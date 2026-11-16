"""Opt-in, true differentiable LR+HR online-encoder MIL with LR->HR SR supervision.

No existing E2E_MetaSR model files, state_dict formats or training entries are changed.
"""
from __future__ import annotations

import torch
from torch import nn

from models.shared_resnet_sr import SharedResNetSRMIL

FORMAT = "c16-e2e-lrhr-shared-resnet-sr-aem-v1"


class MultiScaleSRMIL(nn.Module):
    """One shared ResNet18 processes LR regions and online HR pixels.

    One LR region is fused with K HR local patches, then the original MIL head
    aggregates all regions of a WSI. SR decoder sees only LR stages.
    """

    def __init__(self, config):
        super().__init__()
        cls = dict(config["classification"])
        if cls.get("pooling") != ["attention"]:
            raise ValueError("Multiscale AEM requires pooling: [attention]")
        if cls.get("region_interaction", {}).get("enabled", False):
            raise ValueError("ASRD is not supported in this isolated addon")
        if cls.get("vib", {}).get("enabled", False):
            raise ValueError("VIB is not supported in this isolated addon")
        ms = config["multiscale"]
        self.k = ms["patches_per_region"]
        self.patch_size = ms["hr_patch_size"]
        sr = config["metasr"]
        self.base = SharedResNetSRMIL(
            shared_encoder=config["shared_encoder"], classification=cls,
            scale=sr["scale"], rgb_range=sr["rgb_range"],
            sr_channels=sr["channels"], sr_residual_scale=sr["residual_scale"],
            sr_dropout=sr["dropout"], hr_tile=sr["hr_tile"],
        )
        d = cls["embedding_dim"]
        self.patch_score = nn.Linear(d, 1)
        self.fusion = nn.Sequential(
            nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(),
            nn.Linear(d, d, bias=False),
        )
        # Start near the original LR embedding. The gate and all new weights
        # remain differentiable; HR evidence cannot bypass the LR-only SR head.
        self.fusion_logit = nn.Parameter(torch.tensor(-1.5))

    def encode_multiscale(self, lr, hr, *, need_sr=False):
        if lr.ndim != 4 or hr.ndim != 5:
            raise ValueError("Expected LR [B,3,512,512] and HR [B,K,3,P,P]")
        b, k, ch, h, w = hr.shape
        if (k != self.k or ch != 3 or (h, w) != (self.patch_size, self.patch_size)
                or b != lr.shape[0]):
            raise ValueError("HR local patch dimensions mismatch config")
        stages = self.base.encoder(lr)
        lr_embedding = self.base.classifier_head(stages[-1])
        hr_stages = self.base.encoder(hr.reshape(b * k, 3, h, w))
        hr_embedding = self.base.classifier_head(hr_stages[-1]).reshape(b, k, -1)
        local_weights = self.patch_score(hr_embedding.float()).softmax(dim=1)
        hr_aggregate = (local_weights * hr_embedding.float()).sum(dim=1)
        residual = self.fusion(torch.cat((lr_embedding.float(), hr_aggregate), dim=1))
        fused = lr_embedding.float() + torch.sigmoid(self.fusion_logit) * residual
        sr_features = (self.base.sr.fuse_stages(stages, target_size=lr.shape[-2:])
                       if need_sr else None)
        return fused, sr_features

    def classify(self, embeddings):
        logits, attention = self.base.classifier_head.forward_embeddings(embeddings)
        return logits, attention

    def architecture(self):
        return {
            "type": FORMAT, "shared_encoder": "resnet18",
            "shared_lr_hr_encoder": True, "hr_patches_per_region": self.k,
            "hr_patch_size": self.patch_size, "sr_scale": self.base.sr.scale,
            "embedding_dim": self.base.classifier_head.projection[1].out_features,
        }
