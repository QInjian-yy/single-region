"""Use each Region's LR embedding to weight its own HR patches."""
import torch
from torch import nn

from .multiscale_model import MultiScaleSRMIL

HR_AGGREGATION = "lr_guided_attention"
FORMAT = "c16-e2e-lrhr-guided-hr-attention-v1"


class LRGuidedHRAttention(nn.Module):
    """Aggregate [N,K,D] HR embeddings with one [N,D] LR query per Region."""

    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5

    def forward(self, lr_embedding, hr_embedding):
        query = self.query(lr_embedding).float().unsqueeze(1)
        keys = self.key(hr_embedding).float()
        scores = (query * keys).sum(dim=-1) * self.scale
        weights = scores.softmax(dim=1)
        return (weights.unsqueeze(-1) * hr_embedding.float()).sum(dim=1)


class LRGuidedHRMultiScaleMIL(MultiScaleSRMIL):
    def __init__(self, config):
        super().__init__(config)
        # Keep the existing head optimizer slot. Replacing it removes the old
        # scorer and preserves the baseline initialization of every other layer.
        with torch.random.fork_rng(devices=[]):
            self.patch_score = LRGuidedHRAttention(config["classification"]["embedding_dim"])

    def encode_multiscale(self, lr, hr, *, need_sr=False):
        if lr.ndim != 4 or hr.ndim != 5:
            raise ValueError("Expected LR [N,3,512,512] and HR [N,K,3,P,P]")
        n, k, channels, height, width = hr.shape
        if (n != lr.shape[0] or k != self.k or channels != 3
                or (height, width) != (self.patch_size, self.patch_size)):
            raise ValueError("HR local patch dimensions mismatch config")

        stages = self.base.encoder(lr)
        lr_embedding = self.base.classifier_head(stages[-1])
        hr_stages = self.base.encoder(hr.reshape(n * k, 3, height, width))
        hr_embedding = self.base.classifier_head(hr_stages[-1]).reshape(n, k, -1)

        hr_aggregate = self.patch_score(lr_embedding, hr_embedding)
        residual = self.fusion(torch.cat((lr_embedding.float(), hr_aggregate), dim=1))
        fused = lr_embedding.float() + torch.sigmoid(self.fusion_logit) * residual
        sr_features = (self.base.sr.fuse_stages(stages, target_size=lr.shape[-2:])
                       if need_sr else None)
        return fused, sr_features

    def architecture(self):
        return {**super().architecture(), "type": FORMAT, "hr_aggregation": HR_AGGREGATION}
