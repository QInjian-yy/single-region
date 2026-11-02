"""Classification on pretrained ResNet18(LR); optional MetaRDN SR branch."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from models.metasr import MemoryEfficientMetaRDN

CHECKPOINT_FORMAT = "metasr-dualpath-v1"


def _resnet18_backbone(pretrained: bool = True) -> tuple[nn.Module, int]:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Identity()
    return net, 512


class GatedAttentionMIL(nn.Module):
    """Standard gated attention MIL (same spirit as classic ABMIL)."""

    def __init__(self, in_dim: int = 512, hidden: int = 128):
        super().__init__()
        self.attention_v = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(in_dim, hidden), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.attention_w(self.attention_v(x) * self.attention_u(x))
        a = torch.softmax(a, dim=0)
        z = torch.sum(a * x, dim=0, keepdim=True)
        return z, a


class MeanMIL(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return x.mean(dim=0, keepdim=True), None


class DualPathMetaSRMIL(nn.Module):

    def __init__(
        self,
        mil_pooling: str = "abmil",
        classification_head: str = "linear",
        pretrained_cls: bool = True,
        freeze_cls_stem: bool = True,
        dropout: float = 0.5,
        scale: int = 16,
        rdn_blocks: int = 1,
        checkpoint_rdb: bool = True,
        rgb_range: int = 1,
        freeze_meanshift: bool = True,
        lr_chunk_size: int = 256,
        **kwargs,
    ):
        super().__init__()
        self.mil_pooling = mil_pooling
        self.classification_head = classification_head
        self.dropout_p = dropout

        self.cls_backbone, emb_dim = _resnet18_backbone(pretrained=pretrained_cls)
        if freeze_cls_stem:
            for name, p in self.cls_backbone.named_parameters():
                # freeze conv1 / bn1 / layer1
                if name.startswith("conv1") or name.startswith("bn1") or name.startswith("layer1"):
                    p.requires_grad = False

        self.sr = MemoryEfficientMetaRDN(
            scale=scale,
            rdn_blocks=rdn_blocks,
            checkpoint_rdb=checkpoint_rdb,
            rgb_range=rgb_range,
            freeze_meanshift=freeze_meanshift,
            lr_chunk_size=lr_chunk_size,
        )

        if mil_pooling == "abmil":
            self.mil_head = GatedAttentionMIL(emb_dim, 128)
        elif mil_pooling == "meanmil":
            self.mil_head = MeanMIL()
        else:
            raise ValueError("mil_pooling must be abmil or meanmil")

        self.dropout = nn.Dropout(dropout)
        if classification_head == "linear":
            self.classifier = nn.Linear(emb_dim, 2)
        elif classification_head == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(emb_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, 2),
            )
        else:
            raise ValueError("classification_head must be linear or mlp")

        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

        self._arch = {
            "type": CHECKPOINT_FORMAT,
            "mil_pooling": mil_pooling,
            "classification_head": classification_head,
            "cls_backbone": "resnet18_imagenet",
            "emb_dim": emb_dim,
            "dropout": dropout,
            "freeze_cls_stem": freeze_cls_stem,
            "rdn_blocks": rdn_blocks,
            "scale": scale,
        }

    def _normalize(self, lr: torch.Tensor) -> torch.Tensor:
        return (lr - self.img_mean) / self.img_std

    def encode_regions(self, lr: torch.Tensor):
        x = self._normalize(lr)
        embedding = self.cls_backbone(x)
        feature = self.sr.extract_features(lr)
        return feature, embedding

    def forward_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.mil_head(embeddings)
        return self.classifier(self.dropout(pooled))

    def architecture(self):
        return dict(self._arch)

    @property
    def region_encoder(self):
        return self.cls_backbone


def model_from_checkpoint(saved, device):
    cfg = saved["config"]
    metasr = cfg.get("metasr", {})
    model = DualPathMetaSRMIL(
        mil_pooling=cfg.get("mil_pooling", "abmil"),
        classification_head=cfg.get("classification_head", "linear"),
        pretrained_cls=cfg.get("pretrained_cls", True),
        freeze_cls_stem=cfg.get("freeze_cls_stem", True),
        dropout=cfg.get("dropout", 0.5),
        scale=metasr.get("scale", 16),
        rdn_blocks=metasr.get("rdn_blocks", 1),
        checkpoint_rdb=metasr.get("checkpoint_rdb", True),
        rgb_range=metasr.get("rgb_range", 1),
        freeze_meanshift=metasr.get("freeze_meanshift", True),
        lr_chunk_size=metasr.get("lr_chunk_size", 256),
    ).to(device)
    model.load_state_dict(saved["model_state"], strict=True)
    return model