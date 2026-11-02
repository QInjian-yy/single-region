from __future__ import annotations

import torch
import torch.nn as nn

from models.metasr import MemoryEfficientMetaRDN

CHECKPOINT_FORMAT = "metasr-shared-srcls-v1"


class SpatialPyramidPooling(nn.Module):
    """1x1 + 2x2 + 3x3 + 6x6 max-pool -> 50 * C dims (C=64 => 3200)."""

    def __init__(self, in_channels: int = 64, levels=(1, 2, 3, 6)):
        super().__init__()
        self.levels = levels
        self.out_dim = in_channels * sum(l * l for l in levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = [
            nn.functional.adaptive_max_pool2d(x, output_size=level).flatten(1)
            for level in self.levels
        ]
        return torch.cat(chunks, dim=1)


class GatedAttentionMIL(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.attention_v = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(in_dim, hidden), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        # x: [N, D]
        a = self.attention_w(self.attention_v(x) * self.attention_u(x))
        a = torch.softmax(a, dim=0)
        z = torch.sum(a * x, dim=0, keepdim=True)
        return z, a


class MeanMIL(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.mean(dim=0, keepdim=True), None


class SharedSRClsModel(nn.Module):
    """
    Shared encoder = MetaRDN feature extractor.
    Classification: feature -> SPP -> (dropout) -> MIL -> classifier
    SR: feature -> MetaRDN decode_crop (training only)
    Inference: call encode_regions + forward_embeddings; ignore self.sr decoder.
    """

    def __init__(
        self,
        mil_pooling: str = "abmil",
        classification_head: str = "linear",
        embed_dropout: float = 0.3,
        mil_dropout: float = 0.3,
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

        # ---- shared encoder + SR decoder (one MetaRDN module) ----
        self.sr = MemoryEfficientMetaRDN(
            scale=scale,
            rdn_blocks=rdn_blocks,
            checkpoint_rdb=checkpoint_rdb,
            rgb_range=rgb_range,
            freeze_meanshift=freeze_meanshift,
            lr_chunk_size=lr_chunk_size,
        )

        self.spp = SpatialPyramidPooling(in_channels=64)
        emb_dim = self.spp.out_dim
        self.embed_dropout = nn.Dropout(embed_dropout)

        if mil_pooling == "abmil":
            self.mil_head = GatedAttentionMIL(emb_dim, 128)
        elif mil_pooling == "meanmil":
            self.mil_head = MeanMIL()
        else:
            raise ValueError("mil_pooling must be abmil or meanmil")

        self.mil_dropout = nn.Dropout(mil_dropout)
        if classification_head == "linear":
            self.classifier = nn.Linear(emb_dim, 2)
        elif classification_head == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(emb_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(mil_dropout),
                nn.Linear(128, 2),
            )
        else:
            raise ValueError("classification_head must be linear or mlp")

        self._arch = {
            "type": CHECKPOINT_FORMAT,
            "shared_encoder": "metardn_feature",
            "mil_pooling": mil_pooling,
            "classification_head": classification_head,
            "spp_dim": emb_dim,
            "embed_dropout": embed_dropout,
            "mil_dropout": mil_dropout,
            "rdn_blocks": rdn_blocks,
            "scale": scale,
        }

    def encode_regions(self, lr: torch.Tensor):
        """
        Shared forward:
          feature = encoder(lr)   # also used by SR
          embedding = SPP(feature) then train-time dropout
        """
        feature = self.sr.extract_features(lr)
        embedding = self.spp(feature)
        embedding = self.embed_dropout(embedding)
        return feature, embedding

    def forward_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.mil_head(embeddings)
        return self.classifier(self.mil_dropout(pooled))

    def architecture(self):
        return dict(self._arch)

    # for gradient logging compatibility
    @property
    def region_encoder(self):
        return self.spp


def model_from_checkpoint(saved, device):
    cfg = saved["config"]
    m = cfg.get("metasr", {})
    model = SharedSRClsModel(
        mil_pooling=cfg.get("mil_pooling", "abmil"),
        classification_head=cfg.get("classification_head", "linear"),
        embed_dropout=cfg.get("embed_dropout", 0.3),
        mil_dropout=cfg.get("mil_dropout", 0.3),
        scale=m.get("scale", 16),
        rdn_blocks=m.get("rdn_blocks", 1),
        checkpoint_rdb=m.get("checkpoint_rdb", True),
        rgb_range=m.get("rgb_range", 1),
        freeze_meanshift=m.get("freeze_meanshift", True),
        lr_chunk_size=m.get("lr_chunk_size", 256),
    ).to(device)
    model.load_state_dict(saved["model_state"], strict=True)
    return model