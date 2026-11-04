import torch
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)


_WEIGHTS = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V1),
}


class ImageNetRegionEncoder(nn.Module):

    def __init__(self, backbone="resnet18", pretrained=True,
                 trainable="layer4", embedding_dim=256, dropout=0.2,
                 weights_path=None):
        super().__init__()
        if backbone not in _WEIGHTS:
            raise ValueError("backbone must be resnet18 or resnet50")
        constructor, enum = _WEIGHTS[backbone]
        weights = enum if pretrained else None
        self.backbone = constructor(weights=weights)
        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state, strict=True)
        backbone_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.embedding_dim = embedding_dim
        self.trainable = trainable
        self.project = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim, bias=False),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=True,
        )
        self._set_backbone_trainable(trainable)

    def _set_backbone_trainable(self, trainable):
        if trainable not in ("frozen", "layer4", "all"):
            raise ValueError("classification.trainable must be frozen, layer4 or all")
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if trainable == "all":
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(True)
        elif trainable == "layer4":
            for parameter in self.backbone.layer4.parameters():
                parameter.requires_grad_(True)

    def set_trainable(self, trainable):
        self.trainable = trainable
        self._set_backbone_trainable(trainable)

    def train(self, mode=True):
        super().train(mode)
        # Pretrained BN statistics are stable and small WSI batches are noisy.
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected RGB region images [B,3,H,W]")
        features = self.backbone((images - self.mean) / self.std)
        return self.project(features)


class RobustMILPooling(nn.Module):
    """Select and concatenate gated attention, mean and/or top-k pooling."""

    POOLING_BRANCHES = ("attention", "mean", "topk")

    @classmethod
    def normalize_pooling(cls, pooling):
        if isinstance(pooling, str):
            pooling = (pooling,)
        if not isinstance(pooling, (list, tuple)) or not pooling:
            raise ValueError("classification.pooling must select attention, mean and/or topk")
        if (any(not isinstance(name, str) or name not in cls.POOLING_BRANCHES for name in pooling)
                or len(set(pooling)) != len(pooling)):
            raise ValueError("classification.pooling must contain unique attention, mean or topk values")
        return tuple(name for name in cls.POOLING_BRANCHES if name in pooling)

    def __init__(self, input_dim=256, hidden_dim=128, topk=5,
                 region_dropout=0.1, attention_dropout=0.1, pooling=POOLING_BRANCHES):
        super().__init__()
        self.pooling = self.normalize_pooling(pooling)
        if input_dim < 1 or hidden_dim < 1 or topk < 1:
            raise ValueError("MIL dimensions and topk must be positive")
        if not 0 <= region_dropout < 1 or not 0 <= attention_dropout < 1:
            raise ValueError("MIL dropout must be in [0, 1)")
        self.attention_V = nn.Linear(input_dim, hidden_dim)
        self.attention_U = nn.Linear(input_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.region_dropout = region_dropout
        self.topk = topk
        self.output_dim = input_dim * len(self.pooling)

    def forward(self, embeddings):
        if embeddings.ndim != 2 or embeddings.shape[0] < 1:
            raise ValueError("Expected region embeddings [N,D]")
        # Whole-region dropout prevents the model from memorising one slide's
        # exact region count or a single incidental region.
        if self.training and self.region_dropout and embeddings.shape[0] > 1:
            keep = torch.rand(embeddings.shape[0], device=embeddings.device)
            mask = keep >= self.region_dropout
            if mask.any():
                embeddings = embeddings[mask]
        pooled, weights = [], None
        if "attention" in self.pooling:
            x = self.attention_dropout(embeddings)
            gated = torch.tanh(self.attention_V(x)) * torch.sigmoid(self.attention_U(x))
            scores = self.attention_w(gated).squeeze(-1)
            weights = torch.softmax(scores, dim=0)
            pooled.append((weights.unsqueeze(-1) * embeddings).sum(dim=0, keepdim=True))
        if "mean" in self.pooling:
            pooled.append(embeddings.mean(dim=0, keepdim=True))
        if "topk" in self.pooling:
            k = min(self.topk, embeddings.shape[0])
            pooled.append(embeddings.topk(k, dim=0).values.mean(dim=0, keepdim=True))
        return torch.cat(pooled, dim=1), weights
