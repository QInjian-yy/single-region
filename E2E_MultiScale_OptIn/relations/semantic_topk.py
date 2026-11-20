"""Cosine Top-K messages between regions of exactly one WSI."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .registry import register_relation


@register_relation("semantic_topk")
class SemanticTopKRelation(nn.Module):
    @staticmethod
    def normalize_config(settings):
        defaults = dict(k=3, temperature=1.0, gate_init=0.01)
        if settings.keys() - defaults.keys():
            raise ValueError(f"Unknown semantic_topk settings: {sorted(settings.keys() - defaults.keys())}")
        result = {**defaults, **settings}
        if type(result["k"]) is not int or result["k"] < 1:
            raise ValueError("region_relation.k must be a positive integer")
        for key in ("temperature", "gate_init"):
            value = result[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"region_relation.{key} must be finite")
            result[key] = float(value)
        if result["temperature"] <= 0:
            raise ValueError("region_relation.temperature must be positive")
        if not 0 < result["gate_init"] < 1:
            raise ValueError("region_relation.gate_init must be strictly between 0 and 1")
        return result

    def __init__(self, dim=256, k=3, temperature=1.0, gate_init=0.01):
        super().__init__()
        settings = self.normalize_config(dict(k=k, temperature=temperature, gate_init=gate_init))
        self.dim = dim
        self.k = settings["k"]
        self.temperature = settings["temperature"]
        self.gate_logit = nn.Parameter(torch.tensor(math.log(gate_init / (1 - gate_init))))

    def _validate_features(self, features):
        if (features.ndim != 2 or features.shape[0] < 1 or features.shape[1] != self.dim
                or not features.is_floating_point()):
            raise ValueError(f"Expected floating features [N,{self.dim}], N >= 1")

    def neighbor_weights(self, features):
        """Return [N,K_eff] indices and FP32 weights, with no self edges."""
        self._validate_features(features)
        n = features.shape[0]
        if n == 1:
            return (torch.empty((1, 0), dtype=torch.long, device=features.device),
                    features.new_empty((1, 0), dtype=torch.float32))
        # Explicitly disable an enclosing BF16 autocast for cosine AND softmax.
        with torch.autocast(device_type=features.device.type, enabled=False):
            normalized = F.normalize(features.float(), p=2, dim=-1, eps=1e-6)
            similarities = normalized @ normalized.T
            similarities = similarities.masked_fill(
                torch.eye(n, dtype=torch.bool, device=features.device), -torch.inf)
            scores, indices = similarities.topk(min(self.k, n - 1), dim=-1)
            weights = (scores / self.temperature).softmax(dim=-1)
        return indices, weights

    def forward(self, features, context=None):
        self._validate_features(features)
        if features.shape[0] == 1:
            return features
        indices, weights = self.neighbor_weights(features)
        with torch.autocast(device_type=features.device.type, enabled=False):
            # Actual neighbor values flow into every region; no ABMIL weights are read.
            message = (weights.unsqueeze(-1) * features.float()[indices]).sum(dim=1)
            updated = features.float() + self.gate_logit.float().sigmoid() * message
        return updated.to(dtype=features.dtype)
