"""One pre-normalized global self-attention layer within a single WSI bag."""
import math

import torch
from torch import nn

from .registry import register_relation


@register_relation("global_mhsa")
class GlobalMHSARelation(nn.Module):
    @staticmethod
    def normalize_config(settings):
        defaults = dict(num_heads=4, attention_dropout=0.1, gate_init=0.1)
        unknown = settings.keys() - defaults.keys()
        if unknown:
            raise ValueError(f"Unknown global_mhsa settings: {sorted(unknown)}")
        result = {**defaults, **settings}
        if type(result["num_heads"]) is not int or result["num_heads"] < 1:
            raise ValueError("region_relation.num_heads must be a positive integer")
        for key in ("attention_dropout", "gate_init"):
            value = result[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"region_relation.{key} must be finite")
            result[key] = float(value)
        if not 0 <= result["attention_dropout"] < 1:
            raise ValueError("region_relation.attention_dropout must be in [0, 1)")
        if not 0 < result["gate_init"] < 1:
            raise ValueError("region_relation.gate_init must be strictly between 0 and 1")
        return result

    def __init__(self, dim=256, num_heads=4, attention_dropout=0.1, gate_init=0.1):
        super().__init__()
        settings = self.normalize_config(dict(num_heads=num_heads,
                                              attention_dropout=attention_dropout,
                                              gate_init=gate_init))
        if type(dim) is not int or dim < 1 or dim % settings["num_heads"]:
            raise ValueError("Embedding dimension must be positive and divisible by num_heads")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        # Construct on CPU, then let the existing model.to(device) move all parameters.
        # New weights must not advance the baseline's CPU initialization/shuffle RNG.
        with torch.random.fork_rng(devices=[]):
            self.attention = nn.MultiheadAttention(
                embed_dim=dim, num_heads=settings["num_heads"],
                dropout=settings["attention_dropout"], batch_first=True)
        initial = settings["gate_init"]
        self.gate_logit = nn.Parameter(torch.tensor(math.log(initial / (1 - initial))))

    def forward(self, features, context=None):
        if (features.ndim != 2 or features.shape[0] < 1 or features.shape[1] != self.dim
                or not features.is_floating_point()):
            raise ValueError(f"Expected floating features [N,{self.dim}], N >= 1")
        # The single batch element is exactly one WSI; no positional or label input.
        normalized = self.norm(features.to(dtype=self.norm.weight.dtype)).unsqueeze(0)
        message, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        # Attention follows the caller's autocast; accumulate the residual in FP32.
        updated = features.float() + self.gate_logit.float().sigmoid() * message.squeeze(0).float()
        return updated.to(dtype=features.dtype)
