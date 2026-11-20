"""Exact identity: no parameters, buffers, copies or random draws."""
from torch import nn

from .registry import register_relation


@register_relation("identity")
class IdentityRelation(nn.Module):
    @staticmethod
    def normalize_config(settings):
        if settings:
            raise ValueError(f"Unknown identity settings: {sorted(settings)}")
        return {}

    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim

    def forward(self, features, context=None):
        return features
