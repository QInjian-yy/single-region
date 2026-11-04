"""Diagonal Gaussian variational information bottleneck for region embeddings."""

import math

import torch
from torch import nn


def normalize_vib_config(config=None):
    config = {} if config is None else config
    if not isinstance(config, dict):
        raise ValueError("classification.vib must be a mapping")
    enabled = config.get("enabled", False)
    beta = config.get("beta", 1e-4)
    if not isinstance(enabled, bool):
        raise ValueError("classification.vib.enabled must be a boolean")
    if (isinstance(beta, bool) or not isinstance(beta, (int, float))
            or not math.isfinite(beta) or beta < 0):
        raise ValueError("classification.vib.beta must be finite and nonnegative")
    return {"enabled": enabled, "beta": float(beta)}


class GaussianVIB(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.mu = nn.Linear(embedding_dim, embedding_dim)
        self.logvar = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, embeddings):
        mu = self.mu(embeddings).float()
        logvar = self.logvar(embeddings).float()
        # Keep the exponential, sampling and KL reduction in FP32 under AMP.
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            # expm1(logvar) is exp(logvar)-1, with better precision near zero.
            kl = 0.5 * (mu.square() + logvar.expm1() - logvar)
            kl = kl.sum(dim=-1).mean()
            z = mu
            if self.training:
                z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        return z, kl
