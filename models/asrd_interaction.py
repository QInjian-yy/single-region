"""Adaptive sparse relative messages between regions of a single WSI."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def validate_asrd_parameters(rank, topk, temperature, gamma_init):
    for name, value in (("rank", rank), ("topk", topk)):
        if type(value) is not int or value < 1:
            raise ValueError(f"ASRD {name} must be a positive integer")
    fp32 = torch.finfo(torch.float32)
    if type(temperature) not in (int, float) or not fp32.tiny <= temperature <= fp32.max:
        raise ValueError("ASRD temperature must be positive and representable in float32")
    if type(gamma_init) not in (int, float) or not -fp32.max <= gamma_init <= fp32.max:
        raise ValueError("ASRD gamma_init must be finite in float32")


class AdaptiveSparseRelationalDiffusion(nn.Module):
    def __init__(self, dim=256, rank=16, topk=3, temperature=1.0, gamma_init=0.0):
        super().__init__()
        if type(dim) is not int or dim < 1:
            raise ValueError("ASRD dim must be a positive integer")
        validate_asrd_parameters(rank, topk, temperature, gamma_init)
        self.dim = dim
        self.rank = rank
        self.topk = topk
        self.temperature = float(temperature)
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, rank, bias=False)
        self.key = nn.Linear(dim, rank, bias=False)
        self.value = nn.Linear(dim, rank, bias=False)
        self.output = nn.Linear(rank, dim, bias=False)
        self.gate = nn.Linear(dim, 1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def _neighbors(self, query, key):
        scores = (query @ key.transpose(0, 1)) / math.sqrt(self.rank)
        scores.fill_diagonal_(-torch.inf)
        values, indices = scores.topk(min(self.topk, query.shape[0] - 1), dim=1)
        # Center before dividing so small temperatures cannot overflow to +inf.
        weights = ((values - values[:, :1]) / self.temperature).softmax(dim=1)
        return indices, weights

    def forward(self, embeddings):
        if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] != self.dim:
            raise ValueError(f"ASRD expects a nonempty [N, {self.dim}] tensor")
        if embeddings.shape[0] == 1:
            return embeddings

        # Keep relation scores and messages in FP32 under autocast or BF16 weights.
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            normalized = F.layer_norm(
                embeddings.float(), self.norm.normalized_shape,
                self.norm.weight.float(), self.norm.bias.float(), self.norm.eps,
            )
            query = F.linear(normalized, self.query.weight.float())
            key = F.linear(normalized, self.key.weight.float())
            indices, weights = self._neighbors(query, key)
            value = F.linear(normalized, self.value.weight.float())
            relative = value[indices] - value.unsqueeze(1)
            message = (weights.unsqueeze(-1) * relative).sum(dim=1)
            update = F.linear(message, self.output.weight.float())
            gate = F.linear(normalized, self.gate.weight.float(), self.gate.bias.float()).sigmoid()
            residual = self.gamma.float() * gate * update
        return embeddings + residual.to(embeddings.dtype)
