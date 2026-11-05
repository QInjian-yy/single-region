"""Ascent-only Gradient Centralization + SAM, with an AdamW descent step.

The perturbation uses the post-GC norm (arXiv:2501.11584, Eq. 7).
Unlike the upstream example's subtraction, copies restore weights exactly.
"""

import copy
import math

import torch


class GCSAM(torch.optim.Optimizer):
    def __init__(self, base_optimizer, rho=0.05):
        if not isinstance(base_optimizer, torch.optim.AdamW):
            raise ValueError("GCSAM requires an AdamW base optimizer")
        if (isinstance(rho, bool) or not isinstance(rho, (int, float))
                or not math.isfinite(rho) or rho < 0):
            raise ValueError("GCSAM rho must be finite and nonnegative")
        super().__init__(base_optimizer.param_groups, base_optimizer.defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        self.state = base_optimizer.state
        self.rho = float(rho)
        self._originals = {}
        self._step_pending = False

    @property
    def step_pending(self):
        return self._step_pending

    def _parameters_with_grad(self):
        return [p for group in self.param_groups for p in group["params"]
                if p.requires_grad and p.grad is not None]

    @torch.no_grad()
    def gradient_norm(self):
        parameters = self._parameters_with_grad()
        if not parameters:
            raise RuntimeError("GCSAM requires gradients before each stage")
        return torch.stack([p.grad.norm(p=2) for p in parameters]).norm(p=2)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        if self._step_pending:
            raise RuntimeError("Finish or cancel the pending GCSAM step first")
        parameters = self._parameters_with_grad()
        for p in parameters:
            if p.grad.ndim > 1:
                p.grad.sub_(p.grad.mean(dim=tuple(range(1, p.grad.ndim)), keepdim=True))
        # Normalize only AFTER centralization; 1-D gradients enter unchanged.
        norm = self.gradient_norm()
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite centralized GCSAM gradients")
        self._originals = {p: p.detach().clone() for p in parameters}
        self._step_pending = True
        try:
            scale = self.rho / (norm + 1e-12)
            for p in parameters:
                p.add_(p.grad * scale)
        except BaseException:
            self.cancel_step()
            raise
        if zero_grad:
            self.zero_grad(set_to_none=True)
        return norm

    @torch.no_grad()
    def restore_parameters(self):
        """Restore every ascended parameter, including ones without a second grad."""
        for p, original in self._originals.items():
            p.copy_(original)
        self._originals.clear()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        if not self._step_pending:
            raise RuntimeError("GCSAM second_step requires one preceding first_step")
        if self._originals:
            self.restore_parameters()
        if not torch.isfinite(self.gradient_norm()):
            self.cancel_step()
            raise FloatingPointError("Non-finite GCSAM descent gradients")
        # Descent gradients are neither centralized nor adaptively rescaled.
        self.base_optimizer.step()
        self._step_pending = False
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Use engine_gcsam.train_wsi for the two forward/backward passes")
        return self.second_step()

    @torch.no_grad()
    def cancel_step(self):
        self.restore_parameters()
        self._step_pending = False

    def state_dict(self):
        if self._step_pending:
            raise RuntimeError("Cannot checkpoint between GCSAM ascent and descent")
        saved = self.base_optimizer.state_dict()
        saved["gcsam"] = {"version": 1, "base_optimizer": "adamw", "rho": self.rho,
                          "normalization": "after_gc"}
        return saved

    def load_state_dict(self, state_dict):
        if self._step_pending:
            raise RuntimeError("Cannot load a checkpoint during a GCSAM step")
        saved = copy.deepcopy(state_dict)
        metadata = saved.pop("gcsam")
        if (metadata.get("version") != 1 or metadata.get("base_optimizer") != "adamw"
                or metadata.get("normalization") != "after_gc"):
            raise ValueError("Incompatible GCSAM optimizer state")
        rho = float(metadata["rho"])
        if not math.isfinite(rho) or rho < 0:
            raise ValueError("Invalid rho in GCSAM optimizer state")
        self.base_optimizer.load_state_dict(saved)
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        self.rho = rho
