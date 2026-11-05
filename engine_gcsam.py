"""Isolated two-stage WSI training; evaluation and WSI losses stay in engine.py."""

import random

import numpy as np
import torch

from engine import gradient_norm, memory_summary, wsi_loss
from optimizers.gcsam import GCSAM


def _rng_state(device):
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if device.type == "cuda" else None)


def _restore_rng(state):
    python, numpy, cpu, cuda = state
    random.setstate(python)
    np.random.set_state(numpy)
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


def train_wsi(model, sample, optimizer, device, config, log_gradients=False):
    if not isinstance(optimizer, GCSAM):
        raise TypeError("engine_gcsam requires the isolated GCSAM optimizer")
    if float(config["lambda_sr"]) != 0 or getattr(model, "vib_enabled", False):
        raise ValueError("This GCSAM experiment requires SR=0 and VIB disabled")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    before = _rng_state(device)
    after_first = None
    try:
        # sample is already selected/augmented by Dataset, once per WSI access.
        total, cls_first, sr, _ = wsi_loss(model, sample, device, config)
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite first GCSAM loss")
        total.backward()
        after_first = _rng_state(device)
        first_norm = optimizer.gradient_norm().item()
        if not np.isfinite(first_norm):
            raise FloatingPointError("Non-finite first GCSAM gradients")
        gc_norm = optimizer.first_step(zero_grad=True).item()

        # Replay the SAME sample and RNG; do not ask Dataset for another bag.
        _restore_rng(before)
        second_total, cls_second, _, _ = wsi_loss(model, sample, device, config)
        if not torch.isfinite(second_total):
            raise FloatingPointError("Non-finite second GCSAM loss")
        second_total.backward()
        second_norm = optimizer.gradient_norm().item()
        if not np.isfinite(second_norm):
            raise FloatingPointError("Non-finite second GCSAM gradients")

        # Exact restore precedes clipping and the ONLY AdamW update.
        optimizer.restore_parameters()
        norms = {"rdn": gradient_norm(model.sr),
                 "region_encoder": gradient_norm(model.region_encoder),
                 "abmil": gradient_norm(model.classifier_head.mil),
                 "classifier": gradient_norm(model.classifier_head.classifier),
                 "p2w": 0.0, "shared_encoder": gradient_norm(model.encoder)}
        if log_gradients:
            print(f"[gradient] slide_id={sample['slide_id']} N={sample['n_regions']} {norms}",
                  flush=True)
        clip = float(config.get("gradient_clip_norm", 0.0))
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
        optimizer.second_step()
    except BaseException:
        optimizer.cancel_step()
        optimizer.zero_grad(set_to_none=True)
        raise
    finally:
        # Two passes consume the random stream of just the first pass.
        _restore_rng(after_first if after_first is not None else before)

    return {"optimizer": "gcsam", "base_optimizer": "adamw", "rho": optimizer.rho,
            "train_loss_cls": cls_first.item(), "train_loss_cls_first": cls_first.item(),
            "train_loss_cls_second": cls_second.item(), "train_loss_sr": sr.item(),
            "train_loss_total": total.item(), "grad_norm": norms["shared_encoder"],
            "grad_norm_first": first_norm, "grad_norm_gc": gc_norm,
            "grad_norm_second": second_norm,
            **{f"lr_{name}": next(group["lr"] for group in optimizer.param_groups
                                  if group["name"].startswith(name))
               for name in ("encoder", "head")},
            **memory_summary(device)}
