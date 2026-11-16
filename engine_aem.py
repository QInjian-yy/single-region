"""Opt-in AEM training, reusing the original full-bag forward and CE/SR losses."""

import math

import torch

import engine


def normalize_aem(settings):
    defaults = {"enabled": False, "weight": 0.01, "anneal_epochs": 50}
    if settings is None:
        settings = {}
    if not isinstance(settings, dict) or settings.keys() - defaults.keys():
        raise ValueError("Invalid classification.aem settings")
    result = {**defaults, **settings}
    if type(result["enabled"]) is not bool:
        raise ValueError("classification.aem.enabled must be a boolean")
    weight = result["weight"]
    if type(weight) not in (int, float) or not 0 <= weight <= torch.finfo(torch.float32).max:
        raise ValueError("classification.aem.weight must be nonnegative and finite in float32")
    if type(result["anneal_epochs"]) is not int or result["anneal_epochs"] < 2:
        raise ValueError("classification.aem.anneal_epochs must be an integer >= 2")
    result["weight"] = float(weight)
    return result


def aem_coefficient(settings, epoch):
    """Epoch 1 uses the initial weight; anneal_epochs and later use zero."""
    if type(epoch) is not int or epoch < 1:
        raise ValueError("AEM epoch must be a positive integer")
    if not settings["enabled"]:
        return 0.0
    progress = min((epoch - 1) / (settings["anneal_epochs"] - 1), 1.0)
    return settings["weight"] * 0.5 * (1.0 + math.cos(math.pi * progress))


def negative_attention_entropy(attention):
    """Sum alpha*log(alpha) over already-normalized MIL weights, in FP32."""
    if attention.ndim != 1 or attention.numel() == 0:
        raise ValueError("AEM expects nonempty one-dimensional MIL attention weights")
    with torch.autocast(device_type=attention.device.type, enabled=False):
        probabilities = attention.float()
        log_probabilities = probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
        return (probabilities * log_probabilities).sum()


def wsi_loss(model, sample, device, config, *, weight):
    # A temporary hook preserves the original pooling math and avoids another forward.
    captured = []

    def capture_attention(module, inputs, output):
        captured.append(output[1])

    handle = model.classifier_head.mil.register_forward_hook(capture_attention)
    try:
        total, cls, sr, logits, kl = engine.wsi_loss(model, sample, device, config, return_kl=True)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0] is None:
        raise RuntimeError("AEM requires exactly one full-bag AttentionMIL forward")
    aem = negative_attention_entropy(captured[0])
    if weight:
        total = total + weight * aem
    return total, cls, sr, logits, kl, aem


def train_wsi(model, sample, optimizer, device, config, *, epoch):
    settings = normalize_aem(config["classification"].get("aem"))
    weight = aem_coefficient(settings, epoch)
    if not settings["enabled"]:
        result = engine.train_wsi(model, sample, optimizer, device, config)
        return dict(result, train_loss_aem=0.0, train_loss_aem_weighted=0.0,
                    train_attention_entropy=0.0, aem_weight=0.0)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total, cls, sr, logits, kl, aem = wsi_loss(model, sample, device, config, weight=weight)
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite AEM joint loss")
    total.backward()
    primary_gradient = max(engine.gradient_norm(model.encoder), engine.gradient_norm(model.sr))
    clip = float(config.get("gradient_clip_norm", 0.0))
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip > 0 else math.inf,
                                  error_if_nonfinite=True)
    optimizer.step()
    result = {
        "train_loss_cls": cls.item(), "train_loss_sr": sr.item(), "train_loss_total": total.item(),
        "grad_norm": primary_gradient, "train_loss_aem": aem.item(),
        "train_loss_aem_weighted": weight * aem.item(), "train_attention_entropy": -aem.item(),
        "aem_weight": weight, **engine.memory_summary(device),
    }
    if getattr(model, "vib_enabled", False):
        result.update(train_loss_kl=kl.item(), train_loss_vib=(model.vib_beta * kl).item())
    return result
