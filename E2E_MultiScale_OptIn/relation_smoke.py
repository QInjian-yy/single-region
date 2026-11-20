"""A bounded training probe using the real multiscale step, EMA and strict reload."""
import copy
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch

from data_modes import data_spec
from augmentation import AugmentedPath
from wsi_data import Camelyon16WSI
from .multiscale_relation_model import model_from_checkpoint
from .multiscale_relation_train import baseline, build_model, build_optimizer


def smoke_steps(config, samples, cache, device, *, steps=2):
    if not samples or steps < 1:
        raise ValueError("Smoke test needs samples and positive steps")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    ema = baseline.camil.shared.ModelEMA(model, config["ema_decay"]) if config["ema_decay"] > 0 else None
    training_samples = Camelyon16WSI(samples, augmentation=config.get("augmentation"),
                                    max_regions=config["training"]["max_train_regions"])
    encoder_shapes, stage_outputs, mil_shapes, relation_shapes = [], [], [], []

    def encoder_hook(module, inputs, output):
        encoder_shapes.append(list(inputs[0].shape))
        if output[-1].requires_grad:
            output[-1].retain_grad()
            stage_outputs.append(output[-1])

    def mil_hook(module, inputs, output):
        mil_shapes.append(list(inputs[0].shape))

    hooks = [model.base.encoder.register_forward_hook(encoder_hook),
             model.base.classifier_head.mil.register_forward_hook(mil_hook)]
    relation = getattr(model, "region_relation", None)
    if relation is not None:
        hooks.append(relation.register_forward_hook(
            lambda module, inputs, output: relation_shapes.append(list(inputs[0].shape))))
    records = []
    try:
        for step in range(steps):
            sample = training_samples[step % len(training_samples)]
            encoder_shapes.clear()
            stage_outputs.clear()
            mil_shapes.clear()
            relation_shapes.clear()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            result = baseline.train_wsi(model, sample, cache, optimizer, device, config, epoch=1)
            if ema is not None:
                ema.update(model)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            gradients = {name: float(parameter.grad.float().norm())
                         for name, parameter in model.named_parameters() if parameter.grad is not None}
            if not gradients or not all(np.isfinite(value) for value in gradients.values()):
                raise AssertionError("Missing or nonfinite gradients")
            if relation is not None and sample["n_regions"] > 1:
                for name, parameter in relation.named_parameters():
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all() or not parameter.grad.any():
                        raise AssertionError(f"No valid relation gradient: {name}")
            stage_norms = [float(stage.grad.float().norm()) for stage in stage_outputs]
            if not stage_norms or not all(np.isfinite(x) and x > 0 for x in stage_norms):
                raise AssertionError("Missing LR/HR encoder-output gradients")
            if mil_shapes != [[sample["n_regions"], config["classification"]["embedding_dim"]]]:
                raise AssertionError("MIL did not receive one complete WSI")
            if relation is not None and relation_shapes != mil_shapes:
                raise AssertionError("Relation did not receive one complete WSI")
            records.append(dict(
                step=step + 1, slide_id=sample["slide_id"], n_regions=sample["n_regions"],
                augmentation_applied=all(isinstance(p, AugmentedPath) for p in sample["lr_paths"]),
                seconds=time.perf_counter() - started, metrics=result,
                encoder_inputs=list(encoder_shapes), mil_inputs=list(mil_shapes),
                relation_inputs=list(relation_shapes), stage_gradient_norms=stage_norms,
                relation_gradient_norms={k: v for k, v in gradients.items() if k.startswith("region_relation.")},
                fusion_gradient_norm=float(model.fusion[1].weight.grad.float().norm()),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            ))
            stage_outputs.clear()
            print(f"[smoke] step={step+1}/{steps} WSI={sample['slide_id']} "
                  f"N={sample['n_regions']} loss={result['train_loss_total']:.6f}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    eval_model = (ema.module if ema is not None else model).eval()
    saved = dict(format=eval_model.architecture()["type"], architecture=eval_model.architecture(),
                 config=copy.deepcopy(config), data_spec=data_spec(config).metadata(),
                 model_state=eval_model.state_dict())
    # Test checkpoints are removed on exit; only the diagnostic JSON is retained.
    with tempfile.TemporaryDirectory(prefix="region_relation_checkpoint_") as temporary:
        path = Path(temporary) / "smoke.pth"
        torch.save(saved, path)
        restored = model_from_checkpoint(torch.load(path, map_location="cpu", weights_only=True), device).eval()
        with torch.no_grad():
            reference = baseline.wsi_forward(eval_model, samples[0], cache, device, config)[4]
            recovered = baseline.wsi_forward(restored, samples[0], cache, device, config)[4]
        torch.testing.assert_close(reference, recovered, rtol=0, atol=0)
    return dict(status="passed", steps=records, config=copy.deepcopy(config),
                device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                ema_updates=ema.updates if ema is not None else 0,
                checkpoint_format=saved["format"], strict_reload_prediction_equal=True,
                optimizer_parameter_count=sum(len(g["params"]) for g in optimizer.param_groups),
                relation_parameters=sum(p.numel() for p in relation.parameters()) if relation is not None else 0)
