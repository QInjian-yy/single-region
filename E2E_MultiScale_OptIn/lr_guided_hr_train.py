"""Train the K=4 LR-guided HR experiment with the existing multiscale loop."""
import argparse
import copy
import hashlib
from pathlib import Path
import random
from statistics import mean, stdev
import sys

import numpy as np
import torch

# The existing multiscale trainer imports its sibling modules by bare name.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import camil_data
from . import multiscale_train as baseline
from .lr_guided_hr_model import HR_AGGREGATION, LRGuidedHRMultiScaleMIL
from .ssd_support.relation_paths import remap_lr_samples


def load_config(path):
    config = baseline.load_config(path)
    if config.get("hr_aggregation") != HR_AGGREGATION:
        raise ValueError(f"Use hr_aggregation: {HR_AGGREGATION} with this entry")
    ms = config["multiscale"]
    if ms["patches_per_region"] != 4 or ms["hr_patch_size"] != 256:
        raise ValueError("This experiment requires the K=4 HR256 cache")
    if (config.get("region_relation") or {}).get("enabled"):
        raise ValueError("Use the original Attention MIL without a Region Relation module")
    return config


def smoke_steps(config, train, cache, device, *, steps):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    model = LRGuidedHRMultiScaleMIL(config).to(device)
    optimizer = baseline.build_optimizer(model, config)
    ema = (baseline.camil.shared.ModelEMA(model, config["ema_decay"])
           if config["ema_decay"] > 0 else None)
    indices = sorted(range(len(train)), key=lambda i: train.samples[i]["n_regions"])[:steps]
    history = []
    parameters = dict(model.patch_score.named_parameters())
    for step in range(steps):
        sample = train[indices[step % len(indices)]]
        before = {name: param.detach().clone() for name, param in parameters.items()}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        metrics = baseline.train_wsi(model, sample, cache, optimizer, device, config, epoch=1)
        norms = {}
        for name, param in parameters.items():
            grad = param.grad
            if grad is None or not torch.isfinite(grad).all() or not grad.any():
                raise RuntimeError(f"Missing or nonfinite LR-guided attention gradient: {name}")
            if torch.equal(before[name], param):
                raise RuntimeError(f"Optimizer did not update LR-guided attention: {name}")
            norms[name] = grad.float().norm().item()
        if ema is not None:
            ema.update(model)
        history.append(dict(step=step+1, slide_id=sample["slide_id"], n_regions=sample["n_regions"],
                            metrics=metrics, hr_attention_grad_norms=norms,
                            peak_cuda_allocated_bytes=(torch.cuda.max_memory_allocated(device)
                                                       if device.type == "cuda" else None)))
        print(f"[smoke] step={step+1}/{steps} WSI={sample['slide_id']} "
              f"N={sample['n_regions']} K={cache.k} loss={metrics['train_loss_total']:.6f} "
              f"Q_grad={norms['query.weight']:.6g} K_grad={norms['key.weight']:.6g}", flush=True)

    evaluated = (ema.module if ema is not None else model).eval()
    restore_config = copy.deepcopy(config)
    restore_config["shared_encoder"]["pretrained"] = False
    restore_config["shared_encoder"].pop("weights_path", None)
    restored = LRGuidedHRMultiScaleMIL(restore_config).to(device).eval()
    restored.load_state_dict(evaluated.state_dict(), strict=True)
    sample = train.evaluation_view()[indices[0]]
    with torch.no_grad():
        expected = baseline.wsi_forward(evaluated, sample, cache, device, config)[4]
        actual = baseline.wsi_forward(restored, sample, cache, device, config)[4]
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    return dict(status="passed", device=str(device), precision=config["precision"],
                architecture=model.architecture(), steps=history,
                ema_updates=ema.updates if ema is not None else 0,
                strict_reload_prediction_equal=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--ssd-root", type=Path, default=Path("/home/ub/wsi_fastcache"))
    parser.add_argument("--train-root", type=Path)
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold", type=int, choices=camil_data.FOLDS)
    parser.add_argument("--smoke-steps", type=int, help="Run only these many WSI training steps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.smoke_steps is not None and args.smoke_steps < 1:
        parser.error("--smoke-steps must be positive")

    config = load_config(args.config)
    protocol = camil_data.read_camil_protocol(args.protocol_dir)
    original, provenance = camil_data.load_camil_samples(
        protocol, train_root=args.train_root, test_root=args.test_root, require_hr=True)
    ms = config["multiscale"]
    cache = baseline.open_cache(original, args.cache_dir, patch_size=ms["hr_patch_size"],
                                scale=config["metasr"]["scale"], k=ms["patches_per_region"])
    samples, ssd = remap_lr_samples(original, args.ssd_root)
    cache_index = args.cache_dir.resolve() / "index.json"
    provenance = dict(provenance, ssd_lr=ssd, hr_cache=dict(
        path=str(cache_index), sha256=hashlib.sha256(cache_index.read_bytes()).hexdigest(),
        k=cache.k, patch_size=cache.size))
    print(f"[data] {len(samples)} WSI, {ssd['regions']} Regions; HR cache K={cache.k}", flush=True)
    print(f"[hr_aggregation] {HR_AGGREGATION}", flush=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 not supported on GPU")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    out = args.output.resolve()
    input_roots = [args.protocol_dir, args.cache_dir, args.ssd_root, *(
        Path(provenance[f"{source}_manifest"]["data_root"]) for source in protocol.sources)]
    if any(out.is_relative_to(path.resolve()) for path in input_roots):
        raise ValueError("Output must be outside the data, protocol and cache roots")
    out.mkdir(parents=True, exist_ok=False)

    if args.smoke_steps is not None:
        fold = 0 if args.fold is None else args.fold
        train, _, _ = camil_data.make_camil_datasets(
            protocol, samples, fold, augmentation=config.get("augmentation"),
            max_train_regions=config["training"]["max_train_regions"])
        baseline.camil.configure_class_weights(config, train)
        result = smoke_steps(config, train, cache, device, steps=args.smoke_steps)
        baseline.camil._write_json(out / "smoke.json", dict(result, fold=fold, data_provenance=provenance))
        return

    folds = camil_data.FOLDS if args.fold is None else (args.fold,)
    results = [baseline.train_fold(config, protocol, samples, provenance, cache, fold,
                                   out/f"fold_{fold}", device, model_factory=LRGuidedHRMultiScaleMIL,
                                   training_entry=Path(__file__).name) for fold in folds]
    if args.fold is None:
        aucs = [result["auc"] for result in results]
        summary = dict(fold_count=len(results), test_auc_mean=mean(aucs),
                       test_auc_std=stdev(aucs), folds=results)
        baseline.camil._write_json(out/"summary.json", summary)
        print(f"[five folds] Test AUC {summary['test_auc_mean']:.6f} "
              f"+/- {summary['test_auc_std']:.6f}", flush=True)


if __name__ == "__main__":
    main()
