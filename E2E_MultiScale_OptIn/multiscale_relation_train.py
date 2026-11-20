"""Opt-in region relations: reuse the existing multiscale training loop and losses."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from statistics import mean, stdev
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import torch

import camil_data
import multiscale_train as baseline
from E2E_MultiScale_OptIn.multiscale_relation_model import MultiScaleRelationMIL, build_model
from E2E_MultiScale_OptIn.relations import normalize_relation_config
from E2E_MultiScale_OptIn.ssd_support.relation_paths import remap_lr_samples


def load_config(path):
    config = baseline.load_config(path)
    settings = normalize_relation_config(config.get("region_relation"))
    if "region_relation" in config:
        config["region_relation"] = settings
    return config


def build_optimizer(model, config):
    original = model.multiscale if isinstance(model, MultiScaleRelationMIL) else model
    optimizer = baseline.build_optimizer(original, config)
    if isinstance(model, MultiScaleRelationMIL):
        parameters = [p for p in model.region_relation.parameters() if p.requires_grad]
        for decay in (True, False):
            group = [p for p in parameters if (p.ndim > 1) == decay]
            if group:
                optimizer.add_param_group({
                    "params": group, "lr": float(config["classification"]["head_lr"]),
                    "weight_decay": float(config["weight_decay"]) if decay else 0.0,
                    "name": f"head_relation_{'decay' if decay else 'no_decay'}",
                })
    identifiers = [id(p) for group in optimizer.param_groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != expected:
        raise RuntimeError("Optimizer misses or repeats trainable region-relation parameters")
    return optimizer


def train_fold(config, protocol, samples, provenance, cache, fold, out, device):
    return baseline.train_fold(
        config, protocol, samples, provenance, cache, fold, out, device,
        model_factory=build_model, optimizer_factory=build_optimizer,
        training_entry="multiscale_relation_train.py")


def main(argv=None, *, default_ssd_root=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--train-root", type=Path)
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--ssd-root", type=Path, default=default_ssd_root)
    parser.add_argument("--fold", type=int, choices=camil_data.FOLDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-steps", type=int,
                        help="Only this many WSI training steps, EMA and checkpoint checks; no fold run")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-data", action="store_true")
    mode.add_argument("--prepare-cache", action="store_true")
    mode.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.smoke_steps is not None and (args.smoke_steps < 1 or args.output is None):
        parser.error("--smoke-steps requires a positive value and --output")
    config = load_config(args.config)
    protocol = camil_data.read_camil_protocol(args.protocol_dir)
    original_samples, provenance = camil_data.load_camil_samples(
        protocol, train_root=args.train_root, test_root=args.test_root, require_hr=True)
    print(f"[data] {len(original_samples)} WSI, "
          f"{sum(s['n_regions'] for s in original_samples.values())} Regions; "
          f"grouping={protocol.patient_grouping}", flush=True)
    samples = original_samples
    if args.ssd_root is not None:
        samples, ssd_record = remap_lr_samples(original_samples, args.ssd_root)
        provenance = dict(provenance, ssd_lr=ssd_record)
        print(f"[ssd] {ssd_record['regions']} LR paths -> {ssd_record['root']}", flush=True)
    print(f"[relation] {json.dumps(normalize_relation_config(config.get('region_relation')))}", flush=True)
    if args.check_data:
        return
    ms = config["multiscale"]
    cache_settings = dict(patch_size=ms["hr_patch_size"], scale=config["metasr"]["scale"],
                          k=ms["patches_per_region"])
    # Validate the HR cache against ORIGINAL paths, independently of SSD read paths.
    if args.prepare_cache:
        baseline.build_cache(original_samples, args.cache_dir, **cache_settings)
        return
    cache = baseline.open_cache(original_samples, args.cache_dir, **cache_settings)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 not supported on GPU")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    out = args.output.resolve()
    roots = [args.protocol_dir, args.cache_dir,
             *(provenance[f"{source}_manifest"]["data_root"] for source in protocol.sources)]
    if args.ssd_root is not None:
        roots.append(args.ssd_root)
    if any(out.is_relative_to(Path(root).resolve()) for root in roots):
        raise ValueError("Output must be outside data, protocol and cache roots")
    out.mkdir(parents=True, exist_ok=False)
    if args.smoke_steps is not None:
        from E2E_MultiScale_OptIn.relation_smoke import smoke_steps
        fold = args.fold if args.fold is not None else 0
        config = copy.deepcopy(config)
        train = baseline.camil.make_camil_datasets(protocol, samples, fold)[0]
        baseline.camil.configure_class_weights(config, train)
        candidates = [samples[s] for s in protocol.folds[fold]["train"]]
        # Use complete, small training bags; never read Val/Test for smoke-step selection.
        candidates.sort(key=lambda sample: (sample["n_regions"] < 2, sample["n_regions"],
                                            sample["slide_id"]))
        result = smoke_steps(config, candidates[:2], cache, device, steps=args.smoke_steps)
        baseline.camil._write_json(out / "smoke.json", dict(result, data_provenance=provenance))
        return
    folds = camil_data.FOLDS if args.fold is None else (args.fold,)
    results = [train_fold(config, protocol, samples, provenance, cache, f, out / f"fold_{f}", device)
               for f in folds]
    if folds == camil_data.FOLDS:
        aucs = [float(result["auc"]) for result in results]
        summary = dict(fold_count=5, test_auc_mean=mean(aucs), test_auc_std=stdev(aucs), folds=results)
        baseline.camil._write_json(out / "summary.json", summary)
        print(f"[five folds] Test AUC {summary['test_auc_mean']:.6f} "
              f"+/- {summary['test_auc_std']:.6f}", flush=True)


if __name__ == "__main__":
    main()
