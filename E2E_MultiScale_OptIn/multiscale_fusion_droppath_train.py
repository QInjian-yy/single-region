"""Train multiscale MIL with stochastic depth on the LR-HR fusion residual."""
from __future__ import annotations

import argparse
from inspect import signature
from pathlib import Path
from statistics import mean, stdev

import torch
from torchvision.ops import StochasticDepth

import camil_data
from .multiscale_relation_train import baseline
from .ssd_support.relation_paths import remap_lr_samples


FORMAT = "c16-e2e-lrhr-fusion-droppath-v1"


class FusionDropPathMIL(baseline.MultiScaleSRMIL):
    def __init__(self, config):
        super().__init__(config)
        self.fusion.add_module(
            "drop_path", StochasticDepth(p=config["fusion_drop_path"], mode="row")
        )

    def architecture(self):
        return {
            **super().architecture(),
            "type": FORMAT,
            "fusion_drop_path": self.fusion.drop_path.p,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--ssd-root", type=Path, default=Path("/home/ub/wsi_fastcache"))
    parser.add_argument("--train-root", type=Path)
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--fold", type=int, choices=camil_data.FOLDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--drop-probability", type=float, default=0.10)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-data", action="store_true")
    mode.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 0 <= args.drop_probability < 1:
        parser.error("--drop-probability must be in [0, 1)")

    if "model_factory" not in signature(baseline.train_fold).parameters:
        parser.error("Use the relation-enabled multiscale trainer from the Seed123 SSD runs")

    config = baseline.load_config(args.config)
    relation = config.get("region_relation") or {}
    if relation.get("enabled", False) or config.get("hr_aggregation") not in (None, "patch_score"):
        parser.error("Fusion DropPath requires the original multiscale patch_score model")
    config["fusion_drop_path"] = args.drop_probability

    protocol = camil_data.read_camil_protocol(args.protocol_dir)
    original, provenance = camil_data.load_camil_samples(
        protocol, train_root=args.train_root, test_root=args.test_root, require_hr=True
    )
    samples, ssd = remap_lr_samples(original, args.ssd_root)
    provenance["ssd_lr"] = ssd
    multiscale = config["multiscale"]
    cache = baseline.open_cache(
        original, args.cache_dir, patch_size=multiscale["hr_patch_size"],
        scale=config["metasr"]["scale"], k=multiscale["patches_per_region"],
    )
    print(f"[data] {len(samples)} WSI, {ssd['regions']} Regions; "
          f"grouping={protocol.patient_grouping}", flush=True)
    print(f"[fusion_drop_path] p={args.drop_probability:g}, mode=row, HR cache K={cache.k}",
          flush=True)
    if args.check_data:
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 not supported on GPU")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    out = args.output.resolve()
    input_roots = [args.protocol_dir, args.cache_dir, args.ssd_root, *(
        Path(provenance[f"{source}_manifest"]["data_root"]) for source in protocol.sources
    )]
    if any(out.is_relative_to(root.resolve()) for root in input_roots):
        raise ValueError("Output must be outside the data, protocol and cache roots")
    out.mkdir(parents=True, exist_ok=False)

    folds = camil_data.FOLDS if args.fold is None else (args.fold,)
    results = [
        baseline.train_fold(
            config, protocol, samples, provenance, cache, fold, out / f"fold_{fold}",
            device, model_factory=FusionDropPathMIL, training_entry=Path(__file__).name,
        )
        for fold in folds
    ]
    if args.fold is None:
        aucs = [result["auc"] for result in results]
        baseline.camil._write_json(out / "summary.json", dict(
            fold_count=len(results), test_auc_mean=mean(aucs), test_auc_std=stdev(aucs),
            folds=results,
        ))
        print(f"[five folds] Test AUC {mean(aucs):.6f} +/- {stdev(aucs):.6f}", flush=True)


if __name__ == "__main__":
    main()
