"""Evaluate a saved model on the headerless CAMELYON16 reference CSV."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from engine import evaluate
from models.baseline import model_from_checkpoint
from native_data import load_native_regions
from wsi_data import Camelyon16WSI, collate_one_wsi


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None, *, model_loader=None, checkpoint_formats=("metasr-shared-resnet-srcls-v2",)):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-ema", action="store_true",
                        help="Load ema_model_state from a last.pth checkpoint")
    args = parser.parse_args(argv)

    # The user's checkpoint is trusted and its SHA256 was verified before transfer.
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("format") not in checkpoint_formats:
        raise ValueError("Expected the shared ResNet18 SR/classification checkpoint")
    weight_source = "ema_model_state" if args.use_ema else "model_state"
    if weight_source not in saved:
        raise ValueError(f"Checkpoint does not contain {weight_source}")
    config = saved["config"]
    if config.get("data_mode") != "native512" or "decision_threshold" not in config:
        raise ValueError("Checkpoint must contain native512 mode and a saved threshold")

    labels = {}
    with args.reference_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if not row or not any(value.strip() for value in row):
                continue
            if len(row) < 2:
                raise ValueError(f"Invalid reference row: {row}")
            slide_id, name = row[0].strip(), row[1].strip().lower()
            if not slide_id or slide_id in labels or name not in ("normal", "tumor"):
                raise ValueError(f"Invalid or duplicate reference row: {row}")
            labels[slide_id] = int(name == "tumor")
    if set(labels.values()) != {0, 1}:
        raise ValueError("Reference must contain both Normal and Tumor")

    manifest = args.data_root / "manifests" / "patch_manifest.csv"
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    print(f"Reference slides: {len(labels)}; labels: {dict(Counter(labels.values()))}", flush=True)
    print("Validating manifest, verification report and 512 images...", flush=True)
    regions, report = load_native_regions(args.data_root, rows, require_hr=False)
    missing = sorted(set(labels) - set(regions))
    unlabeled = sorted(set(regions) - set(labels))
    if missing or unlabeled:
        raise ValueError(f"Slide mismatch: missing regions={missing}; missing labels={unlabeled}")

    samples = [
        {"slide_id": slide_id, "case_id": slide_id, "label": label,
         "lr_paths": [lr for lr, _ in regions[slide_id]],
         "hr_paths": [hr for _, hr in regions[slide_id]],
         "n_regions": len(regions[slide_id]), "data_mode": "native512"}
        for slide_id, label in labels.items()
    ]
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; activate the correct environment or use --device cpu")
        torch.cuda.set_device(device)
        if config.get("precision") == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support BF16; use --device cpu for FP32 evaluation")

    args.output.mkdir(parents=True, exist_ok=True)
    for filename in ("metrics.json", "predictions.csv"):
        if (args.output / filename).exists():
            raise FileExistsError(f"Refusing to overwrite {args.output / filename}")
    print(f"Checkpoint epoch: {saved['epoch']}; fold: {saved['fold']}; "
          f"weights: {weight_source}; threshold: {config['decision_threshold']}; device: {device}", flush=True)
    print("The saved threshold is fixed; no test-set threshold selection is performed.", flush=True)
    selected = dict(saved, model_state=saved[weight_source])
    model = (model_loader or model_from_checkpoint)(selected, device)
    loader = DataLoader(Camelyon16WSI(samples), batch_size=1, collate_fn=collate_one_wsi)

    def progress():
        for index, sample in enumerate(loader, 1):
            print(f"[{index}/{len(samples)}] {sample['slide_id']} regions={sample['n_regions']}", flush=True)
            yield sample

    result = evaluate(model, progress(), device, config)
    predictions = result.pop("predictions")
    confusion = Counter((row["label"], row["prediction"]) for row in predictions)
    result.update({
        "checkpoint_epoch": saved["epoch"], "checkpoint_fold": saved["fold"],
        "weight_source": weight_source,
        "decision_threshold": config["decision_threshold"], "slide_count": len(samples),
        "TN": confusion[(0, 0)], "FP": confusion[(0, 1)],
        "FN": confusion[(1, 0)], "TP": confusion[(1, 1)],
        "checkpoint_sha256": sha256(args.checkpoint),
        "reference_sha256": sha256(args.reference_csv),
        "manifest_sha256": sha256(manifest), "verification_sha256": sha256(report),
        "code_sha256": {
            name: sha256(Path(__file__).resolve().parent / name)
            for name in ("engine.py", "models/shared_resnet_sr.py", "models/robust_mil.py",
                         "wsi_data.py", "native_data.py", "data_modes.py", "augmentation.py")
        },
        "device": str(device), "precision": config.get("precision") if device.type == "cuda" else "fp32",
    })
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Results:", json.dumps(result, indent=2), flush=True)
    print(f"Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
