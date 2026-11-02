import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from augmentation import load_augmentation_config
from data_modes import data_spec, validate_data_config
from engine import evaluate, train_wsi
from models.shared_sr_cls_model import CHECKPOINT_FORMAT, SharedSRClsModel
from train_e2e import ROOT, provenance, run_metadata
from validation import run_equivalence
from wsi_data import collate_one_wsi, load_fold_datasets


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    config.setdefault("mil_pooling", "abmil")
    config.setdefault("classification_head", "linear")
    config.setdefault("embed_dropout", 0.3)
    config.setdefault("mil_dropout", 0.3)
    if config["mil_pooling"] not in ("abmil", "meanmil"):
        raise ValueError("mil_pooling must be abmil or meanmil")

    validate_data_config(config)
    if config["precision"] not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")

    metasr = config.setdefault("metasr", {})
    metasr.setdefault("rdn_blocks", 1)
    metasr.setdefault("checkpoint_rdb", True)
    metasr.setdefault("freeze_meanshift", True)
    metasr.setdefault("lr_chunk_size", 256)

    training = config.setdefault("training", {})
    training.setdefault("use_region_microbatch", True)
    training.setdefault("region_microbatch_size", 4)
    training.setdefault("label_smoothing", 0.05)
    training.setdefault("region_dropout", 0.1)
    training.setdefault("grad_clip", 1.0)

    if float(config.get("lambda_sr", 0.0)) < 0:
        raise ValueError("lambda_sr must be >= 0")
    # paper dual-task: SR should usually be on; allow 0 only for debug
    return config


def build_scheduler(optimizer, config):
    if str(config.get("lr_scheduler", "")).lower() != "cosine":
        return None
    warmup = int(config.get("warmup_epochs", 0) or 0)
    min_lr = float(config.get("min_lr", 1e-6))
    base_lr = float(config["lr"])
    total = int(config["epochs"])

    def lr_lambda(epoch_idx):
        if warmup > 0 and epoch_idx < warmup:
            return float(epoch_idx + 1) / float(max(1, warmup))
        progress = (epoch_idx - warmup) / max(1, total - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return max(min_lr / base_lr, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mil-pooling", choices=("abmil", "meanmil"))
    parser.add_argument("--augmentation-config", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--labels-csv", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mil_pooling is not None:
        config["mil_pooling"] = args.mil_pooling
    spec = data_spec(config)

    if spec.mode == "legacy256":
        args.labels_csv = args.labels_csv or ROOT / "downstream_train/camelyon16_labels.csv"
        args.split_dir = args.split_dir or ROOT / "downstream_train"
    if args.augmentation_config is not None:
        config["augmentation"] = asdict(load_augmentation_config(args.augmentation_config))

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 unsupported; use fp32")

    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new or empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(min(4, torch.get_num_threads()))
    native_options = (
        {
            "data_mode": spec.mode,
            "manifest_csv": config.get("data_manifest", "manifests/patch_manifest.csv"),
        }
        if spec.mode == "native512"
        else {}
    )

    gate = run_equivalence(
        rdn_blocks=config["metasr"].get("rdn_blocks", 1),
        **({"scales": (2, 3, 4, 16)} if native_options else {}),
    )
    (args.output / "equivalence.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(f"Decoder equivalence gate passed: {gate['selected_config']}", flush=True)
    print(
        f"[shared] mil={config['mil_pooling']} lambda_sr={config.get('lambda_sr')} "
        f"rdn_blocks={config['metasr'].get('rdn_blocks')}",
        flush=True,
    )

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False

    train_data, val_data = load_fold_datasets(
        args.data_root,
        args.fold,
        args.labels_csv,
        args.split_dir,
        require_hr=float(config.get("lambda_sr", 0.0)) > 0,
        augmentation=config.get("augmentation"),
        **native_options,
    )
    if set(s["label"] for s in val_data.samples) != {0, 1}:
        raise ValueError("Validation needs both classes")

    loaders = [
        DataLoader(data, batch_size=1, shuffle=shuffle, collate_fn=collate_one_wsi)
        for data, shuffle in (
            (train_data, True),
            (train_data.evaluation_view(), False),
            (val_data, False),
        )
    ]

    m = config["metasr"]
    model = SharedSRClsModel(
        mil_pooling=config["mil_pooling"],
        classification_head=config["classification_head"],
        embed_dropout=config.get("embed_dropout", 0.3),
        mil_dropout=config.get("mil_dropout", 0.3),
        scale=m.get("scale", 16),
        rdn_blocks=m.get("rdn_blocks", 1),
        checkpoint_rdb=m.get("checkpoint_rdb", True),
        rgb_range=m.get("rgb_range", 1),
        freeze_meanshift=m.get("freeze_meanshift", True),
        lr_chunk_size=m.get("lr_chunk_size", 256),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = build_scheduler(optimizer, config)

    data_info = provenance(args.data_root, train_data)
    (args.output / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (args.output / "data_provenance.json").write_text(
        json.dumps(data_info, indent=2), encoding="utf-8"
    )

    info = run_metadata(args.fold, device)
    info["model"] = CHECKPOINT_FORMAT
    info_file = args.output / "run_info.json"
    info_file.write_text(json.dumps(info, indent=2), encoding="utf-8")

    best_auc, stale, iteration = -float("inf"), 0, 0
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for epoch in range(1, int(config["epochs"]) + 1):
            online = []
            for sample in loaders[0]:
                iteration += 1
                result = train_wsi(
                    model, sample, optimizer, device, config, iteration % 10 == 0
                )
                online.append(result)
                print(
                    f"[WSI] epoch={epoch} iteration={iteration} slide={sample['slide_id']} "
                    f"N={sample['n_regions']} {result}",
                    flush=True,
                )

            train_result = evaluate(model, loaders[1], device, config)
            val_result = evaluate(model, loaders[2], device, config)
            row = {
                "epoch": epoch,
                **{
                    k: sum(r[k] for r in online) / len(online)
                    for k in ("train_loss_cls", "train_loss_sr", "train_loss_total")
                },
                **{f"train_{k}": train_result[k] for k in ("auc", "acc", "bacc")},
                "train_eval_loss": train_result["loss"],
                **{f"val_{k}": val_result[k] for k in ("loss", "auc", "acc", "bacc")},
                **{
                    k: max((r[k] or 0) for r in online)
                    for k in ("max_memory_allocated", "max_memory_reserved")
                },
                "lr": optimizer.param_groups[0]["lr"],
            }
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()

            improved = val_result["auc"] > best_auc
            if improved:
                best_auc, stale = val_result["auc"], 0
            else:
                stale += 1

            saved = {
                "format": CHECKPOINT_FORMAT,
                "architecture": model.architecture(),
                "data_spec": spec.metadata(),
                "epoch": epoch,
                "fold": args.fold,
                "config": config,
                "data_provenance": data_info,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_auc": best_auc,
            }
            torch.save(saved, args.output / "last.pth")
            if improved:
                torch.save(saved, args.output / "best.pth")
                with (args.output / "best_val_predictions.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as out:
                    preds = val_result["predictions"]
                    table = csv.DictWriter(out, fieldnames=list(preds[0]))
                    table.writeheader()
                    table.writerows(preds)

            print(f"[epoch] {row}", flush=True)
            if scheduler is not None:
                scheduler.step()
            if stale >= int(config["early_stopping_patience"]):
                break

    info["status"] = (
        "early_stopped"
        if stale >= int(config["early_stopping_patience"])
        else "completed"
    )
    info["finished_at"] = datetime.now().astimezone().isoformat()
    info_file.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(
        f"[run] {info['experiment_id']} {info['status']} best_val_auc={best_auc}",
        flush=True,
    )


if __name__ == "__main__":
    main()