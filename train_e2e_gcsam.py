"""Isolated GCSAM + AdamW training, with the shared-model evaluation protocol.

The baseline validation/main-loop code is deliberately copied to keep the
original entry and engine untouched. Shared utilities are imported unchanged.
"""

from __future__ import annotations

import argparse
import copy
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
from engine import evaluate
from engine_gcsam import train_wsi
from models.robust_mil import RobustMILPooling
from models.shared_resnet_sr import CHECKPOINT_FORMAT, SharedResNetSRMIL
from models.vib import normalize_vib_config
from optimizers.gcsam import GCSAM
from train_e2e import ROOT, configure_class_weights, provenance, run_metadata
from train_e2e_shared import (
    ModelEMA,
    apply_threshold,
    balanced_threshold,
    build_optimizer as build_adamw_optimizer,
    build_scheduler,
)
from wsi_data import collate_one_wsi, load_fold_datasets


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_data_config(config)
    if config.get("precision") not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")

    encoder = config.setdefault("shared_encoder", {})
    if encoder.get("backbone", "resnet18") != "resnet18":
        raise ValueError("shared_encoder.backbone must be resnet18")
    encoder.setdefault("pretrained", True)
    encoder.setdefault("trainable", "layer4")
    encoder.setdefault("freeze_bn", True)
    if encoder["trainable"] not in ("frozen", "layer3", "layer4", "all"):
        raise ValueError("shared_encoder.trainable must be frozen, layer3, layer4, or all")

    classification = config.setdefault("classification", {})
    classification["pooling"] = list(RobustMILPooling.normalize_pooling(
        classification.get("pooling", RobustMILPooling.POOLING_BRANCHES)
    ))
    classification.setdefault("embedding_dim", 256)
    classification.setdefault("attention_hidden_dim", 128)
    classification.setdefault("topk", 5)
    classification.setdefault("region_dropout", 0.2)
    classification.setdefault("attention_dropout", 0.2)
    classification.setdefault("projection_dropout", 0.2)
    classification.setdefault("classifier_dropout", 0.4)
    classification.setdefault("head_lr", 2e-4)
    classification.setdefault("encoder_lr", 1e-5)
    classification["vib"] = normalize_vib_config(classification.get("vib"))

    metasr = config.setdefault("metasr", {})
    metasr.setdefault("channels", 64)
    metasr.setdefault("residual_scale", 0.1)
    metasr.setdefault("dropout", 0.05)
    metasr.setdefault("hr_tile", 256)

    training = config.setdefault("training", {})
    training.setdefault("use_region_microbatch", True)
    training.setdefault("region_microbatch_size", 2)
    training.setdefault("class_weight", "balanced")
    training.setdefault("max_train_regions", 24)
    sr_epochs = training.get("sr_stop_after_epochs")
    if sr_epochs is not None and (type(sr_epochs) is not int or sr_epochs < 1):
        raise ValueError("training.sr_stop_after_epochs must be a positive integer or null")
    if training["class_weight"] not in ("none", "balanced"):
        raise ValueError("training.class_weight must be none or balanced")

    for name in (
        "lambda_sr",
        "lr",
        "weight_decay",
        "label_smoothing",
        "gradient_clip_norm",
        "min_lr",
        "early_stopping_min_delta",
        "ema_decay",
    ):
        value = config.get(name)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number")
    if float(config["lambda_sr"]) < 0:
        raise ValueError("lambda_sr must be nonnegative")
    if not 0 <= float(config["label_smoothing"]) < 0.5:
        raise ValueError("label_smoothing must be in [0, 0.5)")
    if not 0 <= float(config["ema_decay"]) < 1:
        raise ValueError("ema_decay must be in [0, 1)")

    for name in ("epochs", "min_epochs", "early_stopping_patience", "sr_train_crop"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config["min_epochs"] > config["epochs"]:
        raise ValueError("min_epochs cannot exceed epochs")
    if config["sr_train_crop"] >= data_spec(config).hr_size:
        raise ValueError("sr_train_crop must be smaller than the HR image")
    config.setdefault("optimizer", "gcsam")
    if config["optimizer"] != "gcsam":
        raise ValueError("The isolated GCSAM entry requires optimizer: gcsam")
    config.setdefault("checkpoint_metric", "val_auc")
    if config["checkpoint_metric"] not in (
        "val_auc",
        "val_loss",
        "val_balanced_accuracy",
    ):
        raise ValueError(
            "checkpoint_metric must be val_auc, val_loss, or val_balanced_accuracy"
        )
    config.setdefault("scheduler", "plateau")
    if config["scheduler"] not in ("none", "plateau", "cosine"):
        raise ValueError("scheduler must be none, plateau, or cosine")
    gcsam = config.setdefault("gcsam", {})
    if not isinstance(gcsam, dict):
        raise ValueError("gcsam must be a mapping")
    gcsam.setdefault("enabled", True)
    gcsam.setdefault("base_optimizer", "adamw")
    gcsam.setdefault("rho", 0.05)
    gcsam.setdefault("start_epoch", 1)
    if gcsam["enabled"] is not True or gcsam["base_optimizer"] != "adamw":
        raise ValueError("GCSAM must be enabled with base_optimizer: adamw")
    if type(gcsam["start_epoch"]) is not int or gcsam["start_epoch"] != 1:
        raise ValueError("gcsam.start_epoch must be 1; no AdamW warm-start")
    rho = gcsam["rho"]
    if (isinstance(rho, bool) or not isinstance(rho, (int, float))
            or not math.isfinite(rho) or rho < 0):
        raise ValueError("gcsam.rho must be finite and nonnegative")
    gcsam["rho"] = float(rho)
    if float(config["lambda_sr"]) != 0 or classification["vib"]["enabled"]:
        raise ValueError("GCSAM requires lambda_sr: 0 and VIB disabled")
    for name in ("region_dropout", "attention_dropout", "projection_dropout", "classifier_dropout"):
        if classification[name] != 0:
            raise ValueError(f"GCSAM requires classification.{name}: 0")
    if metasr["dropout"] != 0:
        raise ValueError("GCSAM requires metasr.dropout: 0")
    if encoder["trainable"] != "layer4" or encoder["freeze_bn"] is not True:
        raise ValueError("GCSAM requires layer4 fine-tuning and frozen BN")
    config.setdefault("split_scheme", "5fold")
    if config["split_scheme"] != "5fold":
        raise ValueError("GCSAM requires split_scheme: 5fold")
    return config


def build_optimizer(model, config):
    # Reuse the baseline's exact parameter groups, LR and decay exclusions.
    base = build_adamw_optimizer(model, dict(config, optimizer="adamw"))
    return GCSAM(base, rho=config["gcsam"]["rho"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--augmentation-config", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--labels-csv", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--split-scheme", choices=("5fold",))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.split_scheme is not None:
        config["split_scheme"] = args.split_scheme
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
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    native_options = (
        {
            "data_mode": spec.mode,
            "manifest_csv": config.get("data_manifest", "manifests/patch_manifest.csv"),
        }
        if spec.mode == "native512"
        else {}
    )
    train_data, val_data = load_fold_datasets(
        args.data_root,
        args.fold,
        args.labels_csv,
        args.split_dir,
        require_hr=float(config["lambda_sr"]) > 0,
        augmentation=config.get("augmentation"),
        max_train_regions=config["training"]["max_train_regions"],
        split_scheme=config.get("split_scheme", "3fold"),
        **native_options,
    )
    if {sample["label"] for sample in val_data.samples} != {0, 1}:
        raise ValueError("Validation needs both classes")
    configure_class_weights(config, train_data)
    loaders = [
        DataLoader(data, batch_size=1, shuffle=shuffle, collate_fn=collate_one_wsi)
        for data, shuffle in (
            (train_data, True),
            (train_data.evaluation_view(), False),
            (val_data, False),
        )
    ]

    metasr = config["metasr"]
    model = SharedResNetSRMIL(
        shared_encoder=config["shared_encoder"],
        classification=config["classification"],
        scale=metasr["scale"],
        rgb_range=metasr["rgb_range"],
        sr_channels=metasr["channels"],
        sr_residual_scale=metasr["residual_scale"],
        sr_dropout=metasr["dropout"],
        hr_tile=metasr["hr_tile"],
    ).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer.base_optimizer, config)
    ema = ModelEMA(model, config["ema_decay"])
    print(f"[optimizer] GCSAM + AdamW from epoch 1; rho={optimizer.rho}; "
          "GC=ascent-only; normalization=after_gc", flush=True)

    data_info = provenance(args.data_root, train_data)
    (args.output / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (args.output / "data_provenance.json").write_text(
        json.dumps(data_info, indent=2), encoding="utf-8"
    )
    info = run_metadata(args.fold, device)
    info["model"] = CHECKPOINT_FORMAT
    info.update(optimizer="gcsam", base_optimizer="adamw", rho=optimizer.rho,
                training_entry="train_e2e_gcsam.py")
    info_file = args.output / "run_info.json"
    info_file.write_text(json.dumps(info, indent=2), encoding="utf-8")

    lower_is_better = config["checkpoint_metric"] == "val_loss"
    best_score = math.inf if lower_is_better else -math.inf
    best_auc = -math.inf
    best_epoch = 0
    best_threshold = 0.5
    stale = 0
    iteration = 0
    sr_epochs = config["training"].get("sr_stop_after_epochs")
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for epoch in range(1, config["epochs"] + 1):
            epoch_config = dict(config)
            if sr_epochs is not None and epoch > sr_epochs:
                epoch_config["lambda_sr"] = 0.0
            if sr_epochs is not None and epoch == sr_epochs + 1 and config["lambda_sr"] > 0:
                print(f"[SR] disabled after {sr_epochs} epochs", flush=True)
            online = []
            for sample in loaders[0]:
                iteration += 1
                result = train_wsi(
                    model, sample, optimizer, device, epoch_config, iteration % 10 == 0
                )
                ema.update(model)
                online.append(result)
                print(
                    f"[WSI] epoch={epoch} iteration={iteration} "
                    f"slide={sample['slide_id']} N={sample['n_regions']} {result}",
                    flush=True,
                )

            train_result = evaluate(ema.module, loaders[1], device, config)
            val_result = evaluate(ema.module, loaders[2], device, config)
            threshold = balanced_threshold(val_result["predictions"])
            apply_threshold(val_result, threshold)
            row = {
                "epoch": epoch,
                "lambda_sr": epoch_config["lambda_sr"],
                **{
                    key: sum(r.get(key, 0.0) for r in online) / len(online)
                    for key in (
                        "train_loss_cls",
                        "train_loss_sr",
                        "train_loss_total",
                        "grad_norm",
                    )
                },
                **{
                    f"train_{key}": train_result[key]
                    for key in ("auc", "acc", "bacc")
                },
                "train_eval_loss": train_result["loss"],
                **{
                    f"val_{key}": val_result[key]
                    for key in ("loss", "auc", "acc", "bacc")
                },
                "val_sensitivity": val_result["sensitivity"],
                "val_specificity": val_result["specificity"],
                "val_decision_threshold": threshold,
                "lr_head": next(
                    group["lr"]
                    for group in optimizer.param_groups
                    if group["name"].startswith("head")
                ),
                **{
                    key: max((r.get(key) or 0) for r in online)
                    for key in ("max_memory_allocated", "max_memory_reserved")
                },
            }
            row.update(
                optimizer="gcsam", base_optimizer="adamw", rho=optimizer.rho,
                lr_encoder=next(group["lr"] for group in optimizer.param_groups
                                if group["name"].startswith("encoder")),
                **{key: sum(r[key] for r in online) / len(online)
                   for key in ("train_loss_cls_first", "train_loss_cls_second",
                               "grad_norm_first", "grad_norm_gc", "grad_norm_second")},
            )
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()

            score = {
                "val_auc": val_result["auc"],
                "val_balanced_accuracy": val_result["bacc"],
                "val_loss": val_result["loss"],
            }[config["checkpoint_metric"]]
            improved = (
                score < best_score - config["early_stopping_min_delta"]
                if lower_is_better
                else score > best_score + config["early_stopping_min_delta"]
            )
            eligible_for_selection = epoch >= config["min_epochs"]
            if improved and eligible_for_selection:
                best_score = score
                best_auc = val_result["auc"]
                best_epoch = epoch
                best_threshold = threshold
                stale = 0
            elif eligible_for_selection:
                stale += 1

            if config["scheduler"] == "plateau":
                scheduler.step(val_result["loss"])
            elif scheduler is not None:
                scheduler.step()

            checkpoint_config = copy.deepcopy(config)
            checkpoint_config["decision_threshold"] = threshold
            common = {
                "format": CHECKPOINT_FORMAT,
                "architecture": ema.module.architecture(),
                "data_spec": spec.metadata(),
                "epoch": epoch,
                "fold": args.fold,
                "config": checkpoint_config,
                "data_provenance": data_info,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "ema_updates": ema.updates,
                "best_val_auc": best_auc,
                "best_epoch": best_epoch,
                "best_selection_score": best_score,
                "checkpoint_metric": config["checkpoint_metric"],
                "epoch_lambda_sr": epoch_config["lambda_sr"],
            }
            torch.save(
                {
                    **common,
                    "model_state": model.state_dict(),
                    "ema_model_state": ema.module.state_dict(),
                },
                args.output / "last.pth",
            )
            if improved and eligible_for_selection:
                best_config = copy.deepcopy(checkpoint_config)
                best_config["decision_threshold"] = best_threshold
                torch.save(
                    {
                        **common,
                        "config": best_config,
                        "model_state": ema.module.state_dict(),
                    },
                    args.output / "best.pth",
                )
                with (args.output / "best_val_predictions.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as out:
                    predictions = val_result["predictions"]
                    table = csv.DictWriter(out, fieldnames=list(predictions[0]))
                    table.writeheader()
                    table.writerows(predictions)
            print(f"[epoch] {row}", flush=True)
            if (
                epoch >= config["min_epochs"]
                and stale >= config["early_stopping_patience"]
            ):
                break

    info["status"] = (
        "early_stopped"
        if stale >= config["early_stopping_patience"]
        else "completed"
    )
    info["best_epoch"] = best_epoch
    info["best_val_auc"] = best_auc
    info["best_decision_threshold"] = best_threshold
    info["best_selection_score"] = best_score
    info["finished_at"] = datetime.now().astimezone().isoformat()
    info_file.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(
        f"[run] {info['experiment_id']} {info['status']} "
        f"best_epoch={best_epoch} best_val_auc={best_auc} "
        f"threshold={best_threshold}",
        flush=True,
    )


if __name__ == "__main__":
    main()
