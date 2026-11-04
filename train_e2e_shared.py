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
from engine import evaluate, train_wsi
from models.robust_mil import RobustMILPooling
from models.vib import normalize_vib_config
from models.shared_resnet_sr import (
    CHECKPOINT_FORMAT,
    SharedResNetSRMIL,
)
from train_e2e import ROOT, configure_class_weights, provenance, run_metadata
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
    config.setdefault("optimizer", "adamw")
    if config["optimizer"] not in ("adam", "adamw"):
        raise ValueError("optimizer must be adam or adamw")
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
    return config


def build_optimizer(model, config):
    encoder_lr = float(config["shared_encoder"]["encoder_lr"])
    sr_lr = float(config.get("sr_lr", config["lr"]))
    head_lr = float(config["classification"]["head_lr"])
    weight_decay = float(config["weight_decay"])
    buckets = {
        (name, decay): []
        for name in ("encoder", "sr", "head")
        for decay in (True, False)
    }
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = (
            "encoder"
            if parameter_name.startswith("encoder.")
            else "sr"
            if parameter_name.startswith("sr.")
            else "head"
        )
        decay = parameter.ndim > 1 and not parameter_name.endswith(".bias")
        buckets[(group, decay)].append(parameter)
    rates = {"encoder": encoder_lr, "sr": sr_lr, "head": head_lr}
    groups = []
    for (group, decay), parameters in buckets.items():
        if parameters:
            groups.append(
                {
                    "params": parameters,
                    "lr": rates[group],
                    "weight_decay": weight_decay if decay else 0.0,
                    "name": group + ("_decay" if decay else "_no_decay"),
                }
            )
    constructor = torch.optim.AdamW if config["optimizer"] == "adamw" else torch.optim.Adam
    return constructor(groups, betas=(0.9, 0.999))


def build_scheduler(optimizer, config):
    if config["scheduler"] == "none":
        return None
    if config["scheduler"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config.get("scheduler_factor", 0.5)),
            patience=int(config.get("scheduler_patience", 5)),
            min_lr=float(config["min_lr"]),
        )
    warmup = int(config.get("warmup_epochs", 0))
    base_rates = [group["lr"] for group in optimizer.param_groups]
    total = int(config["epochs"])
    floor = float(config["min_lr"])

    def factor(epoch):
        if warmup and epoch < warmup:
            return float(epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, total - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return max(floor / max(base_rates), cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class ModelEMA:
    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        self.updates = 0
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        if self.decay <= 0:
            return
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for ema_value, live_value in zip(
            self.module.state_dict().values(), model.state_dict().values()
        ):
            if ema_value.is_floating_point():
                ema_value.lerp_(live_value.detach(), 1.0 - decay)
            else:
                ema_value.copy_(live_value)


def balanced_threshold(predictions):
    """Choose a validation threshold without allowing an all-normal collapse."""
    labels = np.asarray([row["label"] for row in predictions], dtype=np.int64)
    scores = np.asarray([row["p_tumor"] for row in predictions], dtype=np.float64)
    unique = np.unique(scores)
    candidates = {0.5, 0.0, 1.0}
    candidates.update((unique[:-1] + unique[1:]) / 2.0 if len(unique) > 1 else [])
    best = None
    for threshold in sorted(candidates):
        decisions = scores >= threshold
        positives = labels == 1
        negatives = ~positives
        sensitivity = decisions[positives].mean() if positives.any() else 1.0
        specificity = (~decisions[negatives]).mean() if negatives.any() else 1.0
        bacc = 0.5 * (sensitivity + specificity)
        key = (bacc, min(sensitivity, specificity), -abs(float(threshold) - 0.5))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1]


def apply_threshold(result, threshold):
    predictions = result["predictions"]
    labels = [row["label"] for row in predictions]
    decisions = [int(row["p_tumor"] >= threshold) for row in predictions]
    tn = sum(label == 0 and decision == 0 for label, decision in zip(labels, decisions))
    fp = sum(label == 0 and decision == 1 for label, decision in zip(labels, decisions))
    fn = sum(label == 1 and decision == 0 for label, decision in zip(labels, decisions))
    tp = sum(label == 1 and decision == 1 for label, decision in zip(labels, decisions))
    result["acc"] = (tn + tp) / len(labels)
    result["sensitivity"] = tp / (tp + fn) if tp + fn else float("nan")
    result["specificity"] = tn / (tn + fp) if tn + fp else float("nan")
    result["bacc"] = 0.5 * (result["sensitivity"] + result["specificity"])
    result["decision_threshold"] = float(threshold)
    for row, decision in zip(predictions, decisions):
        row["prediction"] = decision
    return result


def run_shared_model_gate():
    torch.manual_seed(20261102)
    config = {
        "data_mode": "native512",
        "metasr": {"scale": 2, "rgb_range": 1},
    }
    model = SharedResNetSRMIL(
        shared_encoder={"pretrained": False, "trainable": "layer4"},
        classification={"embedding_dim": 32, "attention_hidden_dim": 16, "topk": 2},
        scale=2,
        sr_channels=16,
    )
    model.train()
    lr = torch.rand(2, 3, 16, 16)
    target = torch.tensor([1])
    sr_feature, embeddings = model.encode_regions(lr)
    logits = model.forward_embeddings(embeddings[:1])
    sr_crop = model.sr.decode_crop(sr_feature[:1], (3, 5, 12, 14))
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss = loss + torch.nn.functional.l1_loss(
        sr_crop, torch.rand_like(sr_crop)
    )
    loss.backward()
    for name in (
        "encoder.layer4.0.conv1.weight",
        "classifier_head.projection.1.weight",
        "sr.fuse.0.weight",
    ):
        parameter = dict(model.named_parameters())[name]
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Shared-model gate failed for {name}")
    return {
        "passed": True,
        "embedding_shape": list(embeddings.shape),
        "sr_crop_shape": list(sr_crop.shape),
        "data_contract": config,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--augmentation-config", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--labels-csv", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--split-scheme", choices=("3fold", "5fold"))
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
    gate = run_shared_model_gate()
    (args.output / "equivalence.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    print(f"Shared encoder gate passed: {gate}", flush=True)

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
    scheduler = build_scheduler(optimizer, config)
    ema = ModelEMA(model, config["ema_decay"])

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
            if model.vib_enabled:
                row.update(
                    vib_beta=model.vib_beta,
                    **{key: sum(r[key] for r in online) / len(online)
                       for key in ("train_loss_kl", "train_loss_vib")},
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
