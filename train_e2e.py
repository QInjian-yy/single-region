import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from augmentation import AugmentationConfig, load_augmentation_config
from data_modes import data_spec, validate_data_config
from engine import evaluate, train_wsi
from experiments.summarize import summarize
from models.baseline import MetaSRABMIL
from models.edsr_sr import run_compact_sr_gate
from validation import run_equivalence
from wsi_data import collate_one_wsi, load_fold_datasets

ROOT = Path(__file__).resolve().parent


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    config.setdefault("classification_encoder", "resnet18")
    encoders = (
        "resnet18", "spp", "spp_ln", "spp_avg", "spp_1344",
        "dual_resnet18", "dual_resnet50",
    )
    if config["classification_encoder"] not in encoders:
        raise ValueError("classification_encoder must be one of " + ", ".join(encoders))
    config.setdefault("classification_head", "linear")
    if config["classification_head"] not in ("linear", "mlp"):
        raise ValueError("classification_head must be linear or mlp")
    if config["classification_head"] == "mlp" and config["classification_encoder"] != "spp":
        raise ValueError("classification_head=mlp requires classification_encoder=spp")
    config.setdefault("sr_model", "rdn")
    if config["sr_model"] not in ("rdn", "edsr"):
        raise ValueError("sr_model must be rdn or edsr")
    classification = config.setdefault("classification", {})
    if not isinstance(classification, dict):
        raise ValueError("classification must be a mapping")
    if config["classification_encoder"].startswith("dual_"):
        classification.setdefault("pretrained", True)
        classification.setdefault("trainable", "layer4")
        classification.setdefault("embedding_dim", 256)
        classification.setdefault("region_dropout", 0.2)
        classification.setdefault("attention_hidden_dim", 128)
        classification.setdefault("topk", 5)
        classification.setdefault("bag_region_dropout", 0.1)
        classification.setdefault("attention_dropout", 0.1)
        classification.setdefault("classifier_dropout", 0.3)
        classification.setdefault("freeze_backbone_epochs", 0)
        if not isinstance(classification["pretrained"], bool):
            raise ValueError("classification.pretrained must be a boolean")
        if classification["trainable"] not in ("frozen", "layer4", "all"):
            raise ValueError("classification.trainable must be frozen, layer4 or all")
        for name in ("embedding_dim", "attention_hidden_dim", "topk"):
            value = classification[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"classification.{name} must be a positive integer")
        freeze_epochs = classification["freeze_backbone_epochs"]
        if isinstance(freeze_epochs, bool) or not isinstance(freeze_epochs, int) or freeze_epochs < 0:
            raise ValueError("classification.freeze_backbone_epochs must be a nonnegative integer")
        if freeze_epochs > config["epochs"]:
            raise ValueError("classification.freeze_backbone_epochs cannot exceed epochs")
    spec = validate_data_config(config)
    if config["precision"] not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    if config["sr_model"] == "rdn":
        if not isinstance(config["metasr"]["checkpoint_rdb"], bool):
            raise ValueError("checkpoint_rdb must be a boolean")
        config["metasr"].setdefault("freeze_meanshift", False)
        if not isinstance(config["metasr"]["freeze_meanshift"], bool):
            raise ValueError("metasr.freeze_meanshift must be a boolean")
        rdn_blocks = config["metasr"].get("rdn_blocks", 16)
        if isinstance(rdn_blocks, bool) or not isinstance(rdn_blocks, int) or not 1 <= rdn_blocks <= 16:
            raise ValueError("metasr.rdn_blocks must be an integer from 1 to 16")
    training = config["training"]
    if not isinstance(training["use_region_microbatch"], bool):
        raise ValueError("training.use_region_microbatch must be a boolean")
    for name in ("region_microbatch_size",):
        value = training[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    max_regions = training.get("max_train_regions")
    if max_regions is not None and (
            isinstance(max_regions, bool) or not isinstance(max_regions, int)
            or max_regions < 1):
        raise ValueError("training.max_train_regions must be a positive integer or null")
    training.setdefault("class_weight", "none")
    if training["class_weight"] not in ("none", "balanced"):
        raise ValueError("training.class_weight must be none or balanced")
    for name in ("sr_train_crop", "epochs", "early_stopping_patience"):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config["sr_train_crop"] >= spec.hr_size:
        raise ValueError("Training requires a crop smaller than 8192; use infer_sr.py for full reconstruction")
    if not math.isfinite(config["lambda_sr"]) or config["lambda_sr"] < 0:
        raise ValueError("lambda_sr must be finite and nonnegative")
    config.setdefault("label_smoothing", 0.0)
    if not math.isfinite(config["label_smoothing"]) or not 0 <= config["label_smoothing"] < 0.5:
        raise ValueError("label_smoothing must be finite and in [0, 0.5)")
    config.setdefault("optimizer", "adamw")
    if config["optimizer"] not in ("adam", "adamw"):
        raise ValueError("optimizer must be adam or adamw")
    config.setdefault("gradient_clip_norm", 0.0)
    if not math.isfinite(config["gradient_clip_norm"]) or config["gradient_clip_norm"] < 0:
        raise ValueError("gradient_clip_norm must be finite and nonnegative")
    config.setdefault("checkpoint_metric", "val_auc")
    if config["checkpoint_metric"] not in ("val_auc", "val_loss"):
        raise ValueError("checkpoint_metric must be val_auc or val_loss")
    config.setdefault("early_stopping_min_delta", 0.0)
    if not math.isfinite(config["early_stopping_min_delta"]) or config["early_stopping_min_delta"] < 0:
        raise ValueError("early_stopping_min_delta must be finite and nonnegative")
    config.setdefault("min_epochs", 1)
    if isinstance(config["min_epochs"], bool) or not isinstance(config["min_epochs"], int) or config["min_epochs"] < 1:
        raise ValueError("min_epochs must be a positive integer")
    if config["min_epochs"] > config["epochs"]:
        raise ValueError("min_epochs cannot exceed epochs")
    config.setdefault("scheduler", "none")
    if config["scheduler"] not in ("none", "plateau"):
        raise ValueError("scheduler must be none or plateau")
    config.setdefault("scheduler_patience", 5)
    if isinstance(config["scheduler_patience"], bool) or not isinstance(config["scheduler_patience"], int) \
            or config["scheduler_patience"] < 1:
        raise ValueError("scheduler_patience must be a positive integer")
    augmentation = config.get("augmentation", {})
    if not isinstance(augmentation, dict):
        raise ValueError("augmentation must be a mapping")
    AugmentationConfig(**augmentation)
    return config


def build_optimizer(model, config):
    """AdamW with separate low-LR backbone and SR groups plus decay filtering."""
    classification = config.get("classification") or {}
    base_lr = float(config["lr"])
    rates = {
        "backbone": float(classification.get("backbone_lr", min(base_lr, 1e-5))),
        "sr": float(config.get("sr_lr", base_lr)),
        "head": float(classification.get("head_lr", base_lr)),
    }
    weight_decay = float(config["weight_decay"])
    groups = {
        (name, decay): [] for name in rates for decay in (True, False)
    }
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group_name = (
            "backbone" if parameter_name.startswith("region_encoder.backbone.")
            else "sr" if parameter_name.startswith("sr.")
            else "head"
        )
        decay = parameter.ndim > 1 and not parameter_name.endswith(".bias")
        groups[(group_name, decay)].append(parameter)
    options = []
    for (group_name, decay), parameters in groups.items():
        if parameters:
            options.append({
                "params": parameters,
                "lr": rates[group_name],
                "weight_decay": weight_decay if decay else 0.0,
                "name": group_name + ("_decay" if decay else "_no_decay"),
            })
    constructor = (
        torch.optim.AdamW if config.get("optimizer", "adamw") == "adamw"
        else torch.optim.Adam
    )
    return constructor(options, betas=(0.9, 0.999))


def configure_class_weights(config, dataset):
    if (config.get("training") or {}).get("class_weight", "none") not in ("balanced", "wsi_balanced"):
        config.pop("class_weight_values", None)
        return
    counts = Counter(sample["label"] for sample in dataset.samples)
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError("Balanced class loss requires both classes in the training split")
    total = sum(counts.values())
    weights = [total / (2 * counts[label]) for label in (0, 1)]
    mean = sum(weights) / 2
    config["class_weight_values"] = [value / mean for value in weights]


def provenance(data_root, dataset):
    paths = {"manifest": getattr(dataset, "manifest_csv", Path(data_root) / "manifests/patch_manifest.csv"),
             "labels": dataset.labels_csv, "split": dataset.split_csv}
    if hasattr(dataset, "verification_json"):
        paths["verification"] = dataset.verification_json
    return {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in paths.items()}


def run_metadata(fold, device):
    started = datetime.now().astimezone()
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True)
        commit = result.stdout.strip() if result.returncode == 0 else ""
    except FileNotFoundError:
        commit = ""
    return {"experiment_id": "EXP-" + started.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:12],
            "started_at": started.isoformat(), "fold": fold, "git_commit": commit,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "status": "incomplete"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.yaml")
    parser.add_argument("--augmentation-config", type=Path,
                        help="Optional geometry YAML; overrides the main config's augmentation mapping")
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
            raise RuntimeError("BF16 unsupported on this GPU; configure fp32")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new or empty output directory to preserve existing runs")
    args.output.mkdir(parents=True, exist_ok=True)
    # Check the selected depth against the vendored forward before training.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    native_options = ({"data_mode": spec.mode,
                       "manifest_csv": config.get("data_manifest", "manifests/patch_manifest.csv")}
                      if spec.mode == "native512" else {})
    sr_model = config.get("sr_model", "rdn")
    if sr_model == "rdn":
        gate = run_equivalence(
            rdn_blocks=config["metasr"].get("rdn_blocks", 16),
            **({"scales": (2, 3, 4, 16)} if native_options else {}),
        )
    else:
        gate = run_compact_sr_gate(scale=config["metasr"]["scale"])
    (args.output / "equivalence.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(f"Decoder equivalence and checkpoint gate passed: {sr_model}", flush=True)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    train_data, val_data = load_fold_datasets(args.data_root, args.fold, args.labels_csv,
        args.split_dir, require_hr=config["lambda_sr"] > 0, augmentation=config.get("augmentation"),
        max_train_regions=(config.get("training") or {}).get("max_train_regions"),
        split_scheme=config.get("split_scheme", "3fold"),
        **native_options)
    if set(sample["label"] for sample in val_data.samples) != {0, 1}:
        raise ValueError("Validation needs both classes for val_auc checkpoint selection")
    configure_class_weights(config, train_data)
    loaders = [DataLoader(data, batch_size=1, shuffle=shuffle, collate_fn=collate_one_wsi)
               for data, shuffle in ((train_data, True), (train_data.evaluation_view(), False), (val_data, False))]
    model = MetaSRABMIL(classification_encoder=config["classification_encoder"],
                       classification_head=config.get("classification_head", "linear"),
                       sr_model=sr_model,
                       classification=config.get("classification"),
                       **config["metasr"]).to(device)
    optimizer = build_optimizer(model, config)
    freeze_backbone_epochs = (config.get("classification") or {}).get(
        "freeze_backbone_epochs", 0
    )
    if config["classification_encoder"].startswith("dual_") and freeze_backbone_epochs:
        # Optimizer groups already contain layer4; disabling requires_grad only
        # during warm-up lets training update it after the head is calibrated.
        model.set_backbone_trainable("frozen")
    scheduler = None
    if config.get("scheduler", "none") == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.get("scheduler_factor", 0.5),
            patience=config.get("scheduler_patience", 5),
            min_lr=config.get("min_lr", 1e-6),
        )
    checkpoint_format = (
        "metasr-abmil-v3"
        if sr_model == "edsr"
        or config["classification_encoder"].startswith("dual_")
        else "metasr-abmil-v2"
    )
    data_info = provenance(args.data_root, train_data)
    (args.output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (args.output / "data_provenance.json").write_text(json.dumps(data_info, indent=2), encoding="utf-8")
    info = run_metadata(args.fold, device)
    info_file = args.output / "run_info.json"
    info_file.write_text(json.dumps(info, indent=2), encoding="utf-8")
    checkpoint_metric = config.get("checkpoint_metric", "val_auc")
    lower_is_better = checkpoint_metric == "val_loss"
    best_score = math.inf if lower_is_better else -math.inf
    best_auc = -math.inf
    best_epoch = 0
    stale, iteration = 0, 0
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for epoch in range(1, config["epochs"] + 1):
            if (config["classification_encoder"].startswith("dual_")
                    and freeze_backbone_epochs
                    and epoch == freeze_backbone_epochs + 1):
                model.set_backbone_trainable(
                    (config.get("classification") or {}).get("trainable", "layer4")
                )
            online = []
            for sample in loaders[0]:
                iteration += 1
                result = train_wsi(model, sample, optimizer, device, config, iteration % 10 == 0)
                online.append(result)
                print(f"[WSI] epoch={epoch} iteration={iteration} slide={sample['slide_id']} "
                      f"N={sample['n_regions']} {result}", flush=True)
            train_result = evaluate(model, loaders[1], device, config)
            val_result = evaluate(model, loaders[2], device, config)
            row = {"epoch": epoch, **{key: sum(r.get(key, 0.0) for r in online) / len(online)
                   for key in ("train_loss_cls", "train_loss_sr", "train_loss_total", "grad_norm")},
                   **{"train_"+key: train_result[key] for key in ("auc", "acc", "bacc")},
                   "train_eval_loss": train_result["loss"],
                   **{"val_"+key: val_result[key] for key in ("loss", "auc", "acc", "bacc")},
                   "val_sensitivity": val_result.get("sensitivity", float("nan")),
                   "val_specificity": val_result.get("specificity", float("nan")),
                   "lr_head": next(
                       group["lr"] for group in optimizer.param_groups
                       if group["name"].startswith("head")
                   ),
                   **{key: max((r.get(key) or 0) for r in online)
                      for key in ("max_memory_allocated", "max_memory_reserved")}}
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            score = val_result["loss"] if lower_is_better else val_result["auc"]
            delta = config.get("early_stopping_min_delta", 0.0)
            improved = score < best_score - delta if lower_is_better else score > best_score + delta
            if improved:
                best_score, best_auc, best_epoch, stale = score, val_result["auc"], epoch, 0
            else:
                stale += 1
            if scheduler is not None:
                scheduler.step(val_result["loss"])
            saved = {"format": checkpoint_format, "architecture": model.architecture(),
                     "data_spec": spec.metadata(),
                     "epoch": epoch, "fold": args.fold,
                     "config": config, "data_provenance": data_info, "model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict(), "best_val_auc": best_auc,
                     "best_epoch": best_epoch, "best_selection_score": best_score,
                     "checkpoint_metric": checkpoint_metric}
            torch.save(saved, args.output / "last.pth")
            if improved:
                torch.save(saved, args.output / "best.pth")
                with (args.output / "best_val_predictions.csv").open("w", newline="", encoding="utf-8") as out:
                    predictions = val_result["predictions"]
                    table = csv.DictWriter(out, fieldnames=list(predictions[0]))
                    table.writeheader()
                    table.writerows(predictions)
            print(f"[epoch] {row}", flush=True)
            if (epoch >= config.get("min_epochs", 1)
                    and stale >= config["early_stopping_patience"]):
                break

    info["status"] = "early_stopped" if stale >= config["early_stopping_patience"] else "completed"
    info["best_epoch"] = best_epoch
    info["best_selection_score"] = best_score
    info["finished_at"] = datetime.now().astimezone().isoformat()
    info_file.write_text(json.dumps(info, indent=2), encoding="utf-8")
    try:
        result = summarize(args.output)
        print(f"[summary] {result['experiment_id']} val_auc={result['val_auc']}", flush=True)
    except (OSError, ValueError) as error:
        print(f"[summary] Not saved: {error}. Run experiments/summarize.py with the output directory.",
              flush=True)


if __name__ == "__main__":
    main()
