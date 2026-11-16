"""Isolated CAMIL five-fold E2E AttentionMIL + AdamW training with optional AEM."""

import argparse
import copy
import csv
from datetime import datetime
import math
from pathlib import Path
import random
from statistics import mean, stdev

import numpy as np
import torch
import yaml

import engine
import engine_aem
import train_e2e_camil as camil
from data_modes import data_spec
from models.shared_resnet_sr_asrd import build_model, normalize_region_interaction


def load_config(path):
    config = camil.load_config(path)
    classification = config["classification"]
    settings = engine_aem.normalize_aem(classification.get("aem"))
    if settings["enabled"]:
        if config["optimizer"] != "adamw":
            raise ValueError("This AEM entry requires optimizer: adamw when AEM is enabled")
        if classification["pooling"] != ["attention"]:
            raise ValueError("Enabled AEM requires classification.pooling: [attention]")
        normalize_region_interaction(classification.get("region_interaction"))
    classification["aem"] = settings
    return config


def train_fold(config, protocol, samples, provenance, fold, output, device):
    settings = config["classification"]["aem"]
    if not settings["enabled"]:
        return camil.train_fold(config, protocol, samples, provenance, fold, output, device)
    output.mkdir()
    config = copy.deepcopy(config)
    train, val, test = camil.make_camil_datasets(
        protocol, samples, fold, augmentation=config.get("augmentation"),
        max_train_regions=config["training"]["max_train_regions"],
    )
    camil.configure_class_weights(config, train)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False

    sr = config["metasr"]
    model = build_model(
        shared_encoder=config["shared_encoder"], classification=config["classification"],
        scale=sr["scale"], rgb_range=sr["rgb_range"], sr_channels=sr["channels"],
        sr_residual_scale=sr["residual_scale"], sr_dropout=sr["dropout"], hr_tile=sr["hr_tile"],
    ).to(device)
    optimizer = camil.shared.build_optimizer(model, config)
    scheduler = camil.shared.build_scheduler(optimizer, config)
    ema = camil.shared.ModelEMA(model, config["ema_decay"]) if config["ema_decay"] > 0 else None
    eval_model = ema.module if ema is not None else model
    train_loader, train_eval_loader, val_loader = (
        camil._loader(train, shuffle=True), camil._loader(train.evaluation_view()), camil._loader(val)
    )

    regularization = dict(type="aem-negative-entropy-v1", **settings)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    camil._write_json(output / "data_provenance.json", provenance)
    info = camil.run_metadata(fold, device)
    info.update(
        model=model.architecture()["type"], optimizer=config["optimizer"],
        training_entry=Path(__file__).name, split_protocol="camil_5fold",
        training_regularization=regularization, patient_grouping=protocol.patient_grouping,
        split_counts={"train": len(train), "val": len(val), "test": len(test)},
    )
    camil._write_json(output / "run_info.json", info)

    # Keep the baseline's checkpoint, scheduler, threshold and early-stopping policy.
    best_auc, stale = -math.inf, 0
    sr_epochs = config["training"].get("sr_stop_after_epochs")
    checkpoint_path = output / "best.pth"
    with (output / "history.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = None
        for epoch in range(1, config["epochs"] + 1):
            epoch_config = dict(config)
            if sr_epochs is not None and epoch > sr_epochs:
                epoch_config["lambda_sr"] = 0.0
            steps = []
            for sample in train_loader:
                steps.append(engine_aem.train_wsi(model, sample, optimizer, device, epoch_config, epoch=epoch))
                if ema is not None:
                    ema.update(model)
            train_result = engine.evaluate(eval_model, train_eval_loader, device, config)
            val_result = engine.evaluate(eval_model, val_loader, device, config)
            threshold = camil.shared.balanced_threshold(val_result["predictions"])
            camil.shared.apply_threshold(val_result, threshold)
            auc = float(val_result["auc"])
            if not math.isfinite(auc):
                raise ValueError(f"Fold {fold} epoch {epoch} Val AUC is non-finite")
            row = dict(epoch=epoch, lambda_sr=epoch_config["lambda_sr"], val_decision_threshold=threshold,
                       aem_weight=engine_aem.aem_coefficient(settings, epoch))
            row.update({key: mean(step[key] for step in steps) for key in (
                "train_loss_cls", "train_loss_sr", "train_loss_total", "grad_norm",
                "train_loss_aem", "train_loss_aem_weighted", "train_attention_entropy",
            )})
            for name, result in (("train", train_result), ("val", val_result)):
                row.update({f"{name}_{key}": result[key] for key in ("loss", "auc", "acc", "bacc")})
            row.update(val_sensitivity=val_result["sensitivity"], val_specificity=val_result["specificity"])
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            print(f"[fold {fold}] epoch={epoch} train_auc={train_result['auc']:.6f} val_auc={auc:.6f} "
                  f"aem_weight={row['aem_weight']:.6f} attention_entropy={row['train_attention_entropy']:.6f}",
                  flush=True)

            if scheduler is not None:
                if config["scheduler"] == "plateau":
                    scheduler.step(val_result["loss"])
                else:
                    scheduler.step()
            if epoch < config["min_epochs"]:
                continue
            if auc > best_auc + config["early_stopping_min_delta"]:
                best_auc, stale = auc, 0
                torch.save({
                    "format": info["model"], "architecture": eval_model.architecture(),
                    "data_spec": data_spec(config).metadata(), "fold": fold, "epoch": epoch,
                    "config": dict(config, decision_threshold=threshold),
                    "data_provenance": provenance, "split_protocol": "camil_5fold",
                    "training_entry": Path(__file__).name, "training_regularization": regularization,
                    "epoch_aem_weight": row["aem_weight"], "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "model_state": eval_model.state_dict(), "ema_updates": ema.updates if ema else 0,
                    "best_val_auc": best_auc, "best_epoch": epoch, "best_selection_score": best_auc,
                    "checkpoint_metric": "val_auc", "epoch_lambda_sr": epoch_config["lambda_sr"],
                }, checkpoint_path)
                camil._write_predictions(output / "best_val_predictions.csv", val_result["predictions"])
            else:
                stale += 1
                if stale >= config["early_stopping_patience"]:
                    break

    best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    eval_model.load_state_dict(best["model_state"], strict=True)
    result = engine.evaluate(eval_model, camil._loader(test), device, best["config"])
    predictions = result.pop("predictions")
    result.update(
        fold=fold, split="test", checkpoint_epoch=best["epoch"], slide_count=len(test),
        best_val_auc=best["best_val_auc"], decision_threshold=best["config"]["decision_threshold"],
    )
    camil._write_predictions(output / "test_predictions.csv", predictions)
    camil._write_json(output / "test_metrics.json", result)
    info.update(
        status="early_stopped" if stale >= config["early_stopping_patience"] else "completed",
        best_epoch=best["epoch"], best_val_auc=best["best_val_auc"],
        best_decision_threshold=best["config"]["decision_threshold"],
        finished_at=datetime.now().astimezone().isoformat(),
    )
    camil._write_json(output / "run_info.json", info)
    print(f"[fold {fold}] Test AUC={result['auc']:.6f}, best_epoch={best['epoch']}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--train-root", type=Path)
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--fold", type=int, choices=camil.FOLDS, help="Default: run folds 0-4 sequentially")
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="A new run directory; existing paths are rejected")
    mode.add_argument("--check-data", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if not config["classification"]["aem"]["enabled"]:
        return camil.main(argv)
    protocol = camil.read_camil_protocol(args.protocol_dir)
    samples, provenance = camil.load_camil_samples(
        protocol, train_root=args.train_root, test_root=args.test_root, require_hr=config["lambda_sr"] > 0,
    )
    print(f"[data] {len(samples)} WSI, {sum(s['n_regions'] for s in samples.values())} Regions; "
          f"grouping={protocol.patient_grouping}", flush=True)
    if args.check_data:
        return
    output = args.output.resolve()
    protected = [args.protocol_dir.resolve()]
    protected.extend(Path(provenance[f"{s}_manifest"]["data_root"]) for s in protocol.sources)
    if any(output.is_relative_to(root) for root in protected):
        raise ValueError("Output must be outside the protocol and dataset roots")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 unsupported; use fp32")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    output.mkdir(parents=True, exist_ok=False)
    folds = camil.FOLDS if args.fold is None else (args.fold,)
    results = [train_fold(config, protocol, samples, provenance, fold, output / f"fold_{fold}", device)
               for fold in folds]
    if folds == camil.FOLDS:
        aucs = [float(result["auc"]) for result in results]
        summary = dict(fold_count=len(aucs), test_auc_mean=mean(aucs), test_auc_std=stdev(aucs), folds=results)
        camil._write_json(output / "summary.json", summary)
        print(f"[five folds] Test AUC {summary['test_auc_mean']:.6f} +/- {summary['test_auc_std']:.6f}",
              flush=True)


if __name__ == "__main__":
    main()
