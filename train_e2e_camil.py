"""Train the existing Native512 E2E models on the CAMIL-style pooled five folds."""

import argparse
import copy
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import random
from statistics import mean, stdev

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

import engine
import engine_gcsam
import train_e2e_gcsam as gcsam
import train_e2e_shared as shared
from camil_data import FOLDS, load_camil_samples, make_camil_datasets, read_camil_protocol
from data_modes import data_spec
from models.shared_resnet_sr import SharedResNetSRMIL
from train_e2e import configure_class_weights, run_metadata
from wsi_data import collate_one_wsi


def load_config(path):
    recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict):
        raise ValueError("Training config must be a YAML mapping")
    trainer = gcsam if recipe.get("optimizer") == "gcsam" else shared
    config = trainer.load_config(path)
    if config.get("data_mode") != "native512" or config["checkpoint_metric"] != "val_auc":
        raise ValueError("CAMIL requires data_mode: native512 and checkpoint_metric: val_auc")
    config["early_stopping_min_delta"] = float(config["early_stopping_min_delta"])
    if config["early_stopping_min_delta"] < 0:
        raise ValueError("early_stopping_min_delta must be nonnegative")
    config.pop("split_protocol", None)
    config.pop("decision_threshold", None)
    config["split_scheme"] = "5fold"
    return config


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_predictions(path, predictions):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)


def _loader(dataset, shuffle=False):
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, collate_fn=collate_one_wsi)


def train_fold(config, protocol, samples, provenance, fold, output, device):
    output.mkdir()
    config = copy.deepcopy(config)
    train, val, test = make_camil_datasets(
        protocol, samples, fold, augmentation=config.get("augmentation"),
        max_train_regions=config["training"]["max_train_regions"],
    )
    configure_class_weights(config, train)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False

    model_factory = SharedResNetSRMIL
    if "region_interaction" in config["classification"]:
        from models.shared_resnet_sr_asrd import build_model
        model_factory = build_model
    sr = config["metasr"]
    model = model_factory(
        shared_encoder=config["shared_encoder"], classification=config["classification"],
        scale=sr["scale"], rgb_range=sr["rgb_range"], sr_channels=sr["channels"],
        sr_residual_scale=sr["residual_scale"], sr_dropout=sr["dropout"], hr_tile=sr["hr_tile"],
    ).to(device)

    use_gcsam = config["optimizer"] == "gcsam"
    trainer = gcsam if use_gcsam else shared
    train_wsi = engine_gcsam.train_wsi if use_gcsam else engine.train_wsi
    optimizer = trainer.build_optimizer(model, config)
    scheduler = shared.build_scheduler(optimizer.base_optimizer if use_gcsam else optimizer, config)
    ema = shared.ModelEMA(model, config["ema_decay"]) if config["ema_decay"] > 0 else None
    eval_model = ema.module if ema is not None else model
    train_loader, train_eval_loader, val_loader = (
        _loader(train, shuffle=True), _loader(train.evaluation_view()), _loader(val)
    )

    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write_json(output / "data_provenance.json", provenance)
    info = run_metadata(fold, device)
    info.update(
        model=model.architecture()["type"], optimizer=config["optimizer"],
        training_entry=Path(__file__).name, split_protocol="camil_5fold",
        patient_grouping=protocol.patient_grouping,
        split_counts={"train": len(train), "val": len(val), "test": len(test)},
    )
    if use_gcsam:
        info.update(base_optimizer="adamw", rho=optimizer.rho)
    _write_json(output / "run_info.json", info)

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
                steps.append(train_wsi(model, sample, optimizer, device, epoch_config))
                if ema is not None:
                    ema.update(model)

            train_result = engine.evaluate(eval_model, train_eval_loader, device, config)
            val_result = engine.evaluate(eval_model, val_loader, device, config)
            threshold = shared.balanced_threshold(val_result["predictions"])
            shared.apply_threshold(val_result, threshold)
            auc = float(val_result["auc"])
            if not math.isfinite(auc):
                raise ValueError(f"Fold {fold} epoch {epoch} Val AUC is non-finite")
            row = dict(epoch=epoch, lambda_sr=epoch_config["lambda_sr"],
                       val_decision_threshold=threshold)
            row.update({key: mean(step[key] for step in steps)
                        for key in ("train_loss_cls", "train_loss_sr", "train_loss_total", "grad_norm")})
            for name, result in (("train", train_result), ("val", val_result)):
                row.update({f"{name}_{key}": result[key] for key in ("loss", "auc", "acc", "bacc")})
            row.update(val_sensitivity=val_result["sensitivity"], val_specificity=val_result["specificity"])
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            print(f"[fold {fold}] epoch={epoch} train_auc={train_result['auc']:.6f} val_auc={auc:.6f}",
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
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "model_state": eval_model.state_dict(), "ema_updates": ema.updates if ema else 0,
                    "best_val_auc": best_auc, "best_epoch": epoch, "best_selection_score": best_auc,
                    "checkpoint_metric": "val_auc", "epoch_lambda_sr": epoch_config["lambda_sr"],
                }, checkpoint_path)
                _write_predictions(output / "best_val_predictions.csv", val_result["predictions"])
            else:
                stale += 1
                if stale >= config["early_stopping_patience"]:
                    break

    best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    eval_model.load_state_dict(best["model_state"], strict=True)
    result = engine.evaluate(eval_model, _loader(test), device, best["config"])
    predictions = result.pop("predictions")
    result.update(
        fold=fold, split="test", checkpoint_epoch=best["epoch"], slide_count=len(test),
        best_val_auc=best["best_val_auc"], decision_threshold=best["config"]["decision_threshold"],
    )
    _write_predictions(output / "test_predictions.csv", predictions)
    _write_json(output / "test_metrics.json", result)
    info.update(
        status="early_stopped" if stale >= config["early_stopping_patience"] else "completed",
        best_epoch=best["epoch"], best_val_auc=best["best_val_auc"],
        best_decision_threshold=best["config"]["decision_threshold"],
        finished_at=datetime.now().astimezone().isoformat(),
    )
    _write_json(output / "run_info.json", info)
    print(f"[fold {fold}] Test AUC={result['auc']:.6f}, best_epoch={best['epoch']}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, help="Override the official_train root in dataset_sources.json")
    parser.add_argument("--test-root", type=Path, help="Override the official_test root in dataset_sources.json")
    parser.add_argument("--fold", type=int, choices=FOLDS, help="Run one fold. The default runs all five in order.")
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="A new directory for this CAMIL run")
    mode.add_argument("--check-data", action="store_true", help="Validate all folds and both data roots")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    protocol = read_camil_protocol(args.protocol_dir)
    samples, provenance = load_camil_samples(
        protocol, train_root=args.train_root, test_root=args.test_root,
        require_hr=config["lambda_sr"] > 0,
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
    folds = FOLDS if args.fold is None else (args.fold,)
    results = [train_fold(config, protocol, samples, provenance, fold, output / f"fold_{fold}", device)
               for fold in folds]
    if folds == FOLDS:
        aucs = [float(result["auc"]) for result in results]
        summary = dict(fold_count=len(aucs), test_auc_mean=mean(aucs),
                       test_auc_std=stdev(aucs), folds=results)
        _write_json(output / "summary.json", summary)
        print(f"[five folds] Test AUC {summary['test_auc_mean']:.6f} +/- {summary['test_auc_std']:.6f}",
              flush=True)


if __name__ == "__main__":
    main()

