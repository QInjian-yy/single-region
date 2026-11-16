"""Isolated E2E multiscale LR+HR (+ optional LR->HR SR) CAMIL fivefold trainer.

Never edits existing E2E_MetaSR_GCSAM source, checkpoints, config or protocol.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import random
import sys
from statistics import mean, stdev
from datetime import datetime

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score

import camil_data
import engine
import engine_aem
import train_e2e_camil as camil
import train_e2e_camil_aem as camil_aem
from data_modes import data_spec
from models.robust_mil import RobustMILPooling
from multiscale_data import build_cache, open_cache
from multiscale_model import MultiScaleSRMIL, FORMAT


def _validate_config(config):
    if config["optimizer"] != "adamw" or config["data_mode"] != "native512":
        raise ValueError("This addon requires AdamW and native512")
    if config["classification"]["pooling"] != ["attention"]:
        raise ValueError("classification.pooling must be [attention]")
    if config["classification"].get("region_interaction", {}).get("enabled", False):
        raise ValueError("Disable ASRD in addon experiments")
    if config["classification"].get("vib", {}).get("enabled", False):
        raise ValueError("Disable VIB in addon experiments")
    ms = config.get("multiscale")
    if not isinstance(ms, dict) or set(ms) != {"hr_patch_size", "patches_per_region", "sr_regions_per_wsi"}:
        raise ValueError("Required multiscale keys: hr_patch_size, patches_per_region, sr_regions_per_wsi")
    p, k, s = ms["hr_patch_size"], ms["patches_per_region"], ms["sr_regions_per_wsi"]
    if any(type(x) is not int for x in (p, k, s)) or p < 128 or p > 1024 or p % 16 or k < 1 or k > 8 or s < 0:
        raise ValueError("Invalid multiscale dimensions / SR region count")
    if config["lambda_sr"] > 0 and s == 0:
        raise ValueError("SR-on requires sr_regions_per_wsi >= 1")
    if not (0 < config["sr_train_crop"] <= p):
        raise ValueError("SR crop must fit inside cached HR patch")
    if config["metasr"]["scale"] != 16:
        raise ValueError("native512 model must use scale16")
    if config["training"].get("sr_stop_after_epochs") is not None:
        raise ValueError("This experiment requires constant SR weight (sr_stop_after_epochs: null)")
    if config["training"].get("max_train_regions") is not None:
        raise ValueError("Training WSI subsampling disabled to preserve all-region supervision")
    if config["classification"].get("spatial_topk_k", 0) != 0:
        raise ValueError("Spatial TopK must be disabled for isolated multiscale test")
    bs = config["training"].get("region_microbatch_size", 1)
    if type(bs) is not int or bs < 1 or bs > 4:
        raise ValueError("Set training.region_microbatch_size to 1..4")
    return config


def load_config(path):
    # Existing loaders retain original class-weight, EMA, LR scheduler, and AEM semantics.
    config = camil_aem.load_config(Path(path))
    return _validate_config(config)


def build_optimizer(model, config):
    groups = []
    for label, subset, lr in (
        ("encoder", model.base.encoder, float(config["shared_encoder"]["encoder_lr"])),
        ("sr", model.base.sr, float(config["sr_lr"])),
        ("head", [model.base.classifier_head, model.fusion, model.patch_score],
         float(config["classification"]["head_lr"])),
        ("fusion_gate", [model.fusion_logit], float(config["classification"]["head_lr"])),
    ):
        # Use named parameter object identity; make sure every trainable parameter is present once.
        parameters = ([p for item in subset
                       for p in ([item] if isinstance(item, torch.nn.Parameter)
                                 else list(item.parameters()))] if isinstance(subset, list)
                      else [subset] if isinstance(subset, torch.nn.Parameter)
                      else list(subset.parameters()))
        with_decay = [p for p in parameters if p.requires_grad and p.ndim > 1]
        no_decay = [p for p in parameters if p.requires_grad and p.ndim <= 1]
        for decay, params in ((True, with_decay), (False, no_decay)):
            if params:
                groups.append({"params": params, "lr": lr,
                               "weight_decay": float(config["weight_decay"]) if decay else 0.0,
                               "name": f"{label}_{'decay' if decay else 'no_decay'}"})
    identifiers = [id(p) for g in groups for p in g["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != expected:
        raise RuntimeError("Optimizer parameter grouping misses/repeats trainable parameters")
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


def wsi_forward(model, sample, cache, device, config, *, epoch=None):
    n = sample["n_regions"]
    if n < 1:
        raise ValueError("Empty WSI bag")
    bsz = engine.region_batch_size(config, n)
    sr_coeff = float(config["lambda_sr"]) if epoch is not None else 0.0
    sr_count = min(n, config["multiscale"]["sr_regions_per_wsi"]) if sr_coeff else 0
    sr_indices = set(torch.randperm(n)[:sr_count].tolist()) if sr_count else set()
    embedding_parts = []
    sr_terms = []
    for start in range(0, n, bsz):
        end = min(start + bsz, n)
        lr = engine.load_images(sample["lr_paths"][start:end], 512).to(device)
        hr, boxes = cache.read(sample["hr_paths"][start:end])
        hr = hr.to(device)
        with engine.autocast(device, config["precision"]):
            need_sr = any(i in sr_indices for i in range(start, end))
            region_embeddings, sr_feature = model.encode_multiscale(lr, hr, need_sr=need_sr)
            embedding_parts.append(region_embeddings.float())
            if need_sr:
                crop = int(config["sr_train_crop"])
                offset = (cache.size - crop) // 2
                for local in range(end-start):
                    if start + local not in sr_indices:
                        continue
                    y, x, _, _ = boxes[local][0]
                    box = (y+offset, x+offset, crop, crop)
                    pred = model.base.sr.decode_crop(sr_feature[local:local+1], box)
                    target = hr[local:local+1, 0, :, offset:offset+crop, offset:offset+crop]
                    sr_terms.append(F.l1_loss(pred.float(), target.float()))
    with engine.autocast(device, config["precision"]):
        logits, attention = model.classify(torch.cat(embedding_parts, 0))
        label = torch.tensor([sample["label"]], dtype=torch.long, device=device)
        cls = engine.classification_loss(logits, label, config)
    sr_loss = torch.stack(sr_terms).mean() if sr_terms else cls.new_zeros(())
    negative_entropy = engine_aem.negative_attention_entropy(attention)
    aem_cfg = config["classification"]["aem"]
    aem_weight = engine_aem.aem_coefficient(aem_cfg, epoch) if epoch is not None else 0.0
    total = cls + sr_coeff * sr_loss + aem_weight * negative_entropy
    return total, cls, sr_loss, negative_entropy, logits, aem_weight


def train_wsi(model, sample, cache, optimizer, device, config, epoch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, cls, sr, neg_entropy, _, aem_weight = wsi_forward(
        model, sample, cache, device, config, epoch=epoch)
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite multiscale E2E loss")
    total.backward()
    # Must have gradients in both shared encoder and new HR fusion.
    if model.fusion[1].weight.grad is None or model.base.encoder.layer4[0].conv1.weight.grad is None:
        raise RuntimeError("Missing E2E gradient to fusion / shared encoder")
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(config["gradient_clip_norm"]), error_if_nonfinite=True)
    optimizer.step()
    return {"train_loss_cls": cls.item(), "train_loss_sr": sr.item(),
            "train_loss_total": total.item(), "train_attention_entropy": -neg_entropy.item(),
            "grad_norm": norm.item(), "aem_weight": aem_weight}


@torch.no_grad()
def evaluate(model, loader, cache, device, config):
    model.eval()
    labels, scores, predictions = [], [], []
    loss_total = 0.0
    threshold = float(config.get("decision_threshold", 0.5))
    for sample in loader:
        _, _, _, _, logits, _ = wsi_forward(model, sample, cache, device, config)
        label = torch.tensor([sample["label"]], dtype=torch.long, device=device)
        loss_total += engine.classification_loss(
            logits, label, config, use_class_weights=False).item()
        score = torch.softmax(logits.float(), dim=1)[0, 1].item()
        labels.append(int(sample["label"]))
        scores.append(score)
        predictions.append({"slide_id": sample["slide_id"], "label": int(sample["label"]),
                            "p_tumor": score, "prediction": int(score >= threshold),
                            "n_regions": sample["n_regions"]})
    decisions = [int(x >= threshold) for x in scores]
    tp = sum(a == 1 and b == 1 for a, b in zip(labels, decisions))
    tn = sum(a == 0 and b == 0 for a, b in zip(labels, decisions))
    fp = sum(a == 0 and b == 1 for a, b in zip(labels, decisions))
    fn = sum(a == 1 and b == 0 for a, b in zip(labels, decisions))
    return {"loss": loss_total / len(labels),
            "auc": float(roc_auc_score(labels, scores)),
            "acc": float(accuracy_score(labels, decisions)),
            "bacc": float(balanced_accuracy_score(labels, decisions)),
            "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "predictions": predictions}


def train_fold(config, protocol, samples, provenance, cache, fold, out, device):
    out.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(config)
    train, val, test = camil.make_camil_datasets(
        protocol, samples, fold, augmentation=config.get("augmentation"),
        max_train_regions=config["training"]["max_train_regions"])
    camil.configure_class_weights(config, train)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False

    model = MultiScaleSRMIL(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = camil.shared.build_scheduler(optimizer, config)
    ema = camil.shared.ModelEMA(model, config["ema_decay"]) if config["ema_decay"] > 0 else None
    eval_model = ema.module if ema else model
    train_loader = camil._loader(train, shuffle=True)
    val_loader = camil._loader(val)
    # No need to compute train AUC each epoch; can get train losses from steps.
    info = camil.run_metadata(fold, device)
    info.update(model=FORMAT, split_protocol="camil_5fold", training_entry="multiscale_train.py",
                optimizer=config["optimizer"], patient_grouping=protocol.patient_grouping,
                split_counts={"train": len(train), "val": len(val), "test": len(test)},
                hr_used_at_inference=True, feature_cache=False,
                hr_pixel_cache=True, hr_patch_selection="unsupervised_lr_tissue_diverse_v1")
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    camil._write_json(out / "data_provenance.json", provenance)
    camil._write_json(out / "run_info.json", info)

    best_auc, stale = -math.inf, 0
    checkpoint = out / "best.pth"
    with (out / "history.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = None
        for epoch in range(1, config["epochs"]+1):
            steps = []
            for sample in train_loader:
                steps.append(train_wsi(model, sample, cache, optimizer, device, config, epoch))
                if ema is not None:
                    ema.update(model)
            val_result = evaluate(eval_model, val_loader, cache, device, config)
            threshold = camil.shared.balanced_threshold(val_result["predictions"])
            camil.shared.apply_threshold(val_result, threshold)
            auc = float(val_result["auc"])
            if not math.isfinite(auc):
                raise ValueError(f"Nonfinite fold {fold} Val AUC")
            row = {"epoch": epoch, "lambda_sr": config["lambda_sr"],
                   "val_decision_threshold": threshold,
                   **{key: mean(step[key] for step in steps) for key in steps[0]},
                   **{f"val_{key}": val_result[key] for key in ("loss", "auc", "acc", "bacc")},
                   "val_sensitivity": val_result["sensitivity"],
                   "val_specificity": val_result["specificity"]}
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            print(f"[fold {fold}] epoch={epoch} train_loss_cls={row['train_loss_cls']:.5f} "
                  f"train_loss_sr={row['train_loss_sr']:.5f} val_auc={auc:.6f} "
                  f"aem_weight={row['aem_weight']:.6f}", flush=True)
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
                    "format": FORMAT, "architecture": model.architecture(),
                    "data_spec": data_spec(config).metadata(), "fold": fold,
                    "epoch": epoch, "config": dict(config, decision_threshold=threshold),
                    "data_provenance": provenance,
                    "model_state": eval_model.state_dict(),
                    "ema_updates": ema.updates if ema else 0,
                    "best_val_auc": auc, "best_epoch": epoch,
                }, checkpoint)
                camil._write_predictions(out / "best_val_predictions.csv", val_result["predictions"])
            else:
                stale += 1
                if stale >= config["early_stopping_patience"]:
                    break

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    eval_model.load_state_dict(saved["model_state"], strict=True)
    test_result = evaluate(eval_model, camil._loader(test), cache, device, saved["config"])
    predictions = test_result.pop("predictions")
    test_result.update(fold=fold, split="test", checkpoint_epoch=saved["epoch"],
                       slide_count=len(test), best_val_auc=saved["best_val_auc"],
                       decision_threshold=saved["config"]["decision_threshold"])
    camil._write_predictions(out / "test_predictions.csv", predictions)
    camil._write_json(out / "test_metrics.json", test_result)
    info.update(status="completed", best_epoch=saved["epoch"],
                finished_at=datetime.now().astimezone().isoformat())
    camil._write_json(out / "run_info.json", info)
    print(f"[fold {fold}] Test AUC={test_result['auc']:.6f}, best_epoch={saved['epoch']}", flush=True)
    return test_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--train-root", type=Path)
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--fold", type=int, choices=camil_data.FOLDS)
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-data", action="store_true")
    mode.add_argument("--prepare-cache", action="store_true")
    mode.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    protocol = camil_data.read_camil_protocol(args.protocol_dir)
    samples, provenance = camil_data.load_camil_samples(
        protocol, train_root=args.train_root, test_root=args.test_root, require_hr=True)
    print(f"[data] {len(samples)} WSI, {sum(s['n_regions'] for s in samples.values())} Regions; "
          f"grouping={protocol.patient_grouping}", flush=True)
    if args.check_data:
        return
    ms = config["multiscale"]
    if args.prepare_cache:
        build_cache(samples, args.cache_dir, patch_size=ms["hr_patch_size"],
                    scale=config["metasr"]["scale"], k=ms["patches_per_region"])
        return
    cache = open_cache(samples, args.cache_dir, patch_size=ms["hr_patch_size"],
                       scale=config["metasr"]["scale"], k=ms["patches_per_region"])
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 not supported on GPU")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Refuse to overwrite experiment directory: {out}")
    if any(out.is_relative_to(root) for root in (
            Path(args.protocol_dir).resolve(), *(
                Path(provenance[f"{s}_manifest"]["data_root"]) for s in protocol.sources))):
        raise ValueError("Output must be outside dataset and protocol roots")
    out.mkdir(parents=True, exist_ok=False)
    folds = camil_data.FOLDS if args.fold is None else (args.fold,)
    results = [train_fold(config, protocol, samples, provenance, cache, f, out / f"fold_{f}", device)
               for f in folds]
    if folds == camil_data.FOLDS:
        aucs = [float(r["auc"]) for r in results]
        summary = {"fold_count": 5, "test_auc_mean": mean(aucs),
                   "test_auc_std": stdev(aucs), "folds": results}
        camil._write_json(out / "summary.json", summary)
        print(f"[five folds] Test AUC {summary['test_auc_mean']:.6f} "
              f"+/- {summary['test_auc_std']:.6f}", flush=True)


if __name__ == "__main__":
    main()
