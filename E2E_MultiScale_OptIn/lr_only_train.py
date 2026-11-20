"""LR-only MIL ablation: no HR image loading, HR cache, fusion or SR."""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from E2E_MultiScale_OptIn import multiscale_train as base
from E2E_MultiScale_OptIn.lr_only_model import LROnlyMIL

import camil_data
import engine
import engine_aem
import train_e2e_camil as camil


def make_model(config):
    return LROnlyMIL(config)


def make_optimizer(model, config):
    encoder_lr = float(config["shared_encoder"]["encoder_lr"])
    head_lr = float(config["classification"]["head_lr"])
    wd = float(config["weight_decay"])

    groups = []
    for name, module, lr in (
        ("encoder", model.encoder, encoder_lr),
        ("head", model.classifier_head, head_lr),
    ):
        for decay in (True, False):
            params = [
                p for p in module.parameters()
                if p.requires_grad and ((p.ndim > 1) == decay)
            ]
            if params:
                groups.append({
                    "params": params,
                    "lr": lr,
                    "weight_decay": wd if decay else 0.0,
                    "name": name + ("_decay" if decay else "_no_decay"),
                })

    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for g in groups for p in g["params"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError("Optimizer parameter mismatch")

    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


def forward_wsi(model, sample, device, config, epoch=None):
    n = int(sample["n_regions"])
    if n < 1:
        raise ValueError("Empty WSI")

    batch_size = engine.region_batch_size(config, n)
    pieces = []

    for start in range(0, n, batch_size):
        paths = sample["lr_paths"][start:start + batch_size]
        lr = engine.load_images(paths, 512).to(device)

        with engine.autocast(device, config["precision"]):
            pieces.append(model.encode_lr(lr))

    with engine.autocast(device, config["precision"]):
        embeddings = torch.cat(pieces, dim=0)
        logits, attention = model.classify(embeddings)

        label = torch.tensor(
            [int(sample["label"])], dtype=torch.long, device=device
        )
        cls = engine.classification_loss(logits, label, config)

        aem_cfg = config["classification"]["aem"]
        weight = (
            engine_aem.aem_coefficient(aem_cfg, epoch)
            if epoch is not None else 0.0
        )
        negative_entropy = engine_aem.negative_attention_entropy(attention)
        total = cls + weight * negative_entropy

    return total, cls, negative_entropy, logits, weight


def train_wsi(model, sample, cache, optimizer, device, config, epoch):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total, cls, neg_entropy, _, weight = forward_wsi(
        model, sample, device, config, epoch=epoch
    )

    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite LR-only loss")

    total.backward()

    if (
        model.encoder.layer4[0].conv1.weight.grad is None
        or not any(
            p.grad is not None
            for p in model.classifier_head.parameters()
            if p.requires_grad
        )
    ):
        raise RuntimeError("Missing LR-only E2E gradients")

    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        float(config["gradient_clip_norm"]),
        error_if_nonfinite=True,
    )
    optimizer.step()

    return {
        "train_loss_cls": cls.item(),
        "train_loss_sr": 0.0,
        "train_loss_total": total.item(),
        "train_attention_entropy": -neg_entropy.item(),
        "grad_norm": norm.item(),
        "aem_weight": weight,
    }


@torch.no_grad()
def evaluate(model, loader, cache, device, config):
    model.eval()

    labels = []
    scores = []
    predictions = []
    loss_sum = 0.0

    threshold = float(config.get("decision_threshold", 0.5))

    for sample in loader:
        _, cls, _, logits, _ = forward_wsi(
            model, sample, device, config
        )
        score = torch.softmax(logits.float(), dim=1)[0, 1].item()
        label = int(sample["label"])

        labels.append(label)
        scores.append(score)
        loss_sum += cls.item()

        predictions.append({
            "slide_id": sample["slide_id"],
            "label": label,
            "p_tumor": score,
            "prediction": int(score >= threshold),
            "n_regions": sample["n_regions"],
        })

    from sklearn.metrics import (
        roc_auc_score, accuracy_score, balanced_accuracy_score
    )

    decisions = [int(s >= threshold) for s in scores]

    tp = sum(a == 1 and b == 1 for a, b in zip(labels, decisions))
    tn = sum(a == 0 and b == 0 for a, b in zip(labels, decisions))
    fp = sum(a == 0 and b == 1 for a, b in zip(labels, decisions))
    fn = sum(a == 1 and b == 0 for a, b in zip(labels, decisions))

    return {
        "loss": loss_sum / len(labels),
        "auc": float(roc_auc_score(labels, scores)),
        "acc": float(accuracy_score(labels, decisions)),
        "bacc": float(balanced_accuracy_score(labels, decisions)),
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()

    config = base.load_config(args.config)

    if float(config["lambda_sr"]) != 0:
        raise ValueError("LR-only requires lambda_sr=0")

    protocol = camil_data.read_camil_protocol(args.protocol_dir)
    samples, provenance = camil_data.load_camil_samples(
        protocol, require_hr=False
    )

    print(
        f"[LR-only] {len(samples)} WSI, "
        f"{sum(s['n_regions'] for s in samples.values())} Regions",
        flush=True,
    )

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.set_num_threads(min(4, torch.get_num_threads()))

    if args.smoke_steps:
        import train_e2e_camil as camil_train

        fold = args.fold if args.fold is not None else 0
        train, _, _ = camil_train.make_camil_datasets(
            protocol,
            samples,
            fold,
            augmentation=config.get("augmentation"),
            max_train_regions=config["training"]["max_train_regions"],
        )
        camil_train.configure_class_weights(config, train)

        model = make_model(config).to(device)
        optimizer = make_optimizer(model, config)

        for step, sample in enumerate(camil_train._loader(train, shuffle=False)):
            if step >= args.smoke_steps:
                break
            result = train_wsi(
                model, sample, None, optimizer, device, config, epoch=1
            )
            print(
                f"[smoke] step={step + 1} "
                f"WSI={sample['slide_id']} "
                f"N={sample['n_regions']} "
                f"loss={result['train_loss_total']:.6f}",
                flush=True,
            )

        print("LR_ONLY_SMOKE_OK", flush=True)
        return

    # Reuse the original fivefold training machinery.
    # Replace only the LR-only forward/evaluation functions.
    base.train_wsi = train_wsi
    base.evaluate = evaluate

    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Output exists: {out}")
    out.mkdir(parents=True)

    folds = range(5) if args.fold is None else [args.fold]
    results = []

    for fold in folds:
        result = base.train_fold(
            config,
            protocol,
            samples,
            provenance,
            None,
            fold,
            out / f"fold_{fold}",
            device,
            model_factory=make_model,
            optimizer_factory=make_optimizer,
            training_entry="lr_only_train.py",
        )

        info_path = out / f"fold_{fold}" / "run_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info.update({
            "hr_used_at_inference": False,
            "hr_pixel_cache": False,
            "hr_patch_selection": None,
            "ablation": "lr_only",
        })
        info_path.write_text(
            json.dumps(info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        results.append(result)

    if len(results) == 5:
        from statistics import mean, stdev

        aucs = [float(x["auc"]) for x in results]
        summary = {
            "fold_count": 5,
            "test_auc_mean": mean(aucs),
            "test_auc_std": stdev(aucs),
            "folds": results,
            "ablation": "lr_only",
        }
        (out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[five folds] LR-only Test AUC "
            f"{summary['test_auc_mean']:.6f} "
            f"+/- {summary['test_auc_std']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
