import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from augmentation import AugmentedPath
from data_modes import data_spec
from paired_geometry import crop_transformed
from wsi_data import load_images


def autocast(device, precision):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=precision == "bf16" and device.type == "cuda")


def load_hr_crops(paths, box, size=8192):
    """Decode GT on CPU and transfer only RGB float crops, never an 8K GPU tensor."""
    y, x, h, w = box
    crops = []
    for path in paths:
        with Image.open(os.fspath(path)) as image:
            if image.size != (size, size):
                raise ValueError(f"{path}: expected HR {(size, size)}, got {image.size}")
            if min(y, x) < 0 or min(h, w) < 1 or y+h > size or x+w > size:
                raise ValueError("Invalid HR crop")
            crop = (crop_transformed(image, box, path.params) if isinstance(path, AugmentedPath)
                    else image.crop((x, y, x+w, y+h)))
            array = np.array(crop.convert("RGB"), dtype=np.uint8)
        crops.append(torch.from_numpy(array).permute(2, 0, 1).float().div_(255))
    return torch.stack(crops)


def region_batch_size(config, n_regions):
    training = config["training"]
    if not training["use_region_microbatch"]:
        return n_regions
    return min(training["region_microbatch_size"], n_regions)


def classification_loss(logits, target, config, use_class_weights=True):
    values = config.get("class_weight_values") if use_class_weights else None
    weight = None
    if values is not None:
        weight = torch.tensor(values, dtype=logits.dtype, device=logits.device)
    return F.cross_entropy(
        logits,
        target,
        weight=weight,
        label_smoothing=float(config.get("label_smoothing", 0.0)),
    )


def wsi_loss(model, sample, device, config, *, return_kl=False):
    """Retain a normal autograd graph; no gradient cache, BN replay or RNG restore."""
    n = sample["n_regions"]
    spec = _sample_data_spec(model, sample, config)
    batch_size = region_batch_size(config, n)
    coefficient = config["lambda_sr"]
    if coefficient:
        crop = config["sr_train_crop"]
        # Sample once per region before batching, so batching does not change the crops.
        boxes = [(y, x, crop, crop)
                 for y, x in torch.randint(spec.hr_size - crop + 1, (n, 2)).tolist()]
    embeddings, sr_terms = [], []
    for start in range(0, n, batch_size):
        lr = load_images(sample["lr_paths"][start:start+batch_size], spec.lr_size).to(device)
        with autocast(device, config["precision"]):
            if not coefficient and hasattr(model, "encode_classification_regions"):
                embedding = model.encode_classification_regions(lr)
            else:
                feature, embedding = model.encode_regions(lr)
            embeddings.append(embedding.float())
            if coefficient:
                for index in range(lr.shape[0]):
                    box = boxes[start+index]
                    truth = load_hr_crops([sample["hr_paths"][start+index]], box).to(device)
                    prediction = model.sr.decode_crop(feature[index:index+1], box)
                    sr_terms.append(F.l1_loss(prediction.float(), truth) / n)
    with autocast(device, config["precision"]):
        if getattr(model, "vib_enabled", False):
            logits, kl = model.forward_embeddings_with_kl(torch.cat(embeddings))
        else:
            logits = model.forward_embeddings(torch.cat(embeddings))
            kl = logits.new_zeros((), dtype=torch.float32)
        target = torch.tensor([sample["label"]], dtype=torch.long, device=device)
        cls_loss = classification_loss(logits, target, config)
    sr_loss = torch.stack(sr_terms).sum() if sr_terms else cls_loss.new_zeros(())
    total = cls_loss + coefficient * sr_loss
    if getattr(model, "vib_enabled", False):
        total = total + model.vib_beta * kl
    result = (total, cls_loss, sr_loss, logits)
    return (*result, kl) if return_kl else result


def _sample_data_spec(model, sample, config):
    spec = data_spec(config)
    if sample.get("data_mode", "legacy256") != spec.mode:
        raise ValueError("WSI data mode differs from the configured data mode")
    if spec.mode == "native512" and model.sr.scale != spec.scale:
        raise ValueError("native512 requires a scale=16 model")
    return spec


def memory_summary(device):
    if device.type != "cuda":
        return {"max_memory_allocated": None, "max_memory_reserved": None}
    torch.cuda.synchronize(device)
    return {"max_memory_allocated": torch.cuda.max_memory_allocated(device),
            "max_memory_reserved": torch.cuda.max_memory_reserved(device)}


def gradient_norm(module):
    values = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
    return torch.stack(values).norm().item() if values else 0.0


def _gradient_l2(gradients):
    values = [gradient.detach().float().norm() for gradient in gradients if gradient is not None]
    return torch.stack(values).norm().item() if values else 0.0


def task_gradient_metrics(parameters, loss_ce, loss_sr, coefficient):
    """Inspect the existing graph without accumulating into parameter .grad."""
    metrics = {
        "grad_ce_shared": None,
        "grad_sr_shared_raw": None,
        "grad_sr_shared_weighted": None,
        "grad_ratio": None,
        "grad_cosine": None,
        "loss_ce": loss_ce.detach().item(),
        "loss_sr": loss_sr.detach().item(),
        "lambda_sr": float(coefficient),
        "sr_status": "not_applicable_lambda_sr_zero",
    }
    if not parameters:
        metrics["sr_status"] = "not_applicable_no_trainable_layer4"
        return metrics
    ce_gradients = torch.autograd.grad(
        loss_ce, parameters, retain_graph=True, allow_unused=True
    )
    ce_norm = _gradient_l2(ce_gradients)
    metrics["grad_ce_shared"] = ce_norm
    if not coefficient:
        return metrics
    sr_gradients = torch.autograd.grad(
        loss_sr, parameters, retain_graph=True, allow_unused=True
    )
    sr_norm = _gradient_l2(sr_gradients)
    weighted_norm = abs(float(coefficient)) * sr_norm
    metrics.update(
        grad_sr_shared_raw=sr_norm,
        grad_sr_shared_weighted=weighted_norm,
        grad_ratio=weighted_norm / ce_norm if ce_norm else None,
        sr_status="active",
    )
    if ce_norm and sr_norm:
        products = [(ce.detach().float() * sr.detach().float()).sum()
                    for ce, sr in zip(ce_gradients, sr_gradients)
                    if ce is not None and sr is not None]
        dot = torch.stack(products).sum().item() if products else 0.0
        metrics["grad_cosine"] = max(-1.0, min(1.0, dot / (ce_norm * sr_norm)))
    return metrics


def train_wsi(model, sample, optimizer, device, config, log_gradients=False,
              *, gradient_task_step=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total, cls, sr, logits, kl = wsi_loss(model, sample, device, config, return_kl=True)
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite joint loss")
    if gradient_task_step is not None:
        diagnostic_parameters = tuple(p for p in model.encoder.layer4.parameters() if p.requires_grad)
        task_metrics = task_gradient_metrics(diagnostic_parameters, cls, sr, config["lambda_sr"])
    total.backward()
    if gradient_task_step is not None:
        task_metrics["grad_total_shared_preclip"] = (
            gradient_norm(model.encoder.layer4) if diagnostic_parameters else None
        )
    norms = {"rdn": gradient_norm(model.sr), "region_encoder": gradient_norm(model.region_encoder),
             "abmil": (gradient_norm(model.mil_head)
                       if hasattr(model, "mil_head")
                       else gradient_norm(model.classifier_head.mil)),
             "classifier": (gradient_norm(model.classifier)
                            if hasattr(model, "classifier")
                            else gradient_norm(model.classifier_head.classifier)),
             "p2w": (gradient_norm(model.sr.P2W)
                     if hasattr(model.sr, "P2W") else 0.0)}
    if hasattr(model, "encoder"):
        norms["shared_encoder"] = gradient_norm(model.encoder)
    if not all(math.isfinite(value) for value in norms.values()):
        raise FloatingPointError(f"Non-finite gradients: {norms}")
    if log_gradients:
        print(f"[gradient] slide_id={sample['slide_id']} N={sample['n_regions']} {norms}", flush=True)
    clip = float(config.get("gradient_clip_norm", 0.0))
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    if gradient_task_step is not None:
        task_metrics["grad_total_shared_postclip"] = (
            gradient_norm(model.encoder.layer4) if diagnostic_parameters else None
        )
    optimizer.step()
    if gradient_task_step is not None:
        print(f"[gradient_tasks] step={gradient_task_step} slide_id={sample['slide_id']} "
              f"N={sample['n_regions']} {task_metrics}", flush=True)
    primary_gradient = max(
        norms.get("shared_encoder", 0.0),
        norms.get("rdn", 0.0),
    )
    result = {"train_loss_cls": cls.item(), "train_loss_sr": sr.item(),
              "train_loss_total": total.item(), "grad_norm": primary_gradient,
              **memory_summary(device)}
    if getattr(model, "vib_enabled", False):
        result.update(train_loss_kl=kl.item(), train_loss_vib=(model.vib_beta * kl).item())
    return result


@torch.no_grad()
def evaluate(model, loader, device, config):
    model.eval()
    decision_threshold = float(config.get("decision_threshold", 0.5))
    labels, scores, decisions, predictions = [], [], [], []
    loss_sum = 0.0
    for sample in loader:
        spec = _sample_data_spec(model, sample, config)
        chunks = []
        batch_size = region_batch_size(config, sample["n_regions"])
        for start in range(0, sample["n_regions"], batch_size):
            paths = sample["lr_paths"][start:start+batch_size]
            with autocast(device, config["precision"]):
                lr = load_images(paths, spec.lr_size).to(device)
                if hasattr(model, "encode_classification_regions"):
                    embedding = model.encode_classification_regions(lr)
                else:
                    _, embedding = model.encode_regions(lr)
            chunks.append(embedding.float())
        with autocast(device, config["precision"]):
            logits = model.forward_embeddings(torch.cat(chunks))
            target = torch.tensor([sample["label"]], dtype=torch.long, device=device)
            # Keep validation CE comparable to the unweighted C16 baseline even
            # when balanced CE is used for optimization.
            loss_sum += classification_loss(
                logits, target, config, use_class_weights=False
            ).item()
        score = torch.softmax(logits.float(), dim=1)[0, 1].item()
        decision = int(score >= decision_threshold)
        labels.append(sample["label"])
        scores.append(score)
        decisions.append(decision)
        predictions.append({"slide_id": sample["slide_id"], "label": sample["label"],
                            "p_tumor": score, "prediction": decision, "n_regions": sample["n_regions"]})
    if not labels:
        raise ValueError("Cannot evaluate an empty split")
    tn = sum(label == 0 and decision == 0 for label, decision in zip(labels, decisions))
    fp = sum(label == 0 and decision == 1 for label, decision in zip(labels, decisions))
    fn = sum(label == 1 and decision == 0 for label, decision in zip(labels, decisions))
    tp = sum(label == 1 and decision == 1 for label, decision in zip(labels, decisions))
    return {"loss": loss_sum / len(labels),
            "auc": roc_auc_score(labels, scores) if len(set(labels)) == 2 else float("nan"),
            "acc": accuracy_score(labels, decisions),
            "bacc": balanced_accuracy_score(labels, decisions),
            "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
            "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "predictions": predictions}
