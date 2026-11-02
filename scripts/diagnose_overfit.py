from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else None


def diagnose(history_path, predictions_path=None):
    history = read_csv(history_path)
    if not history:
        raise ValueError("history.csv has no epoch rows")
    loss_rows = [
        row
        for row in history
        if number(row, "train_eval_loss") is not None
        and number(row, "val_loss") is not None
    ]
    auc_rows = [
        row
        for row in history
        if number(row, "train_auc") is not None
        and number(row, "val_auc") is not None
    ]
    result = {
        "history_epochs": len(history),
        "first_epoch": int(float(history[0]["epoch"])),
        "last_epoch": int(float(history[-1]["epoch"])),
    }
    if loss_rows:
        gaps = [
            number(row, "val_loss") - number(row, "train_eval_loss")
            for row in loss_rows
        ]
        result["loss"] = {
            "last_train_eval": number(loss_rows[-1], "train_eval_loss"),
            "last_val": number(loss_rows[-1], "val_loss"),
            "last_val_minus_train_eval": gaps[-1],
            "max_val_minus_train_eval": max(gaps),
            "max_gap_epoch": int(float(loss_rows[gaps.index(max(gaps))]["epoch"])),
        }
    if auc_rows:
        auc_gaps = [
            number(row, "train_auc") - number(row, "val_auc")
            for row in auc_rows
        ]
        largest = max(auc_gaps)
        largest_row = auc_rows[auc_gaps.index(largest)]
        result["auc"] = {
            "last_train": number(auc_rows[-1], "train_auc"),
            "last_val": number(auc_rows[-1], "val_auc"),
            "last_train_minus_val": auc_gaps[-1],
            "max_train_minus_val": largest,
            "max_gap_epoch": int(float(largest_row["epoch"])),
        }

    flags = []
    if result.get("auc", {}).get("max_train_minus_val", 0.0) >= 0.20:
        flags.append("overfitting_auc_gap")
    if result.get("loss", {}).get("max_val_minus_train_eval", 0.0) >= 0.50:
        flags.append("overfitting_loss_gap")

    if predictions_path is not None:
        predictions = read_csv(predictions_path)
        by_label = defaultdict(list)
        for row in predictions:
            by_label[int(row["label"])].append(row)
        groups = {}
        for label, rows in sorted(by_label.items()):
            scores = [float(row["p_tumor"]) for row in rows]
            decisions = Counter(int(row["prediction"]) for row in rows)
            groups[str(label)] = {
                "count": len(rows),
                "predicted_normal": decisions[0],
                "predicted_tumor": decisions[1],
                "p_tumor_mean": sum(scores) / len(scores),
                "p_tumor_min": min(scores),
                "p_tumor_max": max(scores),
            }
        result["predictions"] = {
            "count": len(predictions),
            "by_label": groups,
        }
        if float(history[-1].get("val_auc", "nan")) < 0.60:
            flags.append("weak_validation_ranking")
        overall_decisions = Counter(int(row["prediction"]) for row in predictions)
        if not overall_decisions[0] or not overall_decisions[1]:
            flags.append("class_degenerate_predictions")

    if "overfitting_auc_gap" in flags or "overfitting_loss_gap" in flags:
        assessment = "overfitting_supported"
    elif "class_degenerate_predictions" in flags or "weak_validation_ranking" in flags:
        assessment = "underfit_or_representation_collapse"
    else:
        assessment = "insufficient_evidence"
    result["flags"] = flags
    result["assessment"] = assessment
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = diagnose(args.history, args.predictions)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
