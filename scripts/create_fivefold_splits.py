"""Create CAMELYON16 five-fold splits without changing the source CSVs."""

import argparse
import csv
import re
from collections import Counter
from itertools import zip_longest
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def read_csv(path, required):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path} requires columns {required}")
        return [{key: (value or "").strip() for key, value in row.items()}
                for row in reader]


def validate_labels(rows):
    slides = set()
    for row in rows:
        slide, case, label = row["slide_id"], row["case_id"], row["label"]
        match = re.fullmatch(r"(normal|tumor)_(\d{3})", slide)
        if not case or not match or slide in slides:
            raise ValueError(f"Invalid or duplicate CAMELYON16 training slide: {slide}")
        kind, number = match.group(1), int(match.group(2))
        official = (kind == "normal" and 1 <= number <= 160 and number != 86
                    or kind == "tumor" and 1 <= number <= 111)
        if not official or label != ("0" if kind == "normal" else "1"):
            raise ValueError(f"Not an official training slide or label mismatch: {slide}")
        slides.add(slide)
    if not slides:
        raise ValueError("No labelled training slides")


def existing_participants(directory, rows):
    known = {row["case_id"] for row in rows}
    universe, validation_counts = None, Counter()
    for fold in range(3):
        path = directory / f"splits_{fold}.csv"
        old = read_csv(path, ("train", "val"))
        train = [row["train"] for row in old if row["train"]]
        val = [row["val"] for row in old if row["val"]]
        if (not train or not val or len(train) != len(set(train))
                or len(val) != len(set(val)) or set(train) & set(val)):
            raise ValueError(f"Invalid or overlapping original fold: {path}")
        cases = set(train) | set(val)
        if cases - known or (universe is not None and cases != universe):
            raise ValueError(f"Unknown cases or inconsistent original fold: {path}")
        universe = cases
        validation_counts.update(val)
    if validation_counts != Counter({case: 1 for case in universe}):
        raise ValueError("Original folds must validate each participating case once")
    return universe


def make_folds(rows):
    """Apply exact StratifiedKFold, grouping only if a case has multiple slides."""
    labels = [int(row["label"]) for row in rows]
    cases = [row["case_id"] for row in rows]
    if min(Counter(labels).get(label, 0) for label in (0, 1)) < 5:
        raise ValueError("Each class needs at least five slides")
    grouped = len(set(cases)) != len(cases)
    splitter = (StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                if grouped else StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    folds = []
    indices = splitter.split(cases, labels, groups=cases) if grouped else splitter.split(cases, labels)
    for train_idx, val_idx in indices:
        train = sorted({cases[i] for i in train_idx})
        val = sorted({cases[i] for i in val_idx})
        if set(train) & set(val) or not train or not val:
            raise ValueError("Patient/case overlap or empty fold")
        folds.append((train, val))
    counts = Counter(case for _, val in folds for case in val)
    if counts != Counter({case: 1 for case in cases}):
        raise ValueError("Each participating case must validate exactly once")
    return folds, grouped


def write_folds(output_dir, folds):
    paths = [output_dir / f"splits_5fold_{i}.csv" for i in range(5)]
    if any(path.exists() for path in paths):
        raise FileExistsError("A five-fold split already exists; no files were overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, (train, val) in zip(paths, folds):
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("train", "val"))
            writer.writerows(zip_longest(train, val, fillvalue=""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downstream-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    rows = read_csv(args.downstream_dir / "camelyon16_labels.csv", ("case_id", "slide_id", "label"))
    validate_labels(rows)
    participants = existing_participants(args.downstream_dir, rows)
    rows = [row for row in rows if row["case_id"] in participants]
    folds, grouped = make_folds(rows)
    output = args.output_dir or args.downstream_dir
    write_folds(output, folds)
    print("StratifiedGroupKFold" if grouped else "StratifiedKFold", "shuffle=True random_state=42")
    for index, (train, val) in enumerate(folds):
        parts = []
        for name, cases in (("train", set(train)), ("val", set(val))):
            counts = Counter(row["label"] for row in rows if row["case_id"] in cases)
            total = sum(counts.values())
            parts.append(f"{name}: {total} WSI, positive={counts['1']} ({counts['1']/total:.2%}), "
                         f"negative={counts['0']} ({counts['0']/total:.2%})")
        print(f"fold {index}: {'; '.join(parts)}")
    print("WSI/case disjoint; validation covers all participating cases exactly once")
    print("Patient-level verification requires genuine patient IDs; case_id may be a slide ID.")


if __name__ == "__main__":
    main()
