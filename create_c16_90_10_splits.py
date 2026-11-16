#!/usr/bin/env python3
"""Create reproducible CAMELYON16 90/10 repeated stratified holdouts.

This emits the `splits_5fold_<run>.csv` *filename convention* required by the
existing `wsi_data.py` loader. The contents are repeated 90/10 holdouts, NOT
mutually exclusive 5-fold CV. Always describe the protocol accordingly.

Example:
    python create_c16_90_10_splits.py \
      --labels-csv "$TRAIN/downstream_train/camelyon16_labels.csv" \
      --manifest "$TRAIN/Manifests/patch_manifest.csv" \
      --official-test-manifest "$TEST/manifests/patch_manifest.csv" \
      --output-dir "$TRAIN/splits_90_10_repeated_seed42" \
      --repeats 5 --seed 42
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit


def csv_rows(path: Path, columns):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not set(columns).issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} requires columns {columns}; found {reader.fieldnames}")
        return [{k: (v or "").strip() for k, v in row.items()} for row in reader]


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(labels_path, manifest_path, official_test_manifest):
    labels = csv_rows(labels_path, ("case_id", "slide_id", "label"))
    manifest = csv_rows(manifest_path, ("slide_id",))
    if not labels or not manifest:
        raise ValueError("Empty training labels or manifest")

    slides = set()
    cases = defaultdict(list)
    for row in labels:
        case, slide, label = row["case_id"], row["slide_id"], row["label"]
        if not case or not slide or label not in ("0", "1"):
            raise ValueError(f"Invalid label entry: {row}")
        if slide in slides:
            raise ValueError(f"Duplicate labelled slide_id: {slide}")
        slides.add(slide)
        cases[case].append((slide, int(label)))

    by_slide = Counter(row["slide_id"] for row in manifest)
    if "" in by_slide:
        raise ValueError("Manifest contains empty slide_id")
    missing = sorted(slides - by_slide.keys())
    unknown = sorted(by_slide.keys() - slides)
    if missing or unknown:
        raise ValueError(
            f"Training labels and manifest disagree: "
            f"labelled slides without regions={missing[:15]} (n={len(missing)}); "
            f"manifest slides without labels={unknown[:15]} (n={len(unknown)})"
        )

    # A mixed-label group cannot be stratified at the case level with a single
    # label. Refuse rather than silently split the patient's slides apart.
    for case_id, entries in cases.items():
        if len({label for _, label in entries}) != 1:
            raise ValueError(f"case_id {case_id} has mixed labels; grouping policy needs review")

    if official_test_manifest is not None:
        test_rows = csv_rows(official_test_manifest, ("slide_id",))
        test_slides = {row["slide_id"] for row in test_rows}
        overlap = slides & test_slides
        if overlap:
            raise ValueError(f"Official test slides overlap train set: {sorted(overlap)[:15]}")
        print(f"[isolation] official test slides={len(test_slides)}, slide_id overlap=0")

    case_ids = sorted(cases)
    case_labels = [cases[case][0][1] for case in case_ids]
    counts = Counter(case_labels)
    if min(counts.get(0, 0), counts.get(1, 0)) < 2:
        raise ValueError(f"Need both classes with multiple cases: {counts}")
    print(f"[data] WSI={len(slides)}, regions={len(manifest)}, "
          f"cases={len(case_ids)}, normal={sum(v==0 for v in case_labels)}, "
          f"tumor={sum(v==1 for v in case_labels)} (case-level)")
    if len(cases) == len(slides):
        print("[notice] Every case_id maps to one slide; verify these are true "
              "patient identifiers before claiming patient-level independence.")
    return cases, case_ids, case_labels, len(slides), len(manifest)


def build_splits(case_ids, case_labels, repeats, seed, val_fraction):
    outcome = []
    seen_validation = set()
    all_cases = set(case_ids)
    for i in range(repeats):
        run_seed = seed + i
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_fraction, random_state=run_seed
        )
        train_idx, val_idx = next(splitter.split(case_ids, case_labels))
        train = sorted(case_ids[j] for j in train_idx)
        val = sorted(case_ids[j] for j in val_idx)
        train_set, val_set = set(train), set(val)
        if train_set & val_set or train_set | val_set != all_cases:
            raise AssertionError(f"Invalid grouping in repeat {i}")
        train_cls = Counter(case_labels[j] for j in train_idx)
        val_cls = Counter(case_labels[j] for j in val_idx)
        if min(train_cls.get(0, 0), train_cls.get(1, 0),
               val_cls.get(0, 0), val_cls.get(1, 0)) == 0:
            raise ValueError(f"Both classes are required in both splits: run {i}")
        key = tuple(val)
        if key in seen_validation:
            raise ValueError("Duplicate validation set across repeats; choose different seeds")
        seen_validation.add(key)
        outcome.append({"run": i, "seed": run_seed, "train": train, "val": val,
                        "train_class_cases": train_cls, "val_class_cases": val_cls})
    return outcome


def save(args, cases, splits, n_slides, n_regions):
    outdir = args.output_dir
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Will not overwrite existing output: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    summary = []
    protocol = {"protocol": "repeated stratified 90/10 holdout, NOT 5-fold CV",
                "purpose": "internal train/validation model selection; official test not used",
                "validation_fraction": args.val_frac, "repeats": args.repeats,
                "base_seed": args.seed, "case_grouping": "case_id",
                "train_wsis_total": n_slides, "train_regions_total": n_regions,
                "labels_path": str(args.labels_csv.resolve()),
                "labels_sha256": sha256(args.labels_csv),
                "manifest_path": str(args.manifest.resolve()),
                "manifest_sha256": sha256(args.manifest),
                "test_manifest_path": (str(args.official_test_manifest.resolve())
                                       if args.official_test_manifest else None),
                "test_manifest_sha256": (sha256(args.official_test_manifest)
                                         if args.official_test_manifest else None),
                "created_utc": datetime.now(timezone.utc).isoformat(), "splits": []}

    for sp in splits:
        run, seed, train, val = (sp[k] for k in ("run", "seed", "train", "val"))
        path = outdir / f"splits_5fold_{run}.csv"
        with path.open("x", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["train", "val"])
            w.writerows(zip_longest(train, val, fillvalue=""))
        train_slides = [slide for c in train for slide, _ in cases[c]]
        val_slides = [slide for c in val for slide, _ in cases[c]]
        train_counts = Counter(label for c in train for _, label in cases[c])
        val_counts = Counter(label for c in val for _, label in cases[c])
        line = {"run": run, "seed": seed, "train_cases": len(train),
                "val_cases": len(val), "train_wsis": len(train_slides),
                "val_wsis": len(val_slides), "train_normal": train_counts[0],
                "train_tumor": train_counts[1], "val_normal": val_counts[0],
                "val_tumor": val_counts[1], "csv_sha256": sha256(path)}
        summary.append(line)
        protocol["splits"].append({**line, "file": path.name,
                                  "train_case_ids": train, "val_case_ids": val})
        print(f"[run {run} / seed {seed}] Train {len(train_slides)} WSI "
              f"(normal={train_counts[0]}, tumor={train_counts[1]}), "
              f"Val {len(val_slides)} WSI "
              f"(normal={val_counts[0]}, tumor={val_counts[1]})")

    with (outdir / "summary.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (outdir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (outdir / "README.md").write_text(
        "# CAMELYON16 repeated stratified 90/10 splits\n\n"
        "These are **five repeated random 90/10 train/validation splits**, "
        "**not 5-fold cross-validation**. Validation subjects may appear "
        "in more than one repeat. `splits_5fold_<run>.csv` names are "
        "used only because the existing loader recognizes this convention.\n\n"
        "Point your training CLI at this directory with `--split-dir` "
        "and `--split-scheme 5fold --fold <run>`; do not modify the original "
        "fivefold split files. Select best epochs by validation AUC only, "
        "and use the official test set solely for final evaluation.\n\n"
        "See `protocol.json` for seed, data hashes, and exact case membership.\n",
        encoding="utf-8",
    )
    print(f"[saved] {outdir.resolve()}")
    print("[protocol] REPEATED 90/10 HOLDOUT, NOT FIVEFOLD CROSS-VALIDATION")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels-csv", required=True, type=Path,
                   help="Native512 official TRAIN labels: case_id,slide_id,label")
    p.add_argument("--manifest", required=True, type=Path,
                   help="Native512 TRAIN detailed patch manifest")
    p.add_argument("--official-test-manifest", type=Path,
                   help="Optional official TEST manifest for slide-ID isolation audit")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="New directory; refuses to overwrite any existing files")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    args = p.parse_args()
    if args.repeats < 1 or args.repeats > 100:
        p.error("--repeats must be between 1 and 100")
    if not math.isfinite(args.val_frac) or not 0 < args.val_frac < 0.5:
        p.error("--val-frac must be between 0 and 0.5")
    cases, case_ids, case_labels, n_slides, n_regions = prepare(
        args.labels_csv, args.manifest, args.official_test_manifest
    )
    splits = build_splits(case_ids, case_labels, args.repeats, args.seed, args.val_frac)
    save(args, cases, splits, n_slides, n_regions)


if __name__ == "__main__":
    main()
