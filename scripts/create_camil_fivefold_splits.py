#!/usr/bin/env python3
"""Create non-destructive CAMELYON16 CAMIL-style 3:1:1 5-fold index files.

Does NOT move/copy images or modify any existing training code.
Requires numpy and scikit-learn. Run against verified data roots on the server.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path, PurePosixPath, PureWindowsPath

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

DATASET_STEM = "c16_native_l1_8192_l5_512_tissue20_jpg90_nonoverlap_v2"
DEFAULT_BASE = Path(
    "/media/ub/shidang_plu/MT/py_project/yujian/double_wsi/CAMELYON16_WSI_DATA"
)


def read_dict_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path} missing columns {sorted(required - set(reader.fieldnames or []))}; columns={reader.fieldnames}")
        rows = [{k: (v or "").strip() for k, v in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_labels(base: Path, train_root: Path) -> tuple[list[dict], dict]:
    train_path = train_root / "downstream_train/camelyon16_labels.csv"
    test_path = base / "testing/reference.csv"
    train_rows = read_dict_rows(train_path, {"case_id", "slide_id", "label"})
    test_rows: list[dict[str, str]] = []
    with test_path.open(newline="", encoding="utf-8-sig") as fh:
        for lineno, row in enumerate(csv.reader(fh), 1):
            if not row or not any(item.strip() for item in row):
                continue
            if len(row) < 2:
                raise ValueError(f"{test_path}:{lineno} missing label column")
            slide = row[0].strip()
            label = row[1].strip().lower()
            if label not in ("normal", "tumor") or not slide.startswith("test_"):
                raise ValueError(f"{test_path}:{lineno} invalid test record {row}")
            test_rows.append({"case_id": slide, "slide_id": slide, "label": str(int(label == "tumor"))})
    if not test_rows:
        raise ValueError("Empty test labels")

    records: list[dict] = []
    seen: set[str] = set()
    for source, rows in (("official_train", train_rows), ("official_test", test_rows)):
        for row in rows:
            slide = row["slide_id"]
            case = row["case_id"]
            label = row["label"]
            if not slide or not case or label not in ("0", "1") or slide in seen:
                raise ValueError(f"Invalid/duplicate label row: {source}: {row}")
            if source == "official_train" and not slide.startswith(("normal_", "tumor_")):
                raise ValueError(f"Unexpected training slide ID: {slide}")
            if source == "official_train" and (slide.startswith("tumor_") != (label == "1")):
                raise ValueError(f"Training label conflicts with slide ID: {row}")
            seen.add(slide)
            records.append({"case_id": case, "slide_id": slide, "label": int(label), "source": source})
    return records, {"train": train_path, "test": test_path}


def attach_groups(records: list[dict], patient_map_path: Path | None) -> str:
    if patient_map_path is None:
        for r in records:
            # Compatibility only. This is NOT evidence of genuine patient independence.
            r["group_id"] = r["slide_id"]
        return "slide_id_proxy_NO_PATIENT_LEVEL_GUARANTEE"

    mapping_rows = read_dict_rows(patient_map_path, {"slide_id", "patient_id"})
    mapping: dict[str, str] = {}
    for row in mapping_rows:
        slide, patient = row["slide_id"], row["patient_id"]
        if not slide or not patient or slide in mapping:
            raise ValueError(f"Invalid/duplicate patient mapping: {row}")
        mapping[slide] = patient
    required = {r["slide_id"] for r in records}
    if set(mapping) != required:
        raise ValueError(f"Patient map mismatch: missing={sorted(required - set(mapping))[:10]}, extra={sorted(set(mapping) - required)[:10]}")
    for r in records:
        r["group_id"] = mapping[r["slide_id"]]
    return "supplied_patient_id_map_verify_authenticity"


def validate_relpath(value: str, prefix: str) -> PurePosixPath:
    p = PurePosixPath(value)
    if (not value or "\\" in value or ":" in value or PureWindowsPath(value).drive
        or p.is_absolute() or ".." in p.parts or len(p.parts) < 2
        or p.parts[0] != prefix or p.suffix.lower() not in (".jpg", ".jpeg", ".png")):
        raise ValueError(f"Unsafe or unexpected image path: {value}")
    return p


def inspect_manifest(root: Path, manifest_rel: str, expected_slide_ids: set[str], check_files: bool) -> tuple[dict[str, int], dict]:
    manifest = root / manifest_rel
    required = {"slide_id", "x", "y", "hr_level", "lr_level", "scale", "hr_path", "lr_path"}
    rows = read_dict_rows(manifest, required)
    counts: Counter[str] = Counter()
    seen_coord: set[tuple] = set()
    seen_image_paths: set[tuple[str, str]] = set()
    for i, row in enumerate(rows, 2):
        slide = row["slide_id"]
        if not slide:
            raise ValueError(f"{manifest}:{i} missing slide_id")
        if tuple(row[field] for field in ("hr_level", "lr_level", "scale")) != ("1", "5", "16"):
            raise ValueError(f"{manifest}:{i} invalid geometry")
        try:
            xy = (int(row["x"]), int(row["y"]))
        except ValueError as err:
            raise ValueError(f"{manifest}:{i} invalid x/y") from err
        if min(xy) < 0 or (slide, *xy) in seen_coord:
            raise ValueError(f"{manifest}:{i} duplicate/negative coordinates")
        seen_coord.add((slide, *xy))
        lr = validate_relpath(row["lr_path"], "images_512")
        hr = validate_relpath(row["hr_path"], "images_8192")
        if lr.name != hr.name:
            raise ValueError(f"{manifest}:{i} LR/HR basename mismatch")
        key = (str(lr), str(hr))
        if key in seen_image_paths:
            raise ValueError(f"{manifest}:{i} duplicate image pair")
        seen_image_paths.add(key)
        if check_files:
            if not root.joinpath(*lr.parts).is_file() or not root.joinpath(*hr.parts).is_file():
                raise FileNotFoundError(f"{manifest}:{i} missing LR or HR pair: {lr}, {hr}")
        counts[slide] += 1
    if set(counts) != expected_slide_ids:
        raise ValueError(f"{root.name} WSI mismatch: missing={sorted(expected_slide_ids - set(counts))}, extra={sorted(set(counts) - expected_slide_ids)}")
    report = root / "verification/report.json"
    with report.open(encoding="utf-8-sig") as fh:
        metadata = json.load(fh)
    if metadata.get("status") != "complete":
        raise ValueError(f"Incomplete verification: {report}")
    info = {
        "root": str(root.resolve()),
        "manifest": str(manifest.resolve()),
        "verification": str(report.resolve()),
        "manifest_sha256": sha256(manifest),
        "verification_sha256": sha256(report),
        "wsis": len(counts),
        "regions": sum(counts.values()),
    }
    return dict(counts), info


def make_folds(records: list[dict], group_source: str, seed: int):
    ordered = sorted(records, key=lambda r: r["slide_id"])
    y = [r["label"] for r in ordered]
    groups = [r["group_id"] for r in ordered]
    if min(Counter(y).values()) < 5:
        raise ValueError("Need at least five WSI per class")
    if group_source.startswith("slide_id_proxy"):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        iterator = splitter.split(ordered, y)
    else:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        iterator = splitter.split(ordered, y, groups=groups)
    parts: list[list[dict]] = []
    for _, test_idx in iterator:
        parts.append([ordered[i] for i in test_idx])

    seen = Counter(r["slide_id"] for part in parts for r in part)
    assert seen == Counter({r["slide_id"]: 1 for r in ordered}), "Incomplete/non-disjoint outer folds"

    folds = []
    for i in range(5):
        test = parts[i]
        val = parts[(i + 1) % 5]
        train = [r for j, part in enumerate(parts) if j not in (i, (i + 1) % 5) for r in part]
        sets = {name: {r["slide_id"] for r in part} for name, part in (("train", train), ("val", val), ("test", test))}
        assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
        assert set.union(*sets.values()) == set(seen)
        by_group = {name: {r["group_id"] for r in part} for name, part in (("train", train), ("val", val), ("test", test))}
        assert not (by_group["train"] & by_group["val"] or by_group["train"] & by_group["test"] or by_group["val"] & by_group["test"])
        for name, part in (("train", train), ("val", val), ("test", test)):
            if {r["label"] for r in part} != {0, 1}:
                raise ValueError(f"Fold {i} {name} missing a class; change seed or grouping")
        folds.append({"train": sorted(sets["train"]), "val": sorted(sets["val"]), "test": sorted(sets["test"])})
    return folds


def write_csv(path: Path, columns: list[str], rows: list[dict]):
    with path.open("x", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=DEFAULT_BASE, help="CAMELYON16_WSI_DATA directory")
    p.add_argument("--output", type=Path, help="New, non-existing directory for generated CSVs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patient-map", type=Path, help="Optional CSV slide_id,patient_id; must cover ALL slides")
    p.add_argument("--skip-file-check", action="store_true", help="Skip repeated per-file existence check (previously verified)")
    args = p.parse_args()

    base = args.base.resolve(strict=True)
    train_root = base / f"{DATASET_STEM}_train"
    test_root = base / f"{DATASET_STEM}_test"
    output = (args.output or base / "E2E_MetaSR_GCSAM/data_protocols/camil_5fold").resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    # Only creates new output directories; all original dataset paths remain untouched.
    output.parent.mkdir(parents=True, exist_ok=True)

    records, label_paths = read_labels(base, train_root)
    group_source = attach_groups(records, args.patient_map)
    ids = {
        s: {r["slide_id"] for r in records if r["source"] == s}
        for s in ("official_train", "official_test")
    }
    train_counts, train_info = inspect_manifest(train_root, "Manifests/patch_manifest.csv", ids["official_train"], not args.skip_file_check)
    test_counts, test_info = inspect_manifest(test_root, "manifests/patch_manifest.csv", ids["official_test"], not args.skip_file_check)
    counts = dict(train_counts)
    counts.update(test_counts)
    for r in records:
        r["n_regions"] = counts[r["slide_id"]]
    folds = make_folds(records, group_source, args.seed)
    record_by_id = {r["slide_id"]: r for r in records}
    print(f"[dataset] {len(records)} WSI, {sum(counts.values())} regions; labels={dict(Counter(r['label'] for r in records))}")
    print(f"[groups] {group_source} (unique groups={len(set(r['group_id'] for r in records))})")

    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".camil_5fold_staging_") as tmp:
        tmp_dir = Path(tmp)
        write_csv(tmp_dir / "labels_all.csv", ["case_id", "slide_id", "label", "source", "group_id"], [
            {"case_id": r["case_id"], "slide_id": r["slide_id"], "label": r["label"], "source": r["source"], "group_id": r["group_id"]}
            for r in sorted(records, key=lambda r: r["slide_id"])
        ])
        sources = {
            "note": "Images stay in original source roots; dataset_sources.json is an index only.",
            "root_by_source": {"official_train": train_info, "official_test": test_info},
            "slides": {r["slide_id"]: {"source": r["source"], "label": r["label"], "group_id": r["group_id"], "n_regions": r["n_regions"]} for r in sorted(records, key=lambda r: r["slide_id"])},
        }
        (tmp_dir / "dataset_sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        fold_info = []
        summary_rows = []
        for i, fold in enumerate(folds):
            split_file = tmp_dir / f"splits_5fold_{i}.csv"
            packed = [dict(zip(("train", "val", "test"), values))
                      for values in zip_longest(fold["train"], fold["val"], fold["test"], fillvalue="")]
            write_csv(split_file, ["train", "val", "test"], packed)
            stat = {split: {
                "wsis": len(fold[split]),
                "normal": sum(record_by_id[x]["label"] == 0 for x in fold[split]),
                "tumor": sum(record_by_id[x]["label"] == 1 for x in fold[split]),
                "regions": sum(counts[x] for x in fold[split]),
                "original_train_wsis": sum(record_by_id[x]["source"] == "official_train" for x in fold[split]),
                "original_test_wsis": sum(record_by_id[x]["source"] == "official_test" for x in fold[split]),
            } for split in ("train", "val", "test")}
            fold_info.append({"fold": i, "file": split_file.name, "sha256": sha256(split_file), "stats": stat})
            summary_rows.append({"fold": i, **{f"{part}_{k}": v for part, values in stat.items() for k, v in values.items()}})
            print(f"[fold {i}] " + " | ".join(f"{p}: {stat[p]['wsis']} ({stat[p]['normal']}N/{stat[p]['tumor']}T)" for p in ("train", "val", "test")))
        write_csv(tmp_dir / "summary.csv", list(summary_rows[0]), summary_rows)
        protocol = {
            "protocol": "CAMIL-style pooled CAMELYON16, disjoint outer 5-fold test, cyclic validation fold",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "fold_count": 5,
            "val_assignment": "val is outer part (fold + 1) % 5; train is other three parts",
            "patient_grouping": group_source,
            "patient_independence_verified": False,
            "patient_independence_note": "Patient-map is user supplied and not independently authenticated; without map, separation is slide-level only.",
            "original_official_test_is_no_longer_independent": True,
            "source_label_files": {k: {"path": str(v.resolve()), "sha256": sha256(v)} for k, v in label_paths.items()},
            "source_wsi_counts": {k: len(v) for k, v in ids.items()},
            "total_wsis": len(records),
            "total_regions": sum(counts.values()),
            "label_counts": {str(label): sum(r["label"] == label for r in records) for label in (0, 1)},
            "group_count": len(set(r["group_id"] for r in records)),
            "source_index_sha256": sha256(tmp_dir / "dataset_sources.json"),
            "labels_all_sha256": sha256(tmp_dir / "labels_all.csv"),
            "folds": fold_info,
        }
        (tmp_dir / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_dir.rename(output)
    print(f"[done] Created {output}; NO original data or code was changed")
    if group_source.startswith("slide_id_proxy"):
        print("[IMPORTANT] NO true patient IDs supplied. Folds guarantee slide-level separation, NOT verified patient-level separation.")
    print("[IMPORTANT] Existing training loader only accepts Train/Val; a separate 3-way loader and held-out Test evaluation are needed before training.")


if __name__ == "__main__":
    main()
