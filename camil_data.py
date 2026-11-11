"""Read the existing CAMIL-style Native512 protocol across the two source roots."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from native_data import NATIVE_COLUMNS, load_native_regions, native_manifest_path
from wsi_data import Camelyon16WSI, _read_csv

FOLDS = tuple(range(5))
PARTITIONS = ("train", "val", "test")
MANIFESTS = {
    "official_train": "Manifests/patch_manifest.csv",
    "official_test": "manifests/patch_manifest.csv",
}


@dataclass(frozen=True)
class CamilProtocol:
    sources: dict
    slides: dict
    folds: dict
    patient_grouping: str
    provenance: dict


def _file_record(path, expected=None):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {digest}")
    return {"path": str(path.resolve()), "sha256": digest}


def read_camil_protocol(directory):
    directory = Path(directory).resolve()
    protocol_path = directory / "protocol.json"
    index_path = directory / "dataset_sources.json"
    labels_path = directory / "labels_all.csv"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    provenance = {
        "protocol": _file_record(protocol_path),
        "source_index": _file_record(index_path, protocol["source_index_sha256"]),
        "labels": _file_record(labels_path, protocol["labels_all_sha256"]),
    }
    index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    sources = index["root_by_source"]
    if set(sources) != set(MANIFESTS):
        raise ValueError("dataset_sources.json must index official_train and official_test")

    slides = {}
    for row in _read_csv(labels_path, ("case_id", "slide_id", "label", "source", "group_id")):
        slide_id = row["slide_id"]
        if (not slide_id or slide_id in slides or not row["case_id"] or not row["group_id"]
                or row["source"] not in sources or row["label"] not in ("0", "1")):
            raise ValueError(f"Invalid or duplicate pooled label row: {row}")
        entry = index["slides"].get(slide_id)
        label = int(row["label"])
        if (entry is None or entry["source"] != row["source"]
                or type(entry["label"]) is not int or entry["label"] != label
                or entry["group_id"] != row["group_id"]
                or type(entry["n_regions"]) is not int or entry["n_regions"] < 1):
            raise ValueError(f"Dataset index disagrees with pooled labels for {slide_id}")
        slides[slide_id] = dict(row, label=label, n_regions=entry["n_regions"])
    if set(slides) != set(index["slides"]) or len(slides) != protocol["total_wsis"]:
        raise ValueError("Pooled label and dataset index WSI inventories disagree")
    if Counter(s["source"] for s in slides.values()) != protocol["source_wsi_counts"]:
        raise ValueError("Protocol source WSI counts disagree with labels")
    if sum(s["n_regions"] for s in slides.values()) != protocol["total_regions"]:
        raise ValueError("Protocol region count disagrees with dataset index")

    records = {record["fold"]: record for record in protocol["folds"]}
    if (protocol["fold_count"] != len(FOLDS) or len(protocol["folds"]) != len(FOLDS)
            or set(records) != set(FOLDS)):
        raise ValueError("Expected exactly five CAMIL fold records, numbered 0 through 4")

    folds, test_counts = {}, Counter()
    for fold in FOLDS:
        name = f"splits_5fold_{fold}.csv"
        record = records[fold]
        if record["file"] != name:
            raise ValueError(f"Fold {fold} must reference {name}")
        path = directory / name
        provenance[f"split_{fold}"] = _file_record(path, record["sha256"])
        rows = _read_csv(path, PARTITIONS)
        split = {part: [r[part] for r in rows if r[part]] for part in PARTITIONS}
        assigned = [slide_id for ids in split.values() for slide_id in ids]
        if len(assigned) != len(set(assigned)):
            raise ValueError(f"Fold {fold} has a duplicate or overlapping WSI")
        if set(assigned) != set(slides):
            missing = sorted(set(slides) - set(assigned))
            extra = sorted(set(assigned) - set(slides))
            raise ValueError(f"Fold {fold} WSI coverage differs: missing={missing}, extra={extra}")
        used_groups = set()
        for part, ids in split.items():
            groups = {slides[slide_id]["group_id"] for slide_id in ids}
            if used_groups & groups:
                raise ValueError(f"Fold {fold} has overlapping group IDs in {part}")
            used_groups.update(groups)
            if {slides[slide_id]["label"] for slide_id in ids} != {0, 1}:
                raise ValueError(f"Fold {fold} {part} requires both classes")
        test_counts.update(split["test"])
        folds[fold] = split
    if test_counts != Counter(slides.keys()):
        raise ValueError("Each pooled WSI must occur in exactly one outer Test split")
    return CamilProtocol(sources, slides, folds, protocol["patient_grouping"], provenance)


def load_camil_samples(protocol, *, train_root=None, test_root=None, require_hr=False):
    """Validate both exports once and index paths without moving or copying image data."""
    overrides = {"official_train": train_root, "official_test": test_root}
    roots = {name: Path(overrides[name] or source["root"]).resolve()
             for name, source in protocol.sources.items()}
    if roots["official_train"] == roots["official_test"]:
        raise ValueError("CAMIL requires two distinct original data roots")

    samples, provenance = {}, dict(protocol.provenance)
    for name, relative_manifest in MANIFESTS.items():
        root = roots[name]
        source = protocol.sources[name]
        manifest = native_manifest_path(root, relative_manifest)
        report = root / "verification/report.json"
        provenance[f"{name}_manifest"] = dict(
            _file_record(manifest, source["manifest_sha256"]), data_root=str(root)
        )
        provenance[f"{name}_verification"] = _file_record(report, source["verification_sha256"])
        regions, _ = load_native_regions(root, _read_csv(manifest, NATIVE_COLUMNS), require_hr=require_hr)
        expected = {slide_id: record["n_regions"] for slide_id, record in protocol.slides.items()
                    if record["source"] == name}
        if (len(expected) != source["wsis"] or sum(expected.values()) != source["regions"]
                or {slide_id: len(pairs) for slide_id, pairs in regions.items()} != expected):
            raise ValueError(f"Manifest WSI/region inventory disagrees with dataset_sources.json for {name}")
        for slide_id, pairs in regions.items():
            samples[slide_id] = dict(
                protocol.slides[slide_id],
                lr_paths=[lr for lr, _ in pairs], hr_paths=[hr for _, hr in pairs],
                data_mode="native512",
            )
    return samples, provenance


def make_camil_datasets(protocol, samples, fold, *, augmentation=None, max_train_regions=None):
    split = protocol.folds[fold]
    train = Camelyon16WSI([samples[s] for s in split["train"]], augmentation, max_train_regions)
    val = Camelyon16WSI([samples[s] for s in split["val"]])
    test = Camelyon16WSI([samples[s] for s in split["test"]])
    return train, val, test

