"""The existing SSD LR layout, mapped before Dataset augmentation (no patches)."""
import hashlib
import json
from pathlib import Path

from augmentation import AugmentedPath

SSD_FOLDERS = {"official_train": "images_512_train", "official_test": "images_512_test"}


def ssd_lr_path(item, source, ssd_root):
    if source not in SSD_FOLDERS:
        raise ValueError(f"Unknown LR source: {source}")
    src = Path(item.path if isinstance(item, AugmentedPath) else item)
    if src.parts.count("images_512") != 1:
        raise ValueError(f"Expected one images_512 component in {src}")
    index = src.parts.index("images_512")
    folder = Path(ssd_root).resolve() / SSD_FOLDERS[source]
    destination = folder.joinpath(*src.parts[index + 1:])
    if not destination.resolve().is_relative_to(folder):
        raise ValueError("SSD LR path escapes the configured source directory")
    if not destination.is_file():
        raise FileNotFoundError(f"SSD image missing: {destination}")
    return (AugmentedPath(destination, item.params) if isinstance(item, AugmentedPath)
            else destination)


def remap_lr_samples(samples, ssd_root):
    """Leave HR paths unchanged: the opened original HR-cache fingerprint stays valid."""
    remapped, pairs = {}, []
    for slide_id, sample in samples.items():
        paths = [ssd_lr_path(p, sample["source"], ssd_root) for p in sample["lr_paths"]]
        remapped[slide_id] = dict(sample, lr_paths=paths)
        pairs.extend((str(a), str(b)) for a, b in zip(sample["lr_paths"], paths))
    return remapped, {
        "root": str(Path(ssd_root).resolve()), "layout": SSD_FOLDERS,
        "regions": len(pairs),
        "mapping_sha256": hashlib.sha256(json.dumps(sorted(pairs)).encode()).hexdigest(),
    }
