"""Unsupervised local HR pixel cache (not feature cache): gradients stay online.

Crop-selection uses only each region's own LR pixels, never labels/predictions.
Cache carries original HR coordinates, so SR targets can share these crops.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from PIL import Image

from augmentation import AugmentedPath, apply_transform

VERSION = "lr-tissue-diverse-grid-v1"


def _raw_path(path):
    return Path(path.path if isinstance(path, AugmentedPath) else path).resolve()


def cache_key(path):
    return hashlib.sha256(str(_raw_path(path)).encode("utf-8")).hexdigest()


def _tissue_positions(lr_image, patch_size, scale, k):
    """Pick top tissue/contrast 256px HR tiles, enforcing spatial diversity."""
    if patch_size % scale:
        raise ValueError("Patch size must be divisible by scale")
    lr_step = patch_size // scale
    lr = np.asarray(lr_image.convert("RGB"), dtype=np.float32)
    if lr.shape[:2] != (512, 512) or 512 % lr_step:
        raise ValueError("Expected 512x512 LR and a whole-number HR tiling")
    grid = 512 // lr_step
    # Tissue fraction and colour contrast are computed on the local LR image.
    # This is intentionally label-free, and has no patch-level supervision.
    avg = lr.mean(axis=2)
    tissue = (avg < 232.0) & (avg > 25.0)
    texture = lr.std(axis=2) / 255.0
    tissue_score = tissue.reshape(grid, lr_step, grid, lr_step).mean(axis=(1, 3))
    texture_score = texture.reshape(grid, lr_step, grid, lr_step).mean(axis=(1, 3))
    yy, xx = np.indices((grid, grid))
    candidates = [(float(tissue_score[i, j] + 0.15 * texture_score[i, j]), int(i), int(j))
                  for i in range(grid) for j in range(grid)]
    candidates.sort(key=lambda t: (-t[0], abs(t[1] - (grid-1)/2) + abs(t[2] - (grid-1)/2), t[1], t[2]))
    selected = []
    # Minimum spatial diversity, relaxed if a region has too little tissue.
    for min_distance in (max(2, grid // max(k, 3)), 2, 0):
        for score, i, j in candidates:
            if (i, j) in selected:
                continue
            if all((i - a)**2 + (j - b)**2 >= min_distance**2 for a, b in selected):
                selected.append((i, j))
                if len(selected) >= k:
                    break
        if len(selected) >= k:
            break
    if len(selected) != k:
        raise ValueError("Not enough HR patch positions")
    return [(i * patch_size, j * patch_size, patch_size, patch_size) for i, j in selected]


def _fingerprint(samples, patch_size, scale, k):
    keys = sorted((str(_raw_path(hr)), str(_raw_path(lr)))
                  for sample in samples.values()
                  for lr, hr in zip(sample["lr_paths"], sample["hr_paths"]))
    payload = json.dumps([VERSION, patch_size, scale, k, keys], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def build_cache(samples, cache_dir, *, patch_size=256, scale=16, k=2):
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = cache_dir / "patches"
    patch_dir.mkdir(exist_ok=True)
    index_file = cache_dir / "index.json"
    fingerprint = _fingerprint(samples, patch_size, scale, k)
    pairs = sorted(
        [(lr, hr) for sample in samples.values()
         for lr, hr in zip(sample["lr_paths"], sample["hr_paths"])],
        key=lambda item: str(item[1]),
    )
    if index_file.exists():
        index = json.loads(index_file.read_text(encoding="utf-8"))
        if index.get("fingerprint") != fingerprint:
            raise ValueError("Cache exists for different images/settings; use a NEW cache directory")
        missing = [item["filename"] for entries in index["regions"].values()
                   for item in entries if not (patch_dir / item["filename"]).is_file()]
        if not missing and len(index["regions"]) == len(pairs):
            print(f"[cache] reuse {len(pairs)} region pairs, K={k}", flush=True)
            return index
        print(f"[cache] {len(missing)} missing patches: resume preparation", flush=True)

    # Reconstruct the same deterministic crop coordinates for each image.
    # Existing completed PNG tiles are reused on interruption/restart.
    regions = {}
    for n, (lr_path, hr_path) in enumerate(pairs, start=1):
        key = cache_key(hr_path)
        if key in regions:
            raise ValueError("Duplicate HR source in dataset")
        with Image.open(lr_path) as im:
            coords = _tissue_positions(im, patch_size, scale, k)
        records = []
        for j, (y, x, height, width) in enumerate(coords):
            filename = f"{key}_{j}.png"
            records.append({"filename": filename, "box": [y, x, height, width]})
        if any(not (patch_dir / rec["filename"]).is_file() for rec in records):
            with Image.open(hr_path) as im:
                if im.size != (8192, 8192):
                    raise ValueError(f"Expected 8192x8192 HR image: {hr_path}")
                for rec in records:
                    dst = patch_dir / rec["filename"]
                    if dst.is_file():
                        continue
                    y, x, h, w = rec["box"]
                    crop = im.crop((x, y, x + w, y + h)).convert("RGB")
                    with tempfile.NamedTemporaryFile(dir=patch_dir, prefix=key, suffix=".png", delete=False) as temp:
                        name = temp.name
                    try:
                        crop.save(name, format="PNG")
                        os.replace(name, dst)
                    finally:
                        if os.path.exists(name):
                            os.unlink(name)
        regions[key] = records
        if n % 100 == 0 or n == len(pairs):
            print(f"[cache] {n}/{len(pairs)} HR regions processed", flush=True)
    index = {"version": VERSION, "fingerprint": fingerprint, "patch_size": patch_size,
             "scale": scale, "k": k, "regions": regions, "num_regions": len(pairs)}
    tmp = index_file.with_suffix(".json.part")
    tmp.write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, index_file)
    return index


def open_cache(samples, cache_dir, *, patch_size=256, scale=16, k=2):
    cache_dir = Path(cache_dir).resolve()
    index = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
    if (index.get("fingerprint") != _fingerprint(samples, patch_size, scale, k)
            or index.get("k") != k or index.get("patch_size") != patch_size
            or index.get("scale") != scale):
        raise ValueError("HR cache manifest does not match current dataset/settings")
    return PixelCache(cache_dir, index)


class PixelCache:
    def __init__(self, cache_dir, index):
        self.directory = Path(cache_dir) / "patches"
        self.regions = index["regions"]
        self.k = index["k"]
        self.size = index["patch_size"]

    def read(self, hr_paths):
        batches, boxes = [], []
        for hr_path in hr_paths:
            records = self.regions[cache_key(hr_path)]
            params = hr_path.params if isinstance(hr_path, AugmentedPath) else None
            tiles, tile_boxes = [], []
            for record in records:
                with Image.open(self.directory / record["filename"]) as im:
                    tile = im.convert("RGB")
                    if tile.size != (self.size, self.size):
                        raise ValueError("HR cache tile has wrong size")
                    if params is not None:
                        tile = apply_transform(tile, params)
                    array = np.array(tile, dtype=np.uint8, copy=True)
                tiles.append(torch.from_numpy(array).permute(2, 0, 1).float().div_(255))
                box = record["box"]
                if params is not None:
                    box = transform_box(box, params, 8192)
                tile_boxes.append(box)
            batches.append(torch.stack(tiles))
            boxes.append(tile_boxes)
        return torch.stack(batches), boxes


def transform_box(box, params, side):
    """Original HR [top,left,h,w] -> flipped/rotated HR coordinates."""
    y, x, h, w = box
    if params.horizontal_flip:
        x = side - x - w
    if params.vertical_flip:
        y = side - y - h
    for _ in range(params.rotation_k):
        y, x, h, w = side - x - w, y, w, h
    if not (0 <= y < side and 0 <= x < side and y + h <= side and x + w <= side):
        raise ValueError("Augmented HR crop out of bounds")
    return [y, x, h, w]
