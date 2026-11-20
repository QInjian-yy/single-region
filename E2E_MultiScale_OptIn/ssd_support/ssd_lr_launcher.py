import sys
import runpy
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import engine
from augmentation import AugmentedPath

original_loader = engine.load_images
ssd_root = Path("/home/ub/wsi_fastcache")

def ssd_load_images(paths, size):
    mapped = []

    for item in paths:
        src = Path(item.path if isinstance(item, AugmentedPath) else item)
        parts = src.parts
        idx = parts.index("images_512")
        source = parts[idx - 1]

        if source.endswith("_train"):
            folder = "images_512_train"
        elif source.endswith("_test"):
            folder = "images_512_test"
        else:
            raise ValueError(f"Unknown dataset source: {source}")

        dst = ssd_root / folder / Path(*parts[idx + 1:])

        if not dst.is_file():
            raise FileNotFoundError(f"SSD image missing: {dst}")

        if isinstance(item, AugmentedPath):
            mapped.append(AugmentedPath(dst, item.params))
        else:
            mapped.append(dst)

    return original_loader(mapped, size)

engine.load_images = ssd_load_images

runpy.run_path(
    "E2E_MultiScale_OptIn/multiscale_train.py",
    run_name="__main__"
)
