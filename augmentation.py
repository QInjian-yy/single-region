import random
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageEnhance

 
@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = False
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.5
    rotation90: bool = True
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0

    def __post_init__(self):
        for name in ("enabled", "rotation90"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError("augmentation.{} must be a boolean".format(name))
        for name in (
            "horizontal_flip", "vertical_flip", "brightness", "contrast",
            "saturation", "hue",
        ):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not 0 <= value <= 1):
                raise ValueError("augmentation.{} must be in [0, 1]".format(name))


def load_augmentation_config(path=None):
    if path is None:
        return AugmentationConfig()
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("augmentation"), dict):
        raise ValueError("--augmentation-config requires a top-level 'augmentation' mapping")
    return AugmentationConfig(**config["augmentation"])


@dataclass(frozen=True)
class TransformParams:
    horizontal_flip: bool
    vertical_flip: bool
    rotation_k: int
    brightness_factor: float = 1.0
    contrast_factor: float = 1.0
    saturation_factor: float = 1.0
    hue_shift: float = 0.0


def _factor(amount):
    return random.uniform(1.0 - amount, 1.0 + amount) if amount else 1.0


def generate_transform_params(config):
    """Draw once per pair per WSI access, never while decoding or replaying."""
    if not config.enabled:
        return TransformParams(False, False, 0)
    return TransformParams(
        random.random() < config.horizontal_flip,
        random.random() < config.vertical_flip,
        random.randrange(4) if config.rotation90 else 0,
        _factor(config.brightness),
        _factor(config.contrast),
        _factor(config.saturation),
        random.uniform(-config.hue, config.hue) if config.hue else 0.0,
    )


@dataclass(frozen=True)
class AugmentedPath:
    path: Path
    params: TransformParams

    def __fspath__(self):
        return str(self.path)


def apply_to_pair(lr, hr, params):
    """Bind identical geometry to two paths; decoding stays micro-batch lazy."""
    return AugmentedPath(lr, params), AugmentedPath(hr, params)


def apply_transform(image, params):
    """Apply paired geometry and photometric jitter before tensor transfer."""
    if params.horizontal_flip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if params.vertical_flip:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if params.rotation_k:
        rotations = {1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180,
                     3: Image.Transpose.ROTATE_270}
        image = image.transpose(rotations[params.rotation_k])
    if params.brightness_factor != 1.0:
        image = ImageEnhance.Brightness(image).enhance(params.brightness_factor)
    if params.contrast_factor != 1.0:
        # Fixed per-pixel mapping keeps LR and HR crops photometrically paired;
        # Pillow's normal Contrast enhancement depends on each crop's mean.
        table = [
            max(0, min(255, round((pixel - 127.5) * params.contrast_factor + 127.5)))
            for pixel in range(256)
        ]
        image = image.point(table * 3)
    if params.saturation_factor != 1.0:
        image = ImageEnhance.Color(image).enhance(params.saturation_factor)
    if params.hue_shift:
        hue, saturation, value = image.convert("HSV").split()
        shift = int(round(params.hue_shift * 255))
        hue = hue.point(lambda pixel: (pixel + shift) % 256)
        image = Image.merge("HSV", (hue, saturation, value)).convert("RGB")
    return image
