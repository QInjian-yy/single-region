import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class ResidualBlock(nn.Module):
    def __init__(self, channels, residual_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.residual_scale = residual_scale

    def forward(self, x):
        return x + self.residual_scale * self.body(x)


class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.shuffle(self.conv(x)))


class CompactEDSR(nn.Module):

    def __init__(self, scale=16, rgb_range=1, channels=64, blocks=8,
                 residual_scale=0.1, dropout=0.0, checkpoint_blocks=True,
                 hr_tile=256):
        super().__init__()
        if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
            raise ValueError("scale must be a positive integer")
        stages = int(math.log2(scale))
        if 2 ** stages != scale:
            raise ValueError("CompactEDSR currently requires a power-of-two scale")
        if channels < 8 or blocks < 1:
            raise ValueError("channels must be >= 8 and blocks must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if hr_tile < 1:
            raise ValueError("hr_tile must be positive")
        self.scale = scale
        self.rgb_range = rgb_range
        self.channels = channels
        self.blocks = blocks
        self.checkpoint_blocks = checkpoint_blocks
        self.hr_tile = hr_tile
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[
            ResidualBlock(channels, residual_scale=residual_scale) for _ in range(blocks)
        ])
        self.body_dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.upsample = nn.Sequential(*[
            UpsampleBlock(channels) for _ in range(stages)
        ])
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def extract_features(self, lr):
        if lr.ndim != 4 or lr.shape[1] != 3:
            raise ValueError("Expected RGB LR images [B,3,H,W]")
        x = self.head(lr)
        for block in self.body:
            if self.checkpoint_blocks and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.body_dropout(x)

    def _upsample_patch(self, feature):
        output = self.upsample(feature)
        return torch.sigmoid(self.tail(output)).mul_(self.rgb_range)

    def decode_crop(self, feature, box):
        if feature.ndim != 4 or feature.shape[1] != self.channels:
            raise ValueError("Expected SR features [B,channels,H,W]")
        top, left, height, width = (int(value) for value in box)
        if min(height, width) < 1 or min(top, left) < 0:
            raise ValueError("Invalid HR crop")
        lr_h, lr_w = feature.shape[-2:]
        bottom, right = top + height, left + width
        if bottom > lr_h * self.scale or right > lr_w * self.scale:
            raise ValueError("HR crop lies outside the scaled feature extent")

        stages = int(math.log2(self.scale))
        halo = stages
        lr_top = max(0, top // self.scale - halo)
        lr_left = max(0, left // self.scale - halo)
        lr_bottom = min(lr_h, (bottom - 1) // self.scale + 1 + halo)
        lr_right = min(lr_w, (right - 1) // self.scale + 1 + halo)
        patch = self._upsample_patch(
            feature[:, :, lr_top:lr_bottom, lr_left:lr_right]
        )
        y0 = top - lr_top * self.scale
        x0 = left - lr_left * self.scale
        return patch[:, :, y0:y0 + height, x0:x0 + width]

    @torch.no_grad()
    def iter_full_sr(self, feature):
        """Yield non-overlapping HR tiles without retaining a full SR graph."""
        hr_h = feature.shape[-2] * self.scale
        hr_w = feature.shape[-1] * self.scale
        for top in range(0, hr_h, self.hr_tile):
            height = min(self.hr_tile, hr_h - top)
            for left in range(0, hr_w, self.hr_tile):
                width = min(self.hr_tile, hr_w - left)
                yield top, left, self.decode_crop(
                    feature, (top, left, height, width)
                )

    def forward(self, lr, box):
        return self.decode_crop(self.extract_features(lr), box)


def run_compact_sr_gate(scale=16, device="cpu"):
    """Cheap forward/gradient gate used before starting a real run."""
    torch.manual_seed(20261030)
    device = torch.device(device)
    model = CompactEDSR(
        scale=scale, channels=8, blocks=1, checkpoint_blocks=False, hr_tile=16
    ).to(device).eval()
    lr = torch.rand(1, 3, 2, 3, device=device)
    feature = model.extract_features(lr)
    box = (1, 2, 13, 19)
    crop = model.decode_crop(feature, box)
    full = model._upsample_patch(feature)
    expected = full[:, :, box[0]:box[0] + box[2], box[1]:box[1] + box[3]]
    torch.testing.assert_close(crop, expected, atol=2e-6, rtol=2e-5)
    crop.square().mean().backward()
    parameters = [
        parameter for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not parameters or not all(torch.isfinite(p.grad).all() for p in parameters):
        raise AssertionError("Compact EDSR gate produced missing or non-finite gradients")
    return {
        "device": str(device),
        "scale": scale,
        "crop_shape": list(crop.shape),
        "forward_max_abs_error": float((crop - expected).abs().max()),
        "passed": True,
    }
