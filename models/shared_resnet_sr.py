from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from data_modes import checkpoint_data_spec
from models.edsr_sr import ResidualBlock, UpsampleBlock
from models.robust_mil import RobustMILPooling

CHECKPOINT_FORMAT = "metasr-shared-resnet-srcls-v2"


class SharedResNet18Encoder(nn.Module):
    """ResNet18 trunk whose four stages are reused by both task heads."""

    _STAGES = (64, 128, 256, 512)

    def __init__(
        self,
        pretrained: bool = True,
        trainable: str = "layer4",
        freeze_bn: bool = True,
        weights_path: str | None = None,
    ):
        super().__init__()
        if trainable not in ("frozen", "layer3", "layer4", "all"):
            raise ValueError(
                "shared_encoder.trainable must be frozen, layer3, layer4, or all"
            )
        if not isinstance(freeze_bn, bool):
            raise ValueError("shared_encoder.freeze_bn must be a boolean")
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        if weights_path:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            net.load_state_dict(state, strict=True)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.stage_channels = self._STAGES
        self.trainable = trainable
        self.freeze_bn = freeze_bn
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=True,
        )
        self.set_trainable(trainable)

    def set_trainable(self, mode: str) -> None:
        if mode not in ("frozen", "layer3", "layer4", "all"):
            raise ValueError(
                "shared_encoder.trainable must be frozen, layer3, layer4, or all"
            )
        for parameter in self.stem.parameters():
            parameter.requires_grad_(mode == "all")
        for name in ("layer1", "layer2"):
            for parameter in getattr(self, name).parameters():
                parameter.requires_grad_(mode == "all")
        for name in ("layer3", "layer4"):
            enabled = mode == "all"
            if mode == "layer3":
                enabled = name in ("layer3", "layer4")
            elif mode == "layer4":
                enabled = name == "layer4"
            for parameter in getattr(self, name).parameters():
                parameter.requires_grad_(enabled)
        self.trainable = mode

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_bn:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, images: torch.Tensor):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected RGB region images [B,3,H,W]")
        x = self.stem((images - self.mean) / self.std)
        stage1 = self.layer1(x)
        stage2 = self.layer2(stage1)
        stage3 = self.layer3(stage2)
        stage4 = self.layer4(stage3)
        return stage1, stage2, stage3, stage4


class SharedClassifierHead(nn.Module):
    """Compact region embedding followed by robust bag-level MIL."""

    def __init__(
        self,
        input_dim: int = 512,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        topk: int = 5,
        region_dropout: float = 0.2,
        attention_dropout: float = 0.2,
        projection_dropout: float = 0.2,
        classifier_dropout: float = 0.4,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim * 2),
            nn.Linear(input_dim * 2, embedding_dim, bias=False),
            nn.GELU(),
            nn.Dropout(projection_dropout),
        )
        self.mil = RobustMILPooling(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            topk=topk,
            region_dropout=region_dropout,
            attention_dropout=attention_dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.mil.output_dim),
            nn.Dropout(classifier_dropout),
            nn.Linear(self.mil.output_dim, 2),
        )

    def forward(self, stage4: torch.Tensor):
        mean = F.adaptive_avg_pool2d(stage4, 1).flatten(1)
        variance = F.adaptive_avg_pool2d(stage4.square(), 1).flatten(1) - mean.square()
        std = variance.clamp_min(1e-6).sqrt()
        return self.projection(torch.cat((mean, std), dim=1))

    def forward_embeddings(self, embeddings: torch.Tensor):
        pooled, attention = self.mil(embeddings)
        return self.classifier(pooled), attention


class SharedFeatureSRDecoder(nn.Module):
    """Fuse all shared ResNet stages, then learn the scale factor."""

    def __init__(
        self,
        stage_channels=(64, 128, 256, 512),
        scale: int = 16,
        channels: int = 64,
        residual_scale: float = 0.1,
        rgb_range: int = 1,
        dropout: float = 0.05,
        hr_tile: int = 256,
    ):
        super().__init__()
        if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
            raise ValueError("scale must be a positive integer")
        stages = int(math.log2(scale))
        if 2 ** stages != scale:
            raise ValueError("SharedFeatureSRDecoder requires a power-of-two scale")
        if channels < 8 or hr_tile < 1:
            raise ValueError("channels must be >= 8 and hr_tile positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.scale = scale
        self.channels = channels
        self.rgb_range = rgb_range
        self.hr_tile = hr_tile
        self.stage_proj = nn.ModuleList(
            nn.Conv2d(width, channels, 1) for width in stage_channels
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(channels, residual_scale),
        )
        self.body_dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.upsample = nn.Sequential(
            *[UpsampleBlock(channels) for _ in range(stages)]
        )
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def fuse_stages(self, stages, target_size=None) -> torch.Tensor:
        if len(stages) != 4:
            raise ValueError("SR decoder expects the four shared ResNet stages")
        target = stages[0].shape[-2:]
        projected = []
        for stage, projection in zip(stages, self.stage_proj):
            value = projection(stage)
            if value.shape[-2:] != target:
                value = F.interpolate(
                    value, size=target, mode="bilinear", align_corners=False
                )
            projected.append(value)
        fused = self.fuse(torch.cat(projected, dim=1))
        if target_size is not None and fused.shape[-2:] != torch.Size(target_size):
            fused = F.interpolate(
                fused, size=target_size, mode="bilinear", align_corners=False
            )
        return self.body_dropout(fused)

    def _upsample_patch(self, feature: torch.Tensor):
        return torch.sigmoid(self.tail(self.upsample(feature))) * self.rgb_range

    def decode_crop(self, feature: torch.Tensor, box) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[1] != self.channels:
            raise ValueError("Expected fused SR features [B,channels,H,W]")
        top, left, height, width = (int(value) for value in box)
        if min(height, width) < 1 or min(top, left) < 0:
            raise ValueError("Invalid HR crop")
        lr_h, lr_w = feature.shape[-2:]
        bottom, right = top + height, left + width
        if bottom > lr_h * self.scale or right > lr_w * self.scale:
            raise ValueError("HR crop lies outside the scaled feature extent")

        halo = int(math.log2(self.scale))
        lr_top = max(0, top // self.scale - halo)
        lr_left = max(0, left // self.scale - halo)
        lr_bottom = min(lr_h, (bottom - 1) // self.scale + 1 + halo)
        lr_right = min(lr_w, (right - 1) // self.scale + 1 + halo)
        patch = self._upsample_patch(feature[:, :, lr_top:lr_bottom, lr_left:lr_right])
        y0 = top - lr_top * self.scale
        x0 = left - lr_left * self.scale
        return patch[:, :, y0:y0 + height, x0:x0 + width]

    @torch.no_grad()
    def iter_full_sr(self, feature: torch.Tensor):
        hr_h = feature.shape[-2] * self.scale
        hr_w = feature.shape[-1] * self.scale
        for top in range(0, hr_h, self.hr_tile):
            height = min(self.hr_tile, hr_h - top)
            for left in range(0, hr_w, self.hr_tile):
                width = min(self.hr_tile, hr_w - left)
                yield top, left, self.decode_crop(
                    feature, (top, left, height, width)
                )

    def forward(self, feature: torch.Tensor, box):
        return self.decode_crop(feature, box)


class SharedResNetSRMIL(nn.Module):
    """WSI MIL and SR heads conditioned on exactly one encoder instance."""

    def __init__(
        self,
        shared_encoder: dict | None = None,
        classification: dict | None = None,
        scale: int = 16,
        rgb_range: int = 1,
        sr_channels: int = 64,
        sr_residual_scale: float = 0.1,
        sr_dropout: float = 0.05,
        hr_tile: int = 256,
    ):
        super().__init__()
        encoder = dict(shared_encoder or {})
        classifier = dict(classification or {})
        if encoder.get("backbone", "resnet18") != "resnet18":
            raise ValueError("shared_encoder.backbone must be resnet18")
        self.encoder = SharedResNet18Encoder(
            pretrained=encoder.get("pretrained", True),
            trainable=encoder.get("trainable", "layer4"),
            freeze_bn=encoder.get("freeze_bn", True),
            weights_path=encoder.get("weights_path"),
        )
        self.classifier_head = SharedClassifierHead(
            input_dim=self.encoder.stage_channels[-1],
            embedding_dim=classifier.get("embedding_dim", 256),
            hidden_dim=classifier.get("attention_hidden_dim", 128),
            topk=classifier.get("topk", 5),
            region_dropout=classifier.get("region_dropout", 0.2),
            attention_dropout=classifier.get("attention_dropout", 0.2),
            projection_dropout=classifier.get("projection_dropout", 0.2),
            classifier_dropout=classifier.get("classifier_dropout", 0.4),
        )
        self.sr = SharedFeatureSRDecoder(
            stage_channels=self.encoder.stage_channels,
            scale=scale,
            channels=sr_channels,
            residual_scale=sr_residual_scale,
            rgb_range=rgb_range,
            dropout=sr_dropout,
            hr_tile=hr_tile,
        )
        self._arch = {
            "type": CHECKPOINT_FORMAT,
            "shared_encoder": "resnet18",
            "encoder_trainable": self.encoder.trainable,
            "embedding_dim": classifier.get("embedding_dim", 256),
            "attention_hidden_dim": classifier.get("attention_hidden_dim", 128),
            "mil_topk": classifier.get("topk", 5),
            "sr_channels": sr_channels,
            "scale": scale,
            "num_classes": 2,
        }

    @property
    def region_encoder(self):
        return self.classifier_head

    def encode_regions(self, lr: torch.Tensor):
        stages = self.encoder(lr)
        embeddings = self.classifier_head(stages[-1])
        sr_feature = self.sr.fuse_stages(stages, target_size=lr.shape[-2:])
        return sr_feature, embeddings

    def forward_embeddings(self, embeddings: torch.Tensor):
        logits, _ = self.classifier_head.forward_embeddings(embeddings)
        return logits

    def encode_sr_features(self, lr: torch.Tensor) -> torch.Tensor:
        return self.sr.fuse_stages(
            self.encoder(lr), target_size=lr.shape[-2:]
        )

    def forward(self, lr: torch.Tensor):
        _, embeddings = self.encode_regions(lr)
        logits = self.forward_embeddings(embeddings)
        return logits

    def architecture(self):
        return dict(self._arch)

    def set_encoder_trainable(self, mode: str):
        self.encoder.set_trainable(mode)
        self._arch["encoder_trainable"] = mode

    def joint_loss(
        self,
        lr: torch.Tensor,
        target: torch.Tensor,
        *,
        lambda_sr: float = 0.1,
        hr_crop: torch.Tensor | None = None,
        box=None,
    ):
        sr_feature, embeddings = self.encode_regions(lr)
        logits = self.forward_embeddings(embeddings)
        cls_loss = F.cross_entropy(logits, target)
        sr_loss = cls_loss.new_zeros(())
        if lambda_sr:
            if hr_crop is None or box is None:
                raise ValueError("SR supervision requires an HR crop and coordinates")
            prediction = self.sr.decode_crop(sr_feature, box)
            if prediction.shape != hr_crop.shape:
                raise ValueError("Predicted SR crop and GT crop must have identical shapes")
            sr_loss = F.l1_loss(prediction.float(), hr_crop.float())
        return cls_loss + lambda_sr * sr_loss, cls_loss, sr_loss, logits


def model_from_checkpoint(saved, device="cpu"):
    if saved.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Expected {CHECKPOINT_FORMAT}, got {saved.get('format')}")
    checkpoint_data_spec(saved)
    config = saved["config"]
    encoder = dict(config.get("shared_encoder") or {})
    # All trainable weights are in model_state, so restoration must not download.
    encoder["pretrained"] = False
    encoder.pop("weights_path", None)
    metasr = dict(config.get("metasr") or {})
    model = SharedResNetSRMIL(
        shared_encoder=encoder,
        classification=config.get("classification"),
        scale=metasr.get("scale", 16),
        rgb_range=metasr.get("rgb_range", 1),
        sr_channels=metasr.get("channels", 64),
        sr_residual_scale=metasr.get("residual_scale", 0.1),
        sr_dropout=metasr.get("dropout", 0.05),
        hr_tile=metasr.get("hr_tile", 256),
    ).to(device)
    model.load_state_dict(saved["model_state"], strict=True)
    return model
