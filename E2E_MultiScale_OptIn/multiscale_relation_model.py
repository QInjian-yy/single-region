"""Wrap the original LR/HR model only between full-bag fusion and original MIL."""
import copy

from torch import nn

from data_modes import checkpoint_data_spec
from .multiscale_model import MultiScaleSRMIL, FORMAT as BASE_FORMAT
from .relations import build_relation, normalize_relation_config

FORMAT = "c16-e2e-lrhr-region-relation-v1"


class MultiScaleRelationMIL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.relation_settings = normalize_relation_config(config.get("region_relation"))
        if not self.relation_settings["enabled"]:
            raise ValueError("Use build_model for the parameter-free disabled path")
        self.multiscale = MultiScaleSRMIL(config)
        self.region_relation = build_relation(
            self.relation_settings, dim=config["classification"]["embedding_dim"])

    @property
    def base(self):
        return self.multiscale.base

    @property
    def fusion(self):
        return self.multiscale.fusion

    def encode_multiscale(self, lr, hr, *, need_sr=False):
        return self.multiscale.encode_multiscale(lr, hr, need_sr=need_sr)

    def classify(self, embeddings, context=None):
        embeddings = self.region_relation(embeddings, context=context)
        return self.multiscale.classify(embeddings)

    def classify_bag(self, embeddings, sample):
        # Future loaders may supply aligned coordinates/metadata. Never pass labels.
        keys = ("slide_id", "n_regions", "region_coordinates", "region_metadata")
        context = {key: sample[key] for key in keys if key in sample}
        return self.classify(embeddings, context=context)

    def architecture(self):
        return {**self.multiscale.architecture(), "type": FORMAT,
                "base_format": BASE_FORMAT,
                "region_relation": copy.deepcopy(self.relation_settings)}


def build_model(config):
    settings = normalize_relation_config(config.get("region_relation"))
    if not settings["enabled"]:
        # No wrapper, new parameters, buffers or random-number consumption.
        return MultiScaleSRMIL(config)
    return MultiScaleRelationMIL(config)


def model_from_checkpoint(saved, device="cpu"):
    """Strict restoration of relation and legacy multiscale checkpoints."""
    kind = saved.get("format")
    if kind not in (FORMAT, BASE_FORMAT):
        raise ValueError(f"Unsupported multiscale checkpoint format: {kind}")
    checkpoint_data_spec(saved)
    config = copy.deepcopy(saved["config"])
    enabled = normalize_relation_config(config.get("region_relation"))["enabled"]
    if enabled != (kind == FORMAT):
        raise ValueError("Checkpoint format disagrees with region_relation.enabled")
    # All encoder tensors are restored strictly, without downloading ImageNet weights.
    config["shared_encoder"]["pretrained"] = False
    config["shared_encoder"].pop("weights_path", None)
    model = build_model(config).to(device)
    if saved.get("architecture") != model.architecture():
        raise ValueError("Checkpoint architecture disagrees with model configuration")
    model.load_state_dict(saved["model_state"], strict=True)
    return model
