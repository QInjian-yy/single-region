"""Opt-in ASRD between full-bag embeddings and the existing MIL head."""

import torch

from data_modes import checkpoint_data_spec
from models.asrd_interaction import AdaptiveSparseRelationalDiffusion, validate_asrd_parameters
from models.baseline import model_from_checkpoint as load_baseline_checkpoint
from models.shared_resnet_sr import SharedResNetSRMIL


ASRD_CHECKPOINT_FORMAT = "metasr-shared-resnet-srcls-asrd-v1"


def normalize_region_interaction(settings):
    if settings is None:
        settings = {}
    defaults = {
        "enabled": False, "type": "asrd", "rank": 16, "topk": 3,
        "temperature": 1.0, "gamma_init": 0.0,
    }
    if not isinstance(settings, dict) or settings.keys() - defaults.keys():
        raise ValueError("Invalid classification.region_interaction settings")
    interaction = {**defaults, **settings}
    if type(interaction["enabled"]) is not bool or interaction["type"] != "asrd":
        raise ValueError("region_interaction requires a boolean enabled and type: asrd")
    validate_asrd_parameters(
        interaction["rank"], interaction["topk"],
        interaction["temperature"], interaction["gamma_init"],
    )
    return interaction


class SharedResNetSRMILASRD(SharedResNetSRMIL):
    def __init__(self, *, classification, interaction, **kwargs):
        super().__init__(classification=classification, **kwargs)
        # Match the baseline RNG stream used by the training DataLoader.
        with torch.random.fork_rng(devices=[]):
            self.region_interaction = AdaptiveSparseRelationalDiffusion(
                dim=self._arch["embedding_dim"], rank=interaction["rank"],
                topk=interaction["topk"], temperature=interaction["temperature"],
                gamma_init=interaction["gamma_init"],
            )
        self._arch.update(
            type=ASRD_CHECKPOINT_FORMAT, mil_pooling=list(self.classifier_head.mil.pooling),
            region_interaction=dict(interaction),
        )

    def forward_embeddings(self, embeddings):
        return super().forward_embeddings(self.region_interaction(embeddings))

    def forward_embeddings_with_kl(self, embeddings):
        return super().forward_embeddings_with_kl(self.region_interaction(embeddings))


def build_model(*, classification=None, **kwargs):
    interaction = normalize_region_interaction((classification or {}).get("region_interaction"))
    if interaction["enabled"]:
        return SharedResNetSRMILASRD(classification=classification, interaction=interaction, **kwargs)
    return SharedResNetSRMIL(classification=classification, **kwargs)


def model_from_checkpoint(saved, device="cpu"):
    classification = saved["config"].get("classification")
    interaction = normalize_region_interaction((classification or {}).get("region_interaction"))
    if saved.get("format") != ASRD_CHECKPOINT_FORMAT:
        if interaction["enabled"]:
            raise ValueError("Enabled ASRD requires the ASRD checkpoint format")
        return load_baseline_checkpoint(saved, device)
    if not interaction["enabled"]:
        raise ValueError("ASRD checkpoint requires enabled region_interaction")

    checkpoint_data_spec(saved)
    config = saved["config"]
    encoder = dict(config.get("shared_encoder") or {})
    encoder["pretrained"] = False
    encoder.pop("weights_path", None)
    metasr = config.get("metasr") or {}
    model = build_model(
        shared_encoder=encoder, classification=classification, scale=metasr.get("scale", 16),
        rgb_range=metasr.get("rgb_range", 1), sr_channels=metasr.get("channels", 64),
        sr_residual_scale=metasr.get("residual_scale", 0.1), sr_dropout=metasr.get("dropout", 0.05),
        hr_tile=metasr.get("hr_tile", 256),
    ).to(device)
    if saved.get("architecture") != model.architecture():
        raise ValueError("ASRD checkpoint architecture disagrees with its configuration")
    model.load_state_dict(saved["model_state"], strict=True)
    return model
