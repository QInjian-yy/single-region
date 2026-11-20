"""LR-only baseline. No HR encoder, HR fusion or SR decoder."""

import torch
from torch import nn

from E2E_MultiScale_OptIn.multiscale_model import MultiScaleSRMIL


class LROnlyMIL(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Reuse the original model's encoder and MIL head.
        original = MultiScaleSRMIL(config)

        self.encoder = original.base.encoder
        self.classifier_head = original.base.classifier_head

    def encode_lr(self, lr):
        stages = self.encoder(lr)
        return self.classifier_head(stages[-1])

    def classify(self, embeddings):
        return self.classifier_head.forward_embeddings(embeddings)


    def architecture(self):
        return {
            "type": "lr_only_mil",
            "backbone": "resnet18",
            "embedding_dim": 256,
        }
