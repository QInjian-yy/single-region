import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from infer_sr import reconstruct
from models.baseline import model_from_checkpoint
from models.shared_resnet_sr import CHECKPOINT_FORMAT, SharedResNetSRMIL
from train_e2e_shared import balanced_threshold, load_config


class SharedResNetSRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def make_model(self):
        torch.manual_seed(41)
        return SharedResNetSRMIL(
            shared_encoder={"pretrained": False, "trainable": "layer4"},
            classification={
                "embedding_dim": 32,
                "attention_hidden_dim": 16,
                "topk": 2,
            },
            scale=2,
            sr_channels=16,
        )

    def test_one_encoder_feeds_classification_and_sr(self):
        model = self.make_model().train()
        lr = torch.rand(2, 3, 16, 16)
        sr_feature, embeddings = model.encode_regions(lr)
        self.assertEqual(sr_feature.shape, (2, 16, 16, 16))
        self.assertEqual(embeddings.shape, (2, 32))
        self.assertEqual(
            sum(module.__class__.__name__ == "SharedResNet18Encoder"
                for module in model.modules()),
            1,
        )
        logits = model.forward_embeddings(embeddings[:1])
        crop = model.sr.decode_crop(sr_feature[:1], (3, 5, 12, 14))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
        loss = loss + crop.square().mean()
        loss.backward()
        for parameter in (
            model.encoder.layer4[0].conv1.weight,
            model.classifier_head.projection[1].weight,
            model.sr.fuse[0].weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_checkpoint_roundtrip_uses_baseline_factory(self):
        model = self.make_model().eval()
        config = {
            "metasr": {"scale": 2, "rgb_range": 1, "channels": 16},
            "shared_encoder": {
                "backbone": "resnet18",
                "pretrained": True,
                "trainable": "layer4",
                "freeze_bn": True,
            },
            "classification": {
                "embedding_dim": 32,
                "attention_hidden_dim": 16,
                "topk": 2,
            },
        }
        saved = {
            "format": CHECKPOINT_FORMAT,
            "architecture": model.architecture(),
            "config": copy.deepcopy(config),
            "model_state": model.state_dict(),
        }
        restored = model_from_checkpoint(saved).eval()
        lr = torch.rand(1, 3, 16, 16)
        with torch.no_grad():
            first = model.forward_embeddings(model.encode_regions(lr)[1])
            second = restored.forward_embeddings(restored.encode_regions(lr)[1])
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_encoder_freeze_modes_only_select_requested_stages(self):
        model = self.make_model()
        model.encoder.set_trainable("frozen")
        self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))
        model.encoder.set_trainable("layer4")
        self.assertTrue(
            all(p.requires_grad for p in model.encoder.layer4.parameters())
        )
        self.assertFalse(
            any(p.requires_grad for p in model.encoder.layer3.parameters())
        )
        model.encoder.set_trainable("layer3")
        self.assertTrue(
            all(p.requires_grad for p in model.encoder.layer3.parameters())
        )

    def test_streamed_full_sr_uses_shared_encoder_features(self):
        model = self.make_model().eval()
        lr = torch.rand(1, 3, 16, 16)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "sr.npy"
            with torch.no_grad():
                reconstruct(model, lr, output)
            saved = np.load(output, mmap_mode="r")
            self.assertEqual(saved.shape, (1, 3, 32, 32))
            self.assertTrue(np.isfinite(saved).all())
            del saved

    def test_threshold_prevents_all_normal_collapse_when_scores_are_separable(self):
        predictions = [
            {"label": 0, "p_tumor": 0.20},
            {"label": 0, "p_tumor": 0.25},
            {"label": 1, "p_tumor": 0.30},
            {"label": 1, "p_tumor": 0.35},
        ]
        threshold = balanced_threshold(predictions)
        decisions = [row["p_tumor"] >= threshold for row in predictions]
        self.assertEqual(decisions, [False, False, True, True])

    def test_robust_config_declares_one_shared_encoder(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs/native512_shared_resnet18_robust.yaml"
        )
        config = load_config(config_path)
        self.assertEqual(config["shared_encoder"]["backbone"], "resnet18")
        self.assertEqual(config["shared_encoder"]["trainable"], "layer4")
        self.assertNotIn("classification_encoder", config)
        self.assertEqual(config["training"]["class_weight"], "balanced")


if __name__ == "__main__":
    unittest.main()
