"""Opt-in spatial Top-K: no changes to default MIL or existing checkpoints."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from engine import train_wsi
from data_modes import data_spec
from models.baseline import model_from_checkpoint
from models.shared_resnet_sr import CHECKPOINT_FORMAT, SharedClassifierHead, SharedResNetSRMIL
from models.shared_resnet_sr_asrd import build_model, model_from_checkpoint as load_asrd
from train_e2e_shared import load_config


class SpatialTopKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def create_model(self, k=None, *, asrd=False):
        cls = {"embedding_dim": 32, "attention_hidden_dim": 16, "pooling": ["attention"],
               "region_dropout": 0.0, "attention_dropout": 0.0,
               "projection_dropout": 0.0, "classifier_dropout": 0.0}
        if k is not None:
            cls["spatial_topk_k"] = k
        if asrd:
            cls["region_interaction"] = {"enabled": True}
        torch.manual_seed(123)
        make = build_model if asrd else SharedResNetSRMIL
        return make(shared_encoder={"pretrained": False, "trainable": "layer4"},
                    classification=cls, scale=16, sr_channels=8).eval(), cls

    def make_saved(self, model, cls):
        config = {"data_mode": "native512", "metasr": {"scale": 16, "rgb_range": 1, "channels": 8},
                  "shared_encoder": {"pretrained": False, "trainable": "layer4"},
                  "classification": copy.deepcopy(cls)}
        return {"format": model.architecture()["type"],
                "architecture": model.architecture(),
                "model_state": copy.deepcopy(model.state_dict()),
                "config": config, "data_spec": data_spec(config).metadata()}

    def test_default_absent_and_zero_are_exactly_identical(self):
        default, _ = self.create_model(None)
        explicit_zero, _ = self.create_model(0)
        self.assertEqual(default.architecture(), explicit_zero.architecture())
        self.assertNotIn("spatial_topk_k", default.architecture())
        self.assertEqual(list(default.state_dict()), list(explicit_zero.state_dict()))
        for key, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit_zero.state_dict()[key], rtol=0, atol=0)
        stage4 = torch.randn(3, 512, 4, 4)
        with torch.no_grad():
            actual = default.classifier_head(stage4)
            mean = stage4.mean((2, 3))
            std = (stage4.square().mean((2,3)) - mean.square()).clamp_min(1e-6).sqrt()
            expected = default.classifier_head.projection(torch.cat((mean, std), dim=1))
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_topk_shape_value_gradient_and_large_k(self):
        for k in (1, 4, 30):
            with self.subTest(k=k):
                head = SharedClassifierHead(input_dim=2, embedding_dim=4,
                                            hidden_dim=3, spatial_topk_k=k,
                                            pooling=["attention"], projection_dropout=0,
                                            classifier_dropout=0).eval()
                stage4 = torch.arange(1., 17.).reshape(1,1,4,4).repeat(2,2,1,1)
                stage4.requires_grad_(True)
                with patch.object(head.projection, "forward", wraps=head.projection.forward) as projection:
                    embedding = head(stage4)
                    self.assertEqual(embedding.shape, (2,4))
                    self.assertEqual(projection.call_args.args[0].shape, (2,6))
                    topk = projection.call_args.args[0][:, 4:]
                    expected = stage4.flatten(2).topk(min(k, 16),dim=2).values.mean(dim=2)
                    torch.testing.assert_close(topk, expected)
                embedding.square().sum().backward()
                self.assertTrue(torch.isfinite(stage4.grad).all())
                self.assertGreater(stage4.grad.abs().sum().item(), 0)

    def test_topk_preserves_mil_encoder_sr_initialization(self):
        baseline, _ = self.create_model(None)
        extra, _ = self.create_model(4)
        self.assertEqual(baseline.classifier_head.projection[1].weight.shape, (32, 1024))
        self.assertEqual(extra.classifier_head.projection[1].weight.shape, (32, 1536))
        for key, value in baseline.state_dict().items():
            if key.startswith(("encoder.", "sr.", "classifier_head.mil.", "classifier_head.classifier.")):
                torch.testing.assert_close(value, extra.state_dict()[key], atol=0, rtol=0)

    def test_legacy_checkpoint_and_topk_roundtrip(self):
        for k in (None, 0, 4):
            with self.subTest(k=k):
                model, cls = self.create_model(k)
                saved = self.make_saved(model, cls)
                restored = model_from_checkpoint(saved).eval()
                self.assertEqual(model.architecture(), restored.architecture())
                lr = torch.randn(2,3,32,32)
                with torch.no_grad():
                    a = model.forward_embeddings(model.encode_classification_regions(lr))
                    b = restored.forward_embeddings(restored.encode_classification_regions(lr))
                torch.testing.assert_close(a, b, atol=0, rtol=0)
                if k == 4:
                    invalid = copy.deepcopy(saved)
                    invalid["architecture"].pop("spatial_topk_k")
                    with self.assertRaisesRegex(ValueError, "spatial_topk_k"):
                        model_from_checkpoint(invalid)
                    invalid = copy.deepcopy(saved)
                    invalid["config"]["classification"]["spatial_topk_k"] = 0
                    with self.assertRaisesRegex(ValueError, "spatial_topk_k"):
                        model_from_checkpoint(invalid)

    def test_asrd_enabled_and_topk_are_compatible(self):
        model, cls = self.create_model(4, asrd=True)
        saved = self.make_saved(model, cls)
        restored = load_asrd(saved).eval()
        self.assertEqual(restored.architecture(), model.architecture())
        h = torch.randn(3, 32)
        with torch.no_grad():
            torch.testing.assert_close(model.forward_embeddings(h), restored.forward_embeddings(h), atol=0, rtol=0)

    def test_training_backward_and_sr_off_with_topk(self):
        model, _ = self.create_model(4)
        sample = {"slide_id": "fake", "data_mode": "native512", "label": 1,
                  "n_regions": 2, "lr_paths": [0,1], "hr_paths": [0,1]}
        config = {"training": {"use_region_microbatch": False,
                               "region_microbatch_size": 2},
                  "data_mode": "native512", "lambda_sr": 0.0, "precision": "fp32",
                  "gradient_clip_norm": 1.0, "label_smoothing": 0.0}
        images = torch.randn(2,3,32,32)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        with patch("engine.load_images", return_value=images), \
             patch("engine.load_hr_crops", side_effect=AssertionError("HR loaded")), \
             patch.object(model.sr, "fuse_stages", side_effect=AssertionError("SR fused")):
            result = train_wsi(model, sample, optimizer, torch.device("cpu"), config)
        self.assertGreater(result["train_loss_total"], 0)
        self.assertIsNotNone(model.classifier_head.projection[1].weight.grad)
        self.assertGreater(model.classifier_head.projection[1].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.sr.fuse[0].weight.grad)

    def test_config_optin_and_invalid_values(self):
        config = yaml.safe_load((Path(__file__).parents[1] / "configs/camil_attn_adamw_seed123.yaml").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'config.yaml'
            p.write_text(yaml.safe_dump(config))
            original = load_config(p)
            self.assertNotIn("spatial_topk_k", original["classification"])
            config["classification"]["spatial_topk_k"] = 4
            p.write_text(yaml.safe_dump(config))
            modified = load_config(p)
            self.assertEqual(modified["classification"]["spatial_topk_k"], 4)
            for bad in (-1, 1.1, True, False, '4', None):
                with self.subTest(bad=bad):
                    config["classification"]["spatial_topk_k"] = bad
                    p.write_text(yaml.safe_dump(config))
                    with self.assertRaisesRegex(ValueError, "spatial_topk_k"):
                        load_config(p)


if __name__ == "__main__":
    unittest.main()
