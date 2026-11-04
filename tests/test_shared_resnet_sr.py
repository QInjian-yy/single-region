import copy
import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from engine import evaluate, wsi_loss
from infer_sr import reconstruct
from models.baseline import model_from_checkpoint
from models.shared_resnet_sr import CHECKPOINT_FORMAT, SharedResNetSRMIL
from train_e2e_shared import balanced_threshold, load_config, main
from wsi_data import Camelyon16WSI


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

    def test_sr_off_skips_sr_fusion_in_training_and_evaluation(self):
        model = self.make_model().eval()
        lr = torch.rand(2, 3, 16, 16)
        sample = {"slide_id": "test", "n_regions": 2, "label": 1,
                  "lr_paths": [0, 1], "hr_paths": [0, 1]}
        config = {"training": {"use_region_microbatch": True, "region_microbatch_size": 2},
                  "lambda_sr": 0.0, "precision": "fp32"}
        with torch.no_grad():
            expected = model.forward_embeddings(model.encode_regions(lr)[1])
        with patch("engine.load_images", side_effect=lambda paths, size: lr[paths]), \
             patch("engine.load_hr_crops", side_effect=AssertionError("HR was loaded")), \
             patch.object(model.sr, "fuse_stages", side_effect=AssertionError("SR was fused")), \
             patch.object(model.sr, "decode_crop", side_effect=AssertionError("SR was decoded")):
            total, cls, sr, logits = wsi_loss(model, sample, torch.device("cpu"), config)
            result = evaluate(model, [sample, dict(sample, label=0)], torch.device("cpu"), config)
        torch.testing.assert_close(logits, expected)
        self.assertEqual(sr.item(), 0.0)
        torch.testing.assert_close(total, cls)
        self.assertEqual(len(result["predictions"]), 2)
        total.backward()
        self.assertGreater(model.encoder.layer4[0].conv1.weight.grad.norm().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.sr.parameters()))

    def test_sr_on_keeps_hr_and_sr_supervision(self):
        model = self.make_model().eval()
        lr = torch.rand(2, 3, 16, 16)
        sample = {"slide_id": "test", "n_regions": 2, "label": 1,
                  "lr_paths": [0, 1], "hr_paths": [0, 1]}
        config = {"training": {"use_region_microbatch": True, "region_microbatch_size": 2},
                  "lambda_sr": 0.05, "sr_train_crop": 8, "precision": "fp32"}
        with patch("engine.load_images", side_effect=lambda paths, size: lr[paths]), \
             patch("engine.load_hr_crops", return_value=torch.rand(1, 3, 8, 8)) as hr, \
             patch("engine.torch.randint", return_value=torch.tensor([[0, 0], [1, 1]])), \
             patch.object(model.sr, "fuse_stages", wraps=model.sr.fuse_stages) as fuse, \
             patch.object(model.sr, "decode_crop", wraps=model.sr.decode_crop) as decode:
            total, cls, sr, _ = wsi_loss(model, sample, torch.device("cpu"), config)
        self.assertEqual((hr.call_count, fuse.call_count, decode.call_count), (2, 1, 2))
        self.assertGreater(sr.item(), 0.0)
        torch.testing.assert_close(total, cls + 0.05 * sr)

    def test_sr_stop_after_epochs_rejects_invalid_values(self):
        config_path = Path(__file__).resolve().parents[1] / "configs/native512_shared_resnet18_robust.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.yaml"
            for value in (0, -1, True, 1.5, "2"):
                with self.subTest(value=value):
                    config["training"]["sr_stop_after_epochs"] = value
                    path.write_text(yaml.safe_dump(config), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "sr_stop_after_epochs"):
                        load_config(path)
            config["training"].pop("sr_stop_after_epochs")
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            self.assertNotIn("sr_stop_after_epochs", load_config(path)["training"])

    def test_schedule_switches_after_configured_epoch_and_records_weight(self):
        config_path = Path(__file__).resolve().parents[1] / "configs/native512_shared_resnet18_robust.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config.update(epochs=3, min_epochs=1, early_stopping_patience=3, scheduler="none")
        config["shared_encoder"]["pretrained"] = False
        samples = [{"slide_id": str(label), "label": label, "n_regions": 1,
                    "lr_paths": [0], "hr_paths": [0]} for label in (0, 1)]
        train_data, val_data = Camelyon16WSI(samples), Camelyon16WSI(samples)
        weights, checkpoints, saved_schemes = [], [], []

        def train(model, sample, optimizer, device, current, log_gradients):
            weights.append(current["lambda_sr"])
            return {"train_loss_cls": 1.0, "train_loss_sr": 1.0,
                    "train_loss_total": 1.0, "grad_norm": 1.0,
                    "max_memory_allocated": None, "max_memory_reserved": None}

        def validate(*args):
            return {"loss": 1.0, "auc": 0.5, "acc": 0.5, "bacc": 0.5,
                    "predictions": [{"slide_id": "0", "label": 0, "p_tumor": 0.2,
                                     "prediction": 0, "n_regions": 1},
                                    {"slide_id": "1", "label": 1, "p_tumor": 0.8,
                                     "prediction": 1, "n_regions": 1}]}

        def capture(saved, path):
            checkpoints.append((path.name, saved["epoch"], saved["epoch_lambda_sr"],
                                saved["config"]["lambda_sr"]))
            saved_schemes.append((saved["fold"], saved["config"].get("split_scheme")))

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for sr_epochs, expected in ((2, [0.05, 0.05, 0.0]),
                                        (None, [0.05, 0.05, 0.05])):
                with self.subTest(sr_epochs=sr_epochs):
                    config["training"]["sr_stop_after_epochs"] = sr_epochs
                    path = root / "config.yaml"
                    path.write_text(yaml.safe_dump(config), encoding="utf-8")
                    output = root / f"run_{sr_epochs}"
                    args = ["train_e2e_shared.py", "--config", str(path), "--data-root", str(root),
                            "--output", str(output), "--device", "cpu",
                            "--fold", "4", "--split-scheme", "5fold"]
                    weights.clear()
                    checkpoints.clear()
                    saved_schemes.clear()
                    with patch.object(sys, "argv", args), \
                         patch("train_e2e_shared.run_shared_model_gate", return_value={}), \
                         patch("train_e2e_shared.load_fold_datasets", return_value=(train_data, val_data)) as loader, \
                         patch("train_e2e_shared.train_wsi", side_effect=train), \
                         patch("train_e2e_shared.evaluate", side_effect=validate), \
                         patch("train_e2e_shared.provenance", return_value={}), \
                         patch("train_e2e_shared.run_metadata", return_value={"experiment_id": "test"}), \
                         patch("train_e2e_shared.torch.save", side_effect=capture), \
                         redirect_stdout(io.StringIO()):
                        main()
                    with (output / "history.csv").open(newline="", encoding="utf-8") as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertEqual(weights, [value for value in expected for _ in range(2)])
                    self.assertEqual([float(row["lambda_sr"]) for row in rows], expected)
                    self.assertIn(("last.pth", 3, expected[-1], 0.05), checkpoints)
                    self.assertEqual(loader.call_args.kwargs["split_scheme"], "5fold")
                    self.assertTrue(saved_schemes)
                    self.assertTrue(all(item == (4, "5fold") for item in saved_schemes))


if __name__ == "__main__":
    unittest.main()
