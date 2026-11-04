import copy
import csv
import io
import itertools
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from torch.distributions import Normal, kl_divergence

from engine import evaluate, train_wsi, wsi_loss
from models.baseline import model_from_checkpoint
from models.robust_mil import RobustMILPooling
from models.shared_resnet_sr import CHECKPOINT_FORMAT, SharedResNetSRMIL
from models.vib import GaussianVIB
from train_e2e_shared import ModelEMA, build_optimizer, load_config, main
from wsi_data import Camelyon16WSI


ROOT = Path(__file__).resolve().parents[1]
BRANCHES = RobustMILPooling.POOLING_BRANCHES
COMBINATIONS = [choice for n in (1, 2, 3) for choice in itertools.combinations(BRANCHES, n)]


class VIBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def make_model(self, enabled=True, pooling=BRANCHES):
        torch.manual_seed(41)
        return SharedResNetSRMIL(
            shared_encoder={"pretrained": False, "trainable": "layer4"},
            classification={"embedding_dim": 32, "attention_hidden_dim": 16,
                            "pooling": pooling, "topk": 2, "region_dropout": 0,
                            "attention_dropout": 0, "projection_dropout": 0,
                            "classifier_dropout": 0,
                            "vib": {"enabled": enabled, "beta": 1e-3}},
            scale=2, sr_channels=16, sr_dropout=0,
        )

    def fixture(self, micro=False, coefficient=0.0):
        lr = torch.linspace(0, 1, 3 * 3 * 16 * 16).reshape(3, 3, 16, 16)
        sample = {"slide_id": "synthetic", "n_regions": 3, "label": 1,
                  "lr_paths": [0, 1, 2], "hr_paths": [0, 1, 2]}
        config = {"training": {"use_region_microbatch": micro, "region_microbatch_size": 2},
                  "precision": "fp32", "lambda_sr": coefficient, "sr_train_crop": 8}
        return lr, sample, config

    def test_gaussian_kl_matches_distribution_oracle_and_has_gradients(self):
        torch.manual_seed(7)
        vib = GaussianVIB(4).train()
        embeddings = torch.randn(3, 4, requires_grad=True)
        noise = torch.linspace(-1, 1, 12).reshape(3, 4)
        with patch("models.vib.torch.randn_like", return_value=noise):
            z, kl = vib(embeddings)
        mu, logvar = vib.mu(embeddings), vib.logvar(embeddings)
        q = Normal(mu, (0.5 * logvar).exp())
        prior = Normal(torch.zeros_like(mu), torch.ones_like(mu))
        torch.testing.assert_close(kl, kl_divergence(q, prior).sum(-1).mean())
        torch.testing.assert_close(z, mu + q.scale * noise)
        torch.testing.assert_close(kl, vib(embeddings.repeat(2, 1))[1])
        self.assertEqual(z.shape, (3, 4))
        self.assertEqual(kl.shape, ())
        kl.backward()
        for parameter in (embeddings, vib.mu.weight, vib.logvar.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        with torch.no_grad():
            for parameter in vib.parameters():
                parameter.zero_()
        self.assertEqual(vib(embeddings)[1].item(), 0.0)
        vib.eval()
        with patch("models.vib.torch.randn_like", side_effect=AssertionError("Evaluation sampled")):
            torch.testing.assert_close(vib(embeddings)[0], torch.zeros_like(embeddings))

    def test_switch_preserves_existing_weights_rng_sr_and_disabled_path(self):
        off = self.make_model(False).eval()
        expected_rng = torch.get_rng_state().clone()
        on = self.make_model().eval()
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        self.assertIsNone(off.classifier_head.vib)
        self.assertNotIn("vib", off.architecture())
        for name, value in off.state_dict().items():
            torch.testing.assert_close(value, on.state_dict()[name], atol=0, rtol=0)
        lr, _, _ = self.fixture()
        with torch.no_grad():
            off_sr, h = off.encode_regions(lr)
            on_sr, on_h = on.encode_regions(lr)
            torch.testing.assert_close(off_sr, on_sr, atol=0, rtol=0)
            torch.testing.assert_close(h, on_h, atol=0, rtol=0)
            expected = off.classifier_head.classifier(off.classifier_head.mil(h)[0])
            with patch("models.vib.torch.randn_like", side_effect=AssertionError("VIB off sampled")):
                torch.testing.assert_close(off.forward_embeddings(h), expected, atol=0, rtol=0)
                logits, kl = off.forward_embeddings_with_kl(h)
            torch.testing.assert_close(logits, expected, atol=0, rtol=0)
            self.assertEqual(kl.item(), 0)

    def test_engine_applies_one_full_bag_vib_for_all_pooling_sr_and_batch_modes(self):
        for pooling, coefficient, micro in itertools.product(COMBINATIONS, (0.0, 0.05), (False, True)):
            with self.subTest(pooling=pooling, lambda_sr=coefficient, micro=micro):
                model = self.make_model(pooling=pooling).train()
                lr, sample, config = self.fixture(micro, coefficient)
                with patch("engine.load_images", side_effect=lambda paths, size: lr[paths]), \
                     patch("engine.load_hr_crops", return_value=torch.zeros(1, 3, 8, 8)) as hr, \
                     patch("engine.torch.randint", return_value=torch.zeros(3, 2, dtype=torch.long)), \
                     patch.object(model.encoder, "forward", wraps=model.encoder.forward) as encoder, \
                     patch.object(model.classifier_head.vib, "forward",
                                  wraps=model.classifier_head.vib.forward) as bottleneck:
                    total, cls, sr, logits, kl = wsi_loss(
                        model, sample, torch.device("cpu"), config, return_kl=True)
                self.assertEqual(encoder.call_count, 2 if micro else 1)
                self.assertEqual(hr.call_count, 3 if coefficient else 0)
                self.assertEqual(bottleneck.call_count, 1)
                self.assertEqual(bottleneck.call_args.args[0].shape, (3, 32))
                self.assertEqual(logits.shape, (1, 2))
                self.assertGreater(kl.item(), 0)
                torch.testing.assert_close(total, cls + coefficient * sr + model.vib_beta * kl)
                total.backward()
                for parameter in (model.encoder.layer4[0].conv1.weight,
                                  model.classifier_head.projection[1].weight,
                                  model.classifier_head.vib.mu.weight,
                                  model.classifier_head.vib.logvar.weight):
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
                if not coefficient:
                    self.assertTrue(all(p.grad is None for p in model.sr.parameters()))

    def test_optimizer_ema_logs_and_deterministic_evaluation(self):
        model = self.make_model(pooling=("mean",))
        lr, sample, config = self.fixture()
        config.update(optimizer="adamw", lr=2e-4, weight_decay=0.01,
                      shared_encoder={"encoder_lr": 1e-5}, classification={"head_lr": 2e-4})
        optimizer = build_optimizer(model, config)
        head_ids = {id(p) for group in optimizer.param_groups if group["name"].startswith("head")
                    for p in group["params"]}
        self.assertTrue(all(id(p) in head_ids for p in model.classifier_head.vib.parameters()))
        ema = ModelEMA(model, 0.99)
        before = {name: p.detach().clone() for name, p in model.classifier_head.vib.named_parameters()}
        with patch("engine.load_images", side_effect=lambda paths, size: lr[paths]), \
             patch.object(model.sr, "fuse_stages", side_effect=AssertionError("SR0 fused")):
            result = train_wsi(model, sample, optimizer, torch.device("cpu"), config)
            ema.update(model)
            with patch("models.vib.torch.randn_like", side_effect=AssertionError("Evaluation sampled")):
                split = [dict(sample, label=0), sample]
                first = evaluate(model, split, torch.device("cpu"), config)
                second = evaluate(model, split, torch.device("cpu"), config)
        self.assertEqual(first, second)
        self.assertAlmostEqual(result["train_loss_vib"], model.vib_beta * result["train_loss_kl"], places=6)
        self.assertAlmostEqual(result["train_loss_total"], result["train_loss_cls"]
                               + result["train_loss_vib"], places=6)
        for name, p in model.classifier_head.vib.named_parameters():
            self.assertFalse(torch.equal(p, before[name]))
            self.assertFalse(torch.equal(dict(ema.module.classifier_head.vib.named_parameters())[name],
                                         before[name]))
        with torch.no_grad():
            logits = model.forward_embeddings(model.encode_classification_regions(lr))
            expected_ce = torch.nn.functional.cross_entropy(logits.repeat(2, 1), torch.tensor([0, 1])).item()
        self.assertAlmostEqual(first["loss"], expected_ce, places=6)

    def test_joint_loss_includes_kl_and_keeps_four_outputs(self):
        model = self.make_model().eval()
        lr, _, _ = self.fixture()
        for coefficient in (0.0, 0.05):
            with self.subTest(lambda_sr=coefficient):
                total, cls, sr, _ = model.joint_loss(
                    lr, torch.tensor([1]), lambda_sr=coefficient,
                    hr_crop=torch.zeros(3, 3, 8, 8), box=(0, 0, 8, 8))
                _, kl = model.forward_embeddings_with_kl(model.encode_classification_regions(lr))
                torch.testing.assert_close(total, cls + coefficient * sr + model.vib_beta * kl)

    def test_config_is_additive_defaults_off_and_rejects_invalid_settings(self):
        base = load_config(ROOT / "configs/native512_shared_resnet18_mean_sr0_full.yaml")
        new = load_config(ROOT / "configs/native512_shared_resnet18_mean_sr0_full_vib.yaml")
        self.assertFalse(base["classification"]["vib"]["enabled"])
        self.assertTrue(new["classification"]["vib"]["enabled"])
        del base["classification"]["vib"], new["classification"]["vib"]
        self.assertEqual(base, new)
        for invalid in (True, {"enabled": "false"}, {"beta": -1}, {"beta": float("nan")},
                        {"beta": float("inf")}, {"beta": True}):
            with self.subTest(vib=invalid), \
                 patch("train_e2e_shared.yaml.safe_load", return_value=dict(
                     base, classification=dict(base["classification"], vib=invalid))), \
                 self.assertRaisesRegex(ValueError, "vib"):
                load_config(ROOT / "configs/native512_shared_resnet18_mean_sr0_full.yaml")

    def test_training_entry_records_kl_and_restores_ema_checkpoint_strictly(self):
        config = load_config(ROOT / "configs/native512_shared_resnet18_mean_sr0_full_vib.yaml")
        config.update(epochs=1, min_epochs=1, precision="fp32", scheduler="none")
        config["shared_encoder"]["pretrained"] = False
        config["classification"].update(embedding_dim=32, attention_hidden_dim=16)
        config["metasr"]["channels"] = 16
        lr, sample, _ = self.fixture()
        samples = [dict(sample, label=label, slide_id=str(label), data_mode="native512") for label in (0, 1)]
        dataset = Camelyon16WSI(samples)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path, output = root / "config.yaml", root / "run"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            args = ["train_e2e_shared.py", "--config", str(config_path), "--data-root", str(root),
                    "--output", str(output), "--device", "cpu"]
            with patch.object(sys, "argv", args), \
                 patch("train_e2e_shared.run_shared_model_gate", return_value={}), \
                 patch("train_e2e_shared.load_fold_datasets", return_value=(dataset, dataset)), \
                 patch("train_e2e_shared.provenance", return_value={}), \
                 patch("train_e2e_shared.run_metadata", return_value={"experiment_id": "vib-test"}), \
                 patch("engine.load_images", side_effect=lambda paths, size: lr[paths]), \
                 redirect_stdout(io.StringIO()):
                main()
            with (output / "history.csv").open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertGreater(float(row["train_loss_kl"]), 0)
            self.assertAlmostEqual(float(row["train_loss_vib"]),
                                   float(row["vib_beta"]) * float(row["train_loss_kl"]), places=6)
            saved = torch.load(output / "best.pth", map_location="cpu", weights_only=False)
            last = torch.load(output / "last.pth", map_location="cpu", weights_only=False)
            self.assertEqual(saved["format"], CHECKPOINT_FORMAT)
            self.assertTrue(saved["config"]["classification"]["vib"]["enabled"])
            self.assertEqual(last["ema_updates"], 2)
            for name, value in saved["model_state"].items():
                torch.testing.assert_close(value, last["ema_model_state"][name], atol=0, rtol=0)
            restored = model_from_checkpoint(saved).eval()
            with torch.no_grad(), \
                 patch("models.vib.torch.randn_like", side_effect=AssertionError("Restore sampled")):
                h = restored.encode_classification_regions(lr)
                torch.testing.assert_close(restored.forward_embeddings(h),
                                           model_from_checkpoint(saved).eval().forward_embeddings(h), atol=0, rtol=0)
            bad = dict(saved, config=copy.deepcopy(saved["config"]))
            bad["config"]["classification"]["vib"]["enabled"] = False
            with self.assertRaisesRegex(ValueError, "VIB"):
                model_from_checkpoint(bad)
            bad = dict(saved, model_state=dict(saved["model_state"]))
            bad["model_state"].pop("classifier_head.vib.mu.weight")
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                model_from_checkpoint(bad)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 required")
    def test_cuda_bf16_forward_backward_keeps_kl_fp32(self):
        device = torch.device("cuda:0")
        for coefficient in (0.0, 0.05):
            with self.subTest(lambda_sr=coefficient):
                model = self.make_model().to(device).train()
                lr, sample, config = self.fixture(coefficient=coefficient)
                config["precision"] = "bf16"
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                with patch("engine.load_images", side_effect=lambda paths, size: lr[paths]), \
                     patch("engine.load_hr_crops", return_value=torch.zeros(1, 3, 8, 8)), \
                     patch("engine.torch.randint", return_value=torch.zeros(3, 2, dtype=torch.long)):
                    total, _, _, _, kl = wsi_loss(model, sample, device, config, return_kl=True)
                self.assertEqual(kl.dtype, torch.float32)
                self.assertTrue(torch.isfinite(total))
                total.backward()
                for parameter in model.classifier_head.vib.parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
                optimizer.step()


if __name__ == "__main__":
    unittest.main()
