import copy
from contextlib import redirect_stdout
import csv
import io
import json
import math
from pathlib import Path
from statistics import mean, stdev
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

import engine
import engine_aem as aem
from models.shared_resnet_sr_asrd import build_model, model_from_checkpoint
from tests import test_camil as camil_tests
import train_e2e_camil_aem as trainer


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "configs/camil_attn_adamw_asrd_aem_5fold_seed1.yaml"


def small_config():
    config = trainer.load_config(RECIPE)
    config["shared_encoder"]["pretrained"] = False
    config["classification"].update(embedding_dim=8, attention_hidden_dim=8)
    config["classification"]["region_interaction"]["rank"] = 4
    config["metasr"].update(channels=8, hr_tile=16)
    config.update(precision="fp32", epochs=2, min_epochs=1, scheduler="none")
    config["augmentation"] = {"enabled": False}
    return config


def small_model(config):
    return build_model(shared_encoder=config["shared_encoder"], classification=config["classification"],
                       sr_channels=8, sr_dropout=0.0, hr_tile=16)


class AEMTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(71)

    def test_negative_entropy_sign_gradient_singleton_and_extreme_probabilities(self):
        for n in (1, 2, 5, 16, 37):
            uniform = torch.full((n,), 1.0 / n)
            self.assertAlmostEqual(aem.negative_attention_entropy(uniform).item(), -math.log(n), places=5)
        scores = torch.tensor([2.0, 0.5, -1.0], requires_grad=True)
        probabilities = scores.softmax(0)
        loss = aem.negative_attention_entropy(probabilities)
        expected = probabilities.detach() * (probabilities.detach().log() - loss.detach())
        loss.backward()
        torch.testing.assert_close(scores.grad, expected)
        self.assertGreater(loss.item(), -math.log(3))
        self.assertGreater(scores.grad[0].item(), 0.0)  # Descent reduces the dominant logit.
        for dtype in (torch.float32, torch.bfloat16):
            probabilities = torch.tensor([0.0, 1.0], dtype=dtype, requires_grad=True)
            loss = aem.negative_attention_entropy(probabilities)
            self.assertEqual(loss.dtype, torch.float32)
            loss.backward()
            self.assertTrue(torch.isfinite(probabilities.grad).all())
        with self.assertRaises(ValueError):
            aem.negative_attention_entropy(torch.ones(2, 2))

    def test_validation_and_cosine_schedule(self):
        settings = aem.normalize_aem({"enabled": True})
        weights = [aem.aem_coefficient(settings, epoch) for epoch in range(1, 61)]
        self.assertEqual(weights[0], 0.01)
        self.assertEqual(weights[49:], [0.0] * 11)
        self.assertTrue(all(a >= b for a, b in zip(weights, weights[1:])))
        self.assertEqual(aem.aem_coefficient(aem.normalize_aem(None), 1), 0.0)
        invalid = [[], {"other": 1}, {"enabled": 1}, {"weight": -0.01}, {"weight": True},
                   {"weight": "0.01"}, {"weight": float("nan")}, {"weight": float("inf")},
                   {"weight": 1.0e300},
                   {"anneal_epochs": 1}, {"anneal_epochs": 2.5}, {"anneal_epochs": True}]
        for value in invalid:
            with self.subTest(settings=value), self.assertRaises(ValueError):
                aem.normalize_aem(value)
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                aem.aem_coefficient(settings, value)

    def test_configs_only_change_the_requested_regularizer_and_asrd_switch(self):
        baseline = yaml.safe_load((ROOT / "configs/camil_attn_adamw_asrd_5fold_seed1.yaml").read_text())
        recipe = yaml.safe_load(RECIPE.read_text())
        self.assertEqual(recipe["classification"].pop("aem"),
                         {"enabled": True, "weight": 0.01, "anneal_epochs": 50})
        self.assertEqual(recipe, baseline)
        control = yaml.safe_load((ROOT / "configs/camil_attn_adamw_aem_5fold_seed1.yaml").read_text())
        self.assertFalse(control["classification"]["region_interaction"]["enabled"])
        control["classification"]["region_interaction"]["enabled"] = True
        self.assertEqual(control, yaml.safe_load(RECIPE.read_text()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for changes in ({"optimizer": "gcsam"}, {"classification": {**baseline["classification"],
                                                       "pooling": ["mean"], "aem": {"enabled": True}}}):
                invalid = {**yaml.safe_load(RECIPE.read_text()), **changes}
                path.write_text(yaml.safe_dump(invalid))
                with self.assertRaisesRegex(ValueError, "AEM"):
                    trainer.load_config(path)

    def test_zero_weight_and_disabled_match_original_update_exactly(self):
        images = torch.rand(5, 3, 32, 32)
        sample = dict(slide_id="small", n_regions=5, label=1, lr_paths=list(range(5)), data_mode="native512")
        for enabled, weight in ((True, 0.0), (False, 0.01)):
            config = small_config()
            config["classification"]["aem"].update(enabled=enabled, weight=weight)
            original = small_model(config)
            regularized = copy.deepcopy(original)
            optimizer = trainer.camil.shared.build_optimizer(original, config)
            aem_optimizer = trainer.camil.shared.build_optimizer(regularized, config)
            rng = torch.get_rng_state()
            with patch.object(engine, "load_images", side_effect=lambda paths, size: images[paths]):
                expected = engine.train_wsi(original, sample, optimizer, torch.device("cpu"), config)
                torch.set_rng_state(rng)
                actual = aem.train_wsi(regularized, sample, aem_optimizer, torch.device("cpu"), config, epoch=1)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            for name, value in original.state_dict().items():
                torch.testing.assert_close(regularized.state_dict()[name], value, rtol=0, atol=0)
            for first, second in zip(original.parameters(), regularized.parameters()):
                if first.grad is not None:
                    torch.testing.assert_close(second.grad, first.grad, rtol=0, atol=0)
            for key in ("train_loss_cls", "train_loss_sr", "train_loss_total", "grad_norm"):
                self.assertEqual(actual[key], expected[key])
            self.assertEqual(len(regularized.classifier_head.mil._forward_hooks), 0)

    def test_aem_gradient_reaches_encoder_asrd_and_attention_without_extra_forwards(self):
        config = small_config()
        model = small_model(config)
        model.region_interaction.gamma.data.fill_(0.2)
        images = torch.rand(5, 3, 32, 32)
        sample = dict(slide_id="small", n_regions=5, label=1, lr_paths=list(range(5)), data_mode="native512")
        with patch.object(engine, "load_images", side_effect=lambda paths, size: images[paths]), \
             patch.object(model.encoder, "forward", wraps=model.encoder.forward) as encode:
            total, cls, sr, logits, kl, negative_entropy = aem.wsi_loss(
                model, sample, torch.device("cpu"), config, weight=0.01)
        self.assertEqual(encode.call_count, 1)
        torch.testing.assert_close(total, cls + 0.01 * negative_entropy)
        negative_entropy.backward()
        for module in (model.encoder.layer4, model.region_interaction, model.classifier_head.mil):
            gradients = [p.grad for p in module.parameters() if p.requires_grad]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.0)
        self.assertEqual(sr.item(), 0.0)
        self.assertEqual(len(model.classifier_head.mil._forward_hooks), 0)

    def test_existing_sr_and_vib_terms_are_preserved(self):
        config = small_config()
        config["lambda_sr"] = 0.05
        config["sr_train_crop"] = 8
        config["classification"]["vib"]["enabled"] = True
        model = small_model(config).train()
        images = torch.rand(3, 3, 32, 32)
        sample = dict(slide_id="joint", n_regions=3, label=1, lr_paths=list(range(3)),
                      hr_paths=list(range(3)), data_mode="native512")
        with patch.object(engine, "load_images", side_effect=lambda paths, size: images[paths]), \
             patch.object(engine, "load_hr_crops", return_value=torch.zeros(1, 3, 8, 8)), \
             patch.object(model.sr, "decode_crop", side_effect=lambda feature, box: feature[:, :3, :8, :8]):
            total, cls, sr, logits, kl, entropy = aem.wsi_loss(
                model, sample, torch.device("cpu"), config, weight=0.01)
        torch.testing.assert_close(total, cls + 0.05 * sr + model.vib_beta * kl + 0.01 * entropy)
        self.assertGreater(sr.item(), 0)
        self.assertGreater(kl.item(), 0)
        total.backward()
        self.assertGreater(engine.gradient_norm(model.sr), 0.0)
        self.assertGreater(engine.gradient_norm(model.classifier_head.vib), 0.0)
        with patch.object(engine, "wsi_loss", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                aem.wsi_loss(model, sample, torch.device("cpu"), config, weight=0.01)
        self.assertEqual(len(model.classifier_head.mil._forward_hooks), 0)


class AEMIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.fixture = camil_tests.CamilTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def run_fold(self, asrd, device):
        fixture = self.fixture
        config = small_config()
        config["classification"]["region_interaction"]["enabled"] = asrd
        config["precision"] = "bf16" if device == "cuda:0" else "fp32"
        # Make the two regions different, using files only in the temporary fixture.
        for root in fixture.roots.values():
            for index, path in enumerate(sorted((root / "images_512").glob("*.jpg"))):
                pixels = torch.randint(0, 40, (512, 512, 3), generator=torch.Generator().manual_seed(index))
                pixels = (pixels + (60 if index % 2 else 150)).byte().numpy()
                camil_tests.Image.fromarray(pixels).save(path)
        before = fixture._input_hashes()
        path = fixture.root / "aem.yaml"
        path.write_text(yaml.safe_dump(config))
        output = fixture.root / "aem-run"
        args = ["--config", str(path), "--protocol-dir", str(fixture.protocol), "--fold", "0",
                "--device", device, "--output", str(output)]
        with patch.object(trainer.engine, "evaluate", wraps=engine.evaluate) as evaluate, \
             redirect_stdout(io.StringIO()):
            trainer.main(args)
        self.assertEqual(evaluate.call_count, 5)  # Train/Val each epoch, then Test exactly once.
        fold = output / "fold_0"
        best = torch.load(fold / "best.pth", weights_only=True, map_location="cpu")
        self.assertEqual(best["training_regularization"]["type"], "aem-negative-entropy-v1")
        self.assertEqual(best["ema_updates"], 6 * best["epoch"])
        self.assertEqual(best["config"]["classification"]["aem"], config["classification"]["aem"])
        restored = model_from_checkpoint(best, device).eval()
        self.assertEqual(restored.architecture(), best["architecture"])
        self.assertEqual(len(restored.classifier_head.mil._forward_hooks), 0)
        groups = best["optimizer_state"]["param_groups"]
        self.assertEqual(sum(len(g["params"]) for g in groups),
                         sum(p.requires_grad for p in small_model(config).parameters()))
        protocol = trainer.camil.read_camil_protocol(fixture.protocol)
        samples, _ = trainer.camil.load_camil_samples(protocol)
        _, _, test = trainer.camil.make_camil_datasets(protocol, samples, 0)
        evaluated = engine.evaluate(restored, trainer.camil._loader(test), torch.device(device), best["config"])
        with (fold / "test_predictions.csv").open(newline="") as handle:
            saved = list(csv.DictReader(handle))
        self.assertEqual([float(r["p_tumor"]) for r in saved],
                         [r["p_tumor"] for r in evaluated["predictions"]])
        with (fold / "history.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            self.assertGreater(float(row["train_attention_entropy"]), 0.0)
            self.assertAlmostEqual(float(row["train_loss_total"]),
                                   float(row["train_loss_cls"]) + float(row["train_loss_aem_weighted"]), places=6)
        self.assertEqual(fixture._input_hashes(), before)
        metrics_before = (fold / "test_metrics.json").read_bytes()
        with redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            trainer.main(args)
        self.assertEqual((fold / "test_metrics.json").read_bytes(), metrics_before)
        damaged = copy.deepcopy(best)
        damaged["model_state"].pop(next(iter(damaged["model_state"])))
        with self.assertRaises(RuntimeError):
            model_from_checkpoint(damaged)

    def test_native512_adamw_aem_ema_checkpoint_and_test_without_asrd(self):
        self.run_fold(False, "cpu")

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 required")
    def test_native512_bf16_adamw_asrd_aem_ema_checkpoint_and_test(self):
        self.run_fold(True, "cuda:0")

    def test_disabled_entry_delegates_to_unchanged_baseline(self):
        config = small_config()
        config["classification"]["aem"]["enabled"] = False
        path = self.fixture.root / "off.yaml"
        path.write_text(yaml.safe_dump(config))
        args = ["--config", str(path), "--protocol-dir", str(self.fixture.protocol), "--check-data"]
        with patch.object(trainer.camil, "main") as original:
            trainer.main(args)
        original.assert_called_once_with(args)

    def test_five_folds_select_val_best_then_test_once_and_summarize(self):
        fixture = self.fixture
        config = small_config()
        path = fixture.root / "five.yaml"
        path.write_text(yaml.safe_dump(config))
        output = fixture.root / "five"
        protocol = trainer.camil.read_camil_protocol(fixture.protocol)
        expected_order = iter((fold, split, epoch) for fold in trainer.camil.FOLDS
                              for split, epoch in (("train", 1), ("val", 1), ("train", 2), ("val", 2), ("test", 1)))
        selected = {}
        original_evaluate, original_load_images = engine.evaluate, engine.load_images

        def evaluate(model, loader, device, current):
            fold, split, epoch = next(expected_order)
            self.assertEqual([s["slide_id"] for s in loader.dataset.samples], protocol.folds[fold][split])
            self.assertEqual(len(model.classifier_head.mil._forward_hooks), 0)
            if split == "test":
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, selected[fold][name], atol=0, rtol=0)
            result = original_evaluate(model, loader, device, current)
            if split == "val":
                # Force a known selection order, independent of this tiny synthetic AUC.
                result["auc"] = 0.8 if epoch == 1 else 0.7
                if epoch == 1:
                    selected[fold] = {k: v.clone() for k, v in model.state_dict().items()}
            return result

        def small_images(paths, size):
            return original_load_images(paths, size)[:, :, :32, :32]

        before = fixture._input_hashes()
        with patch.object(trainer.engine, "evaluate", side_effect=evaluate), \
             patch.object(engine, "load_images", side_effect=small_images), redirect_stdout(io.StringIO()):
            trainer.main(["--config", str(path), "--protocol-dir", str(fixture.protocol),
                          "--device", "cpu", "--output", str(output)])
        self.assertIsNone(next(expected_order, None))
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["fold_count"], 5)
        self.assertEqual([f["checkpoint_epoch"] for f in summary["folds"]], [1] * 5)
        aucs = [f["auc"] for f in summary["folds"]]
        self.assertEqual(summary["test_auc_mean"], mean(aucs))
        self.assertEqual(summary["test_auc_std"], stdev(aucs))
        self.assertEqual(fixture._input_hashes(), before)


if __name__ == "__main__":
    unittest.main()
