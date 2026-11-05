import copy
import csv
import io
import math
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
import yaml
from PIL import Image

import engine
import engine_gcsam
import eval_e2e
import train_e2e_gcsam as trainer
import train_e2e_shared as baseline
from models.baseline import model_from_checkpoint
from models.shared_resnet_sr import CHECKPOINT_FORMAT, SharedResNetSRMIL
from optimizers.gcsam import GCSAM
from wsi_data import Camelyon16WSI, load_images


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/native512_gcsam.yaml"


class GCSAMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def config(self):
        config = trainer.load_config(CONFIG)
        config["shared_encoder"]["pretrained"] = False
        config["classification"].update(embedding_dim=32, attention_hidden_dim=16)
        config["metasr"]["channels"] = 16
        config["precision"] = "fp32"
        return config

    def model(self, config):
        torch.manual_seed(41)
        sr = config["metasr"]
        return SharedResNetSRMIL(
            shared_encoder=config["shared_encoder"], classification=config["classification"],
            scale=sr["scale"], sr_channels=sr["channels"], sr_dropout=sr["dropout"],
        )

    def fixture(self):
        lr = torch.linspace(0, 1, 3 * 3 * 16 * 16).reshape(3, 3, 16, 16)
        sample = {"slide_id": "synthetic", "label": 1, "n_regions": 3,
                  "lr_paths": [0, 1, 2], "hr_paths": [0, 1, 2], "data_mode": "native512"}
        return lr, sample

    def test_post_gc_normalization_exact_restore_and_uncentered_single_adamw_step(self):
        parameters = [torch.nn.Parameter(torch.arange(6, dtype=torch.float64).reshape(2, 3)),
                      torch.nn.Parameter(torch.linspace(-1, 1, 24, dtype=torch.float64).reshape(2, 3, 2, 2)),
                      torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))]
        frozen = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float64), requires_grad=False)
        original = [p.detach().clone() for p in parameters]
        gradients = [torch.tensor([[1., 2., 3.], [4., 8., 12.]], dtype=torch.float64),
                     torch.arange(24, dtype=torch.float64).reshape(2, 3, 2, 2),
                     torch.tensor([-4., 2.], dtype=torch.float64)]
        centered = [grad - grad.mean(tuple(range(1, grad.ndim)), keepdim=True)
                    if grad.ndim > 1 else grad.clone() for grad in gradients]
        expected_norm = torch.cat([g.flatten() for g in centered]).norm()
        base = torch.optim.AdamW(parameters + [frozen], lr=2e-4, weight_decay=0.01)
        optimizer = GCSAM(base, rho=0.05)
        for p, g in zip(parameters, gradients):
            p.grad = g.clone()
        with patch.object(base, "step", wraps=base.step) as update:
            torch.testing.assert_close(optimizer.first_step(zero_grad=True), expected_norm)
            self.assertEqual(update.call_count, 0)
            self.assertFalse(base.state)
            for p, before, g in zip(parameters, original, centered):
                torch.testing.assert_close(p, before + 0.05 * g / (expected_norm + 1e-12))
                self.assertIsNone(p.grad)
            delta = torch.cat([(p - before).flatten() for p, before in zip(parameters, original)])
            torch.testing.assert_close(delta.norm(), torch.tensor(0.05, dtype=torch.float64))
            with self.assertRaisesRegex(RuntimeError, "checkpoint"):
                optimizer.state_dict()
            optimizer.restore_parameters()
            for p, before in zip(parameters, original):
                self.assertTrue(torch.equal(p, before))
            # Nonzero-mean second gradients must be passed to AdamW unchanged.
            second = [torch.full_like(p, 3.0) for p in parameters]
            reference = [torch.nn.Parameter(before.clone()) for before in original]
            reference_optimizer = torch.optim.AdamW(reference, lr=2e-4, weight_decay=0.01)
            for p, g, ref in zip(parameters, second, reference):
                p.grad, ref.grad = g.clone(), g.clone()
            optimizer.second_step()
            reference_optimizer.step()
            self.assertEqual(update.call_count, 1)
            with self.assertRaisesRegex(RuntimeError, "preceding"):
                optimizer.second_step()
            self.assertEqual(update.call_count, 1)
        for p, ref, g in zip(parameters, reference, second):
            torch.testing.assert_close(p, ref, atol=0, rtol=0)
            torch.testing.assert_close(p.grad, g, atol=0, rtol=0)
        self.assertTrue(torch.equal(frozen, torch.ones_like(frozen)))
        self.assertFalse(optimizer.step_pending)

    def test_parameter_groups_lr_decay_exclusions_and_scheduler_use_actual_adamw(self):
        config = self.config()
        model = self.model(config)
        optimizer = trainer.build_optimizer(model, config)
        reference = baseline.build_optimizer(model, dict(config, optimizer="adamw"))
        for actual, expected in zip(optimizer.param_groups, reference.param_groups):
            for name in ("name", "lr", "weight_decay"):
                self.assertEqual(actual[name], expected[name])
            self.assertEqual([id(p) for p in actual["params"]], [id(p) for p in expected["params"]])
            for p in actual["params"]:
                self.assertTrue(p.requires_grad)
                if p.ndim == 1:
                    self.assertEqual(actual["weight_decay"], 0)
        self.assertEqual({g["lr"] for g in optimizer.param_groups if g["name"].startswith("encoder")}, {1e-5})
        self.assertEqual({g["lr"] for g in optimizer.param_groups if g["name"].startswith("head")}, {2e-4})
        config["scheduler_patience"] = 0
        scheduler = trainer.build_scheduler(optimizer.base_optimizer, config)
        self.assertIs(scheduler.optimizer, optimizer.base_optimizer)
        scheduler.step(1.0)
        scheduler.step(2.0)
        self.assertEqual({g["lr"] for g in optimizer.param_groups if g["name"].startswith("head")}, {1e-4})

    def test_two_pass_order_clipping_frozen_parameters_bn_and_single_update(self):
        for micro in (False, True):
            with self.subTest(micro=micro):
                config = self.config()
                config["training"]["use_region_microbatch"] = micro
                model = self.model(config)
                optimizer = trainer.build_optimizer(model, config)
                lr, sample = self.fixture()
                before = {name: value.detach().clone() for name, value in model.state_dict().items()}
                events, inputs = [], []
                first_step = optimizer.first_step
                base_step = optimizer.base_optimizer.step
                original_clip = torch.nn.utils.clip_grad_norm_

                def ascent(*args, **kwargs):
                    result = first_step(*args, **kwargs)
                    self.assertTrue(any(not torch.equal(p, before[n]) for n, p in model.named_parameters()))
                    self.assertTrue(all(p.grad is None for p in model.parameters()))
                    events.append("ascent")
                    return result

                def assert_restored():
                    for name, p in model.named_parameters():
                        self.assertTrue(torch.equal(p, before[name]), name)

                def clip(*args, **kwargs):
                    assert_restored()
                    events.append("clip")
                    return original_clip(*args, **kwargs)

                def update(*args, **kwargs):
                    assert_restored()
                    self.assertEqual(events, ["ascent", "clip"])
                    self.assertLessEqual(optimizer.gradient_norm().item(), 1.00001)
                    events.append("adamw")
                    return base_step(*args, **kwargs)

                hook = model.encoder.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
                with patch("engine.load_images", side_effect=lambda paths, size: lr[paths]), \
                     patch("engine.load_hr_crops", side_effect=AssertionError("SR0 read HR")), \
                     patch.object(model.sr, "fuse_stages", side_effect=AssertionError("SR0 fused")), \
                     patch.object(optimizer, "first_step", side_effect=ascent), \
                     patch.object(optimizer.base_optimizer, "step", side_effect=update) as step, \
                     patch("engine_gcsam.torch.nn.utils.clip_grad_norm_", side_effect=clip):
                    result = engine_gcsam.train_wsi(model, sample, optimizer, torch.device("cpu"), config)
                hook.remove()
                self.assertEqual(step.call_count, 1)
                self.assertEqual(events, ["ascent", "clip", "adamw"])
                chunks = 2 if micro else 1
                self.assertEqual(len(inputs), 2 * chunks)
                for a, b in zip(inputs[:chunks], inputs[chunks:]):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        self.assertTrue(torch.equal(p, before[name]), name)
                for name, value in model.named_buffers():
                    self.assertTrue(torch.equal(value, before[name]), name)
                self.assertTrue(all(not m.training for m in model.encoder.modules()
                                    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)))
                self.assertTrue(any(not torch.equal(p, before[n]) for n, p in model.named_parameters()))
                for p in (model.encoder.layer4[0].conv1.weight, model.classifier_head.projection[1].weight,
                          model.classifier_head.classifier[-1].weight):
                    self.assertTrue(torch.isfinite(p.grad).all())
                    self.assertGreater(p.grad.abs().sum().item(), 0)
                self.assertTrue(all(p.grad is None for p in model.sr.parameters()))
                self.assertFalse(optimizer.step_pending)
                self.assertTrue(all(math.isfinite(result[key]) for key in
                                    ("train_loss_cls_first", "train_loss_cls_second", "grad_norm_first",
                                     "grad_norm_gc", "grad_norm_second")))

    def test_same_region_selection_real_paired_augmentation_pixels_and_rng_replay(self):
        config = self.config()
        model = self.model(config)
        optimizer = trainer.build_optimizer(model, config)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = []
            for index in range(27):
                path = root / f"region_{index}.png"
                pixels = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3) + index
                Image.fromarray(pixels).save(path)
                paths.append(path)
            sample = {"slide_id": "augmented", "label": 1, "n_regions": 27,
                      "lr_paths": paths, "hr_paths": paths, "data_mode": "native512"}
            dataset = Camelyon16WSI([sample], max_regions=24, augmentation=config["augmentation"])
            import wsi_data
            seen, markers, signatures, sample_ids, labels = [], [], [], [], []
            after_first_load = []
            original_loss = engine_gcsam.wsi_loss

            def loss(model, selected, *args, **kwargs):
                sample_ids.append(id(selected))
                labels.append(selected["label"])
                return original_loss(model, selected, *args, **kwargs)

            def images(selected, size):
                self.assertEqual(size, 512)
                signatures.append([(str(p.path), p.params) for p in selected])
                markers.append((random.random(), np.random.rand(), torch.rand(1).item()))
                value = load_images(selected, 16)
                seen.append(value.clone())
                if not after_first_load:
                    after_first_load.append(engine_gcsam._rng_state(torch.device("cpu")))
                return value

            with patch("wsi_data.random.sample", wraps=random.sample) as select, \
                 patch("wsi_data.generate_transform_params", wraps=wsi_data.generate_transform_params) as augment, \
                 patch("engine.load_images", side_effect=images), \
                 patch("engine_gcsam.wsi_loss", side_effect=loss):
                selected = dataset[0]
                engine_gcsam.train_wsi(model, selected, optimizer, torch.device("cpu"), config)
            self.assertEqual(select.call_count, 1)
            self.assertEqual(augment.call_count, 24)
            self.assertEqual(selected["n_regions"], 24)
            self.assertEqual(sample_ids, [id(selected), id(selected)])
            self.assertEqual(labels, [1, 1])
            self.assertEqual(signatures[0], signatures[1])
            self.assertEqual(markers[0], markers[1])
            torch.testing.assert_close(seen[0], seen[1], atol=0, rtol=0)
            current = engine_gcsam._rng_state(torch.device("cpu"))
            self.assertEqual(current[0], after_first_load[0][0])
            np.testing.assert_equal(current[1], after_first_load[0][1])
            self.assertTrue(torch.equal(current[2], after_first_load[0][2]))

    def test_nan_second_stage_restores_weights_and_does_not_update(self):
        config = self.config()
        model = self.model(config)
        optimizer = trainer.build_optimizer(model, config)
        lr, sample = self.fixture()
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        original_loss = engine_gcsam.wsi_loss
        calls = []

        def loss(*args, **kwargs):
            result = original_loss(*args, **kwargs)
            calls.append(1)
            return (result[0] * float("nan"), *result[1:]) if len(calls) == 2 else result

        with patch("engine.load_images", return_value=lr), \
             patch("engine_gcsam.wsi_loss", side_effect=loss), \
             patch.object(optimizer.base_optimizer, "step", wraps=optimizer.base_optimizer.step) as update:
            with self.assertRaisesRegex(FloatingPointError, "second"):
                engine_gcsam.train_wsi(model, sample, optimizer, torch.device("cpu"), config)
        self.assertEqual(update.call_count, 0)
        self.assertFalse(optimizer.step_pending)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_optimizer_state_restores_adamw_moments_rho_and_next_update(self):
        p = torch.nn.Parameter(torch.tensor([[1., 2.], [3., 5.]]))
        optimizer = GCSAM(torch.optim.AdamW([p], lr=1e-3, weight_decay=0.01), rho=0.1)

        def update(opt, parameter):
            opt.zero_grad(set_to_none=True)
            parameter.square().sum().backward()
            opt.first_step(zero_grad=True)
            parameter.square().sum().backward()
            opt.restore_parameters()
            opt.second_step()

        update(optimizer, p)
        saved = copy.deepcopy(optimizer.state_dict())
        self.assertTrue(saved["state"])
        self.assertTrue(all("exp_avg" in value and "exp_avg_sq" in value for value in saved["state"].values()))
        restored_p = torch.nn.Parameter(p.detach().clone())
        restored = GCSAM(torch.optim.AdamW([restored_p], lr=0.3), rho=0.9)
        restored.load_state_dict(saved)
        self.assertEqual(restored.rho, 0.1)
        self.assertIs(restored.state, restored.base_optimizer.state)
        self.assertIs(restored.param_groups, restored.base_optimizer.param_groups)
        update(optimizer, p)
        update(restored, restored_p)
        torch.testing.assert_close(p, restored_p, atol=0, rtol=0)

    def test_nonfinite_first_gradient_and_zero_centered_gradient_are_safe(self):
        p = torch.nn.Parameter(torch.ones(2, 3))
        optimizer = GCSAM(torch.optim.AdamW([p], lr=1e-3))
        original = p.detach().clone()
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                p.grad = torch.full_like(p, value)
                with self.assertRaises(FloatingPointError):
                    optimizer.first_step(zero_grad=True)
                self.assertTrue(torch.equal(p, original))
                self.assertFalse(optimizer.step_pending)
        p.grad = torch.ones_like(p)
        self.assertEqual(optimizer.first_step(zero_grad=True).item(), 0)
        self.assertTrue(torch.equal(p, original))
        p.square().sum().backward()
        optimizer.restore_parameters()
        optimizer.second_step()
        self.assertTrue(torch.isfinite(p).all())
        self.assertFalse(optimizer.step_pending)

    def test_rho_zero_matches_unchanged_baseline_adamw_update(self):
        config = self.config()
        config["gcsam"]["rho"] = 0.0
        model = self.model(config)
        reference = copy.deepcopy(model)
        optimizer = trainer.build_optimizer(model, config)
        ordinary = baseline.build_optimizer(reference, dict(config, optimizer="adamw"))
        lr, sample = self.fixture()
        with patch("engine.load_images", return_value=lr):
            actual = engine_gcsam.train_wsi(model, sample, optimizer, torch.device("cpu"), config)
            expected = engine.train_wsi(reference, sample, ordinary, torch.device("cpu"), config)
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
        for key in ("train_loss_cls", "train_loss_sr", "train_loss_total", "grad_norm"):
            self.assertEqual(actual[key], expected[key])

    def test_full_config_is_additive_and_invalid_gcsam_settings_are_rejected(self):
        config = trainer.load_config(CONFIG)
        expected = baseline.load_config(ROOT / "configs/native512_shared_resnet18_mean_sr0_full.yaml")
        expected["classification"].update(region_dropout=0, attention_dropout=0,
                                          projection_dropout=0, classifier_dropout=0)
        expected["metasr"]["dropout"] = 0
        expected.update(optimizer="gcsam", gcsam={"enabled": True, "base_optimizer": "adamw",
                                                  "rho": 0.05, "start_epoch": 1})
        self.assertEqual(config, expected)
        with self.assertRaisesRegex(ValueError, "optimizer"):
            baseline.load_config(CONFIG)
        invalid = [("gcsam", "rho", -1), ("gcsam", "rho", float("nan")),
                   ("gcsam", "rho", True), ("gcsam", "enabled", False),
                   ("gcsam", "base_optimizer", "adam"), ("gcsam", "start_epoch", 2),
                   ("gcsam", "start_epoch", True), ("classification", "projection_dropout", 0.2),
                   ("metasr", "dropout", 0.1), ("shared_encoder", "freeze_bn", False)]
        for section, key, value in invalid:
            with self.subTest(section=section, key=key, value=value):
                candidate = copy.deepcopy(config)
                candidate[section][key] = value
                with patch("train_e2e_gcsam.yaml.safe_load", return_value=candidate), self.assertRaises(ValueError):
                    trainer.load_config(CONFIG)

    def test_training_entry_updates_ema_once_schedules_base_and_keeps_eval_checkpoint_compatible(self):
        config = self.config()
        config.update(epochs=2, min_epochs=1, scheduler_patience=0)
        lr, sample = self.fixture()
        dataset = Camelyon16WSI([dict(sample, slide_id=str(label), label=label) for label in (0, 1)])
        optimizers, emas, schedulers = [], [], []
        real_build, real_ema, real_scheduler = trainer.build_optimizer, baseline.ModelEMA, baseline.build_scheduler

        def build(model, current):
            optimizer = real_build(model, current)
            optimizer.base_optimizer.step = Mock(wraps=optimizer.base_optimizer.step)
            optimizers.append(optimizer)
            return optimizer

        def ema(model, decay):
            instance = real_ema(model, decay)
            original = instance.update

            def update(current):
                self.assertFalse(optimizers[0].step_pending)
                self.assertEqual(optimizers[0].base_optimizer.step.call_count, instance.updates + 1)
                original(current)

            instance.update = Mock(side_effect=update)
            emas.append(instance)
            return instance

        def scheduler(base, current):
            self.assertIs(base, optimizers[0].base_optimizer)
            instance = real_scheduler(base, current)

            class SchedulerSpy:
                step = Mock(wraps=instance.step)

                def state_dict(self):
                    return instance.state_dict()

            observed = SchedulerSpy()
            schedulers.append(observed)
            return observed

        def evaluate(*args):
            return {"loss": 1.0, "auc": 0.5, "acc": 0.5, "bacc": 0.5,
                    "predictions": [{"slide_id": "0", "label": 0, "p_tumor": 0.2,
                                     "prediction": 0, "n_regions": 3},
                                    {"slide_id": "1", "label": 1, "p_tumor": 0.8,
                                     "prediction": 1, "n_regions": 3}]}

        self.assertIs(trainer.evaluate, engine.evaluate)
        self.assertIs(trainer.balanced_threshold, baseline.balanced_threshold)
        self.assertIs(trainer.apply_threshold, baseline.apply_threshold)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path, output = root / "config.yaml", root / "gcsam-run"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            args = ["train_e2e_gcsam.py", "--config", str(config_path), "--data-root", str(root),
                    "--output", str(output), "--device", "cpu", "--fold", "4", "--split-scheme", "5fold"]
            with patch.object(sys, "argv", args), \
                 patch("train_e2e_gcsam.load_fold_datasets", return_value=(dataset, dataset)) as loader, \
                 patch("train_e2e_gcsam.provenance", return_value={}), \
                 patch("train_e2e_gcsam.run_metadata", return_value={"experiment_id": "gcsam-test"}), \
                 patch("train_e2e_gcsam.build_optimizer", side_effect=build), \
                 patch("train_e2e_gcsam.ModelEMA", side_effect=ema), \
                 patch("train_e2e_gcsam.build_scheduler", side_effect=scheduler), \
                 patch("train_e2e_gcsam.evaluate", side_effect=evaluate), \
                 patch("engine.load_images", side_effect=lambda paths, size: lr[paths]), \
                 redirect_stdout(io.StringIO()):
                trainer.main()
            self.assertEqual(optimizers[0].base_optimizer.step.call_count, 4)
            self.assertEqual(emas[0].update.call_count, 4)
            self.assertEqual(emas[0].updates, 4)
            self.assertEqual(schedulers[0].step.call_count, 2)
            self.assertFalse(loader.call_args.kwargs["require_hr"])
            self.assertEqual(loader.call_args.kwargs["max_train_regions"], 24)
            self.assertEqual(loader.call_args.kwargs["split_scheme"], "5fold")
            self.assertEqual(loader.call_args.kwargs["augmentation"], config["augmentation"])
            with (output / "history.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["optimizer"] == "gcsam" and float(row["rho"]) == 0.05 for row in rows))
            self.assertTrue(all(row["train_loss_cls_first"] == row["train_loss_cls"] for row in rows))
            saved = torch.load(output / "last.pth", map_location="cpu", weights_only=False)
            best = torch.load(output / "best.pth", map_location="cpu", weights_only=False)
            self.assertEqual(saved["format"], CHECKPOINT_FORMAT)
            self.assertEqual(saved["ema_updates"], 4)
            self.assertTrue(saved["optimizer_state"]["state"])
            self.assertEqual(saved["optimizer_state"]["gcsam"]["normalization"], "after_gc")
            self.assertIsNotNone(saved["scheduler_state"])
            self.assertEqual(saved["fold"], 4)
            self.assertEqual(saved["config"]["optimizer"], "gcsam")
            self.assertTrue(all(g["lr"] == optimizers[0].param_groups[i]["lr"]
                                for i, g in enumerate(saved["optimizer_state"]["param_groups"])))
            restored = model_from_checkpoint(best).eval()
            with torch.no_grad():
                expected_logits = restored.forward_embeddings(restored.encode_classification_regions(lr))
            predictions = root / "predictions.csv"
            args = ["eval_e2e.py", "--checkpoint", str(output / "best.pth"), "--data-root", str(root),
                    "--device", "cpu", "--predictions-csv", str(predictions)]
            with patch.object(sys, "argv", args), \
                 patch("eval_e2e.load_fold_datasets", return_value=(dataset, dataset)) as eval_loader, \
                 patch("eval_e2e.provenance", return_value={}), \
                 patch("engine.load_images", side_effect=lambda paths, size: lr[paths]), \
                 redirect_stdout(io.StringIO()):
                eval_e2e.main()
            self.assertEqual(eval_loader.call_args.kwargs["split_scheme"], "5fold")
            with predictions.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(float(rows[0]["p_tumor"]), expected_logits.softmax(1)[0, 1].item(), places=6)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 required")
    def test_cuda_bf16_two_pass_update_and_cuda_rng_replay(self):
        config = self.config()
        config["precision"] = "bf16"
        device = torch.device("cuda:0")
        model = self.model(config).to(device)
        optimizer = trainer.build_optimizer(model, config)
        lr, sample = self.fixture()
        markers, first_rng = [], []

        def images(paths, size):
            markers.append(torch.rand(4, device=device).cpu())
            if not first_rng:
                first_rng.append(torch.cuda.get_rng_state_all())
            return lr[paths]

        before = model.classifier_head.classifier[-1].weight.detach().clone()
        with patch("engine.load_images", side_effect=images), \
             patch.object(optimizer.base_optimizer, "step", wraps=optimizer.base_optimizer.step) as update:
            result = engine_gcsam.train_wsi(model, sample, optimizer, device, config)
        self.assertEqual(update.call_count, 1)
        torch.testing.assert_close(markers[0], markers[1], atol=0, rtol=0)
        for a, b in zip(torch.cuda.get_rng_state_all(), first_rng[0]):
            self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(before, model.classifier_head.classifier[-1].weight))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertTrue(math.isfinite(result["train_loss_cls_second"]))
        self.assertFalse(optimizer.step_pending)


if __name__ == "__main__":
    unittest.main()
