import ast
import copy
import io
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from data_modes import DataSpec
from engine import gradient_norm, task_gradient_metrics, train_wsi
from models.shared_resnet_sr import SharedResNetSRMIL
from train_e2e_shared import ModelEMA, build_optimizer, main
from wsi_data import Camelyon16WSI


class GradientDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def assert_exact(self, first, second):
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second), "Tensor changed with diagnostics")
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_exact(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_exact(left, right)
        else:
            self.assertEqual(first, second)

    def test_task_gradients_match_oracle_without_touching_grad_or_freeing_graph(self):
        vector = torch.nn.Parameter(torch.tensor([1., 2.]))
        scalar = torch.nn.Parameter(torch.tensor(1.))
        unused = torch.nn.Parameter(torch.tensor(3.))
        vector.grad = torch.tensor([11., 12.])
        parameters = (vector, scalar, unused)
        ce = (vector * torch.tensor([3., 4.])).sum() + 12 * scalar
        sr = (vector * torch.tensor([-4., 3.])).sum()
        metrics = task_gradient_metrics(parameters, ce, sr, 0.2)
        self.assertEqual(metrics["grad_ce_shared"], 13.)
        self.assertEqual(metrics["grad_sr_shared_raw"], 5.)
        self.assertEqual(metrics["grad_sr_shared_weighted"], 1.)
        self.assertEqual(metrics["grad_ratio"], 1 / 13)
        self.assertEqual(metrics["grad_cosine"], 0.)
        self.assertEqual(metrics["sr_status"], "active")
        self.assertTrue(torch.equal(vector.grad, torch.tensor([11., 12.])))
        self.assertIsNone(scalar.grad)
        self.assertIsNone(unused.grad)
        (ce + 0.2 * sr).backward()
        torch.testing.assert_close(vector.grad, torch.tensor([13.2, 16.6]))
        self.assertEqual(scalar.grad.item(), 12.)

    def test_sr_zero_and_frozen_layer4_skip_inapplicable_gradients(self):
        parameter = torch.nn.Parameter(torch.ones(2))
        ce = parameter.sum()
        with patch("engine.torch.autograd.grad", wraps=torch.autograd.grad) as extract:
            metrics = task_gradient_metrics((parameter,), ce, ce.new_zeros(()), 0.)
        self.assertEqual(extract.call_count, 1)
        self.assertEqual(metrics["sr_status"], "not_applicable_lambda_sr_zero")
        for name in ("grad_sr_shared_raw", "grad_sr_shared_weighted", "grad_ratio", "grad_cosine"):
            self.assertIsNone(metrics[name])
        self.assertIsNone(parameter.grad)
        with patch("engine.torch.autograd.grad", side_effect=AssertionError("Frozen layer4")):
            metrics = task_gradient_metrics((), ce, ce, 0.1)
        self.assertIsNone(metrics["grad_ce_shared"])
        self.assertEqual(metrics["sr_status"], "not_applicable_no_trainable_layer4")
        metrics = task_gradient_metrics((parameter,), (parameter * 0).sum(), ce, 0.1)
        self.assertEqual(metrics["grad_ce_shared"], 0.)
        self.assertIsNone(metrics["grad_ratio"])
        self.assertIsNone(metrics["grad_cosine"])

    def one_update(self, enabled, coefficient, micro=False, vib=False, native=False):
        device = torch.device("cuda:0" if native else "cpu")
        spec = DataSpec("native512", 512, 8192, 16) if native else DataSpec("legacy256", 16, 32, 2)
        n = 1 if native else 3
        clip = 0.01
        torch.manual_seed(41)
        model = SharedResNetSRMIL(
            shared_encoder={"pretrained": False, "trainable": "layer4", "freeze_bn": True},
            classification={"embedding_dim": 256 if native else 32, "attention_hidden_dim": 16,
                            "topk": 2, "region_dropout": 0.2, "attention_dropout": 0.2,
                            "projection_dropout": 0.2, "classifier_dropout": 0.4,
                            "vib": {"enabled": vib, "beta": 1e-3}},
            scale=spec.scale, sr_channels=64 if native else 16, sr_dropout=0.05,
        ).to(device)
        config = {"training": {"use_region_microbatch": micro, "region_microbatch_size": 2},
                  "precision": "bf16" if native else "fp32", "lambda_sr": coefficient,
                  "sr_train_crop": 128 if native else 8, "gradient_clip_norm": clip,
                  "shared_encoder": {"encoder_lr": 1e-5}, "classification": {"head_lr": 2e-4},
                  "optimizer": "adamw", "lr": 2e-4, "sr_lr": 1e-4, "weight_decay": 0.01,
                  "class_weight_values": [0.8, 1.2], "label_smoothing": 0.1}
        optimizer = build_optimizer(model, config)
        ema = ModelEMA(model, 0.999)
        lr = torch.linspace(0, 1, n * 3 * spec.lr_size**2).reshape(n, 3, spec.lr_size, spec.lr_size)
        crop = config["sr_train_crop"]
        truth = torch.linspace(0, 1, 3 * crop**2).reshape(1, 3, crop, crop)
        sample = {"slide_id": "synthetic", "n_regions": n, "label": 1,
                  "lr_paths": list(range(n)), "hr_paths": list(range(n))}
        clipping = {}
        original_clip = torch.nn.utils.clip_grad_norm_

        def record_clip(parameters, maximum):
            clipping["pre"] = gradient_norm(model.encoder.layer4)
            result = original_clip(parameters, maximum)
            clipping["post"] = gradient_norm(model.encoder.layer4)
            return result

        original_grad = torch.autograd.grad

        def extract(*args, **kwargs):
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            result = original_grad(*args, **kwargs)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            return result

        torch.manual_seed(2026)
        torch.cuda.manual_seed_all(2026)
        output = io.StringIO()
        with patch("engine._sample_data_spec", return_value=spec), \
             patch("engine.load_images", side_effect=lambda paths, size: lr[paths]), \
             patch("engine.load_hr_crops", return_value=truth) as hr, \
             patch.object(model.encoder, "forward", wraps=model.encoder.forward) as forward, \
             patch("engine.torch.autograd.grad", side_effect=extract) as gradients, \
             patch("engine.torch.nn.utils.clip_grad_norm_", side_effect=record_clip), \
             redirect_stdout(output):
            result = train_wsi(model, sample, optimizer, device, config,
                               **({"gradient_task_step": 10} if enabled else {}))
        ema.update(model)
        self.assertEqual(forward.call_count, math.ceil(n / 2) if micro else 1)
        self.assertEqual(hr.call_count, n if coefficient else 0)
        self.assertEqual(gradients.call_count, (2 if coefficient else 1) if enabled else 0)
        lines = output.getvalue().splitlines()
        if enabled:
            self.assertEqual(len(lines), 1)
            self.assertTrue(lines[0].startswith("[gradient_tasks] step=10 slide_id=synthetic N="))
            metrics = ast.literal_eval(lines[0][lines[0].index("{"):])
            self.assertEqual(metrics["loss_ce"], result["train_loss_cls"])
            self.assertEqual(metrics["loss_sr"], result["train_loss_sr"])
            self.assertEqual(metrics["lambda_sr"], coefficient)
            self.assertEqual(metrics["grad_total_shared_preclip"], clipping["pre"])
            self.assertEqual(metrics["grad_total_shared_postclip"], clipping["post"])
            self.assertGreater(clipping["pre"], clipping["post"])
            self.assertLessEqual(clipping["post"], clip)
            if coefficient:
                self.assertGreater(metrics["grad_ce_shared"], 0.)
                self.assertGreater(metrics["grad_sr_shared_raw"], 0.)
                self.assertEqual(metrics["grad_sr_shared_weighted"], coefficient * metrics["grad_sr_shared_raw"])
                self.assertEqual(metrics["grad_ratio"], metrics["grad_sr_shared_weighted"] / metrics["grad_ce_shared"])
                self.assertLessEqual(abs(metrics["grad_cosine"]), 1.)
            else:
                self.assertEqual(metrics["sr_status"], "not_applicable_lambda_sr_zero")
                self.assertIsNone(metrics["grad_sr_shared_raw"])
                if not vib:
                    self.assertEqual(metrics["grad_ce_shared"], clipping["pre"])
        else:
            self.assertEqual(lines, [])
        result.pop("max_memory_allocated")
        result.pop("max_memory_reserved")
        return {"result": result, "model": copy.deepcopy(model.state_dict()),
                "ema": copy.deepcopy(ema.module.state_dict()),
                "gradients": [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()],
                "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if native else None}

    def test_fixed_seed_updates_are_identical_with_sr_full_microbatch_and_vib(self):
        for coefficient, micro, vib in ((0., False, False), (0.1, False, False),
                                        (0.1, True, False), (0.1, False, True)):
            with self.subTest(coefficient=coefficient, micro=micro, vib=vib):
                self.assert_exact(self.one_update(False, coefficient, micro, vib),
                                  self.one_update(True, coefficient, micro, vib))

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 required")
    def test_native512_cuda_bf16_updates_are_identical(self):
        for coefficient in (0., 0.1):
            with self.subTest(coefficient=coefficient):
                self.assert_exact(self.one_update(False, coefficient, native=True),
                                  self.one_update(True, coefficient, native=True))

    def test_cli_defaults_off_and_schedules_every_ten_global_steps(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / "configs/native512_shared_resnet18_robust.yaml").read_text(encoding="utf-8"))
        config.update(epochs=2, min_epochs=1, early_stopping_patience=3, scheduler="none")
        config["shared_encoder"]["pretrained"] = False
        config["training"]["sr_stop_after_epochs"] = 1
        samples = [{"slide_id": str(i), "label": i % 2, "n_regions": 1,
                    "lr_paths": [0], "hr_paths": [0]} for i in range(11)]
        train_data, val_data = Camelyon16WSI(samples), Camelyon16WSI(samples[:2])
        calls = []

        def train(model, sample, optimizer, device, current, log_gradients, **kwargs):
            calls.append((kwargs, current["lambda_sr"]))
            return {"train_loss_cls": 1., "train_loss_sr": 1., "train_loss_total": 1., "grad_norm": 1.}

        def validate(*args):
            return {"loss": 1., "auc": 1., "acc": 1., "bacc": 1.,
                    "predictions": [{"slide_id": str(i), "label": i, "p_tumor": 0.2 + 0.6 * i,
                                     "prediction": i, "n_regions": 1} for i in (0, 1)]}

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = directory / "config.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            for enabled in (False, True):
                args = ["train_e2e_shared.py", "--config", str(path), "--data-root", str(directory),
                        "--output", str(directory / str(enabled)), "--device", "cpu"]
                if enabled:
                    args.append("--gradient-diagnostics")
                calls.clear()
                with patch.object(sys, "argv", args), \
                     patch("train_e2e_shared.run_shared_model_gate", return_value={}), \
                     patch("train_e2e_shared.load_fold_datasets", return_value=(train_data, val_data)), \
                     patch("train_e2e_shared.train_wsi", side_effect=train), \
                     patch("train_e2e_shared.evaluate", side_effect=validate), \
                     patch("train_e2e_shared.provenance", return_value={}), \
                     patch("train_e2e_shared.run_metadata", return_value={"experiment_id": "test"}), \
                     patch("train_e2e_shared.torch.save"), redirect_stdout(io.StringIO()):
                    main()
                selected = [(index, kwargs["gradient_task_step"], coefficient)
                            for index, (kwargs, coefficient) in enumerate(calls, 1) if kwargs]
                self.assertEqual(selected, [(10, 10, config["lambda_sr"]), (20, 20, 0.)] if enabled else [])


if __name__ == "__main__":
    unittest.main()
