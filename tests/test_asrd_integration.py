import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

import engine
import eval_e2e
import eval_e2e_gcsam_asrd
import eval_native512_reference
import train_e2e_gcsam as trainer
import train_e2e_gcsam_asrd
from data_modes import data_spec
from engine_gcsam import train_wsi
from models.baseline import model_from_checkpoint as load_old
from models.shared_resnet_sr import CHECKPOINT_FORMAT, SharedResNetSRMIL
from models.shared_resnet_sr_asrd import (
    ASRD_CHECKPOINT_FORMAT, build_model, model_from_checkpoint, normalize_region_interaction,
)
from train_e2e_shared import ModelEMA
from wsi_data import Camelyon16WSI


ROOT = Path(__file__).resolve().parents[1]


class ASRDIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def config(self, pooling, enabled):
        name = "mean" if pooling == "mean" else "attn"
        config = trainer.load_config(ROOT / f"configs/gcsam_{name}_asrd_90_10_seed42.yaml")
        config["shared_encoder"]["pretrained"] = False
        config["classification"]["region_interaction"]["enabled"] = enabled
        config.update(precision="fp32", epochs=1, min_epochs=1)
        return config

    def assert_tensor_equal(self, actual, expected):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_disabled_and_omitted_are_bit_exact_and_keep_rng(self):
        for pooling in ("mean", "attention"):
            config = self.config(pooling, False)
            embeddings = torch.randn(5, 256)
            for omitted in (False, True):
                with self.subTest(pooling=pooling, omitted=omitted):
                    classification = copy.deepcopy(config["classification"])
                    if omitted:
                        classification.pop("region_interaction")
                    baseline_config = {k: v for k, v in classification.items() if k != "region_interaction"}
                    kwargs = {"shared_encoder": config["shared_encoder"], "sr_dropout": 0}
                    torch.manual_seed(19)
                    baseline = SharedResNetSRMIL(classification=baseline_config, **kwargs)
                    expected_rng = torch.get_rng_state()
                    torch.manual_seed(19)
                    candidate = build_model(classification=classification, **kwargs)
                    self.assertIs(type(candidate), SharedResNetSRMIL)
                    self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                    self.assertEqual(candidate.architecture(), baseline.architecture())
                    for name, weight in baseline.state_dict().items():
                        self.assert_tensor_equal(candidate.state_dict()[name], weight)
                    self.assert_tensor_equal(candidate.forward_embeddings(embeddings),
                                             baseline.forward_embeddings(embeddings))

            torch.manual_seed(19)
            enabled = build_model(classification=dict(
                config["classification"], region_interaction={"enabled": True},
            ), **kwargs)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            self.assert_tensor_equal(enabled.forward_embeddings(embeddings),
                                     baseline.forward_embeddings(embeddings))

    def test_kl_entry_applies_asrd_once_and_strict_loader_checks_schema(self):
        config = self.config("mean", True)
        config["classification"]["vib"]["enabled"] = True
        config["classification"]["region_interaction"]["gamma_init"] = 0.4
        model = build_model(shared_encoder=config["shared_encoder"],
                            classification=config["classification"], sr_dropout=0).eval()
        h = torch.randn(4, 256)
        with patch.object(model.region_interaction, "forward", wraps=model.region_interaction.forward) as interaction:
            logits, kl = model.forward_embeddings_with_kl(h)
            self.assertEqual(interaction.call_count, 1)
            self.assertEqual(logits.shape, (1, 2))
            self.assertTrue(kl.isfinite())
        saved = {"format": ASRD_CHECKPOINT_FORMAT, "config": config,
                 "data_spec": data_spec(config).metadata(), "architecture": model.architecture(),
                 "model_state": dict(model.state_dict())}
        self.assert_tensor_equal(model_from_checkpoint(saved).eval().forward_embeddings(h),
                                 model.forward_embeddings(h))
        with self.assertRaisesRegex(ValueError, "architecture"):
            model_from_checkpoint(dict(saved, architecture=dict(saved["architecture"], mil_pooling=["attention"])))
        saved["model_state"].pop("region_interaction.gamma")
        with self.assertRaisesRegex(RuntimeError, "region_interaction.gamma"):
            model_from_checkpoint(saved)
        with self.assertRaisesRegex(ValueError, "ASRD checkpoint format"):
            model_from_checkpoint(dict(saved, format=CHECKPOINT_FORMAT))
        for settings in ([], {"enabled": 1}, {"enabled": True, "type": "other"}, {"rank": 0}, {"top_k": 3}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                normalize_region_interaction(settings)

    def test_training_saves_checkpoints_and_both_evaluation_routes(self):
        for pooling in ("mean", "attention"):
            for enabled in (False, True):
                with self.subTest(pooling=pooling, enabled=enabled):
                    self.check_entrypoints(pooling, enabled)

    def check_entrypoints(self, pooling, enabled):
        config = self.config(pooling, enabled)
        fmt = ASRD_CHECKPOINT_FORMAT if enabled else CHECKPOINT_FORMAT
        regions = torch.randn(5, 3, 32, 32)
        samples = [
            {"slide_id": f"test_{label}", "label": label, "n_regions": len(regions),
             "lr_paths": list(range(len(regions))), "data_mode": "native512"}
            for label in (0, 1)
        ]
        dataset = Camelyon16WSI(samples)
        splits = ROOT / "splits_90_10_repeated_seed42"
        protocol = json.loads((splits / "protocol.json").read_text(encoding="utf-8"))
        provenance = {
            "manifest": {"sha256": protocol["manifest_sha256"]},
            "labels": {"sha256": protocol["labels_sha256"]},
            "split": {"sha256": protocol["splits"][0]["csv_sha256"]},
        }
        with tempfile.TemporaryDirectory(prefix="asrd-test-") as temp:
            root = Path(temp)
            output = root / "run"
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            args = ["train_e2e_gcsam_asrd.py", "--config", str(config_path),
                    "--data-root", str(root), "--split-dir", str(splits),
                    "--output", str(output), "--device", "cpu"]
            with patch.object(sys, "argv", args), \
                 patch.object(trainer, "load_fold_datasets", return_value=(dataset, dataset)), \
                 patch.object(trainer, "provenance", return_value=provenance), \
                 patch.object(engine, "load_images", return_value=regions), redirect_stdout(io.StringIO()):
                train_e2e_gcsam_asrd.main()
            for name in ("config.yaml", "history.csv", "best_val_predictions.csv",
                         "run_info.json", "data_provenance.json"):
                self.assertTrue((output / name).is_file())
            best = torch.load(output / "best.pth", weights_only=False, map_location="cpu")
            last = torch.load(output / "last.pth", weights_only=False, map_location="cpu")
            self.assertEqual(best["format"], fmt)
            self.assertEqual(last["format"], fmt)
            self.assertEqual(last["ema_updates"], 2)
            self.assertEqual("region_interaction.gamma" in last["model_state"], enabled)
            self.assertEqual(json.loads((output / "run_info.json").read_text())["model"], fmt)
            if enabled:
                self.assertEqual(best["architecture"]["mil_pooling"], [pooling])
                self.assertEqual(best["architecture"]["region_interaction"]["rank"], 16)
            else:
                old = load_old(best).eval()
                h = torch.randn(5, 256)
                self.assert_tensor_equal(old.forward_embeddings(h), model_from_checkpoint(best).eval().forward_embeddings(h))
            for key, tensor in best["model_state"].items():
                self.assert_tensor_equal(tensor, last["ema_model_state"][key])
            self.assertEqual(model_from_checkpoint(last).architecture(), last["architecture"])
            prediction_path = root / "validation.csv"
            eval_args = ["--checkpoint", str(output / "best.pth"), "--data-root", str(root), "--device", "cpu"]
            with patch.object(eval_e2e, "load_fold_datasets", return_value=(dataset, dataset)), \
                 patch.object(eval_e2e, "provenance", return_value=provenance), \
                 patch.object(engine, "load_images", return_value=regions), redirect_stdout(io.StringIO()):
                eval_e2e_gcsam_asrd.main([*eval_args, "--predictions-csv", str(prediction_path)])
            self.assertEqual(prediction_path.read_bytes(), (output / "best_val_predictions.csv").read_bytes())

            if enabled:
                reference = root / "reference.csv"
                reference.write_text("test_0,Normal\ntest_1,Tumor\n", encoding="utf-8")
                manifest = root / "manifests" / "patch_manifest.csv"
                manifest.parent.mkdir()
                manifest.write_text("slide_id\n", encoding="utf-8")
                report = root / "report.json"
                report.write_text("{}", encoding="utf-8")
                test_regions = {s["slide_id"]: [(i, i) for i in s["lr_paths"]] for s in samples}
                official = root / "official"
                with patch.object(eval_native512_reference, "load_native_regions", return_value=(test_regions, report)), \
                     patch.object(engine, "load_images", return_value=regions), redirect_stdout(io.StringIO()):
                    eval_e2e_gcsam_asrd.main(["official-test", *eval_args, "--reference-csv", str(reference),
                                             "--output", str(official)])
                self.assertEqual((official / "predictions.csv").read_bytes(), prediction_path.read_bytes())

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 required")
    def test_cuda_bf16_gcsam_gradients_and_ema(self):
        device = torch.device("cuda:0")
        for pooling in ("mean", "attention"):
            with self.subTest(pooling=pooling):
                config = self.config(pooling, True)
                config["precision"] = "bf16"
                torch.manual_seed(49)
                model = build_model(shared_encoder=config["shared_encoder"],
                                    classification=config["classification"], sr_dropout=0).to(device)
                interaction = model.region_interaction
                optimizer = trainer.build_optimizer(model, config)
                groups = {id(p): group for group in optimizer.param_groups for p in group["params"]}
                self.assertEqual(len(groups), sum(len(group["params"]) for group in optimizer.param_groups))
                for p in interaction.parameters():
                    self.assertTrue(groups[id(p)]["name"].startswith("head_"))
                    self.assertEqual(groups[id(p)]["lr"], config["classification"]["head_lr"])
                ema = ModelEMA(model, config["ema_decay"])
                region_data = torch.rand(5, 3, 48, 48)
                sample = {"slide_id": "synthetic", "label": 1, "n_regions": 5,
                          "lr_paths": list(range(5)), "data_mode": "native512"}
                initial = interaction.query.weight.detach().clone()
                gamma_gradients, query_gradients, forwards = [], [], []
                handles = [
                    interaction.gamma.register_hook(lambda grad: gamma_gradients.append(grad.detach().clone())),
                    interaction.query.weight.register_hook(lambda grad: query_gradients.append(grad.detach().norm().item())),
                    interaction.register_forward_hook(lambda *args: forwards.append(1)),
                ]
                try:
                    with patch.object(engine, "load_images", return_value=region_data):
                        for _ in range(2):
                            stats = train_wsi(model, sample, optimizer, device, config)
                            self.assertTrue(torch.tensor(stats["grad_norm_second"]).isfinite())
                            ema.update(model)
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(len(forwards), 4)
                self.assertEqual(len(gamma_gradients), 4)
                self.assertGreater(gamma_gradients[0].abs().item(), 0)
                self.assertEqual(query_gradients[0], 0)
                self.assertGreater(query_gradients[2], 0)
                self.assertNotEqual(interaction.gamma.item(), 0)
                self.assertFalse(torch.equal(interaction.query.weight, initial))
                for p in interaction.parameters():
                    self.assertTrue(p.grad.isfinite().all())
                    self.assertIn(p, optimizer.base_optimizer.state)
                self.assertEqual(ema.updates, 2)
                self.assertNotEqual(ema.module.region_interaction.gamma.item(), 0)
