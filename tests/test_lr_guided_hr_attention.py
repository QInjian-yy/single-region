"""Checks for LR-guided aggregation and the separate K=4 launch path."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from E2E_MultiScale_OptIn import lr_guided_hr_train as guided
from E2E_MultiScale_OptIn.lr_guided_hr_model import FORMAT, LRGuidedHRAttention, LRGuidedHRMultiScaleMIL
from E2E_MultiScale_OptIn.multiscale_model import MultiScaleSRMIL
from test_multiscale_relations import experiment_fixture_config, pixel_fixture
from wsi_data import Camelyon16WSI


CONFIG = Path(__file__).resolve().parents[1] / (
    "E2E_MultiScale_OptIn/configs/camil_s123_lrhr_aem02_wholebag_ssd_hr4_lr_guided.yaml")
BASELINE = CONFIG.with_name("camil_s123_lrhr_aem02_wholebag_ssd_hr4.yaml")


def probe_config():
    config = experiment_fixture_config()
    config.pop("region_relation")
    config["shared_encoder"]["pretrained"] = False
    config["multiscale"]["patches_per_region"] = 4
    config["hr_aggregation"] = guided.HR_AGGREGATION
    return config


class LRGuidedHRTests(unittest.TestCase):
    def test_scaled_dot_product_and_region_local_gradients(self):
        torch.manual_seed(23)
        attention = LRGuidedHRAttention(8)
        lr = torch.randn(2, 8, requires_grad=True)
        hr = torch.randn(2, 4, 8, requires_grad=True)
        result = attention(lr, hr)
        expected = torch.nn.functional.scaled_dot_product_attention(
            attention.query(lr)[:, None, :], attention.key(hr), hr).squeeze(1)
        torch.testing.assert_close(result, expected)
        self.assertEqual(result.shape, (2, 8))
        changed_lr = lr.detach().clone()
        changed_lr[0].neg_()
        conditioned = attention(changed_lr, hr)
        self.assertFalse(torch.allclose(result[0], conditioned[0]))
        torch.testing.assert_close(result[1], conditioned[1], rtol=0, atol=0)
        result.square().sum().backward()
        for gradient in (lr.grad, hr.grad, *(p.grad for p in attention.parameters())):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.norm().item(), 0)

    def test_config_changes_only_aggregation(self):
        if not CONFIG.is_file() or not BASELINE.is_file():
            self.skipTest("User-owned experiment YAML not present")
        raw = yaml.safe_load(CONFIG.read_bytes())
        self.assertEqual(raw.pop("hr_aggregation"), guided.HR_AGGREGATION)
        self.assertEqual(raw, yaml.safe_load(BASELINE.read_bytes()))
        guided.load_config(CONFIG)

    def test_only_aggregation_changes_and_optimizer_owns_it(self):
        config = probe_config()
        torch.manual_seed(1)
        original = MultiScaleSRMIL(config)
        original_rng = torch.random.get_rng_state()
        torch.manual_seed(1)
        model = LRGuidedHRMultiScaleMIL(config)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), original_rng))
        for name, value in original.state_dict().items():
            if not name.startswith("patch_score."):
                self.assertTrue(torch.equal(model.state_dict()[name], value), name)
        self.assertEqual(set(model.patch_score.state_dict()), {"query.weight", "key.weight"})
        self.assertEqual(sum(p.numel() for p in model.patch_score.parameters()), 131072)
        optimizer = guided.baseline.build_optimizer(model, config)
        learning_rates = {id(p): group["lr"] for group in optimizer.param_groups for p in group["params"]}
        for parameter in model.patch_score.parameters():
            self.assertEqual(learning_rates[id(parameter)], config["classification"]["head_lr"])

    def test_single_fold_entry_writes_and_tests_guided_checkpoint(self):
        config = probe_config()
        config.update(epochs=1, min_epochs=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples, _ = pixel_fixture(root, n=1, count=6, size=256, k=4)
            names = list(samples)
            protocol = SimpleNamespace(sources={"official_train": {}}, patient_grouping="synthetic",
                                       folds={0: dict(train=names[:2], val=names[2:4], test=names[4:])})
            source = dict(official_train_manifest={"data_root": str(root / "source_train")})
            ssd = root / "ssd/images_512_train"
            ssd.mkdir(parents=True)
            for sample in samples.values():
                for path in sample["lr_paths"]:
                    shutil.copyfile(path, ssd / path.name)
            cfg = root / "config.yaml"
            cfg.write_text(yaml.safe_dump(config), encoding="utf-8")
            output = root / "out"
            with patch.object(guided.camil_data, "read_camil_protocol", return_value=protocol), \
                    patch.object(guided.camil_data, "load_camil_samples", return_value=(samples, source)):
                guided.main(["--config", str(cfg), "--protocol-dir", str(root/"protocol"),
                             "--cache-dir", str(root/"cache"), "--ssd-root", str(ssd.parent),
                             "--device", "cpu", "--fold", "0", "--output", str(output)])
            result = output / "fold_0"
            checkpoint = torch.load(result/"best.pth", weights_only=True)
            self.assertEqual(checkpoint["format"], FORMAT)
            self.assertEqual(checkpoint["architecture"]["hr_patches_per_region"], 4)
            self.assertEqual(checkpoint["data_provenance"]["hr_cache"]["k"], 4)
            self.assertEqual(checkpoint["ema_updates"], 2)
            self.assertIn("patch_score.query.weight", checkpoint["model_state"])
            scores = json.loads((result/"test_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(scores["checkpoint_epoch"], 1)
            self.assertEqual(scores["slide_count"], 2)


def gpu_smoke():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU smoke requires CUDA BF16 support")
    config = probe_config()
    with tempfile.TemporaryDirectory(prefix="lr_guided_hr_gpu_") as tmp:
        samples, _ = pixel_fixture(tmp, n=2, count=2, size=256, k=4)
        cache = guided.baseline.open_cache(samples, Path(tmp)/"cache", patch_size=256, scale=16, k=4)
        train = Camelyon16WSI(list(samples.values()), augmentation=config.get("augmentation"))
        guided.baseline.camil.configure_class_weights(config, train)
        result = guided.smoke_steps(config, train, cache, device, steps=2)
    assert result["status"] == "passed" and len(result["steps"]) == result["ema_updates"] == 2
    assert result["strict_reload_prediction_equal"]
    assert all(record["metrics"]["aem_weight"] == 0.02 for record in result["steps"])
    print(json.dumps(dict(result, gpu=torch.cuda.get_device_name(device), data="synthetic"),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(2)
    if sys.argv[1:] == ["--gpu-smoke"]:
        gpu_smoke()
    else:
        unittest.main(verbosity=2)
