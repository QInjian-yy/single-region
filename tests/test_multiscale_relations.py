"""CPU checks for opt-in relations, original MIL/AEM, SSD paths and checkpoints."""
import ast
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "E2E_MultiScale_OptIn"))

import numpy as np
from PIL import Image
import torch
import yaml

from augmentation import AugmentedPath, TransformParams
from data_modes import data_spec
from E2E_MultiScale_OptIn.multiscale_data import PixelCache, cache_key, _fingerprint, open_cache
from E2E_MultiScale_OptIn.multiscale_model import MultiScaleSRMIL, FORMAT as BASE_FORMAT
from E2E_MultiScale_OptIn.multiscale_relation_model import FORMAT, build_model, model_from_checkpoint
from E2E_MultiScale_OptIn.multiscale_relation_train import baseline, build_optimizer, train_fold
from E2E_MultiScale_OptIn.relations import build_relation, normalize_relation_config, register_relation
from E2E_MultiScale_OptIn.ssd_support.relation_paths import ssd_lr_path, remap_lr_samples

CONFIG = ROOT / "E2E_MultiScale_OptIn/configs/camil_s123_lrhr_aem02_wholebag_ssd_topk.yaml"


def experiment_fixture_config():
    """Self-contained test recipe; experiment YAML remains user-owned."""
    return {   'data_mode': 'native512',
        'data_manifest': 'Manifests/patch_manifest.csv',
        'split_scheme': '5fold',
        'shared_encoder': {   'backbone': 'resnet18',
                              'pretrained': True,
                              'trainable': 'layer4',
                              'freeze_bn': True,
                              'encoder_lr': 1e-05},
        'classification': {   'pooling': ['attention'],
                              'embedding_dim': 256,
                              'attention_hidden_dim': 128,
                              'topk': 5,
                              'region_dropout': 0.0,
                              'attention_dropout': 0.0,
                              'projection_dropout': 0.0,
                              'classifier_dropout': 0.0,
                              'head_lr': 0.0002,
                              'encoder_lr': 1e-05,
                              'vib': {'enabled': False, 'beta': 0.0001},
                              'aem': {'enabled': True, 'weight': 0.02, 'anneal_epochs': 50}},
        'metasr': {   'scale': 16,
                      'rgb_range': 1,
                      'channels': 64,
                      'residual_scale': 0.1,
                      'dropout': 0.0,
                      'hr_tile': 256},
        'lambda_sr': 0.0,
        'sr_train_crop': 128,
        'sr_lr': 0.0001,
        'training': {   'use_region_microbatch': False,
                        'region_microbatch_size': 1,
                        'max_train_regions': None,
                        'class_weight': 'balanced',
                        'sr_stop_after_epochs': None},
        'optimizer': 'adamw',
        'precision': 'bf16',
        'lr': 0.0002,
        'weight_decay': 0.01,
        'label_smoothing': 0.1,
        'gradient_clip_norm': 1.0,
        'epochs': 300,
        'min_epochs': 5,
        'early_stopping_patience': 12,
        'early_stopping_min_delta': 0.001,
        'checkpoint_metric': 'val_auc',
        'scheduler': 'plateau',
        'scheduler_patience': 4,
        'scheduler_factor': 0.5,
        'min_lr': 1e-06,
        'ema_decay': 0.999,
        'seed': 1,
        'augmentation': {   'enabled': True,
                            'horizontal_flip': 0.5,
                            'vertical_flip': 0.5,
                            'rotation90': True,
                            'brightness': 0.15,
                            'contrast': 0.15,
                            'saturation': 0.1,
                            'hue': 0.01},
        'class_weight_values': [0.8170212765957447, 1.1829787234042555],
        'multiscale': {'hr_patch_size': 256, 'patches_per_region': 2, 'sr_regions_per_wsi': 2},
        'region_relation': {   'enabled': True,
                               'name': 'semantic_topk',
                               'k': 3,
                               'temperature': 1.0,
                               'gate_init': 0.01}}


def small_config():
    cfg = experiment_fixture_config()
    cfg["shared_encoder"]["pretrained"] = False
    cfg["classification"].update(embedding_dim=32, attention_hidden_dim=16)
    cfg["metasr"]["channels"] = 16
    cfg["multiscale"]["hr_patch_size"] = 128
    cfg["precision"] = "fp32"
    return cfg


def pixel_fixture(directory, *, n=2, count=1, size=128, k=2):
    """Real PNG decoding, synthetic pixels, and the actual PixelCache reader."""
    directory = Path(directory)
    lr_dir = directory / "source_train/images_512"
    patches = directory / "cache/patches"
    lr_dir.mkdir(parents=True)
    patches.mkdir(parents=True)
    rng = np.random.default_rng(123)
    regions, samples = {}, {}
    for slide in range(count):
        sample = dict(slide_id=f"synthetic_{slide}", label=slide % 2, source="official_train",
                      n_regions=n, data_mode="native512", lr_paths=[], hr_paths=[])
        for region in range(n):
            filename = f"{slide}_{region}.png"
            lr = lr_dir / filename
            hr = directory / "source_train/images_8192" / filename
            Image.fromarray(rng.integers(30, 230, (512, 512, 3), dtype=np.uint8)).save(lr)
            sample["lr_paths"].append(lr)
            sample["hr_paths"].append(hr)
            entries = []
            for j in range(k):
                tile = f"{slide}_{region}_{j}.png"
                Image.fromarray(rng.integers(30, 230, (size, size, 3), dtype=np.uint8)).save(patches / tile)
                entries.append(dict(filename=tile, box=[j * size, j * size, size, size]))
            regions[cache_key(hr)] = entries
        samples[sample["slide_id"]] = sample
    index = dict(k=k, patch_size=size, scale=16, regions=regions,
                 fingerprint=_fingerprint(samples, size, 16, k))
    (directory / "cache/index.json").write_text(json.dumps(index), encoding="utf-8")
    return samples, PixelCache(directory / "cache", index)


class RelationModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_identity_exact_no_parameters_or_rng(self):
        features = torch.randn(5, 256, requires_grad=True)
        rng = torch.get_rng_state().clone()
        module = build_relation(None)
        self.assertIs(module(features, context={"slide_id": "a"}), features)
        self.assertEqual(list(module.parameters()), [])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        module(features).sum().backward()
        self.assertTrue(torch.equal(features.grad, torch.ones_like(features)))

    def test_shapes_neighbors_and_backward(self):
        for n in (1, 2, 3, 5, 20):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(n=n, dtype=dtype):
                    torch.manual_seed(n)
                    module = build_relation(dict(enabled=True, name="semantic_topk"))
                    features = torch.randn(n, 256, dtype=dtype, requires_grad=True)
                    output = module(features)
                    indices, weights = module.neighbor_weights(features)
                    self.assertEqual(output.shape, features.shape)
                    self.assertEqual(output.dtype, dtype)
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertEqual(indices.shape, (n, min(3, n - 1)))
                    self.assertEqual(weights.dtype, torch.float32)
                    if n == 1:
                        self.assertIs(output, features)
                    else:
                        self.assertFalse((indices == torch.arange(n)[:, None]).any())
                        torch.testing.assert_close(weights.sum(-1), torch.ones(n))
                    output.float().square().sum().backward()
                    self.assertTrue(torch.isfinite(features.grad).all())
                    if n > 1:
                        self.assertIsNotNone(module.gate_logit.grad)
                        self.assertTrue(torch.isfinite(module.gate_logit.grad))
                        self.assertGreater(module.gate_logit.grad.abs().item(), 0)

    def test_fp32_similarity_inside_bf16_autocast(self):
        module = build_relation(dict(enabled=True, name="semantic_topk"))
        features = torch.randn(5, 256)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, weights = module.neighbor_weights(features)
            output = module(features)
        self.assertEqual(weights.dtype, torch.float32)
        self.assertEqual(output.dtype, features.dtype)

    def test_cosine_topk_matches_explicit_reference(self):
        x = torch.tensor([[1., 0.], [.9, .1], [-1., 0.], [0., 1.]])
        module = build_relation(dict(enabled=True, name="semantic_topk", k=1), dim=2)
        indices, weights = module.neighbor_weights(x)
        self.assertEqual(indices[:, 0].tolist(), [1, 0, 3, 1])
        expected = x + .01 * x[indices[:, 0]]
        torch.testing.assert_close(module(x), expected)
        self.assertTrue(torch.equal(weights, torch.ones_like(weights)))

    def test_actual_cross_region_information(self):
        module = build_relation(dict(enabled=True, name="semantic_topk", k=1), dim=2)
        x = torch.tensor([[1., .2], [.3, 1.]], requires_grad=True)
        original = module(x)
        changed = x.detach().clone()
        changed[1, 1] += 2
        self.assertFalse(torch.equal(original[0], module(changed)[0]))
        gradient = torch.autograd.grad(original[0].sum(), x)[0]
        self.assertGreater(gradient[1].abs().sum().item(), 0)

    def test_permutation_equivariance_without_ties(self):
        torch.manual_seed(24)
        x = torch.randn(20, 256)
        module = build_relation(dict(enabled=True, name="semantic_topk"))
        order = torch.randperm(20)
        torch.testing.assert_close(module(x[order]), module(x)[order], rtol=1e-6, atol=1e-7)

    def test_zero_features_and_bag_independence(self):
        module = build_relation(dict(enabled=True, name="semantic_topk"))
        self.assertTrue(torch.equal(module(torch.zeros(5, 256)), torch.zeros(5, 256)))
        first, second = torch.randn(3, 256), torch.randn(5, 256)
        before = module(first).clone()
        module(second, context={"slide_id": "another_wsi"})
        self.assertTrue(torch.equal(before, module(first)))

    def test_configuration_validation(self):
        for config in (dict(enabled=1), dict(enabled=True, name="missing"),
                       dict(enabled=True, name="semantic_topk", k=0),
                       dict(enabled=True, name="semantic_topk", k=True),
                       dict(enabled=True, name="semantic_topk", temperature=0),
                       dict(enabled=True, name="semantic_topk", temperature=float("nan")),
                       dict(enabled=True, name="semantic_topk", gate_init=0),
                       dict(enabled=True, name="semantic_topk", gate_init=1),
                       dict(enabled=True, name="semantic_topk", unused=True)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                build_relation(config)
        for shape in ((0, 256), (2, 255), (2, 3, 256)):
            with self.assertRaises(ValueError):
                build_relation(dict(enabled=True, name="semantic_topk"))(torch.zeros(shape))

    def test_registry_rejects_duplicate_registration(self):
        with self.assertRaises(ValueError):
            register_relation("identity")(torch.nn.Identity)

    def test_factory_accepts_full_yaml_or_relation_block(self):
        settings = dict(enabled=True, name="semantic_topk", k=2)
        x = torch.randn(5, 256)
        self.assertTrue(torch.equal(build_relation(settings)(x),
                                    build_relation(dict(region_relation=settings))(x)))

    def test_another_registered_module_and_context_need_no_trainer_changes(self):
        @register_relation("unit_test_context")
        class ContextRelation(torch.nn.Module):
            @staticmethod
            def normalize_config(settings):
                return {}

            def __init__(self, dim):
                super().__init__()
                self.offset = torch.nn.Parameter(torch.zeros(dim))

            def forward(self, features, context=None):
                return features + self.offset + context["shift"]

        cfg = small_config()
        cfg["region_relation"] = dict(enabled=True, name="unit_test_context")
        model = build_model(cfg).eval()
        features = torch.randn(3, 32)
        context = dict(shift=torch.randn(3, 32))
        expected = model.multiscale.classify(features + context["shift"])
        actual = model.classify(features, context=context)
        for x, y in zip(expected, actual):
            self.assertTrue(torch.equal(x, y))
        optimizer = build_optimizer(model, cfg)
        self.assertTrue(any(model.region_relation.offset is p
                            for group in optimizer.param_groups for p in group["params"]))


class MultiScaleIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_changes_only_region_relation(self):
        if not CONFIG.is_file() or not CONFIG.with_name("camil_s123_lrhr_aem02_wholebag_ssd.yaml").is_file():
            self.skipTest("Optional integration check: user-owned experiment YAML not present")
        base = yaml.safe_load(CONFIG.with_name("camil_s123_lrhr_aem02_wholebag_ssd.yaml").read_text())
        changed = yaml.safe_load(CONFIG.read_text())
        changed.pop("region_relation")
        self.assertEqual(base, changed)

    def test_disabled_exact_initialization_output_and_update(self):
        cfg = small_config()
        cfg.pop("region_relation")
        torch.manual_seed(3)
        original = MultiScaleSRMIL(cfg)
        reference_rng = torch.get_rng_state().clone()
        torch.manual_seed(3)
        restored = build_model(dict(cfg, region_relation=dict(enabled=False)))
        self.assertTrue(torch.equal(reference_rng, torch.get_rng_state()))
        self.assertEqual(original.state_dict().keys(), restored.state_dict().keys())
        lr = torch.randn(3, 3, 64, 64)
        hr = torch.randn(3, 2, 3, 128, 128)
        states, outputs = [], []
        for model in (original, restored):
            model.train()
            optimizer = build_optimizer(model, cfg)
            features, _ = model.encode_multiscale(lr, hr)
            logits, attention = model.classify(features)
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
            loss += .02 * baseline.engine_aem.negative_attention_entropy(attention)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            outputs.append(logits.detach())
            states.append({k: v.clone() for k, v in model.state_dict().items()})
        self.assertTrue(torch.equal(*outputs))
        for key in states[0]:
            self.assertTrue(torch.equal(states[0][key], states[1][key]), key)

    def test_identity_wrapper_preserves_logits_and_attention(self):
        cfg = small_config()
        cfg["region_relation"] = dict(enabled=True, name="identity")
        torch.manual_seed(7)
        original = MultiScaleSRMIL(cfg).eval()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(7)
        wrapped = build_model(cfg).eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        x = torch.randn(5, 32)
        for expected, actual in zip(original.classify(x), wrapped.classify(x)):
            self.assertTrue(torch.equal(expected, actual))

    def test_shared_lr_hr_fusion_and_relation_gradients(self):
        model = build_model(small_config()).train()
        lr = torch.randn(3, 3, 64, 64, requires_grad=True)
        hr = torch.randn(3, 2, 3, 128, 128, requires_grad=True)
        features, _ = model.encode_multiscale(lr, hr)
        logits, attention = model.classify(features)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
        loss += .02 * baseline.engine_aem.negative_attention_entropy(attention)
        loss.backward()
        for tensor in (lr, hr):
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(tensor.grad).all())
        for parameter in (model.base.encoder.layer4[0].conv1.weight, model.fusion[1].weight,
                          model.multiscale.patch_score.weight, model.region_relation.gate_logit):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_optimizer_and_ema_cover_relation(self):
        cfg = small_config()
        model = build_model(cfg)
        optimizer = build_optimizer(model, cfg)
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})
        gate = model.region_relation.gate_logit
        group = next(g for g in optimizer.param_groups if any(p is gate for p in g["params"]))
        self.assertEqual(group["lr"], cfg["classification"]["head_lr"])
        self.assertEqual(group["weight_decay"], 0)
        ema = baseline.camil.shared.ModelEMA(model, .999)
        old = ema.module.region_relation.gate_logit.clone()
        with torch.no_grad():
            gate.add_(.2)
        ema.update(model)
        torch.testing.assert_close(ema.module.region_relation.gate_logit,
                                   old.lerp(gate.detach(), 1 - 2/11), rtol=0, atol=0)

    def test_sr_path_remains_differentiable(self):
        cfg = small_config()
        cfg["lambda_sr"] = .05
        model = build_model(cfg).train()
        features, sr_features = model.encode_multiscale(
            torch.rand(2, 3, 64, 64), torch.rand(2, 2, 3, 128, 128), need_sr=True)
        prediction = model.base.sr.decode_crop(sr_features[:1], (32, 32, 64, 64))
        logits, _ = model.classify(features)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1])) + .05 * prediction.abs().mean()
        loss.backward()
        self.assertGreater(model.base.sr.tail.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.region_relation.gate_logit.grad.abs().item(), 0)

    def test_bounded_smoke_and_checkpoint(self):
        from E2E_MultiScale_OptIn.relation_smoke import smoke_steps
        cfg = small_config()
        with tempfile.TemporaryDirectory() as tmp:
            samples, cache = pixel_fixture(tmp, n=3)
            result = smoke_steps(cfg, list(samples.values()), cache, torch.device("cpu"), steps=2)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["ema_updates"], 2)
        self.assertTrue(result["strict_reload_prediction_equal"])
        self.assertEqual(result["relation_parameters"], 1)
        self.assertTrue(all(step["augmentation_applied"] for step in result["steps"]))

    def test_checkpoint_legacy_relation_and_strict_errors(self):
        for mode in (None, dict(enabled=False), dict(enabled=True, name="identity"),
                     dict(enabled=True, name="semantic_topk")):
            with self.subTest(mode=mode):
                cfg = small_config()
                if mode is None:
                    cfg.pop("region_relation")
                else:
                    cfg["region_relation"] = mode
                model = build_model(cfg).eval()
                saved = dict(format=model.architecture()["type"], architecture=model.architecture(),
                             config=cfg, data_spec=data_spec(cfg).metadata(), model_state=model.state_dict())
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "check.pth"
                    torch.save(saved, path)
                    restored = model_from_checkpoint(torch.load(path, weights_only=True)).eval()
                x = torch.randn(5, 32)
                for expected, actual in zip(model.classify(x), restored.classify(x)):
                    self.assertTrue(torch.equal(expected, actual))
                bad = dict(saved, model_state={**saved["model_state"], "unexpected.weight": torch.ones(1)})
                with self.assertRaises(RuntimeError):
                    model_from_checkpoint(bad)
                bad = dict(saved, format=BASE_FORMAT if saved["format"] == FORMAT else FORMAT)
                with self.assertRaises(ValueError):
                    model_from_checkpoint(bad)
                if saved["format"] == FORMAT and mode["name"] == "semantic_topk":
                    missing = dict(saved["model_state"])
                    missing.pop("region_relation.gate_logit")
                    with self.assertRaises(RuntimeError):
                        model_from_checkpoint(dict(saved, model_state=missing))

    def test_full_bag_once_after_all_microbatches_and_original_aem(self):
        cfg = small_config()
        cfg["training"].update(use_region_microbatch=True, region_microbatch_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            samples, cache = pixel_fixture(tmp, n=5)
            sample = next(iter(samples.values()))
            model = build_model(cfg).train()
            relation_inputs, attentions, contexts = [], [], []
            h1 = model.region_relation.register_forward_hook(
                lambda m, i, o: relation_inputs.append(tuple(i[0].shape)))
            h2 = model.base.classifier_head.mil.register_forward_hook(lambda m, i, o: attentions.append(o[1]))
            h3 = model.region_relation.register_forward_hook(
                lambda m, i, kwargs, o: contexts.append(kwargs["context"]), with_kwargs=True)
            total, cls, sr, negative_entropy, logits, weight = baseline.wsi_forward(
                model, sample, cache, torch.device("cpu"), cfg, epoch=1)
            h1.remove()
            h2.remove()
            h3.remove()
            self.assertEqual(relation_inputs, [(5, 32)])
            self.assertEqual(contexts, [dict(slide_id=sample["slide_id"], n_regions=5)])
            self.assertNotIn("label", contexts[0])
            self.assertEqual(len(attentions), 1)
            torch.testing.assert_close(negative_entropy,
                                       baseline.engine_aem.negative_attention_entropy(attentions[0]), rtol=0, atol=0)
            torch.testing.assert_close(total, cls + weight * negative_entropy, rtol=0, atol=0)
            self.assertEqual(sr.item(), 0)
            self.assertEqual(tuple(logits.shape), (1, 2))

    def test_one_epoch_train_validation_test_outputs(self):
        cfg = small_config()
        cfg.update(epochs=1, min_epochs=1)
        with tempfile.TemporaryDirectory() as tmp:
            samples, cache = pixel_fixture(Path(tmp) / "pixels", count=6)
            names = list(samples)
            protocol = SimpleNamespace(
                folds={0: dict(train=names[:2], val=names[2:4], test=names[4:])},
                patient_grouping="synthetic_only")
            result = train_fold(cfg, protocol, samples, {}, cache, 0,
                                Path(tmp) / "run", torch.device("cpu"))
            self.assertTrue(np.isfinite(result["auc"]))
            for name in ("config.yaml", "history.csv", "run_info.json", "data_provenance.json",
                         "best.pth", "best_val_predictions.csv", "test_metrics.json", "test_predictions.csv"):
                self.assertTrue((Path(tmp) / "run" / name).is_file(), name)
            saved = torch.load(Path(tmp) / "run/best.pth", weights_only=True)
            self.assertEqual(saved["format"], FORMAT)
            self.assertIn("region_relation.gate_logit", saved["model_state"])
            model_from_checkpoint(saved)


class SSDCompatibility(unittest.TestCase):
    def test_exact_existing_launcher_mapping_and_augmented_pixels(self):
        path = ROOT / "E2E_MultiScale_OptIn/ssd_support/ssd_lr_launcher.py"
        if not path.is_file():
            self.skipTest("Optional integration check: original server SSD launcher not present")
        source = ast.parse(path.read_text())
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "ssd_load_images")
        with tempfile.TemporaryDirectory() as tmp:
            namespace = dict(Path=Path, AugmentedPath=AugmentedPath,
                             ssd_root=Path(tmp) / "ssd", original_loader=baseline.engine.load_images)
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
            params = TransformParams(True, False, 1)
            for source_name, suffix in (("official_train", "train"), ("official_test", "test")):
                src = Path(tmp) / f"dataset_{suffix}/images_512/sub/tile.png"
                dst = Path(tmp) / f"ssd/images_512_{suffix}/sub/tile.png"
                dst.parent.mkdir(parents=True)
                Image.fromarray(np.random.default_rng(7).integers(0, 255, (512, 512, 3), dtype=np.uint8)).save(dst)
                for item in (src, AugmentedPath(src, params)):
                    mapped = ssd_lr_path(item, source_name, namespace["ssd_root"])
                    expected = namespace["ssd_load_images"]([item], 512)
                    actual = baseline.engine.load_images([mapped], 512)
                    self.assertTrue(torch.equal(expected, actual))

    def test_hr_cache_original_fingerprint_survives_ssd_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples, _ = pixel_fixture(root, size=128)
            for sample in samples.values():
                for path in sample["lr_paths"]:
                    dst = root / "ssd/images_512_train" / path.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(path.read_bytes())
            remapped, record = remap_lr_samples(samples, root / "ssd")
            original_cache = open_cache(samples, root / "cache", patch_size=128, k=2)
            self.assertEqual(record["regions"], 2)
            for key in samples:
                self.assertEqual(samples[key]["hr_paths"], remapped[key]["hr_paths"])
                original_cache.read(remapped[key]["hr_paths"])
            with self.assertRaises(ValueError):
                open_cache(remapped, root / "cache", patch_size=128, k=2)

    def test_missing_ssd_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileNotFoundError):
            ssd_lr_path(Path(tmp) / "source_train/images_512/missing.png", "official_train", tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
