"""CPU module and existing multiscale integration checks for Global MHSA."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import torch
import yaml

from test_multiscale_relations import experiment_fixture_config, pixel_fixture
from data_modes import data_spec
from E2E_MultiScale_OptIn.multiscale_model import MultiScaleSRMIL
from E2E_MultiScale_OptIn.multiscale_relation_model import build_model, model_from_checkpoint
from E2E_MultiScale_OptIn.multiscale_relation_train import baseline, build_optimizer, load_config
from E2E_MultiScale_OptIn.relation_smoke import smoke_steps
from E2E_MultiScale_OptIn.relations import GlobalMHSARelation, build_relation

SETTINGS = dict(enabled=True, name="global_mhsa", num_heads=4,
                attention_dropout=0.1, gate_init=0.1)
CONFIG = ROOT / "E2E_MultiScale_OptIn/configs/camil_s123_lrhr_aem02_wholebag_ssd_mhsa.yaml"


def small_config():
    config = experiment_fixture_config()
    config["region_relation"] = dict(SETTINGS)
    config["shared_encoder"]["pretrained"] = False
    config["classification"].update(embedding_dim=32, attention_hidden_dim=16)
    config["metasr"]["channels"] = 16
    config["multiscale"]["hr_patch_size"] = 128
    config["precision"] = "fp32"
    return config


class GlobalMHSAModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_requested_architecture_and_parameter_count(self):
        module = build_relation(SETTINGS)
        self.assertIsInstance(module, GlobalMHSARelation)
        self.assertEqual(sum(isinstance(m, torch.nn.MultiheadAttention) for m in module.modules()), 1)
        self.assertIsInstance(module.norm, torch.nn.LayerNorm)
        self.assertTrue(module.attention.batch_first)
        self.assertEqual(module.attention.num_heads, 4)
        self.assertEqual(module.attention.dropout, 0.1)
        self.assertAlmostEqual(module.gate_logit.sigmoid().item(), .1, places=7)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 263681)

    def test_n1_and_larger_forward_backward(self):
        for n in (1, 2, 3, 5, 20):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(n=n, dtype=dtype):
                    torch.manual_seed(101 + n)
                    module = build_relation(SETTINGS)
                    features = torch.randn(n, 256, dtype=dtype, requires_grad=True)
                    output = module(features)
                    self.assertEqual(tuple(output.shape), (n, 256))
                    self.assertEqual(output.dtype, dtype)
                    self.assertTrue(torch.isfinite(output).all())
                    output.float().square().mean().backward()
                    self.assertIsNotNone(features.grad)
                    self.assertTrue(torch.isfinite(features.grad).all())
                    for name, parameter in module.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_cpu_bf16_autocast(self):
        module = build_relation(SETTINGS).train()
        features = torch.randn(5, 256, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = module(features)
            loss = output.square().mean()
        loss.backward()
        self.assertEqual(output.dtype, features.dtype)
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreater(module.gate_logit.grad.abs().item(), 0)

    def test_nonzero_gradients_and_optimizer_step(self):
        torch.manual_seed(13)
        module = build_relation(SETTINGS)
        optimizer = torch.optim.AdamW(module.parameters(), lr=2e-4)
        before = {name: value.detach().clone() for name, value in module.named_parameters()}
        features = torch.randn(5, 256, requires_grad=True)
        output = module(features)
        ((output - torch.randn_like(output)).square().mean()).backward()
        for name, parameter in module.named_parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        optimizer.step()
        self.assertFalse(torch.equal(before["gate_logit"], module.gate_logit))
        self.assertFalse(torch.equal(before["attention.in_proj_weight"], module.attention.in_proj_weight))

    def test_pre_norm_and_residual_match_explicit_attention(self):
        module = build_relation(SETTINGS).eval()
        features = torch.randn(5, 256)
        normalized = module.norm(features).unsqueeze(0)
        message, _ = module.attention(normalized, normalized, normalized, need_weights=False)
        expected = features + module.gate_logit.sigmoid() * message.squeeze(0)
        torch.testing.assert_close(module(features), expected, rtol=0, atol=0)

    def test_global_messages_no_labels_no_cross_bag_state(self):
        module = build_relation(SETTINGS).eval()
        features = torch.randn(5, 256, requires_grad=True)
        output = module(features, context={"label": 0})
        self.assertTrue(torch.equal(output, module(features, context={"label": 1})))
        gradient = torch.autograd.grad(output[0, 0], features)[0]
        for other in range(1, 5):
            self.assertGreater(gradient[other].abs().sum().item(), 0)
        module(torch.randn(3, 256), context={"slide_id": "another_wsi"})
        self.assertTrue(torch.equal(output, module(features)))
        with self.assertRaises(ValueError):
            module(torch.randn(2, 5, 256))

    def test_no_position_encoding_permutation_equivariance(self):
        module = build_relation(SETTINGS).eval()
        features = torch.randn(9, 256)
        permutation = torch.randperm(9)
        torch.testing.assert_close(module(features[permutation]), module(features)[permutation],
                                   rtol=1e-5, atol=1e-6)

    def test_initialization_preserves_original_rng_and_weights(self):
        config = small_config()
        torch.manual_seed(31)
        original = MultiScaleSRMIL(config)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(31)
        wrapped = build_model(config)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, wrapped.multiscale.state_dict()[name]), name)

    def test_state_dict_restores_exact_evaluation(self):
        module = build_relation(SETTINGS).eval()
        restored = build_relation(SETTINGS).eval()
        restored.load_state_dict(module.state_dict(), strict=True)
        features = torch.randn(5, 256)
        self.assertTrue(torch.equal(module(features), restored(features)))

    def test_config_and_input_validation(self):
        for change in (dict(num_heads=0), dict(num_heads=True), dict(attention_dropout=-.1),
                       dict(attention_dropout=1), dict(gate_init=0), dict(gate_init=1),
                       dict(gate_init=float("nan")), dict(k=3)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_relation({**SETTINGS, **change})
        with self.assertRaises(ValueError):
            build_relation(dict(SETTINGS, num_heads=3), dim=256)
        module = build_relation(SETTINGS)
        for features in (torch.empty(0, 256), torch.randn(2, 255), torch.ones(2, 256, dtype=torch.long)):
            with self.assertRaises(ValueError):
                module(features)


class GlobalMHSAIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_changes_only_relation_block(self):
        source = CONFIG.with_name("camil_s123_lrhr_aem02_wholebag_ssd_topk.yaml")
        if not CONFIG.is_file() or not source.is_file():
            self.skipTest("User-owned experiment YAML not present")
        before = yaml.safe_load(source.read_text())
        after = yaml.safe_load(CONFIG.read_text())
        self.assertEqual(after.pop("region_relation"), SETTINGS)
        before.pop("region_relation")
        self.assertEqual(before, after)
        validated = load_config(CONFIG)
        self.assertEqual(validated["region_relation"]["name"], "global_mhsa")

    def test_optimizer_head_groups_have_every_relation_parameter_once(self):
        config = small_config()
        model = build_model(config)
        optimizer = build_optimizer(model, config)
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})
        relation_ids = {id(p) for p in model.region_relation.parameters()}
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if id(parameter) in relation_ids:
                    self.assertEqual(group["lr"], config["classification"]["head_lr"])
                    self.assertEqual(group["weight_decay"], config["weight_decay"] if parameter.ndim > 1 else 0)

    def test_original_mil_and_aem_with_complete_bag(self):
        config = small_config()
        model = build_model(config).train()
        features, _ = model.encode_multiscale(torch.randn(3, 3, 64, 64),
                                             torch.randn(3, 2, 3, 128, 128))
        logits, attention = model.classify_bag(features, dict(slide_id="synthetic", n_regions=3, label=1))
        self.assertEqual(tuple(logits.shape), (1, 2))
        self.assertEqual(tuple(attention.shape), (3,))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
        loss += .02 * baseline.engine_aem.negative_attention_entropy(attention)
        loss.backward()
        for parameter in (model.base.encoder.layer4[0].conv1.weight, model.fusion[1].weight,
                          model.region_relation.attention.in_proj_weight, model.region_relation.gate_logit):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_ema_and_strict_model_checkpoint_restore(self):
        config = small_config()
        model = build_model(config)
        ema = baseline.camil.shared.ModelEMA(model, .999)
        before = {name: p.clone() for name, p in ema.module.region_relation.named_parameters()}
        with torch.no_grad():
            for parameter in model.region_relation.parameters():
                parameter.add_(.01)
        ema.update(model)
        for name, parameter in model.region_relation.named_parameters():
            expected = before[name].lerp(parameter.detach(), 1 - 2/11)
            torch.testing.assert_close(ema.module.region_relation.state_dict()[name], expected, rtol=0, atol=0)
        saved = dict(format=model.architecture()["type"], architecture=model.architecture(),
                     config=copy.deepcopy(config), data_spec=data_spec(config).metadata(),
                     model_state=ema.module.state_dict())
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "mhsa.pth"
            torch.save(saved, checkpoint)
            restored = model_from_checkpoint(torch.load(checkpoint, weights_only=True)).eval()
        features = torch.randn(5, 32)
        with torch.no_grad():
            expected = ema.module.classify(features)[0]
            actual = restored.classify(features)[0]
        self.assertTrue(torch.equal(expected, actual))
        missing = dict(saved["model_state"])
        missing.pop("region_relation.attention.in_proj_weight")
        with self.assertRaises(RuntimeError):
            model_from_checkpoint(dict(saved, model_state=missing))

    def test_bounded_cpu_wsi_smoke_with_original_training_step(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as temporary:
            samples, cache = pixel_fixture(temporary, n=3)
            result = smoke_steps(config, list(samples.values()), cache, torch.device("cpu"), steps=2)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["ema_updates"], 2)
        self.assertTrue(result["strict_reload_prediction_equal"])
        for step in result["steps"]:
            self.assertEqual(step["relation_inputs"], [[3, 32]])
            self.assertGreater(step["relation_gradient_norms"]["region_relation.gate_logit"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
