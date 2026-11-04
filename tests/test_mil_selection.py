import copy
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from engine import train_wsi
from models.baseline import model_from_checkpoint
from models.robust_mil import RobustMILPooling
from models.shared_resnet_sr import CHECKPOINT_FORMAT, SharedResNetSRMIL
from train_e2e_shared import load_config


BRANCHES = ("attention", "mean", "topk")
COMBINATIONS = [choice for size in (1, 2, 3)
                for choice in itertools.combinations(BRANCHES, size)]


class MILSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def make_model(self, pooling=BRANCHES):
        torch.manual_seed(41)
        return SharedResNetSRMIL(
            shared_encoder={"pretrained": False, "trainable": "layer4"},
            classification={"embedding_dim": 32, "attention_hidden_dim": 16,
                            "topk": 2, "pooling": pooling},
            scale=2, sr_channels=16,
        ).eval()

    def test_all_combinations_select_existing_branches_and_backpropagate(self):
        reference = RobustMILPooling(4, 3, topk=2).eval()
        embeddings = torch.tensor([[1., 7., 3., 2.], [5., 2., 8., 4.],
                                   [3., 9., 1., 6.]])
        expected, _ = reference(embeddings)
        chunks = dict(zip(BRANCHES, expected.detach().split(4, dim=1)))
        torch.testing.assert_close(chunks["mean"], embeddings.mean(0, keepdim=True))
        torch.testing.assert_close(chunks["topk"], torch.tensor([[4., 8., 5.5, 5.]]))
        for choice in COMBINATIONS:
            with self.subTest(pooling=choice):
                pool = RobustMILPooling(4, 3, topk=2, pooling=choice).eval()
                pool.load_state_dict(reference.state_dict(), strict=True)
                inputs = embeddings.clone().requires_grad_()
                if "attention" in choice:
                    pooled, weights = pool(inputs)
                    torch.testing.assert_close(weights.sum(), torch.tensor(1.))
                else:
                    with patch.object(pool.attention_V, "forward",
                                      side_effect=AssertionError("Attention was executed")):
                        pooled, weights = pool(inputs)
                    self.assertIsNone(weights)
                torch.testing.assert_close(pooled, torch.cat([chunks[x] for x in choice], 1))
                self.assertEqual(pooled.shape, (1, 4 * len(choice)))
                pooled.square().sum().backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())
                self.assertGreater(inputs.grad.abs().sum().item(), 0)
                self.assertEqual(pool(torch.ones(1, 4))[0].shape, (1, 4 * len(choice)))

    def test_invalid_configurations_fail_and_single_string_is_supported(self):
        path = Path(__file__).resolve().parents[1] / "configs/native512_shared_resnet18_robust.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "config.yaml"
            for choice in ([], None, True, "unknown", ["mean", "mean"], ["topk", "bad"]):
                with self.subTest(pooling=choice):
                    config["classification"]["pooling"] = choice
                    target.write_text(yaml.safe_dump(config), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "pooling"):
                        load_config(target)
            config["classification"]["pooling"] = "mean"
            target.write_text(yaml.safe_dump(config), encoding="utf-8")
            self.assertEqual(load_config(target)["classification"]["pooling"], ["mean"])
            config["classification"].pop("pooling")
            target.write_text(yaml.safe_dump(config), encoding="utf-8")
            self.assertEqual(load_config(target)["classification"]["pooling"], list(BRANCHES))
        self.assertEqual(RobustMILPooling.normalize_pooling(["topk", "attention"]),
                         ("attention", "topk"))

    def test_engine_trains_each_combination_with_sr_on_and_off(self):
        lr = torch.linspace(0, 1, 3 * 3 * 16 * 16).reshape(3, 3, 16, 16)
        sample = {"slide_id": "test", "n_regions": 3, "label": 1,
                  "lr_paths": [0, 1, 2], "hr_paths": [0, 1, 2]}
        for choice, coefficient in itertools.product(COMBINATIONS, (0.0, 0.05)):
            with self.subTest(pooling=choice, lambda_sr=coefficient):
                model = self.make_model(choice)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                config = {"training": {"use_region_microbatch": True,
                                       "region_microbatch_size": 2},
                          "lambda_sr": coefficient, "sr_train_crop": 8,
                          "precision": "fp32"}
                parameters = [model.encoder.layer4[0].conv1.weight,
                              model.classifier_head.projection[1].weight,
                              model.classifier_head.classifier[-1].weight]
                if coefficient:
                    parameters.append(model.sr.fuse[0].weight)
                before = [p.detach().clone() for p in parameters]
                with patch("engine.load_images", side_effect=lambda paths, size: lr[paths]), \
                     patch("engine.load_hr_crops", return_value=torch.zeros(1, 3, 8, 8)) as hr, \
                     patch("engine.torch.randint", return_value=torch.zeros(3, 2, dtype=torch.long)), \
                     patch.object(model.sr, "fuse_stages", wraps=model.sr.fuse_stages) as fuse, \
                     patch.object(model.sr, "decode_crop", wraps=model.sr.decode_crop) as decode, \
                     patch.object(model.classifier_head.mil, "forward",
                                  wraps=model.classifier_head.mil.forward) as pool:
                    result = train_wsi(model, sample, optimizer, torch.device("cpu"), config)
                self.assertEqual(pool.call_count, 1)
                self.assertEqual(pool.call_args.args[0].shape, (3, 32))
                self.assertEqual((hr.call_count, fuse.call_count, decode.call_count),
                                 (3, 2, 3) if coefficient else (0, 0, 0))
                self.assertAlmostEqual(result["train_loss_total"], result["train_loss_cls"]
                                       + coefficient * result["train_loss_sr"], places=6)
                for parameter, original in zip(parameters, before):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
                    self.assertFalse(torch.equal(parameter, original))
                if not coefficient:
                    self.assertTrue(all(p.grad is None for p in model.sr.parameters()))
                if "attention" not in choice:
                    self.assertTrue(all(p.grad is None for p in model.classifier_head.mil.parameters()))

    def test_selection_preserves_encoder_projection_sr_and_initialization_rng(self):
        reference = self.make_model()
        expected_rng = torch.get_rng_state().clone()
        reference_state = reference.state_dict()
        lr = torch.linspace(0, 1, 2 * 3 * 16 * 16).reshape(2, 3, 16, 16)
        with torch.no_grad():
            sr_feature, embedding = reference.encode_regions(lr)
            expected_sr = reference.sr.decode_crop(sr_feature[:1], (3, 5, 8, 8))
        for choice in COMBINATIONS:
            with self.subTest(pooling=choice):
                model = self.make_model(choice)
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                for name, value in model.state_dict().items():
                    if name.startswith(("encoder.", "sr.", "classifier_head.projection.")):
                        torch.testing.assert_close(value, reference_state[name], atol=0, rtol=0)
                with torch.no_grad():
                    feature, actual_embedding = model.encode_regions(lr)
                    actual_sr = model.sr.decode_crop(feature[:1], (3, 5, 8, 8))
                    logits = model.forward_embeddings(actual_embedding)
                torch.testing.assert_close(actual_embedding, embedding, atol=0, rtol=0)
                torch.testing.assert_close(actual_sr, expected_sr, atol=0, rtol=0)
                self.assertEqual(logits.shape, (1, 2))
                self.assertEqual(model.classifier_head.classifier[-1].in_features,
                                 32 * len(choice))

    def test_checkpoint_roundtrip_legacy_default_and_same_width_mismatch(self):
        for choice in COMBINATIONS:
            with self.subTest(pooling=choice):
                model = self.make_model(choice)
                config = {"metasr": {"scale": 2, "rgb_range": 1, "channels": 16},
                          "shared_encoder": {"pretrained": False, "trainable": "layer4"},
                          "classification": {"embedding_dim": 32, "attention_hidden_dim": 16,
                                             "topk": 2, "pooling": list(choice)}}
                saved = {"format": CHECKPOINT_FORMAT, "architecture": model.architecture(),
                         "config": config, "model_state": model.state_dict()}
                if choice == BRANCHES:
                    config["classification"].pop("pooling")
                    saved["architecture"].pop("mil_pooling", None)
                restored = model_from_checkpoint(saved).eval()
                embeddings = torch.linspace(-1, 1, 3 * 32).reshape(3, 32)
                with torch.no_grad():
                    torch.testing.assert_close(restored.forward_embeddings(embeddings),
                                               model.forward_embeddings(embeddings), atol=0, rtol=0)
                if choice == ("attention",):
                    invalid = dict(saved, config=copy.deepcopy(config))
                    invalid["config"]["classification"]["pooling"] = ["mean"]
                    with self.assertRaisesRegex(ValueError, "pooling"):
                        model_from_checkpoint(invalid)


if __name__ == "__main__":
    unittest.main()
