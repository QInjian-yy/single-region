import math
import unittest

import torch

from models.asrd_interaction import AdaptiveSparseRelationalDiffusion


class ASRDInteractionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)

    def test_identity_and_backward_for_requested_bag_sizes(self):
        for count in (1, 2, 3, 5, 16, 37):
            with self.subTest(count=count):
                module = AdaptiveSparseRelationalDiffusion()
                h = torch.randn(count, 256, requires_grad=True)
                result = module(h)
                self.assertEqual(result.shape, h.shape)
                self.assertTrue(torch.equal(result, h))
                self.assertTrue(result.isfinite().all())
                if count == 1:
                    self.assertIs(result, h)
                result.square().mean().backward()
                self.assertTrue(h.grad.isfinite().all())
                if count > 1:
                    self.assertGreater(module.gamma.grad.abs().item(), 0)
                    for name, parameter in module.named_parameters():
                        if name != "gamma":
                            self.assertEqual(parameter.grad.count_nonzero().item(), 0)

    def test_topk_excludes_self_and_matches_relative_message_formula(self):
        for count in (2, 3, 5, 16, 37):
            with self.subTest(count=count):
                module = AdaptiveSparseRelationalDiffusion(dim=16, rank=4, temperature=0.7, gamma_init=0.5)
                h = torch.randn(count, 16, requires_grad=True)
                normalized = module.norm(h)
                query, key = module.query(normalized), module.key(normalized)
                scores = query @ key.T / math.sqrt(module.rank)
                self.assertFalse(torch.equal(scores, scores.T))
                indices, weights = module._neighbors(query, key)
                k = min(3, count - 1)
                self.assertEqual(indices.shape, (count, k))
                self.assertTrue((indices != torch.arange(count)[:, None]).all())
                torch.testing.assert_close(weights.sum(dim=1), torch.ones(count))

                values = module.value(normalized)
                expected = []
                for i in range(count):
                    neighbors = sorted((j for j in range(count) if j != i),
                                       key=lambda j: float(scores[i, j].detach()), reverse=True)[:k]
                    self.assertEqual(indices[i].tolist(), neighbors)
                    probability = (scores[i, neighbors] / 0.7).softmax(dim=0)
                    message = sum(probability[t] * (values[j] - values[i]) for t, j in enumerate(neighbors))
                    expected.append(h[i] + 0.5 * module.gate(normalized[i]).sigmoid() * module.output(message))
                result = module(h)
                torch.testing.assert_close(result, torch.stack(expected))
                self.assertTrue(result.isfinite().all())
                result.square().mean().backward()
                for parameter in module.parameters():
                    self.assertTrue(parameter.grad.isfinite().all())
                if k > 1:
                    for projection in (module.query, module.key, module.value, module.output):
                        self.assertGreater(projection.weight.grad.norm().item(), 0)

    def test_another_region_changes_the_current_region_output(self):
        module = AdaptiveSparseRelationalDiffusion(dim=8, rank=4, gamma_init=0.8)
        h = torch.randn(2, 8, requires_grad=True)
        original = module(h)
        altered = h.detach().clone()
        altered[1, 3] += 2
        self.assertFalse(torch.allclose(original[0], module(altered)[0]))
        cross_region_gradient, = torch.autograd.grad(original[0].square().sum(), h)
        self.assertGreater(cross_region_gradient[1].norm().item(), 0)

    def test_permutation_equivariance_with_distinct_scores(self):
        module = AdaptiveSparseRelationalDiffusion(dim=32, rank=8, gamma_init=0.7)
        h = torch.randn(37, 32)
        normalized = module.norm(h)
        scores = module.query(normalized) @ module.key(normalized).T
        self.assertTrue(all(row.unique().numel() == len(h) for row in scores))
        order = torch.randperm(len(h))
        torch.testing.assert_close(module(h[order]), module(h)[order], atol=2e-6, rtol=1e-5)

    def test_bfloat16_inputs_and_weights(self):
        module = AdaptiveSparseRelationalDiffusion(dim=32, gamma_init=0.25).bfloat16()
        h = torch.randn(5, 32, dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = module(h)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(output.isfinite().all())
        output.float().square().mean().backward()
        self.assertTrue(h.grad.isfinite().all())
        self.assertTrue(all(p.grad.isfinite().all() for p in module.parameters()))

    def test_invalid_settings_and_bags(self):
        for kwargs in ({"dim": 0}, {"rank": True}, {"topk": 0}, {"temperature": 0},
                       {"temperature": float("nan")}, {"gamma_init": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AdaptiveSparseRelationalDiffusion(**kwargs)
        module = AdaptiveSparseRelationalDiffusion()
        for h in (torch.empty(0, 256), torch.ones(2, 255), torch.ones(256)):
            with self.assertRaises(ValueError):
                module(h)
