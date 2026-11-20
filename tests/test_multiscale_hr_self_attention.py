import unittest

import torch
from torch import nn

from E2E_MultiScale_OptIn.multiscale_hr_self_attention_train import (
    HRSelfAttentionMIL,
    HRSelfAttentionPool,
    baseline,
)
from test_multiscale_relations import small_config


class HRSelfAttentionTests(unittest.TestCase):
    def test_identity_at_initialization_and_interaction_is_local(self):
        torch.set_num_threads(2)
        torch.manual_seed(1)
        pool = HRSelfAttentionPool(nn.Linear(256, 1)).eval()
        hr = torch.randn(3, 4, 256)
        with torch.no_grad():
            torch.testing.assert_close(pool.interact(hr), hr, rtol=0, atol=0)
            expected = (pool.scorer(hr).softmax(dim=1) * hr).sum(dim=1)
            torch.testing.assert_close(pool(hr), expected, rtol=0, atol=0)
            pool.gamma.fill_(0.1)
            altered = hr.clone()
            altered[0, 0, 0] += 5
            original = pool.interact(hr)
            updated = pool.interact(altered)
            self.assertEqual(tuple(updated.shape), (3, 4, 256))
            self.assertFalse(torch.allclose(original[0, 1:], updated[0, 1:]))
            torch.testing.assert_close(original[1:], updated[1:], rtol=0, atol=0)

    def test_gamma_and_attention_are_optimized(self):
        torch.set_num_threads(2)
        torch.manual_seed(2)
        config = small_config()
        config["multiscale"]["patches_per_region"] = 4
        model = HRSelfAttentionMIL(config)
        optimizer = baseline.build_optimizer(model, config)
        hr = torch.randn(3, 4, config["classification"]["embedding_dim"])
        optimizer.zero_grad(set_to_none=True)
        model.patch_score(hr).square().mean().backward()
        self.assertEqual(model.patch_score.gamma.item(), 0.0)
        self.assertGreater(abs(model.patch_score.gamma.grad.item()), 0.0)
        weight = model.patch_score.attention.in_proj_weight
        self.assertEqual(torch.count_nonzero(weight.grad).item(), 0)
        optimizer.step()

        self.assertNotEqual(model.patch_score.gamma.item(), 0.0)
        optimizer.zero_grad(set_to_none=True)
        model.patch_score(hr).square().mean().backward()
        self.assertTrue(torch.isfinite(weight.grad).all())
        self.assertGreater(torch.count_nonzero(weight.grad).item(), 0)


if __name__ == "__main__":
    unittest.main()
