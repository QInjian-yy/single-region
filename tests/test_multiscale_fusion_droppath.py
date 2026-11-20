import unittest

import torch

from E2E_MultiScale_OptIn.multiscale_fusion_droppath_train import FusionDropPathMIL, baseline
from test_multiscale_relations import small_config


class FusionDropPathTests(unittest.TestCase):
    def test_training_preserves_lr_skip_and_evaluation_matches_baseline(self):
        torch.set_num_threads(2)
        config = small_config()
        config["multiscale"]["patches_per_region"] = 4
        config["classification"]["region_dropout"] = 0.0
        config["fusion_drop_path"] = 0.10
        model = FusionDropPathMIL(config)
        control = baseline.MultiScaleSRMIL(config)
        control.load_state_dict(model.state_dict(), strict=True)
        generator = torch.Generator().manual_seed(1)
        lr = torch.rand(4, 3, 64, 64, generator=generator)
        hr = torch.rand(4, 4, 3, 128, 128, generator=generator)

        with torch.no_grad():
            model.eval()
            control.eval()
            expected, _ = control.encode_multiscale(lr, hr)
            observed, _ = model.encode_multiscale(lr, hr)
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)

            model.train()
            for seed in range(128):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    dropped = model.fusion.drop_path(torch.ones(4, 1)).flatten() == 0
                if dropped.any() and not dropped.all():
                    break
            else:
                self.fail("Could not find a mixed stochastic-depth mask")

            lr_only = model.base.classifier_head(model.base.encoder(lr)[-1]).float()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                observed, _ = model.encode_multiscale(lr, hr)
            torch.testing.assert_close(observed[dropped], lr_only[dropped], rtol=0, atol=0)
            self.assertFalse(torch.equal(observed[~dropped], lr_only[~dropped]))
            logits, attention = model.classify(observed)
            self.assertEqual(tuple(logits.shape), (1, 2))
            self.assertEqual(tuple(attention.shape), (4,))


if __name__ == "__main__":
    unittest.main()
