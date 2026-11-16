"""Opt-in self-check; direct invocation executes tests (unlike 0-test imports)."""
from __future__ import annotations
import sys
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory
from dataclasses import replace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import torch
import numpy as np
from PIL import Image
import yaml
from augmentation import TransformParams, apply_transform
from paired_geometry import crop_transformed
from multiscale_data import transform_box, _tissue_positions
from multiscale_model import MultiScaleSRMIL
from multiscale_train import load_config, build_optimizer, train_wsi, evaluate, camil


class MultiScaleAddonSmoke(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.cfg = load_config(HERE/'configs/camil_s123_lrhr_aem_sr005.yaml')
        self.cfg['shared_encoder']['pretrained'] = False
        self.cfg['classification']['embedding_dim'] = 32
        self.cfg['classification']['attention_hidden_dim'] = 16
        self.cfg['metasr']['channels'] = 16
        self.cfg['multiscale']['hr_patch_size'] = 128
        self.cfg['multiscale']['patches_per_region'] = 1
        self.cfg['sr_train_crop'] = 64
        self.cfg['precision'] = 'fp32'

    def test_configs_valid(self):
        self.assertEqual(load_config(HERE/'configs/camil_s123_lrhr_aem_sr0.yaml')['lambda_sr'], 0)
        self.assertEqual(load_config(HERE/'configs/camil_s123_lrhr_aem_sr005.yaml')['lambda_sr'], .05)

    def test_no_model_source_changes_needed(self):
        self.assertTrue((ROOT/'models/shared_resnet_sr.py').is_file())
        self.assertEqual(self.cfg['classification']['pooling'], ['attention'])

    def test_shared_gradient_and_sr(self):
        torch.manual_seed(7)
        model=MultiScaleSRMIL(self.cfg)
        model.train()
        lr=torch.rand(2,3,64,64)
        hr=torch.rand(2,1,3,128,128)
        feat, sr_feat=model.encode_multiscale(lr,hr,need_sr=True)
        self.assertEqual(tuple(feat.shape),(2,32))
        logits, attention=model.classify(feat)
        self.assertEqual(tuple(logits.shape),(1,2))
        self.assertEqual(tuple(attention.shape),(2,))
        sr_pred=model.base.sr.decode_crop(sr_feat[:1],(32,32,64,64))
        self.assertEqual(tuple(sr_pred.shape),(1,3,64,64))
        (torch.nn.functional.cross_entropy(logits,torch.tensor([1])) +
          .05*sr_pred.square().mean()).backward()
        self.assertIsNotNone(model.base.encoder.layer4[0].conv1.weight.grad)
        self.assertIsNotNone(model.fusion[1].weight.grad)
        self.assertIsNotNone(model.base.sr.tail.weight.grad)
        optimizer=build_optimizer(model,self.cfg)
        self.assertTrue(len(optimizer.param_groups)>1)

    def test_augmented_tile_coordinate_alignment(self):
        src=np.arange(16*16*3,dtype=np.uint8).reshape(16,16,3)
        image=Image.fromarray(src,'RGB')
        box=(4,3,5,5)
        for hflip in [False,True]:
            for vflip in [False,True]:
                for k in range(4):
                    params=TransformParams(hflip,vflip,k)
                    y,x,hh,ww=transform_box(box,params,16)
                    transformed=apply_transform(image.crop((box[1],box[0],box[1]+box[3],box[0]+box[2])),params)
                    expected=crop_transformed(image,(y,x,hh,ww),params)
                    self.assertTrue(np.array_equal(np.asarray(transformed),np.asarray(expected)))

    def test_one_wsi_joint_backprop_online_pixels(self):
        # Simulates the actual bag path instead of just direct model.forward().
        with TemporaryDirectory() as tmp:
            paths = []
            for i in range(2):
                path = Path(tmp) / f"lr_{i}.png"
                Image.fromarray(np.full((512,512,3), 110+i*20, dtype=np.uint8),"RGB").save(path)
                paths.append(path)

            class FakePixels:
                size = 128
                def read(self, names):
                    b = len(names)
                    return torch.rand(b,1,3,128,128), [[[0,0,128,128]] for _ in range(b)]

            torch.manual_seed(4)
            model = MultiScaleSRMIL(self.cfg)
            optimizer = build_optimizer(model, self.cfg)
            cfg = dict(self.cfg)
            cfg['training'] = dict(self.cfg['training'], region_microbatch_size=1)
            sample = {'n_regions':2,'label':1, 'slide_id':'fake',
                      'lr_paths':paths, 'hr_paths':[Path('a'),Path('b')]}
            outcome = train_wsi(model, sample, FakePixels(), optimizer,
                                torch.device('cpu'), cfg, 1)
            self.assertTrue(np.isfinite(outcome['train_loss_total']))
            self.assertGreater(outcome['train_loss_sr'],0)
            self.assertGreater(model.fusion[1].weight.grad.abs().sum().item(),0)
            self.assertGreater(model.base.sr.tail.weight.grad.abs().sum().item(),0)

    def test_tissue_positions_are_valid_unique(self):
        image=Image.fromarray(np.full((512,512,3),150,dtype=np.uint8),'RGB')
        boxes=_tissue_positions(image,256,16,2)
        self.assertEqual(len(boxes),2)
        self.assertEqual(len({tuple(x) for x in boxes}),2)
        for y,x,h,w in boxes:
            self.assertTrue(0 <= y <= 8192-256 and 0 <= x <= 8192-256)


if __name__=='__main__':
    unittest.main(verbosity=2)
