"""Synthetic CUDA probe through the actual multiscale train_wsi and EMA path."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'E2E_MultiScale_OptIn'), str(ROOT)]
import torch
import multiscale_train as attention_train
import multiscale_train_mean as mean_train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--pooling', choices=('mean', 'attention'), default='mean')
    parser.add_argument('--microbatch', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.n > 0 and not args.output.exists()
    trainer = mean_train if args.pooling == 'mean' else attention_train
    config_path = ROOT / 'E2E_MultiScale_OptIn/configs' / (
        'camil_s123_lrhr_mean_sr0.yaml' if args.pooling == 'mean'
        else 'camil_s123_lrhr_aem_sr0.yaml')
    config = trainer.load_config(config_path)
    # No pretrained cache on this machine: only weight values change, not architecture.
    config['shared_encoder']['pretrained'] = False
    config['training']['use_region_microbatch'] = args.microbatch
    config['training']['region_microbatch_size'] = 1
    trainer._validate_config(config)
    assert config['precision'] == 'bf16' and config['lambda_sr'] == 0
    assert torch.cuda.is_bf16_supported()
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    n = args.n
    lr = torch.rand(n, 3, 512, 512)
    k, p = config['multiscale']['patches_per_region'], config['multiscale']['hr_patch_size']
    hr = torch.rand(n, k, 3, p, p)
    sample = dict(n_regions=n, label=1, slide_id='synthetic-full-bag',
                  lr_paths=list(range(n)), hr_paths=list(range(n)))

    class Pixels:
        size = p
        def read(self, paths):
            return hr[paths], [[[0, 0, p, p]] * k for _ in paths]

    def snapshot():
        torch.cuda.synchronize()
        return dict(allocated_bytes=torch.cuda.memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())

    report = dict(synthetic=True, config=config, config_path=str(config_path),
                  pooling=args.pooling, n=n, mode='micro1' if args.microbatch else 'full',
                  torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), total_vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (Path(trainer.__file__), ROOT/'engine.py',
                                              ROOT/'E2E_MultiScale_OptIn/multiscale_model_mean.py',
                                              ROOT/'E2E_MultiScale_OptIn/multiscale_model.py')}, steps=[])
    gpu_log = args.output.with_suffix('.gpu.csv')
    gpu_handle = gpu_log.open('x', encoding='utf-8')
    monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,memory.used,memory.free',
                                '--format=csv,noheader,nounits', '-lms', '100'],
                               stdout=gpu_handle, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    trace = {}
    try:
        model = trainer.MultiScaleSRMIL(config).to(device)
        optimizer = trainer.build_optimizer(model, config)
        ema = trainer.camil.shared.ModelEMA(model, config['ema_decay'])
        report['model_and_ema'] = snapshot()
        original_encode, original_classify = model.encode_multiscale, model.classify
        original_forward, original_backward = trainer.wsi_forward, torch.autograd.backward
        original_step = optimizer.step

        def encoder_input(module, inputs):
            trace['encoder_inputs'].append(list(inputs[0].shape))

        def stage4_output(module, inputs, output):
            trace['stage4'].append(dict(shape=list(output.shape), dtype=str(output.dtype),
                                        requires_grad=output.requires_grad))
            output.register_hook(lambda grad: trace['stage4_gradients'].append(list(grad.shape)))

        def encode(lr_batch, hr_batch, **kwargs):
            trace['region_batches'].append(dict(lr=list(lr_batch.shape), hr=list(hr_batch.shape)))
            embeddings, sr = original_encode(lr_batch, hr_batch, **kwargs)
            assert embeddings.requires_grad and embeddings.grad_fn is not None
            embeddings.register_hook(lambda grad: trace['fused_gradients'].append(list(grad.shape)))
            return embeddings, sr

        def classify(embeddings):
            trace['mil_inputs'].append(list(embeddings.shape))
            assert embeddings.requires_grad and embeddings.grad_fn is not None
            return original_classify(embeddings)

        def forward(*a, **kw):
            result = original_forward(*a, **kw)
            trace['after_forward'] = snapshot()
            return result

        def backward(*a, **kw):
            result = original_backward(*a, **kw)
            trace['backward_calls'] += 1
            trace['after_backward'] = snapshot()
            return result

        def step(*a, **kw):
            result = original_step(*a, **kw)
            trace['optimizer_steps'] += 1
            trace['after_optimizer'] = snapshot()
            return result

        model.base.encoder.register_forward_pre_hook(encoder_input)
        model.base.encoder.layer4.register_forward_hook(stage4_output)
        # Plain callables avoid Mock call histories retaining CUDA input tensors.
        with patch.object(trainer.engine, 'load_images', new=lambda paths, size: lr[paths]), \
             patch.object(model, 'encode_multiscale', new=encode), \
             patch.object(model, 'classify', new=classify), \
             patch.object(trainer, 'wsi_forward', new=forward), \
             patch.object(torch.autograd, 'backward', new=backward), \
             patch.object(optimizer, 'step', new=step):
            for index in range(3):
                trace = dict(index=index+1, region_batches=[], encoder_inputs=[], stage4=[],
                             stage4_gradients=[], fused_gradients=[], mil_inputs=[],
                             backward_calls=0, optimizer_steps=0)
                before = model.fusion[1].weight.detach().cpu().clone()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                trace['outcome'] = trainer.train_wsi(model, sample, Pixels(), optimizer, device, config, 1)
                ema.update(model)
                trace['after_ema'] = snapshot()
                trace['seconds'] = time.perf_counter()-started
                batches = [1]*n if args.microbatch else [n]
                assert [x['lr'][0] for x in trace['region_batches']] == batches
                assert trace['encoder_inputs'] == [shape for b in batches
                    for shape in ([b,3,512,512], [b*k,3,p,p])]
                assert trace['mil_inputs'] == [[n,256]]
                assert trace['backward_calls'] == trace['optimizer_steps'] == 1
                assert len(trace['fused_gradients']) == len(batches)
                assert len(trace['stage4_gradients']) == 2*len(batches)
                assert all(x['dtype']=='torch.bfloat16' and x['requires_grad'] for x in trace['stage4'])
                assert model.base.encoder.layer4[0].conv1.weight.grad is not None
                assert model.patch_score.weight.grad is not None
                assert not torch.equal(before, model.fusion[1].weight.detach().cpu())
                report['steps'].append(trace)
                print(f"STEP_OK {args.pooling} {report['mode']} N={n} step={index+1} "
                      f"peak_allocated_MiB={trace['after_ema']['peak_allocated_bytes']/2**20:.1f} "
                      f"peak_reserved_MiB={trace['after_ema']['peak_reserved_bytes']/2**20:.1f}", flush=True)
        assert ema.updates == 3
        report.update(status='passed', ema_updates=ema.updates)
    except torch.OutOfMemoryError as error:
        report.update(status='oom', error=str(error))
    finally:
        monitor.terminate()
        monitor.wait(timeout=5)
        gpu_handle.close()
        samples = list(csv.reader(gpu_log.read_text(encoding='utf-8').splitlines()))
        used = [float(row[1]) for row in samples if len(row)==3]
        report['nvidia_smi_100ms'] = dict(samples=len(used), used_before_mib=used[0] if used else None,
                                        peak_device_used_mib=max(used) if used else None)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    assert report['status'] == 'passed', report.get('error')


if __name__ == '__main__':
    main()
