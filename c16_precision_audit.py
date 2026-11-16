"""Read-only BF16/FP32 re-evaluation of saved CAMIL fold checkpoints.

Run in E2E_MetaSR_GCSAM project root; no training or file writes.
"""
import argparse
import copy
import csv
from collections import Counter
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import engine
from camil_data import load_camil_samples, make_camil_datasets, read_camil_protocol
from models.shared_resnet_sr import model_from_checkpoint
from wsi_data import collate_one_wsi


def distribution(predictions):
    scores = [float(p['p_tumor']) for p in predictions]
    histogram = Counter(scores)
    median = lambda v: sorted(v)[len(v)//2] if v else float('nan')
    normal = [float(p['p_tumor']) for p in predictions if int(p['label']) == 0]
    tumor = [float(p['p_tumor']) for p in predictions if int(p['label']) == 1]
    return (f'n={len(scores)}, unique={len(histogram)}, tie_extra={len(scores)-len(histogram)}, '
            f'normal_med={median(normal):.6f}, tumor_med={median(tumor):.6f}, '
            f'min={min(scores):.6f}, max={max(scores):.6f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--protocol-dir', type=Path, default=Path('data_protocols/camil_5fold'))
    parser.add_argument('--fold', type=int, choices=range(5), default=2)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    fold_dir = args.run / f'fold_{args.fold}'
    ckpt = fold_dir / 'best.pth'
    if not ckpt.is_file():
        parser.error(f'Checkpoint does not exist: {ckpt}')
    saved = torch.load(ckpt, map_location='cpu', weights_only=True)
    device = torch.device(args.device)
    model = model_from_checkpoint(saved, device)
    model.eval()
    protocol = read_camil_protocol(args.protocol_dir)
    samples, _ = load_camil_samples(protocol, require_hr=False)
    _, val, test = make_camil_datasets(protocol, samples, args.fold)
    datasets = [('VAL', val), ('TEST', test)]
    print('Fold:', args.fold, '| checkpoint_epoch:', saved['epoch'], '| no model or data modification', flush=True)
    for name, dataset in datasets:
        results = {}
        for precision in ('bf16', 'fp32'):
            cfg = copy.deepcopy(saved['config'])
            cfg['precision'] = precision
            loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_one_wsi)
            output = engine.evaluate(model, loader, device, cfg)
            results[precision] = output
            print(f'{name} {precision}: auc={output["auc"]:.6f}; {distribution(output["predictions"])}', flush=True)
        p_bf = {p['slide_id']: p for p in results['bf16']['predictions']}
        p_fp = {p['slide_id']: p for p in results['fp32']['predictions']}
        changes = [abs(float(p_bf[s]['p_tumor']) - float(p_fp[s]['p_tumor'])) for s in p_bf]
        print(f'{name}: fp32-minus-bf16 AUC={results["fp32"]["auc"]-results["bf16"]["auc"]:+.6f}; '
              f'max_absolute_probability_change={max(changes):.6f}', flush=True)
    print('Completed. Checkpoints, existing results, and project files were not changed.', flush=True)


if __name__ == '__main__':
    main()
