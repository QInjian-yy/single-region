#!/usr/bin/env python3
"""Read-only CAMELYON16 pooled five-fold test diagnostics.

Does not train, select checkpoints, tune a threshold, or write files.
Usage from E2E_MetaSR_GCSAM root:
  python c16_test_diagnostics.py --run runs/NAME
"""
import argparse
import csv
import json
import statistics
from pathlib import Path


def auc(rows):
    positives = [r['score'] for r in rows if r['label'] == 1]
    negatives = [r['score'] for r in rows if r['label'] == 0]
    if not positives or not negatives:
        return None
    count = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return count / (len(positives) * len(negatives))


def fmt(x):
    return 'N/A' if x is None else f'{x:.4f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='runs/... folder containing fold_0..fold_4')
    parser.add_argument('--protocol-dir', type=Path, default=Path('data_protocols/camil_5fold'))
    args = parser.parse_args()
    run = args.run.resolve()
    protocol = json.loads((args.protocol_dir / 'protocol.json').read_text(encoding='utf-8'))
    source_index = json.loads((args.protocol_dir / 'dataset_sources.json').read_text(encoding='utf-8'))
    slides = source_index['slides']
    print(f"Run: {run}")
    print(f"Protocol: {protocol['total_wsis']} WSI, {protocol['total_regions']} regions; patient-independence verified={protocol.get('patient_independence_verified')}")
    print('NOTE: official_train / official_test denote original dataset partitions, not clinical hospitals.')
    pooled = []
    fold_aucs = []
    for fold in range(5):
        prediction_path = run / f'fold_{fold}' / 'test_predictions.csv'
        split_path = args.protocol_dir / f'splits_5fold_{fold}.csv'
        with split_path.open(encoding='utf-8-sig', newline='') as f:
            test_ids = {r['test'] for r in csv.DictReader(f) if r['test']}
        if not prediction_path.exists():
            print(f'Fold {fold}: MISSING {prediction_path}')
            continue
        rows = []
        with prediction_path.open(encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                slide_id = r['slide_id']
                if slide_id not in test_ids or slide_id not in slides:
                    raise ValueError(f'Fold {fold}: wrong test slide {slide_id}')
                label = int(r['label'])
                if label != slides[slide_id]['label']:
                    raise ValueError(f'Fold {fold}: label mismatch for {slide_id}')
                rows.append({'slide': slide_id, 'label': label,
                             'score': float(r['p_tumor']),
                             'source': slides[slide_id]['source'],
                             'n_regions': slides[slide_id]['n_regions'],
                             'fold': fold})
        if len(rows) != len(test_ids) or len({r['slide'] for r in rows}) != len(rows):
            raise ValueError(f'Fold {fold}: incomplete or duplicate test predictions')
        pooled.extend(rows)
        fold_auc = auc(rows)
        fold_aucs.append(fold_auc)
        print(f'\nFold {fold} n={len(rows)} overall_AUC={fmt(fold_auc)}')
        for source in ('official_train', 'official_test'):
            group = [r for r in rows if r['source'] == source]
            print(f"  {source:14s} n={len(group):2d} tumor={sum(r['label'] for r in group):2d} "
                  f"AUC={fmt(auc(group))} "
                  f"median_regions={statistics.median(r['n_regions'] for r in group) if group else 'N/A'}")
        positives = sorted((r for r in rows if r['label'] == 1), key=lambda r: r['score'])
        negatives = sorted((r for r in rows if r['label'] == 0), key=lambda r: r['score'], reverse=True)
        print('  lowest tumor scores:', ', '.join(f"{r['slide']}:{r['score']:.3f}" for r in positives[:3]))
        print('  highest normal scores:', ', '.join(f"{r['slide']}:{r['score']:.3f}" for r in negatives[:3]))

    if pooled:
        print(f'\nPooled test predictions n={len(pooled)} (one OOF prediction per slide expected)')
        print('Mean per-fold AUC:', fmt(statistics.mean(fold_aucs)))
        print('Pooled OOF AUC (not equivalent to mean fold AUC):', fmt(auc(pooled)))
        for source in ('official_train', 'official_test'):
            group = [r for r in pooled if r['source'] == source]
            print(f"{source}: n={len(group)}, AUC={fmt(auc(group))}")
        if len(pooled) == protocol['total_wsis']:
            if len({r['slide'] for r in pooled}) != len(pooled):
                raise ValueError('Pooled predictions contain slide overlap!')
            print('All pooled WSIs accounted for exactly once.')
        else:
            print('WARNING: Not all pooled WSIs present.')
    print('\nInterpretation: source-specific AUCs are exploratory diagnostics, not a basis for test-set hyperparameter tuning.')


if __name__ == '__main__':
    main()
