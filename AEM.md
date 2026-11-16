# Opt-in AEM for CAMIL-style five-fold E2E training

This adds the AEM objective from **Attention Entropy Maximization for Multiple
Instance Learning based Whole Slide Image Classification (MICCAI 2025)** to the
existing Native512 AttentionMIL + AdamW pipeline, with optional ASRD.

Sources:

- Final paper: https://papers.miccai.org/miccai-2025/paper/5183_paper.pdf
- Author implementation: https://github.com/dazhangyu123/AEM/blob/main/main.py

The published paper specifies cosine weight annealing. The inspected public
`main.py` uses a constant coefficient in its loss. This integration implements
the paper's negative-entropy term and an explicit cosine schedule. It preserves
this project's E2E encoder and CAMIL data protocol; it is not a reproduction of
the paper's feature extraction, splits or reported benchmark scores.

## Files to synchronize

Required runtime additions:

- `engine_aem.py`
- `train_e2e_camil_aem.py`
- `configs/camil_attn_adamw_asrd_aem_5fold_seed1.yaml` (primary ASRD + AEM experiment)
- `configs/camil_attn_adamw_aem_5fold_seed1.yaml` (AttentionMIL + AEM control)

Also provided: `tests/test_aem.py` and this document. Existing Python files,
configs, split CSVs, checkpoints, runs and logs are not modified. The existing
ASRD, CAMIL loader and shared training dependencies must already be present.

**Use `train_e2e_camil_aem.py` for AEM. The existing `train_e2e_camil.py` does not
compute this new regularizer.** No formal training is started by this change.

## Objective and data flow

`Native512 RGB regions -> ResNet18 -> Projection [N,256] -> optional ASRD
-> original AttentionMIL -> original binary classifier`.

Let `alpha` be the normalized, differentiable Region weights returned by
**AttentionMIL**, not ASRD's Top-k edge weights:

    L_AEM = sum_i alpha_i * log(alpha_i)
    L_train = L_existing + lambda(epoch) * L_AEM

`L_existing` is computed by the original `engine.wsi_loss`, retaining its class
weights, label smoothing, optional SR term and optional VIB term. A temporary
forward hook captures AttentionMIL's existing output and is removed even if
the forward fails. There is no additional encoder pass, random mask, trainable
parameter, optimizer, or second application of ASRD. The negative-entropy term
is computed in FP32 with a finite logarithm at zero.

The gradient remains attached through AttentionMIL, ASRD, Projection and the
trainable encoder stages. A single-Region bag has attention `[1]`, AEM loss zero
and no AEM gradient. Nothing is dropped. All added parameters: **0**.

AEM is a training objective, so checkpoint model formats and state keys remain
those of the existing baseline/ASRD model. Extra `training_regularization`,
`training_entry` and `epoch_aem_weight` metadata distinguish the training recipe.
The original strict baseline/ASRD model loaders restore inference weights.
EMA uses the existing `ModelEMA`; no live hook or attention graph is stored in
the model or EMA copy. Evaluation uses the original unweighted classification
loss and prediction path, without an AEM term.

## Configuration and controls

```yaml
classification:
  pooling: [attention]
  aem:
    enabled: true
    weight: 0.01
    anneal_epochs: 50
optimizer: adamw
```

For epoch `e >= 1`, `t = min((e-1)/(anneal_epochs-1), 1)` and
`lambda(e) = weight * (1 + cos(pi*t)) / 2`.
With the supplied settings, epoch 1 uses 0.01, epoch 50 and later use zero.
The 50-epoch window is an explicit experiment choice, independent of the
unchanged maximum of 300 epochs. Early stopping may end training before the
coefficient reaches zero. This schedule changes only the AEM coefficient, not
the learning-rate scheduler or EMA decay.

Disable it using `classification.aem.enabled: false`, or omit `aem`. The new
entry delegates to the original CAMIL entry in that case. Setting `weight: 0`
also leaves the optimized loss and parameter updates unchanged. With AEM
enabled, this entry requires `pooling: [attention]` and `optimizer: adamw`.
MeanMIL has fixed weights, so this regularizer is not offered for MeanMIL.
GCSAM is not enabled by the AEM entry; existing GCSAM paths remain unchanged.

Both new recipes are exact copies of the existing ASRD recipe plus the AEM
settings. The AEM-only control additionally sets `region_interaction.enabled`
to false. Encoder/head learning rates, all zero Dropouts, weight decay, label
smoothing, augmentations, seed, EMA and data protocol remain unchanged.

Validation, checkpoints and stopping retain the baseline policy: maximum 300
epochs, min_epochs 5, patience 12 and min_delta 0.001. **As in the existing
entry, epochs 1-4 cannot become best.pth, and improvements <= min_delta do not
replace best.pth.** The plateau scheduler monitors Val classification loss.
After training, the selected EMA checkpoint is strictly reloaded, the selected
Val threshold is reused, and Test is evaluated once. Test does not drive
training or stopping. Default execution runs folds 0-4 sequentially and records
the Test AUC mean and sample standard deviation in `summary.json`.

Per-fold outputs remain `config.yaml`, `data_provenance.json`, `run_info.json`,
`history.csv`, `best.pth`, `best_val_predictions.csv`, `test_predictions.csv`
and `test_metrics.json`. New history columns are `aem_weight`, `train_loss_aem`,
`train_loss_aem_weighted` and `train_attention_entropy`. Entropy is averaged
over training WSI steps; its maximum depends on Region count (`log N`).
The AEM term is nonpositive, so its addition can lower total loss even without
better classification. Assess the experiment using validation metrics, not
the magnitude of total training loss or entropy alone.

## Local automated checks

```powershell
& 'D:\conda\envs\clam_latest\python.exe' -B -m unittest tests.test_aem -v
& 'D:\conda\envs\clam_latest\python.exe' -B -m unittest discover -s tests
```

Tests use temporary synthetic inputs and delete their files after completion.
They cover entropy sign/gradients/edge cases, configuration validation, cosine
endpoints, unchanged disabled/zero-weight updates, encoder/ASRD/MIL gradients,
SR/VIB loss composition, Native512 CPU and CUDA BF16 training, EMA and strict
checkpoint restoration, Val-only checkpoint selection, one final Test per
fold, five-fold aggregation, input preservation and overwrite rejection.
Synthetic checks do not establish real-data AUC improvements.

Local verification on 2026-11-11: all **180 tests passed** (186.578 seconds),
including the CUDA BF16 integration test. The real local protocol reader also
validated 393 WSI across five folds. The original 121 code/config/protocol files
retained identical SHA-256 hashes. With the default 256-dimensional model and
the same seed, baseline ASRD and ASRD+AEM initialization and parameter counts
matched exactly. No real-data training or server execution was performed.

## Server commands (run manually after synchronization)

Set up the environment and validate the existing protocol and both roots:

```bash
conda activate clam_latest
cd "/media/ub/shidang_plu/MT/py_project/yujian/double_wsi/CAMELYON16_WSI_DATA/E2E_MetaSR_GCSAM"
CFG="$PWD/configs/camil_attn_adamw_asrd_aem_5fold_seed1.yaml"
PROTO="$PWD/data_protocols/camil_5fold"
python -B "$PWD/train_e2e_camil_aem.py" --config "$CFG" --protocol-dir "$PROTO" --check-data
```

Run **Fold 0 only** in a new screen session:

```bash
RUN="camil_attn_adamw_asrd_aem_f0_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PWD/logs"
python -B "$PWD/train_e2e_camil_aem.py" --config "$CFG" --protocol-dir "$PROTO" --check-data &&
screen -L -Logfile "$PWD/logs/$RUN.log" -dmS "$RUN" \
  python -u -B "$PWD/train_e2e_camil_aem.py" \
  --config "$CFG" --protocol-dir "$PROTO" --fold 0 --device cuda:0 --output "$PWD/runs/$RUN"
```

Or run **all five folds sequentially** in a new screen session:

```bash
RUN="camil_attn_adamw_asrd_aem_5fold_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PWD/logs"
python -B "$PWD/train_e2e_camil_aem.py" --config "$CFG" --protocol-dir "$PROTO" --check-data &&
screen -L -Logfile "$PWD/logs/$RUN.log" -dmS "$RUN" \
  python -u -B "$PWD/train_e2e_camil_aem.py" \
  --config "$CFG" --protocol-dir "$PROTO" --device cuda:0 --output "$PWD/runs/$RUN"
```

Do not start both blocks if only one run is intended. Check the selected run:

```bash
tail -n 20 -F "$PWD/logs/$RUN.log"
```

For the AEM-only control, change `CFG` to
`$PWD/configs/camil_attn_adamw_aem_5fold_seed1.yaml` and choose a distinct RUN.
Compare the ASRD+AEM experiment to the completed ASRD+AdamW run; compare the
AEM-only experiment to AttentionMIL+AdamW. Select hyperparameters using each
fold's Val data and report all five final Test results.
