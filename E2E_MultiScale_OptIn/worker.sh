#!/usr/bin/env bash
set -euo pipefail
ROOT="$1"
PYTHON="$2"
HERE="$3"
PROTO="$4"
CACHE="$5"
RUN0="$6"
RUNSR="$7"
LOG="$8"
cd "$ROOT"

# `noclobber` prevents accidental clobber even if a race happened since launch.
set -C
exec > "$LOG" 2>&1
set +C
trap 'code=$?; echo "TRAIN_EXIT_CODE=$code"; exit "$code"' EXIT

echo "[start] $(date -Is) E2E multiscale, original files untouched"
echo "[cache] storing HR pixels (PNG), NOT frozen/feature embeddings"
echo "[cache] decode 8K images once; can be slow and consume disk space"
"$PYTHON" -u -B "$HERE/multiscale_train.py" \
    --config "$HERE/configs/camil_s123_lrhr_aem_sr0.yaml" \
    --protocol-dir "$PROTO" --cache-dir "$CACHE" --prepare-cache

echo "===== EXPERIMENT 1: LR+HR online shared-encoder E2E, SR=OFF ====="
"$PYTHON" -u -B "$HERE/multiscale_train.py" \
    --config "$HERE/configs/camil_s123_lrhr_aem_sr0.yaml" \
    --protocol-dir "$PROTO" --cache-dir "$CACHE" \
    --device cuda:0 --output "$RUN0"

echo "===== EXPERIMENT 2: LR+HR online E2E, LR->HR SR=ON (lambda 0.05) ====="
"$PYTHON" -u -B "$HERE/multiscale_train.py" \
    --config "$HERE/configs/camil_s123_lrhr_aem_sr005.yaml" \
    --protocol-dir "$PROTO" --cache-dir "$CACHE" \
    --device cuda:0 --output "$RUNSR"

echo "[done] $(date -Is) both fivefold experiments completed"
