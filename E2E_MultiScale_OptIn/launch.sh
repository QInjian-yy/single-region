#!/usr/bin/env bash
# One-click opt-in launcher. Writes only *new* addon cache/runs/logs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$ROOT"
PYTHON="${MULTISCALE_PYTHON:-/home/ub/anaconda3/envs/clam_latest/bin/python}"
PROTO="${MULTISCALE_PROTOCOL:-$ROOT/data_protocols/camil_5fold_seed123_sourcestrat}"
CACHE="${MULTISCALE_CACHE:-$HERE/cache/lr512_hr256_k2_v1}"

[[ -f "$ROOT/models/shared_resnet_sr.py" ]] || { echo 'Missing original ResNet model; run from extracted project.' >&2; exit 2; }
[[ -f "$PROTO/protocol.json" ]] || { echo "Missing Seed 123 protocol: $PROTO" >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "Python not executable: $PYTHON" >&2; exit 2; }
command -v screen >/dev/null || { echo 'GNU screen not found' >&2; exit 2; }
mkdir -p logs runs "$HERE/cache"

# CPU smoke test before background launch; no weights download.
"$PYTHON" "$HERE/smoke_test.py"
for yaml in "$HERE/configs/camil_s123_lrhr_aem_sr0.yaml" "$HERE/configs/camil_s123_lrhr_aem_sr005.yaml"; do
    "$PYTHON" "$HERE/multiscale_train.py" --config "$yaml" \
        --protocol-dir "$PROTO" --cache-dir "$CACHE" --check-data
done

STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION="ms_e2e_s123_${STAMP}"
RUN0="$ROOT/runs/s123_e2e_multiscale_sr0_${STAMP}"
RUNSR="$ROOT/runs/s123_e2e_multiscale_sr005_${STAMP}"
LOG="$ROOT/logs/s123_e2e_multiscale_${STAMP}.log"
for dir in "$RUN0" "$RUNSR"; do
    [[ ! -e "$dir" ]] || { echo "Output already exists; refuse overwrite: $dir" >&2; exit 2; }
done
[[ ! -e "$LOG" ]] || { echo "Log already exists; refuse overwrite: $LOG" >&2; exit 2; }

# All further work (cache preprocessing and the two full E2E 5fold runs) is sequential.
screen -dmS "$SESSION" bash "$HERE/worker.sh" "$ROOT" "$PYTHON" "$HERE" "$PROTO" "$CACHE" "$RUN0" "$RUNSR" "$LOG"
echo "LAUNCHED screen=$SESSION"
echo "CACHE=$CACHE"
echo "RUN_SR0=$RUN0"
echo "RUN_SR005=$RUNSR"
echo "LOG=$LOG"
echo "Watch: tail -f '$LOG'"
echo "Inspect: screen -ls"
