#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KIT="$PROJECT/C16_Seed123_Run_Kit"
cd "$PROJECT"
TRAIN_PYTHON="${TRAIN_PYTHON:-/home/ub/anaconda3/envs/clam_latest/bin/python}"
if [[ ! -x "$TRAIN_PYTHON" ]]; then echo "Python missing: $TRAIN_PYTHON"; exit 2; fi
if ! command -v screen >/dev/null; then echo "Please install GNU screen"; exit 2; fi
# Protocol data is supplied separately from the source repository.
PROTOCOL="${C16_PROTOCOL_DIR:-$PROJECT/data_protocols/camil_5fold_seed123_sourcestrat}"
CONFIG="${C16_CONFIG:-$KIT/camil_attn_adamw_seed123.yaml}"
if [[ ! -f "$PROTOCOL/protocol.json" || ! -f "$CONFIG" ]]; then
  echo "Provide an existing C16_PROTOCOL_DIR and C16_CONFIG: $PROTOCOL / $CONFIG"; exit 2
fi
"$TRAIN_PYTHON" "$KIT/verify_protocol.py" "$PROTOCOL"
# Verify original source manifests and all WSI count matches before starting.
"$TRAIN_PYTHON" -u -B train_e2e_camil.py --config "$CONFIG" --protocol-dir "$PROTOCOL" --check-data
mkdir -p runs logs
NAME="camil_seed123_attn_$(date +%Y%m%d_%H%M%S)"
OUTPUT="$PROJECT/runs/$NAME"
LOG="$PROJECT/logs/$NAME.log"
if [[ -e "$OUTPUT" || -e "$LOG" ]]; then echo "Output/log already exists; refusing to overwrite"; exit 2; fi
SESSION="camil_s123_$(date +%H%M%S)"
export PROJECT TRAIN_PYTHON CONFIG PROTOCOL OUTPUT LOG
screen -dmS "$SESSION" bash "$KIT/worker.sh"
echo "LAUNCHED screen=$SESSION"
echo "RUN=$OUTPUT"
echo "LOG=$LOG"
echo "Watch: tail -f '$LOG'"
echo "Inspect: screen -ls"
