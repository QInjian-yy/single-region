#!/usr/bin/env bash
set -u
cd "$PROJECT" || exit 1
"$TRAIN_PYTHON" -u -B train_e2e_camil.py --config "$CONFIG" --protocol-dir "$PROTOCOL" --device cuda:0 --output "$OUTPUT" > "$LOG" 2>&1
code=$?
echo "TRAIN_EXIT_CODE=$code" >> "$LOG"
exit "$code"
