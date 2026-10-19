#!/bin/bash
# Complete the frozen five-split table on the CURRENT (abliterated) backbone.
# Read-only archive replay: no retrain, no re-extraction, no production change.
set -uo pipefail
JEV_DATA=${JEV_DATA:-$HOME/jev-data}
JEV_PY=${JEV_PY:-python3}
cd "$JEV_DATA/jev-abliterated-eval-20261018"

ID="$JEV_DATA/jev-abliterated-eval-20261018/runtime-identity.run-20261018.json"
DEADLINE='2026-10-19 00:30:00'
for split in ood cal; do
  echo "STAGE $split start $(date '+%F %T')"
  "$PY" eval_head_abliterated.py --split "$split" --out "rows-$split" \
      --identity "$ID" --base http://127.0.0.1:8100 --confirm-window --deadline "$DEADLINE"
  echo "STAGE $split rc=$? $(date '+%F %T')"
  "$PY" finalize_rows.py --split "$split" --rows "rows-$split.jsonl" \
      --summary "rows-$split-summary.json" --identity "$ID"
done
echo "ALL DONE $(date '+%F %T')"
