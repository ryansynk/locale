#!/usr/bin/env bash
# Full SW audit for the three most recent sra500 indexes; builds missing sidecars.
set -euo pipefail
cd /pscratch/sd/r/rsynk/locale
for label in locale@pc6t1x8j-proj-w128s64-endwin30-minlen100 locale@pc6t1x8j-proj-endwin50-minlen100 locale@8vqiabk9-w128s64-endwin30-minlen100; do
  echo "##### $label  $(date)"
  .venv/bin/python experiments/topk_alignment/sw_audit.py run \
    --index_dir benchmark/indexes/sra500/$label \
    --results_dir benchmark/results/sra500/$label/exact/top100 \
    --out_dir experiments/topk_alignment --workers 120
done
echo "##### done $(date)"
