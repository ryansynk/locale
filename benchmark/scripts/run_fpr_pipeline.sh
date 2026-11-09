#!/usr/bin/env bash
#
# Post-search steps of the calibrated-FPR experiment for one negative set:
# calibrate cutoffs on the negatives, rescore the sra500 positives at each
# FPR level, print the metrics. Runs in seconds on a login node.
#
#   scripts/run_fpr_pipeline.sh <negative set name> [positive results dir]
#   scripts/run_fpr_pipeline.sh sra500_negatives2k
#
# Inputs: results/<neg set>/ (run_benchmark output for the negatives),
# locale-data/constructed/<neg set>/bundle, and the positives' results under
# results/sra500 (default). Every model with negative results is picked up;
# the positives tree may hold any number of other models.
# Outputs: results/<neg set>/calibration/{thresholds.json,thresholds.csv,tpr.csv,scores_mut*.png}
# and results/<neg set>_fpr/fpr<level>/<model>/ with print_results tables.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

NEG=${1:?negative set name, e.g. sra500_negatives2k}
POS_RESULTS=${2:-results/sra500}
NEG_RESULTS=results/$NEG
NEG_BUNDLE=/pscratch/sd/r/rsynk/locale-data/constructed/$NEG/bundle
POS_BUNDLE=/pscratch/sd/r/rsynk/locale-data/constructed/sra500/bundle
OUT=results/${NEG}_fpr
export POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-4}

uv run python scripts/calibrate_fpr.py \
    --negative_results_dir "$NEG_RESULTS" --negative_bundle "$NEG_BUNDLE" \
    --positive_results_dir "$POS_RESULTS" --positive_bundle "$POS_BUNDLE" \
    --index_dir indexes/sra500

rm -rf "$OUT"
uv run python scripts/apply_fpr_thresholds.py \
    --thresholds "$NEG_RESULTS/calibration/thresholds.json" \
    --positive_results_dir "$POS_RESULTS" --positive_bundle "$POS_BUNDLE" \
    --index_dir indexes/sra500 --output_root "$OUT"

for level in "$OUT"/fpr*; do
    echo
    echo "################ $level"
    uv run python scripts/print_results.py "$level" \
        "$POS_BUNDLE/queries_mut0.00.parquet" "$POS_BUNDLE/accs.txt" --bootstrap_samples 1000
done
