#!/usr/bin/env bash
#
# Run the ESA (dna2vec, untrained) baseline over the 500-accession sra500
# benchmark at eval mutation rates 0.0 / 0.05 / 0.1, matching the rest of the
# benchmark.
#
# Run it inside an allocation holding every GPU you want used -- the index build
# and the search both fan out over torch.cuda.device_count() on their own.
#
#   ./eval_esa_sra500.sh                # the three rates below
#   RATES="0.0 0.1" ./eval_esa_sra500.sh
#
# Keep the rates at or below ~0.5. mutation_rate is applied as
# Augmenter.augment(identity=1-rate), so 1.0 means zero identity to the original
# read -- the query is randomised outright and recall measures nothing but
# chance.
#
# The 0.0 rate runs first because it is what builds the index; the other two
# reuse it (the index depends on the model, not on the mutation rate). After the
# build the index is checked for all 500 accessions -- a build that loses a
# worker still gets its .done marker, and every later run then loads a truncated
# index and reports quietly wrong recall.
set -euo pipefail

BENCHMARK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK"

CONFIG="configs/nexus_esa_sra500.yaml"
read -r -a RATE_LIST <<<"${RATES:-0.0 0.05 0.1}"
EXPECTED_ACCESSIONS="${EXPECTED_ACCESSIONS:-500}"

INDEX_PATH="/fs/nexus-projects/sra_search/indexes_sra500/dna2vec/maxlen256_poolmax_chunkstride"
RESULTS_DIR="results_sra500"
DATASET_DIR="/fs/nexus-projects/sra_search/hf/sra500"

check_index_complete() {
    local meta="${INDEX_PATH}/meta.parquet"
    if [ ! -f "$meta" ]; then
        echo "[check] FAILED: no ${meta}" >&2
        return 1
    fi
    uv run python -c '
import sys
import polars as pl
n = pl.read_parquet(sys.argv[1]).select("srr_id").n_unique()
expected = int(sys.argv[2])
if n != expected:
    sys.exit(f"[check] FAILED: index holds {n} accessions, expected {expected}")
print(f"[check] index holds {n} accessions")
' "$meta" "$EXPECTED_ACCESSIONS"
}

for rate in "${RATE_LIST[@]}"; do
    echo "=== ESA / sra500 / mutation_rate ${rate} ==="
    uv run python run_benchmark.py --config "$CONFIG" --mutation_rate "$rate"
    check_index_complete
done

echo "=== results ==="
# print_results takes these positionally, not as flags.
uv run python print_results.py \
    "$RESULTS_DIR" \
    "${DATASET_DIR}/queries.parquet" \
    "${DATASET_DIR}/accs.txt"
