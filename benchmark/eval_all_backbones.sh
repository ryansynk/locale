#!/usr/bin/env bash
#
# Run the Table 3 eval for every backbone ladder: each of nt50m, hyenadna and
# dna2vec, each of the four augmentation rungs, each at eval mutation rate
# 0.0 / 0.05 / 0.1.
#
# Run it inside an allocation holding every GPU you want used -- the index
# build and the search both fan out over torch.cuda.device_count() on their own.
#
#   ./eval_all_backbones.sh                  # all three backbones
#   ./eval_all_backbones.sh dna2vec          # one backbone
#   RUNGS="none heavy" ./eval_all_backbones.sh
#
# Per rung the 0.0 rate is run first because it is what builds the index; the
# other two reuse it (the index depends on the checkpoint, not on the mutation
# rate). After the build the index is checked for the full accession count --
# a build that loses a worker still gets its .done marker, and every later run
# then loads a truncated index and reports quietly wrong recall.
#
# Rungs that have not been trained yet are skipped rather than failed, so this
# is safe to run against a partially finished ladder.
set -euo pipefail

BENCHMARK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK"

BACKBONES=("$@")
if [ ${#BACKBONES[@]} -eq 0 ]; then
    BACKBONES=(nt50m hyenadna dna2vec)
fi
read -r -a RUNG_LIST <<<"${RUNGS:-none light medium heavy}"
read -r -a RATE_LIST <<<"${RATES:-0.0 0.05 0.1}"
EXPECTED_ACCESSIONS="${EXPECTED_ACCESSIONS:-47}"

skipped=()
failed=()

yaml_get() {
    # yaml_get FILE dotted.key -- prints nothing if the key is absent.
    uv run python -c '
import sys, yaml
node = yaml.safe_load(open(sys.argv[1]))
for part in sys.argv[2].split("."):
    if not isinstance(node, dict) or part not in node:
        sys.exit(0)
    node = node[part]
print(node if node is not None else "")
' "$1" "$2"
}

# meta.parquet holds one row per accession in the index. Anything short of the
# full set means the build dropped accessions.
check_index_complete() {
    local index_dir="$1" run_id="$2" step="$3" config_tag="$4"
    local meta="${index_dir}/locale/${run_id}/${step}/${config_tag}/meta.parquet"
    if [ ! -f "$meta" ]; then
        echo "[warn] no $meta to verify" >&2
        return 0
    fi
    local n
    n="$(uv run python -c '
import sys, polars as pl
print(pl.read_parquet(sys.argv[1])["srr_id"].n_unique())
' "$meta")"
    if [ "$n" -ne "$EXPECTED_ACCESSIONS" ]; then
        echo "[FAIL] index at $meta has $n accessions, expected $EXPECTED_ACCESSIONS." >&2
        echo "       A worker was probably lost. Delete the index dir to force a rebuild." >&2
        return 1
    fi
    echo "[ok] index has $n accessions"
}

for backbone in "${BACKBONES[@]}"; do
    runs_yaml="runs_${backbone}.yaml"
    if [ ! -f "$runs_yaml" ]; then
        echo "No $runs_yaml -- unknown backbone '$backbone'." >&2
        exit 1
    fi
    step="$(yaml_get "$runs_yaml" step)"

    for rung in "${RUNG_LIST[@]}"; do
        config="configs/${backbone}_${rung}.yaml"
        label="${backbone}/${rung}"

        run_id="$(yaml_get "$runs_yaml" "runs.${rung}")"
        if [ -z "$run_id" ]; then
            echo "[skip] $label not trained yet (no run id in $runs_yaml)"
            skipped+=("$label")
            continue
        fi

        ckpt="$(yaml_get "$config" model.checkpoint_path)"
        if [ ! -f "$ckpt" ]; then
            echo "[skip] $label checkpoint missing: $ckpt"
            skipped+=("$label")
            continue
        fi

        index_dir="$(yaml_get "$config" index_dir)"
        max_len="$(yaml_get "$config" model.max_seq_len)"
        pooling="$(yaml_get "$config" model.pooling)"
        config_tag="maxlen${max_len}_pool${pooling}_chunkstride"

        for rate in "${RATE_LIST[@]}"; do
            echo "=== $label @ mutation_rate $rate ==="
            if ! uv run python run_benchmark.py --config "$config" --mutation_rate "$rate"; then
                echo "[FAIL] $label @ $rate" >&2
                failed+=("$label @ $rate")
                break
            fi
            if [ "$rate" = "${RATE_LIST[0]}" ]; then
                if ! check_index_complete "$index_dir" "$run_id" "$step" "$config_tag"; then
                    failed+=("$label (truncated index)")
                    break
                fi
            fi
        done
    done

    echo
    echo "=== Table 3, $backbone ==="
    results_dir="$(yaml_get "configs/${backbone}_heavy.yaml" results_dir)"
    queries="$(yaml_get "configs/${backbone}_heavy.yaml" dataset_dir)/queries.parquet"
    uv run python make_table3.py --results_dir "$results_dir" \
        --queries "$queries" --runs "$runs_yaml" || true
done

echo
if [ ${#skipped[@]} -gt 0 ]; then
    echo "Skipped (not trained): ${skipped[*]}"
fi
if [ ${#failed[@]} -gt 0 ]; then
    echo "Failed: ${failed[*]}" >&2
    exit 1
fi
