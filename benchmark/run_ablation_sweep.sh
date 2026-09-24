#!/usr/bin/env bash
#
# Evaluate the seven ablation-table checkpoints (configs/ablation_sweep/) on
# sra50v2 at every eval mutation rate, one multi-node srun step at a time, then
# print the recall tables. Run it from inside a multi-node GPU allocation, e.g.
#
#   salloc -A m5408_g -C gpu -q interactive -N 4 -t 4:00:00 \
#          --ntasks-per-node=1 --gpus-per-node=4 -c 128
#   cd /pscratch/sd/r/rsynk/locale/benchmark && ./run_ablation_sweep.sh
#
# Same mechanics as run_backbone_sweep.sh: each srun step launches one
# run_benchmark.py per node, which builds one index shard per node and has
# rank 0 merge; plain `python run_benchmark.py` inside a multi-node allocation
# would wait forever for peers. Per config the first rate builds the index and
# later rates reuse it, so steps run sequentially.
#
# data_reference is the paper checkpoint (8vqiabk9), which already has a
# finished sra50v2 index under indexes/sra50/. It is symlinked into this
# sweep's index tree so that row skips the build.
#
#   ./run_ablation_sweep.sh                         # all rows, all rates
#   ./run_ablation_sweep.sh crop_overlap crop_both  # a subset of rows
#   RATES="0.1" ./run_ablation_sweep.sh mut_none
#
# Override GPUS_PER_NODE / CPUS_PER_TASK if the allocation is smaller.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Not inside a Slurm allocation; run salloc first." >&2
    exit 1
fi

CONFIG_DIR=configs/ablation_sweep
INDEX_DIR=indexes/sra50_ablation_sweep
RESULTS_DIR=results/sra50_ablation_sweep
BUNDLE=/pscratch/sd/r/rsynk/locale-data/constructed/sra50/bundle

ROWS=("$@")
if [ ${#ROWS[@]} -eq 0 ]; then
    # Table order.
    ROWS=(baseline_heavy_contain_logan crop_overlap crop_both data_reference
          mut_none mut_light mut_medium)
fi
read -r -a RATE_LIST <<<"${RATES:-0.0 0.05 0.1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-128}"

export PYTHONUNBUFFERED=1

# Reuse the paper checkpoint's finished sra50v2 index rather than rebuilding it.
if [ -e indexes/sra50/locale/8vqiabk9 ] && [ ! -e "$INDEX_DIR/locale/8vqiabk9" ]; then
    mkdir -p "$INDEX_DIR/locale"
    ln -s ../../sra50/locale/8vqiabk9 "$INDEX_DIR/locale/8vqiabk9"
    echo "[link] $INDEX_DIR/locale/8vqiabk9 -> indexes/sra50/locale/8vqiabk9"
fi

failed=()
for row in "${ROWS[@]}"; do
    cfg="$CONFIG_DIR/$row.yaml"
    if [ ! -f "$cfg" ]; then
        echo "[skip] no config $cfg"
        continue
    fi
    for rate in "${RATE_LIST[@]}"; do
        echo "=== $cfg @ mutation_rate $rate  ($(date))  nodes=${SLURM_JOB_NUM_NODES:-?} ==="
        if ! srun --unbuffered --ntasks-per-node=1 \
                  --gpus-per-node="$GPUS_PER_NODE" \
                  --cpus-per-task="$CPUS_PER_TASK" \
                  uv run python run_benchmark.py \
                    --config "$cfg" --mutation_rate "$rate"; then
            echo "[FAIL] $cfg @ $rate" >&2
            failed+=("$cfg @ $rate")
            # Later rates need the index this rate builds, so move on to the
            # next row rather than retrying against a broken build.
            break
        fi
    done
done

echo
if [ ${#failed[@]} -gt 0 ]; then
    echo "Failed: ${failed[*]}" >&2
    exit 1
fi

echo "All ablation sweep runs finished. Recall tables (R-precision = Recall@R_q):"
uv run python print_results.py "$RESULTS_DIR" "$BUNDLE/queries_mut0.00.parquet" "$BUNDLE/accs.txt" \
    --full_model_names true || true
