#!/bin/bash
# sra4571 PCA 768 -> 128 bf16 (default: the endwin0-minlen100 index, see CONFIG below)
# on every node of an existing allocation: a sharded build (stage engine;
# rank 0 fits the projector, each node projects 1/N of the rows into one
# bf16 file, src/pca.py; 2.06 B rows -> 527 GB took ~12 min on 4 nodes), then a sharded top-100 search at every noise
# level, 0.0 0.05 0.1. Run it inside the allocation, or from a login node with
# SLURM_JOB_ID set to it (every step is srun --overlap):
#
#   salloc -A m5408_g -C gpu -q interactive -N 4 --gpus-per-node=4 -t 1:00:00
#   bash slurm_scripts/sra4571_pca_bf16.sh
#
# Resumable: a finished build is skipped (its .done) and a node whose rows are
# written skips them (shard_rank_<r>.json), as long as the node count is the
# same; a finished search reuses its saved hits. Rank ids come from
# SLURM_PROCID / SLURM_NTASKS (one task per node), which count the step's
# nodes. Logs: slurm_logs/sra4571_pca_d128_bf16_<step>.log
set -euo pipefail
: "${SLURM_JOB_ID:?run inside an allocation, or set SLURM_JOB_ID to one}"
cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1
# CONFIG=... (on the same line as the command, or exported) picks another PCA
# config over another encoder index. Default: the end-window + min_seq_len
# index; the original locale@8vqiabk9 PCA engine is already built, so
# configs/sra4571/perlmutter_locale_sra4571_pca_d128_bf16.yaml only re-searches.
CONFIG=${CONFIG:-configs/sra4571/perlmutter_locale_sra4571_endwin0_minlen100_pca_d128_bf16.yaml}
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG"; exit 1; }
echo "config: $CONFIG"
mapfile -t NODES < <(scontrol show hostnames "$(squeue -h -j "$SLURM_JOB_ID" -o %N)")
NODELIST=$(IFS=,; echo "${NODES[*]}")
STEP=(srun --jobid "$SLURM_JOB_ID" --overlap --unbuffered -N "${#NODES[@]}"
    -n "${#NODES[@]}" --ntasks-per-node=1 -w "$NODELIST"
    --gpus-per-node=4 --cpus-per-task=128)
SHARDED=(bash -c 'exec "$@" --shard "$SLURM_PROCID" --num_shards "$SLURM_NTASKS"' _)
RUN=(uv run --no-sync python scripts/run_benchmark.py --config "$CONFIG")

echo "=== ${#NODES[@]} node(s): ${NODES[*]}"
showquota 2>/dev/null | grep pscratch || true  # the build adds rows x 128 x 2 B (~213 GB for 830 M rows)

step() {  # <name> <args...>: run on every node, log, time
    local name=$1 log=slurm_logs/sra4571_pca_d128_bf16_$1.log t0=$SECONDS
    shift
    echo "=== $name ($(date +%T)) -> $log"
    "${STEP[@]}" "${SHARDED[@]}" "${RUN[@]}" "$@" >"$log" 2>&1 ||
        { echo "FAILED: $name (see $log)"; tail -5 "$log"; exit 1; }
    echo "    done in $((SECONDS - t0))s"
}

step build --stage engine
for rate in 0.0 0.05 0.1; do
    step "search_mut$rate" --stage search --mutation_rate "$rate"
done
echo "=== all done ($(date +%T)); results under results/sra4571/ for $CONFIG"
