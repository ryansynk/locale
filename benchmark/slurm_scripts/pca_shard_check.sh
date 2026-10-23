#!/bin/bash
# Check the sharded PCA build against a single-node build of the same index
# (sra500, PCA 128-d bf16), then search the sharded one at every noise level.
# Needs an allocation of >= 2 GPU nodes; run it inside it, or from a login
# node with SLURM_JOB_ID set to it (every step is srun --overlap):
#
#   salloc -A m5408_g -C gpu -q interactive -N 2 --gpus-per-node=4 -t 1:00:00
#   bash slurm_scripts/pca_shard_check.sh
#
#   1. 1-node build  -> indexes/sra500/locale@8vqiabk9/pca-d128-bf16-1node
#   2. 2-node build  -> indexes/sra500/locale@8vqiabk9/pca-d128-bf16
#   3. slurm_scripts/pca_shard_check.py on the two (exits non-zero on mismatch)
#   4. 2-node search of pca-d128-bf16 at 0.0 0.05 0.1
#
# Both index dirs are pre-created with 32-way striping (as sra4571's is) so
# the output's writes spread like they will there; the sra500 input itself is
# single-stripe. Rank ids come from SLURM_PROCID / SLURM_NTASKS (one task per
# node), which count the step's nodes, not the job's.
# Logs: slurm_logs/pca_shard_check_<step>.log
set -euo pipefail
: "${SLURM_JOB_ID:?run inside an allocation, or set SLURM_JOB_ID to one}"
cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1
mapfile -t NODES < <(scontrol show hostnames "$(squeue -h -j "$SLURM_JOB_ID" -o %N)")
((${#NODES[@]} >= 2)) || { echo "need >= 2 nodes, job has ${#NODES[@]}" >&2; exit 1; }
ONE=configs/sra500/perlmutter_locale_sra500v2_pca_d128_bf16_1node.yaml
TWO=configs/sra500/perlmutter_locale_sra500v2_pca_d128_bf16.yaml
IDX=indexes/sra500/locale@8vqiabk9
STEP=(srun --jobid "$SLURM_JOB_ID" --overlap --unbuffered --ntasks-per-node=1
    --gpus-per-node=4 --cpus-per-task=128)
SHARDED=(bash -c 'exec "$@" --shard "$SLURM_PROCID" --num_shards "$SLURM_NTASKS"' _)
RUN=(uv run --no-sync python scripts/run_benchmark.py)

for label in pca-d128-bf16-1node pca-d128-bf16; do
    if [[ ! -d "$IDX/$label" ]]; then
        mkdir -p "$IDX/$label"
        lfs setstripe -c 32 "$IDX/$label"
    fi
done

step() {  # <name> <srun args...>: run, log, time
    local name=$1 log=slurm_logs/pca_shard_check_$1.log t0=$SECONDS
    shift
    echo "=== $name ($(date +%T)) -> $log"
    "$@" >"$log" 2>&1 || { echo "FAILED: $name (see $log)"; tail -5 "$log"; exit 1; }
    echo "    done in $((SECONDS - t0))s"
}

step build_1node "${STEP[@]}" -N1 -n1 -w "${NODES[0]}" \
    "${RUN[@]}" --config "$ONE" --stage engine
step build_2node "${STEP[@]}" -N2 -n2 -w "${NODES[0]},${NODES[1]}" \
    "${SHARDED[@]}" "${RUN[@]}" --config "$TWO" --stage engine
step compare "${STEP[@]}" -N1 -n1 -w "${NODES[0]}" \
    uv run --no-sync python slurm_scripts/pca_shard_check.py \
    "$IDX/pca-d128-bf16-1node" "$IDX/pca-d128-bf16"
grep -E "^(projector|values|PASS|FAIL)" slurm_logs/pca_shard_check_compare.log
for rate in 0.0 0.05 0.1; do
    step "search_mut$rate" "${STEP[@]}" -N2 -n2 -w "${NODES[0]},${NODES[1]}" \
        "${SHARDED[@]}" "${RUN[@]}" --config "$TWO" --mutation_rate "$rate" --stage search
done
echo "=== all done ($(date +%T))"
