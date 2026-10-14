#!/bin/bash
# Contig-length cutoff sweep on sra500 (endwin50 index): one exact scan per
# noise level, every cutoff at once, sharded over every node of an existing
# GPU allocation (same srun pattern as benchmark/slurm_scripts/sra500_endwin.sh):
#
#   salloc -A m5408_g -C gpu -q interactive -N 4 --gpus-per-node=4 -t 3:00:00
#   bash experiments/seqlen_cutoff/sweep.sh
#
# Needs the length map first (CPU node):
#   experiments/seqlen_cutoff/vector_contig_lengths.py -> sra500_endwin50_contig_len.u32
#
# Override CUTOFFS / RATES / TOPK in the environment, e.g.
#   CUTOFFS="[0,45,77,100]" RATES="[0.0]" bash experiments/seqlen_cutoff/sweep.sh
# Results: benchmark/results/sra500_minlen/<encoder>/exact-minlen<c>/top100/mut<rate>.parquet
# Logs:    benchmark/slurm_logs/seqlen_sweep.log
set -euo pipefail
: "${SLURM_JOB_ID:?run inside an allocation, or set SLURM_JOB_ID to one}"
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../../benchmark"
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONFIG=${CONFIG:-configs/sra500/perlmutter_locale_sra500v2_endwin50.yaml}
LENGTHS=${LENGTHS:-$HERE/sra500_endwin50_contig_len.u32}
CUTOFFS=${CUTOFFS:-"[0,45,64,77,100,128,150,200,256]"}
RATES=${RATES:-"[0.0,0.05,0.1]"}
TOPK=${TOPK:-100}
[ -f "$LENGTHS" ] || { echo "length map missing: $LENGTHS" >&2; exit 1; }

mapfile -t NODES < <(scontrol show hostnames "$(squeue -h -j "$SLURM_JOB_ID" -o %N)")
NODELIST=$(IFS=,; echo "${NODES[*]}")
echo "=== ${#NODES[@]} node(s): ${NODES[*]}"
log=slurm_logs/seqlen_sweep.log
t0=$SECONDS
srun --jobid "$SLURM_JOB_ID" --overlap --unbuffered -N "${#NODES[@]}" \
    -n "${#NODES[@]}" --ntasks-per-node=1 -w "$NODELIST" \
    --gpus-per-node=4 --cpus-per-task=128 \
    bash -c 'exec "$@" --shard "$SLURM_PROCID" --num_shards "$SLURM_NTASKS"' _ \
    uv run --no-sync python "$HERE/sweep_search.py" --bench_config "$CONFIG" \
    --lengths "$LENGTHS" --cutoffs "$CUTOFFS" --rates "$RATES" --top_k "$TOPK" \
    >"$log" 2>&1 || { echo "FAILED (see $log)"; tail -20 "$log"; exit 1; }
echo "=== done in $((SECONDS - t0))s; results in results/sra500_minlen/"
