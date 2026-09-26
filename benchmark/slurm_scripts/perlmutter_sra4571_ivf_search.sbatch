#!/bin/bash
#SBATCH -A m5408
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -c 256
#SBATCH -J locale-sra4571-ivf-search
#SBATCH -o slurm_logs/sra4571_ivf_search_%j.out
#SBATCH -e slurm_logs/sra4571_ivf_search_%j.err

# Single-CPU-node IVF-RaBitQ search on sra4571 (index built by
# perlmutter_sra4571_ivf_build.sbatch). The first run merges the 8 build
# shards into ivf<nlist>_rabitq1_fs.faiss (~230 GB, needs this node's RAM).
#
# Default: ivf_sweep.py over nprobe x rerank at every rate with the index
# loaded once (results -> results/sra4571/<experiment_id>/, hits ->
# results/sra4571_topk_hits/). With MODE=timing it instead runs
# run_benchmark --do_timing per rate for the config as given (5 searches,
# mean of runs 2-5, query embedding on this node's CPUs included) into
# results/sra4571_timing/, beside the metagraph timing.
#   sbatch slurm_scripts/perlmutter_sra4571_ivf_search.sbatch
#   MODE=timing sbatch slurm_scripts/perlmutter_sra4571_ivf_search.sbatch
CONFIG="${CONFIG:-configs/sra4571/perlmutter_locale_sra4571_ivf.yaml}"
set -euo pipefail
cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1
read -r -a RATE_LIST <<<"${RATES:-0.0 0.05 0.1}"
if [ "${MODE:-sweep}" = timing ]; then
    for rate in "${RATE_LIST[@]}"; do
        uv run python run_benchmark.py --config "$CONFIG" --mutation_rate "$rate" \
            --do_timing true --results_dir results/sra4571_timing ${EXTRA:-}
    done
else
    uv run python ivf_sweep.py --config "$CONFIG" \
        --nprobes "${NPROBES:-128,256,512,1024}" --reranks "${RERANKS:-300}" \
        --rates "$(echo "${RATE_LIST[@]}" | tr ' ' ,)" ${EXTRA:-}
fi
