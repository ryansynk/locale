#!/bin/bash
#SBATCH -A m5408_g
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -J locale-sra4571-ivfpq-search
#SBATCH -o slurm_logs/sra4571_ivfpq_search_%j.out
#SBATCH -e slurm_logs/sra4571_ivfpq_search_%j.err

# Single-node GPU IVF-PQ search on sra4571: the 16 shards built by
# perlmutter_sra4571_ivfpq_build.sbatch (same CONFIG) are loaded 4 per
# A100-80GB (~77 GiB each) by one worker process per GPU, which also embeds
# its slice of the queries. Default CONFIG is the metagraph-parity setting.
#
# MODE=timing (default): run_benchmark --do_timing per rate for the config
# as given -- 5 full searches (embed + scan + optional rerank + regroup),
# avg_time = mean of runs 2-5 -- into results/sra4571_timing/, beside
# metagraph_k31{,_p64}. MODE=sweep: ivf_sweep.py over NPROBES x RERANKS with
# the shards loaded once (results -> results/sra4571/, hits ->
# results/sra4571_topk_hits/).
#   sbatch slurm_scripts/perlmutter_sra4571_ivfpq_search.sbatch
#   EXTRA="--model.ivfpq_nprobe 68 --model.ivfpq_rerank 10" sbatch ...
#   MODE=sweep NPROBES=136,272,544 RERANKS=0,10 sbatch ...
CONFIG="${CONFIG:-configs/sra4571/perlmutter_locale_sra4571_ivfpq_L16k.yaml}"
set -euo pipefail
cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1
read -r -a RATE_LIST <<<"${RATES:-0.0 0.05 0.1}"
if [ "${MODE:-timing}" = timing ]; then
    for rate in "${RATE_LIST[@]}"; do
        uv run python run_benchmark.py --config "$CONFIG" --mutation_rate "$rate" \
            --do_timing true --results_dir results/sra4571_timing ${EXTRA:-}
    done
else
    uv run python ivf_sweep.py --config "$CONFIG" \
        --nprobes "${NPROBES:-136,272,544}" --reranks "${RERANKS:-0,10}" \
        --rates "$(echo "${RATE_LIST[@]}" | tr ' ' ,)" ${EXTRA:-}
fi
