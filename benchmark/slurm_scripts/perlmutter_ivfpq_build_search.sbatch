#!/bin/bash
#SBATCH -A m5408_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -o slurm_logs/ivfpq_build_search_%j.out
#SBATCH -e slurm_logs/ivfpq_build_search_%j.err

# One-node GPU IVF-PQ build + timed search for a dataset whose index shards fit
# on this node (sra50 / sra55viral / sra500). Step 1 builds the config's
# ivfpq_num_shards shards, one srun task per GPU (skipped when present);
# step 2 runs run_benchmark --do_timing at every rate (5 searches, avg_time =
# mean of runs 2-5, query embedding included) into results/<dataset>_timing/.
#   sbatch -J ivfpq-sra50 slurm_scripts/perlmutter_ivfpq_build_search.sbatch configs/sra50/perlmutter_locale_sra50v2_ivfpq.yaml
CONFIG="${1:?usage: sbatch perlmutter_ivfpq_build_search.sbatch <config.yaml>}"
set -euo pipefail
cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1
SHARDS=$(grep -E "^\s*ivfpq_num_shards:" "$CONFIG" | awk '{print $2}')
RESULTS=$(grep -E "^results_dir:" "$CONFIG" | awk '{print $2}')_timing
read -r -a RATE_LIST <<<"${RATES:-0.0 0.05 0.1}"
echo "=== build $SHARDS shard(s) ($(date))"
srun --unbuffered -N1 -n "$SHARDS" --gpus-per-task=1 --cpus-per-task=$((128 / SHARDS)) \
  uv run python build_ivfpq.py --config "$CONFIG"
for rate in "${RATE_LIST[@]}"; do
    echo "=== search rate $rate ($(date))"
    srun --unbuffered -N1 -n1 --gpus-per-node=4 --cpus-per-task=128 \
      /usr/bin/env SLURM_NNODES=1 SLURM_NODEID=0 SLURM_PROCID=0 SLURM_NTASKS=1 \
      uv run python run_benchmark.py --config "$CONFIG" --mutation_rate "$rate" \
        --do_timing true --results_dir "$RESULTS"
done
echo "=== done ($(date))"
