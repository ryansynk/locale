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
# index.num_shards shards, one srun task per GPU (skipped when present);
# step 2 runs run_benchmark --do_timing at every rate (5 searches, avg_time =
# mean of runs 2-5, query embedding included), written as mut<rate>.timing.parquet
# beside the accuracy results.
#   sbatch -J ivfpq-sra50 slurm_scripts/perlmutter_ivfpq_build_search.sbatch configs/sra50/perlmutter_locale_sra50v2_ivfpq.yaml
CONFIG="${1:?usage: sbatch perlmutter_ivfpq_build_search.sbatch <config.yaml>}"
set -euo pipefail
cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1
SHARDS=$(grep -E "^\s*num_shards:" "$CONFIG" | awk '{print $2}')
read -r -a RATE_LIST <<<"${RATES:-0.0 0.05 0.1}"
echo "=== build $SHARDS shard(s) ($(date))"
srun --unbuffered -N1 -n "$SHARDS" --gpus-per-task=1 --cpus-per-task=$((128 / SHARDS)) \
  bash -c 'exec "$@" --shard "$SLURM_PROCID" --num_shards "$SLURM_NTASKS"' _ \
  uv run python run_benchmark.py --config "$CONFIG" --stage engine
for rate in "${RATE_LIST[@]}"; do
    echo "=== search rate $rate ($(date))"
    srun --unbuffered -N1 -n1 --gpus-per-node=4 --cpus-per-task=128 \
      uv run python run_benchmark.py --config "$CONFIG" --mutation_rate "$rate" \
        --stage search --num_shards 1 --do_timing true
done
echo "=== done ($(date))"
