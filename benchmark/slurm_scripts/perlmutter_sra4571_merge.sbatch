#!/bin/bash
#SBATCH -A m5408
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 6:00:00
#SBATCH -c 128
#SBATCH -J locale-sra4571-merge
#SBATCH -o slurm_logs/sra4571_merge_%j.out
#SBATCH -e slurm_logs/sra4571_merge_%j.err

# Step 2 of 3: merge the per-node shards written by perlmutter_sra4571.sbatch
# into one embeddings.fbin and mark the index .done. Pure CPU + Lustre I/O, no
# GPU. Resumable: completed shard copies are recorded in .merge_progress.
# The 32-shard 100studies merge ran ~3h at ~750 MB/s.
#
# Usage: [CONFIG=... INDEX_PATH=...] sbatch perlmutter_sra4571_merge.sbatch [NUM_SHARDS]
# NUM_SHARDS must equal the -N of the embed job (default 32). CONFIG and
# INDEX_PATH default to the original sra4571 index; set both for another
# encoder (INDEX_PATH is indexes/sra4571/<encoder_label>, striped before the
# merged file is created).

set -euo pipefail

cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1

CONFIG=${CONFIG:-configs/sra4571/perlmutter_locale_sra4571.yaml}
INDEX_PATH=${INDEX_PATH:-indexes/sra4571/locale@8vqiabk9}
NUM_SHARDS=${1:-32}
echo "config: $CONFIG  index: $INDEX_PATH  shards: $NUM_SHARDS"

# Stripe the merged file across OSTs before it is created; a stripe-1 file
# capped the first 100studies merge at ~160 MB/s and would cap search reads.
lfs setstripe -c 32 -S 16M "$INDEX_PATH" || true

uv run python run_benchmark.py --config "$CONFIG" \
  --stage merge --num_shards "$NUM_SHARDS"
