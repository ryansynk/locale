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

# Step 2 of 3: merge the per-shard outputs written by the perlmutter_sra4571.sbatch array
# into one embeddings.fbin and mark the index .done. Pure CPU + Lustre I/O, no
# GPU. Resumable: completed shard copies are recorded in .merge_progress.
# The 32-shard 100studies merge ran ~3h at ~750 MB/s.
#
# Usage: [CONFIG=...] sbatch perlmutter_sra4571_merge.sbatch [NUM_SHARDS]
# NUM_SHARDS must equal NUM_SHARDS of the embed array (default 32). The index
# directory is derived from the config (index_dir/encoder_label) and striped
# before the merged file is created. Default config: the end-window +
# min_seq_len index. The original locale@8vqiabk9 index is already merged;
# submitting with its config exits at once (job 59995872 was such a no-op).

set -euo pipefail

cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1

CONFIG=${CONFIG:-configs/sra4571/perlmutter_locale_sra4571_endwin0_minlen100.yaml}
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG"; exit 1; }
NUM_SHARDS=${1:-32}
INDEX_PATH=$(uv run python -c "
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(c['index_dir'].rstrip('/') + '/' + c['model']['encoder_label'])" "$CONFIG")
echo "config: $CONFIG  index: $INDEX_PATH  shards: $NUM_SHARDS"
[ ! -f "$INDEX_PATH/.done" ] || { echo "$INDEX_PATH is already merged and .done; nothing to do"; exit 1; }
n_done=$(ls -d "$INDEX_PATH"/shard_*/.done 2>/dev/null | wc -l)
echo "shards .done: $n_done of $NUM_SHARDS"
[ "$n_done" -eq "$NUM_SHARDS" ] || { echo "not every shard is .done; run the embed array first"; exit 1; }

# Stripe the merged file across OSTs before it is created; a stripe-1 file
# capped the first 100studies merge at ~160 MB/s and would cap search reads.
lfs setstripe -c 32 -S 16M "$INDEX_PATH" || true

uv run python run_benchmark.py --config "$CONFIG" \
  --stage merge --num_shards "$NUM_SHARDS"
