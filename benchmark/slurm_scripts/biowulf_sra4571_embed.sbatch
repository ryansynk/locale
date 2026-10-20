#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=220g
#SBATCH --time=6:00:00
#SBATCH --array=0-15
#SBATCH --job-name=locale-sra4571-embed
#SBATCH --output=/data/synkrp/locale/slurm_logs/sra4571_embed_%A_%a.out
#SBATCH --error=/data/synkrp/locale/slurm_logs/sra4571_embed_%A_%a.err

# Step 1 of 3 on Biowulf: embed the sra4571 dense locale index as a job array,
# one single-node 4x A100-80GB task per shard. Array task i builds shard i of
# NUM_SHARDS (default 16, the --array range above) under
# <index_dir>/<encoder_label>/shard_<i>/ and exits (--stage embed), so no GPU
# node is held for the serial merge. Biowulf's gpu partition does not allow
# multi-node GPU jobs, hence an array instead of Perlmutter's -N 32 srun; the
# per-user cap of 56 GPUs means at most 14 tasks run at once.
#
# Sizing: Perlmutter embedded the original 2.06 B-row index on 32 nodes of
# 4x A100-40GB in ~2.5 h (batch_size 1000). This index is ~1.1 B rows on 16
# nodes of 4x A100-80GB at batch_size 2000, so expect ~2.5-3 h per task plus
# /data I/O; 6 h is headroom (shorter limits backfill sooner). Then:
#   sbatch slurm_scripts/biowulf_sra4571_merge.sbatch          (CPU)
#   sbatch slurm_scripts/biowulf_sra4571_search.sbatch 0.00    (GPU array)
# or let slurm_scripts/biowulf_sra4571.sh chain all three.
#
# Shard striping depends on NUM_SHARDS: shard_* under the index directory must
# be empty on the first run at a new count, and a resubmission must keep the
# same NUM_SHARDS so completed shards (.done) are valid to reuse. To redo only
# some shards after a timeout:
#   sbatch --array=3,7 slurm_scripts/biowulf_sra4571_embed.sbatch
# (NUM_SHARDS stays 16; the task id is the shard, not the array size.)
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/data/$USER/.cache/uv
export UV_PYTHON_INSTALL_DIR=/data/$USER/uv/python
export UV_TOOL_DIR=/data/$USER/uv/tools
export UV_PROJECT_ENVIRONMENT=/data/$USER/venvs/locale
export PYTHONUNBUFFERED=1

cd /home/synkrp/locale/benchmark

CONFIG=${CONFIG:-configs/sra4571/biowulf_locale_sra4571_endwin0_minlen100.yaml}
NUM_SHARDS=${NUM_SHARDS:-16}
SHARD=${SLURM_ARRAY_TASK_ID:?run as a job array (--array=0-$((NUM_SHARDS - 1)))}
[ "$SHARD" -lt "$NUM_SHARDS" ] || { echo "array task $SHARD >= NUM_SHARDS=$NUM_SHARDS"; exit 1; }

echo "config: $CONFIG  shard: $SHARD / $NUM_SHARDS  node: $(hostname)  gpus: ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run --no-sync python run_benchmark.py --config "$CONFIG" \
  --stage embed --shard "$SHARD" --num_shards "$NUM_SHARDS"
