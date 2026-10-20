#!/bin/bash
#SBATCH --partition=norm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=12:00:00
#SBATCH --job-name=locale-sra4571-merge
#SBATCH --output=/data/synkrp/locale/slurm_logs/sra4571_merge_%j.out
#SBATCH --error=/data/synkrp/locale/slurm_logs/sra4571_merge_%j.err

# Step 2 of 3 on Biowulf: merge the shards written by the
# biowulf_sra4571_embed.sbatch array into one embeddings.fbin and mark the
# index .done. Pure CPU + /data I/O (8 concurrent raw byte-range copies, no
# RAM to speak of). Resumable: completed shard copies are recorded in
# .merge_progress. ~3.2 TB to copy; 12 h covers ~75 MB/s.
#
# Usage: [CONFIG=...] sbatch biowulf_sra4571_merge.sbatch [NUM_SHARDS]
# NUM_SHARDS must equal the embed array's shard count (default 16). The run
# refuses to start unless every shard_<r>/ is .done.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/data/$USER/.cache/uv
export UV_PYTHON_INSTALL_DIR=/data/$USER/uv/python
export UV_TOOL_DIR=/data/$USER/uv/tools
export UV_PROJECT_ENVIRONMENT=/data/$USER/venvs/locale
export PYTHONUNBUFFERED=1

cd /home/synkrp/locale/benchmark

CONFIG=${CONFIG:-configs/sra4571/biowulf_locale_sra4571_endwin0_minlen100.yaml}
NUM_SHARDS=${1:-${NUM_SHARDS:-16}}
echo "config: $CONFIG  shards: $NUM_SHARDS"
checkquota 2>/dev/null | grep -E '^Mount|^/data' || true  # the merge adds ~3.2 TB

uv run --no-sync python run_benchmark.py --config "$CONFIG" \
  --stage merge --num_shards "$NUM_SHARDS"
