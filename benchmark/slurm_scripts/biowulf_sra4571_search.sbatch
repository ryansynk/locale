#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=220g
#SBATCH --time=4:00:00
#SBATCH --array=0-3
#SBATCH --job-name=locale-sra4571-search
#SBATCH --output=/data/synkrp/locale/slurm_logs/sra4571_search_%A_%a.out
#SBATCH --error=/data/synkrp/locale/slurm_logs/sra4571_search_%A_%a.err

# Step 3 of 3 on Biowulf: sharded search over the merged sra4571 index as a
# 4-task job array (single-node 4x A100-80GB each). Array task r scans rows
# [r/4, (r+1)/4) of the fbin for every query's top-k and writes a partial;
# task 0 waits for the other partials (up to 2 h, run_benchmark._wait_for_files),
# unions them, regroups to accessions and writes the result parquet. The index
# must be .done (run the merge first). Partials are keyed by
# (mutation_rate, num_queries, seed) and reused on rerun, so if task 0 times
# out waiting, resubmit just it: sbatch --array=0 ... <rate> [config].
#
# Usage: sbatch biowulf_sra4571_search.sbatch <mutation_rate> [config.yaml]
#   mutation_rate: 0.00 / 0.05 / 0.10 (selects the pre-mutated query file)
#   config: defaults to the exact top-100 both-strands config; any config over
#           the same encoder index works (e.g. the PCA bf16 one once its
#           engine is built). NUM_SHARDS (default 4) must match the --array
#           range. The three rates write distinct files and may run at once.
#
# Sizing: on Perlmutter 4 nodes scanned the 2.06 B-row index in ~30 min; this
# one is ~1.1 B rows but reads ~800 GB per task from /data, so allow ~1-2 h.
MUT_RATE="${1:?usage: sbatch biowulf_sra4571_search.sbatch <mutation_rate> [config.yaml]}"
CONFIG="${2:-configs/sra4571/biowulf_locale_sra4571_endwin0_minlen100.yaml}"

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/data/$USER/.cache/uv
export UV_PYTHON_INSTALL_DIR=/data/$USER/uv/python
export UV_TOOL_DIR=/data/$USER/uv/tools
export UV_PROJECT_ENVIRONMENT=/data/$USER/venvs/locale
export PYTHONUNBUFFERED=1
# The exact scan allocates and frees a ~15 GB logits tile per block; without
# expandable segments the caching allocator fragments and can fail a request
# with plenty reported free (seen on the Perlmutter 2026-09-22 runs).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/synkrp/locale/benchmark

NUM_SHARDS=${NUM_SHARDS:-4}
SHARD=${SLURM_ARRAY_TASK_ID:?run as a job array (--array=0-$((NUM_SHARDS - 1)))}
[ "$SHARD" -lt "$NUM_SHARDS" ] || { echo "array task $SHARD >= NUM_SHARDS=$NUM_SHARDS"; exit 1; }
echo "config: $CONFIG  mutation_rate: $MUT_RATE  shard: $SHARD / $NUM_SHARDS  node: $(hostname)"

# Default stage (all), not --stage search: the index is .done so no build
# runs, but the engine artifact (a no-op directory for the exact scan) is
# created and marked .done here on first use; --stage search refuses unless
# it already exists (the 2026-10-18 search arrays failed on exactly that).
uv run --no-sync python scripts/run_benchmark.py --config "$CONFIG" \
  --mutation_rate "$MUT_RATE" \
  --shard "$SHARD" --num_shards "$NUM_SHARDS"
