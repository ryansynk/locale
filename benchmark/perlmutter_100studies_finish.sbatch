#!/bin/bash
#SBATCH -A m5408_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -J locale-100studies-finish
#SBATCH -o slurm_logs/%j.out
#SBATCH -e slurm_logs/%j.err

# Recovery for the timed-out 32-node run (58208345): all 32 shards are built,
# so this single node finishes the merge and runs the search. The index dir is
# striped (lfs setstripe -c 32), so the merge's parallel byte copies and the
# search's 7TB streaming read are no longer bottlenecked on one OST.
#
# Both steps are resumable: the merge records completed shards in
# .merge_progress, and run_benchmark skips build entirely once .done exists.

set -euo pipefail

cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1

INDEX_PATH=indexes_100studies/locale/8vqiabk9/5859/maxlen256_poolmean_chunkstride

uv run python finish_merge.py "$INDEX_PATH" 32
uv run python run_benchmark.py --config configs/perlmutter_locale_sra100studies.yaml
