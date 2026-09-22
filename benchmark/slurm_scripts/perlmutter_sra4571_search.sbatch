#!/bin/bash
#SBATCH -A m5408_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 8
#SBATCH -t 8:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -J locale-sra4571-search
#SBATCH -o slurm_logs/sra4571_search_%j.out
#SBATCH -e slurm_logs/sra4571_search_%j.err

# Step 3 of 3: multi-node search over the merged sra4571 dense index.
# Mutation rate is the first argument (0.00 / 0.05 / 0.10), passed as a CLI
# override of the config's mutation_rate. Results land in
# results_sra4571/<experiment_id>/raw_read_mut_<rate>.parquet, so the three
# rates never collide and can run concurrently. The index must be .done (run
# the merge first); all nodes skip build and go straight to search.
#
# 8 nodes / 8h matches the 100studies searches, which took ~3h50m each.
# Partials are keyed by (mutation_rate, num_queries, seed) and reused on
# rerun, so a resubmit after a timeout resumes rather than rescoring.
#
# Optional second argument: config path. Default is the full-dense
# (streaming, exact-over-all-accessions) baseline. For the top-k-then-regroup
# yardstick pass configs/perlmutter_locale_sra4571_exact_topk1000.yaml: same
# index, each node scans an equal range of vector rows, node 0 merges the
# per-query top-1000 hits (under results_sra4571_topk_hits/ for
# rescore_topk.py) plus a results parquet under experiment_id ..._exacttop1000.
MUT_RATE="${1:?usage: sbatch perlmutter_sra4571_search.sbatch <mutation_rate> [config.yaml]}"
CONFIG="${2:-configs/perlmutter_locale_sra4571.yaml}"

set -euo pipefail

cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1

srun --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=128 \
  uv run python run_benchmark.py \
    --config "$CONFIG" \
    --mutation_rate "$MUT_RATE"
