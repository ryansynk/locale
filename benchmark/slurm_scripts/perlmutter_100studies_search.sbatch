#!/bin/bash
#SBATCH -A m5408_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 8
#SBATCH -t 8:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -J locale-100studies-search
#SBATCH -o slurm_logs/search_%j.out
#SBATCH -e slurm_logs/search_%j.err

# Multi-node exact search over the merged sra100studies dense index.
# Mutation rate is the first argument (0.00 / 0.05 / 0.10), passed as a CLI
# override of the config's mutation_rate. Results land in
# results_100studies/<experiment_id>/raw_read_mut_<rate>.parquet, so the three
# rates never collide and can run concurrently. The index is read-only and
# already .done, so all nodes skip build and go straight to search; each node
# scores a stride of the accessions and node 0 merges the partials.
#
# 8 nodes / 8h: a 4-node/4h interactive run wrote no partials before timing
# out, so this doubles both parallelism and budget (~4x headroom). Partials
# are keyed by (mutation_rate, num_queries, seed) and reused on rerun, so a
# resubmit after a timeout resumes rather than rescoring.

set -euo pipefail

# Optional second argument: config path. The default is the full-dense
# (streaming, exact-over-all-accessions) baseline. For the top-k-then-regroup
# experiment pass configs/perlmutter_locale_sra100studies_exact_topk1000.yaml:
# same index, but each node scans an equal range of vector rows and node 0
# merges the per-query top-1000 hits (written under results_100studies_topk_hits/
# for rescore_topk.py) plus a standard results parquet under a distinct
# experiment_id (..._exacttop1000), so the baseline is never overwritten.
MUT_RATE="${1:?usage: sbatch perlmutter_100studies_search.sbatch <mutation_rate> [config.yaml]}"
CONFIG="${2:-configs/perlmutter_locale_sra100studies.yaml}"

cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1

srun --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=128 \
  uv run python run_benchmark.py \
    --config "$CONFIG" \
    --mutation_rate "$MUT_RATE"
