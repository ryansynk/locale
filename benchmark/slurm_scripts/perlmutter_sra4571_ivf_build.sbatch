#!/bin/bash
#SBATCH -A m5408_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 8
#SBATCH -t 1:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -J locale-sra4571-ivf-build
#SBATCH -o slurm_logs/sra4571_ivf_build_%j.out
#SBATCH -e slurm_logs/sra4571_ivf_build_%j.err

# Build the IVF-RaBitQ index over the merged sra4571 fbin (2.06 B x 768).
# Each node pread-streams 1/8 of the rows (~0.8 TB), assigns them to the
# nearest of ivf_nlist centroids on its 4 GPUs and faiss-encodes 1-bit RaBitQ
# residual codes on its CPUs (~2.4 s per 1M rows, ~11 min per node), writing
# <index>/ivf/ivf<nlist>_rabitq1_shard_<r>_of_8.faiss (~29 GB each). Keep -N
# fixed across resubmits: finished shards are skipped, and the shard count is
# in the file names. The search job (perlmutter_sra4571_ivf_search.sbatch, one
# CPU node) merges the shards into one file on first load.
CONFIG="${1:-configs/sra4571/perlmutter_locale_sra4571_ivf.yaml}"
set -euo pipefail
cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1
srun --unbuffered --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=128 \
  uv run python build_ivf.py --config "$CONFIG"
