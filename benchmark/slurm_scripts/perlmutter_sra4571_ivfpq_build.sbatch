#!/bin/bash
#SBATCH -A m5408_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 4
#SBATCH -t 1:30:00
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH -c 32
#SBATCH -J locale-sra4571-ivfpq-build
#SBATCH -o slurm_logs/sra4571_ivfpq_build_%j.out
#SBATCH -e slurm_logs/sra4571_ivfpq_build_%j.err

# Build the 16 cuVS IVF-PQ shards of the sra4571 fbin: one task per GPU
# (SLURM_PROCID = shard), each pread-streams its 1/16 of the rows (~395 GB),
# trains 4096 lists + 128x8 PQ codebooks on a 1M-row sample of its range and
# extends in 2M-row blocks, then saves
# indexes/sra4571/.../ivfpq/pq128x8_L4096/shard_<i>_of_16.cuvs (~17.5 GB).
# Keep 16 tasks across resubmits; finished shards are skipped.
CONFIG="${1:-configs/sra4571/perlmutter_locale_sra4571_ivfpq.yaml}"
set -euo pipefail
cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1
srun --unbuffered --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=32 \
  uv run python build_ivfpq.py --config "$CONFIG"
