#!/bin/bash
#SBATCH -A m5408
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --image=ghcr.io/ratschlab/metagraph:master
#SBATCH -J locale-sra55viral-metagraph
#SBATCH -o slurm_logs/sra55viral_metagraph_%j.out
#SBATCH -e slurm_logs/sra55viral_metagraph_%j.err

# Metagraph k-mer baseline on sra55viral (clean queries only, see
# configs/sra55viral/). Single CPU node: metagraph is not sharded, and the
# published container must be requested on the job itself (--image above), so
# this runs outside run_sra55viral.sh. Results land in
# results/sra55viral/metagraph_k31/raw_read_mut_0.0.parquet.
set -euo pipefail
cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1

# The 55 contigs are already under the bundle's logan_accessions/ (fetched by
# run_sra55viral.sh); run_benchmark would re-download only if they were gone.
srun --unbuffered -N1 -n1 --cpus-per-task=128 \
  /usr/bin/env SLURM_NNODES=1 SLURM_NODEID=0 \
  uv run python run_benchmark.py \
    --config configs/sra55viral/perlmutter_metagraph_sra55viral.yaml \
    --mutation_rate 0.0
