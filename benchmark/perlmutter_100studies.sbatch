#!/bin/bash
#SBATCH -A m5408_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 32
#SBATCH -t 10:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -J locale-100studies
#SBATCH -o slurm_logs/%j.out
#SBATCH -e slurm_logs/%j.err

# 32-node build+search of the sra100studies dense locale index.
# ~97 node-hours of embedding -> ~3h across 32 nodes, plus a serial tail on
# node 0 (7TB shard merge + search, ~2-4h). 10h walltime covers a slow day;
# charge is for elapsed time, not the request.
#
# One task per node: run_benchmark shards accessions by SLURM_NODEID and each
# task fans out over its node's 4 GPUs internally.
#
# Shard striping depends on the node count: the shard_* dirs under
# indexes_100studies must be empty on the first run at a new -N, and any
# resubmission after a failure must reuse the SAME -N so completed shards
# (.done) are valid to reuse.
#
# When premium access lands, either change -q to premium here, or move an
# already-pending job over without losing queue position:
#   scontrol update JobId=<id> QOS=premium

set -euo pipefail

cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1

srun uv run python run_benchmark.py --config configs/perlmutter_locale_sra100studies.yaml
