#!/bin/bash
#SBATCH --account=m5083_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus=4
#SBATCH --output=logs/job_%j.out
#SBATCH --exclusive

export SLURM_CPU_BIND="cores"

cd /pscratch/sd/r/rsynk/rawbert/experiments/sra_recall
uv run python run_benchmark.py --config configs/perlmutter_evo2.yaml --query_type raw_read