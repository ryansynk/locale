#!/bin/bash
#SBATCH --account=m5083_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=1:00:00
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/job_%j.out

cd /pscratch/sd/r/rsynk/rawbert/experiments/sra_recall
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 uv run --frozen python run_benchmark.py --config configs/perlmutter_evo2.yaml --model.batch_size 128 --query_type raw_read --mutation_rate 0.0
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 uv run --frozen python run_benchmark.py --config configs/perlmutter_evo2.yaml --model.batch_size 128 --query_type raw_read --mutation_rate 0.05
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 uv run --frozen python run_benchmark.py --config configs/perlmutter_evo2.yaml --model.batch_size 128 --query_type raw_read --mutation_rate 0.10