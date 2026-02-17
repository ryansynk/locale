#!/bin/bash
#SBATCH --account=m5083_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/job_%j.out

export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS=1

uv run torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --config=configs/supervised_perlmutter.yaml