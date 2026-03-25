#!/bin/bash
#SBATCH --account=m5083_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=5:00:00        # Dropped time significantly since we are 32x faster
#SBATCH --nodes=8             # Scale up to 8 nodes to hit the desired queue
#SBATCH --ntasks-per-node=4   # 1 task per GPU
#SBATCH --gpus-per-node=4     
#SBATCH --cpus-per-task=32    # 128 cpus / 4 tasks
#SBATCH --output=logs/job_%j_node_%N.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=ryansynk@umd.edu
#SBATCH --exclusive

export SLURM_CPU_BIND="cores"

cd /pscratch/sd/r/rsynk/rawbert/experiments/sra_recall

# srun will execute 32 parallel instances of run_benchmark.py
srun uv run python build_index_parallel.py --config configs/perlmutter_evo2.yaml