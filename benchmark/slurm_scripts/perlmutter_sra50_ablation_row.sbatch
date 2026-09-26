#!/bin/bash
#SBATCH -A m5408_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -J locale-sra50-ablation-row
#SBATCH -o slurm_logs/sra50_ablation_row_%j.out
#SBATCH -e slurm_logs/sra50_ablation_row_%j.err

# Evaluate one or more configs/ablation_sweep/ rows on sra50v2 on a single GPU
# node (index build at 0% then 5% and 10% reuse it), via run_ablation_sweep.sh.
#   ROWS="mut_none_logan" sbatch slurm_scripts/perlmutter_sra50_ablation_row.sbatch
set -euo pipefail
cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs
./run_ablation_sweep.sh ${ROWS:-mut_none_logan}
