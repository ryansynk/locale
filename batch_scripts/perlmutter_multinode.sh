#!/bin/bash
#SBATCH --account=m5083_g
#SBATCH --constraint=gpu
#SBATCH --qos=preempt
#SBATCH --time=2:00:00
#SBATCH --nodes=16-31
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/job_%j.out

export MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

cd /pscratch/sd/r/rsynk/rawbert

srun uv run python -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py --config configs/unsupervised_perlmutter_containment.yaml \
    --dataset_path /pscratch/sd/r/rsynk/rawbert_data/reference_genomes/reference_genome_dataset.parquet \
    --use_hard_negatives True 