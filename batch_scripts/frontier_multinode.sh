#!/bin/bash
#SBATCH --account=LRN089
#SBATCH --qos=normal
#SBATCH --time=2:00:00
#SBATCH --nodes=16-31
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
#SBATCH --gpus-per-node=8
#SBATCH --output=logs/job_%j.out

export MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

cd /lustre/orion/lrn089/scratch/ryansynk/locale

srun uv run python -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOBID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py --config configs/unsupervised_frontier_containment.yaml \
    --use_hard_negatives True 