#!/bin/bash

#SBATCH --account=m5083_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=0:30:00
#SBATCH --nodes=88-127
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --array=0-3

export MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

cd /pscratch/sd/r/rsynk/rawbert

echo $SLURM_ARRAY_TASK_ID

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    NOISE_ARGS="--augment_config.disable_mutations False --augment_config.identity_mean 90 --augment_config.identity_max 98 --augment_config.identity_stdev 6.0"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    NOISE_ARGS="--augment_config.disable_mutations False --augment_config.identity_mean 95 --augment_config.identity_max 99 --augment_config.identity_stdev 2.5"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    NOISE_ARGS="--augment_config.disable_mutations False --augment_config.identity_mean 80 --augment_config.identity_max 88 --augment_config.identity_stdev 6.0"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    NOISE_ARGS="--augment_config.disable_mutations True"
fi

srun uv run python -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py --config configs/unsupervised_perlmutter_containment.yaml \
    --dataset_path /pscratch/sd/r/rsynk/rawbert_data/reference_genomes/reference_genome_dataset.parquet \
    --use_hard_negatives True \
    --moco_filter_queue_identity_cutoff 0.7 \
    --total_samples 6_000_000 \
    $NOISE_ARGS

RUN_ID=$(cat logs/wandb_run_id_${SLURM_ARRAY_TASK_ID}.txt)
CHECKPOINT_PATH=$(ls /pscratch/sd/r/rsynk/rawbert/checkpoints/$RUN_ID/checkpoint[0-9]*.pth.tar | sort -V | tail -1)

cd /pscratch/sd/r/rsynk/rawbert/experiments/sra_recall

srun --nodes=$SLURM_NNODES --ntasks-per-node=1 uv run --no-sync python run_benchmark.py \
    --config configs/perlmutter_rawbert.yaml \
    --model.checkpoint_path $CHECKPOINT_PATH \
    --query_type raw_read \
    --mutation_rate 0.0

srun --nodes=$SLURM_NNODES --ntasks-per-node=1 uv run --no-sync python run_benchmark.py \
    --config configs/perlmutter_rawbert.yaml \
    --model.checkpoint_path $CHECKPOINT_PATH \
    --query_type raw_read \
    --mutation_rate 0.05

srun --nodes=$SLURM_NNODES --ntasks-per-node=1 uv run --no-sync python run_benchmark.py \
    --config configs/perlmutter_rawbert.yaml \
    --model.checkpoint_path $CHECKPOINT_PATH \
    --query_type raw_read \
    --mutation_rate 0.10