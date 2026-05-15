#!/bin/bash

#SBATCH --account=m5083_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=1:00:00
#SBATCH --nodes=12-15
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --array=0-1
#SBATCH --job-name=queue_filtering_ablation

export MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

cd /pscratch/sd/r/rsynk/locale

echo $SLURM_ARRAY_TASK_ID

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    FILTER_ARGS="--moco_filter_queue_identity_cutoff 0.9 --moco_filter_queue True"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    FILTER_ARGS="--moco_filter_queue False"
fi

srun uv run --no-sync python -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py --config configs/unsupervised_perlmutter_containment.yaml \
    --dataset_path /pscratch/sd/r/rsynk/locale_data/reference_genomes/reference_genome_dataset.parquet \
    --use_hard_negatives True \
    --total_samples 6_000_000 \
    --augment_config.disable_mutations False \
    --augment_config.identity_mean 90 \
    --augment_config.identity_max 98 \
    --augment_config.identity_stdev 6 \
    --moco_queue_size 32768 \
    $FILTER_ARGS

RUN_ID=$(cat logs/wandb_run_id_${SLURM_ARRAY_TASK_ID}.txt)
CHECKPOINT_PATH=$(ls /pscratch/sd/r/rsynk/locale/checkpoints/$RUN_ID/checkpoint[0-9]*.pth.tar | sort -V | tail -1)

cd /pscratch/sd/r/rsynk/locale/experiments/sra_recall

srun --nodes=$SLURM_NNODES --ntasks-per-node=1 uv run --no-sync python run_benchmark.py \
    --config configs/perlmutter_locale.yaml \
    --model.checkpoint_path $CHECKPOINT_PATH \
    --query_type raw_read \
    --mutation_rate 0.0

srun --nodes=$SLURM_NNODES --ntasks-per-node=1 uv run --no-sync python run_benchmark.py \
    --config configs/perlmutter_locale.yaml \
    --model.checkpoint_path $CHECKPOINT_PATH \
    --query_type raw_read \
    --mutation_rate 0.05

srun --nodes=$SLURM_NNODES --ntasks-per-node=1 uv run --no-sync python run_benchmark.py \
    --config configs/perlmutter_locale.yaml \
    --model.checkpoint_path $CHECKPOINT_PATH \
    --query_type raw_read \
    --mutation_rate 0.10