#!/bin/bash

# Check if user provided an argument
if [ -z "$1" ]; then
    echo "Usage: ./launch.sh <num_gpus>"
    exit 1
fi

NUM_GPUS=$1

echo "Launching training on $NUM_GPUS GPUs..."

torchrun --standalone --nproc_per_node=$NUM_GPUS train.py \
    --dataset_path="./data/gencode.v49.transcripts.train.fa" \
    --test_dataset_path="./data/gencode.v49.transcripts.test.fa" \
    --batch_size=128 \
    --lr=1e-6 \
    --epochs=10 \
    --dim=128 \
    --moco_queue_size 65536 \
    --checkpoint_interval 1000 \
