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
    --lr=4e-6 \
    --epochs=5 \
    --dim=128 \
    --moco_queue_size 131072 \
    --moco_momentum 0.9995 \
    --moco_softmax_temp 0.05 \
    --checkpoint_interval 1000 \
