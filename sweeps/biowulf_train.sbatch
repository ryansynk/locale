#!/bin/bash
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:h200:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=768g
#SBATCH --time=24:00:00
#SBATCH --job-name=locale-train
#SBATCH --output=/data/synkrp/locale/slurm_logs/train_%j.out
#SBATCH --error=/data/synkrp/locale/slurm_logs/train_%j.err

# One training run of train.py on one 8 x H200 node (global batch 8 x 64 =
# 512; training.py sqrt-scales the LRs off reference_global_batch_size).
# Same environment as biowulf_lr_sweep.sbatch, but a plain config instead of
# a wandb sweep agent:
#
#   CONFIG=configs/paper_biowulf_config_nohm_edloss.yaml sbatch sweeps/biowulf_train.sbatch
#
# Extra train.py flags can follow through EXTRA_ARGS, e.g.
#   EXTRA_ARGS="--seed 1" CONFIG=... sbatch sweeps/biowulf_train.sbatch
# For a 4 h slot that starts quickly use: sbatch -p quick -t 4:00:00 ...
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/data/$USER/.cache/uv
export UV_PYTHON_INSTALL_DIR=/data/$USER/uv/python
export UV_TOOL_DIR=/data/$USER/uv/tools
export UV_PROJECT_ENVIRONMENT=/data/$USER/venvs/locale
export PYTHONUNBUFFERED=1

# Compute nodes reach wandb (and HF, for the val bundle) only through the
# Biowulf web proxy; .bashrc is not read by batch jobs, so set it here.
if [ -f /etc/profile.d/b2webproxy.sh ]; then
    source /etc/profile.d/b2webproxy.sh
fi

# sbatch propagates the login shell's soft `ulimit -u` of 1024; threads count
# against it and 8 ranks blow through it in the first read_parquet. Raise it.
ulimit -u "$(ulimit -Hu)"
echo "ulimit -u: $(ulimit -Su)"

cd /home/synkrp/locale

CONFIG=${CONFIG:?set CONFIG=<path to a train config yaml>}
echo "config: $CONFIG  job: ${SLURM_JOB_ID:-?}  node: $(hostname)  gpus: ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086
uv run --no-sync python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=8 \
    train.py --config "$CONFIG" ${EXTRA_ARGS:-}
