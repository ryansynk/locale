#!/bin/bash
#SBATCH --partition=quick
#SBATCH --gres=gpu:h200:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=768g
#SBATCH --time=4:00:00
#SBATCH --job-name=locale-moco-queue-sweep
#SBATCH --output=/data/synkrp/locale/slurm_logs/moco_queue_sweep_%A_%a.out
#SBATCH --error=/data/synkrp/locale/slurm_logs/moco_queue_sweep_%A_%a.err

# One wandb sweep agent per array task, one run per agent (--count 1), so each
# task takes one grid point from the sweep and releases the node. Submit with
# the sweep id from `wandb sweep sweeps/moco_queue_grid.yaml` and one task per grid
# point (5 for the queue grid):
#
#   SWEEP_ID=<id> sbatch --array=0-4 sweeps/biowulf_moco_queue_sweep.sbatch
#
# Grid sweeps hand out points in order, so if a task dies before its run
# finishes, resubmit a single extra task (--array=5) and the agent picks up
# the point that was never completed. Biowulf has 7 H200 nodes, so at most
# 7 tasks run at once; the rest queue.
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

# sbatch propagates the submitting shell's soft limits (PropagateResourceLimits
# = ALL), and the login node caps user processes at 1024. Threads count against
# that, and 8 ranks x (polars pool + torch + dataloader workers) blow through
# it in the first read_parquet with "could not spawn threads: Resource
# temporarily unavailable". Raise it to the hard limit before launching.
ulimit -u "$(ulimit -Hu)"
echo "ulimit -u: $(ulimit -Su)"

cd /home/synkrp/locale

SWEEP_ID=${SWEEP_ID:?set SWEEP_ID=<entity/project/id or bare id> from 'wandb sweep sweeps/moco_queue_grid.yaml'}
case "$SWEEP_ID" in
    */*) ;;
    *) SWEEP_ID="tomg-group-umd/locale/$SWEEP_ID" ;;
esac

echo "sweep: $SWEEP_ID  task: ${SLURM_ARRAY_TASK_ID:-0}  node: $(hostname)  gpus: ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run --no-sync wandb agent --count 1 "$SWEEP_ID"
