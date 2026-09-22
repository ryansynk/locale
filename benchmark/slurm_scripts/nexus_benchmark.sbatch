#!/bin/bash
#SBATCH --job-name=locale-benchmark
#SBATCH --output=slurm_logs/benchmark_%j.out
#SBATCH --error=slurm_logs/benchmark_%j.err
#SBATCH --partition=cml-dpart
#SBATCH --qos=cml-very_high
#SBATCH --account=cml-tomg
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --mem=128G
#SBATCH -c 16

# Nexus benchmark run. Partition/qos/account carried over from the pre-cleanup
# nexus_scripts/ (untracked); verified against sacctmgr.
#
# Note: rtxa4000 has 16 GB VRAM. If model.batch_size (1000 in nexus_locale.yaml)
# OOMs, lower it or request a larger card from cml-scavenger.

set -euo pipefail

REPO=/fs/nexus-scratch/ryansynk/rawbert
CONFIG=${1:-configs/nexus_locale.yaml}

export PATH="/nfshomes/ryansynk/.local/bin:$PATH"

cd "$REPO/benchmark"
mkdir -p slurm_logs

uv run python run_benchmark.py --config "$CONFIG" "${@:2}"
