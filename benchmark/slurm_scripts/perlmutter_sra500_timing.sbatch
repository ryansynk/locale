#!/bin/bash
#SBATCH -A m5408_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -J locale-sra500-timing
#SBATCH -o slurm_logs/sra500_timing_%j.out
#SBATCH -e slurm_logs/sra500_timing_%j.err

# Timed single-node searches on sra500v2 for the Runs tab: LOCALE fp32,
# LOCALE + RaBitQ, ESA and LLM-ED, 500 clean queries each, one GPU node
# (4x A100). run_benchmark --do_timing runs index.search 5 times and records
# the mean of the last 4 as avg_time (query embedding for both strands +
# top-100 scan + regroup; the fbin streams from Lustre every pass). Indexes
# are the existing .done ones under indexes/sra500/; the RaBitQ codes are
# built into <locale index>/rabitq/ on first load and kept.
#
# Results go to results/sra500_timing/<experiment_id>/raw_read_mut_0.0.parquet
# (a separate dir so the accuracy results in results/sra500/, produced by the
# 4-node sharded scans, are not overwritten). avg_time and index_size_gb are
# the columns to read off.
#
#   sbatch slurm_scripts/perlmutter_sra500_timing.sbatch
#   METHODS="locale rabitq" sbatch slurm_scripts/perlmutter_sra500_timing.sbatch
set -euo pipefail

cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

declare -A CONFIGS=(
  [locale]=configs/sra500/perlmutter_locale_sra500v2.yaml
  [rabitq]=configs/sra500/perlmutter_locale_sra500v2_rabitq.yaml
  [esa]=configs/sra500/perlmutter_esa_sra500v2.yaml
  [llmed]=configs/sra500/perlmutter_llmed_sra500v2.yaml
)
read -r -a METHOD_LIST <<<"${METHODS:-locale rabitq esa llmed}"

for m in "${METHOD_LIST[@]}"; do
    cfg="${CONFIGS[$m]:?unknown method $m}"
    echo "=== timing $m  ($(date)) ==="
    # SLURM_NNODES=1: do_timing is single-node only (no sharded search).
    srun --unbuffered -N1 -n1 --gpus-per-node=4 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
      /usr/bin/env SLURM_NNODES=1 SLURM_NODEID=0 \
      uv run python run_benchmark.py --config "$cfg" --mutation_rate 0.0 \
        --do_timing true --results_dir results/sra500_timing
    echo "=== done $m  ($(date)) ==="
done

uv run python - <<'PY'
import polars as pl
from pathlib import Path
for f in sorted(Path("results/sra500_timing").rglob("*.parquet")):
    df = pl.read_parquet(f)
    print(f"{f.parts[-2]:75s} avg_time {df['avg_time'][0]:8.2f} s   index {df['index_size_gb'][0]:8.2f} GB")
PY
