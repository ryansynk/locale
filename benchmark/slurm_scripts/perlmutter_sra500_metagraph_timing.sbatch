#!/bin/bash
#SBATCH -A m5408
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 0:40:00
#SBATCH --ntasks-per-node=1
#SBATCH -c 256
#SBATCH --image=ghcr.io/ratschlab/metagraph:master
#SBATCH -J locale-sra500-metagraph-timing
#SBATCH -o slurm_logs/sra500_metagraph_timing_%j.out
#SBATCH -e slurm_logs/sra500_metagraph_timing_%j.err

# Warm, single-CPU-node query timing of the sra500v2 metagraph index (4 GB),
# for the recall-vs-query-time scatter at 500 accessions. Same protocol as the
# sra50 point in the paper: run_benchmark --do_timing, 5 searches of the 500
# queries, avg_time = mean of runs 2-5, single request (server_parallel 1).
# RATES defaults to 0.0 only: query time does not depend on the mutation level
# and the accuracy parquets already live in results/sra500/metagraph_k31.
set -euo pipefail
cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1
read -r -a RATE_LIST <<<"${RATES:-0.0}"
for rate in "${RATE_LIST[@]}"; do
    echo "=== metagraph timing @ $rate ($(date)) ==="
    uv run python run_benchmark.py --config configs/sra500/perlmutter_metagraph_sra500v2.yaml \
        --mutation_rate "$rate" --do_timing true --results_dir results/sra500_timing --model.server_parallel ${PARALLEL:-1}
done
uv run python - <<'PY'
import polars as pl
from pathlib import Path
for f in sorted(Path("results/sra500_timing/metagraph_k31").rglob("*.parquet")):
    df = pl.read_parquet(f)
    print(f"{f.parts[-2]:40s} {f.name:28s} avg_time {df['avg_time'][0]:8.2f} s  index {df['index_size_gb'][0]:8.2f} GB")
PY
