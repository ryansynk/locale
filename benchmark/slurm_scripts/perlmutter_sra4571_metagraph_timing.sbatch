#!/bin/bash
#SBATCH -A m5408
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 3:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -c 256
#SBATCH --image=ghcr.io/ratschlab/metagraph:master
#SBATCH -J locale-sra4571-metagraph-timing
#SBATCH -o slurm_logs/sra4571_metagraph_timing_%j.out
#SBATCH -e slurm_logs/sra4571_metagraph_timing_%j.err

# Warm, single-CPU-node query timing of the existing 8-shard sra4571 metagraph
# index (graphs.csv, 36 GB), for the IVF-vs-metagraph query-time comparison.
# run_benchmark --do_timing: 5 searches of the 1000 queries, avg_time = mean
# of runs 2-5 (run 1 pages the mmap'd index in from Lustre). Same protocol as
# the dense timing runs. PARALLEL=N sets server_parallel (N concurrent requests,
# server -p N, experiment_id metagraph_k31_p<N>); default 1 = the baseline protocol. A background sampler logs CPU utilization so the
# number of cores metagraph actually uses is on record.
set -euo pipefail
cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1
read -r -a RATE_LIST <<<"${RATES:-0.0 0.05 0.1}"
( while true; do echo "--- $(date +%T) load $(cut -d' ' -f1 /proc/loadavg)"; top -bn1 -w 200 | sed -n 8,11p; sleep 20; done ) > "slurm_logs/sra4571_metagraph_timing_${SLURM_JOB_ID}.cpu" 2>&1 &
for rate in "${RATE_LIST[@]}"; do
    echo "=== metagraph timing @ $rate ($(date)) ==="
    uv run python run_benchmark.py --config configs/sra4571/perlmutter_metagraph_sra4571.yaml \
        --mutation_rate "$rate" --do_timing true --results_dir results/sra4571_timing --model.server_parallel ${PARALLEL:-1}
done
uv run python - <<'PY'
import polars as pl
from pathlib import Path
for f in sorted(Path("results/sra4571_timing").rglob("*.parquet")):
    df = pl.read_parquet(f)
    print(f"{f.parts[-2]:60s} {f.name:28s} avg_time {df['avg_time'][0]:8.2f} s  index {df['index_size_gb'][0]:8.2f} GB")
PY
