#!/bin/bash
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:h200:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=768g
#SBATCH --time=6:00:00
#SBATCH --job-name=locale-sra500
#SBATCH --output=/data/synkrp/locale/slurm_logs/sra500_%j.out
#SBATCH --error=/data/synkrp/locale/slurm_logs/sra500_%j.err

# Biowulf runner for one dense configs/sra500/*.yaml method on sra500v2 (the
# counterpart of run_sra500.sh, which is Perlmutter-only: multi-node srun,
# /pscratch paths, shifter). Biowulf allows no multi-node GPU jobs, so this
# is one 8x H200 node and run_benchmark runs with --num_shards 1: the whole
# index is built in one process with one embedding worker per GPU, and the
# exact scan runs across all 8 GPUs. Per rate the first run builds the index
# and the later rates reuse it (the index depends on the model, not on the
# mutation rate); after the build the index is checked for the full
# accession count, then the results table is printed.
#
# Sizing (biowulf_locale_4svch81s_w128_sra500v2.yaml, measured 2026-11-04):
# 132.9 M windows / 136 GB fp32; one H200 encodes ~5.8 k 128-bp windows/s at
# batch_size 3500 (6.9 GB peak), so the embed is >= 48 min on 8 GPUs plus
# chunk iteration and /data writes; allow ~1.5 h. The exact scan streams the
# 136 GB fbin once per rate (the node's 2.2 TB RAM caches it after the first).
#
#   sbatch slurm_scripts/biowulf_sra500.sbatch                   # default CONFIG, 3 rates
#   CONFIG=configs/sra500/biowulf_<x>.yaml sbatch slurm_scripts/biowulf_sra500.sbatch
#   RATES="0.0" sbatch slurm_scripts/biowulf_sra500.sbatch       # build + clean-query eval only
#   RATES="0.05 0.1" sbatch --time=2:00:00 slurm_scripts/biowulf_sra500.sbatch   # index .done
#
# Knobs (env): CONFIG, RATES ("0.0 0.05 0.1").
# Logs: /data/synkrp/locale/slurm_logs/sra500_<jobid>.{out,err}
set -euo pipefail

# Batch jobs inherit the login shell's soft `ulimit -u` 1024 and threads
# count against it; 8 embedding workers plus polars/torch thread pools
# exceed it ("could not spawn threads ... Resource temporarily unavailable").
ulimit -u "$(ulimit -Hu)"

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/data/$USER/.cache/uv
export UV_PYTHON_INSTALL_DIR=/data/$USER/uv/python
export UV_TOOL_DIR=/data/$USER/uv/tools
export UV_PROJECT_ENVIRONMENT=/data/$USER/venvs/locale
export PYTHONUNBUFFERED=1
# The exact scan allocates and frees a large logits tile per block; without
# expandable segments the caching allocator fragments and can fail a request
# with plenty reported free.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/synkrp/locale/benchmark

CONFIG=${CONFIG:-configs/sra500/biowulf_locale_4svch81s_w128_sra500v2.yaml}
read -r -a RATE_LIST <<<"${RATES:-0.0 0.05 0.1}"
[ -f "$CONFIG" ] || { echo "no config $CONFIG" >&2; exit 1; }
echo "config: $CONFIG  rates: ${RATE_LIST[*]}  node: $(hostname)  gpus: $(nvidia-smi -L | wc -l)  cpus: $(nproc)"

# dataset_dir / results_dir come from the config so another sra500 config
# with different paths still works.
read -r BUNDLE RESULTS <<<"$(uv run --no-sync python - "$CONFIG" <<'PY'
import sys
from jsonargparse import CLI
from src.config import ExperimentConfig
cfg = CLI(ExperimentConfig, as_positional=False, args=["--config", sys.argv[1]])
print(cfg.dataset_dir, cfg.results_dir)
PY
)"

# ---- contigs present? (same rglob run_benchmark uses) -------------------------
# With no contigs under logan_accessions, run_benchmark downloads all 500 from
# Logan, which hangs on compute nodes (no internet). Check before holding GPUs.
n_accs=$(grep -c . "$BUNDLE/accs.txt")
n_found=$(uv run --no-sync python -c "from pathlib import Path; print(len(list(Path('$BUNDLE/logan_accessions').rglob('*.contigs.fa'))))")
echo "logan_accessions: $n_found contigs for $n_accs accessions"
if [ "$n_found" -ne "$n_accs" ]; then
    echo "Contig count mismatch; link the contigs into $BUNDLE/logan_accessions first." >&2
    exit 1
fi

# meta.parquet holds one row per accession in a dense index. Anything short
# of the full set means the build dropped accessions.
check_dense_index() {
    uv run --no-sync python - "$1" "$n_accs" <<'PY'
import sys
import polars as pl
from jsonargparse import CLI
from src.config import ExperimentConfig

cfg_path, n_accs = sys.argv[1], int(sys.argv[2])
cfg = CLI(ExperimentConfig, as_positional=False, args=["--config", cfg_path])
meta = cfg.model.index_path(cfg.index_dir) / "meta.parquet"
if not meta.exists():
    sys.exit(f"[FAIL] no {meta}")
n = pl.read_parquet(meta)["srr_id"].n_unique()
if n != n_accs:
    sys.exit(f"[FAIL] index at {meta} has {n} accessions, expected {n_accs}. "
             "A worker was probably lost; delete the index dir to force a rebuild.")
print(f"[ok] index has {n} accessions")
PY
}

for rate in "${RATE_LIST[@]}"; do
    echo "=== $CONFIG @ mutation_rate $rate  ($(date)) ==="
    uv run --no-sync python scripts/run_benchmark.py \
        --config "$CONFIG" --mutation_rate "$rate" --num_shards 1
    if [ "$rate" = "${RATE_LIST[0]}" ]; then
        check_dense_index "$CONFIG"
    fi
done

echo
echo "=== sra500v2 results  ($(date)) ==="
uv run --no-sync python scripts/print_results.py "$RESULTS" "$BUNDLE/queries_mut0.00.parquet" "$BUNDLE/accs.txt" || true
echo "All sra500v2 runs for $CONFIG finished."
