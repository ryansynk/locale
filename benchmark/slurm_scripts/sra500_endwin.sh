#!/bin/bash
# Re-embed sra500v2 with end-aligned contig windows
# (configs/sra500/perlmutter_locale_sra500v2_endwin50.yaml) and run the exact
# top-100 search at every noise level, on every node of an existing
# allocation. Run it inside the allocation, or from a login node with
# SLURM_JOB_ID set to it (every step is srun --overlap):
#
#   salloc -A m5408_g -C gpu -q interactive -N 4 --gpus-per-node=4 -t 3:00:00
#   bash slurm_scripts/sra500_endwin.sh
#
# Same flow as perlmutter_sra500_esa.sbatch: one multi-node run_benchmark per
# noise level. The first one embeds one accession shard per node, rank 0
# merges the shards into indexes/sra500/locale@8vqiabk9-endwin50/ while the
# other ranks wait, then all nodes scan their range of vector rows and rank 0
# merges the top-100 hits. Later noise levels reuse the .done index.
#
# Resumable: shards marked .done are kept on a rerun with the SAME node count
# (shard striping depends on it); a finished search reuses its saved hits.
# Rank ids come from SLURM_PROCID / SLURM_NTASKS (one task per node).
# Logs: slurm_logs/sra500_endwin_mut<rate>.log
set -euo pipefail
: "${SLURM_JOB_ID:?run inside an allocation, or set SLURM_JOB_ID to one}"
cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1
# The exact scan allocates and frees a large logits tile per block; expandable
# segments keep the caching allocator from fragmenting.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONFIG=${CONFIG:-configs/sra500/perlmutter_locale_sra500v2_endwin50.yaml}
BUNDLE=/pscratch/sd/r/rsynk/locale-data/constructed/sra500/bundle
read -r -a RATE_LIST <<<"${RATES:-0.0 0.05 0.1}"
mapfile -t NODES < <(scontrol show hostnames "$(squeue -h -j "$SLURM_JOB_ID" -o %N)")
NODELIST=$(IFS=,; echo "${NODES[*]}")
echo "=== ${#NODES[@]} node(s): ${NODES[*]}"

# Abort before srun unless every contig is linked (otherwise every node would
# re-download the 500 accessions from Logan at once).
n_accs=$(grep -c . "$BUNDLE/accs.txt")
n_found=$(uv run --no-sync python -c "from pathlib import Path; print(len(list(Path('$BUNDLE/logan_accessions').rglob('*.contigs.fa'))))")
echo "logan_accessions: $n_found contigs for $n_accs accessions"
[ "$n_found" -eq "$n_accs" ] || { echo "contig count mismatch; aborting" >&2; exit 1; }

for rate in "${RATE_LIST[@]}"; do
    log=slurm_logs/sra500_endwin_mut$rate.log
    t0=$SECONDS
    echo "=== mutation_rate $rate ($(date +%T)) -> $log"
    srun --jobid "$SLURM_JOB_ID" --overlap --unbuffered -N "${#NODES[@]}" \
        -n "${#NODES[@]}" --ntasks-per-node=1 -w "$NODELIST" \
        --gpus-per-node=4 --cpus-per-task=128 \
        bash -c 'exec "$@" --shard "$SLURM_PROCID" --num_shards "$SLURM_NTASKS"' _ \
        uv run --no-sync python scripts/run_benchmark.py --config "$CONFIG" \
        --mutation_rate "$rate" >"$log" 2>&1 ||
        { echo "FAILED at rate $rate (see $log)"; tail -5 "$log"; exit 1; }
    echo "    done in $((SECONDS - t0))s"
done

# A build that loses a worker still gets its .done marker: check the merged
# index holds every accession, and report how many vectors the end windows
# added over the original sra500 index.
uv run --no-sync python - "$CONFIG" "$n_accs" <<'PY'
import sys
from pathlib import Path

import polars as pl
from jsonargparse import CLI
from src.config import ExperimentConfig

cfg_path, n_accs = sys.argv[1], int(sys.argv[2])
cfg = CLI(ExperimentConfig, as_positional=False, args=["--config", cfg_path])
new = pl.read_parquet(cfg.model.index_path(cfg.index_dir) / "meta.parquet")
old = pl.read_parquet(Path(cfg.index_dir) / "locale@8vqiabk9" / "meta.parquet")
n = new["srr_id"].n_unique()
if n != n_accs:
    sys.exit(f"[FAIL] index has {n} accessions, expected {n_accs}")
nv, ov = int(new["num_rows"].sum()), int(old["num_rows"].sum())
print(f"[ok] index has {n} accessions; {nv:,} vectors vs {ov:,} without end "
      f"windows (+{nv - ov:,}, +{(nv - ov) / ov:.1%})")
PY
echo "=== all done ($(date +%T)); results in" \
    "results/sra500/locale@8vqiabk9-endwin50/exact/top100/"
