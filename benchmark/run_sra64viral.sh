#!/usr/bin/env bash
#
# Run the configs/sra64viral/*.yaml methods on the clean queries (mutation
# rate 0.00), one srun step at a time, then print the results table. Run it
# from inside a GPU allocation, e.g.
#
#   salloc -A m5408_g -C gpu -q interactive -N 4 -t 4:00:00 \
#          --ntasks-per-node=1 --gpus-per-node=4 -c 128 \
#          --image=ghcr.io/ratschlab/metagraph:master
#   cd /pscratch/sd/r/rsynk/locale/benchmark && ./run_sra64viral.sh
#
# A method name is the middle of its config file name:
# configs/sra64viral/perlmutter_<method>_sra64viral.yaml. The default set is
#   locale_8vqiabk9 locale_l6e9itq9_w128 esa llmed mmseqs2 metagraph_k19 metagraph_k31
#
# The --image is only needed for metagraph (executable: shifter metagraph);
# without it the metagraph methods are skipped, the rest still run.
#
# Before any step the bundle's logan_accessions/ is checked for all 64
# accessions' contigs (the bundle ships them as symlinks into
# ../work/logan_contigs); any that are missing are fetched from Logan once,
# on this node, so a multi-node step never downloads concurrently.
#
# Dense methods (locale_*, esa, llmed) run as one process per node across the
# whole allocation: run_benchmark.py gets --shard / --num_shards from
# SLURM_NODEID / SLURM_NNODES, builds one index shard per node, and rank 0
# merges. metagraph_* and mmseqs2 are single-node methods, so they run on one
# node with --num_shards 1. Plain `python scripts/run_benchmark.py` inside a
# multi-node allocation would be a lone rank 0 waiting forever for peers:
# always srun.
#
# Only the clean queries are run by default: the divergence under test is the
# real genotype-D-vs-A/B/C difference (best-hit identity 0.90-1.00, median
# 0.95), not simulated read errors. RATES adds the bundle's mut0.05 / mut0.10
# files, which stack synthetic noise on top. After a dense build the index is
# checked for the full accession count: a build that loses a worker still gets
# its .done marker and would report quietly wrong recall.
#
#   ./run_sra64viral.sh                                   # all seven methods
#   ./run_sra64viral.sh locale_l6e9itq9_w128 mmseqs2      # a subset
#   RATES="0.0 0.05 0.1" ./run_sra64viral.sh              # also the synthetic-noise rates
#
# INDEX=sra500 searches the same queries against the existing sra500 indexes
# instead (configs/sra64viral_sra500idx/, results/sra64viral_sra500idx/;
# search only, nothing is built or downloaded). scripts/merge_index_results.py
# then unions those results with results/sra64viral into the result of one
# index holding sra500 plus the HBV runs:
#
#   INDEX=sra500 ./run_sra64viral.sh
#   uv run python scripts/merge_index_results.py
#
# The srun resource flags must fit inside the allocation; override with
# GPUS_PER_NODE / CPUS_PER_TASK if you asked salloc for less.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Not inside a Slurm allocation; run salloc first." >&2
    exit 1
fi

METHODS=("$@")
if [ ${#METHODS[@]} -eq 0 ]; then
    METHODS=(locale_8vqiabk9 locale_l6e9itq9_w128 esa llmed mmseqs2 metagraph_k19 metagraph_k31)
fi
read -r -a RATE_LIST <<<"${RATES:-0.0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-128}"
BUNDLE=/pscratch/sd/r/rsynk/locale-data/constructed/sra64viral-v2/bundle
RESULTS=/pscratch/sd/r/rsynk/locale/benchmark/results/sra64viral
CONFIG_DIR=configs/sra64viral
if [ "${INDEX:-}" = sra500 ]; then
    BUNDLE=${BUNDLE}_sra500idx
    RESULTS=${RESULTS}_sra500idx
    CONFIG_DIR=configs/sra64viral_sra500idx
elif [ -n "${INDEX:-}" ]; then
    echo "Unknown INDEX=$INDEX (only sra500 is supported)" >&2
    exit 1
fi

# metagraph_* and mmseqs2 build and search on one node
is_single_node() { [[ "$1" == metagraph* || "$1" == mmseqs2 ]]; }

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- contigs: fetch once, verify, before any multi-node step -----------------
n_accs=$(grep -c . "$BUNDLE/accs.txt")
# The sra500 indexes are already built; their contigs are not needed.
[ "${INDEX:-}" = sra500 ] || uv run python - "$BUNDLE" "$n_accs" <<'PY'
import sys
from pathlib import Path
from src.download_accessions import download_accessions

bundle, n_accs = Path(sys.argv[1]), int(sys.argv[2])
accs = [a for a in (bundle / "accs.txt").read_text().splitlines() if a]
acc_dir = bundle / "logan_accessions"
have = {p.name.removesuffix(".contigs.fa") for p in acc_dir.rglob("*.contigs.fa")}
missing = [a for a in accs if a not in have]
if missing:
    print(f"Downloading {len(missing)} of {n_accs} accessions from Logan into {acc_dir}")
    download_accessions(missing, acc_dir)
    have = {p.name.removesuffix(".contigs.fa") for p in acc_dir.rglob("*.contigs.fa")}
still = sorted(set(accs) - have)
if still:
    sys.exit(f"{len(still)} accessions have no contigs after download: {still[:5]}...")
print(f"logan_accessions: {len(have)} contigs for {n_accs} accessions")
PY

# ---- helpers ------------------------------------------------------------------
# meta.parquet holds one row per accession in a dense index. Anything short of
# the full set means the build dropped accessions.
check_dense_index() {
    local cfg="$1"
    uv run python - "$cfg" "$n_accs" <<'PY'
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

run_step() {
    # run_step METHOD CONFIG RATE
    local method="$1" cfg="$2" rate="$3"
    if is_single_node "$method"; then
        # One node, and tell run_benchmark it is alone so it neither shards
        # the build nor waits for peers.
        srun --unbuffered -N1 -n1 --cpus-per-task="$CPUS_PER_TASK" \
            uv run python scripts/run_benchmark.py --config "$cfg" --mutation_rate "$rate" --num_shards 1
    else
        srun --unbuffered --ntasks-per-node=1 \
            --gpus-per-node="$GPUS_PER_NODE" --cpus-per-task="$CPUS_PER_TASK" \
            bash -c 'exec "$@" --shard "$SLURM_NODEID" --num_shards "$SLURM_NNODES"' _ \
            uv run python scripts/run_benchmark.py --config "$cfg" --mutation_rate "$rate"
    fi
}

# ---- main loop ------------------------------------------------------------------
skipped=()
failed=()
for method in "${METHODS[@]}"; do
    cfg="$CONFIG_DIR/perlmutter_${method}_sra64viral.yaml"
    if [ ! -f "$cfg" ]; then
        echo "[skip] no config $cfg"
        skipped+=("$method (no config)")
        continue
    fi
    # shifter exits 0 even with no image, so test the allocation's image
    # request directly. sbatch exports it as SHIFTER_IMAGEREQUEST; the
    # interactive shell of an salloc only has the SLURM_SPANK_-prefixed copy.
    if [[ "$method" == metagraph* ]] && [ -z "${SHIFTER_IMAGEREQUEST:-${SLURM_SPANK_SHIFTER_IMAGEREQUEST:-}}" ]; then
        echo "[skip] $method: allocation has no shifter image; add" \
             "--image=ghcr.io/ratschlab/metagraph:master to salloc"
        skipped+=("$method (no image)")
        continue
    fi
    for rate in "${RATE_LIST[@]}"; do
        echo "=== $method @ mutation_rate $rate  ($(date))  nodes=${SLURM_JOB_NUM_NODES:-?} ==="
        if ! run_step "$method" "$cfg" "$rate"; then
            echo "[FAIL] $method @ $rate" >&2
            failed+=("$method @ $rate")
            # Any later rate needs the index this one builds, so move on to
            # the next method rather than retrying against a broken build.
            break
        fi
        if [ "$rate" = "${RATE_LIST[0]}" ] && ! is_single_node "$method"; then
            if ! check_dense_index "$cfg"; then
                failed+=("$method (truncated index)")
                break
            fi
        fi
    done
done

echo
echo "=== sra64viral results ($RESULTS) ==="
# With INDEX=sra500 this table is against sra500 only, where no run is
# relevant; the meaningful table comes from merge_index_results.py.
[ "${INDEX:-}" = sra500 ] || uv run python scripts/print_results.py "$RESULTS" "$BUNDLE/queries_mut0.00.parquet" "$BUNDLE/accs.txt" || true

echo
if [ ${#skipped[@]} -gt 0 ]; then
    echo "Skipped: ${skipped[*]}"
fi
if [ ${#failed[@]} -gt 0 ]; then
    echo "Failed: ${failed[*]}" >&2
    exit 1
fi
echo "All sra64viral runs finished."
