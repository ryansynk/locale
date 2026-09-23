#!/usr/bin/env bash
#
# Run every configs/sra55viral/*.yaml on the clean queries (mutation rate
# 0.00), one srun step at a time, then print the results table. Run it from
# inside a GPU allocation, e.g.
#
#   salloc -A m5408_g -C gpu -q interactive -N 4 -t 4:00:00 \
#          --ntasks-per-node=1 --gpus-per-node=4 -c 128 \
#          --image=ghcr.io/ratschlab/metagraph:master
#   cd /pscratch/sd/r/rsynk/locale/benchmark && ./run_sra55viral.sh
#
# The --image is only needed for metagraph (executable: shifter metagraph);
# without it that method is skipped, the rest still run.
#
# Before any step, the 55 accessions' contigs are fetched from Logan into the
# bundle's logan_accessions/ if missing -- once, on this node. Left to
# run_benchmark, every node of a multi-node step would download into the same
# directory at once.
#
# Dense methods (locale, esa, llmed) run as one process per node across the
# whole allocation: run_benchmark.py reads SLURM_NODEID / SLURM_NNODES, builds
# one index shard (or scans one range of vector rows) per node, and rank 0
# merges. metagraph and mmseqs2 are single-node methods (the sharded build
# path merges dense indexes only), so they run on one node with the node-count
# variables forced to 1. Plain `python run_benchmark.py` inside a multi-node
# allocation would be a lone rank 0 waiting forever for peers: always srun.
#
# Only the clean queries are run: the divergence under test is the real
# genotype-D-vs-B difference (~11%), not simulated read errors, so the
# mut0.05 / mut0.10 files in the bundle are deliberately not used. RATES can
# still add them if ever wanted. After a dense build the index is checked for
# the full accession count: a build that loses a worker still gets its .done
# marker and would report quietly wrong recall.
#
#   ./run_sra55viral.sh                     # all five methods
#   ./run_sra55viral.sh locale esa          # a subset
#   RATES="0.0 0.05 0.1" ./run_sra55viral.sh   # also the synthetic-noise rates
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
    METHODS=(locale esa llmed metagraph mmseqs2)
fi
read -r -a RATE_LIST <<<"${RATES:-0.0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-128}"
BUNDLE=/pscratch/sd/r/rsynk/locale-data/constructed/sra55viral/bundle
RESULTS=/pscratch/sd/r/rsynk/locale/benchmark/results/sra55viral
SINGLE_NODE_METHODS=" metagraph mmseqs2 "

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- contigs: fetch once, verify, before any multi-node step -----------------
n_accs=$(grep -c . "$BUNDLE/accs.txt")
uv run python - "$BUNDLE" "$n_accs" <<'PY'
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
import yaml
from pathlib import Path
from src.config import DenseConfig, ExperimentConfig

cfg_path, n_accs = sys.argv[1], int(sys.argv[2])
raw = yaml.safe_load(open(cfg_path))
model = DenseConfig(**raw.pop("model"))
raw = {k: (Path(v) if k.endswith("_dir") else v) for k, v in raw.items()}
cfg = ExperimentConfig(model=model, **raw)
meta = cfg.index_dir / model.index_suffix / "meta.parquet"
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
    if [[ "$SINGLE_NODE_METHODS" == *" $method "* ]]; then
        # One node, and tell run_benchmark it is alone so it neither shards
        # the build nor waits for peers.
        # /usr/bin/env by absolute path: ~/.local/bin/env shadows it on PATH
        # here and is not executable (exit 13 from execve, 2026-09-23).
        srun --unbuffered -N1 -n1 --cpus-per-task="$CPUS_PER_TASK" \
            /usr/bin/env SLURM_NNODES=1 SLURM_NODEID=0 \
            uv run python run_benchmark.py --config "$cfg" --mutation_rate "$rate"
    else
        srun --unbuffered --ntasks-per-node=1 \
            --gpus-per-node="$GPUS_PER_NODE" --cpus-per-task="$CPUS_PER_TASK" \
            uv run python run_benchmark.py --config "$cfg" --mutation_rate "$rate"
    fi
}

# ---- main loop ------------------------------------------------------------------
skipped=()
failed=()
for method in "${METHODS[@]}"; do
    cfg="configs/sra55viral/perlmutter_${method}_sra55viral.yaml"
    if [ ! -f "$cfg" ]; then
        echo "[skip] no config $cfg"
        skipped+=("$method (no config)")
        continue
    fi
    # shifter exits 0 even with no image, so test the allocation's image
    # request directly.
    if [ "$method" = metagraph ] && [ -z "${SHIFTER_IMAGEREQUEST:-}" ]; then
        echo "[skip] metagraph: allocation has no shifter image; add" \
             "--image=ghcr.io/ratschlab/metagraph:master to salloc"
        skipped+=("metagraph (no image)")
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
        if [ "$rate" = "${RATE_LIST[0]}" ] && [[ "$SINGLE_NODE_METHODS" != *" $method "* ]]; then
            if ! check_dense_index "$cfg"; then
                failed+=("$method (truncated index)")
                break
            fi
        fi
    done
done

echo
echo "=== sra55viral results ==="
uv run python print_results.py "$RESULTS" "$BUNDLE/queries_mut0.00.parquet" "$BUNDLE/accs.txt" || true

echo
if [ ${#skipped[@]} -gt 0 ]; then
    echo "Skipped: ${skipped[*]}"
fi
if [ ${#failed[@]} -gt 0 ]; then
    echo "Failed: ${failed[*]}" >&2
    exit 1
fi
echo "All sra55viral runs finished."
