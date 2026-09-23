#!/usr/bin/env bash
#
# Run the configs/sra500/*.yaml methods at every eval mutation rate, one srun
# step at a time, then print the results table. Run it from inside a GPU
# allocation, e.g.
#
#   salloc -A m5408_g -C gpu -q interactive -N 4 -t 4:00:00 \
#          --ntasks-per-node=1 --gpus-per-node=4 -c 128 \
#          --image=ghcr.io/ratschlab/metagraph:master
#   cd /pscratch/sd/r/rsynk/locale/benchmark && ./run_sra500.sh
#
# The --image is only needed for metagraph (executable: shifter metagraph);
# without it that method is skipped, the rest still run.
#
#   ./run_sra500.sh                     # locale esa llmed metagraph, all rates
#   ./run_sra500.sh locale              # one method, all rates
#   ./run_sra500.sh locale llmed        # a subset
#   RATES="0.0" ./run_sra500.sh esa     # index build + clean-query eval only
#
# Dense methods (locale, esa, llmed) run as one process per node across the
# whole allocation: run_benchmark.py reads SLURM_NODEID / SLURM_NNODES, builds
# one index shard per node, rank 0 merges, then every node scans its range of
# vector rows and rank 0 merges the hits. metagraph is a single-node method
# (the sharded build path merges dense indexes only), so it runs on one node
# with the node-count variables forced to 1. Plain `python run_benchmark.py`
# inside a multi-node allocation would be a lone rank 0 waiting forever for
# peers: always srun.
#
# Per method the first rate builds the index; later rates reuse it (the index
# depends on the model, not on the mutation rate). After a dense build the
# index is checked for the full accession count: a build that loses a worker
# still gets its .done marker and would report quietly wrong recall.
#
# The bundle's logan_accessions/ must already link the 500 contigs (it does,
# laid out 2026-09-23 from work/logan_contigs); the check below aborts before
# any srun if it does not, rather than let four nodes re-download at once.
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
    METHODS=(locale esa llmed metagraph)
fi
read -r -a RATE_LIST <<<"${RATES:-0.0 0.05 0.1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-128}"
BUNDLE=/pscratch/sd/r/rsynk/locale-data/constructed/sra500/bundle
RESULTS=/pscratch/sd/r/rsynk/locale/benchmark/results/sra500
SINGLE_NODE_METHODS=" metagraph "

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- contigs present? (same rglob run_benchmark uses) -------------------------
n_accs=$(grep -c . "$BUNDLE/accs.txt")
n_found=$(uv run python -c "from pathlib import Path; print(len(list(Path('$BUNDLE/logan_accessions').rglob('*.contigs.fa'))))")
echo "logan_accessions: $n_found contigs for $n_accs accessions"
if [ "$n_found" -ne "$n_accs" ]; then
    echo "Contig count mismatch; link work/logan_contigs into $BUNDLE/logan_accessions" \
         "(see slurm_scripts/perlmutter_sra4571.sbatch) before running." >&2
    exit 1
fi

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
        # One node, told it is alone so it neither shards the build nor waits
        # for peers. /usr/bin/env by absolute path: ~/.local/bin/env shadows
        # it on PATH here and is not executable.
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
    cfg="configs/sra500/perlmutter_${method}_sra500v2.yaml"
    if [ ! -f "$cfg" ]; then
        echo "[skip] no config $cfg (methods: locale esa llmed metagraph)"
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
            # Later rates need the index this rate builds, so move on to the
            # next method rather than retrying against a broken build.
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
echo "=== sra500v2 results ==="
uv run python print_results.py "$RESULTS" "$BUNDLE/queries_mut0.00.parquet" "$BUNDLE/accs.txt" || true

echo
if [ ${#skipped[@]} -gt 0 ]; then
    echo "Skipped: ${skipped[*]}"
fi
if [ ${#failed[@]} -gt 0 ]; then
    echo "Failed: ${failed[*]}" >&2
    exit 1
fi
echo "All sra500v2 runs finished."
