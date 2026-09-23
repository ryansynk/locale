#!/bin/bash
#SBATCH -A m5408_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 4
#SBATCH -t 4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -J locale-sra500-esa
#SBATCH -o slurm_logs/sra500_esa_%j.out
#SBATCH -e slurm_logs/sra500_esa_%j.err

# Embed-Search-Align (dna2vec, "ESA") on sra500v2 at every eval mutation rate,
# 4 nodes x 4 GPUs. Each srun step is one process per node: the first rate
# builds one index shard per node and rank 0 merges (all ranks then stay for
# the search), every rate scans its range of vector rows on all four nodes
# and rank 0 merges the top-100 hits. Later rates reuse the .done index.
#
# Results: results/sra500/dna2vec_maxlen256_poolmax_chunkstride_exacttop100_bothstrands/
# raw hits: results/sra500_topk_hits/<...>_exact_bothstrands/ (reused at k <= 100).
#
#   sbatch slurm_scripts/perlmutter_sra500_esa.sbatch            # all rates
#   RATES="0.0" sbatch slurm_scripts/perlmutter_sra500_esa.sbatch
set -euo pipefail

cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1
# The exact scan allocates and frees a large logits tile per block; expandable
# segments keep the caching allocator from fragmenting.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=configs/sra500/perlmutter_esa_sra500v2.yaml
BUNDLE=/pscratch/sd/r/rsynk/locale-data/constructed/sra500/bundle
read -r -a RATE_LIST <<<"${RATES:-0.0 0.05 0.1}"

# Abort before srun unless every contig is linked (otherwise all four nodes
# would re-download the 500 accessions from Logan at once).
n_accs=$(grep -c . "$BUNDLE/accs.txt")
n_found=$(uv run python -c "from pathlib import Path; print(len(list(Path('$BUNDLE/logan_accessions').rglob('*.contigs.fa'))))")
echo "logan_accessions: $n_found contigs for $n_accs accessions"
[ "$n_found" -eq "$n_accs" ] || { echo "contig count mismatch; aborting" >&2; exit 1; }

for rate in "${RATE_LIST[@]}"; do
    echo "=== esa @ mutation_rate $rate  ($(date)) ==="
    # --unbuffered: srun line-buffers task output and tqdm's \r updates carry
    # no newline, so without it the progress bars only appear at exit.
    srun --unbuffered --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=128 \
      uv run python run_benchmark.py --config "$CONFIG" --mutation_rate "$rate"
done

# A build that loses a worker still gets its .done marker; make sure the
# merged index holds every accession before trusting the numbers.
uv run python - "$CONFIG" "$n_accs" <<'PY'
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
n = pl.read_parquet(meta)["srr_id"].n_unique()
if n != n_accs:
    sys.exit(f"[FAIL] index has {n} accessions, expected {n_accs}: {meta}")
print(f"[ok] index has {n} accessions")
PY

uv run python print_results.py results/sra500 "$BUNDLE/queries_mut0.00.parquet" "$BUNDLE/accs.txt" || true
