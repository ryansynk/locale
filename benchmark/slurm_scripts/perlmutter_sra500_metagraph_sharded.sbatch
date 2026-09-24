#!/bin/bash
#SBATCH -A m5408
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 2
#SBATCH -t 6:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -c 256
#SBATCH --image=ghcr.io/ratschlab/metagraph:master
#SBATCH -J locale-sra500-metagraph-sharded
#SBATCH -o slurm_logs/sra500_metagraph_sharded_%j.out
#SBATCH -e slurm_logs/sra500_metagraph_sharded_%j.err

# Validation of the multi-node metagraph path before sra4571: build sra500v2
# as 2 shards (indexes/sra500_sharded/), search from node 0 over both, and
# compare against the single-node result in results/sra500/metagraph_k31/
# (perlmutter_sra500_metagraph.sbatch). Per-accession k-mer counts are
# independent of the sharding, so the per-query (accession, score) lists must
# agree exactly, up to ties at the top-100 cut. Everything under
# indexes/sra500_sharded and results/sra500_sharded is disposable.
#
#   sbatch slurm_scripts/perlmutter_sra500_metagraph_sharded.sbatch
set -euo pipefail

cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1

CONFIG=configs/sra500/perlmutter_metagraph_sra500v2_sharded.yaml
BUNDLE=/pscratch/sd/r/rsynk/locale-data/constructed/sra500/bundle
REFERENCE=results/sra500/metagraph_k31/raw_read_mut_0.0.parquet
SHARDED=results/sra500_sharded/metagraph_k31/raw_read_mut_0.0.parquet

n_accs=$(grep -c . "$BUNDLE/accs.txt")
n_found=$(uv run python -c "from pathlib import Path; print(len(list(Path('$BUNDLE/logan_accessions').rglob('*.contigs.fa'))))")
echo "logan_accessions: $n_found contigs for $n_accs accessions"
[ "$n_found" -eq "$n_accs" ] || { echo "contig count mismatch; aborting" >&2; exit 1; }

echo "=== metagraph sharded @ mutation_rate 0.0  ($(date)) ==="
srun --unbuffered --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  uv run python run_benchmark.py --config "$CONFIG" --mutation_rate 0.0

n_indexed=$(grep -c . indexes/sra500_sharded/metagraph/k31/contig_manifest.txt)
[ "$n_indexed" -eq "$n_accs" ] || { echo "[FAIL] merged index lists $n_indexed contigs, expected $n_accs" >&2; exit 1; }
echo "[ok] merged index lists $n_indexed contigs"

if [ ! -f "$REFERENCE" ]; then
    echo "[skip] no single-node reference at $REFERENCE yet; rerun the comparison by hand:"
    echo "  uv run python compare_results.py $REFERENCE $SHARDED"
    exit 0
fi
uv run python compare_results.py "$REFERENCE" "$SHARDED"
