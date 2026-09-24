#!/bin/bash
#SBATCH -A m5408
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 8
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -c 256
#SBATCH --image=ghcr.io/ratschlab/metagraph:master
#SBATCH -J locale-sra4571-metagraph
#SBATCH -o slurm_logs/sra4571_metagraph_%j.out
#SBATCH -e slurm_logs/sra4571_metagraph_%j.err

# Metagraph k-mer baseline on sra4571, sharded over 8 CPU nodes. Node r builds
# a complete metagraph index of accessions[r::8] under
# indexes/sra4571/metagraph/k31/shard_r/ (build_metagraph.sh, with disk swap
# and memory caps sized from node RAM), node 0 then writes graphs.csv listing
# every shard and marks the index .done, and node 0 alone starts one
# server_query over all shards for the search. Per-accession k-mer counts do
# not depend on how accessions are split across graphs, so the results equal
# a single joint index; index_size_gb is the sum of the shards, which
# overstates a joint index by the duplicated graph part.
#
# Shard striping depends on -N: shard_* must be empty on the first run at a
# new node count, and a resubmission after a timeout must reuse the SAME -N so
# completed shards (.done) are valid to reuse. Sizing: one shard is ~570
# accessions / ~37 GB of contigs, about 1.6x sra500, whose single-node build
# is the reference (slurm_logs/sra500_metagraph_*.err has per-stage timings).
#
# The first rate builds; the other rates reuse the .done index (7 nodes sit
# idle for those short searches). Results:
#   results/sra4571/metagraph_k31/raw_read_mut_<rate>.parquet
#
#   sbatch slurm_scripts/perlmutter_sra4571_metagraph.sbatch            # all rates
#   RATES="0.0" sbatch slurm_scripts/perlmutter_sra4571_metagraph.sbatch
set -euo pipefail

cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1

CONFIG=configs/sra4571/perlmutter_metagraph_sra4571.yaml
BUNDLE=/pscratch/sd/r/rsynk/locale-data/constructed/sra4571/bundle
read -r -a RATE_LIST <<<"${RATES:-0.0 0.05 0.1}"

# Abort before srun unless every contig is linked (run_benchmark would
# otherwise have all 8 nodes re-download 4571 accessions from Logan at once).
n_accs=$(grep -c . "$BUNDLE/accs.txt")
n_found=$(uv run python -c "from pathlib import Path; print(len(list(Path('$BUNDLE/logan_accessions').rglob('*.contigs.fa'))))")
echo "logan_accessions: $n_found contigs for $n_accs accessions"
[ "$n_found" -eq "$n_accs" ] || { echo "contig count mismatch; aborting" >&2; exit 1; }

for rate in "${RATE_LIST[@]}"; do
    echo "=== metagraph @ mutation_rate $rate  ($(date)) ==="
    # One task per node; run_benchmark reads SLURM_NODEID / SLURM_NNODES to
    # pick its shard. --unbuffered so metagraph's progress lines appear live.
    srun --unbuffered --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
      uv run python run_benchmark.py --config "$CONFIG" --mutation_rate "$rate"
done

# Every accession must be in the merged manifest before trusting the numbers.
n_indexed=$(grep -c . indexes/sra4571/metagraph/k31/contig_manifest.txt)
[ "$n_indexed" -eq "$n_accs" ] || { echo "[FAIL] merged index lists $n_indexed contigs, expected $n_accs" >&2; exit 1; }
echo "[ok] merged index lists $n_indexed contigs"

uv run python print_results.py results/sra4571 "$BUNDLE/queries_mut0.00.parquet" "$BUNDLE/accs.txt" || true
