#!/bin/bash
#SBATCH -A m5408
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -c 256
#SBATCH --image=ghcr.io/ratschlab/metagraph:master
#SBATCH -J locale-sra500-metagraph
#SBATCH -o slurm_logs/sra500_metagraph_%j.out
#SBATCH -e slurm_logs/sra500_metagraph_%j.err

# Metagraph k-mer baseline on sra500v2 at every eval mutation rate. One CPU
# node: 500 accessions build comfortably on one node, and this run is the
# reference the 2-node perlmutter_sra500_metagraph_sharded.sbatch is checked
# against. The whole node's 256 hardware threads go to the
# single task; build_metagraph.sh sizes its parallelism from
# SLURM_CPUS_PER_TASK. The container is requested on the job (--image), which
# is why this is a batch script rather than a run_sra500.sh step.
#
# The first rate builds the index under indexes/sra500/metagraph_k31/ and
# marks it .done; the other two reuse it. Results land in
# results/sra500/metagraph_k31/raw_read_mut_<rate>.parquet.
#
#   sbatch slurm_scripts/perlmutter_sra500_metagraph.sbatch            # all rates
#   RATES="0.0" sbatch slurm_scripts/perlmutter_sra500_metagraph.sbatch
set -euo pipefail

cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1

CONFIG=configs/sra500/perlmutter_metagraph_sra500v2.yaml
BUNDLE=/pscratch/sd/r/rsynk/locale-data/constructed/sra500/bundle
read -r -a RATE_LIST <<<"${RATES:-0.0 0.05 0.1}"

# Abort before srun unless every contig is linked (run_benchmark would
# otherwise re-download all 500 from Logan).
n_accs=$(grep -c . "$BUNDLE/accs.txt")
n_found=$(uv run python -c "from pathlib import Path; print(len(list(Path('$BUNDLE/logan_accessions').rglob('*.contigs.fa'))))")
echo "logan_accessions: $n_found contigs for $n_accs accessions"
[ "$n_found" -eq "$n_accs" ] || { echo "contig count mismatch; aborting" >&2; exit 1; }

for rate in "${RATE_LIST[@]}"; do
    echo "=== metagraph @ mutation_rate $rate  ($(date)) ==="
    # /usr/bin/env by absolute path: ~/.local/bin/env shadows it on PATH and
    # is not executable. SLURM_NNODES=1 keeps run_benchmark on the
    # single-node build path.
    srun --unbuffered -N1 -n1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
      /usr/bin/env SLURM_NNODES=1 SLURM_NODEID=0 \
      uv run python run_benchmark.py --config "$CONFIG" --mutation_rate "$rate"
done

uv run python print_results.py results/sra500 "$BUNDLE/queries_mut0.00.parquet" "$BUNDLE/accs.txt" || true
