#!/bin/bash
#SBATCH -A m5408
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 8:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -c 256
#SBATCH -J locale-sra500-mmseqs
#SBATCH -o slurm_logs/sra500_mmseqs_%j.out
#SBATCH -e slurm_logs/sra500_mmseqs_%j.err

# MMseqs2 alignment baseline on sra500v2 at every eval mutation rate, over ONE
# database of all 500 accessions' contigs (~23 GB; src/mmseqs2_index.py).
# Single CPU node: mmseqs is not sharded. The first rate builds
# indexes/sra500/mmseqs/ (createdb + createindex) and, with --do_timing, times
# the 500-query search (5 runs, first discarded); the other rates reuse the
# index. sra50 (1.6 GB of contigs) took 4 min end to end; the prefilter scales
# with database size, so 8 h is headroom, not an estimate.
# Results: results/sra500/mmseqs/raw_read_mut_<rate>.parquet.
#
#   sbatch slurm_scripts/perlmutter_sra500_mmseqs.sbatch            # all rates
#   RATES="0.0" sbatch slurm_scripts/perlmutter_sra500_mmseqs.sbatch
set -euo pipefail

cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1
export PATH=/pscratch/sd/r/rsynk/mmseqs/bin:$PATH

CONFIG=configs/sra500/perlmutter_mmseqs2_sra500v2.yaml
BUNDLE=/pscratch/sd/r/rsynk/locale-data/constructed/sra500/bundle
read -r -a RATE_LIST <<<"${RATES:-0.0 0.05 0.1}"

n_accs=$(grep -c . "$BUNDLE/accs.txt")
n_found=$(uv run python -c "from pathlib import Path; print(len(list(Path('$BUNDLE/logan_accessions').rglob('*.contigs.fa'))))")
echo "logan_accessions: $n_found contigs for $n_accs accessions"
[ "$n_found" -eq "$n_accs" ] || { echo "contig count mismatch; aborting" >&2; exit 1; }

for rate in "${RATE_LIST[@]}"; do
    echo "=== mmseqs @ mutation_rate $rate  ($(date)) ==="
    timing=()
    [ "$rate" = "0.0" ] && timing=(--do_timing true)
    # SLURM_NNODES=1 keeps run_benchmark on the single-node build path.
    srun --unbuffered -N1 -n1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
      /usr/bin/env SLURM_NNODES=1 SLURM_NODEID=0 \
      uv run python run_benchmark.py --config "$CONFIG" --mutation_rate "$rate" "${timing[@]}"
done

uv run python print_results.py results/sra500 "$BUNDLE/queries_mut0.00.parquet" "$BUNDLE/accs.txt" || true
