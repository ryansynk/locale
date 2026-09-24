#!/bin/bash
#SBATCH -A m5408
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J locale-sra55viral-mmseqs
#SBATCH -o slurm_logs/sra55viral_mmseqs_%j.out
#SBATCH -e slurm_logs/sra55viral_mmseqs_%j.err

# MMseqs2 alignment baseline on sra55viral on the clean queries (0% only, the sra55viral protocol), over ONE
# database of all 55 accessions' contigs (src/mmseqs2_index.py, rebuilt
# 2026-09-24; the earlier per-accession layout is archived under
# indexes/sra55viral/mmseqs_per_accession_old and
# results_archive/sra55viral_mmseqs_per_accession). Single CPU node: mmseqs is not
# sharded. The first rate builds indexes/sra55viral/mmseqs/ and, with --do_timing,
# also times the 500-query search (5 runs, first discarded); the other rates
# reuse the index. Results: results/sra55viral/mmseqs/raw_read_mut_<rate>.parquet.
#
#   sbatch slurm_scripts/perlmutter_sra55viral_mmseqs.sbatch            # all rates
#   RATES="0.0" sbatch slurm_scripts/perlmutter_sra55viral_mmseqs.sbatch
set -euo pipefail

cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1
export PATH=/pscratch/sd/r/rsynk/mmseqs/bin:$PATH

CONFIG=configs/sra55viral/perlmutter_mmseqs2_sra55viral.yaml
BUNDLE=/pscratch/sd/r/rsynk/locale-data/constructed/sra55viral/bundle
read -r -a RATE_LIST <<<"${RATES:-0.0}"

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

uv run python print_results.py results/sra55viral "$BUNDLE/queries_mut0.00.parquet" "$BUNDLE/accs.txt" || true
