#!/bin/bash
#SBATCH -A m5408
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J locale-sra500-binned
#SBATCH -o slurm_logs/sra500_binned_%j.out
#SBATCH -e slurm_logs/sra500_binned_%j.err

# Post hoc stratification of the sra500 results by how close each query is to
# the LOCALE reference-genome training set. No benchmark is re-run: the clean
# queries are aligned to the 7 reference assemblies with minimap2, binned by
# best identity (unaligned, <0.90, 0.90-0.95, 0.95-0.99, >=0.99), and
# print_results.py is called once per bin with that bin's queries parquet
# (every metric inner-joins results to the queries file on query_id, so the
# filtered file restricts the scoring). An "all" run on the full bundle is
# included for reference.
#
# Needs a CPU node: the human and pea indexes take ~30-45 GB each.
#   sbatch scripts/run_sra500_binned.sh
#   RESULTS=results/sra50 BUNDLE=... OUT=results/sra50_binned sbatch scripts/run_sra500_binned.sh
#
# Inside an existing interactive allocation (salloc -N 1 -C cpu -q interactive),
# run it as a plain script; the SBATCH lines above are inert comments:
#   JOBS=7 THREADS=36 bash scripts/run_sra500_binned.sh
set -euo pipefail

cd /pscratch/sd/r/rsynk/locale/benchmark

REFS=${REFS:-/pscratch/sd/r/rsynk/reference_data/refs}
BUNDLE=${BUNDLE:-/pscratch/sd/r/rsynk/locale-data/constructed/sra500/bundle}
RESULTS=${RESULTS:-results/sra500}
OUT=${OUT:-results/sra500_binned}
JOBS=${JOBS:-4}
# minimap2 threads per concurrent assembly. Unset -> the python script uses
# SLURM_CPUS_PER_TASK / JOBS, which is right under sbatch but can come out as
# 1 inside an salloc that did not pass -c, so set THREADS explicitly there.
THREADS=${THREADS:-}
mkdir -p "$OUT"

echo "== aligning queries to references -> $OUT"
uv run python scripts/bin_queries_by_reference.py \
    --refs_dir "$REFS" --bundle "$BUNDLE" --out_dir "$OUT" --jobs "$JOBS" \
    ${THREADS:+--threads "$THREADS"}

# Untruncated model ids so summarize_binned.py can key on them.
export POLARS_FMT_STR_LEN=120

echo "== scoring: all queries"
uv run python scripts/print_results.py "$RESULTS" "$BUNDLE/queries.parquet" "$BUNDLE/accs.txt" \
    > "$OUT/print_results_all.txt"

for q in "$OUT"/queries/queries_*.parquet; do
    bin=$(basename "$q" .parquet); bin=${bin#queries_}
    echo "== scoring: bin $bin"
    uv run python scripts/print_results.py "$RESULTS" "$q" "$BUNDLE/accs.txt" \
        > "$OUT/print_results_${bin}.txt"
done

echo "== summary"
uv run python scripts/summarize_binned.py --out_dir "$OUT" \
    --models 'locale@8vqiabk9/exact/top100|w128|metagraph|mmseqs|esa' | tee "$OUT/summary.txt"
