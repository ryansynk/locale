#!/bin/bash
#SBATCH -A m5408_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 32
#SBATCH -t 5:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -J locale-sra4571-embed
#SBATCH -o slurm_logs/sra4571_embed_%j.out
#SBATCH -e slurm_logs/sra4571_embed_%j.err

# Step 1 of 3: 32-node embed of the sra4571 dense locale index.
# Reference: the 5092-accession 100studies embed (job 58208345) took 2h40m on
# 32 nodes (~85 node-hours); 4571 accessions is ~90% of that, so expect
# ~2.5h. --build_only makes every node exit once its shard is .done, so the
# GPU allocation is not held for the serial merge. Then:
#   sbatch perlmutter_sra4571_merge.sbatch         (CPU, ~3h)
#   sbatch perlmutter_sra4571_search.sbatch 0.00   (GPU)
#
# One task per node: run_benchmark shards accessions by SLURM_NODEID and each
# task fans out over its node's 4 GPUs internally. Shard striping depends on
# the node count: shard_* under indexes_sra4571 must be empty on the first run
# at a new -N, and a resubmission after a timeout must reuse the SAME -N so
# completed shards (.done) are valid to reuse.

set -euo pipefail

cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs

BUNDLE=/pscratch/sd/r/rsynk/locale-data/constructed/sra4571/bundle
CONTIGS=/pscratch/sd/r/rsynk/locale-data/constructed/sra4571/work/logan_contigs

# finalize_query_dataset.py writes accs.txt + queries but not logan_accessions/.
# Link exactly the manifest accessions from the already-downloaded contig tree
# (layout <ACC>/<ACC>.contigs.fa, same as src/download_accessions.py) so
# run_benchmark neither re-downloads nor indexes stray accessions. Idempotent.
mkdir -p "$BUNDLE/logan_accessions"
missing=0
while read -r acc; do
  [ -n "$acc" ] || continue
  src="$CONTIGS/$acc"
  if [ ! -f "$src/$acc.contigs.fa" ]; then echo "missing contigs: $acc"; missing=$((missing+1)); continue; fi
  [ -e "$BUNDLE/logan_accessions/$acc" ] || ln -s "$src" "$BUNDLE/logan_accessions/$acc"
done < "$BUNDLE/accs.txt"
[ "$missing" -eq 0 ] || { echo "$missing accessions lack contigs; aborting"; exit 1; }
echo "logan_accessions: $(ls "$BUNDLE/logan_accessions" | wc -l) links for $(wc -l < "$BUNDLE/accs.txt") accessions"

export PYTHONUNBUFFERED=1

srun uv run python run_benchmark.py --config configs/perlmutter_locale_sra4571.yaml --build_only true
