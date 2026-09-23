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
# so run_benchmark neither re-downloads nor indexes stray accessions. Layout
# must be a REAL directory per accession holding a symlink to the .contigs.fa:
# run_benchmark finds contigs with Path.rglob, which on Python 3.11 does not
# descend into symlinked directories (a per-accession *directory* symlink made
# it find nothing and re-download all 4571, job 58713395). Idempotent; replaces
# any leftover directory symlinks.
mkdir -p "$BUNDLE/logan_accessions"
missing=0
while read -r acc; do
  [ -n "$acc" ] || continue
  src="$CONTIGS/$acc/$acc.contigs.fa"
  dst="$BUNDLE/logan_accessions/$acc"
  if [ ! -f "$src" ]; then echo "missing contigs: $acc"; missing=$((missing+1)); continue; fi
  [ -L "$dst" ] && rm "$dst"
  mkdir -p "$dst"
  [ -e "$dst/$acc.contigs.fa" ] || ln -s "$src" "$dst/$acc.contigs.fa"
done < "$BUNDLE/accs.txt"
[ "$missing" -eq 0 ] || { echo "$missing accessions lack contigs; aborting"; exit 1; }
# Self-check with the same call run_benchmark uses; abort before srun rather
# than let 32 GPU nodes fall back to downloading.
n_found=$(uv run python -c "from pathlib import Path; print(len(list(Path('$BUNDLE/logan_accessions').rglob('*.contigs.fa'))))")
n_accs=$(wc -l < "$BUNDLE/accs.txt")
echo "logan_accessions: rglob finds $n_found contigs for $n_accs accessions"
[ "$n_found" -eq "$n_accs" ] || { echo "rglob count mismatch; aborting before srun"; exit 1; }

export PYTHONUNBUFFERED=1

srun uv run python run_benchmark.py --config configs/sra4571/perlmutter_locale_sra4571.yaml --build_only true
