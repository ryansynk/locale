#!/bin/bash
#SBATCH -A m5408_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 4
#SBATCH -t 2:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -J locale-sra4571-search
#SBATCH -o slurm_logs/sra4571_search_%j.out
#SBATCH -e slurm_logs/sra4571_search_%j.err

# Step 3 of 3: multi-node search over the merged sra4571 dense index.
# Mutation rate is the first argument (0.00 / 0.05 / 0.10), passed as a CLI
# override of the config's mutation_rate. The index must be .done (run the
# merge first); all nodes skip build and go straight to search.
#
# Default config: configs/sra4571/perlmutter_locale_sra4571_exact_topk1000.yaml, the
# exact top-1000 regroup over both strands. Each node scans an equal range of the fbin's vector
# rows for every query's top-1000 nearest vectors; node 0 unions the partials,
# re-takes the global top-1000, regroups to accessions and writes
#   results/sra4571/<experiment_id>_exacttop1000_bothstrands/raw_read_mut_<rate>.parquet
# The raw hits are persisted under results/sra4571_topk_hits/<hits_id>/ and
# serve any later both-strands run of the same encoder at k <= 1000 without a
# rescan, which is what makes the later ANN-vs-exact comparison cheap.
#
# Optional second argument: another config over the same index, e.g.
# configs/sra4571/perlmutter_locale_sra4571.yaml (default protocol, k=100 -- note it
# is NOT the streaming per-accession baseline unless it sets exhaustive: true)
# or configs/sra4571/perlmutter_locale_sra4571_bothstrands.yaml. Both-strand hits live
# in a separate hits_id, so a both-strands run does not reuse single-strand hits.
#
# The three rates write distinct result and hits files and can run
# concurrently. Partials are keyed by (mutation_rate, num_queries, seed) and
# reused on rerun, so a resubmit after a timeout resumes rather than rescoring.
# Sizing: the 8-node run of 2026-09-22 scanned each node's 257M-row range in
# ~15 min, so 4 nodes (515M rows each) finish in ~30 min plus query embedding
# and merge; 2h is generous headroom.
MUT_RATE="${1:?usage: sbatch perlmutter_sra4571_search.sbatch <mutation_rate> [config.yaml]}"
CONFIG="${2:-configs/sra4571/perlmutter_locale_sra4571_exact_topk1000.yaml}"

set -euo pipefail

cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
# The exact scan allocates and frees a ~15 GB logits tile per block; without
# expandable segments the caching allocator fragments and can fail a
# 15 GB request with 12 GB reported free (seen on the 2026-09-22 runs).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --unbuffered: srun line-buffers task output, and tqdm's \r updates carry
# no newline, so without it the progress bars only appear at exit.
srun --unbuffered --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=128 \
  uv run python run_benchmark.py \
    --config "$CONFIG" \
    --mutation_rate "$MUT_RATE"
