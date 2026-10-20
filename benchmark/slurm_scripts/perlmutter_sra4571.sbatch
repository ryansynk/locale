#!/bin/bash
#SBATCH -A m5408_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 7
#SBATCH -t 4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH --array=0-31:7
#SBATCH -J locale-sra4571-embed
#SBATCH -o slurm_logs/sra4571_embed_%A_%a.out
#SBATCH -e slurm_logs/sra4571_embed_%A_%a.err

# Step 1 of 3: embed the sra4571 dense locale index, one single-node shard
# build per task. Each build runs run_benchmark with --shard S --num_shards 32,
# fans out over its node's 4 GPUs internally, and exits once its shard is
# .done (--stage embed). Then:
#   sbatch perlmutter_sra4571_merge.sbatch         (CPU, ~3h)
#   sbatch perlmutter_sra4571_search.sbatch 0.00   (GPU)
#
# Why not one 32-node job: a NODE_FAIL on any one node killed the whole
# 32-node allocation (job 59781048, 1h48m in, no shard finished, ~57
# node-hours lost). Here each shard build is independent: one failure costs
# one shard, and only the missing shards are rerun.
#
# Every run picks shard = BASE + SLURM_NODEID on each node of its allocation,
# so an N-node allocation builds the contiguous block BASE..BASE+N-1 (nodes
# whose shard would be >= NUM_SHARDS exit at once).
#
# Default: a premium job array of 5 tasks x 7 nodes (--array=0-31:7, -N 7).
# gpu_premium caps a user at 5 submitted jobs and every array task counts,
# so 5 tasks is the most premium allows; 7 nodes each covers 32 shards, the
# last task (shards 28-31) idling 3 of its nodes. Each task queues on its own,
# so a 7-node slot is found far sooner than a 32-node one, and a NODE_FAIL
# costs at most one task's 7 shards.
#   CONFIG=... sbatch slurm_scripts/perlmutter_sra4571.sbatch
# Zero-idle variant, same 5 jobs, two submissions (7+7+6+6+6 nodes):
#   CONFIG=... sbatch -N 7 --array=0,7       slurm_scripts/perlmutter_sra4571.sbatch
#   CONFIG=... sbatch -N 6 --array=14,20,26  slurm_scripts/perlmutter_sra4571.sbatch
# Rerun missing shards with -N 1 and explicit ids, e.g. -N 1 --array=3,17,29.
#
# Interactive allocation (BASE = SHARD_BASE): the interactive QOS gives 4
# nodes for 4 h with little wait, 2 jobs per user, but refuses sbatch, so
# run the script as the salloc command (4 shards per allocation):
#   CONFIG=... SHARD_BASE=0 salloc -q interactive -A m5408_g -C gpu -N 4 \
#     -t 4:00:00 --ntasks-per-node=1 --gpus-per-node=4 -c 128 \
#     bash slurm_scripts/perlmutter_sra4571.sbatch
# Shards from either route land in the same shard_<S>/ dirs and the merge
# step does not care which built them.
#
# Timing: the 100studies embed took 2h40m per shard on 32 nodes, so expect
# ~2.5-3h per shard (-t 4:00:00).
#
# Shard striping depends on NUM_SHARDS, not on which shards a run covers:
# shard_* under the index dir must be empty on the first run at a new
# NUM_SHARDS, and every rerun must use the SAME NUM_SHARDS so completed
# shards (.done) are valid to reuse. A shard without .done is rebuilt from
# scratch (its partial embeddings.fbin is overwritten); a shard with .done
# is skipped in seconds, so re-covering a range is cheap. Rerun only the
# missing shards with e.g.  sbatch --array=3,17,29 ...
#
# Usage: [CONFIG=... NUM_SHARDS=32] sbatch [-N n --array=...] perlmutter_sra4571.sbatch
#    or: [CONFIG=... NUM_SHARDS=32] SHARD_BASE=<n> salloc ... bash perlmutter_sra4571.sbatch

set -euo pipefail

cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs

# Which encoder to embed; the index directory comes from its encoder_label.
# Default: the end-window + min_seq_len index (the original locale@8vqiabk9
# index is already built; submitting with that config is a 40 s no-op on
# every node, array 59965075). CONFIG=... sbatch ... picks another.
CONFIG=${CONFIG:-configs/sra4571/perlmutter_locale_sra4571_endwin0_minlen100.yaml}
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG"; exit 1; }
NUM_SHARDS=${NUM_SHARDS:-32}
NNODES=${SLURM_NNODES:?no Slurm allocation: run under sbatch --array or salloc}
BASE=${SLURM_ARRAY_TASK_ID:-${SHARD_BASE:-}}
[ -n "$BASE" ] || { echo "set SHARD_BASE=<first shard> when not running as an array task"; exit 1; }
[ "$BASE" -lt "$NUM_SHARDS" ] || { echo "BASE=$BASE >= NUM_SHARDS=$NUM_SHARDS: nothing to build"; exit 1; }
LAST=$((BASE + NNODES - 1))
if [ "$LAST" -ge "$NUM_SHARDS" ]; then
  echo "block $BASE-$LAST runs past NUM_SHARDS=$NUM_SHARDS; $((LAST - NUM_SHARDS + 1)) node(s) will idle"
  LAST=$((NUM_SHARDS - 1))
fi
echo "config: $CONFIG  shards: $BASE-$LAST of $NUM_SHARDS  job: ${SLURM_JOB_ID}  nodes: ${SLURM_JOB_NODELIST}"

# Refuse to burn GPU nodes on an index that is already merged and .done:
# run_benchmark would have every rank exit at once with "Index built".
INDEX_PATH=$(uv run python -c "
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(c['index_dir'].rstrip('/') + '/' + c['model']['encoder_label'])" "$CONFIG")
echo "index: $INDEX_PATH"
[ ! -f "$INDEX_PATH/.done" ] || { echo "$INDEX_PATH is already .done; nothing to embed"; exit 1; }
echo "shards already .done: $(ls -d "$INDEX_PATH"/shard_*/.done 2>/dev/null | wc -l) of $NUM_SHARDS"

BUNDLE=/pscratch/sd/r/rsynk/locale-data/constructed/sra4571/bundle
CONTIGS=/pscratch/sd/r/rsynk/locale-data/constructed/sra4571/work/logan_contigs

# run_benchmark finds contigs with Path.rglob under logan_accessions and
# falls back to downloading all 4571 accessions when it finds nothing. The
# links are made ONCE, on a login node, before submitting (32 tasks linking
# concurrently would race on the same directories):
#   BUNDLE=$BUNDLE CONTIGS=$CONTIGS bash slurm_scripts/biowulf_link_accessions.sh
# Here only verify, with the same rglob call, and abort the task rather than
# let a GPU node fall back to downloading.
n_found=$(uv run python -c "from pathlib import Path; print(len(list(Path('$BUNDLE/logan_accessions').rglob('*.contigs.fa'))))")
n_accs=$(grep -c . "$BUNDLE/accs.txt")
echo "logan_accessions: rglob finds $n_found contigs for $n_accs accessions"
[ "$n_found" -eq "$n_accs" ] || {
  echo "rglob count mismatch; run biowulf_link_accessions.sh (see comment) and resubmit"; exit 1; }

export PYTHONUNBUFFERED=1 BASE NUM_SHARDS

# One task per node; each picks its shard from its node rank.
# One task per node; each picks its shard from its node rank. Ranks past the
# last shard exit immediately (the allocation is still held by the others).
srun -N "$NNODES" --ntasks-per-node=1 bash -c '
  s=$((BASE + SLURM_NODEID))
  [ "$s" -lt "$NUM_SHARDS" ] || { echo "[rank $SLURM_NODEID] no shard $s; idle"; exit 0; }
  exec "$@" --shard "$s" --num_shards "$NUM_SHARDS"' _ \
  uv run python run_benchmark.py --config "$CONFIG" --stage embed
