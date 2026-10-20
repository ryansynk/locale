#!/bin/bash
# Submit the whole sra4571 pipeline on Biowulf as dependency-chained jobs:
#   embed   16-task job array, 1 node of 4x A100-80GB per shard (GPU)
#   merge   1 CPU job, afterok the embed array
#   search  one 4-task job array per noise level (0.0 0.05 0.1), afterok merge
# Run from a login node:
#   bash slurm_scripts/biowulf_sra4571.sh            # everything
#   FROM=merge  bash slurm_scripts/biowulf_sra4571.sh # shards already .done
#   FROM=search bash slurm_scripts/biowulf_sra4571.sh # index already merged
# Knobs (env): CONFIG (encoder config), EMBED_SHARDS (16), SEARCH_SHARDS (4),
# RATES ("0.0 0.05 0.1"), SEARCH_CONFIG (config for the search arrays,
# default CONFIG). Logs: /data/synkrp/locale/slurm_logs/sra4571_<step>_*.out
#
# A downstream job is cancelled (DependencyNeverSatisfied) if its upstream
# fails or times out; fix, resubmit the failed piece, then rerun this script
# with FROM= set to the next step. Every step is resumable: finished shards,
# merge progress and search partials are reused.
set -euo pipefail
cd /home/synkrp/locale/benchmark
export PATH="$HOME/.local/bin:$PATH"

CONFIG=${CONFIG:-configs/sra4571/biowulf_locale_sra4571_endwin0_minlen100.yaml}
SEARCH_CONFIG=${SEARCH_CONFIG:-$CONFIG}
EMBED_SHARDS=${EMBED_SHARDS:-16}
SEARCH_SHARDS=${SEARCH_SHARDS:-4}
RATES=${RATES:-"0.0 0.05 0.1"}
FROM=${FROM:-embed}
LOGS=/data/synkrp/locale/slurm_logs
mkdir -p "$LOGS"
echo "config: $CONFIG  embed shards: $EMBED_SHARDS  search shards: $SEARCH_SHARDS  from: $FROM"

BUNDLE=/data/synkrp/locale-data/constructed/sra4571/bundle
if [ "$FROM" = embed ]; then
  # run_benchmark downloads all 4571 accessions from Logan when it finds no
  # contigs under <dataset_dir>/logan_accessions, which hangs on compute nodes
  # (no internet). Check with the same rglob it uses, before holding GPUs.
  # If this fails: bash /data/synkrp/locale-data/constructed/sra4571/biowulf_link_accessions.sh
  n_accs=$(grep -c . "$BUNDLE/accs.txt")
  n_found=$(uv run --no-sync python -c "from pathlib import Path; print(len(list(Path('$BUNDLE/logan_accessions').rglob('*.contigs.fa'))))")
  echo "logan_accessions: rglob finds $n_found contigs for $n_accs accessions"
  [ "$n_found" -eq "$n_accs" ] || { echo "contig count mismatch; run biowulf_link_accessions.sh first"; exit 1; }
  checkquota 2>/dev/null | grep -E '^Mount|^/data' || true  # shards + merged index need ~6.5 TB free
fi

dep=""
case "$FROM" in
  embed)
    embed=$(CONFIG=$CONFIG NUM_SHARDS=$EMBED_SHARDS sbatch --parsable \
      --array="0-$((EMBED_SHARDS - 1))" slurm_scripts/biowulf_sra4571_embed.sbatch)
    echo "embed:  $embed  (array 0-$((EMBED_SHARDS - 1)))"
    dep="--dependency=afterok:$embed"
    ;&
  merge)
    merge=$(CONFIG=$CONFIG sbatch --parsable $dep \
      slurm_scripts/biowulf_sra4571_merge.sbatch "$EMBED_SHARDS")
    echo "merge:  $merge  ${dep:+(after embed)}"
    dep="--dependency=afterok:$merge"
    ;&
  search)
    for rate in $RATES; do
      s=$(NUM_SHARDS=$SEARCH_SHARDS sbatch --parsable $dep \
        --array="0-$((SEARCH_SHARDS - 1))" \
        slurm_scripts/biowulf_sra4571_search.sbatch "$rate" "$SEARCH_CONFIG")
      echo "search: $s  mut$rate (array 0-$((SEARCH_SHARDS - 1))) ${dep:+(after merge)}"
    done
    ;;
  *) echo "FROM must be embed, merge or search (got '$FROM')"; exit 1 ;;
esac
echo "squeue -u $USER -o '%.10i %.22j %.8T %.10M %.6D %R' to watch"
