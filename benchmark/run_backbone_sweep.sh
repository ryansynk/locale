#!/usr/bin/env bash
#
# Run every backbone_sweep config at every eval mutation rate, one multi-node
# srun step at a time. Run it from inside a multi-node GPU allocation, e.g.
#
#   salloc -A m5408_g -C gpu -q interactive -N 4 -t 4:00:00 \
#          --ntasks-per-node=1 --gpus-per-node=4 -c 128
#   cd /pscratch/sd/r/rsynk/locale/benchmark && ./run_backbone_sweep.sh
#
# Each srun step launches one run_benchmark.py process per node in the
# allocation. run_benchmark.py reads SLURM_NODEID / SLURM_NNODES, builds one
# index shard (or scans one range of vector rows) per node, and rank 0 merges.
# Plain `python run_benchmark.py` inside a multi-node allocation would run a
# single rank-0 process that waits forever for peers, so always go through srun.
#
# Per config the first rate builds the index; later rates reuse it (the index
# depends on the checkpoint, not on the mutation rate). Steps run sequentially
# because each one occupies the whole allocation and concurrent builds of the
# same index would collide on the shard directories.
#
#   ./run_backbone_sweep.sh                       # all six configs, all rates
#   ./run_backbone_sweep.sh dna2vec               # one backbone
#   RUNGS="heavy" RATES="0.0" ./run_backbone_sweep.sh nt50m hyenadna
#
# The srun resource flags must fit inside the allocation; override with
# GPUS_PER_NODE / CPUS_PER_TASK if you asked salloc for less.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Not inside a Slurm allocation; run salloc first." >&2
    exit 1
fi

BACKBONES=("$@")
if [ ${#BACKBONES[@]} -eq 0 ]; then
    BACKBONES=(nt50m hyenadna dna2vec)
fi
read -r -a RUNG_LIST <<<"${RUNGS:-none heavy}"
read -r -a RATE_LIST <<<"${RATES:-0.0 0.05 0.1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-128}"

export PYTHONUNBUFFERED=1

failed=()
for backbone in "${BACKBONES[@]}"; do
    for rung in "${RUNG_LIST[@]}"; do
        cfg="configs/backbone_sweep/${backbone}_${rung}.yaml"
        if [ ! -f "$cfg" ]; then
            echo "[skip] no config $cfg"
            continue
        fi
        for rate in "${RATE_LIST[@]}"; do
            echo "=== $cfg @ mutation_rate $rate  ($(date))  nodes=${SLURM_JOB_NUM_NODES:-?} ==="
            if ! srun --unbuffered --ntasks-per-node=1 \
                      --gpus-per-node="$GPUS_PER_NODE" \
                      --cpus-per-task="$CPUS_PER_TASK" \
                      uv run python run_benchmark.py \
                        --config "$cfg" --mutation_rate "$rate"; then
                echo "[FAIL] $cfg @ $rate" >&2
                failed+=("$cfg @ $rate")
                # Later rates need the index this rate builds, so move on to
                # the next config rather than retrying against a broken build.
                break
            fi
        done
    done
done

echo
if [ ${#failed[@]} -gt 0 ]; then
    echo "Failed: ${failed[*]}" >&2
    exit 1
fi
echo "All backbone sweep runs finished."
