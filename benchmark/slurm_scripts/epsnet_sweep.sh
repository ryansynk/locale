#!/bin/bash
# Build + search a list of dense configs inside an existing allocation,
# spreading them round-robin over its nodes: node i runs configs i, i+N, ...
# one after another; the nodes run in parallel. Per config: stage engine
# (skipped when the engine dir is .done), then stage search at every rate
# (accuracy only; set TIMING=1 for --do_timing). Meant for the epsilon-net
# sweep but works for any dense config whose embeddings are already built.
#
#   salloc -A m5408_g -C gpu -q interactive -N 4 --gpus-per-node=4 -t 2:00:00
#   bash slurm_scripts/epsnet_sweep.sh configs/sra50/perlmutter_locale_sra50v2_epsnet_e*.yaml
#
# Logs: slurm_logs/sweep_<config name>.log (one per config).
set -euo pipefail
(($#)) || { echo "usage: $0 <config.yaml>..." >&2; exit 1; }
: "${SLURM_JOB_ID:?run inside an allocation (salloc ... first)}"
cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1
read -r -a RATE_LIST <<<"${RATES:-0.0 0.05 0.1}"
TIMING_ARGS=()
[[ "${TIMING:-0}" == 1 ]] && TIMING_ARGS=(--do_timing true)
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
CONFIGS=("$@")
echo "=== ${#CONFIGS[@]} config(s) on ${#NODES[@]} node(s): ${NODES[*]}"

run_config() {  # <node> <config>
    local node=$1 config=$2 log
    log="slurm_logs/sweep_$(basename "$config" .yaml).log"
    echo "[$node] start $config -> $log"
    # set -e does not apply inside a function called from an || list, so every
    # step checks its own status: a failed engine build skips the searches.
    {
        echo "=== engine ($(date))"
        srun --unbuffered -N1 -n1 -w "$node" --gpus-per-node=4 --cpus-per-task=128 \
            uv run --no-sync python run_benchmark.py --config "$config" --stage engine ||
            return 1
        for rate in "${RATE_LIST[@]}"; do
            echo "=== search rate $rate ($(date))"
            srun --unbuffered -N1 -n1 -w "$node" --gpus-per-node=4 --cpus-per-task=128 \
                uv run --no-sync python run_benchmark.py --config "$config" \
                --mutation_rate "$rate" --stage search "${TIMING_ARGS[@]}" ||
                return 1
        done
        echo "=== done ($(date))"
    } >"$log" 2>&1
}

for i in "${!NODES[@]}"; do
    (
        for ((j = i; j < ${#CONFIGS[@]}; j += ${#NODES[@]})); do
            if run_config "${NODES[$i]}" "${CONFIGS[$j]}"; then
                echo "[${NODES[$i]}] done   ${CONFIGS[$j]}"
            else
                echo "[${NODES[$i]}] FAILED ${CONFIGS[$j]} (see slurm_logs/sweep_$(basename "${CONFIGS[$j]}" .yaml).log)"
            fi
        done
    ) &
done
wait
echo "=== all done"
