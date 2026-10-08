#!/bin/bash
# fp16 / bf16 quantized-exact runs (configs/<dataset>/*_quantized_{fp16,bf16}.yaml)
# inside an existing allocation: one config per node, each building its
# 2-byte copy of the encoder fbin (stage engine) then searching at every
# noise level, 0.0 0.05 0.1. The scheduling is dense_sweep.sh's; pass
# configs to run a subset, RATES=... or TIMING=1 as there. A dataset name
# (sra50, the default) picks that dataset's two configs; anything else is
# taken as config paths. Two configs use two nodes; extra nodes sit idle.
#
#   salloc -A m5408_g -C gpu -q interactive -N 4 --gpus-per-node=4 -t 1:00:00
#   bash slurm_scripts/quantized_sweep.sh            # sra50v2
#
# Logs: slurm_logs/sweep_perlmutter_locale_<dataset>v2_quantized_<dtype>.log
set -euo pipefail
cd /pscratch/sd/r/rsynk/locale/benchmark
if (($# == 0)) || [[ $# == 1 && "$1" != *.yaml ]]; then
    ds=${1:-sra50}
    CONFIGS=(configs/"$ds"/perlmutter_locale_"$ds"v2_quantized_{fp16,bf16}.yaml)
else
    CONFIGS=("$@")
fi
for c in "${CONFIGS[@]}"; do
    [[ -f "$c" ]] || { echo "no such config: $c" >&2; exit 1; }
done
printf 'configs:\n'
printf '  %s\n' "${CONFIGS[@]}"
exec bash slurm_scripts/dense_sweep.sh "${CONFIGS[@]}"
