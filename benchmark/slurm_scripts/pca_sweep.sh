#!/bin/bash
# PCA pca_dim sweep (configs/<dataset>/*_pca_d*.yaml) inside an existing
# allocation: one config per node, each building its projected fbin (stage
# engine) then searching at every noise level, 0.0 0.05 0.1. The scheduling
# is epsnet_sweep.sh's; pass configs to run a subset, RATES=... or TIMING=1
# as there. A dataset name (sra50, the default, or sra500) picks that
# dataset's four configs; anything else is taken as config paths.
#
#   salloc -A m5408_g -C gpu -q interactive -N 4 --gpus-per-node=4 -t 2:00:00
#   bash slurm_scripts/pca_sweep.sh            # sra50v2
#   bash slurm_scripts/pca_sweep.sh sra500     # sra500v2
#
# Logs: slurm_logs/sweep_perlmutter_locale_<dataset>v2_pca_d<dim>.log
set -euo pipefail
cd /pscratch/sd/r/rsynk/locale/benchmark
if (($# == 0)) || [[ $# == 1 && "$1" != *.yaml ]]; then
    # DATASET=... still works; a lowercase dataset=... is a different
    # variable and silently fell back to sra50 (2026-10-08), so refuse it.
    if [[ -n "${dataset:-}" && -z "${DATASET:-}" ]]; then
        echo "dataset=$dataset is ignored: pass it as an argument," \
            "bash $0 $dataset" >&2
        exit 1
    fi
    ds=${1:-${DATASET:-sra50}}
    CONFIGS=(configs/"$ds"/perlmutter_locale_"$ds"v2_pca_d{32,64,128,256}.yaml)
else
    CONFIGS=("$@")
fi
for c in "${CONFIGS[@]}"; do
    [[ -f "$c" ]] || { echo "no such config: $c" >&2; exit 1; }
done
printf 'configs:\n'
printf '  %s\n' "${CONFIGS[@]}"
exec bash slurm_scripts/epsnet_sweep.sh "${CONFIGS[@]}"
