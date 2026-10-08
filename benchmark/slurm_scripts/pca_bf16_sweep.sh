#!/bin/bash
# bf16 PCA sweep on sra50v2 and sra500v2 (pca_dim 32 / 64 / 128 / 256, dtype
# bf16; configs/<ds>/*_pca_d<k>_bf16.yaml) inside an existing allocation. The
# scheduling is dense_sweep.sh's: node i runs configs i, i+N, ...; per config
# a single-node build (stage engine, all 4 GPUs; skipped when .done) then a
# search at every noise level, 0.0 0.05 0.1 (RATES=... / TIMING=1 as there).
# The order gives each of 4 nodes one sra500 and one sra50 config.
# sra500 pca-d128-bf16 is already built and searched (the 2-node check), so
# it finishes at once from its .done index and saved hits.
#
#   salloc -A m5408_g -C gpu -q interactive -N 4 --gpus-per-node=4 -t 1:00:00
#   bash slurm_scripts/pca_bf16_sweep.sh
#
# Logs: slurm_logs/sweep_perlmutter_locale_<ds>v2_pca_d<k>_bf16.log
set -euo pipefail
cd /pscratch/sd/r/rsynk/locale/benchmark
CONFIGS=()
for ds in sra500 sra50; do
    for k in 256 64 32 128; do
        CONFIGS+=(configs/$ds/perlmutter_locale_${ds}v2_pca_d${k}_bf16.yaml)
    done
done
for c in "${CONFIGS[@]}"; do
    [[ -f "$c" ]] || { echo "no such config: $c" >&2; exit 1; }
done
printf 'configs:\n'
printf '  %s\n' "${CONFIGS[@]}"
exec bash slurm_scripts/dense_sweep.sh "${CONFIGS[@]}"
