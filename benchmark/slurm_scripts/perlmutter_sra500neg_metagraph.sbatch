#!/bin/bash
#SBATCH -A m5408
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -c 256
#SBATCH --image=ghcr.io/ratschlab/metagraph:master
#SBATCH -J locale-sra500neg-metagraph
#SBATCH -o slurm_logs/sra500neg_metagraph_%j.out
#SBATCH -e slurm_logs/sra500neg_metagraph_%j.err

# Metagraph searched with the sra500 negative queries (calibrated-FPR
# experiment), at every mutation rate of the negatives bundle. The index is
# the built sra500 one, so this is search only and short; one CPU node with
# the container requested on the job, as perlmutter_sra500_metagraph.sbatch.
#
#   sbatch slurm_scripts/perlmutter_sra500neg_metagraph.sbatch          # k=31, all rates
#   RATES="0.0" sbatch slurm_scripts/perlmutter_sra500neg_metagraph.sbatch
#   CONFIG=configs/sra500_negatives/perlmutter_metagraph_k19_sra500neg.yaml \
#     sbatch slurm_scripts/perlmutter_sra500neg_metagraph.sbatch        # k=19
#   CONFIG=configs/sra500_negatives2k/perlmutter_metagraph_sra500neg2k.yaml \
#     sbatch slurm_scripts/perlmutter_sra500neg_metagraph.sbatch        # 2k negative set
#
# (CONFIG=... / RATES=... must be on the sbatch line or exported; an
# assignment on its own line does nothing.)
set -euo pipefail

cd /pscratch/sd/r/rsynk/locale/benchmark
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1

CONFIG=${CONFIG:-configs/sra500_negatives/perlmutter_metagraph_sra500neg.yaml}
# The bundle is the config's dataset_dir, so one script serves every negative set.
BUNDLE=$(uv run python -c "import yaml, sys; print(yaml.safe_load(open(sys.argv[1]))['dataset_dir'])" "$CONFIG")
read -r -a RATE_LIST <<<"${RATES:-0.0 0.05 0.1}"

# Abort before srun unless every contig is linked (run_benchmark would
# otherwise re-download all 500 from Logan).
n_accs=$(grep -c . "$BUNDLE/accs.txt")
n_found=$(uv run python -c "from pathlib import Path; print(len(list(Path('$BUNDLE/logan_accessions').rglob('*.contigs.fa'))))")
echo "logan_accessions: $n_found contigs for $n_accs accessions"
[ "$n_found" -eq "$n_accs" ] || { echo "contig count mismatch; aborting" >&2; exit 1; }

for rate in "${RATE_LIST[@]}"; do
    echo "=== metagraph @ mutation_rate $rate  ($(date))  config=$CONFIG ==="
    # --num_shards 1 keeps run_benchmark on the single-node path.
    srun --unbuffered -N1 -n1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
      uv run python scripts/run_benchmark.py --config "$CONFIG" --mutation_rate "$rate" --num_shards 1
done
echo "done; next: scripts/calibrate_fpr.py then scripts/apply_fpr_thresholds.py"
