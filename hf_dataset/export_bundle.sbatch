#!/bin/bash
#SBATCH -A m5408_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -J locale-hf-export
#SBATCH -o benchmark/slurm_logs/hf_export_%j.out
#SBATCH -e benchmark/slurm_logs/hf_export_%j.err

# Turn a finished benchmark index into a Hugging Face embedding bundle:
#   <STAGE>/embeddings/<LABEL>/{manifest.json, meta.parquet, [projector.pt],
#                               base/, mut0.00/, mut0.05/, mut0.10/}
# in the layout of rsynk/locale-benchmark-sra500's embeddings/locale.
# Steps (all on this node; sharding is I/O-bound and fast):
#   1. hf_dataset/shard_fbin.py       base shards + meta + manifest
#   2. hf_dataset/generate_queries.py every query at every rate, both strands
#   3. hf_dataset/ground_truth.py     exact top-K over the merged base file
#   4. locale-data/benchmark/score.py oracle table for the dataset card
# Then upload (see hf_dataset/README.md).
#
# Usage (from the repo root; CONFIG and LABEL are required):
#   CONFIG=benchmark/configs/sra4571/perlmutter_locale_sra4571_endwin0_minlen100_pca_d128_bf16.yaml \
#   LABEL=locale-pca128-bf16 sbatch hf_dataset/export_bundle.sbatch
# Optional: STAGE (default /pscratch/sd/r/rsynk/hf_bundles/<dataset_name>_embed),
#           K (top-k of the ground truth, default 100), ROWS (rows per shard,
#           default ~1 GB). Works as `bash hf_dataset/export_bundle.sbatch` on
#           an interactive GPU node too.
# The index is taken from CONFIG: a pca / quantized_exact index exports its
# projected or 2-byte file, anything else exports the encoder's fp32 fbin.
# Re-running skips the shards if manifest.json exists and skips a rate's
# gt.bin if present; delete those to redo.

set -euo pipefail

cd "${LOCALE_ROOT:-/pscratch/sd/r/rsynk/locale}"
mkdir -p benchmark/slurm_logs
export PYTHONUNBUFFERED=1

: "${CONFIG:?set CONFIG=<benchmark yaml>}"
: "${LABEL:?set LABEL=<bundle label, e.g. locale-pca128-bf16>}"
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG"; exit 1; }
K=${K:-100}
SCORE=${SCORE:-/pscratch/sd/r/rsynk/locale-data/benchmark/score.py}

# Resolve the index from the config the way run_benchmark does.
eval "$(uv run python - "$CONFIG" <<'EOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
m = c["model"]
enc = f'{c["index_dir"].rstrip("/")}/{m["encoder_label"]}'
engine = m["index"].get("engine", "exact")
src = f'{enc}/{m["index_label"]}' if engine in ("pca", "quantized_exact") else enc
print(f'DATASET={c["dataset_name"]}')
print(f'BUNDLE={c["dataset_dir"]}')
print(f'ENC_DIR={enc}')
print(f'SRC={src}')
EOF
)"
STAGE=${STAGE:-/pscratch/sd/r/rsynk/hf_bundles/${DATASET}_embed}
OUT=$STAGE/embeddings/$LABEL

[ -f "$ENC_DIR/.done" ] || { echo "$ENC_DIR is not .done"; exit 1; }
[ -f "$SRC/.done" ] || { echo "$SRC is not .done"; exit 1; }
[ -f "$BUNDLE/queries_mut0.10.parquet" ] || { echo "no query parquets in $BUNDLE"; exit 1; }
BASE=""
for f in embeddings.fbin embeddings.fp16bin embeddings.bf16bin; do
  [ -f "$SRC/$f" ] && BASE=$SRC/$f
done
[ -n "$BASE" ] || { echo "no embeddings file in $SRC"; exit 1; }
PROJ=()
[ -f "$SRC/projector.pt" ] && PROJ=(--projector "$SRC/projector.pt")

echo "dataset: $DATASET  config: $CONFIG"
echo "base:    $BASE"
echo "out:     $OUT"
nvidia-smi -L || true

echo "== 1/4 shards"
if [ -f "$OUT/manifest.json" ]; then
  echo "manifest.json exists, skipping"
else
  mkdir -p "$STAGE/embeddings"
  lfs setstripe -c 8 -S 16M "$STAGE/embeddings" || true
  uv run python -m hf_dataset.shard_fbin --src "$SRC" --out "$OUT" ${ROWS:+--rows_per_shard "$ROWS"}
fi

echo "== 2/4 query vectors"
uv run python -m hf_dataset.generate_queries \
  --dataset_path "$BUNDLE" --cfg "$CONFIG" --out "$OUT" "${PROJ[@]}"

echo "== 3/4 ground truth (k=$K)"
if ls "$OUT"/mut0.00/gt.bin "$OUT"/mut0.05/gt.bin "$OUT"/mut0.10/gt.bin >/dev/null 2>&1; then
  echo "gt.bin present for every rate, skipping"
else
  uv run python -m hf_dataset.ground_truth --base "$BASE" --queries_dir "$OUT" --k "$K"
fi

echo "== 4/4 oracle scores (for the dataset card)"
for m in 0.00 0.05 0.10; do
  uv run python -I "$SCORE" "$OUT/mut$m/gt.bin" --bundle "$OUT" --mut "$m" \
    --queries "$BUNDLE/queries.parquet" --label "oracle mut$m"
done

echo "== done: $(ls "$OUT/base" | wc -l) shards, $(du -sh "$OUT" | cut -f1) in $OUT"
