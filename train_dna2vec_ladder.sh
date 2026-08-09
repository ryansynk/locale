#!/usr/bin/env bash
#
# Train the full Embed-Search-Align (dna2vec) augmentation ladder: none ->
# light -> medium -> heavy, one rung after another on this node's GPUs.
#
# Run it inside an allocation that already holds every GPU you intend to use
# (salloc/srun --gres=gpu:8 ...); it does not submit anything to SLURM itself.
#
#   ./train_dna2vec_ladder.sh              # all four rungs
#   ./train_dna2vec_ladder.sh medium heavy # just these, in this order
#
# As each rung finishes the script records its wandb run id in
# benchmark/runs_dna2vec.yaml and rewrites the matching
# benchmark/configs/dna2vec_<rung>.yaml checkpoint_path, which is what makes
# benchmark/eval_all_backbones.sh able to pick the ladder up. A rung that is
# already recorded with a checkpoint on disk is skipped, so the script is
# re-runnable after a preemption.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

RUNS_YAML="benchmark/runs_dna2vec.yaml"
CHECKPOINT_DIR="/fs/nexus-scratch/ryansynk/rawbert/checkpoints"
RUNGS=("$@")
if [ ${#RUNGS[@]} -eq 0 ]; then
    RUNGS=(none light medium heavy)
fi

NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [ "$NPROC" -eq 0 ]; then
    echo "No GPUs visible. Request them in the allocation, or set NPROC." >&2
    exit 1
fi

# The step number is baked into runs_dna2vec.yaml and into every eval config's
# checkpoint filename, and it is total_samples / (per_device_batch * NPROC) --
# so it only lands on the recorded value at the batch size the ladder was
# specified for. Changing it silently produces a ladder the eval cannot find.
STEP="$(grep -E '^step:' "$RUNS_YAML" | awk '{print $2}')"
EXPECTED_NPROC=8
if [ "$NPROC" -ne "$EXPECTED_NPROC" ] && [ "${ALLOW_ANY_GPU_COUNT:-0}" != "1" ]; then
    echo "This ladder assumes ${EXPECTED_NPROC} GPUs (effective batch 512, step ${STEP})," >&2
    echo "but $NPROC are visible. The rungs would stop at a different step and every" >&2
    echo "eval config would point at a checkpoint that does not exist." >&2
    echo "Re-run in a full ${EXPECTED_NPROC}-GPU allocation, or set ALLOW_ANY_GPU_COUNT=1" >&2
    echo "and update 'step:' in $RUNS_YAML to match." >&2
    exit 1
fi

already_recorded() {
    grep -qE "^  $1: " "$RUNS_YAML"
}

recorded_id() {
    grep -E "^  $1: " "$RUNS_YAML" | awk '{print $2}'
}

# Append the rung -> run id mapping and repoint the eval config at the
# checkpoint this run actually produced. tests/test_eval_configs.py cross-checks
# both against the train config train.py saved next to the checkpoint, so a
# mix-up here is caught rather than silently mislabelling a Table 3 column.
record_run() {
    local rung="$1" run_id="$2" ckpt="$3"
    echo "  $rung: $run_id" >>"$RUNS_YAML"
    local eval_cfg="benchmark/configs/dna2vec_${rung}.yaml"
    sed -i "s|^  checkpoint_path: .*|  checkpoint_path: ${ckpt}|" "$eval_cfg"
    echo "[record] $rung -> $run_id ($eval_cfg repointed)"
}

for rung in "${RUNGS[@]}"; do
    config="configs/dna2vec_${rung}.yaml"
    if [ ! -f "$config" ]; then
        echo "No such config: $config" >&2
        exit 1
    fi

    if already_recorded "$rung"; then
        existing="$(recorded_id "$rung")"
        if [ -f "${CHECKPOINT_DIR}/${existing}/checkpoint${STEP}.pth.tar" ]; then
            echo "[skip] $rung already trained as $existing"
            continue
        fi
        echo "[warn] $rung recorded as $existing but its checkpoint is missing;" >&2
        echo "       remove that line from $RUNS_YAML to retrain it." >&2
        exit 1
    fi

    # Per-rung log dir: train.py writes the wandb run id to
    # <log_dir>/wandb_run_id_<array task>.txt, so sharing one dir across rungs
    # would overwrite the id we need to record.
    log_dir="logs/dna2vec_${rung}"
    rm -f "${log_dir}/wandb_run_id_0.txt"

    echo "=== training dna2vec / $rung on $NPROC GPUs ==="
    uv run python -m torch.distributed.run \
        --standalone --nnodes=1 --nproc_per_node="$NPROC" \
        train.py --config "$config" --log_dir "$log_dir"

    id_file="${log_dir}/wandb_run_id_0.txt"
    if [ ! -s "$id_file" ]; then
        echo "$rung finished but $id_file is missing; cannot record the run." >&2
        exit 1
    fi
    run_id="$(cat "$id_file")"

    ckpt="${CHECKPOINT_DIR}/${run_id}/checkpoint${STEP}.pth.tar"
    if [ ! -f "$ckpt" ]; then
        echo "$rung (run $run_id) produced no $ckpt." >&2
        echo "Check whether the run stopped early; nothing was recorded." >&2
        exit 1
    fi

    record_run "$rung" "$run_id" "$ckpt"
done

echo
echo "Ladder complete. Evaluate it with:"
echo "  cd benchmark && ./eval_all_backbones.sh dna2vec"
