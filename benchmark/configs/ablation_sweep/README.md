# Ablation sweep on sra50v2

Re-evaluates the seven checkpoints behind the paper's ablation table on the
sra50 v2 bundle. All configs are identical except `model.checkpoint_path`.
Run with `./run_ablation_sweep.sh` from inside a multi-node GPU allocation
(see the script header); it prints the recall tables at the end.

| Config | Table row | Checkpoint (run id, step 5859) |
|---|---|---|
| `baseline_heavy_contain_logan.yaml` | Baseline (Heavy / Contain. / Logan) | dwr9e67f |
| `crop_overlap.yaml` | Cropping: Overlap | ea7c9cd7 |
| `crop_both.yaml` | Cropping: Both | 7lvoe2hj |
| `data_reference.yaml` | Training data: Reference | 8vqiabk9 |
| `mut_none.yaml` | Training mutation: None | ep3zkwec |
| `mut_light.yaml` | Training mutation: Light | sdp7o13s |
| `mut_medium.yaml` | Training mutation: Medium | 355y3fml |

All seven checkpoints are on Perlmutter under `checkpoints/<run id>/`. The
Nexus mirror is `/fs/nexus-scratch/ryansynk/rawbert/checkpoints/<run id>/`.

Caveat carried over from the original table: only `data_reference` is a
single-axis change from the baseline. The crop and mutation rows were trained
on reference genomes, and most also enable hard negatives, so they are
single-axis relative to `data_reference` (8vqiabk9) rather than the Logan
baseline. Logan-trained counterparts exist as checkpoints lgodjvzt (no
mutations), 0zvrmj5m (light) and oupqgxds (reference data, hard negatives off).

The original v1 numbers can be re-derived with
`uv run python print_results.py old/results_ablations
/pscratch/sd/r/rsynk/hf_bundles/sra50/queries.parquet
/pscratch/sd/r/rsynk/hf_bundles/sra50/accs.txt --full_model_names true`.
