# CLAUDE.md

## Project
Creating an embedding model for local alignment of DNA sequences.
The goal is to convert sequence search over the large sets of sequences (like NIH Sequence Read Archive) into vector search. Vector search could be scalable and importantly robust to noise (an advantage over k-mer methods like metagraph)

## Layout
`v1-cleanup` is the working branch and is a strict superset of `main`'s history
(main's tip is the merge-base). The cleanup renamed things; expect stale
references in older notes and commands:
- Python package is `lae/` (was `rawbert/`). Model name in configs is `locale`.
- Benchmark lives in `benchmark/` (was `experiments/sra_recall/`).
- Training config is `configs/config.yaml`.
- `experiments/`, `batch_scripts/`, `scripts/`, and `nexus_scripts/` are gone
  from git. Directories by those names on disk are pre-cleanup leftovers.
  `experiments/` holds ~44 GB of stale indexes and is excluded via
  `.git/info/exclude` (local-only) rather than `.gitignore`, since a fresh
  clone has no such directory. Do not `git add -A` without checking.

## Benchmark
In `benchmark/`. Tests retrieval of relevant SRA accessions given query sequences.
- Each method's `*_index.py` builds an index over accessions and searches it.
- `metagraph_index` is k-mer search; `dense_index` is vector embedding search,
  with different embedding models as subclasses.
- Each method returns a score for (up to) every accession.
- `plot_results.py` sorts these and computes recall@k. It takes three positional
  args: results dir, queries parquet, accessions file.

### Data
Datasets come from the HF Hub, selected by `dataset_name` and materialised into
`dataset_dir`:
- `sra50` -> `rsynk/locale-benchmark-sra50`
- `sra500` -> `rsynk/locale-benchmark-sra500`

`snapshot_download` needs `repo_type="dataset"`; these are dataset repos, and the
default is `"model"`.

**`sra50` has 47 accessions, not 50.** `SRR14483924`, `SRR8745633`, and
`SRR8745637` were never synced from SRA into the Logan release and 404 from
`logan-pub`. No queries target them, so ground truth and recall are unaffected.

Contigs are fetched from `logan-pub` S3 **only if**
`<dataset_dir>/logan_accessions/**/*.contigs.fa` is empty. Symlinking that path
at an existing contig tree skips the download entirely.

### GPUs
Index build and search both fan out over `torch.cuda.device_count()`
automatically — one spawned worker process per GPU for the build, sharded matmul
for search. No torchrun, no flags.

The `SLURM_NNODES` / `SLURM_NODEID` sharding in `run_benchmark.py` is for
**multi-node only**. On a single node just run the script directly, but request
every GPU in the allocation or `device_count()` will not see them.

An index rebuilds only when `<index_dir>/<index_suffix>/.done` is absent. If a
run dies after that file is touched, delete the index dir to force a rebuild.

## Training
- DNABERT base model (could change)
- Momentum Contrast training strategy
- Self-supervised dataset generated from a parent contig into two crops (containment or overlap)
- Noise added to crops
- Config in `configs/config.yaml` (still carries `/pscratch` paths; repoint for nexus)

## Nexus
Data root is `/fs/nexus-projects/sra_search`:
- `hf/sra50/logan_accessions` -> symlink to `data/test/logan_contig`, so the HF
  flow reuses the 47 already-downloaded contig sets instead of refetching.
- `indexes_locale/` is where new indexes go. `indexes/rawbert/...` and
  `indexes_500/rawbert/...` are pre-cleanup, built under the old naming, and
  kept as a fallback — the `locale` rename means the code will not find them.
- Checkpoint used for the paper: `checkpoints/8vqiabk9/checkpoint5859.pth.tar`,
  which yields index suffix `locale/8vqiabk9/5859/maxlen256_poolmean_chunkstride`.

SLURM: `--account=cml-tomg --qos=cml-very_high`, partition `cml-dpart`
(rtxa4000 / rtx2080ti) or `cml-scavenger` (a6000 / h100 / l40s). Configs are
`benchmark/configs/nexus_locale.yaml` and `nexus_metagraph.yaml`; batch wrapper
is `benchmark/nexus_benchmark.sbatch`.

Metagraph binary: `/fs/nexus-scratch/ryansynk/metagraph/metagraph/build/metagraph`
(Perlmutter instead needs `shifter metagraph` inside a container).

## Commands
Benchmark, single node, all GPUs on the node:
```
uv run python run_benchmark.py --config configs/nexus_locale.yaml --num_queries 20
```
`--num_queries` subsamples the *search* only; the index build always consumes
every accession. `--no_search` builds and exits.

Multi-node index build:
```
srun uv run python run_benchmark.py --config configs/nexus_locale.yaml
```

Training:
```
srun uv run python -m torch.distributed.run --nnodes=4 --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train.py --config configs/config.yaml
```

Plotting:
```
uv run python plot_results.py results/ <dataset_dir>/queries.parquet <dataset_dir>/accs.txt
```
