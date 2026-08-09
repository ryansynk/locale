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

### Backbone swap
The encoder is selected by `backbone:` in the training config and resolved in
`lae/modeling/backbones.py`. Everything downstream — mean-pool + L2-norm head,
InfoNCE loss, index, search — is shared, so only the encoder varies.

- Ids: `dnabert2` (default, D=768), `nt50m` (D=512), `hyenadna` (D=256),
  `dna2vec` (D=1020 — Embed-Search-Align's encoder, 54M params).
- `backbone` is written into the checkpoint's `model_args`, and the benchmark's
  `LOCALEEncoder` reads it back to pick the matching encoder *and tokenizer*.
  Pre-swap checkpoints have no such key and fall back to `dnabert2`, so
  `8vqiabk9/checkpoint5859.pth.tar` still loads with `strict=True`.
- Hub gotchas, all already handled — don't "simplify" them away:
  - NT-v2's `auto_map` has **no `AutoModel` entry**, so `AutoModel` falls back
    to native `EsmModel` (plain MLP) and shape-mismatches against NT's SwiGLU
    weights. Load via `AutoModelForMaskedLM` and take `.esm`.
  - `HyenaDNATokenizer` pads but returns **no `attention_mask`**; `get_tokenizer`
    patches `model_input_names` so the pooling head can mask padding.
  - `EsmModel` builds `embeddings.position_embeddings` (dead — NT-v2 is rotary)
    and `contact_head` (only used by `predict_contacts`). Left trainable they
    get no gradient, which aborts DDP's reducer at step 2 and puts a `None`
    into the grad-norm logging. `_freeze_unused` handles it; it raises if the
    prefixes stop matching rather than silently regressing.
  - dna2vec's `DNAEncoder.forward` returns a **bare tensor**, not a tuple. The
    shared head does `model(...)[0]`, which on a bare tensor picks sequence 0
    and still broadcasts against the mask into a plausible-looking
    `(B, S, H)` — it trains on garbage without erroring. `DNA2VecBackbone`
    wraps the output in a tuple; do not unwrap it. It also absorbs the
    `token_type_ids` the tokenizer emits, and its positions are a fixed
    1024-row sinusoidal table (fine for 256bp crops, an index error above it).
- Before any multi-GPU launch, run the opt-in gradient check — it reproduces
  both DDP failures on CPU in seconds:
  ```
  LAE_RUN_BACKBONE_TESTS=1 uv run python -m pytest tests/test_backbone_gradients.py
  ```
- Ladder configs are
  `configs/{nt50m,hyenadna,dna2vec}_{none,light,medium,heavy}.yaml`, generated
  identically and guarded by `tests/test_ladder_configs.py`, which fails if two
  rungs differ anywhere but the mutation settings.
- `dna2vec` is also a benchmark `model.name` — the *untrained* ESA baseline in
  `benchmark/configs/embed-search-align_config.yaml`. Same weights, opposite
  role: `benchmark/configs/dna2vec_*.yaml` are trained LOCALE checkpoints
  (`model.name: locale`). Their indexes live under `locale/<runid>/`, so the
  two never collide on disk.
- `train_contigs.parquet` / `val_contigs.parquet` are the 50-accession train and
  val slices; verified disjoint from each other and from the 47 eval accessions.

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

Metagraph runs from a container, not a native build: `metagraph_master.sif` in
`/fs/nexus-scratch/ryansynk`, pulled from `ghcr.io/ratschlab/metagraph:master`.
The configs invoke it as a multi-word `executable:` (`apptainer exec --bind ...
<sif> metagraph`), which works because `metagraph_index.py` `shlex.split`s it
and `build_metagraph.sh` expands `${METAGRAPH_EXEC}` unquoted — the same trick
Perlmutter uses for `shifter metagraph`. Both binds are **self-referential**
(`/fs/nexus-projects/sra_search` and `/fs/nexus-scratch/ryansynk` mapped to
themselves) and must stay that way: `contig_manifest.txt` is written host-side
with absolute paths and piped to metagraph's stdin, and `annotate
--anno-filename` stores those absolute paths as column labels, which `search()`
parses back into accessions via `.str.split("/").list.get(-2)`.

The old native build at `metagraph/metagraph/build/metagraph` is **broken** —
linked against `libboost_iostreams.so.1.66.0`, which el9 no longer ships. That
checkout is also stale (pinned at `e69e128`, Mar 2026) and has drifted from the
image: upstream made in-place graph construction the default and replaced the
opt-in `--inplace` with an opt-out `--in-ram`, so `build_metagraph.sh` no longer
passes `--inplace`. Do not "restore" that flag against a current image; it is a
hard `Unknown option` error.

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

Augmentation ladder for one backbone (4 array tasks, single node x 8 GPUs each).
Effective batch is 512, not the paper's 1024 — a deviation applied uniformly to
all four rungs, so the within-backbone trend is unaffected:
```
sbatch nexus_train_ladder.sbatch nt50m
```
That sbatch wrapper is **not in the repo** (nor are the old `*_sweep.sh`). For
the dna2vec ladder use `train_dna2vec_ladder.sh`, which trains all four rungs
back-to-back inside an existing 8-GPU allocation, then records each wandb run id
in `benchmark/runs_dna2vec.yaml` and repoints the matching eval config:
```
./train_dna2vec_ladder.sh              # or: ./train_dna2vec_ladder.sh medium heavy
```
It refuses to start on anything but 8 GPUs, because the step number the eval
configs name (11718) is `total_samples / (per_device_batch * num_gpus)`.

Every backbone x rung x eval-noise level in one go — skips rungs with no run id
recorded, verifies the accession count after each index build, and prints each
backbone's Table 3 at the end:
```
cd benchmark && ./eval_all_backbones.sh            # or: ./eval_all_backbones.sh dna2vec
```

Evaluate one ablation rung by hand, once per eval-noise level. Configs are
`benchmark/configs/{nt50m,hyenadna,dna2vec}_{none,light,medium,heavy}.yaml`,
which differ only in `checkpoint_path`; the rung's run id is recorded in
`benchmark/runs_<backbone>.yaml`:
```
cd benchmark
for r in 0.0 0.05 0.1; do
  uv run python run_benchmark.py --config configs/nt50m_heavy.yaml --mutation_rate $r
done
```
The first rate builds the index over all 47 accessions and the other two reuse
it — the index depends on the checkpoint, not the mutation rate. Check
`meta.parquet` has 47 rows before trusting results: a build that loses a worker
still gets a `.done` marker, and later runs then load a truncated index and
report quietly wrong recall.

Table 3 on its own, without re-running any eval (see the module docstring for
the runs YAML):
```
uv run python make_table3.py --results_dir results/ \
  --queries <dataset_dir>/queries.parquet --runs runs_nt50m.yaml
```

Smoke test before committing to a ladder — 200 steps, then an end-to-end eval on
3 accessions. Always give the smoke eval a throwaway `index_dir`: a truncated
index still gets a `.done` marker:
```
uv run python train.py --config configs/smoke_nt50m.yaml   # or smoke_{hyenadna,dna2vec}
uv run python run_benchmark.py --config configs/nexus_locale.yaml \
  --model.checkpoint_path <smoke_ckpt> --index_dir /fs/nexus-scratch/ryansynk/smoke_indexes \
  --max_accessions 3 --num_queries 20
```

Training:
```
srun uv run python -m torch.distributed.run --nnodes=4 --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train.py --config configs/config.yaml
```

Plotting:
```
uv run python plot_results.py results/ <dataset_dir>/queries.parquet <dataset_dir>/accs.txt
```
