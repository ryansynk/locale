# LOCALE

This repository contains the code to train and evaluate the model and baselines contained in the paper "LOCALE: Local-Alignment Embeddings for Noise-Robust DNA Search at SRA Scale" by Synk et. al.


The paper presents an embedding model for local alignment of DNA sequences with applications to large-scale sequence search. The repository contains code for reproducing the training, inference, and benchmarking of our model and other baselines (which require extra dependencies, see below)

## Environment

This project uses [uv](https://github.com/astral-sh/uv) for environment management.

```bash
uv sync
```

All commands below should be prefixed with `uv run`.

### Optional extras

| Extra | Install | Needed for |
|---|---|---|
| `flash` | `uv sync --extra flash` | Faster attention on DNABERT-2. **Requires Python 3.11 + CUDA 12 + torch 2.6** — the wheel is pinned to that exact combination. |
| `evo2` | `uv sync --extra evo2` | The Evo2 baseline only. |
| `dev` | `uv sync --extra dev` | Tests (`pytest`), linting. |

Without `flash`, attention falls back to the model's native path. ALiBi is still
applied, so embeddings remain valid, but throughput is lower and fp16 numerics
can differ slightly from the runs reported in the paper. A warning is printed
whenever the fallback is in use — do not ignore it when comparing against
published numbers.

---

## Quickstart

Smallest end-to-end run: retrieve 20 queries against an index built over 3 SRA
accessions, using the checkpoint published with the paper.

```bash
uv sync
cd benchmark
uv run python run_benchmark.py \
  --config configs/locale_config.yaml \
  --max_accessions 3 \
  --num_queries 20 \
  --index_dir /tmp/locale_smoke_index
```

The throwaway `--index_dir` is not optional hygiene. A truncated index still
gets its `.done` marker, and an index is only rebuilt when that marker is
absent — so a 3-accession smoke index left in the default location would be
silently reused by a later full run, which then reports quietly wrong recall.

`locale_config.yaml` leaves `checkpoint_path` unset, which downloads the paper's
checkpoint (466 MB) from [`rsynk/locale`](https://huggingface.co/rsynk/locale),
pinned to a fixed revision. Everything is cached after the first run.

**Requires at least one CUDA GPU.** Index building fans out one worker process
per GPU and raises `RuntimeError: No GPUs available for building the index.` on
a CPU-only machine. There is currently no CPU path for the benchmark; embedding
sequences directly with `LOCALEEncoder` does work on CPU.

What the first run fetches, all cached afterwards:

| Item | Size | Notes |
|---|---|---|
| `sra50` dataset | 4 files | `accs.txt`, `queries.parquet`, metadata |
| Logan contigs | 1.3 GB | All 47 accessions, ~10 s from the `logan-pub` S3 bucket |
| LOCALE checkpoint | 466 MB | From the Hub, unless `checkpoint_path` is set |
| DNABERT-2 | ~500 MB | Backbone weights and tokenizer |

`--max_accessions` limits what gets **indexed**, not what gets downloaded: the
full accession manifest is verified before truncation, so a smoke run still
fails loudly on an incomplete dataset rather than quietly indexing a subset.

---

## Code Structure

### `lae/`
Core library used for training and embedding:
- `modeling/` — BERT-based model definition (`model.py`, `bert_layers.py`, etc.)
- `training/` — Training logic and data batcher
- `config.py` — Top-level config dataclass

### `train.py`
Entry point for model training. Configured via YAML files in `configs/`.

### `configs/`
YAML configs for training runs.
- `config.yaml` — the configuration used for the model in the paper
- `{nt50m,hyenadna,dna2vec}_{none,light,medium,heavy}.yaml` — the augmentation
  ladder, one file per backbone and augmentation strength
- `smoke_{nt50m,hyenadna,dna2vec}.yaml` — 200-step runs for checking a setup

---

## Training (4 Node Example)

Set up distributed env vars (may vary for different node counts):
```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
```

Launch training:
```bash
srun uv run python -m torch.distributed.run \
    --nnodes=4 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py --config configs/config.yaml
```

---

## Benchmark on SRA Data: `benchmark`

Tests the ability to retrieve relevant SRA accessions given query sequences. Each method implements an index with build and search functionality. Results are scored by recall@k, auprc. Further details are given in the `benchmark` README.

### Structure

- `run_benchmark.py` — Runs a search method against the benchmark queries
- `plot_results.py` — Loads result scores and plots recall@k curves
- `src/` — Index implementations:
  - `dense_index.py` — Vector embedding search (wraps `locale` model); base class for learned methods
  - `metagraph_index.py` — k-mer graph search via Metagraph
  - `mmseqs2_index.py` — Sequence search via MMseqs2
  - `base_index.py` — Abstract base class
  - `config.py` — Config dataclasses for all index types

### External Dependencies (must be in PATH)

The following tools must be available on your PATH to run the full benchmark:

- [`metagraph`](https://github.com/ratschlab/metagraph)
- [`mmseqs`](https://github.com/soedinglab/MMseqs2)

> **Metagraph on Perlmutter:** Metagraph requires running inside a container. Add `--image=ghcr.io/ratschlab/metagraph:master` to your `salloc` command when using Metagraph.

### Example benchmark command

First navigate to benchmark dir
```bash
cd benchmark
```

Then run the benchmark:
```bash
uv run python run_benchmark.py \
  --config configs/locale_config.yaml \
  --mutation_rate 0.0 \
  --do_timing True \
  --model.exact_search True \
  --results_dir results/
```

To parallelize index construction across multiple nodes run with srun. Each node
builds a shard independently; node 0 waits for all shards and merges them before
searching:
```bash
srun --nodes=4 --ntasks-per-node=1 \
  uv run --no-sync python run_benchmark.py \
  --config configs/locale_config.yaml \
  --mutation_rate 0.0 \
  --do_timing True \
  --model.exact_search True \
  --results_dir results/
```

### Plotting results

`plot_results.py` takes three positional arguments: the results directory, the
queries parquet, and the accessions list. The latter two come from the
downloaded dataset — `snapshot_download` reports where it materialised the repo,
or pass `dataset_dir` in the config to choose the location yourself.

```bash
uv run python plot_results.py \
  results/ \
  <dataset_dir>/queries.parquet \
  <dataset_dir>/accs.txt
```

Recall@k curves are written to `plots_matplotlib/` (`--plots_dir` to override).
Rendering uses LaTeX, so a working `latex` installation is required; to print the
metric tables without plotting, use `print_results.py` with the same arguments.

## Citation

If you use LOCALE in your work, you can cite the preprint here:
```
@misc{synk2026locale,
	title = {{LOCALE}: {Local}-{Alignment} {Embeddings} for {Noise}-{Robust} {DNA} {Search} at {SRA} {Scale}},
	shorttitle = {{LOCALE}},
	url = {https://www.biorxiv.org/content/10.64898/2026.05.12.724581v1},
	doi = {10.64898/2026.05.12.724581},
	urldate = {2026-05-15},
	publisher = {bioRxiv},
	author = {Synk, Ryan and Pandey, Prashant and Sahinalp, Cenk and Duraiswami, Ramani},
	month = may,
	year = {2026},
	note = {ISSN: 2692-8205
Pages: 2026.05.12.724581
Section: New Results},
}
```
