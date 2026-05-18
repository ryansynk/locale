# LOCALE

This repository contains the code to train and evaluate the model and baselines contained in the paper "LOCALE: Local-Alignment Embeddings for Noise-Robust DNA Search at SRA Scale" by Synk et. al.


The paper presents an embedding model for local alignment of DNA sequences with applications to large-scale sequence search. The repository contains code for reproducing the training, inference, and benchmarking of our model and other baselines (which require extra dependencies, see below)

## Environment

This project uses [uv](https://github.com/astral-sh/uv) for environment management.

```bash
uv sync
```

All commands below should be prefixed with `uv run`.

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
YAML configs for training runs. Given config file matching model used in paper:
- `config.yaml`

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
  --config configs/perlmutter_rawbert.yaml \
  --query_type raw_read \
  --mutation_rate 0.0 \
  --do_timing True \
  --model.exact_search True \
  --results_dir results/
```

To parallelize index construction across multiple nodes run with srun:
```bash
srun --nodes=4 --ntasks-per-node=1 \
  uv run --no-sync python run_benchmark.py \
  --config configs/perlmutter_rawbert.yaml \
  --query_type raw_read \
  --mutation_rate 0.0 \
  --do_timing True \
  --model.exact_search True \
  --results_dir results/
```

### Plotting results

To view results, run:

```bash
uv run python plot_results.py results/ /pscratch/sd/r/rsynk/locale_data/data/sra_recall/raw_read_queries_final.parquet
```

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

## Todos
- Update dense_index search method to not fill with sentinel values
- Change repo name/ everything name to locale
- Move rabitq to faiss