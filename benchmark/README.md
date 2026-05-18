# SRA Recall Experiment

Benchmarks the ability to retrieve relevant SRA accessions given query sequences. Each method builds an index over a set of accessions, searches with query sequences matched to those accessions, and returns a scored ranking. Detailed methodology can be found in the [paper](https://www.biorxiv.org/content/10.64898/2026.05.12.724581v1). Recall@k is computed from the ranked results. 

---

## External Dependencies

The following tools must be available on your `PATH`:

| Tool | Used by |
|---|---|
| `metagraph` | `metagraph_index.py` |
| `mmseqs` | `mmseqs2_index.py` |

> **Metagraph on Perlmutter:** Metagraph must run inside a container. Add `--image=ghcr.io/ratschlab/metagraph:master` to your `salloc` command.

---

## Methods

| Config | Method | Notes |
|---|---|---|
| `perlmutter_locale.yaml` | LOCALE | This repository |
| `perlmutter_metagraph.yaml` | Metagraph | k-mer graph baseline |
| `perlmutter_mmseqs2.yaml` | MMseqs2 | Sequence alignment baseline |

---

## Running the Benchmark

### Single node

```bash
uv run python run_benchmark.py \
  --config configs/perlmutter_locale.yaml \
  --model.checkpoint_path /path/to/checkpoint.pth.tar \
  --model.pooling mean \
  --model.max_seq_len 256 \
  --model.chunk_type stride \
  --query_type gencode \
  --mutation_rate 0.0
```

Results are written to `results/<method_id>/<query_type>_mut<mutation_rate>.parquet`.

### Multi-node index building (dense methods only)

`run_benchmark.py` has built-in SLURM-aware sharding. When run with `srun` across multiple nodes, each node builds a shard of the index independently, then node 0 waits for all shards and merges them before running search.

```bash
srun uv run python run_benchmark.py --config configs/perlmutter_locale.yaml ...
```

Alternatively, use `build_index_parallel.py` to build index shards across many tasks (one GPU per task) without the merge step — useful when you want to pre-build a large index separately from the search step. Note: this script only supports `DenseConfig` and `MetagraphConfig`.

---

## Plotting Results

### Static plots (saved to disk)

```bash
uv run python plot_results.py results/ \
  /pscratch/sd/r/rsynk/locale_data/data/sra_recall/raw_read_queries_final.parquet \
  /pscratch/sd/r/rsynk/locale_data/data/sra_recall/gencode/gencode_queries.parquet
```

Recall@k curves are written to `plots/`.
