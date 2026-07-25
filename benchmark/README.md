# SRA Recall Experiment

Benchmarks the ability to retrieve relevant SRA accessions given query sequences. Each method builds an index over a set of accessions, searches with query sequences matched to those accessions, and returns a scored ranking. Detailed methodology can be found in the [paper](https://www.biorxiv.org/content/10.64898/2026.05.12.724581v1). Recall@k is computed from the ranked results. 

---

## Datasets

Accessions and queries are pulled from the Hugging Face Hub on first run, selected by `dataset_name`:

| `dataset_name` | Repo | Accessions | Queries |
|---|---|---|---|
| `sra50` | `rsynk/locale-benchmark-sra50` | 47 | 500 |
| `sra500` | `rsynk/locale-benchmark-sra500` | 500 | 500 |

`snapshot_download` materialises the repo into `dataset_dir` (or the default HF cache when unset), yielding `accs.txt`, `queries.parquet`, and `logan_accessions/`. Contigs listed in `accs.txt` are fetched from the public `logan-pub` S3 bucket on first run and cached thereafter.

> `sra50` holds 47 accessions, not 50: three of the original draw were never synced from SRA into the Logan release. No queries target them, so recall is unaffected. See the dataset card for details.

---

## External Dependencies

The following tools must be available on your `PATH`, or given as an absolute path via `model.executable`:

| Tool | Used by |
|---|---|
| `metagraph` | `metagraph_index.py` |
| `mmseqs` | `mmseqs2_index.py` |

> **Metagraph on Perlmutter:** Metagraph must run inside a container. Add `--image=ghcr.io/ratschlab/metagraph:master` to your `salloc` command, and set `executable: shifter metagraph`.
>
> **Metagraph on Nexus:** a native build is used directly; see `configs/nexus_metagraph.yaml`.

---

## Methods

| Config | Method | Notes |
|---|---|---|
| `locale_config.yaml` | LOCALE | This repository |
| `metagraph_config.yaml` | Metagraph | k-mer graph baseline |
| `perlmutter_mmseqs2.yaml` | MMseqs2 | Sequence alignment baseline |
| `nexus_locale.yaml` | LOCALE | Nexus paths |
| `nexus_metagraph.yaml` | Metagraph | Nexus paths, native binary |

---

## Running the Benchmark

### Single node

```bash
uv run python run_benchmark.py \
  --config configs/nexus_locale.yaml \
  --model.checkpoint_path /path/to/checkpoint.pth.tar \
  --model.pooling mean \
  --model.max_seq_len 256 \
  --mutation_rate 0.0
```

Results are written to `<results_dir>/<experiment_id>/raw_read_mut_<mutation_rate>.parquet`, where `experiment_id` encodes the method, checkpoint, and chunking config.

Use `--num_queries` to subsample (default 1000, capped at the dataset size) and `--no_search` to build the index and exit.

### Multi-node index building (dense methods only)

`run_benchmark.py` has built-in SLURM-aware sharding. When run with `srun` across multiple nodes, each node builds a shard of the index independently, then node 0 waits for all shards and merges them before running search.

```bash
srun uv run python run_benchmark.py --config configs/nexus_locale.yaml ...
```

An index is rebuilt only when `<index_dir>/<index_suffix>/.done` is absent, so reruns reuse existing indexes.

---

## Plotting Results

`plot_results.py` takes three positional arguments: the results directory, the queries parquet, and the accessions list. The latter two come from the downloaded dataset.

```bash
uv run python plot_results.py \
  results/ \
  <dataset_dir>/queries.parquet \
  <dataset_dir>/accs.txt
```

Recall@k curves are written to `plots_matplotlib/` by default (`--plots_dir` to override).
