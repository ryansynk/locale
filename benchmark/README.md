# SRA Recall Experiment

Benchmarks the ability to retrieve relevant SRA accessions given query sequences. Each method builds an index over a set of accessions, searches with queries, and returns a scored ranking. Recall@k is computed from the ranked results.

---

## Data Requirements

This experiment depends on data files generated from a separate repository. Either generate them yourself or **ask Ryan for copies**.

Expected paths (configured in `configs/perlmutter_*.yaml`):

| File | Description |
|---|---|
| `/pscratch/sd/r/rsynk/locale_data/data/sra_recall/raw_read_queries_final.parquet` | Raw read queries |
| `/pscratch/sd/r/rsynk/locale_data/data/sra_recall/logan_contig_queries_final.parquet` | Logan contig queries |
| `/pscratch/sd/r/rsynk/locale_data/data/sra_recall/gencode/gencode_queries.parquet` | Gencode queries |
| `/pscratch/sd/r/rsynk/data/test/logan_contig/` | Accession FASTA files (`*.contigs.fa`) |

---

## External Dependencies

The following tools must be available on your `PATH`:

| Tool | Used by |
|---|---|
| `metagraph` | `metagraph_index.py` |
| `mantis` | `mantis_index.py` |
| `squeakr` | `mantis_index.py` (Mantis dependency) |
| `seqtk` | `mantis_index.py` (Mantis dependency) |
| `mmseqs` | `mmseqs2_index.py` |

> **Metagraph on Perlmutter:** Metagraph must run inside a container. Add `--image=ghcr.io/ratschlab/metagraph:master` to your `salloc` command.

---

## Query Types

Three query types are supported via `--query_type`:

- `raw_read` — short raw sequencing reads
- `logan_contig` — assembled contigs from the Logan dataset
- `gencode` — reference transcript sequences from Gencode

---

## Methods

### Actively maintained
| Config | Method | Notes |
|---|---|---|
| `perlmutter_locale.yaml` | Rawbert (this repo) | Requires a checkpoint path |
| `perlmutter_metagraph.yaml` | Metagraph | k-mer graph baseline |
| `perlmutter_mmseqs2.yaml` | MMseqs2 | Sequence alignment baseline |
| `perlmutter_mantis.yaml` | Mantis | k-mer counting baseline |

### Comparison baselines
| Config | Method | Notes |
|---|---|---|
| `perlmutter_dnabert.yaml` | DNABERT | Pretrained DNA BERT |
| `perlmutter_dna2vec.yaml` | DNA2Vec | k-mer embedding baseline |
| `perlmutter_llmed.yaml` | LLMED | LLM-based embedding |
| `perlmutter_neuroseed.yaml` | NeuroSEED | Requires separate NeuroSEED install |
| `perlmutter_generator.yaml` | Generator | |

### Broken
| Config | Method | Notes |
|---|---|---|
| `perlmutter_evo2.yaml` | Evo2 | **Currently broken** |

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

### Interactive Streamlit app

Lets you select any subset of models, mutation rates, and query types from a sidebar and view all plots interactively.

**On Perlmutter**, start the app on a login node:

```bash
cd /pscratch/sd/r/rsynk/locale/experiments/sra_recall
uv run streamlit run app.py
```

Note which login node you are on (e.g. `login25`).

**On your local machine**, open a tunnel to that specific login node:

```bash
ssh -L 8501:login25:8501 rsynk@perlmutter.nersc.gov
```

Then open `http://localhost:8501` in your browser.
