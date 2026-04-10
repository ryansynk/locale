# Rawbert

Embedding model for local alignment of DNA sequences. The goal is to convert sequence search over large sets of sequences (e.g. NIH Sequence Read Archive) into vector search — scalable and robust to noise compared to k-mer methods like Metagraph.

## Environment

This project uses [uv](https://github.com/astral-sh/uv) for environment management.

```bash
uv sync
```

All commands below should be prefixed with `uv run`.

---

## Code Structure

### `rawbert/`
Core library used for training and embedding:
- `modeling/` — BERT-based model definition (`model.py`, `bert_layers.py`, etc.)
- `training/` — Training logic and data batchers (unsupervised, supervised, containment, badread, etc.)
- `config.py` — Top-level config dataclass

### `train.py`
Entry point for model training. Configured via YAML files in `configs/`.

### `configs/`
YAML configs for training runs. Key configs:
- `unsupervised_perlmutter_containment.yaml` — unsupervised MoCo training on Perlmutter
- `kl_finetune_perlmutter.yaml` — KL divergence finetuning on Perlmutter

---

## Training on Perlmutter (4-node)

Allocate nodes:
```bash
salloc --nodes 4 --qos debug --time 0:30:00 --ntasks-per-node 1 -c 128 \
  --mem 0 --constraint gpu --gpus-per-node 4 --account m5083_g
```

Set up distributed env vars:
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
    train.py --config configs/kl_finetune_perlmutter.yaml
```

---

## Main Experiment: `experiments/sra_recall`

Tests the ability to retrieve relevant SRA accessions given query sequences. Each method implements an index with build and search functionality. Results are scored by recall@k.

### Structure

- `run_benchmark.py` — Runs a search method against the benchmark queries
- `build_index_parallel.py` — Parallel index construction
- `plot_results.py` — Loads result scores and plots recall@k curves
- `src/` — Index implementations:
  - `dense_index.py` — Vector embedding search (wraps `rawbert` model); base class for learned methods
  - `metagraph_index.py` — k-mer graph search via Metagraph
  - `mmseqs2_index.py` — Sequence search via MMseqs2
  - `mantis_index.py` — Search via Mantis
  - `evo2_index.py` — Evo2-based index (**currently broken**)
  - `base_index.py` — Abstract base class
  - `config.py` — Config dataclasses for all index types

### External Dependencies (must be in PATH)

The following tools must be available on your PATH to run the full benchmark:

- [`metagraph`](https://github.com/ratschlab/metagraph)
- [`mantis`](https://github.com/splatlab/mantis)
- [`mmseqs`](https://github.com/soedinglab/MMseqs2)

> **Metagraph on Perlmutter:** Metagraph requires running inside a container. Add `--image=ghcr.io/ratschlab/metagraph:master` to your `salloc` command when using Metagraph.

### Example benchmark command

```bash
uv run python run_benchmark.py \
  --config configs/perlmutter_rawbert.yaml \
  --model.checkpoint_path /pscratch/sd/r/rsynk/rawbert/checkpoints/ge6jbfvp/checkpoint7000.pth.tar \
  --model.pooling mean \
  --model.max_seq_len 256 \
  --model.chunk_type stride \
  --query_type gencode \
  --mutation_rate 0.0
```

### Plotting results

```bash
uv run python plot_results.py results/ /pscratch/sd/r/rsynk/rawbert_data/data/sra_recall/raw_read_queries_final.parquet
```

---

## Other Experiments

All other experiment folders are **deprecated** and not actively maintained:

- `experiments/adversarial_benchmark/`
- `experiments/comparison_benchmark/`
- `experiments/cutoff/`
- `experiments/query-read-embedding/`
