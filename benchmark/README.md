# SRA Recall Experiment

Benchmarks the ability to retrieve relevant SRA accessions given query sequences. Each method builds an index over a set of accessions, searches with query sequences matched to those accessions, and returns a scored ranking. Detailed methodology can be found in the [paper](https://www.biorxiv.org/content/10.64898/2026.05.12.724581v1). Recall@k is computed from the ranked results. 

---

## Datasets

`dataset_dir` must already hold `accs.txt` and the per-rate query files (`queries_mut<rate>.parquet`); `run_benchmark.py` downloads neither. Fetch a published bundle with `python fetch_dataset.py <name> <dataset_dir>`:

| `dataset_name` | Repo | Accessions | Queries |
|---|---|---|---|
| `sra50` | `rsynk/locale-benchmark-sra50` | 47 | 500 |
| `sra500` | `rsynk/locale-benchmark-sra500` | 500 | 500 |

Contigs listed in `accs.txt` are fetched from the public `logan-pub` S3 bucket on first run and cached thereafter.

> `sra50` holds 47 accessions, not 50: three of the original draw were never synced from SRA into the Logan release. No queries target them, so recall is unaffected. See the dataset card for details.

---

## External Dependencies

The following tools must be available on your `PATH`, or given as an absolute path via `model.executable`:

| Tool | Used by |
|---|---|
| `metagraph` | `metagraph_index.py` |
| `mmseqs` | `mmseqs2_index.py` |

> **Metagraph on Perlmutter:** Metagraph must run inside a container. Add `--image=ghcr.io/ratschlab/metagraph:master` to your `salloc` command, and set `executable: shifter metagraph`.

---

## Methods

Configs live under `configs/<dataset>/` (`sra50`, `sra500`, `sra4571`,
`sra55viral`) plus the checkpoint sweeps `configs/ablation_sweep/` and
`configs/backbone_sweep/`. Per dataset:

| Config | Method | Notes |
|---|---|---|
| `perlmutter_locale_*.yaml` | LOCALE | This repository; `_ivfpq` / `_rabitq` / `_ivf` variants swap the index engine |
| `perlmutter_esa_*.yaml` | ESA (dna2vec) | Embed-Search-Align encoder, same protocol |
| `perlmutter_llmed_*.yaml` | LLM-ED | Same protocol |
| `perlmutter_metagraph_*.yaml` | Metagraph | k-mer graph baseline |
| `perlmutter_mmseqs2_*.yaml` | MMseqs2 | Sequence alignment baseline |

---

## Running the Benchmark

### Single node

```bash
uv run python run_benchmark.py \
  --config configs/sra50/perlmutter_locale_sra50v2.yaml \
  --model.encoder.checkpoint_path /path/to/checkpoint.pth.tar \
  --model.encoder_label locale@myrun \
  --mutation_rate 0.0
```

`--mutation_rate` selects a pre-mutated query file from the dataset bundle, `queries_mut<rate>.parquet` (0.00 / 0.05 / 0.10; written by `locale-data/benchmark/mutate_queries.py` with mutation-simulator). The benchmark itself never mutates sequences, so every run at a rate searches identical queries.

Results are written to `<results_dir>/<encoder_label>/<index_label>/<search_label>/mut<mutation_rate>.parquet` (see "Encoder + index" below).

Use `--num_queries` to subsample (default 1000, capped at the dataset size) and `--no_search` to build the index and exit.

### Multi-node index building (dense methods only)

`run_benchmark.py` shards by `--shard` / `--num_shards`, which the launcher passes (it reads no scheduler variables). With `--num_shards N`, process `r` builds `shard_r/` of the index; under `--stage all` (default) shard 0 waits for all shards and merges them, then the search runs. `--stage embed` exits once this shard is built, `--stage merge` (CPU) joins the shards and refuses unless all `N` are `.done`, `--stage engine` builds one shard of the engine artifact (IVF-PQ: one process per GPU), and `--stage search` searches built indexes.

```bash
srun bash -c 'exec "$@" --shard "$SLURM_NODEID" --num_shards "$SLURM_NNODES"' _ \
  uv run python run_benchmark.py --config configs/sra500/perlmutter_locale_sra500v2.yaml ...
```

An index is rebuilt only when `<index_dir>/<encoder_label>/.done` is absent, so reruns reuse existing indexes.

### Encoder + index

A dense method is an encoder plus an index (`model.encoder` and
`model.index` in the yaml; `src/config.py`). Encoders write
`<index_dir>/<encoder_label>/embeddings.fbin` (+ `meta.parquet`); indexes read
it and keep their own artifact under `<index_dir>/<encoder_label>/<index_label>/`.
Results land in `<results_dir>/<encoder_label>/<index_label>/<search_label>/`
(`mut<rate>.parquet`, raw vector hits in `hits/`). Labels are directory
names chosen in the yaml; `config.json` in each directory is the truth
(encoder identity, engine + `BUILD` fields, engine + `SEARCH` fields +
strands + query draw) and is checked on load: a run whose config differs
from what a label was built or searched under is refused. Non-dense methods
(metagraph, mmseqs) have no encoder level: `<index_dir>/<label>/` and
`<results_dir>/<label>/<search_label>/`. To add an index engine: one module
with `build/load/topk_hits/size_gb` (see `src/engines.py`), one dataclass with
`BUILD`/`SEARCH` in `src/config.py`, one line in `make_engine`.

### Scoring protocol (dense methods: LOCALE, ESA/dna2vec, LLM-ED, ...)

All dense methods share `DenseIndex`, so these settings apply to every
`DenseMethod` run. Index building is unaffected: every protocol reads the same
`<index_dir>/<encoder_label>/` and the `.done` marker logic is unchanged.

**Default: top-k regroup.** Each query's `model.index.top_k` (default 100)
nearest index *vectors* are retrieved, grouped by accession, each accession is
scored by its max hit and the accessions are ranked; accessions with no hit
get the miss sentinel (-2.0) and rank below every scored one. The vector
engine is `model.index.engine`: `exact` (fp32 scan), `rabitq` (1-bit RaBitQ
codes), `ivfpq` (cuVS GPU IVF-PQ) or `ivfrabitq` (faiss IVF + RaBitQ). Each
index label has its own results subtree, so the results never overwrite an
exhaustive run's.

**Persisted hits and k-sweeps.** The raw hits are written to
`<results_dir>/<encoder>/<index>/<search>/hits/mut<rate>.parquet`: one row
per query with a score-descending list of `{accession, score, vector_id}`
structs plus the run metadata columns; k is in the search `config.json`. A
run whose `top_k` is at most a sibling search label's k (same encoder,
index, strands and engine search settings; queries covered) loads and
truncates that file instead of scanning. A k-sweep is therefore one scan at
the largest k, then reruns at smaller k under their own `search_label`.
`print_results.py` skips `hits/` when it collects results. Timing runs
(`--do_timing`) always search, never persist hits, and write `mut<rate>.timing.parquet` beside the accuracy results. When several runs share a
display name (`LOCALE`), `print_results.py` keeps the full run keys apart in
its tables instead of pooling them (`--full_model_names` overrides).

**Reference: exhaustive.** `model.index.engine: exhaustive` scores every
accession by the max over all its vectors (the original full-dense scan). No
hits are persisted.

**Both strands.** `model.encoder.both_strands: true` (the default) also
embeds each query's reverse complement, retrieves `top_k` hits for both,
unions the two lists (deduplicated by vector, max score, cut back to `top_k`)
and then regroups as above; the timing runs include the second embedding.
Under `exhaustive` the accession keeps the better of its two strand scores.
It is part of the search identity (`config.json`), not the index's. Metagraph
and MMseqs2 already see both strands and have no such flag. Example:
`configs/sra4571/perlmutter_locale_sra4571_bothstrands.yaml`.

`model.exact_search: true` in older configs is accepted as a no-op (that
protocol is now the default).

### Multi-node metagraph

Under a multi-node `srun`, node r builds a complete metagraph index of
`accessions[r::N]` into `<index>/shard_r/` (with disk swap and memory caps
sized from node RAM), and node 0 writes `graphs.csv` listing every shard plus
the joined manifest. Node 0 alone then runs one `server_query` over all
shards; an accession's k-mer count does not depend on which other accessions
share its graph, so the results equal a single joint index (each shard's
top-100 is unioned and cut back to 100). `index_size_gb` sums the shards,
which overstates a joint index by the duplicated graph part. See
`slurm_scripts/perlmutter_sra4571_metagraph.sbatch`; the same node count must
be reused when resuming. MMseqs2 and the centroid index are still single node.

### Multi-node search (dense methods only)

With the index built, a multi-node `srun` also shards the search. The default
top-k protocol has each node scan an equal range of vector *rows* (exact fp32
or RaBitQ) and node 0 merge the per-query top-k lists; the exhaustive
protocol deals accessions out across nodes and node 0 reassembles the
per-accession scores. Engines that hold the whole index on one node (IVF-PQ,
IVF-RaBitQ) search on node 0 alone. A run that finds cached hits skips the
scan on every node.

With `model.index.engine: rabitq` the vector-level path runs over 1-bit
RaBitQ codes instead of fp32 rows. The codes are built once into
`<index_dir>/<encoder_label>/<index_label>/`, one shard per node when the
build itself runs under a multi-node `srun` (the centroid is estimated from
`model.index.sample_rows` sampled rows rather than a full pass), and at
search time each node loads only its row range of packed codes onto its GPUs
(~96 B/vector at 768 dims).

### Single-node IVF search (sra4571 scale)

Two partitioned vector engines search the whole 2.06 B-vector sra4571 index
from one node; both return the standard hits frame (top-k regroup protocol)
and every search knob is part of the experiment id.

**GPU IVF-PQ (`model.use_ivfpq`, cuVS; the metagraph-parity engine).**
`ivfpq_num_shards` (16) independent IVF-PQ indexes over contiguous fbin row
ranges, `ivfpq_lists_per_shard` lists each, 128 x 8-bit PQ codes (PQ at 96 B
ranks clearly worse than 1-bit RaBitQ on these embeddings; 128 B matches it).
The 280 GB of codes + ids sit 4 shards per card on one 4 x A100-80GB node
(`-C "gpu&hbm80g"`), one worker process per GPU that also embeds its slice of
the queries (a cuVS search call blocks its host thread, so threads would run
the GPUs one after another). `ivfpq_nprobe` lists are probed per shard;
`ivfpq_rerank` re-scores each query chunk's best candidates exactly from the
fbin (Lustre preads: cheap at 10 per chunk, too slow for hundreds).

```bash
sbatch slurm_scripts/perlmutter_sra4571_ivfpq_build.sbatch configs/sra4571/perlmutter_locale_sra4571_ivfpq_L16k.yaml  # 4 GPU nodes, ~20 min
CONFIG=configs/sra4571/perlmutter_locale_sra4571_ivfpq_L16k.yaml sbatch slurm_scripts/perlmutter_sra4571_ivfpq_search.sbatch  # do_timing
```

**CPU IVF-RaBitQ (`model.use_ivf`, faiss).** 1-bit RaBitQ residual codes in
`ivf_nlist` spherical k-means cells (~112 B/vector, 231 GB, fits a 512 GB CPU
node), FastScan, flat or HNSW (`ivf_quantizer: hnsw`) coarse quantizer, exact
fp32 rerank of `ivf_rerank` candidates. Near-exact accuracy, but ~60-190 s per
1000 queries on sra4571: the accuracy reference for the GPU engine, not a
parity option. Built by `perlmutter_sra4571_ivf_build.sbatch` (8 GPU nodes,
~11 min); FastScan shards are searched side by side because faiss'
`merge_from` corrupts the heap on FastScan indexes of this size.

`ivf_probe.py` measures, before any build, how many lists must be probed to
keep the exact top-k hits of noised queries (finer cells win at small scanned
fractions); `ivf_sweep.py` loads either index once and sweeps nprobe x rerank
over all rates, writing results/hits exactly as run_benchmark does.

Metagraph's warm query time depends on client parallelism:
`model.server_parallel: N` starts `server_query -p N` and splits the batch
into N concurrent requests (experiment id `metagraph_k31_p<N>`); the default
1 is the original single-request protocol.

## Plotting Results

`plot_results.py` takes three positional arguments: the results directory, the queries parquet, and the accessions list. The latter two come from the downloaded dataset.

```bash
uv run python plot_results.py \
  results/ \
  <dataset_dir>/queries.parquet \
  <dataset_dir>/accs.txt
```

Recall@k curves are written to `plots_matplotlib/` by default (`--plots_dir` to override).
