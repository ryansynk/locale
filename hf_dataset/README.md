# hf_dataset: publishing an index as a Hugging Face embedding bundle

Turns a finished benchmark index into the `embeddings/<label>/` directory the
`rsynk/locale-benchmark-*` dataset repos serve, and computes the ground truth
for it. Published so far: `sra50` and `sra500` (768-d fp32, 2026-08) and
`sra4571` (`locale-pca128-bf16`, 2026-10-19).

## Bundle layout

```
embeddings/<label>/
├── manifest.json      n, dim, dtype, num_shards, rows_per_shard, metric, normalized,
│                      encoder config (from the index's config.json), pca block if any
├── meta.parquet       srr_id, start_row, num_rows   (global row -> accession)
├── projector.pt       only for a PCA index: (encoder_dim, pca_dim) fp32
├── base/embeddings-00000.<fbin|fp16bin|bf16bin> ...   ~1 GB shards, own [n, d] header
└── mut0.00/ mut0.05/ mut0.10/
    ├── query.fbin           one row per query chunk and strand, base's dimension
    ├── query_meta.parquet   query_id, start_row, num_rows, num_forward
    └── gt.bin               exact top-k: uint32 nq, k, ids[nq*k]; float32 dists[nq*k]
```

`score.py` (tracked in `locale-data/benchmark/`, uploaded to each repo root)
scores a result file in the `gt.bin` layout and needs nothing but numpy and
pyarrow.

## One command

```bash
cd /pscratch/sd/r/rsynk/locale
CONFIG=benchmark/configs/sra4571/perlmutter_locale_sra4571_endwin0_minlen100_pca_d128_bf16.yaml \
LABEL=locale-pca128-bf16 sbatch hf_dataset/export_bundle.sbatch
```

`CONFIG` is the benchmark yaml whose index is `.done`; the encoder dir,
sub-index (`pca` / `quantized_exact` export their projected or 2-byte file,
anything else the encoder's fp32 fbin), bundle dir and dataset name all come
from it. `LABEL` names the directory under `embeddings/`; use
`<encoder>[-<index>]`, e.g. `locale` or `locale-pca128-bf16`. The job runs on
one 4-GPU node, about 25 minutes for sra4571, and ends by printing the oracle
table for the card. Output: `/pscratch/sd/r/rsynk/hf_bundles/<dataset>_embed/`
(override with `STAGE`). Re-running skips finished shards and ground truth.
The same file runs as `bash hf_dataset/export_bundle.sbatch` on an
interactive GPU node.

The steps it runs, if one is needed alone:

| Step | Script | Notes |
|---|---|---|
| shards | `python -m hf_dataset.shard_fbin --src <index or sub-index dir> --out <OUT>` | fp32 / fp16 / bf16; values copied verbatim |
| queries | `python -m hf_dataset.generate_queries --dataset_path <bundle> --cfg <yaml> --out <OUT> [--projector ...]` | all queries, both strands, chunked at max_seq_len |
| ground truth | `python -m hf_dataset.ground_truth --base <merged file> --queries_dir <OUT> --k 100` | benchmark's ExactEngine, all GPUs, TF32 off |

## Upload

The staging dir mirrors the repo, so push it from a login node under tmux
(resumable from `.cache/huggingface/` inside the staging dir; keep that until
the last commit lands):

```bash
cd /pscratch/sd/r/rsynk/hf_bundles/<dataset>_embed
/pscratch/sd/r/rsynk/locale-data/.venv/bin/hf upload-large-folder rsynk/locale-benchmark-<dataset> . --repo-type=dataset --num-workers=8
```

Then the card and scorer, from `locale-data`:

```bash
.venv/bin/hf upload rsynk/locale-benchmark-<dataset> cards/<dataset>.md README.md --repo-type=dataset
.venv/bin/hf upload rsynk/locale-benchmark-<dataset> benchmark/score.py score.py --repo-type=dataset
```

The card needs the oracle table the job printed, the encoder config from
`manifest.json`, and a note on anything that differs from the sra500 layout.
Repos above ~300 GB need a request to Hugging Face first.

## Conventions worth keeping

- Row ids are the encoder fbin's rows everywhere: base shards, `meta.parquet`,
  `gt.bin`, and the benchmark's saved hits all agree.
- Both strands: inside a query's row range the forward chunks come first,
  then the reverse complement's; scorers take the max over the range.
- A new encoder or index type only needs the merged file to use the fbin
  header and 4- or 2-byte values; `shard_fbin.py` and `ground_truth.py`
  detect the dtype from the filename.
