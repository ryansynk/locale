# Separability probe

A go/no-go geometry probe on the released LOCALE checkpoint. It asks one
question and nothing else:

> In LOCALE embedding space, is there a radius **wider** than the divergence
> among true carriers of a mobile genetic element, and **narrower** than the
> distance to the nearest functionally different homolog?

If such a window exists, and is wider than what identity clustering achieves,
then a downstream association study can pool every carrier of an element into
one feature column with usable carrier frequency `f_j`. If the bands interleave,
it cannot, and the approach fails before it starts.

This probe **does not** fine-tune anything, build a FAISS/ANN index, touch the
benchmark's `index_dir`, or go through `run_benchmark.py`. It is direct
embedding plus pairwise similarity over a few thousand sequences. Nothing
outside `experiments/separability/` is modified.

The findings are in [`report.md`](report.md).

---

## What it actually measures

Three elements, each with a carrier set that must pool and a confusable set
that must not:

| Element | Carriers | Confusables |
|---|---|---|
| `pks_clb` | `clbA`–`clbS` (colibactin island) in *Escherichia*, *Klebsiella*, *Citrobacter* | yersiniabactin (`irp`/`ybt`, the adjacent NRPS-PKS hybrid), enterobactin (`entB/E/F`) |
| `cdt_abc` | `cdtABC` in *Campylobacter*, *Escherichia*, *Haemophilus*, *Shigella*, *Helicobacter*, *Aggregatibacter* | *Salmonella* typhoid-toxin `cdtB` (same name, same fold, different holotoxin), plus `hlyA`/`stx`/`elt`/`cnf1` |
| `bla_ctxm` | `blaCTX-M` alleles, any genus | `blaTEM`, `blaSHV`, `blaOXA` |

Two granularities, both scored by **max over windows** (late-interaction), never
by mean pooling — pooled accession representation is an open problem in this
codebase and this probe does not need it solved:

- **gene** — query is a whole gene, all of its windows; similarity is the max
  over (query window, reference window) pairs.
- **window** — query is a single 256 bp window, i.e. a simulated read;
  similarity is the max over the reference gene's windows. This is the
  benchmark's own regroup protocol.

Two carrier definitions, because they answer different halves of the question
and conflating them makes any multi-gene element meaningless:

- **`same_gene`** — a query is scored only against carriers of its *own* gene.
  A `clbB` query is compared to other `clbB` sequences; `clbA` is masked out of
  both numerator and denominator, since no method should be expected to retrieve
  it. This is the "is the radius wider than the divergence among true carriers"
  half.
- **`element`** — carriers are *accessions* (strains), captured when any window
  of any of their carrier genes falls inside the radius. This is the benchmark's
  own regroup-to-accession protocol and the half `f_j` consumes. An accession
  holding both a carrier and a confusable gene counts as a carrier, not as
  contamination.
- **`element_ge<N>genes`** — as `element`, but carrier accessions must hold at
  least `N` distinct element genes (`N` = 2 for a small operon, a quarter of the
  genes for a large island). GenBank contigs do not span a 54 kb island, so the
  median pks carrier accession deposits only 2 of 17 clb genes, and an accession
  that deposited only `clbK` cannot be retrieved by a `clbB` query. Without the
  restriction the `element` scope measures deposition practice rather than the
  embedding; treat unrestricted `element` as a lower bound and this as the
  interpretable estimate. Accessions below the threshold are dropped from the
  target set but still allowed as queries.

Without the split, the pks island's 19 `clb` genes cap "fraction of carrier
genes captured" near 1/19 for every method, at any radius. blaCTX-M hides the
problem because it is a single gene family — which is exactly why a probe on one
element would have missed it.

Three noise levels on the **query** side only (0%, 5%, 10%) — the reference set
stays clean, which is the read-against-catalog setting. Clean sequences are
where identity clustering is strongest, so a LOCALE win, if there is one, has
to show up under noise.

Baselines on the same units and the same radius grid: exact canonical 31-mers
(containment), MMseqs2 `easy-search` alignment identity, and MMseqs2
`easy-cluster` catalogs at 95/90/80/70/50% identity with each query assigned to
the catalog by its best alignment.

### Sequence length — read this before changing anything

LOCALE's native window is **256 bp**, not a kilobase. It is
`EncoderConfig.max_seq_len` in `benchmark/src/config.py`, it is what every
config under `benchmark/configs/` sets, and it is `augment_config.max_len` in
the paper's `configs/config.yaml`. A 54 kb pks island is therefore ~500 windows,
not one vector. `src/embed.py` tiles with `chunk_sequence` in stride mode at
`chunk_overlap = 150` (step 106), exactly as `DenseIndex._iter_chunks` does.

---

## Running it

```bash
uv sync                                     # no flash extra needed, no GPU needed
export NCBI_EMAIL=you@example.org           # NCBI asks for this; unset = harder throttling
export NCBI_API_KEY=...                     # optional, raises the rate limit

# 1. assemble the sequence sets (network only -- safe on a login node)
uv run python experiments/separability/data/fetch_elements.py

# 2. run the probe (needs the model; use a compute node)
srun -A <account>_g -C gpu -q interactive -t 60 -N 1 -n 1 --gpus-per-task=1 \
  uv run python experiments/separability/run_probe.py
```

Step 1 writes `data/sequences.parquet` (the CDS table), `data/manifest.yaml`
(every accession retrieved, with taxonomy and set sizes) and `data/cache/*.gb`
(raw GenBank). Re-running reads the cache; `--refetch` ignores it.

Of those, only `manifest.yaml` and `elements.yaml` are tracked: the repo root
ignores `*.parquet` and this directory ignores `cache/`, and both are
regenerated from the cache by the command above. `manifest.yaml` is the
provenance record — it lists every accession, per group, with taxonomy.

The repo root `.gitignore` also has a bare `data` rule that matches any
directory of that name at any depth, so `experiments/separability/.gitignore`
re-includes this one with `!data/`.

Step 2 writes per-element `*_curves.csv`, `*_summary.json`, capture/f_j plots,
and a top-level `summary.json` into `results/`. Only `summary.json` is tracked.

```bash
# 3. regenerate report.md's tables from results/summary.json (no model needed)
uv run python experiments/separability/make_tables.py
```

### On CPU

The benchmark harness requires CUDA, but `LOCALEEncoder` does not:

```bash
uv run python experiments/separability/run_probe.py --device cpu --batch-size 32
```

Slower, otherwise identical. Do not run this on a login node.

### Useful flags

| Flag | Meaning |
|---|---|
| `--element bla_ctxm` | one element only (repeatable) |
| `--noise 0.0 0.05 0.10` | noise rates on the query side |
| `--convention repo\|benchmark` | which mutation model (see below) |
| `--windows-per-gene 3` | read-level query windows sampled per carrier gene |
| `--tau 0.9 --alpha 0.01` | carrier capture required / confusable capture tolerated |
| `--checkpoint PATH` | local checkpoint instead of the Hub download |
| `--no-mmseqs` | skip the MMseqs2 baselines |

### Checkpoint

`--checkpoint` unset downloads the paper checkpoint (466 MB) from
`rsynk/locale` at the revision pinned in
`benchmark/src/config.py::PAPER_CHECKPOINT`, and caches it. Note that
`benchmark/configs/locale_config.yaml`, referenced by the top-level README, no
longer exists — the configs were reorganised into per-dataset directories, and
`PAPER_CHECKPOINT` is now the authoritative pin.

### MMseqs2

The benchmark expects `mmseqs` on `PATH`. If it is not there, `src/baselines.py`
falls back to `/pscratch/sd/r/rsynk/mmseqs/bin/mmseqs`; `MMSEQS_BIN` overrides
both. `--no-mmseqs` drops those baselines entirely.

### Noise conventions

Two exist in this project and they are not the same thing, so both are
implemented and every number says which it used:

- `repo` (default) — `lae.training.batcher.Augmenter.augment`, imported rather
  than reimplemented. This is what the checkpoint was trained against: exactly
  `round(len * rate)` edits, split 40/30/30 across substitution/insertion/
  deletion. Identity lands at `1 - rate` by construction.
- `benchmark` — the convention `locale-data/benchmark/mutate_queries.py` uses
  with `mutation-simulator`: SNP rate `r`, insertion and deletion rates `r/10`,
  indel length 1–2 bp. **Reimplemented in numpy**, not shelled out, because
  `mutation-simulator` is not on `PATH` here and, as its own docstring says, has
  no seed flag — which makes it unusable for a probe that has to be rerunnable.

They are not interchangeable at the same nominal `r`: `repo` gives identity
`1-r`, `benchmark` gives roughly `1-1.2r`. `run_probe.py` measures and reports
the realized identity either way.

---

## Tests

```bash
uv sync --extra dev
uv run python -m pytest experiments/separability/tests -q          # fast: sweep math
uv run python -m pytest experiments/separability/tests -q -m slow  # + encoder parity
```

`test_sweep_math.py` checks the capture/band/AUROC/f_j arithmetic against
hand-computed values. `test_embed_parity.py` checks that `src/embed.py` produces
the same vectors as both `DenseEncoder` and `DenseIndex._embed_queries` for a
fixed input; it loads a checkpoint, so it is marked `slow` and skips when none is
reachable. Run the slow ones on a compute node.

---

## Layout

```
experiments/separability/
  README.md              this file
  report.md              the deliverable: curves, f_j, verdicts
  run_probe.py           single entry point
  make_tables.py         results/summary.json -> the markdown tables in report.md
  data/
    elements.yaml        element definitions (hand-authored input)
    fetch_elements.py    NCBI harvest -> sequences.parquet + manifest.yaml
    manifest.yaml        provenance: every accession, with taxonomy (generated)
    sequences.parquet    the CDS table (generated)
    cache/               raw GenBank (gitignored)
  src/
    embed.py             LOCALE wrapper; mirrors dense_index.py, invents nothing
    noise.py             the two mutation conventions
    baselines.py         31-mers, MMseqs2 search, MMseqs2 catalogs
    sweep.py             radius sweep, band, AUROC, f_j
  tests/
  results/               CSV/JSON/PNG (gitignored except summary.json)
```
