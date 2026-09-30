# Separability probe — report

One question, asked of the released LOCALE checkpoint (`8vqiabk9`, step 5859, the
revision pinned in `benchmark/src/config.py::PAPER_CHECKPOINT`):

> Is there a radius in LOCALE embedding space **wider** than the divergence
> among true carriers of a mobile element, and **narrower** than the distance to
> the nearest functionally different homolog — and is that window wider than
> identity clustering achieves, particularly under read-level noise?

---

## What the repo actually says (four corrections to the brief)

These were read out of the code, not assumed, and three of them contradict the
probe brief:

1. **The native window is 256 bp, not "roughly kilobase".**
   `EncoderConfig.max_seq_len` defaults to 256; every config under
   `benchmark/configs/` sets 256; the paper's own training config
   (`configs/config.yaml`) sets `augment_config.max_len: 256`. A 54 kb pks
   island is therefore ~500 windows, and *everything* in this probe is tiled at
   256 bp with `chunk_overlap: 150` (step 106) via the repo's own
   `chunk_sequence` in stride mode — the same path as
   `DenseIndex._iter_chunks`.

2. **Embeddings are 768-d, not 128-d.** `LOCALEEncoder.encode` mean-pools
   `model.bert_q(**tokens)[0]` — the DNABERT-2 backbone hidden states — and
   never touches the MoCo projection head. `TrainConfig.dim: 128` is the
   contrastive objective's projection width and is not what the benchmark
   indexes. Vectors are L2-normalised, so cosine = inner product and every
   "radius" here is a cosine similarity (higher = closer).

3. **`benchmark/configs/locale_config.yaml` does not exist.** The configs were
   reorganised into per-dataset directories (`benchmark/configs/sra50/`, etc.).
   The top-level README still references the old path. `PAPER_CHECKPOINT` is now
   the authoritative pin, and an unset `checkpoint_path` downloads it.

4. **flash-attn *is* installed in this environment**, contrary to the brief's
   expectation. The run log records
   `Successfully patched BertUnpadSelfAttention with flash_attn library (ALiBi enabled)`,
   so attention took the flash path, not the fallback. (A separate warning that
   Triton is unavailable comes from `lae/modeling/bert_layers.py` and is
   superseded by the flash patch.)

The embedding path is not reimplemented. `src/embed.py` calls the repo's
`LOCALEEncoder` and `chunk_sequence` directly, and
`tests/test_embed_parity.py` asserts bitwise-equal vectors (atol 1e-6) against
both `DenseEncoder` and `DenseIndex._embed_queries` for fixed inputs.

---

## Method

### Two granularities

| | query is | similarity to a reference gene |
|---|---|---|
| **gene** | one whole gene, all its windows | max over (query window, reference window) pairs |
| **window** | one 256 bp window — a simulated read | max over the reference gene's windows |

Both are **max over windows** (late-interaction), never a mean pool. Pooled
accession representation is an open problem in this codebase and this probe does
not need it solved; every gene- and element-level number here is a max and is
labelled as such.

### Two carrier scopes — and why the probe is wrong without them

This is the methodological crux, and getting it wrong silently destroys any
multi-gene element:

- **`same_gene`** — a query is scored only against carriers of its *own* gene. A
  `clbB` query is compared to other `clbB` sequences; `clbA` is masked out of
  both numerator and denominator. This answers *"is the radius wider than the
  divergence among true carriers?"* — strain-level and cross-genus divergence of
  one gene against the nearest different family.
- **`element`** — carriers are *accessions* (strains), captured when any window
  of any of their carrier genes lands inside the radius. This is the benchmark's
  own regroup-to-accession protocol, and it is the quantity `f_j` consumes. An
  accession holding both a carrier and a confusable gene counts as a **carrier**,
  not as contamination — retrieving a genuine pks carrier that also has
  yersiniabactin is not a false positive.

Without the split, the pks island's 19 `clb` genes cap "fraction of carrier genes
captured" near 1/19 for *every* method at *any* radius, because a `clbB` query
cannot and should not retrieve `clbA`. blaCTX-M hides the problem entirely
because it is a single gene family — which is precisely why a probe run on one
element would have missed it.

### Noise

Read-level noise on the **query** side only, at 0% / 5% / 10%; the reference set
stays clean. That is the read-against-catalog setting, and clean sequences are
where identity clustering is strongest — so a LOCALE win, if there is one, has to
appear under noise.

Mutation uses the repo's own path: `lae.training.batcher.Augmenter.augment`,
imported, not reimplemented. It is the augmentation the checkpoint was trained
against — exactly `round(len × rate)` edits split 40/30/30 across
substitution/insertion/deletion, so identity lands on `1 − rate` by
construction. Realized identity is measured per run and reported.

`src/noise.py` also implements the *benchmark's* convention
(`locale-data/benchmark/mutate_queries.py`: SNP rate `r`, insertion and deletion
rates `r/10`, indel length 1–2 bp) as a numpy reimplementation. It is a
reimplementation rather than a call to `mutation-simulator` because that tool is
not on `PATH` here and, as `mutate_queries.py`'s own docstring states, has no
seed flag — which makes it unusable for a probe that must be rerunnable. The two
conventions are not interchangeable at the same nominal `r`: `repo` yields
identity `1−r`, `benchmark` roughly `1−1.2r`.

### Baselines, run generously

Every baseline produces a score on the same [0,1] grid over the same units and
goes through the same sweep:

- **exact 31-mers** — canonical (strand-collapsed) k-mer containment
  `|K_q ∩ K_t| / |K_q|`. Containment rather than Jaccard because it is what a
  k-mer index answers and it does not penalise a short query against a long
  gene. Jaccard is also computed.
- **MMseqs2 `easy-search`** — alignment identity, run at `-s 7.5`, `-e 1e3`,
  `--max-seqs 20000` so a missing entry means MMseqs2 genuinely found nothing
  rather than that the search was truncated. The headline score is raw `fident`,
  **not** identity discounted by coverage, so a short high-identity local hit
  counts as a full match. Since "MMseqs2 matches LOCALE" is a NO-GO verdict,
  being generous to MMseqs2 is the conservative direction;
  `fident × qcov` is reported as the stricter reading.
- **MMseqs2 identity catalogs** at 95/90/80/70/50% — the clean reference set is
  clustered once (that is the catalog), then each possibly-noisy query is
  assigned to it by best alignment, and only if that alignment clears the
  threshold. A query that clears nothing is **unassigned and scores 0**, not
  dropped: that is the fragmentation failure mode and deleting it would hide
  exactly what the probe is looking for.
- **UHGP-style protein catalog** — UHGP-90 and UHGP-50 are MMseqs2 clusterings
  of predicted proteins at 90% and 50% amino-acid identity, so clustering the
  translated CDSs at those thresholds *is* the catalog protocol applied to these
  sequences. It answers the brief's "does an off-the-shelf gut gene catalog put
  all carriers in one cluster" question without downloading UHGP. Clean
  sequences only: frame-1 translation is meaningless once indels have shifted
  the frame, and running it under noise would manufacture a baseline collapse
  that is an artefact of this script rather than of the catalog.

A note on the clustering implementation: `mmseqs easy-cluster` and
`easy-linclust` **segfault** in this MMseqs2 build (`d45e0c4`) on nucleotide
input — `align2clust` reads past the end of the reverse-complement-doubled
database (`getDbKey: local id (548) >= db size (335)`) and dies. `easy-search`
is unaffected, so the catalogs are rebuilt from its all-vs-all alignments using
greedy set cover, which is the algorithm `mmseqs clust --cluster-mode 0` runs.
The identity and bidirectional-coverage (≥0.8 both ways) rules are MMseqs2's;
only the graph traversal is ours, and it is unit-tested.

### Reported statistics

- **capture curves** — carrier and confusable capture at each radius, as the
  distribution over queries (mean, p10/p50/p90, and confusable max), never a
  single arbitrary query. Every carrier is used as a query in turn.
- **AUROC** — carrier vs confusable similarity, threshold-free, with ties at
  0.5. The one number comparable across cosine, containment and alignment
  identity without choosing a radius for any of them.
- **band** — `[lo, hi]` where `hi` is the largest radius still holding mean
  carrier capture ≥ τ (0.90) and `lo` the smallest radius with mean confusable
  capture ≤ α (0.01). `margin = hi − lo`; negative means the bands interleave.
- **carrier capture at ≤α confusable capture** — the headline number, because it
  is what `f_j` is proportional to and it stays meaningful when the band is
  empty and the margin degenerates.
- **`f_j`** — see below.

### Three controls

- **same-genus** — the sweep rerun with carriers *and* confusables both
  restricted to one genus, inside the `same_gene` scope. Without it, a model
  could score well by reading genomic background (GC content, codon usage,
  k-mer style) and never look at the gene. It ran for all three elements
  (*Klebsiella*, *Escherichia*, *Klebsiella*), though for `cdt_abc` it
  necessarily excludes the typhoid-toxin confusable, which is defined by its
  genus.
- **near-duplicate fraction** — the share of carrier–carrier pairs above 0.99
  similarity on clean sequences. Exact duplicates were dropped at fetch time,
  but two accessions depositing the same allele one base apart both survive.
  They inflate every method's absolute carrier capture equally, so they do not
  bias the comparison — but the report states the size of the inflation.
- **label audit** — carrier–confusable pairs at cosine ≥ 0.99 on clean
  sequences. A carrier sitting on top of something labelled a confusable is far
  more likely a labelling error than a finding. This is the check that caught
  the CTX-M contamination described under compromises; it reports **zero pairs
  for all three elements** in the final run.

### Reproducibility

The probe was run twice end to end from the same data and seed. All **192 band
statistics agreed to within 1e-9**, so every number here except the same-genus
control (which was corrected between the two runs) is reproducible rather than a
draw from GPU or clustering nondeterminism.

---

## `f_j` — the conversion the downstream statistics consumes

The separability plot is suggestive; `f_j` is what a power calculation eats. The
model, stated plainly because it is doing real work:

- The element is present in a fraction `P_E` of individuals.
- Carriers are assumed **uniformly distributed over the distinct carrier
  sequences observed**. We have no population frequencies for strain variants,
  so every carrier stands for an equal slice of `P_E`. **This is the single
  biggest modelling assumption in the probe** and it is optimistic: real strain
  frequencies are heavy-tailed, so a feature that captures a random 40% of
  observed carriers will not generally capture 40% of carrying people.
- At radius `r`, a column defined by one query pools the fraction `c(r)` of
  carriers it captures, so `f_j(r) = P_E · c(r)`.
- Non-carriers are swept in when the radius reaches the confusables:
  `f_contam(r) = P_C · x(r)`, with `x(r)` the confusable capture and `P_C` the
  confusable's prevalence.
- `purity = f_j / (f_j + f_contam)`. A column with high `f_j` and low purity is
  worse than useless — it dilutes the very effect it is meant to detect.

### Prevalence assumptions — inputs, not findings

| element | `P_E` | `P_C` | basis |
|---|---|---|---|
| `pks_clb` | 0.20 | 0.35 | pks⁺ *E. coli* carriage in healthy adults is commonly quoted around 20% (isolate surveys of the Nougayrède/Putze era report the island in ~20–35% of commensal B2 *E. coli*); yersiniabactin is more common still. |
| `cdt_abc` | 0.10 | 0.30 | order-of-magnitude placeholder for cdt-bearing Enterobacteriaceae/*Campylobacter* carriage; the confusable pool (hlyA/stx/elt/typhoid-toxin cdtB) is broader. |
| `bla_ctxm` | 0.15 | 0.50 | ESBL (largely CTX-M) faecal carriage varies enormously by region, 15% is mid-range; blaTEM is near-ubiquitous in Enterobacteriaceae. |

**These figures were not verified against primary sources in this environment**
(no full-text access), and they are the weakest link in the probe. `f_j` scales
linearly in `P_E`, so a reader who prefers another number can rescale by eye;
the *ratios* between methods, which is what the verdict turns on, do not depend
on them at all.

---

## Data

All accessions are in [`data/manifest.yaml`](data/manifest.yaml). 2326 CDS
sequences, harvested from NCBI nuccore via Entrez.

| element | carriers | carrier genera | carrier genes | confusables | confusable genera | ref. windows |
|---|---|---|---|---|---|---|
| `pks_clb` | 550 (201 accessions) | 3 — *Citrobacter*, *Escherichia*, *Klebsiella* | 17 (clbA–clbS) | 789 | 7 | 47 971 |
| `cdt_abc` | 393 (237 accessions) | 5 — *Aggregatibacter*, *Campylobacter*, *Escherichia*, *Helicobacter*, *Shigella* | 3 (cdtA/B/C) | 206 | 2 | 4 073 |
| `bla_ctxm` | 117 (117 accessions) | 14 | 1 (CTX-M family) | 271 | 23 | 2 190 |

**Carriers span at least two genera for all three elements**, so the
cross-species pooling claim is testable in every case (the brief asked for this
to be stated explicitly). `pks_clb` is the weakest at 3 genera, and its
*Citrobacter* representation is thin.

Confusables: `pks_clb` — yersiniabactin NRPS-PKS machinery (irp1/irp2 =
HMWP1/HMWP2, ybtE/S/U) and enterobactin NRPS (entB/D/E/F). `cdt_abc` —
*Salmonella* typhoid-toxin cdtB/pltA/pltB (a genuine CdtB homolog in a
different holotoxin) plus hlyA/stx/elt/cnf1. `bla_ctxm` — blaTEM, blaSHV,
blaOXA.

### Data-quality diagnostics

| element | audit: carrier–confusable pairs ≥0.99 | near-duplicate carrier pairs (clean) | carrier genes per accession |
|---|---|---|---|
| `pks_clb` | 0 | 0.094 | median 2 of 17; 42 accessions have 1, 20 have ≥5, max 16 |
| `cdt_abc` | 0 | 0.025 | median 1 of 3; 47 accessions have all 3 |
| `bla_ctxm` | 0 | 0.152 | 1 of 1 (single-gene element) |

The audit finding zero suspect pairs is the check that caught the CTX-M
labelling error described under compromises; it is clean in this run. The
near-duplicate column is the absolute-number inflation to discount: ~9% of pks
and ~15% of blaCTX-M carrier pairs are near-identical alleles, which lifts every
method's carrier capture equally.

---

## Results

Cells below are **AUROC / carrier capture at ≤1% confusable capture**. The
second number is the headline: it is what `f_j` is proportional to, and it stays
meaningful when no band exists. Full tables, all α budgets and per-query
quantiles are in `results/*_curves.csv` and `results/summary.json`; curves are
plotted in `results/*.png`.

### At a glance

Read granularity (256 bp windows) at 10% query noise — the regime the claim
lives in. "best id." is the better of 31-mers and MMseqs2 search. The last two
columns hold confusable capture **equal** between LOCALE and the best identity
catalog, which is the only fair way to compare a curve against a fixed
operating point.

| element | scope | AUROC LOCALE | AUROC best id. | carrier@≤1% LOCALE | carrier@≤1% best id. | best catalog | LOCALE @ same confusable | LOCALE pools more |
|---|---|---|---|---|---|---|---|---|
| `pks_clb` | same_gene | **0.943** | 0.920 | 0.873 | 0.878 | 0.158 (@80%) | **0.227** | yes |
| `pks_clb` | element | **0.679** | 0.620 | 0.245 | 0.239 | 0.158 (@80%) | **0.227** | yes |
| `pks_clb` | element_ge4genes | **0.837** | 0.743 | 0.494 | 0.485 | 0.158 (@80%) | **0.227** | yes |
| `cdt_abc` | same_gene | **0.666** | 0.575 | 0.178 | 0.156 | 0.076 (@50%) | **0.086** | yes |
| `cdt_abc` | element | **0.551** | 0.544 | 0.103 | 0.089 | 0.076 (@50%) | **0.086** | yes |
| `cdt_abc` | element_ge2genes | **0.650** | 0.569 | 0.158 | 0.139 | 0.076 (@50%) | **0.086** | yes |
| `bla_ctxm` | same_gene / element | **0.907** | 0.733 | 0.462 | 0.471 | **0.611** (@70%) | 0.404 | **no** |

LOCALE has the better AUROC in every row. It never has a materially better
operating point than MMseqs2 *search*, and it loses outright to MMseqs2
*clustering* on `bla_ctxm`. Both numbers are reproducible: two independent runs
of the probe agreed on all 192 band statistics to within 1e-9.

### pks / clb island

`same_gene` scope — divergence among carriers of one clb gene vs the nearest
NRPS-PKS homolog:

| granularity | method | 0% | 5% | 10% |
|---|---|---|---|---|
| gene | **LOCALE** | 0.975 / 0.965 | 0.977 / 0.965 | 0.977 / 0.965 |
| gene | 31-mers | 0.980 / 0.972 | 0.978 / 0.965 | 0.973 / 0.928 |
| gene | MMseqs2 | 0.980 / 0.974 | 0.980 / 0.973 | **0.979 / 0.972** |
| window | **LOCALE** | **0.941** / 0.875 | **0.943** / 0.876 | **0.943** / 0.873 |
| window | 31-mers | 0.923 / 0.883 | 0.917 / 0.865 | 0.765 / 0.525 |
| window | MMseqs2 | 0.923 / 0.886 | 0.925 / 0.887 | 0.920 / **0.878** |

A usable window **does exist** here, for LOCALE and for MMseqs2. At gene
granularity and 10% noise LOCALE's band is `[0.655, 0.830]`, margin **+0.175**,
with 96.5% carrier capture at ≤1% confusable capture. But MMseqs2's band margin
is **+0.890** — five times wider — because its confusable capture is *exactly
zero* at every radius above 0: nucleotide alignment finds nothing at all between
clb and ybt/ent genes.

`element_ge4genes` scope (carrier accessions holding ≥4 of 17 clb genes, 20
accessions — the unrestricted `element` scope is reported in the JSON as a lower
bound and is confounded by record incompleteness):

| granularity | method | 0% | 5% | 10% |
|---|---|---|---|---|
| gene | **LOCALE** | 0.767 / 0.532 | 0.794 / 0.533 | 0.798 / 0.534 |
| gene | MMseqs2 | 0.788 / 0.569 | 0.782 / 0.567 | 0.780 / **0.561** |
| window | **LOCALE** | **0.831** / 0.489 | **0.831** / 0.496 | **0.837** / 0.494 |
| window | 31-mers | 0.744 / 0.487 | 0.740 / 0.475 | 0.651 / 0.284 |
| window | MMseqs2 | 0.745 / 0.490 | 0.745 / 0.490 | 0.743 / 0.485 |

Identity catalogs, window granularity, counted in carrier accessions:

| catalog | noise | assigned | carrier capture | confusable capture | carrier clusters | LOCALE @ same confusable |
|---|---|---|---|---|---|---|
| mmseqs @ 95% | 0% | 1.00 | 0.173 | 0.000 | 61 | **0.235** |
| mmseqs @ 95% | 10% | **0.02** | 0.003 | 0.000 | 61 | **0.227** |
| mmseqs @ 90% | 10% | 0.69 | 0.106 | 0.000 | 60 | **0.227** |
| mmseqs @ 80% | 10% | 0.99 | 0.158 | 0.000 | 60 | **0.227** |
| mmseqs @ 50% | 10% | 0.99 | 0.158 | 0.000 | 60 | **0.227** |
| UHGP-style protein @ 90% AA | clean | — | 66/550 = 0.12 | 0 confusables | 66 | — |
| UHGP-style protein @ 50% AA | clean | — | 66/550 = 0.12 | 0 confusables | 66 | — |

Two things here. First, a 95%-identity catalog **rejects 98% of 10%-noise reads
outright** — they align to nothing above threshold and the feature is simply
missing for those people. Second, the catalog fragments into ~60 carrier
clusters and **loosening the threshold from 95% to 50% changes nothing**
(0.158 either way; the protein catalog is identical at 90% and 50% AA). That is
not a tuning failure, it is structural: the 17 clb genes span 513 bp (clbS) to
9621 bp (clbB), and only 17 of 136 gene pairs have a length ratio that even
permits 80% mutual coverage. **No identity threshold can pool different genes of
one element**, because they are not similar to each other.

### cdtABC

`same_gene` scope:

| granularity | method | 0% | 5% | 10% |
|---|---|---|---|---|
| gene | **LOCALE** | **0.760** / 0.227 | **0.741** / 0.207 | **0.730** / **0.206** |
| gene | 31-mers | 0.576 / 0.159 | 0.572 / 0.147 | 0.561 / 0.115 |
| gene | MMseqs2 | 0.624 / **0.243** | 0.595 / 0.192 | 0.585 / 0.174 |
| window | **LOCALE** | **0.697** / **0.202** | **0.682** / **0.189** | **0.666** / **0.178** |
| window | 31-mers | 0.573 / 0.153 | 0.565 / 0.132 | 0.538 / 0.073 |
| window | MMseqs2 | 0.592 / 0.187 | 0.580 / 0.166 | 0.575 / 0.156 |

**Every margin is negative, for every method, scope, granularity and noise
level.** No radius separates cdt carriers from the confusables. LOCALE is the
best method at almost every cell — and it is nowhere near sufficient.

The cause is biological and not fixable by a better embedding: *Salmonella*'s
typhoid-toxin cdtB **is** a CdtB, and cross-genus carrier cdtB (*Campylobacter*
vs *Escherichia* vs *Helicobacter*) is more divergent from itself than from that
confusable. The bands interleave by construction. The protein catalog shows the
same thing from the other side: at 90% AA identity, 83 clusters with the largest
holding 29/393 carriers (7%); loosening to 50% AA pools 125/393 (32%) **but
admits 31 confusables**. That is exactly the "loose enough to pool admits
non-carriers" failure the brief predicted.

### blaCTX-M

`same_gene` = `element` here (single-gene element, 117 accessions each with one
CTX-M):

| granularity | method | 0% | 5% | 10% |
|---|---|---|---|---|
| gene | **LOCALE** | 0.967 / 0.650 | **0.948** / 0.619 | **0.934** / 0.541 |
| gene | 31-mers | 0.729 / 0.461 | 0.724 / 0.452 | 0.700 / 0.373 |
| gene | MMseqs2 | **0.986** / **0.983** | 0.879 / **0.765** | 0.794 / **0.593** |

LOCALE's ranking is far more noise-stable than either baseline (AUROC 0.967 →
0.934 across 0–10%, against MMseqs2's 0.986 → 0.794 and k-mers' 0.729 → 0.700).
This is the clearest demonstration in the probe of the property LOCALE was built
for.

It does not survive contact with the catalog:

| catalog | noise | assigned | carrier capture | confusable capture | carrier clusters | LOCALE @ same confusable | LOCALE wins |
|---|---|---|---|---|---|---|---|
| mmseqs @ 95% | 10% | 0.00 | 0.000 | 0.000 | 9 | 0.290 | yes |
| mmseqs @ 90% | 10% | 0.90 | 0.252 | 0.004 | 7 | 0.452 | yes |
| mmseqs @ 80% | 10% | 1.00 | 0.446 | 0.004 | 6 | 0.456 | yes |
| **mmseqs @ 70%** | **10%** | **1.00** | **0.630** | **0.004** | **3** | 0.456 | **no** |
| **mmseqs @ 70%** | **0%** | 1.00 | **0.662** | 0.004 | 3 | 0.470 | **no** |
| UHGP-style protein @ 50% AA | clean | — | 89/117 = **0.76** | **0** | 8 | — | **no** |

A 70%-identity nucleotide catalog pools 63% of CTX-M carriers into **3
clusters** at 10% noise with 0.4% confusable capture, and is essentially
noise-immune (0.662 / 0.619 / 0.630 across 0–10%) because the catalog is built
on clean references and a 90%-identity read still clears a 70% threshold
comfortably. A UHGP-style protein catalog at 50% AA does better still: 76% of
carriers in 8 clusters, zero confusables. LOCALE at matched confusable capture
reaches 0.456.

### Same-genus control

Carriers **and** confusables both restricted to one genus, in the `same_gene`
scope, so a model cannot score by reading genomic background — GC content, codon
usage, k-mer style — instead of the gene. Window granularity, AUROC / carrier
capture at ≤1% confusable:

| element | genus (carriers/confusables) | method | 0% | 5% | 10% |
|---|---|---|---|---|---|
| `pks_clb` | *Klebsiella* (200/436) | **LOCALE** | **0.946** / 0.866 | **0.943** / 0.865 | **0.947** / 0.861 |
| | | 31-mers | 0.916 / 0.870 | 0.910 / 0.857 | 0.759 / 0.533 |
| | | MMseqs2 | 0.916 / **0.871** | 0.917 / **0.872** | 0.912 / **0.867** |
| `cdt_abc` | *Escherichia* (133/115) | **LOCALE** | **0.755** / 0.462 | **0.755** / 0.460 | **0.744** / 0.447 |
| | | 31-mers | 0.706 / 0.413 | 0.655 / 0.302 | 0.576 / 0.140 |
| | | MMseqs2 | 0.745 / **0.486** | 0.735 / **0.468** | 0.714 / 0.425 |
| `bla_ctxm` | *Klebsiella* (44/139) | **LOCALE** | **0.920** / 0.577 | **0.888** / 0.480 | **0.877** / 0.466 |
| | | 31-mers | 0.730 / 0.460 | 0.726 / 0.446 | 0.636 / 0.246 |
| | | MMseqs2 | 0.883 / **0.765** | 0.775 / **0.550** | 0.739 / **0.478** |

**The control passes.** For `pks_clb` the within-genus numbers are
indistinguishable from the cross-genus ones (LOCALE 0.947 / 0.861 vs 0.943 /
0.873), and LOCALE keeps its AUROC advantage over both baselines inside a single
genus at every noise level. LOCALE's edge is therefore a property of the gene
comparison, not an artefact of carriers and confusables being drawn from
different taxa. Same conclusion for `bla_ctxm` (0.877 vs 0.739 at 10% noise).

**For `cdt_abc` the control changes the diagnosis.** Restricted to
*Escherichia*, carrier capture jumps from 0.178 to 0.447 and AUROC from 0.666 to
0.744 — cdt is substantially separable *within* a genus and only fails *across*
genera. Since the brief's requirement is explicitly pooling "across strain-level
divergence **AND** across host genera", cdt fails on the cross-genus clause
specifically, not on the confusable boundary in general.

Two caveats on that reading, both pushing the same way: the *Escherichia*
confusable set is the other-toxin group (hlyA/stx/elt/cnf1) and **excludes the
Salmonella typhoid-toxin cdtB by construction**, since that confusable is
defined by its genus. So the within-genus numbers are better both because
carriers are less divergent and because the hardest confusable is absent. A
same-genus control against the typhoid cdtB is impossible in this design.

### `f_j`

Implied carrier frequency of the pooled feature column, at the radius that holds
confusable capture ≤1%, under the prevalence assumptions above:

| element | LOCALE | 31-mers | MMseqs2 search | best identity catalog |
|---|---|---|---|---|
| `pks_clb` (window, ge4genes, 10%) | **0.099** | 0.057 | 0.097 | 0.032 |
| `cdt_abc` (window, ge2genes, 10%) | **0.016** | 0.006 | 0.014 | 0.008 |
| `bla_ctxm` (gene, 10%) | 0.081 | 0.056 | 0.089 | **0.095** |

---

## Verdicts

### `pks_clb` — **MARGINAL**

A radius window genuinely exists at the `same_gene` level: at 10% noise LOCALE
holds 96.5% carrier capture with ≤1% confusable capture, band margin +0.175, and
it is the most noise-stable method in the probe (window AUROC 0.941 → 0.943
across 0–10% noise, while 31-mers collapse 0.923 → 0.765). At read granularity
LOCALE has the best AUROC at every noise level and every scope. So the first
half of the GO condition is met.

The second half is not. MMseqs2 matches LOCALE's operating point almost exactly
(0.972 vs 0.965 carrier capture at gene level; 0.878 vs 0.873 at window level)
and has a band **five times wider**, because nucleotide alignment returns
*nothing* for clb-vs-ybt while cosine returns a graded score. Against identity
*clustering* LOCALE does win — 0.227 vs 0.158 carrier capture at matched
confusable capture, a 1.4× improvement in `f_j` — but that win is against
thresholding, not against alignment: MMseqs2 used as a continuous score gets
0.485 where LOCALE gets 0.494.

Marginal, and the honest reading is that the margin is a systems argument rather
than a geometry one: LOCALE's advantage over MMseqs2 here is ~1% carrier capture
and better noise-stability, which is not what decides a study's power. What
would decide it is that embeddings are indexable at SRA scale — which this probe
did not measure and was not asked to.

### `cdt_abc` — **NO-GO**

The bands interleave. Every margin is negative for every method at every scope,
granularity and noise level; the best AUROC anywhere is LOCALE's 0.760, and
`f_j` lands at 0.012–0.023 — a 1–2% feature frequency, which has no power for
any realistic cohort. No threshold works, which is the brief's first NO-GO
clause.

This is a property of the biology, not of the embedding: the *Salmonella*
typhoid-toxin cdtB is a real CdtB homolog and sits *inside* the divergence band
of the cross-genus carrier set. An element whose confusable is the same gene in
a different holotoxin cannot be separated by sequence similarity at any radius,
and pooling carriers loosely enough to capture *Campylobacter* and *Escherichia*
cdt together necessarily admits it — the protein catalog shows the same
trade-off (32% carrier pooling at 50% AA, but 31 confusables admitted).

The same-genus control locates the failure precisely. Restricted to
*Escherichia*, carrier capture rises from 0.178 to 0.447 — cdt is substantially
separable **within** a genus. What fails is the **cross-genus** half of the
requirement, and it fails because *Campylobacter*, *Escherichia* and
*Helicobacter* cdtB are further apart from each other than a carrier is from
Salmonella's typhoid cdtB. So a single pooled cdt feature is unavailable, but
*per-genus* cdt features would be viable — at the cost of splitting `f_j` across
several columns, which is the fragmentation the study was trying to avoid. (Note
the within-genus figure is flattered: the typhoid cdtB is excluded from an
*Escherichia*-restricted confusable set by construction.)

### `bla_ctxm` — **NO-GO**

The brief's second NO-GO clause, met explicitly: MMseqs2 at a specific identity
threshold beats LOCALE across the board. A 70%-identity catalog pools 63% of
carriers into 3 clusters at 10% noise with 0.4% confusable capture and is
noise-immune; the UHGP-style protein catalog at 50% AA reaches 76% in 8 clusters
with zero confusables. LOCALE at matched confusable capture reaches 0.456.

LOCALE's ranking *is* markedly more noise-robust (AUROC 0.934 vs 0.794 at 10%
noise) and this is the cleanest such demonstration in the probe — it survives
the same-genus control. But a better ranking that yields a worse operating point
does not help the downstream statistics, and `f_j` is 0.081 for LOCALE against
0.095 for the best catalog. blaCTX-M is the case identity clustering was designed
for: one ~876 bp gene family with a sharp boundary, where TEM/SHV/OXA sit far
below 70% nucleotide identity.

### Overall — **NO-GO**, with `pks_clb` marginal

Two elements fail outright and the motivating case is marginal. The proposed
approach should not proceed on the strength of LOCALE's embedding geometry.

Three findings matter more than the verdict itself:

1. **Multi-gene element pooling is an annotation problem, not a radius
   problem.** The 17 clb genes span 513–9621 bp; only 17/136 pairs can meet 80%
   mutual coverage. Identity catalogs fragment into ~60 clusters *identically* at
   95% and 50% identity — the coverage constraint binds, not the threshold. No
   embedding radius fixes this either, and it would be wrong if it did: LOCALE is
   trained for local alignment, so clbA and clbB *should* be far apart. A feature
   meaning "carries the pks island" has to be built by mapping genes to elements
   and OR-ing them, after which the radius question applies only *within* a gene.
   This is good news for the downstream study — the hard part is a catalog, not a
   model — but it is not what the probe was asked to find.

2. **Cosine has no "no match" value, and that costs it the exclusion half of
   the question.** MMseqs2's confusable capture is exactly 0.000 at every radius
   above zero for pks, giving an essentially unbounded band. LOCALE assigns
   remote homologs graded similarity, so its confusable curve decays smoothly and
   its band is narrower even when its carrier capture is equal. For "exclude the
   functionally different homolog", returning nothing for non-homologs is exactly
   the right behaviour, and nucleotide alignment has it for free.

3. **Where LOCALE does lead is narrow but real:** noise-stable *ranking* (AUROC
   flat across 0–10% noise where k-mers collapse), short 256 bp read queries
   (its AUROC advantage is largest at window granularity), and beating hard
   clustering, which any fixed threshold condemns to fragmentation. If there is a
   case for LOCALE in this application it rests on those plus indexing at scale,
   not on a wider separability window.

### What would change the verdict

- A real metagenomic carrier set instead of GenBank contigs, so element-level
  pooling can be measured rather than bounded (see compromise 2).
- Protein-space or remote-homology baselines (translated search, HMM profiles),
  which would likely *strengthen* the identity baselines further rather than
  weaken them.
- Testing `f_j` against real strain frequencies instead of the uniform
  assumption, which is optimistic for every method here.

---

## Data-sourcing compromises and how they could bias the result

### 1. NCBI annotation text *is* the ground truth here

Carrier and confusable membership is decided by regex over `/gene` and
`/product` on CDS features pulled from GenBank. That makes the label quality a
function of submitter annotation, and it went wrong three times during
construction. All three are fixed; they are listed because the *class* of error
is what a reader should distrust, not because these specific ones survive.

| what happened | effect | direction of bias |
|---|---|---|
| Anchored patterns (`^cdtB$`) were tested against the concatenation `"gene product"`, where `$` can never anchor. Every symbol pattern matched nothing; only free-text patterns worked. | The typhoid-toxin confusable set came back **empty** — the single hardest confusable in the probe. | Would have made `cdt_abc` look far easier than it is. |
| The CTX-M carrier patterns included `'^bla$'`, which matches the generic `/gene` value `bla` with product "class A beta-lactamase" — i.e. *any* class A enzyme. | Four TEM-like sequences sat inside the CTX-M **carrier** set at cosine >0.99 from real TEMs. | Destroyed the confusable boundary; made every method look worse, and would have produced a spurious NO-GO. |
| `astA` was used as a toxin pattern, but the symbol denotes both the EAST1 enterotoxin and arginine N-succinyltransferase. | 21 copies of a housekeeping metabolic enzyme sat in the "secreted toxin" confusable set. | Padded confusables with trivially separable cases — flatters every method. |

Two guards now exist against recurrence: `fetch_elements.py` prints the top
matched products per group (a generic annotation in a single-family group is
visible immediately), and `run_probe.py` runs a permanent audit flagging
carrier–confusable pairs at cosine ≥ 0.99, which is what surfaced the second
row above. A non-zero audit count should be read as "audit the manifest".

The `ybt` confusable group was also narrowed to the biosynthetic NRPS-PKS
machinery (`irp1`/`irp2` = HMWP1/HMWP2, `ybtE/S/U`), dropping the `ybtP/Q/X`
transporters and the `FyuA` receptor. Those are in the locus but are not
functionally analogous to a clb megasynthase, and including them padded the
confusable set with easy cases. This makes the test **harder**.

### 2. GenBank records are not carriers

The biggest structural compromise. A carrier "individual" is modelled as a
GenBank accession, but accessions are contigs, not genomes. Even after adding
searches that require multiple element genes in one record
(`clbB AND clbN AND 20000:80000[SLEN]`, `cdtA AND cdtB AND cdtC`), the median
pks carrier accession deposits only **2 of 17** clb genes. An accession whose
only clb gene is `clbK` cannot be retrieved by a `clbB` query, and no method
should be expected to do it.

This is why the unrestricted `element` scope is reported as a **lower bound**
and a restricted scope (`element_ge<N>genes`, carrier accessions holding at
least a quarter of the element's genes) is reported beside it. The restriction
biases toward strains that happen to have been sequenced as long contigs, which
may correlate with being reference strains — so the restricted numbers are
optimistic about record quality while the unrestricted ones are pessimistic
about it. The truth for a real metagenome, which would contain the whole island,
is closer to the restricted figure.

### 3. Capping prefers gene-rich accessions

Sequences are capped at 90 accessions per (group, genus) and a 1500-sequence
budget per group, applied to **whole accessions**, richest in element genes
first. Capping individual sequences would have been worse — a half-kept island
is indistinguishable from a strain that only has half the island — but
preferring rich accessions trades allelic diversity for element completeness.

### 4. Length floor

`MIN_CDS_LEN` is 256 bp, one full model window. At the earlier 150 bp threshold,
contig-edge truncations as short as 162 bp were entering the sets and standing
in for whole genes on a pad-heavy embedding.

### 5. No UHGG/UHGP download

The gut-catalog question is answered by clustering translated CDSs with MMseqs2
at 90% and 50% amino-acid identity, which is how UHGP-90 and UHGP-50 are
themselves built, rather than by downloading UHGP. This tests the *protocol* on
these sequences; it does not test UHGP's actual cluster assignments, which were
built over a different (gut-metagenome-derived) sequence universe.

### 6. Prevalence figures are unverified

See the `f_j` section. The figures are the commonly quoted ones and were not
checked against primary sources in this environment. `f_j` scales linearly in
`P_E`, and the *ratios between methods* — which is what the verdict turns on —
do not depend on them at all.

### 7. Confusable taxonomic breadth is uneven

`bla_ctxm` confusables span 21/4/8 genera (TEM/SHV/OXA), but both `cdt_abc`
confusable groups are single-genus by construction: the typhoid-toxin `cdtB` is
a confusable *because* it is Salmonella's, and the other-toxin group is
Escherichia. For that element a same-genus control against the typhoid
confusable is therefore impossible, and a genomic-background shortcut cannot be
fully excluded.

---

## Which attention path, and on what hardware

- **flash-attn 2.7.4.post1, installed and active.** The run log records
  `Successfully patched BertUnpadSelfAttention with flash_attn library (ALiBi
  enabled)`, so attention took the **flash** path, not the native fallback the
  brief expected. A separate `Unable to import Triton` warning from
  `lae/modeling/bert_layers.py` is emitted before the flash patch is applied and
  is superseded by it.
- torch 2.6.0+cu124, one **NVIDIA A100-SXM4-80GB**, one node
  (`srun -A m5408_g -C gpu -q interactive`). No login-node compute.
- Checkpoint `8vqiabk9/5859` — the paper checkpoint, downloaded from the Hub at
  the revision pinned in `PAPER_CHECKPOINT`.
- Window 256 bp, chunk overlap 150 bp, τ = 0.90, α = 0.01, 3 read windows
  sampled per carrier gene, seed 0.
- MMseqs2 `d45e0c4` from `/pscratch/sd/r/rsynk/mmseqs/bin`.
- Wall clock: 290 s (`pks_clb`), 84 s (`cdt_abc`), 58 s (`bla_ctxm`).

## Reproducing

See [`README.md`](README.md). Runtime and settings for the run these numbers come
from are in `results/summary.json` under `runtime` and `settings`; every table
above is regenerated by `make_tables.py` from that file.
