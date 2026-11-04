# Smith-Waterman audit of the sra500 top-100 vector hits (2026-11-02)

Question: are the non-credited top-k hits real homologs the accession-level
benchmark fails to credit (identity mass at 75-95%), or junk (almost nothing
survives a query-coverage filter)?

`sw_audit.py` maps every saved hit (vector_id) to its contig window via the
row->contig sidecar (window_recall.py), fetches the Logan contig, and locally
aligns the mutated query (both strands; blastn-like scores 2/-3/-5/-2) to the
window, to the window + query-length flank ("locus"), and to a decoy locus of
another query (null). `plot_sw_audit.py` draws the histograms.
Outputs: `sw_audit_<label>.parquet` (one row per rate x query x rank, 150k
rows), `sw_audit_<label>.png`, `sw_audit_compare.png`, `run_all.log`.

## Result (pc6t1x8j w128; the w256 and 8vqiabk9-w128 indexes look the same)

Other-accession hits (85-88 of every 100): fraction passing coverage >= 50%
and fraction passing with identity >= 75%

| rate | hits | pass cov | pass & id>=75% | null pass (decoy) |
|---|---|---|---|---|
| 0% | 42,644 | 15.6% | 5.8% | 1.4% |
| 5% | 43,478 | 11.3% | 3.6% | 0.7% |
| 10% | 44,243 | 7.0% | 1.5% | 0.4% |

Identity of the passers peaks at 65-75% with ~15 gaps per alignment (null
median 0.69), i.e. the coverage survivors are mostly low-identity gappy
alignments of low-entropy queries (3-mer entropy 4.4 vs 5.1 for the failers),
not homologs. The 75-95% band holds 1.5-5.8% of other-accession hits; the
>=95% bin at 0% noise (567 hits) is near-identical sequence in other
accessions (related-accession pitfall), gone at 10%.

Ground-truth-accession hits: only 48-60% land on the aligned contig and
51-66% pass coverage; the rest are other windows/contigs of the right
accession, so accession-level credit is itself loose.

Verdict: junk. The top-100 is ~85% hits with no alignable similarity to the
query; the benchmark is not under-crediting homologs in any meaningful amount.
An alignment reranker would remove these, not recover credit.

## Hub / anisotropy check (2026-11-02, same day)

123,492 junk hits (qcov < 50%) land on 114,968 distinct windows; 181 windows
recur in 2 queries, 1 in 3, none in >= 5 (uniform sampling of 133M would
already give ~57 repeats). Per rate 99.8-99.9% of junk hits are windows seen
by exactly one query. Index is near-isotropic: |mean vector| 0.045, random
cosine mean 0.002 / sd 0.092; junk windows' cosine to the mean direction
0.094 vs 0.045 random (mild). Junk windows are not short (2% from contigs
< 128 bp, same as alignable hits), not end windows (12%), and not
low-complexity (3-mer entropy 5.03 vs 5.06 random; dust 2.09 vs 2.03;
`window_complexity.py` / `.parquet`). So the junk is query-specific
near-random background, not hubs. Extreme-value note: a Gaussian tail with
the measured mean/sd predicts the rank-100 background score at 0.44, the
observed junk sits at 0.54-0.61, i.e. the per-query tail is heavier than
the global score distribution suggests.

## Dilution vs mutation (2026-11-02, `dilution_vs_mutation.py`)

All 2,604 ground-truth (query, contig) pairs, pc6t1x8j project=True, 128/64
index settings, both strands (max). Mean cosine:

| rate | mut vs clean read (mutation only) | clean read vs best 128 window (dilution only) | mut read vs exact span | mut read vs best window (index score) |
|---|---|---|---|---|
| 0% | 1.000 | 0.871 | 0.976 | 0.871 |
| 5% | 0.841 | 0.871 | 0.815 | 0.730 |
| 10% | 0.706 | 0.871 | 0.679 | 0.615 |

At 10% the exact aligned span scores 0.68, not 0.85: mutation loss is ~0.29
of the 0.385 drop, dilution ~0.13 (near-additive). Sub-span scoring would
recover at most +0.06 at 10%; the encoder's robustness to 10% edits is the
binding constraint. Dilution cost is length-dependent (reads < 100 bp 0.82,
128-200 bp 0.90-0.91, > 200 bp 0.70); mutation cost is not (~0.70 at every
length).

**Correction (same day):** the cosine attribution above must not be read as
"sub-span scoring is a small lever". Joining each truth pair to its query's
junk tail from the SW audit: the index score beats the query's best junk hit
for 67% of truth pairs at 10% (margin 0.051); the centred 128 window 76%; the
exact span 88% (per query 0.72 / 0.79 / 0.85; top-10 proxy 0.79 / 0.88 /
0.95). That matches experiments/dilution_probe (2026-10-14, 8vqiabk9, W=256
base): AUPRC at 10% w128 stride 0.680, w128 centred 0.813, exact span 0.846.
Mutation robustness sets the absolute true score; granularity sets the margin.

## Junk-tail stability across mutation rates (2026-11-04, `junk_stability.py`)

Same query, same index (pc6t1x8j w128), overlap of the non-relevant top-100
between rates (|a & b| / min): 0% vs 5% mean 0.19, 0% vs 10% 0.06 (23% of
queries share nothing), 5% vs 10% 0.035. Top-20 junk 0% vs 10%: 0.08. Of the
0% top-10 junk, 17% is still anywhere in the 10% top-100. The junk tail is a
fresh draw from the near-threshold sea for every realization of the query, not
a property of the sequence region: mining negatives with the clean window (or
any other draw) does not give the negatives the trainer's mutated read sees.
