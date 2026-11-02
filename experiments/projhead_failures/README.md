# Projection-head checkpoints on sra500v2: where the noisy queries fail (2026-10-27)

`characterize.py` scores every query of the three exact top-100 runs
(pc6t1x8j/21380 proj, qpms6u7j/4276 proj, 8vqiabk9/5859; all endwin50 +
minlen100) per rate and joins bundle attributes; output in `per_query.parquet`,
`characterize.log`, `pc6t1x8j_mut0.10_features.parquet`. Failure = AP < 0.5.

Findings (pc6t1x8j @10% unless noted; n = 500 queries):
- 162 failures, 152 of them noise-induced (fine at 0%); 26 complete misses
  (no relevant accession in the top-100 vectors; baseline 37).
- vs 8vqiabk9: 140 shared failures, 89 rescued, 22 newly failing.
- Dominant factor is read length: fail rate 84% < 100 bp, 48% 100-120,
  36% 120-140, 21% >= 140 bp (Spearman AP~qlen +0.39). Realized edit
  fraction, indel share, GC, entropy, homopolymers: no effect beyond length.
- Mechanism: an 80M-vector index has a background max cosine ~0.57 (90th
  pct 0.66). 10% noise lowers the true window's cosine by ~0.23 regardless
  of outcome; what differs is clean-query headroom above background
  (0.25 ok vs 0.16 fail), and headroom shrinks with read length (0.11 at
  < 100 bp, 0.27 at >= 140 bp). Short reads are a smaller fraction of a
  256 bp window, so they start closer to the floor.
- Secondary: many relevant accessions (> 10: 52% fail, metric effect) and
  lower alignment identity (< 0.95: 54% fail); '+' strand 37% vs '-' 28%
  (Fisher p = 0.045, borderline).
- Rescue over baseline comes from a lower background (0.57 vs 0.60) more
  than from a higher true score; the 22 newly failing are mostly 150 bp
  single-accession reads where the true score now sits just under a
  high-scoring impostor (best_irr 0.63-0.71).

## 128/64 index (2026-10-28): `characterize.py w128 pc6t1x8j` -> characterize_w128.log

pc6t1x8j with 128 bp windows at stride 64 (endwin30, minlen100; 133M
vectors, 136 GB, 1.65x the 256/106 index). AUPRC 0.978 / 0.954 / 0.727 vs
0.959 / 0.898 / 0.660 at 0 / 5 / 10% noise; failures at 10%: 121 vs 162;
complete misses 13 vs 26; at 0% only 3 queries fail.

- The length pattern inverted. Under 140 bp the fail rate at 10% fell from
  84/48/36% (<100, 100-120, 120-140) to 38/14/11%, mean AP delta +0.24 to
  +0.35, misses 14 -> 3: the clean true score of short reads rose 0.67 ->
  0.82 (<100 bp) and 0.76 -> 0.92 (100-120), the dilution fix working.
- Reads of 140 bp and more got WORSE: fail 21% -> 30%, mean AP delta -0.095,
  93 queries lost > 0.1 AP. Decomposition (n=291, 10% noise): clean true
  score 0.916 -> 0.967, but the noise drop grew 0.258 -> 0.279 so the noisy
  true score moved only 0.658 -> 0.687, while the background rose 0.609 ->
  0.650 on 85% of queries (a 150-170 bp noisy read finds closer random
  128 bp windows than random 256 bp ones). Headroom 0.051 -> 0.038. The
  worsened set is defined by background (0.597 -> 0.661) with a flat true
  score; SRR5723xxx (170 bp reads) and SRR9093/9094xxx (151 bp) dominate.
- Low 3-mer entropy is now the secondary factor on long reads (AP 0.55 at
  entropy < 4.5 vs 0.77 above 5.5; background 0.672 vs 0.618).
- Length no longer predicts AP overall (Spearman -0.06); the 13 reads with
  <= 10 edits (50-85 bp) still fail 85%.

Implication: a 150+ bp read is now LONGER than the window, embedded as one
vector against sub-span windows; the natural fix is query-side splitting
(embed the read's own 128 bp windows, union the hit lists as both_strands
does) and/or the alignment reranker absorbing the background rise. The
shortest reads need 64 bp windows or the same split on the index side.
