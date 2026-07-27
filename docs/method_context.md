# Method context (from the LOCALE paper)

Reference for the backbone-swap experiment. These are the values **as written in
the paper** — treat them as the intended recipe, but if the repo's config/code
disagrees, follow the code and flag the discrepancy. The whole experiment depends
on holding this recipe FIXED and varying only the encoder backbone.

---

## Encoder head (§3.3) — the interface the new backbone must match

- Current backbone: **DNABERT-2**, all parameters fine-tuned (not frozen).
- BPE tokenizer, vocab size 4096; produces token-level reps `h_t ∈ R^D`.
- **Head (must stay identical across backbones):** mean-pool token reps over
  non-padding positions → **L2-normalize** onto the unit sphere.
- DNABERT-2 specifics: 117M params, embedding dim **D = 768**, base model
  originally trained on 700 bp.

**Swap requirement:** the substitute backbone must expose token-level
representations, then feed the SAME mean-pool + L2-norm head, so the InfoNCE loss
and the retrieval/eval pipeline are byte-for-byte unchanged. Only the encoder
module changes. If the new backbone has a different hidden dim, the head still
works (mean-pool + L2-norm is dim-agnostic); just confirm downstream index/eval
doesn't hardcode 768.

---

## Training objective (§3.1) — DO NOT CHANGE

- InfoNCE with in-batch negatives.
- Temperature **τ = 0.05**.

---

## Data augmentation (§3.2) — the recipe under test

Positive pairs = two overlapping crops of a parent sequence, one crop corrupted
with mutations.

**Cropping** (hold at **containment**, the baseline, for this experiment):
- Two strategies exist (containment, overlap); baseline uses containment.
- Crop lengths uniform in **[31, 256]** bp.
- Smaller sequence must cover ≥ 40% of the longer.

**Mutation injection** (this is the axis we ablate):
- Sample target identity ρ ∈ [0, ρ_max] from a **Beta distribution**
  parameterized by **mean + std** (Wick/Badread style), not α/β.
- Given crop length L and sampled ρ: apply `n = floor(L * (1 - ρ))` mutations,
  uniformly at random without replacement within the aligned region.
- Each mutation is substitution / insertion / deletion with prob **0.4 / 0.3 / 0.3**.
  - Substitution: uniform among the 3 non-matching bases.
  - Insertion: uniformly random base before the sampled position.
- Mutations applied to **only one crop** in each pair.

**The augmentation ladder (the 4 runs to reproduce):** none / light / medium / heavy.

| Strength | Mean identity % | Std % | Max % |
|----------|----------------|-------|-------|
| none     | (no mutation)  | —     | —     |
| light    | 95.0           | 2.5   | 99.0  |
| medium   | 90.0           | 6     | 98.0  |
| heavy    | 80.0           | 6     | 88.0  |

Heavy is the final-model setting.

---

## Training setup (§3.4) — hold FIXED across backbones

- Corpus: **Logan** contigs, **50-accession** training slice (disjoint from eval).
  6M training pairs. (Reference-genome corpus exists too but Logan is the default.)
- Optimizer: **AdamW**, β1=0.9, β2=0.999, weight decay **1e-2**.
- Peak LR **6e-5**, **460-step linear warmup** → cosine decay.
- Per-device batch 64; effective batch **1024**.
- **5859 steps**, ≈ 1 hr wall-clock (~16 A100-hrs) per run.

Do NOT re-tune these per backbone unless a run diverges — varying them would
confound the recipe-vs-backbone comparison. If a new backbone needs a different
LR to train at all, note it explicitly as a deviation.

---

## Indexing & search (§3.5) — used by eval, unchanged

- Embed every sequence; sequences over the context window are split into sliding
  windows with **150 bp overlap**, each window embedded independently.
- Retrieve top-**m = 10** nearest sequence embeddings; aggregate to accession by
  **max** inner-product similarity; rank accessions by that score.

---

## Evaluation (§4.1) — the target metric

- 50-accession benchmark.
- Eval mutation injected at **0% / 5% / 10%** (query identities 100/95/90%), same
  sub/ins/del 0.4/0.3/0.3 procedure as training.
- Metric: **Average Recall@Rq** (Rq = |relevant accession set|), plus AUPRC.

---

## The result to reproduce (Table 3, augmentation ladder)

DNABERT-2, containment cropping, Logan data — Recall@Rq at 0 / 5 / 10% eval mutation:

| Training mutation | 0%   | 5%   | 10%  |
|-------------------|------|------|------|
| none              | 79.6 | 62.8 | 27.7 |
| light             | 81.1 | 74.6 | 57.8 |
| medium            | 79.7 | 71.4 | 60.0 |
| **heavy** (baseline) | 79.9 | 74.1 | 62.4 |

**The load-bearing signal** is the **within-backbone** trend: as training
augmentation goes none → heavy, 10%-mutation recall should rise sharply
(here 27.7 → 62.4) while 0%-mutation recall stays roughly flat (~80). If the
substitute backbone shows the SAME shape — big gain in the noisy regime from
augmentation, little cost in the clean regime — the recipe generalizes, which is
the reviewer's question.

**Report as a within-backbone delta.** The new backbone may have lower absolute
recall than DNABERT-2; that is expected and irrelevant. The claim is "the
augmentation produces the same robustness trend regardless of backbone," NOT
"the new backbone beats DNABERT-2." Do not present a head-to-head absolute
comparison that could be misread as the recipe failing.