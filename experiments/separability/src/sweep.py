"""Radius sweep, capture curves, band margin, and the f_j conversion.

Everything in this module operates on one object: a similarity matrix ``sim``
of shape (n_queries, n_targets) whose entries are "closeness", higher = closer,
on a [0, 1]-comparable scale. LOCALE supplies cosine, the k-mer baseline
supplies 31-mer containment, MMseqs2 supplies alignment identity. Putting all
three on the same grid is the whole point -- the probe is comparative.

The two granularities differ only in what a query is:

  gene    query = one whole gene, all its windows; sim = max over (query window,
          target window) pairs. Late-interaction max-similarity, never a mean
          pool -- pooled accession representation is an open problem in this
          codebase and this probe does not need it solved.
  window  query = a single 256 bp window, i.e. a simulated read; sim = max over
          the target gene's windows. This is the benchmark's own regroup
          protocol and the realistic read-search setting.

Self-matches are excluded via ``self_mask``: a carrier must be found by *other*
carriers or the capture rate is just measuring that a sequence matches itself.

There are also two different things "carrier capture" can mean, and the probe
reports both because they answer different halves of the question. The
distinction is not cosmetic -- getting it wrong makes the numbers for any
multi-gene element meaningless:

  same_gene  Carriers of the query's *own* gene only. A clbB query is scored
             against other clbB sequences; clbA is neither a carrier to pool
             nor a confusable to exclude, so it is masked out entirely. This is
             the "is the radius wider than the divergence among true carriers"
             half, i.e. strain-level and cross-genus divergence of one gene.
  element    Carriers are *accessions* -- strains, not genes. An accession is
             captured when any window of any of its carrier genes falls inside
             the radius, which is the benchmark's own regroup-to-accession
             protocol. This is the half f_j actually consumes: the fraction of
             carrying individuals a feature column would pool.

Why both are needed: the pks island spans 19 clb genes, so a clbB query can
never retrieve clbA, and "fraction of all carrier genes captured" is capped near
1/19 no matter how good the embedding is. Pooling at the accession level removes
that artefact. blaCTX-M hides the problem because it is a single gene family --
which is exactly why a probe on one element would have missed it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# One grid for every method so LOCALE, k-mer and MMseqs2 curves are directly
# superimposable. Cosine between unrelated DNA embeddings is comfortably
# positive, so starting at 0 loses nothing; the raw distributions are kept
# alongside the grid for anything the grid would round away.
RADII = np.linspace(0.0, 1.0, 201)

# Carrier capture we require of a usable radius. 0.9 is not sacred; the report
# shows the curve, and `band` is recomputable at any tau.
TAU_DEFAULT = 0.90
# Confusable capture we are willing to call "near zero".
ALPHA_DEFAULT = 0.01


@dataclass
class CaptureCurves:
    """Per-radius capture, both as per-query curves and aggregated."""

    radii: np.ndarray
    # (n_queries, n_radii) -- the distribution the brief insists on, never a
    # single arbitrary query.
    carrier_per_query: np.ndarray
    confusable_per_query: np.ndarray
    n_queries: int
    n_carriers: int
    n_confusables: int
    # Raw pooled similarities, kept for AUROC and for the quantile-based band.
    carrier_sims: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    confusable_sims: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    @property
    def carrier_mean(self) -> np.ndarray:
        return self.carrier_per_query.mean(axis=0) if self.n_queries else self.radii * 0

    @property
    def confusable_mean(self) -> np.ndarray:
        return (
            self.confusable_per_query.mean(axis=0) if self.n_queries else self.radii * 0
        )

    def carrier_quantile(self, q: float) -> np.ndarray:
        return np.quantile(self.carrier_per_query, q, axis=0)

    def confusable_quantile(self, q: float) -> np.ndarray:
        return np.quantile(self.confusable_per_query, q, axis=0)


def sweep(
    sim: np.ndarray,
    target_is_carrier: np.ndarray,
    self_mask: np.ndarray | None = None,
    radii: np.ndarray = RADII,
) -> CaptureCurves:
    """Capture rates at every radius.

    sim                 (n_queries, n_targets) similarity, higher = closer
    target_is_carrier   (n_targets,) bool
    self_mask           (n_queries, n_targets) bool, True where the pair must be
                        dropped (a query against itself). Dropped pairwise, so a
                        query's carrier denominator is its own carrier count.
    """
    sim = np.asarray(sim, dtype=np.float64)
    target_is_carrier = np.asarray(target_is_carrier, dtype=bool)
    n_q, n_t = sim.shape
    if target_is_carrier.shape != (n_t,):
        raise ValueError(
            f"target_is_carrier has shape {target_is_carrier.shape}, expected {(n_t,)}"
        )
    valid = np.ones_like(sim, dtype=bool)
    if self_mask is not None:
        self_mask = np.asarray(self_mask, dtype=bool)
        if self_mask.shape != sim.shape:
            raise ValueError("self_mask must match sim's shape")
        valid &= ~self_mask

    car_col = target_is_carrier[None, :]
    car_valid = valid & car_col
    con_valid = valid & ~car_col

    car_n = car_valid.sum(axis=1)
    con_n = con_valid.sum(axis=1)

    # (n_queries, n_radii): count of valid targets at or above each radius.
    # Done by sorting once per query row rather than an (n_q, n_t, n_r) compare.
    def _curve(mask: np.ndarray, denom: np.ndarray) -> np.ndarray:
        out = np.zeros((n_q, radii.size), dtype=np.float64)
        for i in range(n_q):
            if denom[i] == 0:
                continue
            vals = np.sort(sim[i][mask[i]])
            # number of vals >= r  ==  len - searchsorted(vals, r, 'left')
            counts = vals.size - np.searchsorted(vals, radii, side="left")
            out[i] = counts / denom[i]
        return out

    carrier_curve = _curve(car_valid, car_n)
    confusable_curve = _curve(con_valid, con_n)

    return CaptureCurves(
        radii=radii,
        carrier_per_query=carrier_curve,
        confusable_per_query=confusable_curve,
        n_queries=n_q,
        n_carriers=int(car_n.max()) if n_q else 0,
        n_confusables=int(con_n.max()) if n_q else 0,
        carrier_sims=sim[car_valid],
        confusable_sims=sim[con_valid],
    )


def aggregate_to_entities(
    sim: np.ndarray, entity_of_col: np.ndarray, n_entities: int
) -> np.ndarray:
    """Collapse target columns to entities by max.

    ``entity_of_col[j]`` is the entity column j belongs to, or -1 to drop it.
    Max, not mean: an accession is a carrier if *any* of its genes matches, the
    same reduction ``regroup_topk_hits`` applies in the benchmark.
    """
    out = np.full((sim.shape[0], n_entities), -np.inf, dtype=np.float64)
    for e in range(n_entities):
        cols = np.nonzero(entity_of_col == e)[0]
        if cols.size:
            out[:, e] = sim[:, cols].max(axis=1)
    return out


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC of carrier vs confusable similarities, ties at 0.5.

    Threshold-free, so it survives the fact that cosine, containment and
    alignment identity live on different scales -- the one number that compares
    methods without picking a radius for them.
    """
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    # Average ranks within ties so a method that returns a constant scores 0.5
    # instead of 1.0.
    srt = allv[order]
    i = 0
    while i < srt.size:
        j = i
        while j + 1 < srt.size and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    rsum = ranks[: pos.size].sum()
    return float((rsum - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


@dataclass
class Band:
    """The window, if one exists.

    lo   the radius just above every (well, all but alpha of the) confusable --
         the narrowest we are allowed to be
    hi   the radius at which carrier capture is still tau -- the widest we are
         allowed to be
    margin = hi - lo. Positive means a usable window exists; negative means the
    bands interleave and no threshold works, which is the NO-GO condition.
    """

    tau: float
    alpha: float
    lo: float
    hi: float
    margin: float
    exists: bool
    carrier_capture_at_lo: float
    confusable_capture_at_hi: float
    auroc: float


def band(
    curves: CaptureCurves, tau: float = TAU_DEFAULT, alpha: float = ALPHA_DEFAULT
) -> Band:
    """Widest radius window with carrier capture >= tau and confusable <= alpha.

    Both bounds are read off the *aggregate* curves. `hi` is the largest radius
    whose mean carrier capture is still at least tau; `lo` is the smallest
    radius whose mean confusable capture has fallen to alpha. The window is
    [lo, hi] and it exists when lo <= hi.
    """
    r = curves.radii
    car = curves.carrier_mean
    con = curves.confusable_mean

    hi_idx = np.nonzero(car >= tau)[0]
    hi = float(r[hi_idx[-1]]) if hi_idx.size else float("nan")

    lo_idx = np.nonzero(con <= alpha)[0]
    lo = float(r[lo_idx[0]]) if lo_idx.size else float("nan")

    margin = hi - lo
    exists = bool(np.isfinite(margin) and margin >= 0)
    return Band(
        tau=tau,
        alpha=alpha,
        lo=lo,
        hi=hi,
        margin=float(margin),
        exists=exists,
        carrier_capture_at_lo=float(np.interp(lo, r, car)) if np.isfinite(lo) else float("nan"),
        confusable_capture_at_hi=float(np.interp(hi, r, con)) if np.isfinite(hi) else float("nan"),
        auroc=auroc(curves.carrier_sims, curves.confusable_sims),
    )


# --------------------------------------------------------------------------- #
# f_j -- what the downstream statistics actually consumes
# --------------------------------------------------------------------------- #


@dataclass
class FjAssumptions:
    """Cohort assumptions behind the f_j curve. All of these are inputs, not
    findings, and the report states them next to every f_j number.

    element_prevalence      fraction of individuals carrying the element at all
    confusable_prevalence   fraction carrying the confusable (the contamination
                            source when the radius is too wide)
    """

    element_prevalence: float
    confusable_prevalence: float
    source: str


def fj_curve(
    curves: CaptureCurves, assumptions: FjAssumptions
) -> dict[str, np.ndarray]:
    """Implied carrier frequency of the pooled feature column at each radius.

    Model, stated plainly because it is doing real work:

      * The element is present in a fraction ``P_E`` of individuals. Carriers
        are assumed uniformly distributed over the distinct carrier sequences we
        observed -- we have no population frequencies for strain variants, so
        every carrier sequence stands for an equal slice of P_E. This is the
        single biggest modelling assumption in the probe.
      * At radius r a feature column defined by one query pools the fraction
        ``c(r)`` of carriers it captures, so the column's true-carrier frequency
        is ``f_j(r) = P_E * c(r)``.
      * Non-carriers get swept in when the radius reaches the confusables:
        ``f_contam(r) = P_C * x(r)``, with ``x(r)`` the confusable capture.
      * ``purity`` is the fraction of the column that is genuinely a carrier.
        A column with high f_j and low purity is worse than useless: it dilutes
        the effect it is meant to detect.
    """
    c = curves.carrier_mean
    x = curves.confusable_mean
    fj = assumptions.element_prevalence * c
    contam = assumptions.confusable_prevalence * x
    total = fj + contam
    with np.errstate(divide="ignore", invalid="ignore"):
        purity = np.where(total > 0, fj / total, np.nan)
    return {
        "radii": curves.radii,
        "carrier_capture": c,
        "confusable_capture": x,
        "f_j": fj,
        "f_contamination": contam,
        "f_column_total": total,
        "purity": purity,
    }


def fj_for_cluster(
    cluster_carrier_share: float, cluster_confusable_share: float,
    assumptions: FjAssumptions,
) -> dict[str, float]:
    """Single-point f_j for a hard clustering (MMseqs2 / CD-HIT at one identity).

    A clustering has no radius to sweep at eval time: the threshold was spent at
    cluster time, and the feature column is whatever landed in the query's
    cluster. This is the number the identity baselines are compared on.
    """
    fj = assumptions.element_prevalence * cluster_carrier_share
    contam = assumptions.confusable_prevalence * cluster_confusable_share
    total = fj + contam
    return {
        "carrier_share": cluster_carrier_share,
        "confusable_share": cluster_confusable_share,
        "f_j": fj,
        "f_contamination": contam,
        "f_column_total": total,
        "purity": (fj / total) if total > 0 else float("nan"),
    }
