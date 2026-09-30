#!/usr/bin/env python
"""Single entry point for the separability probe.

    uv run python experiments/separability/run_probe.py --results-dir experiments/separability/results

Requires data/sequences.parquet (see data/fetch_elements.py). Embeds on whatever
device --device says; the benchmark harness needs CUDA but LOCALEEncoder does
not, so --device cpu is a valid, slower path.

What one run does, per element:

  1. Tile every sequence (carriers and confusables) at the model's native 256 bp
     window and embed the whole clean reference set once.
  2. For each noise rate, mutate the carrier sequences -- queries only, the
     reference stays clean, which is the read-against-catalog setting -- and
     embed them.
  3. Score every (query, reference gene) pair three ways: LOCALE cosine,
     exact 31-mer containment, MMseqs2 alignment identity. Reduce to gene level
     by max over windows, never by mean pooling.
  4. Sweep the radius, record carrier and confusable capture, find the band,
     convert to f_j.
  5. Add the operating points an identity catalog would actually give: MMseqs2
     clustering of the clean reference at 95 / 90 / 80 / 70 / 50% identity, with
     each query assigned to the catalog by its best alignment.

Everything lands in --results-dir as CSV + JSON + PNG, plus a summary.json that
report.md is written from.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import baselines  # noqa: E402
import embed as emb  # noqa: E402
import noise as noisemod  # noqa: E402
import sweep as sw  # noqa: E402

# Cohort assumptions behind every f_j number. These are INPUTS. They are the
# weakest link in the probe and the report says so next to every f_j figure:
# no primary source was re-read in this environment, the figures are the
# commonly quoted ones, and f_j scales linearly in element_prevalence so a
# reader who prefers another number can rescale by eye.
FJ_ASSUMPTIONS = {
    "pks_clb": sw.FjAssumptions(
        element_prevalence=0.20,
        confusable_prevalence=0.35,
        source=(
            "pks+ E. coli carriage in healthy adults is commonly quoted around "
            "20% (Nougayrede/Putze-era isolate surveys report the pks island in "
            "~20-35% of commensal B2 E. coli); yersiniabactin (the confusable) "
            "is more common still, hence 0.35. Not re-verified against primary "
            "sources here."
        ),
    ),
    "cdt_abc": sw.FjAssumptions(
        element_prevalence=0.10,
        confusable_prevalence=0.30,
        source=(
            "cdt-bearing Enterobacteriaceae/Campylobacter carriage assumed 10%; "
            "the confusable pool (hlyA/stx/elt/typhoid-toxin cdtB) is broader, "
            "assumed 30%. Order-of-magnitude placeholders."
        ),
    ),
    "bla_ctxm": sw.FjAssumptions(
        element_prevalence=0.15,
        confusable_prevalence=0.50,
        source=(
            "ESBL (largely CTX-M) faecal carriage in community cohorts varies "
            "enormously by region; 15% is a mid-range assumption. blaTEM in "
            "particular is near-ubiquitous in Enterobacteriaceae, hence 0.50 "
            "for the confusable pool."
        ),
    ),
}
DEFAULT_ASSUMPTION = sw.FjAssumptions(0.20, 0.35, "fallback: element 20%, confusable 35%")

NOISE_RATES = (0.0, 0.05, 0.10)
CLUSTER_IDENTITIES = (0.95, 0.90, 0.80, 0.70, 0.50)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def locale_window_to_gene(
    qvec: np.ndarray,
    tvec: np.ndarray,
    towner: np.ndarray,
    block: int = 8192,
    device: str = "cpu",
) -> np.ndarray:
    """(n_query_windows, n_target_genes) cosine, max over each gene's windows.

    Blocked over query windows so the (query windows x target windows) product
    never has to exist all at once -- a pks reference set is tens of thousands
    of windows on both sides, and the full product would be tens of GB.

    Runs on the GPU when one is available. This is not premature: a pks run is
    ~25k query windows against ~70k reference windows in 768 dimensions, which
    is a couple of TFLOPs per noise level. On CPU that is minutes per element
    and it was the second-largest cost in the probe after the identity
    alignment; on an A100 it is under a second. The reduction is exact either
    way -- max over each gene's contiguous block of windows.
    """
    n_genes = int(towner[-1]) + 1 if towner.size else 0
    if n_genes == 0 or qvec.shape[0] == 0:
        return np.zeros((qvec.shape[0], n_genes), dtype=np.float32)

    starts = emb.group_starts(towner)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if not use_cuda:
        out = np.zeros((qvec.shape[0], n_genes), dtype=np.float32)
        for s in range(0, qvec.shape[0], block):
            e = min(s + block, qvec.shape[0])
            out[s:e] = emb.max_over_groups(
                emb.cosine(qvec[s:e], tvec), towner, axis=1
            )
        return out

    # Vectors are already unit norm (LOCALEEncoder normalises), so a plain
    # matmul is cosine. Grouped max via index_reduce over gene ids.
    t = torch.from_numpy(tvec).to(device)
    gene_of_window = torch.from_numpy(
        np.repeat(
            np.arange(n_genes),
            np.diff(np.append(starts, towner.size)),
        )
    ).to(device)
    out = np.zeros((qvec.shape[0], n_genes), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, qvec.shape[0], block):
            e = min(s + block, qvec.shape[0])
            q = torch.from_numpy(qvec[s:e]).to(device)
            sim = q @ t.T                                 # (b, n_target_windows)
            red = torch.full((e - s, n_genes), -float("inf"), device=device)
            red.index_reduce_(1, gene_of_window, sim, "amax", include_self=True)
            out[s:e] = red.cpu().numpy()
    del t
    torch.cuda.empty_cache()
    return out


def sample_query_windows(
    owner: np.ndarray, per_gene: int, rng: np.random.Generator
) -> np.ndarray:
    """Up to `per_gene` window rows per gene -- the read-level query set.

    Sampled rather than exhaustive because a 54 kb island contributes ~500
    windows and would otherwise swamp the query distribution with one gene.
    """
    picks: list[int] = []
    starts = emb.group_starts(owner)
    ends = np.append(starts[1:], owner.size)
    for s, e in zip(starts, ends):
        idx = np.arange(s, e)
        if idx.size > per_gene:
            idx = rng.choice(idx, size=per_gene, replace=False)
        picks.extend(sorted(idx.tolist()))
    return np.asarray(picks, dtype=np.int64)


def self_mask_from(query_source: np.ndarray, n_targets: int) -> np.ndarray:
    """True where a query would be scored against the reference it came from."""
    m = np.zeros((query_source.size, n_targets), dtype=bool)
    ok = query_source >= 0
    m[np.nonzero(ok)[0], query_source[ok]] = True
    return m


def _genes_per_accession(
    acc_of_row: list[str], gene_key: list[str], is_carrier
) -> dict[str, int]:
    """Histogram: how many distinct carrier genes each carrier accession has."""
    from collections import Counter, defaultdict

    per: dict[str, set] = defaultdict(set)
    for a, g, c in zip(acc_of_row, gene_key, is_carrier):
        if c:
            per[a].add(g)
    hist = Counter(len(v) for v in per.values())
    return {str(k): int(hist[k]) for k in sorted(hist)}


def pick_matched_genus(
    genus: list[str], is_carrier: np.ndarray, min_each: int = 10
) -> str | None:
    """The genus with the most sequences that has both carriers and confusables.

    The control this exists for: if every carrier came from one genus and every
    confusable from another, a model could score well by reading genomic
    background -- GC content, codon usage, k-mer style -- and never look at the
    gene at all. Restricting both sides to one genus removes that shortcut. It
    is not available for every element (Salmonella's typhoid-toxin cdtB is a
    confusable *because* of its genus), and the report says where it is missing.
    """
    best, best_n = None, 0
    for g in sorted(set(genus)):
        if not g:
            continue
        sel = np.array([x == g for x in genus])
        n_car = int((sel & is_carrier).sum())
        n_con = int((sel & ~is_carrier).sum())
        if n_car >= min_each and n_con >= min_each and n_car + n_con > best_n:
            best, best_n = g, n_car + n_con
    return best


def _same_gene_scope(gene_key: list[str], is_carrier, query_source, base_mask):
    """Scope where a query is scored only against carriers of its own gene.

    Carriers of a *different* gene of the same element are masked out of both
    numerator and denominator: a clbB read is not supposed to retrieve clbA, so
    counting clbA as a missed carrier would penalise every method for something
    none of them should do. Confusables stay in scope -- they are still the
    thing that must not be admitted.
    """
    gk = np.asarray(gene_key, dtype=object)
    is_carrier = np.asarray(is_carrier)
    q_gene = np.where(query_source >= 0, gk[query_source], None)
    # (n_queries, n_targets): carrier target whose gene differs from the query's
    other_gene = is_carrier[None, :] & (q_gene[:, None] != gk[None, :])
    mask = base_mask | other_gene

    def apply(sim):
        return sim, is_carrier, mask

    return apply


def _element_scope(
    acc_of_row: list[str],
    is_carrier,
    query_source,
    base_mask,
    min_genes: int = 1,
    gene_key: list[str] | None = None,
):
    """Scope where carriers and confusables are accessions, not genes.

    A carrier entity is an accession holding at least one carrier gene -- a
    strain. A confusable entity is an accession holding confusable genes and
    **no** carrier gene: an accession carrying both pks and ybt is a genuine
    carrier, so retrieving it is not contamination and counting it as such
    would invent a false-positive rate.

    ``min_genes`` restricts carrier entities to accessions holding at least
    that many distinct carrier genes. This is not cosmetic. GenBank contigs do
    not span a 54 kb island, so the median pks carrier accession here deposits
    2 of 17 clb genes -- and an accession that deposited only clbK cannot be
    retrieved by a clbB query, which makes the unrestricted number a measure of
    deposition practice rather than of the embedding. Requiring substantial
    element content gives an interpretable estimate from the same data.
    Accessions below the threshold are dropped from the *target* set but their
    sequences are still allowed as queries, so query count is unaffected.
    """
    accs = np.asarray(acc_of_row, dtype=object)
    is_carrier = np.asarray(is_carrier)
    all_carrier_accs = {a for a, c in zip(accs, is_carrier) if c}
    carrier_accs = set(all_carrier_accs)
    if min_genes > 1:
        if gene_key is None:
            raise ValueError("min_genes > 1 needs gene_key")
        gk = np.asarray(gene_key, dtype=object)
        per: dict[str, set] = {}
        for a, g, c in zip(accs, gk, is_carrier):
            if c:
                per.setdefault(a, set()).add(g)
        carrier_accs = {a for a in carrier_accs if len(per.get(a, ())) >= min_genes}
        if not carrier_accs:
            return None
    # Subtract the UNRESTRICTED carrier set: an accession with one clb gene
    # plus a ybt cluster is still a genuine pks carrier, and min_genes dropping
    # it from the carrier entities must not silently promote it to a
    # confusable -- that would count retrieving a true carrier as a false
    # positive.
    confusable_accs = {
        a for a, c in zip(accs, is_carrier) if not c
    } - all_carrier_accs

    entities = sorted(carrier_accs) + sorted(confusable_accs)
    if not entities:
        return None
    pos = {a: i for i, a in enumerate(entities)}
    entity_is_carrier = np.array(
        [a in carrier_accs for a in entities], dtype=bool
    )
    # Columns of accessions that hold both carrier and confusable genes are
    # folded into their carrier entity; a confusable gene of a true carrier is
    # not a separate thing to avoid.
    entity_of_col = np.array([pos.get(a, -1) for a in accs], dtype=np.int64)
    q_entity = np.array(
        [pos.get(accs[s], -1) if s >= 0 else -1 for s in query_source],
        dtype=np.int64,
    )
    n_ent = len(entities)
    ent_mask = np.zeros((query_source.size, n_ent), dtype=bool)
    ok = q_entity >= 0
    ent_mask[np.nonzero(ok)[0], q_entity[ok]] = True

    def apply(sim):
        agg = sw.aggregate_to_entities(sim, entity_of_col, n_ent)
        # -inf columns (no genes) cannot happen by construction, but a method
        # that reports nothing for a pair legitimately yields its floor.
        agg = np.where(np.isfinite(agg), agg, 0.0)
        return agg, entity_is_carrier, ent_mask

    return apply


def match_carrier_capture(curves: sw.CaptureCurves, confusable_target: float) -> float:
    """LOCALE's carrier capture at the radius where it admits `confusable_target`
    confusables -- the operating point that matches a clustering threshold.

    Confusable capture falls monotonically with radius, so the matching radius
    is the smallest one whose confusable capture is at or below the target; if
    even radius 0 lets in fewer confusables than the clustering does, LOCALE is
    scored at radius 0, which is the most generous reading for it and still
    caps carrier capture at 1.
    """
    con = curves.confusable_mean
    car = curves.carrier_mean
    ok = np.nonzero(con <= confusable_target + 1e-12)[0]
    if ok.size == 0:
        return float("nan")
    return float(car[ok[0]])


def suspect_label_pairs(
    sim: np.ndarray,
    is_carrier: np.ndarray,
    mask: np.ndarray,
    query_source: np.ndarray,
    rows: pl.DataFrame,
    thresh: float = 0.99,
    top: int = 6,
) -> dict:
    """Carrier-confusable pairs that are near-identical on clean sequences.

    A carrier sitting at cosine >= 0.99 from something labelled a confusable is
    far more likely a labelling error than a real finding, and it corrupts the
    confusable boundary that the whole verdict rests on. This diagnostic is
    here because it caught exactly that: four sequences whose only annotation
    was the generic /gene "bla" + "class A beta-lactamase" had matched an
    over-broad CTX-M pattern and were sitting on top of real TEMs. A non-zero
    count here should be read as "audit the manifest", not as a result.
    """
    acc = rows["accession"].to_list()
    prod = rows["product"].to_list()
    group = rows["group"].to_list()
    valid = (~mask) & (~is_carrier)[None, :]
    hit = valid & (sim >= thresh)
    n_pairs = int(hit.sum())
    worst: list[dict] = []
    if n_pairs:
        per_query = hit.sum(axis=1)
        for k in np.argsort(-per_query)[:top]:
            if per_query[k] == 0:
                break
            src = int(query_source[k])
            js = np.nonzero(hit[k])[0]
            worst.append({
                "query_accession": acc[src] if src >= 0 else None,
                "query_product": prod[src][:70] if src >= 0 else None,
                "n_confusables_within": int(per_query[k]),
                "example_confusable": {
                    "group": group[int(js[0])],
                    "accession": acc[int(js[0])],
                    "product": prod[int(js[0])][:70],
                },
            })
    return {
        "threshold": thresh,
        "n_pairs": n_pairs,
        "fraction_of_valid_pairs": float(hit.sum() / max(valid.sum(), 1)),
        "n_carrier_queries_affected": int((hit.sum(axis=1) > 0).sum()),
        "worst": worst,
    }


def near_duplicate_fraction(sim: np.ndarray, is_carrier: np.ndarray,
                            mask: np.ndarray, thresh: float = 0.99) -> float:
    """Fraction of carrier-carrier pairs that are near-identical.

    Exact duplicates were dropped at fetch time, but two accessions depositing
    the same allele with one base changed both survive. They inflate carrier
    capture for every method equally, so they do not bias the comparison -- but
    they do inflate the absolute numbers, and the report has to say by how much.
    """
    valid = (~mask) & is_carrier[None, :]
    if not valid.any():
        return float("nan")
    return float((sim[valid] >= thresh).mean())


# --------------------------------------------------------------------------- #
# per-element driver
# --------------------------------------------------------------------------- #


def run_element(
    element: str,
    rows: pl.DataFrame,
    encoder,
    args,
    out_dir: Path,
) -> dict:
    t0 = time.time()
    seqs = rows["sequence"].to_list()
    labels = rows["label"].to_list()
    groups = rows["group"].to_list()
    ids = [
        f"{g}|{a}|{s}" for g, a, s in
        zip(groups, rows["accession"].to_list(), rows["start"].to_list())
    ]
    is_carrier = np.array([lab == "carrier" for lab in labels], dtype=bool)
    carrier_idx = np.nonzero(is_carrier)[0]
    gene_key = rows["gene_key"].to_list()
    acc_of_row = rows["accession"].to_list()

    # Accession entities, shared by the `element` scope and the catalog points
    # so both count in the same units. An accession holding both carrier and
    # confusable genes is a carrier: retrieving it is not contamination.
    _carrier_accs = {a for a, c in zip(acc_of_row, is_carrier) if c}
    _confusable_accs = {a for a, c in zip(acc_of_row, is_carrier) if not c} - _carrier_accs
    _entities = sorted(_carrier_accs) + sorted(_confusable_accs)
    _epos = {a: i for i, a in enumerate(_entities)}
    entity_of_col = np.array([_epos.get(a, -1) for a in acc_of_row], dtype=np.int64)
    entity_is_carrier = np.array([a in _carrier_accs for a in _entities], dtype=bool)

    print(f"\n=== {element}: {is_carrier.sum()} carriers, "
          f"{(~is_carrier).sum()} confusables, {len(seqs)} references ===")

    # Reference side: clean, embedded once and reused at every noise level.
    t_windows, t_owner = emb.tile_many(seqs)
    print(f"  reference: {len(t_windows)} windows; embedding...")
    t_vec = emb.embed(encoder, t_windows)
    print(f"  reference embedded in {time.time() - t0:.1f}s")

    rng = np.random.default_rng(args.seed)
    curve_rows: list[dict] = []
    summary: dict = {
        "element": element,
        "n_carriers": int(is_carrier.sum()),
        "n_confusables": int((~is_carrier).sum()),
        "n_reference_windows": int(len(t_windows)),
        "n_carrier_accessions": int(entity_is_carrier.sum()),
        "n_confusable_accessions": int((~entity_is_carrier).sum()),
        "n_accessions_with_both": len(_carrier_accs & {
            a for a, c in zip(acc_of_row, is_carrier) if not c
        }),
        "carrier_gene_keys": sorted({
            k for k, c in zip(gene_key, is_carrier) if c
        }),
        # How much of the element each carrier accession actually carries. This
        # is the single most important caveat on every `element` scope number:
        # an accession that deposited one clb gene cannot be retrieved by a
        # query for a different clb gene, and no method should be expected to.
        # If this histogram is dominated by 1, the element scope is measuring
        # GenBank deposition practice, not embedding geometry.
        "carrier_genes_per_accession": _genes_per_accession(
            acc_of_row, gene_key, is_carrier
        ),
        "carrier_genera": sorted(
            {g for g, c in zip(rows["genus"].to_list(), is_carrier) if c and g}
        ),
        "confusable_genera": sorted(
            {g for g, c in zip(rows["genus"].to_list(), is_carrier) if not c and g}
        ),
        "carrier_groups": sorted({g for g, c in zip(groups, is_carrier) if c}),
        "confusable_groups": sorted({g for g, c in zip(groups, is_carrier) if not c}),
        "bands": [],
        "matched_genus_bands": [],
        "catalog_points": [],
    }
    if len(summary["carrier_genera"]) < 2:
        summary["warning"] = (
            "carriers span fewer than two genera -- the cross-species pooling "
            "claim is UNTESTED for this element"
        )

    assumptions = FJ_ASSUMPTIONS.get(element, DEFAULT_ASSUMPTION)
    summary["fj_assumptions"] = asdict(assumptions)

    # Carrier accessions must hold at least this many distinct element genes
    # to count as a target entity in the restricted element scope. Two for a
    # small operon, a quarter of the genes for a large island -- enough to rule
    # out "the submitter annotated one gene" without demanding a complete
    # island, which GenBank contigs rarely span.
    n_carrier_genes = len({k for k, c in zip(gene_key, is_carrier) if c})
    min_element_genes = 1 if n_carrier_genes < 2 else max(2, n_carrier_genes // 4)
    summary["min_element_genes"] = min_element_genes
    summary["n_carrier_genes"] = n_carrier_genes

    genus_list = rows["genus"].to_list()
    matched_genus = pick_matched_genus(genus_list, is_carrier)
    summary["matched_genus"] = matched_genus
    if matched_genus is None:
        summary["matched_genus_note"] = (
            "no genus holds >=10 carriers AND >=10 confusables, so the "
            "same-genus control could not be run for this element: a genomic-"
            "background shortcut cannot be ruled out here"
        )
    else:
        print(f"  same-genus control: {matched_genus} "
              f"({sum(1 for g, c in zip(genus_list, is_carrier) if g == matched_genus and c)} carriers / "
              f"{sum(1 for g, c in zip(genus_list, is_carrier) if g == matched_genus and not c)} confusables)")

    # Catalogs: cluster the clean reference once per identity threshold. The
    # catalog does not depend on the noise level, only the assignment does.
    catalogs: dict[float, dict[str, str]] = {}
    if not args.no_mmseqs:
        try:
            wd = baselines.scratch_dir(f"{element}_allvall")
            ava = baselines.mmseqs_all_vs_all(ids, seqs, wd)
            for ident in CLUSTER_IDENTITIES:
                catalogs[ident] = baselines.mmseqs_cluster(
                    ids, seqs, ident, wd, precomputed=ava
                )
                reps = set(catalogs[ident].values())
                n_car_clu = len({
                    catalogs[ident][i] for i, c in zip(ids, is_carrier) if c
                })
                print(f"  catalog @ {ident:.0%} identity: {len(reps)} clusters "
                      f"({n_car_clu} of them containing carriers)")
        except Exception as exc:
            print(f"  [warn] clustering failed: {exc}")

    # UHGP-style protein catalog. UHGP-90 and UHGP-50 are MMseqs2 clusterings of
    # predicted proteins at 90% and 50% amino-acid identity, so clustering the
    # translated CDSs at those thresholds *is* the catalog protocol, applied to
    # these sequences. It answers the brief's catalog question directly -- do
    # all carriers land in one cluster -- without downloading UHGP itself.
    # Clean sequences only: frame-1 translation is meaningless once indels have
    # shifted the frame, and pretending otherwise would manufacture a baseline
    # collapse that is an artefact of this script rather than of the catalog.
    if catalogs and not args.no_protein:
        try:
            prot = [baselines.translate_cds(s) for s in seqs]
            wd = baselines.scratch_dir(f"{element}_prot")
            ava_p = baselines.mmseqs_all_vs_all(ids, prot, wd, search_type=1)
            for ident in (0.90, 0.50):
                m = baselines.mmseqs_cluster(
                    ids, prot, ident, wd, precomputed=ava_p
                )
                car_reps = {m[i] for i, c in zip(ids, is_carrier) if c}
                sizes = {
                    r: sum(1 for i, c in zip(ids, is_carrier) if c and m[i] == r)
                    for r in car_reps
                }
                big = max(sizes, key=sizes.get)
                n_con_in_big = sum(
                    1 for i, c in zip(ids, is_carrier) if not c and m[i] == big
                )
                pt = {
                    "catalog": f"UHGP-style protein clustering @ {ident:.0%} AA identity",
                    "min_seq_id": ident,
                    "n_carrier_clusters": len(car_reps),
                    "largest_carrier_cluster": sizes[big],
                    "largest_carrier_cluster_share":
                        sizes[big] / max(int(is_carrier.sum()), 1),
                    "confusables_in_largest_carrier_cluster": n_con_in_big,
                    "all_carriers_in_one_cluster": len(car_reps) == 1,
                }
                summary.setdefault("protein_catalog_points", []).append(pt)
                print(f"  protein catalog @ {ident:.0%} AA: {len(car_reps)} carrier "
                      f"clusters, largest holds {sizes[big]}/{int(is_carrier.sum())} "
                      f"carriers + {n_con_in_big} confusables")
        except Exception as exc:
            print(f"  [warn] protein catalog failed: {exc}")

    for rate in args.noise:
        q_seqs_clean = [seqs[i] for i in carrier_idx]
        if args.convention == "repo":
            # The Augmenter reports the identity it achieved, exactly, for
            # free. Recovering it afterwards with an alignment is O(n*m) in
            # Python and dominates the runtime on kilobase genes.
            pairs = [
                noisemod.mutate_repo_with_identity(sq, rate, args.seed + i)
                for i, sq in enumerate(q_seqs_clean)
            ]
            q_seqs = [a for a, _ in pairs]
            realized = float(np.mean([b for _, b in pairs])) if pairs else 1.0
            identity_source = "exact (Augmenter)"
        else:
            q_seqs = noisemod.mutate_all(
                q_seqs_clean, rate, convention=args.convention, salt=args.seed
            )
            realized = (
                float(np.mean([
                    noisemod.realized_identity(a, b)
                    for a, b in list(zip(q_seqs_clean, q_seqs))[: args.identity_check]
                ]))
                if rate > 0 else 1.0
            )
            identity_source = f"estimated on <=1500 bp prefixes of {args.identity_check} seqs"
        print(f"\n  -- noise {rate:.0%} ({args.convention}), "
              f"realized identity {realized:.3f} [{identity_source}] --")

        q_windows, q_owner = emb.tile_many(q_seqs)
        q_vec = emb.embed(encoder, q_windows)
        W = locale_window_to_gene(
            q_vec, t_vec, t_owner, device=args.device
        )                                                    # (n_qw, n_targets)
        G = emb.max_over_groups(W, q_owner, axis=0)          # (n_genes, n_targets)

        # Read-level query set: a bounded sample of windows per carrier gene.
        pick = sample_query_windows(q_owner, args.windows_per_gene, rng)
        w_src = carrier_idx[q_owner[pick]]

        granularities = {
            "gene": {
                "sim": {"locale": G},
                "query_seqs": q_seqs,
                "query_source": carrier_idx,
                "query_ids": [f"q{i}" for i in range(len(q_seqs))],
            },
            "window": {
                "sim": {"locale": W[pick]},
                "query_seqs": [q_windows[i] for i in pick],
                "query_source": w_src,
                "query_ids": [f"w{i}" for i in range(pick.size)],
            },
        }

        for gran, payload in granularities.items():
            qs = payload["query_seqs"]
            qsrc = np.asarray(payload["query_source"])
            mask = self_mask_from(qsrc, len(seqs))

            payload["sim"]["kmer31"] = baselines.kmer_similarity(qs, seqs)
            if not args.no_mmseqs:
                try:
                    wd = baselines.scratch_dir(f"{element}_{gran}_{int(rate * 100)}")
                    fid, fcov = baselines.mmseqs_search_similarity(
                        payload["query_ids"], qs, ids, seqs, wd
                    )
                    payload["sim"]["mmseqs_fident"] = fid
                    payload["sim"]["mmseqs_fident_x_qcov"] = fcov
                except Exception as exc:
                    print(f"    [warn] mmseqs search failed ({gran}): {exc}")

            # Two carrier definitions per granularity -- see sweep.py's
            # docstring. `same_gene` answers "is the radius wider than carrier
            # divergence"; `element` answers "what f_j would this column have".
            scopes = {
                "same_gene": _same_gene_scope(
                    gene_key, is_carrier, qsrc, mask
                ),
                "element": _element_scope(
                    acc_of_row, is_carrier, qsrc, mask
                ),
            }
            if min_element_genes > 1:
                scopes[f"element_ge{min_element_genes}genes"] = _element_scope(
                    acc_of_row, is_carrier, qsrc, mask,
                    min_genes=min_element_genes, gene_key=gene_key,
                )

            curves_by_method: dict[str, sw.CaptureCurves] = {}
            for scope_name, scope in scopes.items():
                if scope is None:
                    continue
                for method, sim in payload["sim"].items():
                    s_sim, s_car, s_mask = scope(sim)
                    curves = sw.sweep(s_sim, s_car, s_mask)
                    if scope_name == "element":
                        curves_by_method[method] = curves
                    band = sw.band(curves, tau=args.tau, alpha=args.alpha)
                    fj = sw.fj_curve(curves, assumptions)
                    # alpha is an arbitrary budget, so record the whole
                    # trade-off: carrier capture at each confusable budget a
                    # reader might prefer. Lets the report show sensitivity
                    # without a rerun.
                    alpha_sweep = {}
                    for a in (0.001, 0.01, 0.05, 0.10):
                        ba = sw.band(curves, tau=args.tau, alpha=a)
                        alpha_sweep[f"{a:g}"] = {
                            "lo": ba.lo,
                            "carrier_capture": ba.carrier_capture_at_lo,
                        }
                    summary["bands"].append({
                        "granularity": gran, "scope": scope_name,
                        "noise": rate, "method": method,
                        "n_queries": curves.n_queries,
                        "n_carrier_units": int(s_car.sum()),
                        "n_confusable_units": int((~s_car).sum()),
                        "realized_identity": realized,
                        "alpha_sweep": alpha_sweep,
                        **asdict(band),
                    })
                    q10 = curves.carrier_quantile(0.10)
                    q50 = curves.carrier_quantile(0.50)
                    q90 = curves.carrier_quantile(0.90)
                    cmax = curves.confusable_per_query.max(axis=0)
                    for k, r in enumerate(curves.radii):
                        curve_rows.append({
                            "element": element, "granularity": gran,
                            "scope": scope_name, "noise": rate,
                            "method": method, "radius": float(r),
                            "carrier_capture_mean": float(curves.carrier_mean[k]),
                            "carrier_capture_p10": float(q10[k]),
                            "carrier_capture_p50": float(q50[k]),
                            "carrier_capture_p90": float(q90[k]),
                            "confusable_capture_mean": float(curves.confusable_mean[k]),
                            "confusable_capture_max": float(cmax[k]),
                            "f_j": float(fj["f_j"][k]),
                            "f_contamination": float(fj["f_contamination"][k]),
                            "purity": float(fj["purity"][k]),
                        })
                    # The headline number is not the margin but carrier capture
                    # at the tightest radius that still excludes confusables:
                    # that is what f_j is proportional to, and it stays
                    # meaningful when the band is empty (where margin
                    # degenerates).
                    print(f"    {gran:<7} {scope_name:<10} {method:<22} "
                          f"AUROC={band.auroc:.4f}  "
                          f"carrier@conf<={args.alpha:g}: "
                          f"{band.carrier_capture_at_lo:.3f} "
                          f"(r>={band.lo:.3f})  margin={band.margin:+.3f}")

            # Same-genus control: rerun the sweep with carriers and confusables
            # both restricted to one genus, so a model cannot win by reading
            # genomic background instead of the gene.
            if matched_genus is not None:
                col = np.array([g == matched_genus for g in genus_list])
                row = col[qsrc] & (qsrc >= 0)
                if row.sum() >= 5 and col.sum() >= 10:
                    # Restricted to the same_gene scope, not the unrestricted
                    # all-carrier-genes one: inside a single genus the
                    # multi-gene artefact (a clbB query cannot retrieve clbA)
                    # would otherwise dominate and drive every method to near
                    # chance, which says nothing about a genomic-background
                    # shortcut -- the thing this control exists to rule out.
                    sg = _same_gene_scope(gene_key, is_carrier, qsrc, mask)
                    for method, sim in payload["sim"].items():
                        s_sim, s_car, s_mask = sg(sim)
                        c2 = sw.sweep(
                            s_sim[np.ix_(row, col)], s_car[col],
                            s_mask[np.ix_(row, col)],
                        )
                        b2 = sw.band(c2, tau=args.tau, alpha=args.alpha)
                        summary["matched_genus_bands"].append({
                            "genus": matched_genus, "granularity": gran,
                            "noise": rate, "method": method,
                            "scope": "same_gene",
                            "n_queries": int(row.sum()),
                            "n_carriers": int((is_carrier & col).sum()),
                            "n_confusables": int((~is_carrier & col).sum()),
                            **asdict(b2),
                        })

            # Diagnostics on LOCALE's own matrix, clean sequences only.
            if gran == "gene" and "locale" in payload["sim"]:
                summary.setdefault("near_duplicate_fraction", {})[str(rate)] = \
                    near_duplicate_fraction(payload["sim"]["locale"], is_carrier, mask)
                if rate == 0.0:
                    sus = suspect_label_pairs(
                        payload["sim"]["locale"], is_carrier, mask, qsrc, rows
                    )
                    summary["suspect_label_pairs"] = sus
                    if sus["n_pairs"]:
                        print(f"    [audit] {sus['n_pairs']} carrier-confusable "
                              f"pairs at cosine >= 0.99 "
                              f"({sus['n_carrier_queries_affected']} queries "
                              f"affected) -- check the manifest")

            # Catalog operating points: only at gene granularity for the gene
            # queries, and at window granularity for the read queries -- both
            # are assignments against the same clean catalog.
            if catalogs and "mmseqs_fident" in payload["sim"]:
                # Counted in accessions, matching LOCALE's `element` scope --
                # comparing a gene-counted clustering against an
                # accession-counted embedding would be the wrong denominator.
                for ident, mapping in catalogs.items():
                    pt = baselines.catalog_capture(
                        mapping, ids,
                        payload["sim"]["mmseqs_fident"], qsrc, ident,
                        unit_of_target=entity_of_col,
                        unit_is_carrier=entity_is_carrier,
                    )
                    pt.update({
                        "granularity": gran, "scope": "element", "noise": rate,
                        "method": f"mmseqs_cluster@{ident:.0%}",
                    })
                    pt.update(sw.fj_for_cluster(
                        pt["carrier_capture"], pt["confusable_capture"], assumptions
                    ))
                    # The fair head-to-head: hold LOCALE to the SAME confusable
                    # capture this clustering threshold produces, and ask how
                    # many carriers it pools at that point. Comparing LOCALE's
                    # best radius against a clustering's fixed operating point
                    # would be comparing a curve to a dot.
                    lc = curves_by_method.get("locale")
                    if lc is not None:
                        pt["locale_carrier_at_matched_confusable"] = float(
                            match_carrier_capture(lc, pt["confusable_capture"])
                        )
                        pt["locale_beats_cluster"] = bool(
                            pt["locale_carrier_at_matched_confusable"]
                            > pt["carrier_capture"]
                        )
                    summary["catalog_points"].append(pt)
                    print(f"    {gran:<7} cluster@{ident:.0%}          "
                          f"carrier={pt['carrier_capture']:.3f} "
                          f"confusable={pt['confusable_capture']:.3f} "
                          f"assigned={pt['assigned_fraction']:.2f} "
                          f"clusters={pt['n_carrier_clusters']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(curve_rows).write_csv(out_dir / f"{element}_curves.csv")
    (out_dir / f"{element}_summary.json").write_text(json.dumps(summary, indent=2))
    summary["elapsed_s"] = round(time.time() - t0, 1)
    return summary


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #


def make_plots(curves_csv: Path, out_dir: Path, element: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pl.read_csv(curves_csv)
    written: list[str] = []
    methods = [m for m in ("locale", "kmer31", "mmseqs_fident") if m in
               set(df["method"].to_list())]
    colors = {"locale": "#1f77b4", "kmer31": "#d62728", "mmseqs_fident": "#2ca02c"}

    combos = sorted({
        (g, sc) for g, sc in zip(df["granularity"].to_list(), df["scope"].to_list())
    })
    for gran, scope in combos:
        noises = sorted(set(df["noise"].to_list()))
        fig, axes = plt.subplots(
            2, len(noises), figsize=(4.2 * len(noises), 7.0), squeeze=False
        )
        for col, nz in enumerate(noises):
            ax, axf = axes[0][col], axes[1][col]
            for m in methods:
                d = df.filter(
                    (pl.col("granularity") == gran)
                    & (pl.col("scope") == scope)
                    & (pl.col("noise") == nz)
                    & (pl.col("method") == m)
                ).sort("radius")
                if d.is_empty():
                    continue
                r = d["radius"].to_numpy()
                ax.plot(r, d["carrier_capture_mean"].to_numpy(),
                        color=colors[m], lw=1.8, label=f"{m} carrier")
                ax.plot(r, d["confusable_capture_mean"].to_numpy(),
                        color=colors[m], lw=1.4, ls="--", label=f"{m} confusable")
                ax.fill_between(r, d["carrier_capture_p10"].to_numpy(),
                                d["carrier_capture_p90"].to_numpy(),
                                color=colors[m], alpha=0.12, lw=0)
                axf.plot(r, d["f_j"].to_numpy(), color=colors[m], lw=1.8, label=m)
                axf.plot(r, d["f_contamination"].to_numpy(), color=colors[m],
                         lw=1.2, ls=":")
            ax.set_title(f"{element} / {gran} / {scope} / noise {nz:.0%}")
            ax.set_xlabel("radius (similarity)")
            ax.set_ylabel("capture rate")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.25)
            axf.set_xlabel("radius (similarity)")
            axf.set_ylabel("implied $f_j$ (solid) / contamination (dotted)")
            axf.grid(alpha=0.25)
            if col == 0:
                ax.legend(fontsize=6, loc="upper right")
                axf.legend(fontsize=6, loc="upper right")
        fig.suptitle(
            f"{element}: solid = carriers captured, dashed = confusables "
            f"captured (shaded = p10-p90 over queries)", fontsize=9
        )
        fig.tight_layout()
        p = out_dir / f"{element}_{gran}_{scope}.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(p.name)
    return written


# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequences", type=Path, default=HERE / "data" / "sequences.parquet")
    ap.add_argument("--results-dir", type=Path, default=HERE / "results")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--checkpoint", default=None,
                    help="default: the paper checkpoint from the Hub")
    ap.add_argument("--element", action="append", default=None)
    ap.add_argument("--noise", type=float, nargs="*", default=list(NOISE_RATES))
    ap.add_argument("--convention", default="repo",
                    choices=list(noisemod.CONVENTIONS))
    ap.add_argument("--windows-per-gene", type=int, default=3,
                    help="read-level query windows sampled per carrier gene")
    ap.add_argument("--tau", type=float, default=sw.TAU_DEFAULT)
    ap.add_argument("--alpha", type=float, default=sw.ALPHA_DEFAULT)
    ap.add_argument("--identity-check", type=int, default=25,
                    help="sequences per rate to measure realized identity on")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-mmseqs", action="store_true")
    ap.add_argument("--no-protein", action="store_true",
                    help="skip the UHGP-style protein clustering point")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if not args.sequences.exists():
        raise SystemExit(
            f"{args.sequences} not found -- run data/fetch_elements.py first"
        )
    df = pl.read_parquet(args.sequences)
    elements = list(dict.fromkeys(df["element"].to_list()))
    if args.element:
        elements = [e for e in elements if e in set(args.element)]

    runtime = emb.describe_runtime()
    runtime["mmseqs"] = baselines.mmseqs_bin()
    runtime["device_requested"] = args.device
    print(json.dumps(runtime, indent=2))
    if runtime["attention_path"].startswith("native"):
        print("[note] flash-attn is NOT installed: attention is on the model's "
              "native ALiBi fallback. Embeddings are valid; fp16 numerics can "
              "differ slightly from the published runs.")

    encoder = emb.load_encoder(
        emb.encoder_config(
            device=args.device, batch_size=args.batch_size,
            checkpoint_path=args.checkpoint,
        )
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for el in elements:
        rows = df.filter(pl.col("element") == el)
        s = run_element(el, rows, encoder, args, args.results_dir)
        if not args.no_plots:
            try:
                s["plots"] = make_plots(
                    args.results_dir / f"{el}_curves.csv", args.results_dir, el
                )
            except Exception as exc:
                print(f"[warn] plotting failed for {el}: {exc}")
        summaries.append(s)

    out = {
        "runtime": runtime,
        "settings": {
            "noise_rates": args.noise,
            "noise_convention": args.convention,
            "tau": args.tau,
            "alpha": args.alpha,
            "windows_per_gene": args.windows_per_gene,
            "seed": args.seed,
            "cluster_identities": list(CLUSTER_IDENTITIES),
            "window_bp": emb.WINDOW,
            "chunk_overlap_bp": emb.OVERLAP,
        },
        "elements": summaries,
    }
    (args.results_dir / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.results_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
