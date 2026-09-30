#!/usr/bin/env python
"""Render results/summary.json as the markdown tables report.md quotes.

Kept separate from run_probe.py so the report can be regenerated without
re-embedding anything, and so every number in report.md has a command that
reproduces it:

    uv run python experiments/separability/make_tables.py > /tmp/tables.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fmt(x, nd=3):
    if x is None:
        return "--"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        if x != x:  # NaN
            return "--"
        return f"{x:.{nd}f}"
    return str(x)


def sets_table(elements: list[dict]) -> str:
    lines = [
        "| element | carriers | carrier genera | confusables | confusable genera | ref. windows | near-dup pairs |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in elements:
        nd = (e.get("near_duplicate_fraction") or {}).get("0.0")
        lines.append(
            f"| `{e['element']}` | {e['n_carriers']} | "
            f"{len(e['carrier_genera'])} ({', '.join(e['carrier_genera'][:4])}"
            f"{'...' if len(e['carrier_genera']) > 4 else ''}) | "
            f"{e['n_confusables']} | {len(e['confusable_genera'])} | "
            f"{e['n_reference_windows']} | "
            f"{fmt(nd) if nd is not None else '--'} |"
        )
    return "\n".join(lines)


def band_table(e: dict, gran: str, scope: str, methods: list[str]) -> str:
    rows = [b for b in e["bands"] if b["granularity"] == gran
            and b.get("scope") == scope and b["method"] in methods]
    if not rows:
        return "_(no results)_"
    noises = sorted({b["noise"] for b in rows})
    lines = [
        "| method | " + " | ".join(f"noise {n:.0%}" for n in noises) + " |",
        "|---" * (len(noises) + 1) + "|",
    ]
    for m in methods:
        cells = []
        for n in noises:
            b = next((r for r in rows if r["method"] == m and r["noise"] == n), None)
            cells.append(
                f"{fmt(b['auroc'], 3)} / {fmt(b['carrier_capture_at_lo'], 3)}"
                if b else "--"
            )
        lines.append(f"| `{m}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def catalog_table(e: dict, gran: str) -> str:
    rows = [c for c in e["catalog_points"] if c["granularity"] == gran]
    if not rows:
        return "_(no clustering results)_"
    noises = sorted({c["noise"] for c in rows})
    idents = sorted({c["min_seq_id"] for c in rows}, reverse=True)
    lines = [
        "| catalog | noise | assigned | carrier capture | confusable capture "
        "| carrier clusters | LOCALE @ same confusable | LOCALE wins |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ident in idents:
        for n in noises:
            c = next((r for r in rows
                      if r["min_seq_id"] == ident and r["noise"] == n), None)
            if not c:
                continue
            lines.append(
                f"| mmseqs @ {ident:.0%} | {n:.0%} | "
                f"{fmt(c['assigned_fraction'], 2)} | "
                f"{fmt(c['carrier_capture'])} | {fmt(c['confusable_capture'])} | "
                f"{c['n_carrier_clusters']} | "
                f"{fmt(c.get('locale_carrier_at_matched_confusable'))} | "
                f"{fmt(c.get('locale_beats_cluster'))} |"
            )
    return "\n".join(lines)


def matched_table(e: dict) -> str:
    rows = e.get("matched_genus_bands") or []
    if not rows:
        return f"_(not available: {e.get('matched_genus_note', 'no matched genus')})_"
    methods = ["locale", "kmer31", "mmseqs_fident"]
    noises = sorted({r["noise"] for r in rows})
    lines = [
        f"Genus: **{e['matched_genus']}** "
        f"({rows[0]['n_carriers']} carriers / {rows[0]['n_confusables']} confusables). "
        "Cells are AUROC / carrier capture at <=1% confusable.",
        "",
        "| granularity | method | " + " | ".join(f"noise {n:.0%}" for n in noises) + " |",
        "|---|---" + "|---" * len(noises) + "|",
    ]
    for gran in ("gene", "window"):
        for m in methods:
            cells = []
            for n in noises:
                b = next((r for r in rows if r["granularity"] == gran
                          and r["method"] == m and r["noise"] == n), None)
                cells.append(
                    f"{fmt(b['auroc'], 3)} / {fmt(b['carrier_capture_at_lo'], 3)}"
                    if b else "--"
                )
            lines.append(f"| {gran} | `{m}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fj_table(e: dict, gran: str, scope: str) -> str:
    a = e["fj_assumptions"]
    rows = [b for b in e["bands"] if b["granularity"] == gran
            and b.get("scope") == scope]
    noises = sorted({b["noise"] for b in rows})
    lines = [
        f"P(element) = {a['element_prevalence']}, "
        f"P(confusable) = {a['confusable_prevalence']}.",
        "",
        "| method | " + " | ".join(f"f_j @ noise {n:.0%}" for n in noises) + " |",
        "|---" * (len(noises) + 1) + "|",
    ]
    for m in ("locale", "kmer31", "mmseqs_fident"):
        cells = []
        for n in noises:
            b = next((r for r in rows if r["method"] == m and r["noise"] == n), None)
            if b and b["carrier_capture_at_lo"] == b["carrier_capture_at_lo"]:
                cells.append(fmt(a["element_prevalence"] * b["carrier_capture_at_lo"]))
            else:
                cells.append("--")
        lines.append(f"| `{m}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def decision_table(elements: list[dict]) -> str:
    """The comparisons the verdict turns on, one row per element.

    Deliberately narrow. The full tables below show everything; this shows the
    cells that decide GO / NO-GO: separability under the heaviest noise (where
    LOCALE has to win if it wins anywhere), and pooling against the best
    identity catalog (where fragmentation decides f_j).
    """
    lines = [
        "Read at 10% query noise -- the regime the claim lives in. "
        "`best id.` is the best of the 31-mer and MMseqs2 columns.",
        "",
        "| element | scope | AUROC LOCALE | AUROC best id. | carrier@conf<=1% LOCALE "
        "| carrier@conf<=1% best id. | best catalog carrier capture "
        "| LOCALE @ same confusable | LOCALE pools more |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in elements:
        scopes = sorted(
            {b.get("scope") for b in e["bands"] if b.get("scope")},
            key=lambda x: ("same_gene", "element").index(x)
            if x in ("same_gene", "element") else 2,
        )
        for scope in scopes:
            rows = [
                b for b in e["bands"]
                if b.get("scope") == scope and b["granularity"] == "window"
                and b["noise"] == max(x["noise"] for x in e["bands"])
            ]
            if not rows:
                continue

            def get(m, key):
                r = next((x for x in rows if x["method"] == m), None)
                return r[key] if r else float("nan")

            loc_auc, loc_cap = get("locale", "auroc"), get("locale", "carrier_capture_at_lo")
            ids = [
                (get(m, "auroc"), get(m, "carrier_capture_at_lo"))
                for m in ("kmer31", "mmseqs_fident", "mmseqs_fident_x_qcov")
            ]
            ids = [(a, c) for a, c in ids if a == a]
            best_auc = max((a for a, _ in ids), default=float("nan"))
            best_cap = max((c for _, c in ids if c == c), default=float("nan"))

            cats = [
                c for c in e["catalog_points"]
                if c["granularity"] == "window"
                and c["noise"] == max(x["noise"] for x in e["catalog_points"])
            ] if e.get("catalog_points") else []
            best_cat = max(cats, key=lambda c: c["carrier_capture"], default=None)
            cat_cap = best_cat["carrier_capture"] if best_cat else float("nan")
            loc_matched = (
                best_cat.get("locale_carrier_at_matched_confusable")
                if best_cat else None
            )
            wins = (
                loc_matched is not None and loc_matched == loc_matched
                and cat_cap == cat_cap and loc_matched > cat_cap
            )
            lines.append(
                f"| `{e['element']}` | {scope} | {fmt(loc_auc)} | {fmt(best_auc)} "
                f"| {fmt(loc_cap)} | {fmt(best_cap)} "
                f"| {fmt(cat_cap)} (@{best_cat['min_seq_id']:.0%}) "
                f"| {fmt(loc_matched)} | {fmt(wins)} |"
                if best_cat else
                f"| `{e['element']}` | {scope} | {fmt(loc_auc)} | {fmt(best_auc)} "
                f"| {fmt(loc_cap)} | {fmt(best_cap)} | -- | -- | -- |"
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, default=HERE / "results" / "summary.json")
    args = ap.parse_args()
    d = json.loads(args.summary.read_text())

    print("## Runtime\n")
    for k, v in d["runtime"].items():
        print(f"- `{k}`: {v}")
    print("\n## Settings\n")
    for k, v in d["settings"].items():
        print(f"- `{k}`: {v}")

    print("\n## Set sizes\n")
    print(sets_table(d["elements"]))

    print("\n## Decision table\n")
    print(decision_table(d["elements"]))

    methods = ["locale", "kmer31", "mmseqs_fident", "mmseqs_fident_x_qcov"]
    for e in d["elements"]:
        print(f"\n\n# {e['element']}\n")
        if e.get("warning"):
            print(f"> **{e['warning']}**\n")
        print(f"\nCarrier units: {e.get('n_carrier_accessions')} accessions, "
              f"{e['n_carriers']} genes across "
              f"{len(e.get('carrier_gene_keys') or [])} gene(s). "
              f"Confusable-only accessions: "
              f"{e.get('n_confusable_accessions')}; accessions holding both "
              f"(counted as carriers): {e.get('n_accessions_with_both')}.\n")
        scopes = sorted(
            {b.get("scope") for b in e["bands"] if b.get("scope")},
            key=lambda x: ("same_gene", "element").index(x)
            if x in ("same_gene", "element") else 2,
        )
        for gran in ("gene", "window"):
            for scope in scopes:
                print(f"\n### {gran} granularity, {scope} scope -- "
                      f"AUROC / carrier capture at <=1% confusable capture\n")
                print(band_table(e, gran, scope, methods))
                print("\n#### implied f_j\n")
                print(fj_table(e, gran, scope))
            print(f"\n### {gran} granularity -- identity catalogs "
                  f"(element scope, accessions)\n")
            print(catalog_table(e, gran))
        if e.get("protein_catalog_points"):
            print("\n### UHGP-style protein catalog (clean sequences only)\n")
            print("| catalog | carrier clusters | largest holds | "
                  "confusables in it | all carriers in one cluster |")
            print("|---|---|---|---|---|")
            for pt in e["protein_catalog_points"]:
                print(f"| {pt['catalog']} | {pt['n_carrier_clusters']} | "
                      f"{pt['largest_carrier_cluster']} "
                      f"({fmt(pt['largest_carrier_cluster_share'], 2)}) | "
                      f"{pt['confusables_in_largest_carrier_cluster']} | "
                      f"{fmt(pt['all_carriers_in_one_cluster'])} |")
        print("\n### same-genus control\n")
        print(matched_table(e))


if __name__ == "__main__":
    main()
