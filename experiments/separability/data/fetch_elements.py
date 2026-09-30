"""Assemble the carrier and confusable sequence sets from NCBI nuccore.

Reads data/elements.yaml (what we want), writes data/cache/*.gb (raw records,
gitignored), data/sequences.parquet (the extracted CDS table) and
data/manifest.yaml (what we actually got, accession by accession).

Why CDS extraction rather than downloading gene records directly: the three
elements live at different granularities in GenBank. blaCTX-M has thousands of
dedicated ~900 bp CDS records; the pks island mostly exists inside 54 kb island
entries and plasmid contigs. Pulling CDS features out of whatever record came
back normalises all of that to one unit -- a gene -- and hands us the organism
and coordinates for free, which is what the manifest needs.

Network only: no model, no GPU. Safe to run on a login node.

    uv run python experiments/separability/data/fetch_elements.py
"""

import argparse
import hashlib
import os
import re
import sys
import time
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path

import polars as pl
import yaml
from Bio import Entrez, SeqIO
from Bio.Seq import UndefinedSequenceError

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"

# One full model window. A CDS shorter than this cannot fill even one window
# (EncoderConfig.max_seq_len = 256) and is scored on a pad-heavy embedding of a
# gene fragment -- contig-edge truncations as short as 162 bp were getting in at
# the old 150 bp threshold and standing in for whole genes.
MIN_CDS_LEN = 256
# Fraction of the CDS that must be unambiguous ACGT. Records padded with N runs
# produce embeddings of the padding, not of the gene.
MIN_ACGT_FRAC = 0.99
# Caps keep one over-sequenced allele (blaCTX-M-15, say) from dominating a set
# and making within-carrier divergence look artificially small. They are
# applied to whole ACCESSIONS, never to individual sequences: dropping some of
# an accession's genes would make it look like a strain that lacks them, and
# the element scope would then measure our subsampling instead of the
# embedding. Accessions carrying more of the element are kept first, for the
# same reason.
CAP_ACCESSIONS_PER_GENUS = 90
CAP_PER_GROUP = 1500

_ACGT = set("ACGT")
# A bare gene symbol: clbb, irp1, cdta, entf, pltb.
_SYMBOL_RE = re.compile(r"^[a-z]{2,5}[a-z0-9]?$")


def gene_symbol_key(symbol: str, product: str, group: str) -> str:
    """Normalise a CDS to the gene it is, for the same-gene scope.

    Many records leave /gene empty and put everything in /product, so the raw
    symbol can be "cytolethal distending toxin subunit B family protein". Left
    alone those become their own gene, which fragments a 3-gene element into 15
    and makes the same-gene carrier sets far too small. The fallbacks recover
    the subunit letter and rebuild the symbol from the group prefix; anything
    still ambiguous collapses to the group, which is the conservative choice
    (it widens the carrier set rather than inventing a distinction).
    """
    t = symbol.strip().lower()
    if _SYMBOL_RE.match(t):
        return t
    blob = f"{symbol} {product}".lower()
    m = re.search(rf"\b{re.escape(group)}\s*([a-z0-9])\b", blob)
    if m:
        return f"{group}{m.group(1)}"
    m = re.search(r"\bsubunit\s+([a-z0-9])\b", blob)
    if m:
        return f"{group}{m.group(1)}"
    return group


def _entrez_setup(api_key: str | None) -> None:
    # NCBI asks for an identifying email and raises the rate limit with an API
    # key. Both come from the environment so this script never hardcodes a
    # personal address.
    email = os.environ.get("NCBI_EMAIL", "").strip()
    if email:
        Entrez.email = email
    else:
        print(
            "[warn] NCBI_EMAIL is unset. NCBI asks for a contact address and "
            "throttles anonymous callers harder. export NCBI_EMAIL=you@example.org",
            file=sys.stderr,
        )
    key = api_key or os.environ.get("NCBI_API_KEY", "").strip()
    if key:
        Entrez.api_key = key


def _sleep(api_key_set: bool) -> None:
    # 10 req/s with a key, 3 without. Stay comfortably under either.
    time.sleep(0.12 if api_key_set else 0.4)


def _cache_path(element: str, term: str) -> Path:
    h = hashlib.sha1(term.encode()).hexdigest()[:12]
    return CACHE / f"{element}.{h}.gb"


def fetch_term(element: str, term: str, retmax: int, batch: int = 10) -> str:
    """GenBank text for one search term, cached on disk by term hash."""
    path = _cache_path(element, term)
    if path.exists() and path.stat().st_size > 0:
        return path.read_text()

    key_set = bool(getattr(Entrez, "api_key", None))
    with Entrez.esearch(db="nuccore", term=term, retmax=retmax) as h:
        ids = Entrez.read(h)["IdList"]
    _sleep(key_set)
    print(f"    {len(ids):>4} records  <- {term}")
    if not ids:
        path.write_text("")
        return ""

    parts: list[str] = []
    n_lost = 0

    def _efetch(chunk: list[str]) -> None:
        """Fetch one chunk, halving it on failure down to single records.

        A pks island entry can be 150 kb, so 20 of them in one efetch reliably
        trips NCBI's response limit and comes back as an IncompleteRead. Rather
        than pick a batch size that is safe for the largest record and slow for
        every other element, split on failure: big records end up fetched one at
        a time and small ones stay batched.
        """
        nonlocal n_lost
        for attempt in range(3):
            try:
                with Entrez.efetch(
                    db="nuccore", id=",".join(chunk),
                    rettype="gbwithparts", retmode="text",
                ) as h:
                    text = h.read()
                # A truncated body still parses as "some records", which would
                # silently shrink the set. Require the terminator.
                if chunk and not text.rstrip().endswith("//"):
                    raise IOError("truncated GenBank response")
                parts.append(text)
                _sleep(key_set)
                return
            except Exception as exc:
                last = exc
                time.sleep(1.5 * (attempt + 1))
        if len(chunk) == 1:
            n_lost += 1
            print(f"    [warn] dropped {chunk[0]}: {last}", file=sys.stderr)
            return
        mid = len(chunk) // 2
        _efetch(chunk[:mid])
        _efetch(chunk[mid:])

    for i in range(0, len(ids), batch):
        _efetch(ids[i : i + batch])
    if n_lost:
        print(f"    [warn] {n_lost}/{len(ids)} records unavailable", file=sys.stderr)

    text = "".join(parts)
    path.write_text(text)
    return text


def _clean(seq: str) -> str | None:
    s = str(seq).upper()
    if len(s) < MIN_CDS_LEN:
        return None
    good = sum(1 for c in s if c in _ACGT)
    if good / len(s) < MIN_ACGT_FRAC:
        return None
    return s


def harvest(element: dict, refetch: bool = False) -> list[dict]:
    """All CDS features from this element's searches that match a group."""
    name = element["name"]
    slen = element.get("slen", "")
    retmax = int(element.get("retmax", 200))
    batch = int(element.get("batch", 10))

    # Compile once; a CDS is tested against gene and product both.
    groups = []
    for g in element["groups"]:
        groups.append({
            "label": g["label"],
            "group": g["name"],
            "patterns": [re.compile(p, re.I) for p in g["patterns"]],
            "genera": set(g.get("genera") or []),
            "gene_key": g.get("gene_key", "symbol"),
        })

    # Search terms are shared across groups only in that they all feed the same
    # record pool: a record found by a clb search may well contain ybt CDSs, and
    # we want those -- they are the confusables that actually co-occur.
    texts = []
    for g in element["groups"]:
        for term in g["terms"]:
            t = term.format(slen=slen)
            if refetch:
                _cache_path(name, t).unlink(missing_ok=True)
            texts.append(fetch_term(name, t, retmax, batch))

    rows: list[dict] = []
    seen_records: set[str] = set()
    for text in texts:
        if not text.strip():
            continue
        for rec in SeqIO.parse(StringIO(text), "genbank"):
            if rec.id in seen_records:
                continue
            seen_records.add(rec.id)
            organism = rec.annotations.get("organism", "").strip()
            genus = organism.split()[0] if organism else ""
            try:
                whole = str(rec.seq).upper()
            except UndefinedSequenceError:
                # CON / master records carry features but no bases; efetch
                # returns them even with rettype=gbwithparts.
                continue
            if not whole or set(whole) <= {"N"}:
                continue
            for feat in rec.features:
                if feat.type != "CDS":
                    continue
                gene = (feat.qualifiers.get("gene", [""])[0] or "").strip()
                product = (feat.qualifiers.get("product", [""])[0] or "").strip()
                for grp in groups:
                    if grp["genera"] and genus not in grp["genera"]:
                        continue
                    # Patterns are tested against /gene and /product
                    # SEPARATELY, never against their concatenation: an
                    # anchored symbol pattern like ^cdtB$ can only ever match
                    # the gene field, and matching it against "cdtB cytolethal
                    # distending toxin S-CDT" silently matches nothing -- which
                    # is how the typhoid-toxin confusable set came back empty.
                    if not any(
                        p.search(gene) or p.search(product)
                        for p in grp["patterns"]
                    ):
                        continue
                    try:
                        seq = _clean(feat.extract(rec.seq))
                    except Exception:
                        seq = None
                    if seq is None:
                        break
                    loc = feat.location
                    symbol = (gene or product[:40]).strip()
                    rows.append({
                        "element": name,
                        "label": grp["label"],
                        "group": grp["group"],
                        "gene": symbol,
                        # What counts as "the same gene" downstream; see the
                        # gene_key note in elements.yaml.
                        "gene_key": gene_symbol_key(
                            symbol, product, grp["group"]
                        )
                        if grp["gene_key"] == "symbol"
                        else grp["group"],
                        "product": product,
                        "organism": organism,
                        "genus": genus,
                        "accession": rec.id,
                        "start": int(loc.start),
                        "end": int(loc.end),
                        "strand": int(loc.strand or 0),
                        "seq_len": len(seq),
                        "sequence": seq,
                    })
                    break  # first matching group wins; groups are disjoint by design
    return rows


def dedupe_and_cap(rows: list[dict], seed: int = 0) -> list[dict]:
    """Drop exact-duplicate sequences, then cap by accession.

    The dedupe is what stops a single allele deposited a hundred times from
    setting the within-carrier divergence floor. The cap is applied at
    accession granularity and prefers accessions that carry more of the
    element, so a carrier accession is either kept whole or not kept at all --
    a half-kept island would be indistinguishable from a strain that only has
    half the island.
    """
    import random

    rng = random.Random(seed)
    by_seq: dict[str, dict] = {}
    for r in rows:
        by_seq.setdefault(r["sequence"], r)
    uniq = list(by_seq.values())

    # (group, accession) -> its sequences
    by_acc: dict[tuple, list[dict]] = defaultdict(list)
    for r in uniq:
        by_acc[(r["group"], r["accession"])].append(r)

    # Per (group, genus): keep at most CAP_ACCESSIONS_PER_GENUS accessions,
    # richest first, ties broken deterministically at random.
    by_group_genus: dict[tuple, list[tuple]] = defaultdict(list)
    for key, seqs in by_acc.items():
        by_group_genus[(key[0], seqs[0]["genus"])].append(key)

    kept_accs: list[tuple] = []
    for gg in sorted(by_group_genus):
        accs = by_group_genus[gg]
        rng.shuffle(accs)
        accs.sort(key=lambda k: -len({r["gene_key"] for r in by_acc[k]}))
        kept_accs.extend(accs[:CAP_ACCESSIONS_PER_GENUS])

    # Per group: fill a sequence budget with whole accessions, richest first.
    per_group: dict[str, list[tuple]] = defaultdict(list)
    for key in kept_accs:
        per_group[key[0]].append(key)

    out: list[dict] = []
    for group in sorted(per_group):
        accs = per_group[group]
        rng.shuffle(accs)
        accs.sort(key=lambda k: -len({r["gene_key"] for r in by_acc[k]}))
        budget = CAP_PER_GROUP
        for key in accs:
            seqs = by_acc[key]
            if len(seqs) > budget:
                continue
            out.extend(seqs)
            budget -= len(seqs)
            if budget <= 0:
                break
    out.sort(key=lambda r: (r["group"], r["gene"], r["accession"], r["start"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, default=HERE / "elements.yaml")
    ap.add_argument("--out", type=Path, default=HERE / "sequences.parquet")
    ap.add_argument("--manifest", type=Path, default=HERE / "manifest.yaml")
    ap.add_argument("--api-key", default=None, help="NCBI API key (or NCBI_API_KEY)")
    ap.add_argument("--refetch", action="store_true", help="ignore the cache")
    ap.add_argument("--element", action="append", default=None,
                    help="limit to these element names (repeatable)")
    args = ap.parse_args()

    _entrez_setup(args.api_key)
    CACHE.mkdir(parents=True, exist_ok=True)

    spec = yaml.safe_load(args.spec.read_text())
    elements = spec["elements"]
    if args.element:
        elements = [e for e in elements if e["name"] in set(args.element)]
        if not elements:
            raise SystemExit(f"no element matched {args.element}")

    all_rows: list[dict] = []
    manifest = {
        "generated_by": "experiments/separability/data/fetch_elements.py",
        "source": "NCBI nuccore via Entrez E-utilities",
        "filters": {
            "min_cds_len": MIN_CDS_LEN,
            "min_acgt_frac": MIN_ACGT_FRAC,
            "cap_accessions_per_genus": CAP_ACCESSIONS_PER_GENUS,
            "cap_sequences_per_group": CAP_PER_GROUP,
            "capping": "whole accessions, richest in element genes first",
            "exact_duplicate_sequences": "dropped",
        },
        "elements": [],
    }

    for el in elements:
        print(f"[{el['name']}] {el['title']}")
        rows = harvest(el, refetch=args.refetch)
        rows = dedupe_and_cap(rows)
        all_rows.extend(rows)

        by_group: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_group[r["group"]].append(r)

        entry = {
            "name": el["name"],
            "title": el["title"],
            "search_slen": el.get("slen"),
            "groups": [],
        }
        for g in el["groups"]:
            sel = by_group.get(g["name"], [])
            genera = sorted({r["genus"] for r in sel if r["genus"]})
            entry["groups"].append({
                "name": g["name"],
                "label": g["label"],
                "gene_key": g.get("gene_key", "symbol"),
                "n_sequences": len(sel),
                "n_accessions": len({r["accession"] for r in sel}),
                "genera": genera,
                "n_genera": len(genera),
                "genes": sorted({r["gene"] for r in sel}),
                "median_len": int(sorted(r["seq_len"] for r in sel)[len(sel) // 2])
                if sel else 0,
                "search_terms": [t.format(slen=el.get("slen", "")) for t in g["terms"]],
                "accessions": sorted({r["accession"] for r in sel}),
            })
            flag = "" if len(genera) >= 2 or g["label"] == "confusable" else \
                "  <-- SINGLE GENUS: cross-species pooling UNTESTED here"
            print(f"    {g['label']:<10} {g['name']:<14} "
                  f"n={len(sel):<5} genera={len(genera)}{flag}")
            # Print the products actually matched. An over-broad pattern shows
            # up here as a generic annotation ("class A beta-lactamase") sitting
            # in a group that is supposed to be one family, which is how four
            # TEM-like sequences were once found inside the CTX-M carriers.
            prods = Counter(r["product"][:52] for r in sel)
            for pr, n in prods.most_common(4):
                print(f"         {n:>4}  {pr!r}")
        manifest["elements"].append(entry)

    if not all_rows:
        raise SystemExit("nothing harvested -- check network and search terms")

    df = pl.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    args.manifest.write_text(yaml.safe_dump(manifest, sort_keys=False, width=100))
    print(f"\nwrote {len(df)} sequences -> {args.out}")
    print(f"wrote manifest             -> {args.manifest}")


if __name__ == "__main__":
    main()
