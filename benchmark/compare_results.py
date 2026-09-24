"""Compare two run_benchmark result files query by query.

Usage: python compare_results.py <reference.parquet> <candidate.parquet>

Meant for checks that must agree exactly, such as a sharded metagraph index
against the single-node one. For each query_id present in both files the
(accession, score) lists are compared as sets. Lists cut at the same top-k can
legitimately differ in which tied accessions made the cut, so a mismatch is
tolerated when every differing accession scores exactly the lowest retained
score of that query. Exits 1 on any other difference.
"""

import sys
from pathlib import Path

import polars as pl


def hits(df: pl.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for query_id, results in df.select("query_id", "results").iter_rows():
        out[query_id] = {r["accession"]: r["score"] for r in results}
    return out


def main(reference: Path, candidate: Path) -> int:
    ref = hits(pl.read_parquet(reference))
    cand = hits(pl.read_parquet(candidate))
    common = sorted(set(ref) & set(cand))
    print(f"{len(ref)} reference queries, {len(cand)} candidate, {len(common)} in common")
    if not common:
        print("[FAIL] no query_id overlap")
        return 1

    bad = 0
    tie_only = 0
    for qid in common:
        a, b = ref[qid], cand[qid]
        if a == b:
            continue
        # Shared accessions must score identically.
        shared = set(a) & set(b)
        if any(a[acc] != b[acc] for acc in shared):
            bad += 1
            print(f"[diff] {qid}: scores differ on shared accessions")
            continue
        # Extra/missing accessions may only be ties at the cut.
        floor = min(min(a.values(), default=0.0), min(b.values(), default=0.0))
        diff = (set(a) ^ set(b))
        if len(a) != len(b) or any((a.get(acc) or b.get(acc)) != floor for acc in diff):
            bad += 1
            print(f"[diff] {qid}: {len(a)} vs {len(b)} hits; non-tie differences {sorted(diff)[:5]}")
            continue
        tie_only += 1

    print(f"{len(common) - bad - tie_only} identical, {tie_only} differ only in ties at the cut, {bad} real differences")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
