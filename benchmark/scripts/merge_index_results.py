"""Union the results of one query set searched against two indexes.

run_sra64viral.sh searches the sra64viral-v2 queries against the 64-run index
(results/sra64viral) and, with INDEX=sra500, against the existing sra500
indexes (results/sra64viral_sra500idx). This merges the two into the results
of a single index holding both accession sets, without building it:

  dense (exact top-k regroup): the two top-k vector hit lists are unioned and
      the global top-k re-taken (src.topk_regroup.merge_topk_hits, the same
      function that merges per-node shards), then regrouped per accession.
      Both scans are exact, so this equals one exact scan over both indexes.
  metagraph / mmseqs: per-accession scores are unioned, keeping the best score
      of an accession present in both, and the top `max_labels` kept.
      MetaGraph's k-mer counts do not depend on the other labels, so this is
      exact; MMseqs2 bit scores do not either, but its prefilter (max_seqs)
      and e-value cut-off were applied per index, so it is an approximation.

A model directory is paired across the two results trees by its path; `strip`
removes label suffixes that differ between them (the sra500 LOCALE index was
built with a contig-length cutoff, "-minlen100").

Usage:
    uv run python scripts/merge_index_results.py
    uv run python scripts/print_results.py results/sra64viral_union \
        <bundle>/queries_mut0.00.parquet results/sra64viral_union/accs.txt
"""

import json
import shutil
import sys
from pathlib import Path

import polars as pl
from jsonargparse import auto_cli

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.topk_regroup import merge_topk_hits, regroup_topk_hits  # noqa: E402

B = "/pscratch/sd/r/rsynk/locale-data/constructed/sra64viral-v2"


def read_accs(path: Path) -> list[str]:
    return Path(path).read_text().split()


def merge_scored(a: pl.DataFrame, b: pl.DataFrame, max_labels: int) -> pl.DataFrame:
    """Union two per-accession results frames over the same queries."""
    flat = (
        pl.concat([a.select("query_id", "results"), b.select("query_id", "results")])
        .explode("results")
        .drop_nulls("results")
        .unnest("results")
        .group_by("query_id", "accession")
        .agg(pl.col("score").max())
        .sort(["query_id", "score", "accession"], descending=[False, True, False])
        .group_by("query_id", maintain_order=True)
        .head(max_labels)
        .group_by("query_id", maintain_order=True)
        .agg(pl.struct("accession", "score").alias("results"))
    )
    dtype = a.schema["results"]
    return (
        a.select("query_id")
        .join(flat, on="query_id", how="left")
        .with_columns(pl.col("results").fill_null(pl.lit([], dtype=dtype)).cast(dtype))
    )


def main(
    results_a: str = "results/sra64viral",
    results_b: str = "results/sra64viral_sra500idx",
    accs_a: str = f"{B}/bundle/accs.txt",
    accs_b: str = f"{B}/bundle_sra500idx/accs.txt",
    output: str = "results/sra64viral_union",
    strip: list[str] = ["-minlen100"],
    max_labels: int = 100,
):
    """
    Args:
        results_a: results tree whose model labels the output keeps.
        results_b: results tree of the same queries against the other index.
        accs_a: accession list of the first index.
        accs_b: accession list of the second index.
        output: merged results tree; also receives the union accs.txt.
        strip: label substrings removed from results_b paths before pairing.
        max_labels: accessions kept per query for metagraph / mmseqs.
    """
    results_a, results_b, output = Path(results_a), Path(results_b), Path(output)
    accs = list(dict.fromkeys(read_accs(accs_a) + read_accs(accs_b)))
    output.mkdir(parents=True, exist_ok=True)
    (output / "accs.txt").write_text("".join(a + "\n" for a in accs))

    def key(root: Path, f: Path) -> str:
        rel = str(f.relative_to(root))
        for s in strip:
            rel = rel.replace(s, "")
        return rel

    files_b = {
        key(results_b, f): f
        for f in results_b.rglob("mut*.parquet")
        if "hits" not in f.parts and not any(p.startswith("partials_") for p in f.parts)
    }
    merged = 0
    for fa in sorted(results_a.rglob("mut*.parquet")):
        if "hits" in fa.parts or any(p.startswith("partials_") for p in fa.parts):
            continue
        rel = key(results_a, fa)
        fb = files_b.get(rel)
        if fb is None:
            print(f"[skip] {rel}: no counterpart under {results_b}")
            continue
        a, b = pl.read_parquet(fa), pl.read_parquet(fb)
        assert set(a["query_id"]) == set(b["query_id"]), f"{rel}: query sets differ"
        b = a.select("query_id").join(b, on="query_id", how="left")
        hits_a, hits_b = fa.parent / "hits" / fa.name, fb.parent / "hits" / fb.name
        if hits_a.exists() and hits_b.exists():
            top_k = json.loads((fa.parent / "config.json").read_text())["top_k"]
            ha = pl.read_parquet(hits_a, columns=["query_id", "hits"])
            hb = pl.read_parquet(hits_b, columns=["query_id", "hits"])
            ha = a.select("query_id").join(ha, on="query_id", how="left")
            hits = merge_topk_hits([ha, hb], top_k)
            results = regroup_topk_hits(hits, accs).select("results")
            new = a.with_columns(results["results"].alias("results"))
            how = f"dense top-{top_k} hits"
        else:
            new = a.drop("results").join(
                merge_scored(a, b, max_labels), on="query_id", how="left"
            ).select(a.columns)
            how = "per-accession scores"
        if "index_size_gb" in new.columns:
            new = new.with_columns(
                (a["index_size_gb"] + b["index_size_gb"]).alias("index_size_gb")
            )
        out = output / fa.relative_to(results_a)
        out.parent.mkdir(parents=True, exist_ok=True)
        new.write_parquet(out)
        shutil.copy(fa.parent / "config.json", out.parent / "config.json")
        merged += 1
        print(f"[ok] {rel}: merged {how} over {len(accs)} accessions")
    print(f"Merged {merged} result files into {output}")


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
