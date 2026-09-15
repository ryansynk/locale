"""Re-score saved top-k vector hits at smaller k, without re-searching.

An exact_search run (run_benchmark.py with model.exact_search: true) writes,
per mutation rate, each query's exact global top-K nearest index vectors as a
score-descending list under topk_hits_dir. Because the top-k-then-regroup
ranking at any k <= K depends only on the first k entries of that list, every
smaller k is a truncation: take the first k hits, regroup to accessions by max
score, emit every accession with the miss sentinel. This writes one standard
results parquet per (hits file, k), each under its own experiment id
(<experiment_id>_k<k>) so print_results.py reports them as separate models.

Usage:
    uv run python rescore_topk.py \\
        results_100studies_topk_hits/locale_..._exacttop1000 \\
        results_100studies_regroup \\
        /pscratch/sd/r/rsynk/locale-data/100studies_work/bundle/accs.txt \\
        --ks "[10,20,50,100,200,1000]"

    then: uv run python print_results.py results_100studies_regroup \\
              <bundle>/queries.parquet <bundle>/accs.txt

hits_path may be one *_topk<K>.parquet file or a directory (searched
recursively). k = K reproduces the run's own results parquet exactly.
"""

from pathlib import Path

import polars as pl
from jsonargparse import auto_cli
from src.topk_regroup import regroup_topk_hits

# In run_benchmark's column order, so a k = K rescore is byte-for-byte the
# run's own results parquet apart from the model name.
META_COLUMNS = [
    "index_size_gb",
    "avg_time",
    "model",
    "mutation_rate",
    "query_type",
    "checkpoint",
    "max_len",
    "checkpoint_step_num",
    "chunk_type",
]


def rescore_file(
    hits_path: Path, out_dir: Path, accessions: list[str], ks: list[int]
) -> list[Path]:
    hits = pl.read_parquet(hits_path)
    saved_k = int(hits["hits"].list.len().max())
    meta = hits.select(META_COLUMNS).unique()
    assert len(meta) == 1, f"{hits_path}: metadata columns are not constant"
    meta = meta.row(0, named=True)
    written = []
    for k in ks:
        if k > saved_k:
            print(f"  skip k={k}: only {saved_k} hits saved per query")
            continue
        results = regroup_topk_hits(hits, accessions, k=k)
        for col in META_COLUMNS:
            dtype = hits.schema[col]
            value = f"{meta['model']}_k{k}" if col == "model" else meta[col]
            results = results.with_columns(pl.lit(value, dtype=dtype).alias(col))
        out_path = (
            out_dir
            / f"{meta['model']}_k{k}"
            / f"raw_read_mut_{meta['mutation_rate']}.parquet"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.select("query_id", "results", *META_COLUMNS).write_parquet(out_path)
        written.append(out_path)
        print(f"  k={k:>5} -> {out_path}")
    return written


def main(
    hits_path: str,
    out_dir: str,
    accessions: str,
    ks: list[int] | None = None,
):
    """
    Args:
        hits_path: a *_topk<K>.parquet hits file, or a directory to search.
        out_dir: results root; print_results.py takes this as results_dir.
        accessions: accs.txt of the benchmark bundle -- the accession universe
            every query's results must cover (misses get the sentinel).
        ks: vector-level top-k values to derive; each must be <= K. Default
            [10, 20, 50, 100, 200, 1000], the experiment's sweep.
    """
    if ks is None:
        ks = [10, 20, 50, 100, 200, 1000]
    hits_path = Path(hits_path)
    files = (
        [hits_path]
        if hits_path.is_file()
        else sorted(hits_path.rglob("*_topk*.parquet"))
    )
    if not files:
        raise SystemExit(f"No *_topk*.parquet hits under {hits_path}")
    accs = Path(accessions).read_text().splitlines()
    for f in files:
        print(f"{f}:")
        rescore_file(f, Path(out_dir), accs, ks)


if __name__ == "__main__":
    auto_cli(main)
