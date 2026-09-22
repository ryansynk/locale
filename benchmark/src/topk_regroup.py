"""Vector-level top-k hits -> per-accession results.

The top-k engines (exact top-k scan, ANN, RaBitQ) return, per query, its k
nearest index *vectors*. The benchmark scores *accessions*, so the hits are
regrouped: an accession's score is the max over its vectors in the hit list, and
every accession absent from the list gets ``MISS_SCORE`` (plot_results
DEFAULT_SCORE, which its metrics treat as "never returned").

This lives apart from dense_index so it can be used without pulling in
torch/CUDA. Nothing here touches a GPU.

Hits frame schema (one row per query):
    query_id: str
    hits: list[struct{accession: str, score: float, vector_id: int}]
          sorted by score descending, at most top_k entries

Results frame schema (what run_benchmark writes and print_results reads):
    query_id: str
    results: list[struct{accession: str, score: float}], one entry per
             indexed accession
"""

import numpy as np
import polars as pl

# Must stay equal to plot_results.DEFAULT_SCORE (not imported: plot_results
# pulls in matplotlib/LaTeX config at import time).
MISS_SCORE = -2.0

HITS_DTYPE = pl.List(
    pl.Struct({"accession": pl.String, "score": pl.Float64, "vector_id": pl.Int64})
)
RESULTS_DTYPE = pl.List(pl.Struct({"accession": pl.String, "score": pl.Float64}))


def vector_ids_to_accessions(
    vector_ids: np.ndarray, acc_offsets: np.ndarray, acc_names: list[str]
) -> list[str]:
    """Map flat index-vector ids to accession names via the meta.parquet offsets.

    ``acc_offsets`` has len(acc_names)+1 entries: accession i owns rows
    [acc_offsets[i], acc_offsets[i+1]).
    """
    acc_idx = np.searchsorted(acc_offsets, vector_ids, side="right") - 1
    if len(acc_idx) and (acc_idx.min() < 0 or acc_idx.max() >= len(acc_names)):
        raise ValueError("vector id outside the index's accession offsets")
    names = np.asarray(acc_names, dtype=object)
    return names[acc_idx].tolist()


def build_hits_frame(
    query_ids: list[str],
    flat_query_pos: np.ndarray,
    flat_vector_ids: np.ndarray,
    flat_scores: np.ndarray,
    acc_offsets: np.ndarray,
    acc_names: list[str],
    top_k: int,
) -> pl.DataFrame:
    """Assemble a hits frame from flat (query position, vector id, score) triples.

    A query may contribute the same vector more than once (each of a long
    query's chunks keeps its own top-k); duplicates collapse to the max score,
    then each query keeps its top_k by score. Every query in ``query_ids`` gets
    a row, with an empty list if it has no hits.
    """
    flat = pl.DataFrame(
        {
            "query_pos": pl.Series(flat_query_pos, dtype=pl.Int64),
            "vector_id": pl.Series(flat_vector_ids, dtype=pl.Int64),
            "score": pl.Series(flat_scores, dtype=pl.Float64),
        }
    ).filter(pl.col("vector_id") >= 0)
    flat = (
        flat.group_by("query_pos", "vector_id")
        .agg(pl.col("score").max())
        .sort(["query_pos", "score", "vector_id"], descending=[False, True, False])
        .group_by("query_pos", maintain_order=True)
        .head(top_k)
    )
    flat = flat.with_columns(
        pl.Series(
            "accession",
            vector_ids_to_accessions(
                flat["vector_id"].to_numpy(), acc_offsets, acc_names
            ),
            dtype=pl.String,
        )
    )
    grouped = flat.group_by("query_pos", maintain_order=True).agg(
        pl.struct("accession", "score", "vector_id").alias("hits")
    )
    base = pl.DataFrame(
        {
            "query_pos": pl.Series(range(len(query_ids)), dtype=pl.Int64),
            "query_id": pl.Series(query_ids, dtype=pl.String),
        }
    )
    out = base.join(grouped, on="query_pos", how="left").with_columns(
        pl.col("hits").fill_null(pl.lit([], dtype=HITS_DTYPE))
    )
    return out.select("query_id", pl.col("hits").cast(HITS_DTYPE))


def merge_topk_hits(partials: list[pl.DataFrame], top_k: int) -> pl.DataFrame:
    """Union per-shard hits frames (same queries, disjoint vector ranges) and
    re-take each query's global top_k. Each partial is exact over its own
    range, so the union's top_k is the exact global top_k."""
    exploded = (
        pl.concat([p.select("query_id", "hits") for p in partials])
        .explode("hits")
        .drop_nulls("hits")
        .unnest("hits")
        .sort(["query_id", "score", "vector_id"], descending=[False, True, False])
        .group_by("query_id", maintain_order=True)
        .head(top_k)
        .group_by("query_id", maintain_order=True)
        .agg(pl.struct("accession", "score", "vector_id").alias("hits"))
    )
    query_ids = partials[0].select("query_id")
    out = query_ids.join(exploded, on="query_id", how="left").with_columns(
        pl.col("hits").fill_null(pl.lit([], dtype=HITS_DTYPE))
    )
    assert len(out) == len(query_ids)
    return out.select("query_id", pl.col("hits").cast(HITS_DTYPE))


def regroup_topk_hits(
    hits: pl.DataFrame, accessions: list[str], k: int | None = None
) -> pl.DataFrame:
    """Regroup vector hits into the standard per-accession results frame.

    Keeps the first ``k`` hits per query (all of them when k is None) -- the
    lists are score-descending, so this is "the top-k vectors" -- takes the max
    score per accession, and emits every accession in ``accessions`` order,
    scoring the ones outside the top-k at MISS_SCORE.
    """
    n_acc = len(accessions)
    acc_to_idx = {a: i for i, a in enumerate(accessions)}
    n_queries = len(hits)

    hit_col = pl.col("hits").list.head(k) if k is not None else pl.col("hits")
    flat = (
        hits.with_row_index("query_pos")
        .select("query_pos", hit_col.alias("hits"))
        .explode("hits")
        .drop_nulls("hits")
        .unnest("hits")
        .group_by("query_pos", "accession")
        .agg(pl.col("score").max())
    )
    unknown = set(flat["accession"].to_list()) - acc_to_idx.keys()
    if unknown:
        raise ValueError(f"hits name accessions not in the accession list: {unknown}")

    scores = np.full((n_queries, n_acc), MISS_SCORE, dtype=np.float64)
    if len(flat):
        qi = flat["query_pos"].to_numpy().astype(np.int64)
        ai = np.fromiter(
            (acc_to_idx[a] for a in flat["accession"].to_list()),
            dtype=np.int64,
            count=len(flat),
        )
        scores[qi, ai] = flat["score"].to_numpy()

    results = (
        pl.DataFrame(
            {
                "query_pos": np.repeat(np.arange(n_queries, dtype=np.int64), n_acc),
                "accession": pl.Series(accessions * n_queries, dtype=pl.String),
                "score": scores.ravel(),
            }
        )
        .group_by("query_pos", maintain_order=True)
        .agg(pl.struct("accession", "score").alias("results"))
    )
    out = pl.DataFrame({"query_id": hits["query_id"]}).with_row_index("query_pos")
    out = out.with_columns(pl.col("query_pos").cast(pl.Int64))
    results = results.with_columns(pl.col("query_pos").cast(pl.Int64))
    out = out.join(results, on="query_pos", how="left").select(
        "query_id", pl.col("results").cast(RESULTS_DTYPE)
    )
    assert len(out) == n_queries
    assert out["results"].null_count() == 0
    return out
