from pathlib import Path

import altair as alt
import numpy as np
import polars as pl
from jsonargparse import auto_cli


def main(
    results_dir: str,
    plots_dir: str = "plots",
):
    results_dir: Path = Path(results_dir)
    plots_dir: Path = Path(plots_dir)
    if not plots_dir.is_dir():
        plots_dir.mkdir()
    data = []
    schema = pl.Schema(
        {
            "query_read": pl.String,
            "query_accession": pl.String,
            "results": pl.List(
                pl.Struct({"accession": pl.String, "score": pl.Float64})
            ),
            "model": pl.String,
            "mutation_rate": pl.Float64,
        }
    )
    for f in list(results_dir.rglob("*.parquet")):
        df = pl.read_parquet(f, schema=schema)
        data.append(df)
    data = pl.concat(data)

    # 2. Explode, extract structs, and rank results
    df_exploded = (
        data.explode("results")
        .select(
            pl.col("query_read"),
            pl.col("query_accession"),
            pl.col("results").struct.field("score").alias("score"),
            pl.col("results").struct.field("accession").alias("accession"),
            pl.col("model"),
            pl.col("mutation_rate"),
        )
        # Sort by query_read and descending score to ensure strict ranking
        .sort(["query_read", "score"], descending=[False, True])
    )

    # 3. Calculate Precision@K and Recall@K per query
    # Note: Since true_accession is a single string per query, total_relevant = 1
    df_hits = df_exploded.with_columns(
        k=pl.int_range(1, pl.len() + 1).over("query_read"),
        is_relevant=(pl.col("accession") == pl.col("query_accession")).cast(pl.Float64),
    ).with_columns(cum_hits=pl.col("is_relevant").cum_sum().over("query_read"))

    max_k = df_hits.select(pl.col("k").max()).item()
    df_queries = df_hits.select("query_read", "model", "mutation_rate").unique()
    df_k_grid = pl.DataFrame({"k": range(1, max_k + 1)})
    df_grid = df_queries.join(df_k_grid, how="cross")
    df_metrics = (
        df_grid.join(
            df_hits.select("query_read", "k", "cum_hits"),
            on=["query_read", "k"],
            how="left",
        )
        .sort(["query_read", "k"])
        .with_columns(cum_hits=pl.col("cum_hits").forward_fill().over("query_read"))
        .with_columns(
            cum_hits=pl.col("cum_hits").fill_null(
                0.0
            )  # Just in case k=1 was completely empty
        )
        .with_columns(
            precision_at_k=pl.col("cum_hits") / pl.col("k"),
            recall_at_k=pl.col("cum_hits"),
        )
    )

    # 4. Average over all queries to get Mean PR values per K
    df_pr_curve = (
        df_metrics.group_by(["model", "mutation_rate", "k"])
        .agg(
            mean_precision=pl.col("precision_at_k").mean(),
            mean_recall=pl.col("recall_at_k").mean(),
        )
        .sort(["model", "mutation_rate", "mean_recall", "k"])
    )

    # 5. Calculate AUPRC per Model & Mutation Rate group using window functions
    df_auprc = (
        df_pr_curve.with_columns(
            prev_recall=pl.col("mean_recall")
            .shift(1)
            .fill_null(0.0)
            .over(["model", "mutation_rate"]),
            prev_precision=pl.col("mean_precision")
            .shift(1)
            .fill_null(1.0)
            .over(["model", "mutation_rate"]),
        )
        .with_columns(
            auprc_step=(
                (pl.col("mean_recall") - pl.col("prev_recall"))
                * (pl.col("mean_precision") + pl.col("prev_precision"))
                / 2
            )
        )
        .group_by(["model", "mutation_rate"])
        .agg(auprc=pl.col("auprc_step").sum())
    )
    print(df_auprc)

    # 6. Join AUPRC back to the PR Curve to create a descriptive legend label
    df_plot = df_pr_curve.join(df_auprc, on=["model", "mutation_rate"]).with_columns(
        legend_label=pl.concat_str(
            [
                pl.col("model"),
                pl.lit(" (Mut: "),
                pl.col("mutation_rate").cast(pl.String),
                pl.lit(") - AUPRC: "),
                pl.col("auprc").round(3).cast(pl.String),
            ]
        )
    )

    chart = (
        alt.Chart(df_plot)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "mean_precision:Q",
                title="Mean Precision@K",
                scale=alt.Scale(domain=[0, 1.05]),
            ),
            y=alt.Y(
                "mean_recall:Q", title="Mean Recall@K", scale=alt.Scale(domain=[0, 1.0])
            ),
            color=alt.Color(
                "model:N",
                title="Model",
            ),
            strokeDash=alt.StrokeDash(
                "mutation_rate:N"
            ),  # Optional: visually separate mutation rates by line style
        )
        .properties(
            width=650,
            height=450,
        )
        .configure_axis(labelFontSize=15, titleFontSize=20)
        .configure_legend(labelFontSize=14, titleFontSize=16)
    )
    chart.save(plots_dir / "precision_recall_curve.png")

    other_chart = (
        alt.Chart(df_plot)
        .mark_line(point=True)
        .encode(
            x=alt.X("k:Q", title="k"),
            y=alt.Y("mean_recall:Q", title="Mean Recall@K"),
            color=alt.Color(
                "model:N",
                title="Model",
            ),
            strokeDash=alt.StrokeDash(
                "mutation_rate:N"
            ),  # Optional: visually separate mutation rates by line style
        )
        .properties(
            width=650,
            height=450,
        )
        .configure_axis(labelFontSize=15, titleFontSize=20)
        .configure_legend(labelFontSize=14, titleFontSize=16)
    )
    other_chart.save(plots_dir / "recall_vs_k_curve.png")


if __name__ == "__main__":
    auto_cli(main)
