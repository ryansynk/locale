from pathlib import Path
import numpy as np

import altair as alt
import polars as pl
from jsonargparse import auto_cli


def main(results_dir: str, plots_dir: str = "plots", num_identity_bins: int = 11):
    results_dir: Path = Path(results_dir)
    plots_dir: Path = Path(plots_dir)
    if not plots_dir.is_dir():
        plots_dir.mkdir()
    data = []
    schema = pl.Schema(
        {
            "query_read": pl.String,
            "query_accession": pl.String,
            "retrieved_accession": pl.String,
            "identity": pl.Float64,
            "model": pl.String,
            "mutation_rate": pl.Float64,
        }
    )
    for f in list(results_dir.rglob("*.parquet")):
        df = pl.read_parquet(f, schema=schema)
        data.append(df)
    data = pl.concat(data)
    data = data.with_columns(
        (pl.col("query_accession") == pl.col("retrieved_accession"))
        .cast(pl.Int64)
        .alias("correct")
    )

    identity_cutoffs = np.linspace(0.0, 1.0, num=num_identity_bins)

    dfs = []
    for name, df in data.group_by("model", "mutation_rate"):
        model = name[0]
        mutation_rate = name[1]
        recalls = []
        for c in identity_cutoffs:
            recall = (
                df.filter(pl.col("identity") > c)
                .group_by("query_read")
                .agg(pl.col("correct").any().cast(pl.Int64))["correct"]
                .mean()
            )
            recalls.append(recall)
        df = pl.DataFrame({"identity_cutoff": identity_cutoffs, "recall": recalls})
        df = df.with_columns(
            pl.lit(model).alias("model"), pl.lit(mutation_rate).alias("mutation_rate")
        )
        dfs.append(df)

    recall_df = pl.concat(dfs).fill_null(0.0)
    chart = (
        alt.Chart(recall_df)
        .mark_line()
        .encode(
            x=alt.X("identity_cutoff", title="Sequence identity cut-off (%)"),
            y=alt.Y("recall", title="Recall (%)"),
            color=alt.Color("model:N", title="Model"),
            strokeDash=alt.StrokeDash("mutation_rate:N", title="Mutation Rate"),
        )
    )
    chart.save(plots_dir / "recall_curve.png")


if __name__ == "__main__":
    auto_cli(main)
