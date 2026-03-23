from pathlib import Path

import altair as alt
import polars as pl
from jsonargparse import auto_cli
from jsonargparse.typing import Path_fr


def calculate_recall_precision(
    retrieval_results: list[dict],
    ground_truth_results: list[str],
    read_id: str,
    model: str,
    mutation_rate: float,
    query_type: str,
    max_k: int,
):
    gt_set: set[str] = set(ground_truth_results)
    num_gt = len(gt_set)
    retrieval_results = sorted(
        retrieval_results, key=lambda x: x.get("score", 0.0), reverse=True
    )

    outputs = []
    true_positives = set()

    for k in range(1, max_k + 1):
        if k <= len(retrieval_results):
            accession = retrieval_results[k - 1].get("accession")
            if accession in gt_set:
                true_positives.add(accession)
        precision = len(true_positives) / k
        recall = len(true_positives) / num_gt
        outputs.append(
            {
                "read_id": read_id,
                "model": model,
                "mutation_rate": mutation_rate,
                "query_type": query_type,
                "k": k,
                "precision": precision,
                "recall": recall,
            }
        )
    return outputs


def calculate_recall_precision_df(queries_df: pl.DataFrame, data: pl.DataFrame):
    all_recalls_precisions = []

    # Largest number of matches in ground truth over all queries
    max_k_gt = (
        queries_df.with_columns(
            pl.col("contig_accession").list.len().alias("num_results")
        )
        .select(pl.col("num_results").max())
        .item()
    )

    # Largest number of returned results over all queries
    max_k_results = (
        data.with_columns(pl.col("results").list.len().alias("num_results"))
        .select(pl.col("num_results").max())
        .item()
    )

    max_k = max(max_k_gt, max_k_results)

    for row in data.iter_rows(named=True):
        gt_results = queries_df.filter(pl.col("read_id") == row["query_read"])[
            "contig_accession"
        ].item()
        recalls_precisions = calculate_recall_precision(
            row["results"],
            gt_results,
            row["query_read"],
            row["model"],
            row["mutation_rate"],
            row["query_type"],
            max_k=max_k,
        )
        all_recalls_precisions.extend(recalls_precisions)

    return pl.from_dicts(all_recalls_precisions)


def plot_recall_precision(recall_precision_df: pl.DataFrame, plots_dir: Path):
    recall_precision_df = (
        recall_precision_df.group_by(["model", "mutation_rate", "k", "query_type"])
        .agg(
            pl.col("recall").mean().alias("average_recall"),
            pl.col("precision").mean().alias("average_precision"),
        )
        .sort("average_recall")
    )
    recall_precision_df = recall_precision_df.sort("average_recall")
    for name, data in recall_precision_df.group_by("mutation_rate", "query_type"):
        mut_rate = name[0]
        query_type = name[1]
        title = ""
        match query_type:
            case "raw_read":
                title = "Average Precision-Recall Curve for Raw Read Queries"
            case "logan_contig":
                title = "Average Precision-Recall Curve for Logan Contig Queries"
        chart = (
            alt.Chart(data)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "average_recall:Q",
                    title="Mean Recall@K",
                    scale=alt.Scale(domain=[0, 1.0]),
                ),
                y=alt.Y(
                    "average_precision:Q",
                    title="Mean Precision@K",
                    scale=alt.Scale(domain=[0, 1.05]),
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
                title=title,
                width=650,
                height=450,
            )
            .configure_axis(labelFontSize=15, titleFontSize=20)
            .configure_legend(labelFontSize=14, titleFontSize=16)
        )
        chart.save(
            plots_dir
            / f"{query_type}_precision_recall_curve_mutation_rate_{str(mut_rate)}.png"
        )


def plot_recall_at_k(recall_precision_df: pl.DataFrame, plots_dir: Path):
    recall_precision_df = (
        recall_precision_df.group_by(["model", "mutation_rate", "k", "query_type"])
        .agg(
            pl.col("recall").mean().alias("average_recall"),
            pl.col("precision").mean().alias("average_precision"),
        )
        .sort("average_recall")
    )
    recall_precision_df = recall_precision_df.sort("average_recall")
    for name, data in recall_precision_df.group_by("mutation_rate", "query_type"):
        mut_rate = name[0]
        query_type = name[1]
        title = ""
        match query_type:
            case "raw_read":
                title = "Recall @ k for Raw Read Queries"
            case "logan_contig":
                title = "Recall @ k for Logan Contig Queries"
        chart = (
            alt.Chart(data)
            .mark_line(point=True)
            .encode(
                x=alt.X("k:Q", title="k", scale=alt.Scale(domain=[1, 47])),
                y=alt.Y("average_recall:Q", title="Mean Recall@K"),
                color=alt.Color(
                    "model:N",
                    title="Model",
                ),
            )
            .properties(
                title=title,
                width=650,
                height=450,
            )
            .configure_axis(labelFontSize=15, titleFontSize=20)
            .configure_legend(labelFontSize=14, titleFontSize=16)
        )
        chart.save(
            plots_dir / f"{query_type}_recall_vs_k_curve_mutation_rate_{mut_rate}.png"
        )


def plot_auprc(recall_precision_df: pl.DataFrame, plots_dir: Path):
    sorted_df = recall_precision_df.sort(
        ["model", "mutation_rate", "read_id", "query_type", "recall"]
    )
    macro_auprc = sorted_df.group_by(
        ["model", "mutation_rate", "read_id", "query_type"], maintain_order=True
    ).agg(
        # 2. Apply trapezoidal rule to the explicitly sorted columns
        (
            0.5
            * (pl.col("recall") - pl.col("recall").shift(1))
            * (pl.col("precision") + pl.col("precision").shift(1))
        )
        .fill_null(0.0)
        .sum()
        .alias("read_auprc")
    )
    macro_auprc = macro_auprc.group_by(["model", "mutation_rate", "query_type"]).agg(
        pl.col("read_auprc").mean().alias("auprc")
    )

    for name, df in macro_auprc.group_by("query_type"):
        query_type = name[0]
        chart = (
            alt.Chart(macro_auprc)
            .mark_bar()
            .encode(
                x=alt.X("model:N"),
                y=alt.Y("auprc:Q", scale=alt.Scale(domain=[0, 1.0])),
                column="mutation_rate:Q",
            )
        )
        chart.save(plots_dir / f"{query_type}_auprc_bar_chart.png")


def main(
    results_dir: str,
    queries_path: Path_fr,
    plots_dir: str = "plots",
):
    results_dir: Path = Path(results_dir)
    queries_path: Path = Path(queries_path)
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
            "query_type": pl.String,
        }
    )
    for f in list(results_dir.rglob("*.parquet")):
        df = pl.read_parquet(f, schema=schema)
        data.append(df)
    data = pl.concat(data)

    queries_df = pl.read_parquet(queries_path)
    recall_precision_df = calculate_recall_precision_df(queries_df, data)
    plot_recall_precision(recall_precision_df, plots_dir)
    plot_recall_at_k(recall_precision_df, plots_dir)
    plot_auprc(recall_precision_df, plots_dir)


if __name__ == "__main__":
    auto_cli(main)
