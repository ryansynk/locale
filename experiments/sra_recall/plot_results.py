from pathlib import Path

import altair as alt
import polars as pl
from jsonargparse import auto_cli
from jsonargparse.typing import Path_fr


def calculate_recall_precision(
    retrieval_results: list[dict],
    ground_truth_results: list[str],
    query_id: str,
    model: str,
    mutation_rate: float,
    query_type: str,
    checkpoint: str | None,
    max_len: int | None,
    checkpoint_step_num: int | None,
    chunk_type: str | None,
    max_k: int,
    avg_time: float | None,
):
    filtered_retrieval_results = [
        result
        for result in retrieval_results
        if result["score"] and result["accession"]
    ]
    gt_set: set[str] = set(ground_truth_results)
    num_gt = len(gt_set)
    retrieval_results = sorted(
        filtered_retrieval_results, key=lambda x: x.get("score", 0.0), reverse=True
    )

    outputs = []
    true_positives = set()
    precision = 0.0
    recall = 0.0

    for k in range(1, max_k + 1):
        if k <= len(retrieval_results):
            accession = retrieval_results[k - 1].get("accession")
            if accession in gt_set:
                true_positives.add(accession)
            precision = len(true_positives) / k
            recall = len(true_positives) / num_gt
        # else: hold precision and recall at their last values

        outputs.append(
            {
                "query_id": query_id,
                "model": model,
                "checkpoint": checkpoint,
                "max_len": max_len,
                "checkpoint_step_num": checkpoint_step_num,
                "chunk_type": chunk_type,
                "mutation_rate": mutation_rate,
                "query_type": query_type,
                "k": k,
                "precision": precision,
                "recall": recall,
                "avg_time": avg_time,
            }
        )
    return outputs


def add_random_baseline(data, ground_truth, total_num_items):
    # Random baseline
    num_relevant_items = ground_truth.with_columns(
        pl.col("results").list.len().alias("num_results")
    ).select("query_id", "num_results")
    random_baseline = []
    for row in num_relevant_items.iter_rows(named=True):
        query_id = row["query_id"]
        num_relevant = row["num_results"]
        for k in range(1, total_num_items + 1):
            random_baseline.append(
                {
                    "query_id": query_id,
                    "model": "random",
                    "k": k,
                    "precision": num_relevant / total_num_items,
                    "recall": k / total_num_items,
                }
            )

    baseline_df = pl.from_dicts(random_baseline)
    baseline_df = baseline_df.with_columns(
        pl.lit(None).alias("checkpoint"),
        pl.lit(None).alias("max_len"),
        pl.lit(None).alias("checkpoint_step_num"),
        pl.lit(None).alias("chunk_type"),
    )
    combos = data.select("mutation_rate", "query_type").unique()
    baseline_df = baseline_df.join(combos, how="cross")
    return pl.concat([data, baseline_df], how="diagonal")


def calculate_recall_precision_df(ground_truth: pl.DataFrame, data: pl.DataFrame):

    all_recalls_precisions = []

    # Largest number of matches in ground truth over all queries
    max_k_gt = (
        ground_truth.with_columns(pl.col("results").list.len().alias("num_results"))
        .select(pl.col("num_results").max())
        .item()
    )
    total_num_items = 0
    for _, df in data.group_by(
        [
            "model",
            "checkpoint",
            "max_len",
            "checkpoint_step_num",
            "chunk_type",
            "query_type",
        ]
    ):
        # Largest number of returned results over all queries
        max_k_results = (
            df.with_columns(pl.col("results").list.len().alias("num_results"))
            .select(pl.col("num_results").max())
            .item()
        )
        max_k = max(max_k_gt, max_k_results)
        if max_k > total_num_items:
            total_num_items = max_k

        for row in df.iter_rows(named=True):
            gt_results = (
                ground_truth.filter(
                    (pl.col("query_id") == row["query_id"])
                    & (pl.col("query_type") == row["query_type"])
                    & (pl.col("mutation_rate") == row["mutation_rate"])
                )["results"]
                .item()
                .to_list()
            )
            gt_results = [res["accession"] for res in gt_results]
            recalls_precisions = calculate_recall_precision(
                row["results"],
                gt_results,
                row["query_id"],
                row["model"],
                row["mutation_rate"],
                row["query_type"],
                row["checkpoint"],
                row["max_len"],
                row["checkpoint_step_num"],
                row["chunk_type"],
                max_k=max_k,
                avg_time=row["avg_time"],
            )
            all_recalls_precisions.extend(recalls_precisions)

    schema = pl.Schema(
        {
            "query_id": pl.String,
            "model": pl.String,
            "checkpoint": pl.String,
            "max_len": pl.Int64,
            "checkpoint_step_num": pl.Int64,
            "chunk_type": pl.String,
            "mutation_rate": pl.Float64,
            "query_type": pl.String,
            "k": pl.Int64,
            "precision": pl.Float64,
            "recall": pl.Float64,
            "avg_time": pl.Float64,
        }
    )
    data = pl.from_dicts(all_recalls_precisions, schema=schema)
    data = add_random_baseline(data, ground_truth, total_num_items)
    return data


def plot_contig_len_hit_at_k(
    ground_truth: pl.DataFrame, data: pl.DataFrame, plots_dir: Path, k: int = 7
):
    data = data.filter(pl.col("query_type") == "raw_read")
    data = data.filter(
        (pl.col("chunk_type") == "exact") | (pl.col("chunk_type").is_null())
    )
    if data.is_empty():
        return
    ground_truth = ground_truth.filter(pl.col("query_type") == "raw_read")
    results_df = (
        data.explode("results")
        .unnest("results")
        .sort(["model", "mutation_rate", "query_id", "score"], descending=True)
        .group_by("model", "mutation_rate", "query_id", maintain_order=True)
        .agg(pl.col("accession"))
    ).rename({"accession": "retrieved_accession"})
    results_df = results_df.join(
        ground_truth.select(["query_id", "results", "contig_len", "mutation_rate"]),
        on=["query_id", "mutation_rate"],
    )
    # results_df = results_df.explode(["contig_accession", "identity", "contig_len"])
    results_df = results_df.explode(["results", "contig_len"]).unnest("results")
    results_df = results_df.with_columns(
        pl.col("accession")
        .is_in(pl.col("retrieved_accession").list.slice(0, k))
        .alias(f"hit_at_{k}")
    )
    results_df = results_df.with_columns(
        pl.col("contig_len").qcut(8).alias("contig_len_group")
    )
    plot_df = (
        results_df.group_by("model", "mutation_rate", "contig_len_group")
        .agg(pl.col(f"hit_at_{k}").mean(), pl.len())
        .sort("contig_len_group", "model")
    )
    min_contig_len = results_df.select(pl.col("contig_len").min()).item()
    max_contig_len = results_df.select(pl.col("contig_len").max()).item()
    plot_df = plot_df.with_columns(
        pl.col("contig_len_group")
        .cast(pl.String)
        .str.replace("-inf", str(min_contig_len), literal=True)
        .str.replace("inf", str(max_contig_len), literal=True)
    )
    # sort_order = plot_df["contig_len_group"].cat.get_categories().to_list()
    unsorted_bins = plot_df["contig_len_group"].cast(pl.String).unique().to_list()

    # 2. Define a sorting key to extract the lower bound
    def get_lower_bound(interval_str):
        # Strip brackets/parentheses: "(4844, 14290]" -> "4844, 14290"
        clean_str = interval_str.strip("()[]")

        # Split by the comma and grab the first value: "4844"
        lower_bound_str = clean_str.split(",")[0]

        # Convert to float so python handles '-inf' and standard numbers correctly
        return float(lower_bound_str)

    # 3. Sort the list numerically based on that lower bound
    sort_order = sorted(unsorted_bins, key=get_lower_bound)

    for name, mut_df in plot_df.group_by("mutation_rate"):
        mutation_rate = name[0]
        chart = (
            alt.Chart(mut_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "model:O",
                    title=None,
                    axis=alt.Axis(labelAngle=-45, labelFontSize=15),
                ),
                y=alt.Y(
                    f"hit_at_{k}:Q",
                    scale=alt.Scale(domain=[0, 1.0]),
                    title=f"Hit at {k}",
                    axis=alt.Axis(labelFontSize=15, titleFontSize=20),
                ),
                color=alt.Color("model:O"),
                column=alt.Column(
                    "contig_len_group:O",
                    sort=sort_order,
                    title="Length of Matching Contig",
                    header=alt.Header(
                        labelFontSize=10,
                        titleFontSize=18,
                    ),
                ),
            )
        )
        chart.save(plots_dir / f"mut_{mutation_rate}_contig_lens_bar_chart.png")


def get_average_precision_recall_df(recall_precision_df: pl.DataFrame):
    return (
        recall_precision_df.group_by(
            [
                "model",
                "checkpoint",
                "max_len",
                "checkpoint_step_num",
                "chunk_type",
                "mutation_rate",
                "k",
                "query_type",
            ]
        )
        .agg(
            pl.col("recall").mean().alias("average_recall"),
            pl.col("precision").mean().alias("average_precision"),
            pl.col("avg_time").max().alias("avg_time"),
        )
        .sort("average_recall")
    )


def plot_recall_precision(recall_precision_df: pl.DataFrame, plots_dir: Path):
    avg_recall_precision_df = get_average_precision_recall_df(recall_precision_df)
    for name, data in avg_recall_precision_df.group_by("mutation_rate", "query_type"):
        mut_rate = name[0]
        query_type = name[1]
        match query_type:
            case "raw_read":
                title = "Average Precision-Recall Curve for Raw Read Queries"
            case "logan_contig":
                title = "Average Precision-Recall Curve for Logan Contig Queries"
            case "gencode":
                title = "Average Precision-Recall Curve for Gencode Queries"
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
    max_k = recall_precision_df.select(pl.col("k")).max().item()
    avg_recall_precision_df = get_average_precision_recall_df(recall_precision_df)
    for name, data in avg_recall_precision_df.group_by("mutation_rate", "query_type"):
        mut_rate = name[0]
        query_type = name[1]
        match query_type:
            case "raw_read":
                title = "Recall @ k for Raw Read Queries"
            case "logan_contig":
                title = "Recall @ k for Logan Contig Queries"
            case "gencode":
                title = "Recall @ k for Gencode Queries"
        chart = (
            alt.Chart(data)
            .mark_line(point=True)
            .encode(
                x=alt.X("k:Q", title="k", scale=alt.Scale(domain=[1, max_k])),
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
        ["model", "mutation_rate", "query_id", "query_type", "recall"]
    )
    macro_auprc = sorted_df.group_by(
        [
            "model",
            "checkpoint",
            "max_len",
            "checkpoint_step_num",
            "chunk_type",
            "mutation_rate",
            "query_id",
            "query_type",
        ],
        maintain_order=True,
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
    macro_auprc = macro_auprc.group_by(
        [
            "model",
            "checkpoint",
            "max_len",
            "checkpoint_step_num",
            "chunk_type",
            "mutation_rate",
            "query_type",
        ]
    ).agg(pl.col("read_auprc").mean().alias("auprc"))

    for name, df in macro_auprc.group_by("query_type"):
        query_type = name[0]
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("model:N"),
                y=alt.Y(
                    "auprc:Q", scale=alt.Scale(domain=[0, 0.4])
                ),  # oracle performance is around 0.39
                column="mutation_rate:Q",
            )
        )
        chart.save(plots_dir / f"{query_type}_auprc_bar_chart.png")


def raw_read_oracle_results(raw_read_queries_df):
    oracle_df = (
        raw_read_queries_df.select("query_id", "contig_accession", "identity")
        .explode("contig_accession", "identity")
        .rename(
            {
                "contig_accession": "accession",
                "identity": "score",
            }
        )
        .group_by("query_id", "accession")
        .agg(pl.col("score").max())
        .with_columns(pl.struct("accession", "score").alias("results"))
        .drop("accession", "score")
        .group_by("query_id")
        .agg(pl.col("results"))
    ).with_columns(
        pl.lit("oracle").alias("model"),
        pl.lit("raw_read").alias("query_type"),
        pl.lit(None).alias("checkpoint"),
        pl.lit(None).alias("max_len"),
        pl.lit(None).alias("checkpoint_step_num"),
        pl.lit(None).alias("chunk_type"),
        pl.lit(None).alias("avg_time"),
    )
    return oracle_df


def gencode_oracle_results(gencode_queries_df):
    oracle_df = (
        gencode_queries_df.select("query_id", "accessions", "coverages")
        .explode("accessions", "coverages")
        .rename(
            {
                "accessions": "accession",
                "coverages": "score",
            }
        )
        .with_columns(pl.struct("accession", "score").alias("results"))
        .drop("accession", "score")
        .group_by("query_id")
        .agg(pl.col("results"))
    ).with_columns(
        pl.lit("oracle").alias("model"),
        pl.lit("gencode").alias("query_type"),
        pl.lit(None).alias("checkpoint"),
        pl.lit(None).alias("max_len"),
        pl.lit(None).alias("checkpoint_step_num"),
        pl.lit(None).alias("chunk_type"),
        pl.lit(None).alias("avg_time"),
    )
    return oracle_df


def get_ground_truth(raw_read_queries_df, gencode_oracle_data, combos):
    oracle_raw_read_data_with_contig_len = (
        raw_read_queries_df.select(
            "query_id", "contig_accession", "identity", "contig_len"
        )
        .explode("contig_accession", "identity", "contig_len")
        .rename(
            {
                "contig_accession": "accession",
                "identity": "score",
            }
        )
        .group_by("query_id", "accession")
        .agg(pl.col("score").max(), pl.col("contig_len").max())
        .with_columns(pl.struct("accession", "score").alias("results"))
        .drop("accession", "score")
        .group_by("query_id")
        .agg(pl.col("results"), pl.col("contig_len"))
    ).with_columns(
        pl.lit("oracle").alias("model"),
        pl.lit("raw_read").alias("query_type"),
        pl.lit(None).alias("checkpoint"),
        pl.lit(None).alias("max_len"),
        pl.lit(None).alias("checkpoint_step_num"),
        pl.lit(None).alias("chunk_type"),
        pl.lit(None).alias("avg_time"),
    )
    oracle_raw_read_data_with_contig_len = oracle_raw_read_data_with_contig_len.join(
        combos, how="cross"
    )
    return pl.concat(
        [oracle_raw_read_data_with_contig_len, gencode_oracle_data], how="diagonal"
    )


def plot_recall_vs_noise_line(
    recall_precision_df: pl.DataFrame,
    plots_dir: Path,
    k: int = 7,
):
    avg_recall_precision_df = get_average_precision_recall_df(recall_precision_df)
    avg_recall_precision_df = avg_recall_precision_df.filter(pl.col("k") == k)
    avg_recall_precision_df = avg_recall_precision_df.filter(
        ~pl.col("model").is_in(["random", "oracle", "mmseqs"])
    )
    avg_recall_precision_df = avg_recall_precision_df.with_columns(
        pl.col("model").str.split("_").list.get(0)
    )
    title_names = {
        "llmed": "LLM-ED",
        "rawbert": "RawBERT",
        "metagraph": "MetaGraph",
        "dna2vec": "Embed-Search-Align",
    }
    avg_recall_precision_df = avg_recall_precision_df.with_columns(
        pl.col("model").replace(title_names)
    )

    for name, data in avg_recall_precision_df.group_by("query_type"):
        query_type = name[0]
        match query_type:
            case "raw_read":
                title = f"Recall @ {k} for Raw Read Queries vs Mutation Rate"
            case "logan_contig":
                title = f"Recall @ {k} for Logan Contig Queries v Mutation Rate"
            case "gencode":
                title = f"Recall @ {k} for Gencode Queries v Mutation Rate"

        chart = (
            alt.Chart(data)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "mutation_rate:Q",
                    title="Mutation Rate",
                    scale=alt.Scale(domain=[0, 0.1]),
                ),
                y=alt.Y(
                    "average_recall:Q",
                    title=f"Mean Recall@{k}",
                    scale=alt.Scale(domain=[0.2, 1.0]),
                ),
                color=alt.Color(
                    "model:N",
                    title="Model",
                    sort=["RawBERT", "LLM-ED", "Embed-Search-Align", "MetaGraph"],
                ).scale(scheme="viridis"),
            )
            .properties(
                title=title,
                width=650,
                height=450,
            )
            .configure_axis(labelFontSize=15, titleFontSize=20)
            .configure_legend(labelFontSize=14, titleFontSize=16)
        )
        chart.save(plots_dir / f"{query_type}_recall_at_{k}_vs_mut_rate_curve.png")


def plot_recall_vs_noise_bar(
    recall_precision_df: pl.DataFrame,
    plots_dir: Path,
    k: int = 7,
):
    avg_recall_precision_df = get_average_precision_recall_df(recall_precision_df)
    avg_recall_precision_df = avg_recall_precision_df.filter(pl.col("k") == k)
    avg_recall_precision_df = avg_recall_precision_df.filter(
        ~pl.col("model").is_in(["random", "oracle", "mmseqs"])
    )
    for name, data in avg_recall_precision_df.group_by("query_type"):
        query_type = name[0]
        match query_type:
            case "raw_read":
                title = f"Recall @ {k} for Raw Read Queries vs Mutation Rate"
            case "logan_contig":
                title = f"Recall @ {k} for Logan Contig Queries v Mutation Rate"
            case "gencode":
                title = f"Recall @ {k} for Gencode Queries v Mutation Rate"
        chart = (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x=alt.X("model:N"),
                y=alt.Y("average_recall:Q", title=f"Mean Recall@{k}"),
                column="mutation_rate:Q",
            )
            .properties(
                title=title,
                width=650,
                height=450,
            )
            .configure_axis(labelFontSize=15, titleFontSize=20)
            .configure_legend(labelFontSize=14, titleFontSize=16)
        )
        chart.save(plots_dir / f"{query_type}_recall_at_{k}_vs_mut_rate_bar.png")


def plot_recall_vs_time(
    recall_precision_df: pl.DataFrame,
    plots_dir: Path,
    k: int = 7,
    mutation_rate: float = 0.1,
):
    avg_recall_precision_df = get_average_precision_recall_df(recall_precision_df)
    avg_recall_precision_df = avg_recall_precision_df.filter(pl.col("k") == k)
    avg_recall_precision_df = avg_recall_precision_df.filter(
        pl.col("mutation_rate") == mutation_rate
    )
    avg_recall_precision_df = avg_recall_precision_df.filter(
        ~pl.col("model").is_in(["random", "oracle"])
    )
    for name, data in avg_recall_precision_df.group_by("query_type"):
        # data: model, checkpoint, max_len, checkpoint_step_num, chunk_type, avg_time
        query_type = name[0]
        match query_type:
            case "raw_read":
                title = f"Recall @ {k} for Raw Read Queries"
            case "logan_contig":
                title = f"Recall @ {k} for Logan Contig Queries"
            case "gencode":
                title = f"Recall @ {k} for Gencode Queries"
        chart = (
            alt.Chart(data)
            .mark_point()
            .encode(
                x=alt.X(
                    "avg_time:Q",
                    title="Query Time (Log Scale)",
                ).scale(type="log"),
                y=alt.Y("average_recall:Q", title=f"Mean Recall@{k}"),
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
        chart.save(plots_dir / f"{query_type}_recall_at_{k}_vs_time_scatterplot.png")


def main(
    results_dir: str,
    raw_read_queries_path: Path_fr,
    gencode_queries_path: Path_fr,
    plots_dir: str = "plots",
):
    results_dir: Path = Path(results_dir)
    raw_read_queries_path: Path = Path(raw_read_queries_path)
    gencode_queries_path: Path = Path(gencode_queries_path)
    plots_dir: Path = Path(plots_dir)
    if not plots_dir.is_dir():
        plots_dir.mkdir()
    data = []
    schema = pl.Schema(
        {
            "query_id": pl.String,
            "results": pl.List(
                pl.Struct({"accession": pl.String, "score": pl.Float64})
            ),
            "model": pl.String,
            "mutation_rate": pl.Float64,
            "query_type": pl.String,
            "checkpoint": pl.String,
            "max_len": pl.Int64,
            "checkpoint_step_num": pl.Int64,
            "chunk_type": pl.String,
            "avg_time": pl.Float64,
        }
    )
    for f in list(results_dir.rglob("*.parquet")):
        df = pl.read_parquet(f, schema=schema)
        data.append(df)
    data = pl.concat(data)
    raw_read_queries_df = pl.read_parquet(raw_read_queries_path)
    raw_read_oracle_data = raw_read_oracle_results(raw_read_queries_df)
    gencode_queries_df = pl.read_parquet(gencode_queries_path)
    gencode_oracle_data = gencode_oracle_results(gencode_queries_df)

    combos = data.select(
        "mutation_rate",
    ).unique()
    raw_read_oracle_data = raw_read_oracle_data.join(combos, how="cross")
    gencode_oracle_data = gencode_oracle_data.join(combos, how="cross")
    data = pl.concat([data, raw_read_oracle_data, gencode_oracle_data], how="diagonal")
    ground_truth = get_ground_truth(raw_read_queries_df, gencode_oracle_data, combos)
    recall_precision_df = calculate_recall_precision_df(ground_truth, data)
    plot_recall_precision(recall_precision_df, plots_dir)
    plot_recall_at_k(recall_precision_df, plots_dir)
    plot_auprc(recall_precision_df, plots_dir)
    plot_contig_len_hit_at_k(ground_truth, data, plots_dir)
    plot_recall_vs_time(recall_precision_df, plots_dir)
    plot_recall_vs_noise_line(recall_precision_df, plots_dir)
    plot_recall_vs_noise_bar(recall_precision_df, plots_dir)


if __name__ == "__main__":
    auto_cli(main)
