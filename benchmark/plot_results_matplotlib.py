from pathlib import Path

import polars as pl
from jsonargparse import auto_cli
from jsonargparse.typing import Path_fr

from matplotlib import pyplot as plt

# plt.rcParams.update({"text.usetex": True, "font.family": "mathptmx"})
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        "text.latex.preamble": r"\usepackage{mathptmx}",
    }
)


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
        models = mut_df["model"].unique().sort().to_list()
        cmap = plt.colormaps["tab10"]
        colors = [cmap(i) for i in range(10)]
        model_colors = {m: colors[i % len(colors)] for i, m in enumerate(models)}

        n_groups = len(sort_order)
        fig, axes = plt.subplots(1, n_groups, figsize=(3 * n_groups, 5), sharey=True)
        if n_groups == 1:
            axes = [axes]

        for ax, group in zip(axes, sort_order):
            group_df = mut_df.filter(pl.col("contig_len_group") == group)
            for i, model in enumerate(models):
                model_df = group_df.filter(pl.col("model") == model)
                val = model_df[f"hit_at_{k}"].item() if not model_df.is_empty() else 0.0
                ax.bar(i, val, color=model_colors[model], label=model)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=-45, ha="right", fontsize=9)
            ax.set_title(group, fontsize=9)
            ax.set_ylim(0, 1.0)
            ax.yaxis.grid(True)
            ax.set_axisbelow(True)

        axes[0].set_ylabel(f"Hit at {k}", fontsize=20)
        fig.suptitle("Length of Matching Contig", fontsize=18, y=1.02)
        handles = [plt.Rectangle((0, 0), 1, 1, color=model_colors[m]) for m in models]
        fig.legend(handles, models, loc="upper right", fontsize=10)
        plt.tight_layout()
        fig.savefig(
            plots_dir / f"mut_{mutation_rate}_contig_lens_bar_chart.png",
            bbox_inches="tight",
        )
        plt.close(fig)


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
        models = data["model"].unique().sort().to_list()
        cmap = plt.colormaps["tab10"]
        model_colors = {m: cmap(i) for i, m in enumerate(models)}

        fig, ax = plt.subplots(figsize=(8, 6))
        for model in models:
            mdf = data.filter(pl.col("model") == model).sort("average_recall")
            ax.plot(
                mdf["average_recall"].to_list(),
                mdf["average_precision"].to_list(),
                marker="o",
                label=model,
                color=model_colors[model],
            )
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Mean Recall@K", fontsize=20)
        ax.set_ylabel("Mean Precision@K", fontsize=20)
        ax.tick_params(labelsize=15)
        ax.set_title(title)
        ax.grid(True)
        ax.legend(title="Model", fontsize=14, title_fontsize=16)
        plt.tight_layout()
        fig.savefig(
            plots_dir
            / f"{query_type}_precision_recall_curve_mutation_rate_{str(mut_rate)}.png"
        )
        plt.close(fig)


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
        models = data["model"].unique().sort().to_list()
        cmap = plt.colormaps["tab10"]
        model_colors = {m: cmap(i) for i, m in enumerate(models)}

        fig, ax = plt.subplots(figsize=(8, 6))
        for model in models:
            mdf = data.filter(pl.col("model") == model).sort("k")
            ax.plot(
                mdf["k"].to_list(),
                mdf["average_recall"].to_list(),
                marker="o",
                label=model,
                color=model_colors[model],
            )
        ax.set_xlim(1, max_k)
        ax.set_xlabel("k", fontsize=20)
        ax.set_ylabel("Mean Recall@K", fontsize=20)
        ax.tick_params(labelsize=15)
        ax.set_title(title)
        ax.grid(True)
        ax.legend(title="Model", fontsize=14, title_fontsize=16)
        plt.tight_layout()
        fig.savefig(
            plots_dir / f"{query_type}_recall_vs_k_curve_mutation_rate_{mut_rate}.png"
        )
        plt.close(fig)


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
        mut_rates = sorted(df["mutation_rate"].unique().to_list())
        models = df["model"].unique().sort().to_list()
        cmap = plt.colormaps["tab10"]
        model_colors = {m: cmap(i) for i, m in enumerate(models)}

        fig, axes = plt.subplots(
            1, len(mut_rates), figsize=(4 * len(mut_rates), 5), sharey=True
        )
        if len(mut_rates) == 1:
            axes = [axes]

        for ax, mut_rate in zip(axes, mut_rates):
            mdf = df.filter(pl.col("mutation_rate") == mut_rate)
            for i, model in enumerate(models):
                row = mdf.filter(pl.col("model") == model)
                val = row["auprc"].item() if not row.is_empty() else 0.0
                ax.bar(i, val, color=model_colors[model], label=model)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=-45, ha="right", fontsize=9)
            ax.set_title(f"mut rate = {mut_rate}", fontsize=10)
            ax.set_ylim(0, 0.4)
            ax.yaxis.grid(True)
            ax.set_axisbelow(True)

        axes[0].set_ylabel("AUPRC", fontsize=16)
        handles = [plt.Rectangle((0, 0), 1, 1, color=model_colors[m]) for m in models]
        fig.legend(handles, models, loc="upper right", fontsize=10)
        plt.tight_layout()
        fig.savefig(
            plots_dir / f"{query_type}_auprc_bar_chart.png", bbox_inches="tight"
        )
        plt.close(fig)


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
        ~pl.col("model").is_in(["random", "oracle"])
    )
    avg_recall_precision_df = avg_recall_precision_df.with_columns(
        pl.col("model").str.split("_").list.get(0)
    )
    title_names = {
        "mmseqs": "MMseqs2",
        "llmed": "LLM-ED",
        "rawbert": "RawBERT",
        "metagraph": "MetaGraph",
        "dna2vec": "Embed-Search-Align",
    }
    avg_recall_precision_df = avg_recall_precision_df.with_columns(
        pl.col("model").replace(title_names)
    )
    avg_recall_precision_df = avg_recall_precision_df.with_columns(
        pl.col("mutation_rate") * 100
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

        model_order = ["RawBERT", "LLM-ED", "Embed-Search-Align", "MetaGraph"]
        models = [m for m in model_order if m in data["model"].to_list()]
        models += [
            m for m in data["model"].unique().sort().to_list() if m not in model_order
        ]
        viridis = plt.colormaps["viridis"]
        model_colors = {
            m: viridis(i / max(len(models) - 1, 1)) for i, m in enumerate(models)
        }

        fig, ax = plt.subplots(figsize=(8, 6))
        for model in models:
            mdf = data.filter(pl.col("model") == model).sort("mutation_rate")
            ax.plot(
                mdf["mutation_rate"].to_list(),
                mdf["average_recall"].to_list(),
                marker="o",
                label=model,
                color=model_colors[model],
            )
        ax.set_xlim(0, 10)
        ax.set_ylim(0.2, 1.0)
        ax.set_xlabel("Mutation Rate \%", fontsize=20)
        ax.set_ylabel(f"Mean Recall@{k}", fontsize=20)
        ax.tick_params(labelsize=15)
        ax.set_title(title)
        ax.grid(True)
        ax.legend(title="Model", fontsize=14, title_fontsize=16)
        plt.tight_layout()
        fig.savefig(plots_dir / f"{query_type}_recall_at_{k}_vs_mut_rate_curve.png")
        plt.close(fig)


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
        mut_rates = sorted(data["mutation_rate"].unique().to_list())
        models = data["model"].unique().sort().to_list()
        cmap = plt.colormaps["tab10"]
        model_colors = {m: cmap(i) for i, m in enumerate(models)}

        fig, axes = plt.subplots(
            1, len(mut_rates), figsize=(4 * len(mut_rates), 5), sharey=True
        )
        if len(mut_rates) == 1:
            axes = [axes]

        for ax, mut_rate in zip(axes, mut_rates):
            mdf = data.filter(pl.col("mutation_rate") == mut_rate)
            for i, model in enumerate(models):
                row = mdf.filter(pl.col("model") == model)
                val = row["average_recall"].item() if not row.is_empty() else 0.0
                ax.bar(i, val, color=model_colors[model], label=model)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=-45, ha="right", fontsize=9)
            ax.set_title(f"mut rate = {mut_rate}", fontsize=10)
            ax.yaxis.grid(True)
            ax.set_axisbelow(True)

        axes[0].set_ylabel(f"Mean Recall@{k}", fontsize=20)
        fig.suptitle(title, fontsize=16)
        handles = [plt.Rectangle((0, 0), 1, 1, color=model_colors[m]) for m in models]
        fig.legend(handles, models, loc="upper right", fontsize=10)
        plt.tight_layout()
        fig.savefig(
            plots_dir / f"{query_type}_recall_at_{k}_vs_mut_rate_bar.png",
            bbox_inches="tight",
        )
        plt.close(fig)


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
    avg_recall_precision_df = avg_recall_precision_df.with_columns(
        pl.col("model").str.split("_").list.get(0)
    )
    title_names = {
        "mmseqs": "MMseqs2",
        "llmed": "LLM-ED",
        "rawbert": "RawBERT",
        "metagraph": "MetaGraph",
        "dna2vec": "Embed-Search-Align",
    }
    avg_recall_precision_df = avg_recall_precision_df.with_columns(
        pl.col("model").replace(title_names)
    )
    for name, data in avg_recall_precision_df.group_by("query_type"):
        # data: model, checkpoint, max_len, checkpoint_step_num, chunk_type, avg_time
        query_type = name[0]
        match query_type:
            case "raw_read":
                title = f"Recall @ {k} v Query Time for Raw Read Queries"
            case "logan_contig":
                title = f"Recall @ {k} v Query Time for Logan Contig Queries"
            case "gencode":
                title = f"Recall @ {k} v Query Time for Gencode Queries"
        models = data["model"].unique().sort().to_list()
        cmap = plt.colormaps["tab10"]
        model_colors = {m: cmap(i) for i, m in enumerate(models)}

        fig, ax = plt.subplots(figsize=(8, 6))
        for model in models:
            mdf = data.filter(pl.col("model") == model)
            ax.scatter(
                mdf["avg_time"].to_list(),
                mdf["average_recall"].to_list(),
                label=model,
                color=model_colors[model],
            )
        ax.set_xscale("log")
        ax.set_xlabel("Query Time (s, Log Scale)", fontsize=20)
        ax.set_ylabel(f"Mean Recall@{k}", fontsize=20)
        ax.tick_params(labelsize=15)
        ax.set_title(title)
        ax.grid(True)
        ax.legend(title="Model", fontsize=14, title_fontsize=16, loc="lower right")
        plt.tight_layout()
        fig.savefig(plots_dir / f"{query_type}_recall_at_{k}_vs_time_scatterplot.png")
        plt.close(fig)


def main(
    results_dir: str,
    raw_read_queries_path: Path_fr,
    gencode_queries_path: Path_fr,
    plots_dir: str = "plots_matplotlib",
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
