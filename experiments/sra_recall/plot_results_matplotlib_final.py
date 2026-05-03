from pathlib import Path

import numpy as np
import polars as pl
from jsonargparse import auto_cli
from jsonargparse.typing import Path_fr
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score

# plt.rcParams.update({"text.usetex": True, "font.family": "mathptmx"})
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        "text.latex.preamble": r"\usepackage{mathptmx}",
    }
)


def r_precision_per_query(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute R-precision for a single query.

    y_true: binary array of length |U|
    y_score: score array of length |U|
    """
    n_relevant = int(y_true.sum())
    if n_relevant == 0:
        return np.nan  # or skip

    top_nrelevant_indices = (-y_score).argsort()[:n_relevant]

    # Count how many of those are relevant
    n_retrieved_relevant = int(y_true[top_nrelevant_indices].sum())

    return n_retrieved_relevant / n_relevant


def recall_at_k_per_query(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Compute recall at k for a single query.

    y_true: binary array of length |U|
    y_score: score array of length |U|
    """
    n_relevant = int(y_true.sum())
    if n_relevant == 0:
        return np.nan  # or skip

    top_k_indices = (-y_score).argsort()[:k]

    # Count how many of those are relevant
    n_retrieved_relevant = int(y_true[top_k_indices].sum())

    return n_retrieved_relevant / n_relevant


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


def print_auprc(
    data: pl.DataFrame,
    ground_truth: pl.DataFrame,
    accessions: list[str],
):
    accession_order = sorted(accessions)  # canonical, stable ordering
    acc_to_idx = {acc: i for i, acc in enumerate(accession_order)}
    n_acc = len(accession_order)

    DEFAULT_SCORE = -2.0
    auprc_rows = []
    for name, df in data.group_by(
        [
            "model",
            "checkpoint",
            "max_len",
            "checkpoint_step_num",
            "chunk_type",
            "mutation_rate",
            "query_type",
        ]
    ):
        joined_true_pred = (
            ground_truth.filter(pl.col("mutation_rate") == 0.0)
            .select(["query_id", "results"])
            .rename({"results": "true_results"})
            .join(
                df.select(["query_id", "results"]).rename({"results": "pred_results"}),
                on="query_id",
            )
        )

        ap_per_query = []
        for row in joined_true_pred.iter_rows(named=True):
            # Build score vector with defaults
            y_score = np.full(n_acc, DEFAULT_SCORE)
            for p in row["pred_results"]:
                if p["accession"] in acc_to_idx:
                    y_score[acc_to_idx[p["accession"]]] = p["score"]

            # Build truth vector
            y_true = np.zeros(n_acc, dtype=int)
            for t in row["true_results"]:
                if t["accession"] in acc_to_idx:
                    y_true[acc_to_idx[t["accession"]]] = 1

            if y_true.sum() > 0:
                ap_per_query.append(average_precision_score(y_true, y_score))

        auprc_rows.append(
            {
                "model": name[0],
                "checkpoint": name[1],
                "max_len": name[2],
                "checkpoint_step_num": name[3],
                "chunk_type": name[4],
                "mutation_rate": name[5],
                "query_type": name[6],
                "auprc": np.mean(ap_per_query),
            }
        )

    auprc_df = pl.from_dicts(auprc_rows)
    pl.Config.set_tbl_rows(len(auprc_df))
    print("=========== AUPRC DATA =============")
    print(auprc_df.sort("model", "mutation_rate"))


def plot_r_precision_vs_noise_line(
    data: pl.DataFrame,
    ground_truth: pl.DataFrame,
    accessions: list[str],
    plots_dir: Path,
    print_data: bool = True,
):
    data = data.filter(~pl.col("model").is_in(["random", "oracle"]))
    data = data.with_columns(pl.col("model").str.split("_").list.get(0))
    title_names = {
        "mmseqs": "MMseqs2",
        "llmed": "LLM-ED",
        "rawbert": "RawBERT",
        "metagraph": "MetaGraph",
        "dna2vec": "Embed-Search-Align",
    }
    data = data.with_columns(pl.col("model").replace(title_names))
    data = data.with_columns(pl.col("mutation_rate") * 100)

    accession_order = sorted(accessions)  # canonical, stable ordering
    acc_to_idx = {acc: i for i, acc in enumerate(accession_order)}
    n_acc = len(accession_order)

    DEFAULT_SCORE = -2.0
    r_precision_rows = []
    for name, df in data.group_by(
        [
            "model",
            "checkpoint",
            "max_len",
            "checkpoint_step_num",
            "chunk_type",
            "mutation_rate",
            "query_type",
        ]
    ):
        joined_true_pred = (
            ground_truth.filter(pl.col("mutation_rate") == 0.0)
            .select(["query_id", "results"])
            .rename({"results": "true_results"})
            .join(
                df.select(["query_id", "results"]).rename({"results": "pred_results"}),
                on="query_id",
            )
        )

        r_precision_per_query_array = []
        for row in joined_true_pred.iter_rows(named=True):
            # Build score vector with defaults
            y_score = np.full(n_acc, DEFAULT_SCORE)
            for p in row["pred_results"]:
                if p["accession"] in acc_to_idx:
                    y_score[acc_to_idx[p["accession"]]] = p["score"]

            # Build truth vector
            y_true = np.zeros(n_acc, dtype=int)
            for t in row["true_results"]:
                if t["accession"] in acc_to_idx:
                    y_true[acc_to_idx[t["accession"]]] = 1

            if y_true.sum() > 0:
                r_precision_per_query_array.append(
                    r_precision_per_query(y_true, y_score)
                )

        r_precision_rows.append(
            {
                "model": name[0],
                "checkpoint": name[1],
                "max_len": name[2],
                "checkpoint_step_num": name[3],
                "chunk_type": name[4],
                "mutation_rate": name[5],
                "query_type": name[6],
                "average_recall": np.mean(r_precision_per_query_array),
            }
        )

    r_precision_df = pl.from_dicts(r_precision_rows)

    if print_data:
        print("=========== R PRECISION DATA =============")
        pl.Config.set_tbl_rows(len(r_precision_df))
        print(
            r_precision_df.sort("model", "mutation_rate").select(
                ["model", "mutation_rate", "average_recall"]
            )
        )

    for name, data in r_precision_df.group_by("query_type"):
        query_type = name[0]

        model_order = [
            "MMseqs2",
            "RawBERT",
            "LLM-ED",
            "Embed-Search-Align",
            "MetaGraph",
        ]
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
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Mutation Rate \%", fontsize=20)
        ax.set_ylabel("Average Recall@$|\mathcal{A}_q|$ (R-Precision)", fontsize=20)
        ax.tick_params(labelsize=15)
        ax.grid(True)
        ax.legend(title="Model", fontsize=14, title_fontsize=16)
        plt.tight_layout()
        fig.savefig(plots_dir / f"{query_type}_recall_at_Aq_vs_mut_rate_curve.pdf")
        plt.close(fig)


def plot_recall_at_k_vs_noise_line(
    data: pl.DataFrame,
    ground_truth: pl.DataFrame,
    accessions: list[str],
    k: int,
    plots_dir: Path,
    print_data: bool = True,
):
    data = data.filter(~pl.col("model").is_in(["random", "oracle"]))
    data = data.with_columns(pl.col("model").str.split("_").list.get(0))
    title_names = {
        "mmseqs": "MMseqs2",
        "llmed": "LLM-ED",
        "rawbert": "RawBERT",
        "metagraph": "MetaGraph",
        "dna2vec": "Embed-Search-Align",
    }
    data = data.with_columns(pl.col("model").replace(title_names))
    data = data.with_columns(pl.col("mutation_rate") * 100)

    accession_order = sorted(accessions)  # canonical, stable ordering
    acc_to_idx = {acc: i for i, acc in enumerate(accession_order)}
    n_acc = len(accession_order)

    DEFAULT_SCORE = -2.0
    recall_at_k_rows = []

    for name, df in data.group_by(
        [
            "model",
            "checkpoint",
            "max_len",
            "checkpoint_step_num",
            "chunk_type",
            "mutation_rate",
            "query_type",
        ]
    ):
        joined_true_pred = (
            ground_truth.filter(pl.col("mutation_rate") == 0.0)
            .select(["query_id", "results"])
            .rename({"results": "true_results"})
            .join(
                df.select(["query_id", "results"]).rename({"results": "pred_results"}),
                on="query_id",
            )
        )

        recalls_at_k = []
        for row in joined_true_pred.iter_rows(named=True):
            # Build score vector with defaults
            y_score = np.full(n_acc, DEFAULT_SCORE)
            for p in row["pred_results"]:
                if p["accession"] in acc_to_idx:
                    y_score[acc_to_idx[p["accession"]]] = p["score"]

            # Build truth vector
            y_true = np.zeros(n_acc, dtype=int)
            for t in row["true_results"]:
                if t["accession"] in acc_to_idx:
                    y_true[acc_to_idx[t["accession"]]] = 1

            if y_true.sum() > 0:
                recalls_at_k.append(recall_at_k_per_query(y_true, y_score, k))

        recall_at_k_rows.append(
            {
                "model": name[0],
                "checkpoint": name[1],
                "max_len": name[2],
                "checkpoint_step_num": name[3],
                "chunk_type": name[4],
                "mutation_rate": name[5],
                "query_type": name[6],
                "average_recall": np.mean(recalls_at_k),
                "k": k,
            }
        )

    recall_at_k_df = pl.from_dicts(recall_at_k_rows)

    if print_data:
        pl.Config.set_tbl_rows(len(recall_at_k_df))
        print(f"=========== RECALL AT {k} DATA =============")
        print(
            recall_at_k_df.sort("model", "mutation_rate").select(
                ["model", "mutation_rate", "average_recall", "k"]
            )
        )

    for name, data in recall_at_k_df.group_by("query_type"):
        query_type = name[0]

        model_order = [
            "MMseqs2",
            "RawBERT",
            "LLM-ED",
            "Embed-Search-Align",
            "MetaGraph",
        ]
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
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Mutation Rate \%", fontsize=20)
        ax.set_ylabel(f"Average Recall@{k} ", fontsize=20)
        ax.tick_params(labelsize=15)
        ax.grid(True)
        ax.legend(title="Model", fontsize=14, title_fontsize=16)
        plt.tight_layout()
        fig.savefig(plots_dir / f"{query_type}_recall_at_{k}_vs_mut_rate_curve.pdf")
        plt.close(fig)


def main(
    results_dir: str,
    raw_read_queries_path: Path_fr,
    accessions: Path_fr,
    plots_dir: str = "plots_matplotlib",
    k: int = 10,
):
    results_dir: Path = Path(results_dir)
    raw_read_queries_path: Path = Path(raw_read_queries_path)
    with open(accessions) as f:
        accs = f.read().splitlines()

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
            "index_size_gb": pl.Float64,
        }
    )
    for f in list(results_dir.rglob("*.parquet")):
        df = pl.read_parquet(f, schema=schema)
        data.append(df)
    data = pl.concat(data)
    raw_read_queries_df = pl.read_parquet(raw_read_queries_path)
    raw_read_oracle_data = raw_read_oracle_results(raw_read_queries_df)

    combos = data.select(
        "mutation_rate",
    ).unique()
    raw_read_oracle_data = raw_read_oracle_data.join(combos, how="cross")
    # breakpoint()
    # data = pl.concat([data, raw_read_oracle_data], how="diagonal")
    plot_r_precision_vs_noise_line(data, raw_read_oracle_data, accs, plots_dir)
    plot_recall_at_k_vs_noise_line(data, raw_read_oracle_data, accs, k, plots_dir)
    print_auprc(data, raw_read_oracle_data, accs)


if __name__ == "__main__":
    auto_cli(main)
