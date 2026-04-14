from pathlib import Path
from jsonargparse import auto_cli
from jsonargparse.typing import Path_fr
import polars as pl
import matplotlib.pyplot as plt
import numpy as np


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
    )
    oracle_raw_read_data_with_contig_len = oracle_raw_read_data_with_contig_len.join(
        combos, how="cross"
    )
    return pl.concat(
        [oracle_raw_read_data_with_contig_len, gencode_oracle_data], how="diagonal"
    )


def main(
    results_parquet: Path_fr,
    raw_read_queries_path: Path_fr,
    metadata_tsv_path: Path_fr,
    plots_dir: str = "error_plots",
):
    results_parquet: Path = Path(results_parquet)
    raw_read_queries_path: Path = Path(raw_read_queries_path)
    metadata_tsv_path: Path = Path(metadata_tsv_path)
    plots_dir: Path = Path(plots_dir)
    if not plots_dir.is_dir():
        plots_dir.mkdir()
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
        }
    )
    data = pl.read_parquet(results_parquet, schema=schema)
    raw_read_queries_df = pl.read_parquet(raw_read_queries_path)
    metadata = pl.read_csv(
        source=metadata_tsv_path,
        ignore_errors=True,
        separator="\t",
    )

    data = data.filter(pl.col("query_type") == "raw_read")
    if data.is_empty():
        return
    results_df = (
        data.explode("results")
        .unnest("results")
        .rename({"accession": "retrieved_accession"})
        .sort(["model", "mutation_rate", "query_id", "score"], descending=True)
        .group_by("model", "mutation_rate", "query_id", maintain_order=True)
        .agg(pl.col("retrieved_accession"))
    )
    results_df = results_df.with_columns(
        pl.col("query_id").str.split(".").list.get(0).alias("true_accession")
    )
    results_df = results_df.with_columns(
        pl.col("retrieved_accession").list.get(0).alias("top_retrieved_accession")
    )
    results_df = results_df.join(
        metadata.select(["acc", "organism"]).rename(
            {"acc": "true_accession", "organism": "true_organism"}
        ),
        on="true_accession",
        how="left",
    ).join(
        metadata.select(["acc", "organism"]).rename(
            {"acc": "top_retrieved_accession", "organism": "retrieved_organism"}
        ),
        on="top_retrieved_accession",
        how="left",
    )

    for (model, mutation_rate), group in results_df.group_by(
        ["model", "mutation_rate"], maintain_order=True
    ):
        # cm_df = (
        #    group.filter(
        #        pl.col("true_organism").ne_missing(pl.col("retrieved_organism"))
        #    )
        #    .select("true_accession", "top_retrieved_accession")
        #    .group_by("true_accession", "top_retrieved_accession")
        #    .len()
        # )
        cm_df = (
            group.select("true_accession", "top_retrieved_accession")
            .group_by("true_accession", "top_retrieved_accession")
            .len()
        )

        all_accessions = sorted(
            set(cm_df["true_accession"].to_list())
            | set(cm_df["top_retrieved_accession"].to_list())
        )
        idx = {acc: i for i, acc in enumerate(all_accessions)}
        n = len(all_accessions)

        matrix = np.zeros((n, n), dtype=int)
        for row in cm_df.iter_rows(named=True):
            i = idx[row["true_accession"]]
            j = idx[row["top_retrieved_accession"]]
            matrix[i, j] = row["len"]

        fig, ax = plt.subplots(figsize=(max(8, n * 0.4), max(6, n * 0.35)))
        im = ax.imshow(matrix, aspect="auto", cmap="Blues")
        plt.colorbar(im, ax=ax, label="count")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(all_accessions, rotation=90, fontsize=6)
        ax.set_yticklabels(all_accessions, fontsize=6)
        ax.set_xlabel("top_retrieved_accession")
        ax.set_ylabel("true_accession")
        ax.set_title(f"Confusion matrix — model={model}, mutation_rate={mutation_rate}")

        plt.tight_layout()
        safe_model = str(model).replace("/", "_")
        out_path = plots_dir / f"confusion_{safe_model}_mut{mutation_rate}.png"
        fig.savefig(out_path, dpi=1000)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    auto_cli(main)
