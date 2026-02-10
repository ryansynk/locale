import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import altair as alt
import edlib  # ty: ignore unresolved-import
import numpy as np
import polars as pl
from jsonargparse import CLI
from tqdm import tqdm

from rawbert.config import TrainConfig
from rawbert.training.supervised_batcher import SupervisedBatcher


def get_similarity(seq1, seq2):
    if len(seq1) >= len(seq2):
        query = seq2
        target = seq1
    else:
        query = seq1
        target = seq2

    result = edlib.align(query=query, target=target, mode="HW", task="distance")
    similarity = 1.0 - (result["editDistance"] / len(query))
    return similarity


def get_similarities_for_query(q, ref_seqs):
    return [get_similarity(q, ref_seq) for ref_seq in ref_seqs]


def main(cfg_file: TrainConfig):
    """
    Exploratory data analysis. Gets histogram of edit distance similarities between
    aligned and unaligned sequences in order to get a background noise level.
    """
    reader = SupervisedBatcher(
        cfg.val_dataset_path, cfg.augment_config, num_examples=cfg.num_val_keys
    )

    query_seqs = []
    ref_seqs = []
    print("Adding sequences to list")
    for i in range(len(reader)):
        query, ref = reader[i]
        if len(query_seqs) < cfg.num_val_queries:
            query_seqs.append(query)
        ref_seqs.append(ref)

    print("Beginning executor")
    max_workers = os.cpu_count()
    negative_similarities = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, q in enumerate(query_seqs):
            futures.append(
                executor.submit(
                    get_similarities_for_query, q, ref_seqs[:i] + ref_seqs[(i + 1) :]
                )
            )

        for f in tqdm(
            as_completed(futures), total=len(futures), desc="Calculating distance..."
        ):
            negative_similarities.extend(f.result())

    positive_similarities = []
    for query_seq, ref_seq in zip(query_seqs, ref_seqs[: len(query_seqs)]):
        positive_similarities.append(get_similarity(query_seq, ref_seq))
    negative_df = pl.DataFrame(
        {
            "similarity": negative_similarities,
            "type": ["negative"] * len(negative_similarities),
        }
    )
    positive_df = pl.DataFrame(
        {
            "similarity": positive_similarities,
            "type": ["positive"] * len(positive_similarities),
        }
    )

    bins = np.linspace(0.3, 1.0, 21)[:-1]
    dfs = []
    for label, df in zip(["negative", "positive"], [negative_df, positive_df]):
        # hist_df = df["similarity"].hist(bin_count=10)
        hist_df = df["similarity"].hist(bins=bins)
        hist_df = hist_df.with_columns(
            [
                pl.col("category")
                .cast(pl.Utf8)
                .str.strip_chars("[(]")
                .str.split(", ")
                .list.get(0)
                .cast(pl.Float64)
                .alias("bin_start"),
                pl.col("category")
                .cast(pl.Utf8)
                .str.strip_chars("]]")
                .str.split(", ")
                .list.get(1)
                .cast(pl.Float64)
                .alias("bin_end"),
            ]
        ).drop("category")
        hist_df = hist_df.with_columns(
            (pl.col("count") / pl.col("count").sum()).alias("probability")
        )
        hist_df = hist_df.with_columns(pl.lit(label).alias("type"))
        dfs.append(hist_df)
    df = pl.concat(dfs)
    chart = (
        alt.Chart(df)
        .mark_bar(opacity=0.3)
        .encode(
            x=alt.X("bin_start:Q", bin="binned"),
            x2="bin_end",
            y=alt.Y("probability:Q").stack(None),
            color=alt.Color("type:N"),
        )
    )
    chart.save("similarity_hist.png")


if __name__ == "__main__":
    cfg = CLI(TrainConfig, as_positional=False)
    main(cfg)
