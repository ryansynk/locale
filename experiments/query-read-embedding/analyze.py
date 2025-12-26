import polars as pl
from jsonargparse import auto_cli


def main(parquet_file: str, test_dataset: str):
    true_df = pl.read_ndjson(test_dataset)
    df = pl.read_parquet(parquet_file)
    df = df.join(true_df, on="transcript_id", how="left").rename(
        {"accession_list": "gt_accession", "seq": "query_seq"}
    )
    df = df.with_columns(
        # Get the number of common elements
        tp_count=pl.col("accession").list.set_intersection("gt_accession").list.len(),
        fp_count=pl.col("accession").list.set_difference("gt_accession").list.len(),
    ).with_columns(
        precision=pl.col("tp_count") / pl.col("accession").list.len(),
        recall=pl.col("tp_count") / pl.col("gt_accession").list.len(),
    )

    precision = df.select(pl.mean("precision")).item()
    recall = df.select(pl.mean("recall")).item()
    tp_rate = df.select(pl.mean("tp_count")).item()
    fp_rate = df.select(pl.mean("fp_count")).item()

    print(f"Precision = {precision}. Recall = {recall}")
    print(f"True Positive Rate = {tp_rate}. False Positive Rate = {fp_rate}")


if __name__ == "__main__":
    auto_cli(main)
