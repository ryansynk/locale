from jsonargparse import auto_cli
import polars as pl


def main(parquet_file: str, test_dataset: str):
    true_df = pl.read_ndjson(test_dataset)
    df = pl.read_parquet(parquet_file)
    df = df.join(true_df, on="transcript_id", how="left").rename(
        {"accession_list": "gt_accession", "seq": "query_seq"}
    )
    df = (
        df.with_columns(
            # Get the number of common elements
            intersection_count=pl.col("accession")
            .list.set_intersection("gt_accession")
            .list.len()
        )
        .with_columns(
            precision=pl.col("intersection_count") / pl.col("accession").list.len(),
            recall=pl.col("intersection_count") / pl.col("gt_accession").list.len(),
        )
        .drop("intersection_count")
    )

    precision = df.select(pl.mean("precision")).item()
    recall = df.select(pl.mean("recall")).item()

    print(f"Precision = {precision}. Recall = {recall}")


if __name__ == "__main__":
    auto_cli(main)
