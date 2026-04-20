import polars as pl


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


def _build_oracle_raw_read_df(raw_read_queries_df):
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
    return oracle_raw_read_data_with_contig_len


def get_ground_truth_no_gencode(raw_read_queries_df):
    return _build_oracle_raw_read_df(raw_read_queries_df=raw_read_queries_df)


def get_ground_truth(raw_read_queries_df, gencode_oracle_data, combos):
    df = _build_oracle_raw_read_df(raw_read_queries_df).join(combos, how="cross")
    return pl.concat([df, gencode_oracle_data], how="diagonal")
