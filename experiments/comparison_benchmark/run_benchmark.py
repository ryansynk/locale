from pathlib import Path

import polars as pl
import torch
from jsonargparse import CLI
from pyfaidx import Fasta
from src.config import DenseConfig, ExperimentConfig, MMSeqs2Config, SourMashConfig
from src.encoders import DenseEncoder, SourMashEncoder
from src.indexers import DenseIndexer, SourMashIndexer
from src.mmseqs2 import MMSeqs2Searcher


def remove_overlapping_distractors(
    query_df, distractor_df, num_distractors, min_overlap: int = 100
):
    joined = distractor_df.join(query_df, on="chromosome", suffix="_query")

    overlap_start = pl.max_horizontal("reference_start", "reference_start_query")
    overlap_end = pl.min_horizontal("reference_end", "reference_end_query")
    overlap_bp = (overlap_end - overlap_start).clip(lower_bound=0)

    matches = (
        joined.filter(overlap_bp >= min_overlap)
        .select(
            "read",
            "chromosome",
            "strand",
            "reference_start",
            "reference_end",
            "length",
            "free_length",
            "identity",
        )
        .unique()
    )

    distractor_df = distractor_df.join(matches, on=distractor_df.columns, how="anti")
    return distractor_df.sample(num_distractors)


def get_reference_seqs(reference_path, queries_df):
    reference_path: Path = Path(reference_path).resolve()
    reference = Fasta(reference_path)
    ref_seqs = []
    for row in queries_df.iter_rows(named=True):
        ref_seqs.append(
            str(
                reference[row["chromosome"]][
                    row["reference_start"] : row["reference_end"]
                ]
            ).upper()
        )
    return ref_seqs


def combine_data(
    reference_path,
    dataset_path,
    max_seq_len,
    num_distractors,
    identity_bins,
):
    dataset_path: Path = Path(dataset_path).resolve()
    df = pl.read_parquet(dataset_path).filter(
        (pl.col("strand") == "+") & (pl.col("length") <= max_seq_len)
    )
    queries_df = pl.concat(
        [
            df.filter((pl.col("identity") > lo) & (pl.col("identity") <= hi))
            .sample(n=100)
            .with_columns(
                pl.lit(f"({lo}, {hi}]").cast(pl.Categorical).alias("identity_bin")
            )
            for lo, hi in identity_bins
        ]
    )  # Remove exact rows of sampled from the original df
    candidate_distractors = df.join(queries_df, on=df.columns, how="anti")

    # Filter the rest of the rows by overlap of query df
    distractors_df = remove_overlapping_distractors(
        queries_df, candidate_distractors, num_distractors
    )

    queries = get_reference_seqs(reference_path, queries_df)
    keys = queries_df["read"].to_list() + distractors_df["read"].to_list()
    identities = queries_df["identity_bin"].to_list()
    return queries, keys, torch.arange(len(queries)), identities


def get_results(predictions, ground_truth_map, topks, identities):
    cols = {}
    cols["identity_bin"] = identities
    for k in topks:
        top_k_preds = predictions[..., :k]
        is_hit = (top_k_preds == ground_truth_map.unsqueeze(1)).any(dim=1)
        cols[f"hit_at_{k}"] = is_hit.to(torch.int).tolist()
    return pl.from_dict(cols)


def main(cfg: ExperimentConfig):
    assert cfg.reference_path is not None
    assert cfg.dataset_path is not None
    assert cfg.results_dir is not None

    # identity_bins = ([(0.7, 0.75), (0.75, 0.8), (0.8, 0.85), (0.9, 0.95), (0.95, 1.0)],)
    queries, keys, ground_truth_map, identities = combine_data(
        cfg.reference_path,
        cfg.dataset_path,
        cfg.max_seq_len,
        cfg.num_distractors,
        cfg.identity_bins,
    )
    if isinstance(cfg.model, MMSeqs2Config):
        # mmseqs2 is a monolithic CLI tool — no separate encode/index steps
        searcher = MMSeqs2Searcher(cfg.model)
        predictions = searcher.search(
            query_seqs=queries,
            target_seqs=keys,
            topk=max(cfg.topks),
        )
    else:
        if isinstance(cfg.model, SourMashConfig):
            encoder = SourMashEncoder(cfg.model)
            indexer = SourMashIndexer(cfg.model)
        elif isinstance(cfg.model, DenseConfig):
            encoder = DenseEncoder(cfg.model)
            indexer = DenseIndexer()
        else:
            raise ValueError("Unknown model config")

        target_features = encoder.encode(keys)
        indexer.build(target_features)

        query_features = encoder.encode(queries)

        predictions = indexer.search(
            query_features, topk=max(cfg.topks)
        )  # (num_queries, k)

    results = get_results(predictions, ground_truth_map, cfg.topks, identities)
    results = results.group_by("identity_bin").agg(
        pl.col(f"hit_at_{k}").mean() for k in cfg.topks
    )
    results = results.with_columns(
        pl.lit(cfg.model.name).alias("model"),
        pl.lit(cfg.num_distractors, dtype=pl.Int64),
    )
    fname = cfg.results_dir / f"{cfg.model.name}.parquet"
    if fname.exists():
        existing_results = pl.read_parquet(fname)
        combined = existing_results.vstack(results)
        combined.write_parquet(fname)
    else:
        if not cfg.results_dir.is_dir():
            cfg.results_dir.mkdir()
        results.write_parquet(cfg.results_dir / f"{cfg.model.name}.parquet")


if __name__ == "__main__":
    cfg = CLI(ExperimentConfig, as_positional=False)
    main(cfg)
