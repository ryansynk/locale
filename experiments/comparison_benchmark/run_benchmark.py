import edlib
import polars as pl
import torch
from jsonargparse import CLI
from src.config import DenseConfig, ExperimentConfig, MMSeqs2Config, SourMashConfig
from src.encoders import DenseEncoder, SourMashEncoder
from src.indexers import DenseIndexer, SourMashIndexer
from src.mmseqs2 import MMSeqs2Searcher
from tqdm import tqdm


def filter_aligned_sequences(query_seqs, target_seqs, alignment_threshold):
    """
    Filter out target sequences that are aligned above threshold on a per-query basis.

    Args:
        query_seqs: List of query DNA sequences
        target_seqs: List of target DNA sequences
        target_ids: List of target IDs
        alignment_threshold: Minimum similarity to consider sequences aligned

    Returns:
        Tuple of (target_seqs, target_ids, valid_targets_mask)
        where valid_targets_mask is a boolean tensor of shape (num_queries, num_targets)
    """
    num_queries = len(query_seqs)
    num_targets = len(target_seqs)

    # Track which targets are valid for each query
    valid_targets_mask = torch.ones(num_queries, num_targets, dtype=torch.bool)

    total_filtered = 0
    for i in tqdm(range(num_queries), total=num_queries, desc="Filtering..."):
        query_seq = query_seqs[i]

        for j in range(num_targets):
            # Skip the diagonal (true positive pair)
            if i == j:
                continue

            target_seq = target_seqs[j]

            # Check if sequences are aligned using edlib
            # short query, long target
            if len(query_seq) <= len(target_seq):
                q = query_seq
                t = target_seq
            else:
                q = target_seq
                t = query_seq
            result = edlib.align(query=q, target=t, mode="HW", task="distance")
            edit_distance = result["editDistance"]

            # Calculate similarity
            similarity = 1.0 - (edit_distance / len(q))

            # If aligned above threshold, mark as invalid for this query
            if similarity >= alignment_threshold:
                valid_targets_mask[i, j] = False
                total_filtered += 1

    print(
        f"Filtered {total_filtered} query-target pairs (avg {total_filtered / num_queries:.1f} per query)"
    )

    return valid_targets_mask


def combine_data(alignments, distractors, num_distractors):
    queries = alignments["query_seq"].to_list()
    query_ids = alignments["query_name"].to_list()
    keys = alignments["reference_seq"].to_list()
    similarities = alignments["similarity"].to_list()
    distractor_keys = distractors["seq"].to_list()
    all_keys = keys + distractor_keys[:num_distractors]
    return queries, all_keys, torch.arange(len(queries)), similarities, query_ids


def get_results(predictions, ground_truth_map, topks, similarities, query_ids):
    cols = {}
    cols["query_id"] = query_ids
    cols["similarity"] = similarities
    for k in topks:
        top_k_preds = predictions[..., :k]
        is_hit = (top_k_preds == ground_truth_map.unsqueeze(1)).any(dim=1)
        cols[f"hit_at_{k}"] = is_hit.to(torch.int).tolist()
    return pl.from_dict(cols)


def main(cfg: ExperimentConfig):
    assert cfg.alignments_path is not None
    assert cfg.distractors_path is not None
    assert cfg.results_dir is not None

    alignments = pl.read_parquet(cfg.alignments_path)
    distractors = pl.read_parquet(cfg.distractors_path)
    queries, keys, ground_truth_map, similarities, query_ids = combine_data(
        alignments, distractors, cfg.num_distractors
    )
    valid_targets_mask = filter_aligned_sequences(
        queries, keys[: len(queries)], cfg.similarity_threshold
    )
    valid_targets_mask = torch.cat(
        [
            valid_targets_mask,
            torch.zeros(len(queries), len(keys) - len(queries), dtype=torch.bool),
        ],
        dim=1,
    )
    # Ground truth map stays the same - no need to update indices

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
            query_features, topk=max(cfg.topks), valid_targets_mask=valid_targets_mask
        )  # (num_queries, k)

    results = get_results(
        predictions, ground_truth_map, cfg.topks, similarities, query_ids
    )
    results = (
        results.with_columns(
            pl.col("similarity")
            .cut([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            .alias("similarity_bin")
        )
        .group_by("similarity_bin")
        .agg(pl.col(f"hit_at_{k}").mean() for k in cfg.topks)
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
