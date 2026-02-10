import edlib
import polars as pl
import torch
from jsonargparse import CLI
from src.config import DenseConfig, ExperimentConfig, SourMashConfig
from src.encoders import DenseEncoder, SourMashEncoder
from src.indexers import DenseIndexer, SourMashIndexer
from tqdm import tqdm


def load_data(
    dataset_path: str,
    num_keys: int,
    num_queries: int,
    max_seq_len: int,
    min_coverage: float,
):
    df = pl.read_parquet(dataset_path)
    df = df.filter(
        (pl.col("max_len") < max_seq_len) & (pl.col("coverage") > min_coverage)
    )
    df = df.head(num_keys)
    query_ids = df["query_name"].head(num_queries).to_list()
    queries = df["query_seq"].head(num_queries).to_list()
    target_ids = df["reference_name"].to_list()
    targets = df["reference_seq"].to_list()
    return queries, query_ids, targets, target_ids, torch.arange(num_queries)


def filter_aligned_sequences(query_seqs, target_seqs, target_ids, alignment_threshold):
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

    return target_seqs, target_ids, valid_targets_mask


def calculate_hit_at_k(predictions, ground_truth_map, k):
    predictions = predictions[..., :k]
    is_hit = (predictions == ground_truth_map.unsqueeze(1)).any(dim=1)
    return is_hit.float().mean().item()


def main(cfg: ExperimentConfig):
    queries, query_ids, targets, target_ids, ground_truth_map = load_data(
        cfg.dataset_path,
        cfg.num_keys,
        cfg.num_queries,
        cfg.max_seq_len,
        cfg.min_coverage,
    )

    # Filter out targets that are aligned above threshold to queries (per-query basis)
    valid_targets_mask = None
    print(f"Filtering targets with alignment threshold: {cfg.similarity_threshold}")
    targets, target_ids, valid_targets_mask = filter_aligned_sequences(
        queries, targets, target_ids, cfg.similarity_threshold
    )
    # Ground truth map stays the same - no need to update indices

    if isinstance(cfg.model, SourMashConfig):
        encoder = SourMashEncoder(cfg.model)
        indexer = SourMashIndexer(cfg.model)
    elif isinstance(cfg.model, DenseConfig):
        encoder = DenseEncoder(cfg.model)
        indexer = DenseIndexer()
    else:
        raise ValueError("Unknown model config")

    target_features = encoder.encode(targets)
    indexer.build(target_features, target_ids)

    query_features = encoder.encode(queries)

    predictions = indexer.search(
        query_features, topk=max(cfg.topks), valid_targets_mask=valid_targets_mask
    )  # (num_queries, k)

    recalls = [
        calculate_hit_at_k(predictions, ground_truth_map, k=topk) for topk in cfg.topks
    ]
    for recall, topk in zip(recalls, cfg.topks):
        print(f"Recall @{topk} for {cfg.model}: {(recall * 100):.2f}%")


if __name__ == "__main__":
    cfg = CLI(ExperimentConfig, as_positional=False)
    main(cfg)
