import polars as pl
import torch
from jsonargparse import CLI
from src.config import DenseConfig, ExperimentConfig, SourMashConfig
from src.encoders import DenseEncoder, SourMashEncoder
from src.indexers import DenseIndexer, SourMashIndexer


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

    if isinstance(cfg.model, SourMashConfig):
        encoder = SourMashEncoder(cfg.model)
        indexer = SourMashIndexer()
    elif isinstance(cfg.model, DenseConfig):
        encoder = DenseEncoder(cfg.model)
        indexer = DenseIndexer()
    else:
        raise ValueError("Unknown model config")

    target_features = encoder.encode(targets)
    indexer.build(target_features, target_ids)

    query_features = encoder.encode(queries)

    predictions = indexer.search(
        query_features, topk=max(cfg.topks)
    )  # (num_queries, k)
    recalls = [
        calculate_hit_at_k(predictions, ground_truth_map, k=topk) for topk in cfg.topks
    ]
    for recall, topk in zip(recalls, cfg.topks):
        print(f"Recall @{topk} for {cfg.model}: {(recall * 100):.2f}%")


if __name__ == "__main__":
    cfg = CLI(ExperimentConfig, as_positional=False)
    main(cfg)
