from typing import List

import torch
from sourmash.index import IndexSearchResult, LinearIndex
from tqdm import tqdm

from .config import SourMashConfig


class BaseIndexer:
    def build(self, features):
        raise NotImplementedError

    def search(self, query_features, topk, valid_targets_mask):
        raise NotImplementedError


class DenseIndexer(BaseIndexer):
    def __init__(self):
        pass

    def build(self, features):
        self.index = features

    @torch.no_grad()
    def search(self, query_features, topk, valid_targets_mask):
        logits = torch.matmul(query_features, self.index.T)  # (num_queries, num_keys)
        if valid_targets_mask is not None:
            logits[~valid_targets_mask] = -1e9
        _, indices = logits.topk(k=topk, dim=1)  # (num_queries, k)
        return indices.cpu()


class SourMashIndexer:
    def __init__(self, cfg: SourMashConfig):
        """
        Constructs sourmash index.
        threshold=0.0 ensures we get candidates even if they are distant,
        allowing us to fill the top-k buffer.
        """
        self.index = None
        self.threshold = cfg.threshold
        self.num_targets = 0

    def build(self, features):
        """
        Given set of hashes from sourmash and the ids of the sequences, creates an index.

        Note: We overwrite the signature names with their integer indices (0...N)
        to allow O(1) retrieval of the index during search.
        """
        self.num_targets = len(features)

        # We embed the integer index into the signature name for retrieval later
        for i, sig in enumerate(features):
            sig._name = str(i)

        self.index = LinearIndex(features)

    def search(self, query_features, topk, valid_targets_mask):
        """
        Given representation of query sequences and a topk, returns the topk INDICES.

        Returns:
            torch.Tensor: Shape (num_queries, k) containing integer indices.
                          Padded with -1 if fewer than k matches are found.
        """
        num_queries = len(query_features)

        # Initialize tensor with -1 (padding value)
        # Using long (int64) which is standard for indices in PyTorch
        predictions = torch.full((num_queries, topk), -1, dtype=torch.long)

        for i, query_sig in tqdm(
            enumerate(query_features), total=num_queries, desc="Searching queries..."
        ):
            # sourmash search returns list of (score, signature, filename)
            results: List[IndexSearchResult] = self.index.search(
                query_sig, threshold=self.threshold
            )
            filtered_results: List[IndexSearchResult] = []
            # Mask out invalid targets by setting their score to -1e9
            if valid_targets_mask is not None:
                for j, res in enumerate(results):
                    # target_idx = int(res.signature.name)
                    if valid_targets_mask[i, j]:
                        filtered_res = IndexSearchResult(
                            -1e9, res.signature, res.location
                        )
                        filtered_results.append(filtered_res)
                    else:
                        filtered_results.append(res)

            # Sort by similarity (score) descending
            filtered_results.sort(key=lambda x: x.score, reverse=True)

            # Keep only topk results
            top_results = filtered_results[:topk]

            # Extract the integer indices stored in signature names
            # We iterate only up to len(top_results) in case we found fewer than k matches
            for j, res in enumerate(top_results):
                # res.signature.name contains the string version of the index (e.g., "42")
                original_idx = int(res.signature.name)
                predictions[i, j] = original_idx

        return predictions
