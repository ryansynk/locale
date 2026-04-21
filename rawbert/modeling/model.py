"""
This file contains code adapted from the Moco repository:
https://github.com/facebookresearch/moco/tree/main

Original Author: Kaiming He, Yuxin Wu
License: MIT License

Code has been modified for DNA sequence data
"""

import concurrent.futures
import logging
import math
import os
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

import edlib  # ty: ignore unresolved-import
import einops
import torch
import torch.nn as nn
from transformers import BertConfig

from rawbert.modeling.bert_layers import BertModel as DNABertModel

# Try to import the specific varlen function from flash_attn
try:
    FLASH_ATTN_AVAILABLE = True
    from rawbert.utils.patch import patch_with_flash_lib
except ImportError:
    FLASH_ATTN_AVAILABLE = False

logger = logging.getLogger(__name__)


class RawBERT(nn.Module):
    def __init__(
        self,
        pooling: str,
        dim: int = 128,
        K: int = 4096,
        m: float = 0.999,
        T: float = 0.07,
        use_projection_head: bool = False,
    ):
        super().__init__()
        self.config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        if not hasattr(self.config, "pad_token_id") or self.config.pad_token_id is None:
            self.config.pad_token_id = 3  # DNABERT Tokenizer [PAD] token id
        if pooling not in ["class", "mean", "max"]:
            raise ValueError(
                f"Expected pooling to be one of class, mean, max. Got: {pooling}"
            )
        else:
            self.pooling = pooling
        self.dim = dim
        self.K = K
        self.m = m
        self.T = T
        self.use_projection_head = use_projection_head

        self.is_moco = K > 0

        # 1. Load Encoders
        self.bert_q = DNABertModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M",
            trust_remote_code=True,
            config=self.config,
        )
        self._remove_pooler(self.bert_q)
        if FLASH_ATTN_AVAILABLE:
            patch_with_flash_lib(self.bert_q)

        prev_dim = self.config.hidden_size
        if self.use_projection_head:
            self.projector_q = nn.Sequential(
                nn.Linear(prev_dim, prev_dim), nn.ReLU(), nn.Linear(prev_dim, dim)
            )
        else:
            self.projector_q = None

        total_cpus = os.cpu_count() or 4
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            num_gpus = torch.distributed.get_world_size()
        else:
            num_gpus = 1
        workers = max(1, total_cpus // num_gpus)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)

        if self.is_moco:
            self.bert_k = deepcopy(self.bert_q)
            self._remove_pooler(self.bert_k)

            for param_q, param_k in zip(
                self.bert_q.parameters(), self.bert_k.parameters()
            ):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            if self.use_projection_head:
                self.projector_k = nn.Sequential(
                    nn.Linear(prev_dim, prev_dim), nn.ReLU(), nn.Linear(prev_dim, dim)
                )
                assert self.projector_q
                # Initialize Key Projector
                for param_q, param_k in zip(
                    self.projector_q.parameters(), self.projector_k.parameters()
                ):
                    param_k.data.copy_(param_q.data)
                    param_k.requires_grad = False
            else:
                self.projector_k = None

            # Queue setup (unchanged)
            if self.use_projection_head:
                self.register_buffer("queue", torch.randn(dim, K))
            else:
                self.register_buffer("queue", torch.randn(prev_dim, K))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            # Queue for storing actual DNA sequence strings
            # Using a Python list since strings can't be stored in tensors
            self.queue_seqs: List[Optional[str]] = [None] * K

            # k-mer inverted index: kmer_str -> set of queue slot indices
            # Allows O(L) candidate lookup instead of O(K) exhaustive search.
            # k is chosen so that sequences above the similarity cutoff are
            # guaranteed to share at least one k-mer (q-gram lemma).
            # Safe default k=9 works for similarity >= 0.9 on typical DNA reads.
            self.kmer_k: int = 9
            self.kmer_index: Dict[str, Set[int]] = defaultdict(set)
        else:
            self.bert_k = None
            self.projector_k = None

    def set_kmer_k(self, identity_cutoff):
        # k must satisfy k < 1/(1-cutoff) to guarantee shared k-mers for similar seqs.
        # Use ceil - 1 to get the largest valid integer strictly below the bound.
        self.kmer_k = math.ceil(1 / (1 - identity_cutoff)) - 1

    def _get_kmers(self, seq: str) -> List[str]:
        k = self.kmer_k
        return [seq[i : i + k] for i in range(len(seq) - k + 1)]

    def _add_to_kmer_index(self, slot: int, seq: str) -> None:
        for kmer in self._get_kmers(seq):
            self.kmer_index[kmer].add(slot)

    def _remove_from_kmer_index(self, slot: int, seq: str) -> None:
        for kmer in self._get_kmers(seq):
            bucket = self.kmer_index.get(kmer)
            if bucket is not None:
                bucket.discard(slot)

    def _remove_pooler(self, model):
        # Remove unused pooler layers
        if hasattr(model, "pooler") and model.pooler is not None:
            del model.pooler
            model.pooler = None

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def _dequeue_and_enqueue(
        self, keys, is_distributed=False, sequences: Optional[List[str]] = None
    ) -> None:
        if is_distributed:
            keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue), handling wrap-around
        end = ptr + batch_size
        if end <= self.K:
            self.queue[:, ptr:end] = keys.T
        else:
            first = self.K - ptr
            self.queue[:, ptr:] = keys.T[:, :first]
            self.queue[:, : end - self.K] = keys.T[:, first:]

        # Store sequence strings and maintain k-mer index
        if sequences is not None:
            for i, seq in enumerate(sequences):
                slot = (ptr + i) % self.K
                old_seq = self.queue_seqs[slot]
                if old_seq is not None:
                    self._remove_from_kmer_index(slot, old_seq)
                self.queue_seqs[slot] = seq
                self._add_to_kmer_index(slot, seq)

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def _check_alignments(
        self, batch_sequences: List[str], queue_similarity_cutoff: float
    ) -> torch.Tensor:
        """
        Check if any sequences in the batch are locally aligned to sequences in the queue.

        Args:
            batch_sequences: List of DNA sequences in the current batch
            queue_alignment_cutoff: Minimum similarity (num_matches/min_length) to consider aligned

        Returns:
            Boolean mask of shape (K,) where True indicates the queue sequence is aligned
            to at least one sequence in the batch (should be excluded from negatives)
        """
        if not any(seq is not None for seq in self.queue_seqs):
            # Queue not yet populated with sequences
            return torch.zeros(self.K, dtype=torch.bool, device=self.device)

        aligned_mask = torch.zeros(self.K, dtype=torch.bool, device=self.device)

        for i, queue_seq in enumerate(self.queue_seqs):
            if queue_seq is None:
                continue

            # Check if this queue sequence aligns with any sequence in the batch
            for batch_seq in batch_sequences:
                # Use edlib for fast alignment
                if len(batch_seq) <= len(queue_seq):
                    q = batch_seq
                    t = queue_seq
                else:
                    q = queue_seq
                    t = batch_seq
                result = edlib.align(query=q, target=t, mode="HW", task="distance")
                edit_distance = result["editDistance"]

                # Calculate similarity as 1 - (edit_distance / max_length)
                similarity = 1.0 - (edit_distance / len(q))

                if similarity >= queue_similarity_cutoff:
                    aligned_mask[i] = True
                    break  # No need to check other batch sequences for this queue seq

        return aligned_mask

    @torch.no_grad()
    def _check_alignments_fast(
        self, batch_sequences: List[str], queue_similarity_cutoff: float
    ) -> torch.Tensor:
        """
        Check if any sequences in the batch are aligned to sequences in the queue.

        Uses a k-mer inverted index to find candidates in O(batch * L) instead of
        O(K * batch). By the q-gram lemma, any pair with edit distance <= d must
        share at least one k-mer when k <= floor(min_len / (d + 1)), so this
        filter has zero false negatives.

        Only the (typically small) candidate set is passed to edlib.
        """
        aligned_mask = torch.zeros(self.K, dtype=torch.bool, device=self.device)

        # Collect candidate queue slots that share at least one k-mer with any batch seq
        candidates: Set[int] = set()
        for seq in batch_sequences:
            for kmer in self._get_kmers(seq):
                bucket = self.kmer_index.get(kmer)
                if bucket:
                    candidates.update(bucket)

        if not candidates:
            return aligned_mask

        def check_single_candidate(slot: int):
            queue_seq = self.queue_seqs[slot]
            if queue_seq is None:
                return slot, False
            for batch_seq in batch_sequences:
                if len(batch_seq) <= len(queue_seq):
                    q, t = batch_seq, queue_seq
                else:
                    q, t = queue_seq, batch_seq
                max_allowed_distance = int(len(q) * (1.0 - queue_similarity_cutoff))
                result = edlib.align(
                    query=q,
                    target=t,
                    mode="HW",
                    task="distance",
                    k=max_allowed_distance,
                )
                if result["editDistance"] != -1:
                    return slot, True
            return slot, False

        for slot, is_aligned in self._executor.map(check_single_candidate, candidates):
            if is_aligned:
                aligned_mask[slot] = True

        return aligned_mask

    @torch.no_grad()
    def _get_top_k_sw_negatives(
        self, batch_sequences: List[str], k: int
    ) -> Tuple[torch.Tensor, List[List[Optional[str]]], torch.Tensor]:
        """
        For each query, find the top-k queue sequences by Smith-Waterman score
        using the k-mer inverted index for candidate pre-filtering.

        Returns:
            top_k_idx: (B, k) LongTensor of queue slot indices; -1 where no candidate.
            hn_seqs_per_query: B x k lists of sequences; None where no candidate.
            hn_sw_scores: (B, k) FloatTensor of SW identity scores; 0.0 where no candidate.
        """
        B = len(batch_sequences)
        top_k_idx = torch.full((B, k), -1, dtype=torch.long, device=self.device)
        hn_seqs_per_query: List[List[Optional[str]]] = [[None] * k for _ in range(B)]
        hn_sw_scores = torch.zeros((B, k), dtype=torch.float)

        def score_query(args: Tuple[int, str]):
            i, query_seq = args
            candidates: Set[int] = set()
            for kmer in self._get_kmers(query_seq):
                bucket = self.kmer_index.get(kmer)
                if bucket:
                    candidates.update(bucket)

            scored = []
            for slot in candidates:
                queue_seq = self.queue_seqs[slot]
                if queue_seq is None:
                    continue
                if len(query_seq) <= len(queue_seq):
                    q_seq, t_seq = query_seq, queue_seq
                else:
                    q_seq, t_seq = queue_seq, query_seq
                result = edlib.align(
                    query=q_seq, target=t_seq, mode="HW", task="distance"
                )
                sw_score = 1.0 - result["editDistance"] / len(q_seq)
                scored.append((sw_score, slot))

            scored.sort(key=lambda x: x[0], reverse=True)
            return i, scored[:k]

        for i, scored in self._executor.map(score_query, enumerate(batch_sequences)):
            for j, (sw_score, slot) in enumerate(scored):
                top_k_idx[i, j] = slot
                hn_seqs_per_query[i][j] = self.queue_seqs[slot]
                hn_sw_scores[i, j] = sw_score

        return top_k_idx, hn_seqs_per_query, hn_sw_scores

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        # Update both the Encoder and the Projector
        for param_q, param_k in zip(self.bert_q.parameters(), self.bert_k.parameters()):  # ty: ignore possibly-missing-attribute
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        if self.use_projection_head:
            assert self.projector_q
            assert self.projector_k
            for param_q, param_k in zip(
                self.projector_q.parameters(),
                self.projector_k.parameters(),  # ty: ignore possibly-missing-attribute
            ):
                param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def _embed(self, model, projector, seq_ids, pooling):
        # 1. Get Sequence Output (Batch, Seq_Len, Hidden)
        # Index [0] is last_hidden_state
        outputs = model(**seq_ids)[0]

        # 2. Mean Pooling (Correctly implemented)
        # Create attention mask for broadcasting: (Batch, Seq_Len, 1)
        mask = seq_ids.attention_mask.unsqueeze(-1)

        # Sum masked embeddings and divide by valid token count
        if pooling == "class":
            embeddings = outputs[:, 0, :]
        if pooling == "mean":
            embeddings = (outputs * mask).sum(dim=1) / mask.sum(dim=1)
        elif pooling == "max":
            mask_expanded = mask.expand(outputs.size())
            outputs[mask_expanded == 0] = -1e9
            embeddings, _ = outputs.max(dim=1)

        # 3. Apply MLP Projection Head
        if self.use_projection_head:
            outputs = projector(embeddings)
        else:
            outputs = embeddings
        return embeddings

    def forward(
        self,
        query,
        key,
        queue_identity_cutoff: float,
        is_distributed=False,
        query_seqs: Optional[List[str]] = None,
        key_seqs: Optional[List[str]] = None,
        filter_aligned: bool = True,
        neg_tokens: torch.Tensor | None = None,
        neg_seqs: list[str] | None = None,
    ):
        # Calculate Query Embedding
        q = self._embed(
            self.bert_q, self.projector_q, query.to(self.device), pooling=self.pooling
        )
        q = nn.functional.normalize(q, dim=1)

        n = None
        if neg_tokens is not None and neg_seqs is not None:
            n = self._embed(
                self.bert_q,
                self.projector_q,
                neg_tokens.to(self.device),
                pooling=self.pooling,
            )
            n = nn.functional.normalize(n, dim=1)

        if self.is_moco:
            with torch.no_grad():
                self._momentum_update_key_encoder()

                # Calculate Key Embedding
                k = self._embed(
                    self.bert_k,
                    self.projector_k,
                    key.to(self.device),
                    pooling=self.pooling,
                )
                k = nn.functional.normalize(k, dim=1)

            # Positive logits: B x 1
            # l_pos = einops.einsum(q, k, "B D, B D -> B").unsqueeze(-1)
            l_pos = (q * k).sum(dim=-1, keepdim=True)

            # Negative logits: B x K
            l_neg = einops.einsum(q, self.queue.clone().detach(), "B D, D K -> B K")

            # Check for aligned sequences and mask them out from negatives
            if filter_aligned and query_seqs is not None:
                aligned_mask = self._check_alignments_fast(
                    query_seqs, queue_identity_cutoff
                )
                # Set logits for aligned sequences to a very negative value (will be ignored)
                l_neg[:, aligned_mask] = -1e9

            # Logits: B x (1 + [B] + K)
            # Hard negatives are batch negatives: each query is penalized against all
            # hard negatives in the batch. They are NOT enqueued.
            if n is not None:
                l_hard = einops.einsum(q, n, "B D, N D -> B N")  # (B, B)
                logits = torch.cat([l_pos, l_hard, l_neg], dim=1)
            else:
                logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.T

            # Label 0 is always the positive (l_pos is at index 0)
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)

            self._dequeue_and_enqueue(k, is_distributed, sequences=key_seqs)

        else:
            k = self._embed(
                self.bert_q, self.projector_q, key.to(self.device), pooling=self.pooling
            )
            k = nn.functional.normalize(k, dim=1)  # (b, D)

            # 2. Gather Global Keys ONLY (or both if doing symmetric loss)
            if is_distributed:
                # We need all keys to compare our local queries against
                # torch.distributed.nn.all_gather is differentiable
                k_global_list = torch.distributed.nn.all_gather(k)  # ty: ignore possibly-missing-attribute
                k_global = torch.cat(k_global_list, dim=0)  # (N*b, D)
            else:
                k_global = k

            # Append hard negatives to the key matrix so they are treated as
            # additional negatives. The positive indices in labels are unchanged
            # because hard negatives are appended after k_global.
            if n is not None:
                if is_distributed:
                    n_global_list = torch.distributed.nn.all_gather(n)  # ty: ignore possibly-missing-attribute
                    n_global = torch.cat(n_global_list, dim=0)  # (N*b, D)
                else:
                    n_global = n
                keys_and_negs = torch.cat([k_global, n_global], dim=0)  # (2*N*b, D)
            else:
                keys_and_negs = k_global

            # 3. Compute Partial Logits (Memory Saving Step)
            # Instead of (N*b x N*b), we compute (b x N*b)
            # We only calculate logits for the queries sitting on THIS GPU
            logits = (
                torch.matmul(q, keys_and_negs.T) / self.T
            )  # Shape: (b, N*b [+ N*b])

            # 4. Correct Labels
            # The positive key for q_local[i] is at a specific index in k_global.
            # If we assume k_global is ordered by rank, the positive key for the
            # i-th local query is at index: (rank * b) + i

            rank = torch.distributed.get_rank() if is_distributed else 0
            b = q.shape[0]

            # Labels are simply the indices in the global array corresponding to local samples
            labels = torch.arange(b, dtype=torch.long, device=self.device) + (rank * b)

        return logits, labels

    @torch.no_grad()
    def encode(self, sequences):
        """
        Generates representations for downstream tasks.
        Uses bert_q only. Discards projector_q.
        """
        # self.eval()
        # sequences = sequences.to(self.device)

        # 1. Get Hidden States from Query Backbone
        # We access the backbone directly, ignoring the projector
        outputs = self.bert_q(**sequences)[0]

        # 2. Mean Pooling
        mask = sequences.attention_mask.unsqueeze(-1)
        # Sum masked embeddings and divide by valid token count
        if self.pooling == "class":
            embeddings = outputs[:, 0, :]
        if self.pooling == "mean":
            embeddings = (outputs * mask).sum(dim=1) / mask.sum(dim=1)
        elif self.pooling == "max":
            mask_expanded = mask.expand(outputs.size())
            outputs[mask_expanded == 0] = -1e9
            embeddings, _ = outputs.max(dim=1)

        # 3. Normalize (Optional but recommended for cosine similarity tasks)
        embeddings = nn.functional.normalize(embeddings, dim=1)

        return embeddings


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
