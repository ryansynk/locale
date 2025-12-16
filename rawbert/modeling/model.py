"""
This file contains code adapted from the Moco repository:
https://github.com/facebookresearch/moco/tree/main

Original Author: Kaiming He, Yuxin Wu
License: MIT License

Code has been modified for DNA sequence data
"""

import logging

import einops
import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig

# Try to import the specific varlen function from flash_attn
try:
    from flash_attn import flash_attn_varlen_qkvpacked_func

    FLASH_ATTN_AVAILABLE = True
    from rawbert.utils.patch import patch_with_flash_lib
except ImportError:
    FLASH_ATTN_AVAILABLE = False

logger = logging.getLogger(__name__)


class RawBERT(nn.Module):
    def __init__(self, dim: int = 64, K: int = 4096, m: float = 0.999, T: float = 0.07):
        super().__init__()
        self.config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        self.bert_q = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
        self.bert_k = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
        if FLASH_ATTN_AVAILABLE:
            patch_with_flash_lib(self.bert_q)
        else:
            logger.warning("Executing model without flash attention")

        self.bert_q.pooler = nn.Linear(self.config.hidden_size, dim, bias=False)
        self.bert_k.pooler = nn.Linear(self.config.hidden_size, dim, bias=False)
        self.dim = dim

        for param_q, param_k in zip(self.bert_q.parameters(), self.bert_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.K = K
        self.m = m
        self.T = T

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, is_distributed) -> None:
        if is_distributed:
            keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.bert_q.parameters(), self.bert_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def forward(self, query, key, is_distributed=False):
        # seq is (B, N_max, D)
        query = query.to(self.device)
        q = self._embed_q(query)
        q = nn.functional.normalize(q, dim=1)  # (B, D)

        with torch.no_grad():
            # update key encoder
            self._momentum_update_key_encoder()
            key = key.to(self.device)

            k = self._embed_k(key)
            k = nn.functional.normalize(k, dim=1)  # (B, D)

        # Positive logits: B x 1
        l_pos = einops.einsum(q, k, "B D, B D -> B").unsqueeze(-1)

        # Negative logits: B x K
        l_neg = einops.einsum(q, self.queue.clone().detach(), "B D, D K -> B K")

        # Logits: B x (1 + K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        self._dequeue_and_enqueue(k, is_distributed)

        return logits, labels

    def _embed_q(self, seq_ids):
        embeddings = self.bert_q(**seq_ids)[
            1
        ]  # use "pooled" logit output (B, seq_len, self.dim)

        # Mean pooling
        embeddings = embeddings.sum(axis=1) / seq_ids.attention_mask.sum(
            axis=-1
        ).unsqueeze(-1)  # (B, self.dim)

        return embeddings

    def _embed_k(self, seq_ids):
        embeddings = self.bert_k(**seq_ids)[
            1
        ]  # use "pooled" logit output (B, seq_len, self.dim)

        # Mean pooling
        embeddings = embeddings.sum(axis=1) / seq_ids.attention_mask.sum(
            axis=-1
        ).unsqueeze(-1)  # (B, self.dim)

        return embeddings

    def encode(self, sequences):
        sequences = sequences.to(self.device)
        sequences = self._embed_q(sequences)
        sequences = nn.functional.normalize(sequences, dim=1)  # (B, D)
        return sequences


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
