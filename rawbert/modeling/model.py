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
    FLASH_ATTN_AVAILABLE = True
    from rawbert.utils.patch import patch_with_flash_lib
except ImportError:
    FLASH_ATTN_AVAILABLE = False

logger = logging.getLogger(__name__)


class RawBERT(nn.Module):
    def __init__(
        self, dim: int = 128, K: int = 4096, m: float = 0.999, T: float = 0.07
    ):
        super().__init__()
        assert K > 0, f"Expected positive queue size, got K = {K}"
        self.config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")

        # 1. Load Encoders
        self.bert_q = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
        self.bert_k = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )

        # Remove unused pooler layers
        if hasattr(self.bert_q, "pooler") and self.bert_q.pooler is not None:
            del self.bert_q.pooler
            self.bert_q.pooler = None
        if hasattr(self.bert_k, "pooler") and self.bert_k.pooler is not None:
            del self.bert_k.pooler
            self.bert_k.pooler = None

        if FLASH_ATTN_AVAILABLE:
            patch_with_flash_lib(self.bert_q)

        # 2. Define Projection Head (MoCo v2 Style: MLP)
        # Note: We do NOT replace bert.pooler. We act on the hidden states directly.
        prev_dim = self.config.hidden_size
        self.projector_q = nn.Sequential(
            nn.Linear(prev_dim, prev_dim), nn.ReLU(), nn.Linear(prev_dim, dim)
        )
        self.projector_k = nn.Sequential(
            nn.Linear(prev_dim, prev_dim), nn.ReLU(), nn.Linear(prev_dim, dim)
        )

        # 3. Initialize Key Encoder
        for param_q, param_k in zip(self.bert_q.parameters(), self.bert_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Initialize Key Projector
        for param_q, param_k in zip(
            self.projector_q.parameters(), self.projector_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Queue setup (unchanged)
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.dim = dim
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

        ptr = int(self.queue_ptr)  # ty: ignore
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr  # ty: ignore

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        # Update both the Encoder and the Projector
        for param_q, param_k in zip(self.bert_q.parameters(), self.bert_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        for param_q, param_k in zip(
            self.projector_q.parameters(), self.projector_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def _embed(self, model, projector, seq_ids):
        # 1. Get Sequence Output (Batch, Seq_Len, Hidden)
        # Index [0] is last_hidden_state
        outputs = model(**seq_ids)[0]

        # 2. Mean Pooling (Correctly implemented)
        # Create attention mask for broadcasting: (Batch, Seq_Len, 1)
        mask = seq_ids.attention_mask.unsqueeze(-1)

        # Sum masked embeddings and divide by valid token count
        embeddings = (outputs * mask).sum(dim=1) / mask.sum(dim=1)

        # 3. Apply MLP Projection Head
        return projector(embeddings)

    def forward(self, query, key, is_distributed=False):
        # Calculate Query Embedding
        q = self._embed(self.bert_q, self.projector_q, query.to(self.device))
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()

            # Calculate Key Embedding
            k = self._embed(self.bert_k, self.projector_k, key.to(self.device))
            k = nn.functional.normalize(k, dim=1)

        # Positive logits: B x 1
        l_pos = einops.einsum(q, k, "B D, B D -> B").unsqueeze(-1)

        # Negative logits: B x K
        l_neg = einops.einsum(q, self.queue.clone().detach(), "B D, D K -> B K")

        # Logits: B x (1 + K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)

        self._dequeue_and_enqueue(k, is_distributed)

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
        embeddings = (outputs * mask).sum(dim=1) / mask.sum(dim=1)

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
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())  # ty: ignore possibly-missing-attribute
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)  # ty: ignore possibly-missing-attribute

    output = torch.cat(tensors_gather, dim=0)
    return output
