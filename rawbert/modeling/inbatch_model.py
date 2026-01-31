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


class InBatchRawBERT(nn.Module):
    def __init__(self, dim: int = 128, T: float = 0.07):
        super().__init__()
        self.config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")

        # 1. Load Encoders
        self.bert_q = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
        # Remove unused pooler layers
        if hasattr(self.bert_q, "pooler") and self.bert_q.pooler is not None:
            del self.bert_q.pooler
            self.bert_q.pooler = None

        if FLASH_ATTN_AVAILABLE:
            patch_with_flash_lib(self.bert_q)

        # 2. Define Projection Head (MoCo v2 Style: MLP)
        # Note: We do NOT replace bert.pooler. We act on the hidden states directly.
        prev_dim = self.config.hidden_size
        self.projector_q = nn.Sequential(
            nn.Linear(prev_dim, prev_dim), nn.ReLU(), nn.Linear(prev_dim, dim)
        )

        self.dim = dim
        self.T = T

    @property
    def device(self):
        return next(self.parameters()).device

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
        # 1. Local Encoding
        q_local = self._embed(self.bert_q, self.projector_q, query.to(self.device))
        q_local = nn.functional.normalize(q_local, dim=1)  # (b, D)

        k_local = self._embed(self.bert_q, self.projector_q, key.to(self.device))
        k_local = nn.functional.normalize(k_local, dim=1)  # (b, D)

        # 2. Gather Global Keys ONLY (or both if doing symmetric loss)
        if is_distributed:
            # We need all keys to compare our local queries against
            # torch.distributed.nn.all_gather is differentiable
            k_global_list = torch.distributed.nn.all_gather(k_local)
            k_global = torch.cat(k_global_list, dim=0)  # (N*b, D)

            # If you are doing symmetric loss (loss(q,k) + loss(k,q)),
            # you also need q_global.
            # If just uni-directional, you only need k_global.
        else:
            k_global = k_local

        # 3. Compute Partial Logits (Memory Saving Step)
        # Instead of (N*b x N*b), we compute (b x N*b)
        # We only calculate logits for the queries sitting on THIS GPU
        logits = torch.matmul(q_local, k_global.T) / self.T  # Shape: (b, N*b)

        # 4. Correct Labels
        # The positive key for q_local[i] is at a specific index in k_global.
        # If we assume k_global is ordered by rank, the positive key for the
        # i-th local query is at index: (rank * b) + i

        rank = torch.distributed.get_rank() if is_distributed else 0
        b = query.shape[0]

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
