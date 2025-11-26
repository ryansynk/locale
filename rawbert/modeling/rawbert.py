import itertools

import einops
import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig


class RawBERT(nn.Module):
    def __init__(self, dim=64, K=4096, m=0.999):
        super().__init__()
        self.config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        self.bert_q = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
        self.bert_k = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
        self.linear_q = nn.Linear(self.config.hidden_size, dim, bias=False)
        self.linear_k = nn.Linear(self.config.hidden_size, dim, bias=False)
        self.dim = dim

        params_q = itertools.chain(self.bert_q.parameters(), self.linear_q.parameters())
        params_k = itertools.chain(self.bert_k.parameters(), self.linear_k.parameters())

        for param_q, param_k in zip(params_q, params_k):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.K = K
        self.m = m

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys) -> None:
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
        params_q = itertools.chain(self.bert_q.parameters(), self.linear_q.parameters())
        params_k = itertools.chain(self.bert_k.parameters(), self.linear_k.parameters())

        for param_q, param_k in zip(params_q, params_k):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def forward(self, query, key):
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

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        self._dequeue_and_enqueue(k)

        return logits, labels

    def _embed_q(self, seq_ids):
        embeddings = self.bert_q(**seq_ids, output_hidden_states=True)[
            0
        ]  # use raw logit output (B, seq_len, hidden_size)

        # Mean pooling
        embeddings = embeddings.sum(axis=1) / seq_ids.attention_mask.sum(
            axis=-1
        ).unsqueeze(
            -1
        )  # (B, hidden_size)

        # Linear output
        embeddings = self.linear_q(embeddings)  # (B, self.dim)

        return embeddings

    def _embed_k(self, seq_ids):
        embeddings = self.bert_k(**seq_ids, output_hidden_states=True)[
            0
        ]  # use raw logit output (B, seq_len, hidden_size)

        # Mean pooling
        embeddings = embeddings.sum(axis=1) / seq_ids.attention_mask.sum(
            axis=-1
        ).unsqueeze(
            -1
        )  # (B, hidden_size)

        # Linear output
        embeddings = self.linear_k(embeddings)  # (B, self.dim)

        return embeddings
