import string
import einops
import torch
import torch.nn as nn

from transformers import AutoModel, BertConfig


class RawBERT(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        self.bert = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
        self.dim = dim
        self.linear = nn.Linear(self.config.hidden_size, dim, bias=False)

    @property
    def device(self):
        # Get the device of the first parameter
        return next(self.parameters()).device

    @staticmethod
    def pool_reads_last_token(embed, attention_mask):
        # embed: (B, num_reads, num_tokens_per_read, D)
        # attention_mask: (B, num_reads, num_tokens_per_read)
        # output is (B, num_reads, D)

        # Number of valid (unmasked) tokens along the sequence
        lengths = attention_mask.sum(dim=-1)  # (B, H)
        new_mask = (lengths != 0).int()

        # Convert to indices of last valid position (subtract 1)
        last_idx = lengths - 1  # (B, H)
        last_idx[last_idx == -1] = 0

        # Gather the corresponding entries
        idx = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, embed.size(-1))
        embed_last = torch.gather(embed, dim=2, index=idx).squeeze(2)  # (B, H, D)
        return embed_last, new_mask

    def forward(self, Q, R):
        # Q is (B, 1, query_length)
        # R is (B, num_reads)
        Q_mask = Q["attention_mask"].to(self.device)
        Q = self.embed(**Q).squeeze()
        R, R_mask = self.embed(**R, pool_reads=True)
        return self.score(Q, Q_mask.squeeze(), R, R_mask)

    def embed(self, input_ids, attention_mask, pool_reads=False):
        # input_ids is (B, max_num_reads, max_read_length)
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        B, num_reads, read_length = input_ids.shape
        input_ids = input_ids.view(B * num_reads, read_length)
        attention_mask = attention_mask.view(B * num_reads, read_length)
        E = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]

        attention_mask = attention_mask.view(B, num_reads, read_length)
        E = E.view(B, num_reads, read_length, -1)

        if pool_reads:
            E, new_mask = RawBERT.pool_reads_last_token(E, attention_mask)
            E = self.linear(E)  # select last token of each read
            E = torch.nn.functional.normalize(E, p=2, dim=2)
            return E, new_mask
        else:
            E = self.linear(E)  # select last token of each read
            E = torch.nn.functional.normalize(E, p=2, dim=2)
            return E

    def score(self, Q, Q_mask, R, R_mask):
        # Q is (B, max_query_len, D)
        # Q_mask is (B, max_query_len)
        # R is (B, max_num_reads, D)
        # R_mask is (B, max_num_reads)

        # scores is (B, B)
        scores = einops.einsum(
            Q, R, "B q_tokens D, BB n_reads D -> B BB q_tokens n_reads"
        )

        # Need to mask out tokens with -inf before taking max!
        LARGE_NEG = -1e9
        # Expand masks to broadcast shapes
        # Q_mask: (B, q_tokens) -> (B, 1, q_tokens, 1)
        q_mask_exp = Q_mask[:, None, :, None]
        # R_mask: (B, n_reads) -> (1, B, 1, n_reads)
        r_mask_exp = R_mask[None, :, None, :]

        # Combine: valid where both are 1
        valid_mask = q_mask_exp * r_mask_exp  # (B, B, q_tokens, n_reads)

        # Apply mask
        scores = scores.masked_fill(valid_mask == 0, LARGE_NEG)

        # Perform sum-of-max operation
        scores = scores.max(-1).values

        # Zero out queries that are masked
        valid_query_mask = Q_mask.unsqueeze(1).float()  # (B, 1, q_tokens)
        scores = scores * valid_query_mask  # zero out invalid query tokens
        scores = scores.sum(-1)

        return scores
