import logging
import math

import einops
import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig

# Try to import the specific varlen function from flash_attn
try:
    from flash_attn import flash_attn_varlen_qkvpacked_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

logger = logging.getLogger(__name__)


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
        if FLASH_ATTN_AVAILABLE:
            self._patch_with_flash_lib()
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
        for param_q, param_k in zip(self.bert_q.parameters(), self.bert_k.parameters()):
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
        embeddings = self.bert_q(**seq_ids)[1]  # use "pooled" logit output (B, seq_len, self.dim)

        # Mean pooling
        embeddings = embeddings.sum(axis=1) / seq_ids.attention_mask.sum(
            axis=-1
        ).unsqueeze(
            -1
        )  # (B, self.dim)

        return embeddings

    def _embed_k(self, seq_ids):
        embeddings = self.bert_k(**seq_ids)[
            1
        ]  # use "pooled" logit output (B, seq_len, self.dim)

        # Mean pooling
        embeddings = embeddings.sum(axis=1) / seq_ids.attention_mask.sum(
            axis=-1
        ).unsqueeze(
            -1
        )  # (B, self.dim)

        return embeddings

    def _patch_with_flash_lib(self):
        """
        Replaces BertUnpadSelfAttention.forward with a direct call to
        flash_attn_varlen_qkvpacked_func, passing ALiBi slopes directly.
        """
        # 1. Locate the class
        try:
            first_layer_attn = self.bert_q.encoder.layer[0].attention.self
            TargetClass = first_layer_attn.__class__
        except AttributeError:
            print("Warning: Could not locate DNABERT attention layer.")
            return

        if getattr(TargetClass, "_is_patched_by_fa_lib", False):
            return

        # 2. Helper to generate ALiBi slopes (Copied logic from MosaicBERT)
        def get_alibi_slopes(n_heads):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                return [start * start**i for i in range(n)]

            if math.log2(n_heads).is_integer():
                return get_slopes_power_of_2(n_heads)

            closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = get_alibi_slopes(2 * closest_power_of_2)
            slopes_b = slopes_b[0::2][: n_heads - closest_power_of_2]
            return slopes_a + slopes_b

        # 3. Define the new forward function
        def flash_lib_forward(
            attn_self,
            hidden_states,
            cu_seqlens,
            max_seqlen_in_batch,
            indices,
            attn_mask,
            bias,
        ):
            """
            Args:
                hidden_states: (total_nnz, dim) - Already unpadded!
                cu_seqlens: (batch + 1)
                bias: IGNORED (We generate alibi slopes internally)
            """
            # 1. Project QKV
            # Output: [total_nnz, 3 * heads * head_dim]
            qkv = attn_self.Wqkv(hidden_states)

            # Ensure input is fp16/bf16 (Flash Attn requirement)
            dtype_og = qkv.dtype
            if dtype_og not in [torch.float16, torch.bfloat16]:
                qkv = qkv.to(torch.float16)

            # 2. Reshape for Flash Attn
            # Target: [total_nnz, 3, heads, head_dim]
            qkv = einops.rearrange(
                qkv, "n (t h d) -> n t h d", t=3, h=attn_self.num_attention_heads
            )

            # 3. Get ALiBi slopes (Cache them on the module to avoid recomputing)
            if not hasattr(attn_self, "_alibi_slopes"):
                slopes = torch.tensor(
                    get_alibi_slopes(attn_self.num_attention_heads),
                    device=qkv.device,
                    dtype=torch.float32,
                )
                attn_self.register_buffer("_alibi_slopes", slopes, persistent=False)

            # 4. Call Flash Attention Varlen
            # We skip pad_input/unpad_input entirely because we use the varlen kernel
            output = flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_seqlen_in_batch,
                dropout_p=attn_self.p_dropout if attn_self.training else 0.0,
                alibi_slopes=attn_self._alibi_slopes,
                deterministic=False,  # Set True if you need reproducibility at small perf cost
            )

            # Output is [total_nnz, heads, head_dim]
            # 5. Flatten back to [total_nnz, hidden_dim]
            output = einops.rearrange(output, "n h d -> n (h d)")

            return output.to(dtype_og)

        # 4. Apply Patch
        TargetClass.forward = flash_lib_forward
        TargetClass._is_patched_by_fa_lib = True
        print(
            f"Successfully patched {TargetClass.__name__} with flash_attn library (ALiBi enabled)."
        )
