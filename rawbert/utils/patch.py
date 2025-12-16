import math

import einops
import torch
from flash_attn import flash_attn_varlen_qkvpacked_func


def patch_with_flash_lib(model):
    """
    Replaces BertUnpadSelfAttention.forward with a direct call to
    flash_attn_varlen_qkvpacked_func, passing ALiBi slopes directly.
    """
    # 1. Locate the class
    try:
        first_layer_attn = model.encoder.layer[0].attention.self
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
