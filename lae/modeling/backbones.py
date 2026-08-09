"""Encoder backbones for LOCALE.

The backbone-swap experiment holds the whole recipe fixed and varies only the
encoder, so every backbone here must satisfy one interface: a module whose
forward accepts the matching tokenizer's output and returns token-level
representations at index [0], shaped (batch, seq_len, hidden). That is exactly
what ``LOCALE._embed`` consumes, so the mean-pool + L2-normalize head, the
InfoNCE loss, and the retrieval pipeline stay byte-for-byte identical across
backbones.

DNABERT-2 and Nucleotide Transformer already satisfy the interface natively and
are returned unwrapped -- important for DNABERT-2, where wrapping would rename
every state-dict key and break the existing paper checkpoint. HyenaDNA needs a
thin adapter because it takes no ``attention_mask``, and dna2vec needs one
because it returns a bare tensor rather than a tuple.
"""

import torch.nn as nn
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, BertConfig

from lae.modeling.bert_layers import BertModel as DNABertModel

try:
    FLASH_ATTN_AVAILABLE = True
    from lae.utils.patch import patch_with_flash_lib
except ImportError:
    FLASH_ATTN_AVAILABLE = False

# Backbone id -> HuggingFace repo. The id is what goes in the training config
# and is persisted into the checkpoint, so it must stay stable.
BACKBONES: dict[str, str] = {
    "dnabert2": "zhihan1996/DNABERT-2-117M",
    "nt50m": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    "hyenadna": "LongSafari/hyenadna-small-32k-seqlen-hf",
    # Embed-Search-Align's encoder. Note the benchmark also runs this checkpoint
    # untrained as a baseline (DenseConfig name "dna2vec"); this entry is the
    # same weights used as a *trainable* backbone under the LOCALE recipe.
    "dna2vec": "roychowdhuryresearch/dna2vec",
}

DEFAULT_BACKBONE = "dnabert2"


def _unknown(name: str) -> ValueError:
    return ValueError(
        f"Unknown backbone {name!r}. Expected one of: {', '.join(sorted(BACKBONES))}"
    )


# EsmModel builds these but never reads them in forward():
#   - position_embeddings: NT-v2 is rotary (config.position_embedding_type),
#     so the absolute position table is dead weight.
#   - contact_head: only reachable via predict_contacts().
# Left trainable they receive no gradient, which breaks DDP's reducer
# ("parameters that were not used in producing loss") and puts a None into the
# grad-norm logging in training.py. Freezing is a no-op for the recipe -- a
# parameter that never affects the output cannot be fine-tuned in any
# meaningful sense -- and drops them from both the optimizer and DDP.
NT_UNUSED_PREFIXES = ("embeddings.position_embeddings", "contact_head")


def _freeze_unused(model: nn.Module, prefixes: tuple[str, ...]) -> None:
    frozen = 0
    for param_name, param in model.named_parameters():
        if param_name.startswith(prefixes):
            param.requires_grad = False
            frozen += 1
    if frozen == 0:
        raise RuntimeError(
            f"Expected to freeze parameters matching {prefixes}, found none. The "
            "upstream module layout changed; re-check which parameters receive "
            "gradients before training, or DDP will fail at step 2."
        )


class HyenaDNABackbone(nn.Module):
    """Adapter for HyenaDNA, which is convolutional rather than attention-based.

    Its forward takes no ``attention_mask``, so we drop it here. Padded
    positions are still excluded from the embedding because the head masks at
    pool time, and because HyenaDNA is causal, right-padding cannot contaminate
    the representations of the real positions that precede it.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids=input_ids)


class DNA2VecBackbone(nn.Module):
    """Adapter for Embed-Search-Align's dna2vec encoder.

    ``DNAEncoder.forward`` returns the hidden states as a bare tensor, not a
    tuple or a ModelOutput. Handing that to the shared head is silently wrong
    rather than loudly broken: ``_embed`` takes ``[0]``, which on a bare tensor
    selects sequence 0 and yields (seq_len, hidden), and that still broadcasts
    against the (batch, seq_len, 1) mask to a plausible (batch, seq_len,
    hidden). Training would proceed on garbage. Wrapping the output in a tuple
    is what makes ``[0]`` mean last_hidden_state, as every other backbone does.

    The tokenizer also emits ``token_type_ids``, which the encoder has no use
    for; ``**kwargs`` absorbs it.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return (self.model(input_ids=input_ids, attention_mask=attention_mask),)


def build_backbone(name: str = DEFAULT_BACKBONE) -> tuple[nn.Module, int]:
    """Return ``(encoder, hidden_size)`` for the named backbone."""
    if name not in BACKBONES:
        raise _unknown(name)
    repo = BACKBONES[name]

    if name == "dnabert2":
        config = BertConfig.from_pretrained(repo)
        if getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = 3  # DNABERT tokenizer [PAD] token id
        model = DNABertModel.from_pretrained(
            repo, trust_remote_code=True, config=config
        )
        # The pooler is unused by the mean-pool head and is dropped before the
        # key encoder is deep-copied, so it never enters the state dict.
        if getattr(model, "pooler", None) is not None:
            del model.pooler
            model.pooler = None
        if FLASH_ATTN_AVAILABLE:
            patch_with_flash_lib(model)
        return model, config.hidden_size

    if name == "nt50m":
        # Must go through AutoModelForMaskedLM, NOT AutoModel. The repo's
        # auto_map has no AutoModel entry, so AutoModel silently falls back to
        # transformers' native EsmModel -- which builds a plain MLP where NT
        # uses a SwiGLU gated MLP, giving an intermediate.dense of (2048, 512)
        # against the checkpoint's (4096, 512). AutoModelForMaskedLM IS mapped
        # and pulls the repo's own modeling code, which matches the weights.
        mlm = AutoModelForMaskedLM.from_pretrained(repo, trust_remote_code=True)
        model = mlm.esm  # drop the LM head; every backbone param is fine-tuned
        if getattr(model, "pooler", None) is not None:
            del model.pooler
            model.pooler = None
        _freeze_unused(model, NT_UNUSED_PREFIXES)
        # ESM-style: takes attention_mask, returns last_hidden_state at [0].
        return model, model.config.hidden_size

    if name == "hyenadna":
        model = AutoModel.from_pretrained(repo, trust_remote_code=True)
        return HyenaDNABackbone(model), model.config.d_model

    if name == "dna2vec":
        # Unlike NT-v2, the repo's auto_map does map AutoModel, so this pulls
        # the repo's own DNAEncoder. Every parameter of it receives a gradient
        # under the LOCALE loss, so there is nothing to freeze for DDP.
        model = AutoModel.from_pretrained(repo, trust_remote_code=True)
        # Positions are a fixed sinusoidal table of max_position_embeddings
        # (1024) rows, sliced to the input length -- longer inputs are an index
        # error, not a silent truncation. The recipe crops at 256 bp, which the
        # k-mer vocab packs into ~130 tokens, so there is a wide margin.
        return DNA2VecBackbone(model), model.config.embedding_dim

    raise _unknown(name)


def get_tokenizer(name: str = DEFAULT_BACKBONE):
    """Return the tokenizer matching the named backbone.

    Tokenization is the axis the reviewer asked us to vary, so this must always
    be selected by the same id that selected the encoder -- never hardcoded.
    """
    if name not in BACKBONES:
        raise _unknown(name)
    tokenizer = AutoTokenizer.from_pretrained(BACKBONES[name], trust_remote_code=True)

    if name == "hyenadna":
        # HyenaDNATokenizer pads but omits attention_mask entirely, because the
        # model itself never consumes one. The pooling head does: it averages
        # over non-padding positions only. Without this the head would average
        # padding into every embedding.
        if "attention_mask" not in tokenizer.model_input_names:
            tokenizer.model_input_names = list(tokenizer.model_input_names) + [
                "attention_mask"
            ]

    return tokenizer
