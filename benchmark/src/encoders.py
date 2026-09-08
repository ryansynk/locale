"""Sequence encoders: raw DNA strings in, L2-normalized embeddings out.

Every encoder exposes the same `encode(sequences) -> Tensor` interface and ends in
`nn.functional.normalize(..., dim=1)`, so inner product is cosine for all of them.
`DenseEncoder` dispatches on `DenseConfig.name`.

Split out of dense_index.py so an index can pick an encoder without importing the
dense index (and with it cuvs). The dependency runs one way: dense_index
imports from here, never the reverse.
"""

import os
import sys
from itertools import islice
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BertConfig,
)
from transformers.utils import logging as transformers_logging

from lae.modeling.backbones import DEFAULT_BACKBONE, get_tokenizer
from lae.modeling.bert_layers import BertModel as DNABertModel
from lae.modeling.model import LOCALE
from lae.utils.patch import patch_with_flash_lib

from .config import PAPER_CHECKPOINT, DenseConfig


def batched(iterable, n):
    """Batch data into lists of length n. The last batch may be shorter."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


class DenseEncoder:
    def __init__(self, cfg: DenseConfig):
        if cfg.name == "dnabert":
            self._encoder = DNABertEncoder(cfg)
        elif cfg.name == "locale":
            self._encoder = LOCALEEncoder(cfg)
        elif cfg.name == "generator":
            self._encoder = GeneratorEncoder(cfg)
        elif cfg.name == "neuroseed":
            self._encoder = NeuroSEEDEncoder(cfg)
        elif cfg.name == "dna2vec":
            self._encoder = DNA2VecEncoder(cfg)
        elif cfg.name == "llmed":
            self._encoder = LLMEDEncoder(cfg)
        elif cfg.name == "evo2":
            self._encoder = Evo2Encoder(cfg)
        else:
            raise ValueError(f"Unknown model name: {cfg.name}")

    def encode(self, sequences):
        return self._encoder.encode(sequences)


class DNABertEncoder:
    # def __init__(self, model_name, batch_size, pooling, checkpoint_path, device="cuda"):
    def __init__(self, cfg: DenseConfig):
        transformers_logging.set_verbosity_error()

        device = cfg.device
        assert cfg.name == "dnabert"
        bert_config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        if not hasattr(bert_config, "pad_token_id") or bert_config.pad_token_id is None:
            bert_config.pad_token_id = 3  # DNABERT Tokenizer [PAD] token id
        model = DNABertModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M",
            trust_remote_code=True,
            config=bert_config,
        )
        if hasattr(model, "pooler") and model.pooler is not None:
            del model.pooler
            model.pooler = None
        patch_with_flash_lib(model)
        model = model.eval().to(device)
        # forward = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
        self.model = model

        self.batch_size = cfg.batch_size
        self.device = device
        self.pooling = cfg.pooling

    @torch.no_grad()
    def encode(self, sequences):
        embeds_list = []
        assert self.tokenizer
        for batch in batched(sequences, self.batch_size):
            tokens = self.tokenizer(batch, return_tensors="pt", padding=True).to(
                self.device
            )
            outputs = self.model(**tokens)[0]

            mask = tokens.attention_mask.unsqueeze(-1)
            if self.pooling == "class":
                embeddings = outputs[:, 0, :]
            elif self.pooling == "mean":
                embeddings = (outputs * mask).sum(dim=1) / mask.sum(dim=1)
            elif self.pooling == "max":
                mask_expanded = mask.expand(outputs.size())
                outputs = outputs.clone()  # Prevent in-place modification warnings
                outputs[mask_expanded == 0] = -1e9
                embeddings, _ = outputs.max(dim=1)
            else:
                raise ValueError(f"self.pooling got unexpected value {self.pooling}")

            batch_embeds = nn.functional.normalize(embeddings, dim=1)
            embeds_list.append(batch_embeds)

        embeddings = torch.cat(embeds_list, dim=0)
        return embeddings


class LOCALEEncoder:
    # def __init__(self, model_name, batch_size, pooling, checkpoint_path, device="cuda"):
    def __init__(self, cfg: DenseConfig):
        transformers_logging.set_verbosity_error()

        device = cfg.device
        assert cfg.name == "locale"
        if cfg.checkpoint_path is None:
            # Unset checkpoint_path means the checkpoint published with the
            # paper. DenseConfig has already pinned the matching ckpt_id and
            # step, so the index lands where a local copy of the same
            # checkpoint would. Cached after the first call.
            checkpoint_path = Path(
                hf_hub_download(
                    repo_id=PAPER_CHECKPOINT["repo_id"],
                    filename=PAPER_CHECKPOINT["filename"],
                    revision=PAPER_CHECKPOINT["revision"],
                )
            )
        else:
            checkpoint_path = Path(cfg.checkpoint_path).resolve()
            assert checkpoint_path.is_file(), (
                f"Checkpoint does not exist: {checkpoint_path}"
            )
        checkpoint = torch.load(checkpoint_path)
        model_args = checkpoint["model_args"]
        # Checkpoints trained before the backbone swap have no "backbone" key;
        # those are all DNABERT-2, so default accordingly.
        backbone = model_args.get("backbone", DEFAULT_BACKBONE)
        model = LOCALE(
            pooling=model_args["pooling"],
            dim=model_args["dim"],
            K=model_args["K"],
            m=model_args["m"],
            T=model_args["T"],
            backbone=backbone,
        )
        model.load_state_dict(checkpoint["model"])
        model = model.eval().to(device)
        self.tokenizer = get_tokenizer(backbone)

        self.model = model
        self.batch_size = cfg.batch_size
        self.device = device
        self.pooling = cfg.pooling

    @torch.no_grad()
    def encode(self, sequences):
        embeds_list = []
        assert self.tokenizer
        for batch in batched(sequences, self.batch_size):
            tokens = self.tokenizer(batch, return_tensors="pt", padding=True).to(
                self.device
            )
            outputs = self.model.bert_q(**tokens)[0]

            mask = tokens.attention_mask.unsqueeze(-1)
            if self.pooling == "class":
                embeddings = outputs[:, 0, :]
            elif self.pooling == "mean":
                embeddings = (outputs * mask).sum(dim=1) / mask.sum(dim=1)
            elif self.pooling == "max":
                mask_expanded = mask.expand(outputs.size())
                outputs = outputs.clone()  # Prevent in-place modification warnings
                outputs[mask_expanded == 0] = -1e9
                embeddings, _ = outputs.max(dim=1)
            else:
                raise ValueError(f"self.pooling got unexpected value {self.pooling}")

            batch_embeds = nn.functional.normalize(embeddings, dim=1)
            embeds_list.append(batch_embeds)

        embeddings = torch.cat(embeds_list, dim=0)
        return embeddings


class GeneratorEncoder:
    # def __init__(self, model_name, batch_size, pooling, checkpoint_path, device="cuda"):
    def __init__(self, cfg: DenseConfig):
        transformers_logging.set_verbosity_error()

        device = cfg.device
        assert cfg.name == "generator"
        model_str = "GenerTeam/GENERator-v2-eukaryote-1.2b-base"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_str, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_str,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.batch_size = cfg.batch_size
        self.device = device
        self.pooling = cfg.pooling
        self.model = self.model.eval().to(self.device)

    @torch.no_grad()
    def encode(self, sequences):
        embeds_list = []
        assert self.tokenizer
        for batch in batched(sequences, self.batch_size):
            processed_sequences = [
                self.tokenizer.bos_token + seq[: len(seq) // 6 * 6] for seq in batch
            ]
            self.tokenizer.padding_side = "right"
            inputs = self.tokenizer(
                processed_sequences,
                add_special_tokens=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model.config.max_position_embeddings,
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states[-1]
            attention_mask = inputs["attention_mask"]

            if self.pooling == "class":
                raise ValueError("Decoder-only model has no cls token")
            elif self.pooling == "mean":
                expanded_mask = (
                    attention_mask.unsqueeze(-1)
                    .expand(hidden_states.size())
                    .to(torch.float32)
                )
                sum_embeddings = torch.sum(hidden_states * expanded_mask, dim=1)
                embeddings = sum_embeddings / expanded_mask.sum(dim=1)
            elif self.pooling == "max":
                mask_expanded = attention_mask.unsqueeze(-1).expand(
                    hidden_states.size()
                )
                hidden_states = (
                    hidden_states.clone()
                )  # Prevent in-place modification warnings
                hidden_states[mask_expanded == 0] = -1e9
                embeddings, _ = hidden_states.max(dim=1)
            elif self.pooling == "eos":
                last_token_indices = attention_mask.sum(dim=1) - 1
                embeddings = hidden_states[
                    torch.arange(hidden_states.size(0)), last_token_indices, :
                ]
            else:
                raise ValueError(f"self.pooling got unexpected value {self.pooling}")

            batch_embeds = nn.functional.normalize(embeddings, dim=1)
            embeds_list.append(batch_embeds)

        embeddings = torch.cat(embeds_list, dim=0)
        return embeddings


# DNA character → integer index matching NeuroSEED's convention:
# A→0, C→1, G→2, T→3; unknown characters (N, etc.) encoded as all-zeros.
_DNA_ALPHABET: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}


def _strings_to_one_hot(sequences: list[str], len_sequence: int) -> torch.Tensor:
    """Convert a list of DNA strings to a one-hot float tensor of shape
    (batch, len_sequence, 4).  Sequences are truncated or zero-padded to
    ``len_sequence``.  Unknown nucleotides are encoded as all-zeros.
    """
    alphabet_size = 4
    # lookup[i] = one-hot row for nucleotide i; lookup[4] = zeros for padding/-1
    lookup = torch.cat(
        [torch.eye(alphabet_size), torch.zeros(1, alphabet_size)], dim=0
    )  # (5, 4)

    batch_indices = []
    for seq in sequences:
        indices = [
            _DNA_ALPHABET.get(c.upper(), alphabet_size) for c in seq[:len_sequence]
        ]
        if len(indices) < len_sequence:
            indices.extend([alphabet_size] * (len_sequence - len(indices)))
        batch_indices.append(indices)

    index_tensor = torch.tensor(batch_indices, dtype=torch.long)  # (B, L)
    return lookup[index_tensor]  # (B, L, 4)


class NeuroSEEDEncoder:
    """Wraps a pretrained NeuroSEED model for encoding raw DNA strings.

    The checkpoint must be saved in the format produced by NeuroSEED's
    training scripts::

        torch.save(
            (model_class, model_args, embedding_model.state_dict(), distance),
            "checkpoint.pkl",
        )

    Sequences are converted to one-hot tensors (batch × len_sequence × 4)
    before being passed to ``model.encode()``.  Sequences longer than
    ``model_args.len_sequence`` are truncated; shorter ones are zero-padded.
    """

    def __init__(self, cfg: DenseConfig):
        assert cfg.name == "neuroseed"
        assert cfg.checkpoint_path, "No checkpoint_path provided for NeuroSEED!"
        checkpoint_path = Path(cfg.checkpoint_path).resolve()
        assert checkpoint_path.is_file(), f"Checkpoint not found: {checkpoint_path}"

        # NeuroSEED model classes must be importable at torch.load time because
        # the checkpoint serialises the class object itself.
        if cfg.neuroseed_path:
            neuroseed_abs = str(Path(cfg.neuroseed_path).resolve())
            if neuroseed_abs not in sys.path:
                sys.path.insert(0, neuroseed_abs)

        # Checkpoint format: (model_class, model_args, state_dict, distance)
        model_class, model_args, state_dict, distance = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        encoder_model = model_class(**vars(model_args))
        encoder_model.load_state_dict(state_dict)
        encoder_model = encoder_model.eval().to(cfg.device)

        self.model = encoder_model
        self.len_sequence: int = model_args.len_sequence
        self.batch_size = cfg.batch_size
        self.device = cfg.device

    @torch.no_grad()
    def encode(self, sequences: list[str]) -> torch.Tensor:
        embeds_list = []
        for batch in batched(sequences, self.batch_size):
            one_hot = _strings_to_one_hot(batch, self.len_sequence).to(self.device)
            # Base model classes (CNN, Feedforward, etc.) expose forward() not encode().
            # TripletEncoder wraps those and does expose encode(), so handle both.
            if hasattr(self.model, "encode"):
                embeddings = self.model.encode(one_hot)
            else:
                embeddings = self.model(one_hot)
            batch_embeds = nn.functional.normalize(embeddings.float(), dim=1)
            embeds_list.append(batch_embeds)
        return torch.cat(embeds_list, dim=0)


class DNA2VecEncoder:
    # def __init__(self, model_name, batch_size, pooling, checkpoint_path, device="cuda"):
    def __init__(self, cfg: DenseConfig):
        transformers_logging.set_verbosity_error()

        device = cfg.device
        assert cfg.name == "dna2vec"
        self.model = AutoModel.from_pretrained(
            "roychowdhuryresearch/dna2vec", trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "roychowdhuryresearch/dna2vec", trust_remote_code=True
        )
        self.model = self.model.eval().to(device)

        self.batch_size = cfg.batch_size
        self.device = device
        self.pooling = cfg.pooling

    @torch.no_grad()
    def encode(self, sequences):
        embeds_list = []
        assert self.tokenizer
        for batch in batched(sequences, self.batch_size):
            tokens = self.tokenizer(batch, return_tensors="pt", padding=True).to(
                self.device
            )
            outputs = self.model(**tokens)

            mask = tokens.attention_mask.unsqueeze(-1)
            if self.pooling == "class":
                raise ValueError("dna2vec does not have a class token")
            elif self.pooling == "mean":
                embeddings = (outputs * mask).sum(dim=1) / mask.sum(dim=1)
            elif self.pooling == "max":
                mask_expanded = mask.expand(outputs.size())
                outputs = outputs.clone()  # Prevent in-place modification warnings
                outputs[mask_expanded == 0] = -1e9
                embeddings, _ = outputs.max(dim=1)
            else:
                raise ValueError(f"self.pooling got unexpected value {self.pooling}")

            batch_embeds = nn.functional.normalize(embeddings, dim=1)
            embeds_list.append(batch_embeds)

        embeddings = torch.cat(embeds_list, dim=0)
        return embeddings


class LLMEDEncoder:
    # def __init__(self, model_name, batch_size, pooling, checkpoint_path, device="cuda"):
    def __init__(self, cfg: DenseConfig):
        transformers_logging.set_verbosity_error()

        device = cfg.device
        assert cfg.name == "llmed"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
        model = AutoModel.from_pretrained("PSUXL/LLMED-MAE", trust_remote_code=True)
        patch_with_flash_lib(model)
        self.model = model
        self.model = self.model.eval().to(device)

        self.batch_size = cfg.batch_size
        self.device = device
        self.pooling = cfg.pooling
        assert self.tokenizer.vocab_size == self.model.config.vocab_size

    @torch.no_grad()
    def encode(self, sequences):
        embeds_list = []
        assert self.tokenizer
        for batch in batched(sequences, self.batch_size):
            tokens = self.tokenizer(batch, return_tensors="pt", padding=True).to(
                self.device
            )
            outputs = self.model(**tokens)[0]

            mask = tokens.attention_mask.unsqueeze(-1)
            if self.pooling == "class":
                embeddings = outputs[:, 0, :]
            elif self.pooling == "mean":
                embeddings = (outputs * mask).sum(dim=1) / mask.sum(dim=1)
            elif self.pooling == "max":
                mask_expanded = mask.expand(outputs.size())
                outputs = outputs.clone()  # Prevent in-place modification warnings
                outputs[mask_expanded == 0] = -1e9
                embeddings, _ = outputs.max(dim=1)
            else:
                raise ValueError(f"self.pooling got unexpected value {self.pooling}")

            batch_embeds = nn.functional.normalize(embeddings, dim=1)
            embeds_list.append(batch_embeds)

        embeddings = torch.cat(embeds_list, dim=0)
        return embeddings


class Evo2Encoder:
    def __init__(self, cfg):  # Assuming Evo2Config is defined elsewhere
        assert cfg.name == "evo2"
        try:
            from evo2 import Evo2
        except ImportError:
            raise ImportError(
                "Evo2 is missing. For a light install, run: pip install evo2"
            )

        # Evo2 claims all visible GPUs by default, which breaks multi-GPU workers.
        # Restrict visibility to only the target GPU before loading so Evo2
        # initializes a single copy on that device (visible as cuda:0).
        device_str = cfg.device  # e.g. "cuda:2"
        gpu_id = device_str.split(":")[-1] if ":" in device_str else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        model = Evo2("evo2_7b")
        self.model = model
        self.forward = self.model
        self.tokenizer = self.model.tokenizer
        # With CUDA_VISIBLE_DEVICES restricted to one GPU, Evo2 sees it as cuda:0
        self.device = "cuda:0"
        self.batch_size = cfg.batch_size
        self.pooling = cfg.pooling

    def encode(self, sequences):
        embeds_list = []
        assert self.tokenizer
        for batch in batched(sequences, self.batch_size):
            tokenized = [self.tokenizer.tokenize(seq) for seq in batch]
            max_len = max(len(t) for t in tokenized)

            pad_id = getattr(
                self.tokenizer, "pad_token_id", self.tokenizer.tokenize("N")[0]
            )

            input_ids = []
            masks = []
            for t in tokenized:
                pad_len = max_len - len(t)
                input_ids.append(t + [pad_id] * pad_len)
                masks.append([1] * len(t) + [0] * pad_len)

            tokens_tensor = torch.tensor(
                input_ids, dtype=torch.long, device=self.device
            )
            mask = torch.tensor(
                masks, dtype=torch.float32, device=self.device
            ).unsqueeze(-1)

            layer_name = "blocks.28.mlp.l3"
            _, embeddings_dict = self.forward(
                tokens_tensor, return_embeddings=True, layer_names=[layer_name]
            )
            outputs = embeddings_dict[layer_name]

            if self.pooling == "mean":
                embeddings = (outputs * mask).sum(dim=1) / mask.sum(dim=1)
            elif self.pooling == "max":
                mask_expanded = mask.expand(outputs.size())
                outputs = outputs.clone()
                outputs[mask_expanded == 0] = -1e9
                embeddings, _ = outputs.max(dim=1)
            else:
                raise ValueError(f"self.pooling got unexpected value {self.pooling}")

            embeds_list.append(nn.functional.normalize(embeddings, dim=1))
        embeddings = torch.cat(embeds_list, dim=0)
        return embeddings
