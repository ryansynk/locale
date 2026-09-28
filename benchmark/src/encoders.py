"""Sequence encoders: raw DNA strings in, L2-normalized embeddings out.

Every encoder exposes the same `encode(sequences) -> Tensor` interface and ends in
`nn.functional.normalize(..., dim=1)`, so inner product is cosine for all of them.
`DenseEncoder` dispatches on `EncoderConfig.name`.

Split out of dense_index.py so an index can pick an encoder without importing the
dense index (and with it cuvs). The dependency runs one way: dense_index
imports from here, never the reverse.
"""

from itertools import islice
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as transformers_logging

from lae.modeling.backbones import DEFAULT_BACKBONE, get_tokenizer
from lae.modeling.model import LOCALE
from lae.utils.patch import patch_with_flash_lib

from .config import PAPER_CHECKPOINT, EncoderConfig


def batched(iterable, n):
    """Batch data into lists of length n. The last batch may be shorter."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


class DenseEncoder:
    def __init__(self, cfg: EncoderConfig):
        if cfg.name == "locale":
            self._encoder = LOCALEEncoder(cfg)
        elif cfg.name == "dna2vec":
            self._encoder = DNA2VecEncoder(cfg)
        elif cfg.name == "llmed":
            self._encoder = LLMEDEncoder(cfg)
        else:
            raise ValueError(f"Unknown model name: {cfg.name}")

    def encode(self, sequences):
        return self._encoder.encode(sequences)


class LOCALEEncoder:
    # def __init__(self, model_name, batch_size, pooling, checkpoint_path, device="cuda"):
    def __init__(self, cfg: EncoderConfig):
        transformers_logging.set_verbosity_error()

        device = cfg.device
        assert cfg.name == "locale"
        if cfg.checkpoint_path is None:
            # Unset checkpoint_path means the checkpoint published with the
            # paper. EncoderConfig.identity() pins the matching ckpt_id and
            # step, so the index matches one built from a local copy of the
            # same checkpoint. Cached after the first call.
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
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
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


class DNA2VecEncoder:
    # def __init__(self, model_name, batch_size, pooling, checkpoint_path, device="cuda"):
    def __init__(self, cfg: EncoderConfig):
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
    def __init__(self, cfg: EncoderConfig):
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
