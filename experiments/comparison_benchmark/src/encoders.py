import math
from itertools import batched  # ty: ignore unresolved-import
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as transformers_logging

from rawbert.modeling.model import RawBERT
from rawbert.utils.patch import patch_with_flash_lib

from .config import DenseConfig


class BaseEncoder:
    def encode(self, sequences: list[str]):
        """Returns a list of feature vectors (or MinHash objects)."""
        raise NotImplementedError


class DenseEncoder(BaseEncoder):
    # def __init__(self, model_name, batch_size, pooling, checkpoint_path, device="cuda"):
    def __init__(self, cfg: DenseConfig, device: str = "cuda"):
        transformers_logging.set_verbosity_error()
        # Load your PyTorch model or DNABERT here
        if cfg.name == "rawbert":
            assert cfg.checkpoint_path, "No checkpoint provided!"
            checkpoint_path = Path(cfg.checkpoint_path).resolve()
            assert checkpoint_path.is_file(), "Checkpoint does not exist!"
            checkpoint = torch.load(checkpoint_path)
            model = RawBERT(
                pooling="max",
                dim=checkpoint["model_args"]["dim"],
                K=checkpoint["model_args"]["K"],
                m=checkpoint["model_args"]["m"],
                T=checkpoint["model_args"]["T"],
            )
            model.load_state_dict(checkpoint["model"])
            model = model.eval().to(device)
            forward = model.bert_q
        elif cfg.name == "dnabert":
            model = AutoModel.from_pretrained(
                "zhihan1996/DNABERT-2-117M", trust_remote_code=True
            )
            patch_with_flash_lib(model)
            model = model.eval().to(device)
            forward = model
        else:
            raise ValueError(
                f"Expected model_name to be 'rawbert' or 'dnabert', got: {cfg.name}"
            )

        self.model = model
        self.batch_size = cfg.batch_size
        self.device = device
        self.forward = forward
        self.pooling = cfg.pooling
        self.tokenizer = AutoTokenizer.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )

    @torch.no_grad()
    def encode(self, sequences):
        embeds_list = []
        num_batches = int(math.ceil(len(sequences) / self.batch_size))
        for batch in tqdm(
            batched(sequences, self.batch_size), total=num_batches, desc="Encoding..."
        ):
            tokens = self.tokenizer(batch, return_tensors="pt", padding=True).to(
                self.device
            )
            outputs = self.forward(**tokens)[0]

            mask = tokens.attention_mask.unsqueeze(-1)
            if self.pooling == "class":
                embeddings = outputs[:, 0, :]
            if self.pooling == "mean":
                embeddings = (outputs * mask).sum(dim=1) / mask.sum(dim=1)
            elif self.pooling == "max":
                mask_expanded = mask.expand(outputs.size())
                outputs[mask_expanded == 0] = -1e9
                embeddings, _ = outputs.max(dim=1)

            batch_embeds = nn.functional.normalize(embeddings, dim=1)
            embeds_list.append(batch_embeds)

        embeddings = torch.cat(embeds_list, dim=0)
        return embeddings


class SourMashEncoder(BaseEncoder):
    def __init__(self, k=6, num_perm=128):
        raise NotImplementedError

    def encode(self, sequences):
        raise NotImplementedError
