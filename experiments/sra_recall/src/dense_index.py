# import base64
import polars as pl

# import io
import math

# import os
# from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, BertConfig
from transformers.utils import logging as transformers_logging

from rawbert.modeling.bert_layers import BertModel as DNABertModel
from rawbert.modeling.model import RawBERT
from rawbert.utils.patch import patch_with_flash_lib

from .config import ExperimentConfig, DenseConfig
from .base_index import BaseIndex


def batched(iterable, n):
    """Batch data into lists of length n. The last batch may be shorter."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


class DenseIndex(BaseIndex):
    def __init__(self, cfg: ExperimentConfig):
        assert isinstance(cfg.model, DenseConfig)
        self.model = DenseEncoder(cfg.model)
        self.indexed = []
        self.accessions_tensor_map: dict[str, torch.Tensor] = {}
        self.k = cfg.model.k

    def load(self, index_path: Path):
        raise NotImplementedError

    def build(self, accessions: list[Path]):
        raise NotImplementedError

    @torch.no_grad()
    def search(self, queries: pl.DataFrame) -> pl.DataFrame:
        query_features = self.model.encode(queries)
        all_logits = []
        for acc, acc_tensor in self.accessions_tensor_map.items():
            per_accession_logits = torch.matmul(
                query_features, acc_tensor.T
            )  # (num_queries, num_seqs_in_accession)
            all_logits.append(per_accession_logits)

        logits = torch.cat(all_logits, dim=1)
        _, indices = logits.topk(k=self.k, dim=1)  # (num_queries, k)
        # return indices.cpu()
        raise NotImplementedError

    def indexed_accessions(self) -> list[Path]:
        raise NotImplementedError

    def save(self, output_path: Path):
        raise NotImplementedError


class DenseEncoder:
    # def __init__(self, model_name, batch_size, pooling, checkpoint_path, device="cuda"):
    def __init__(self, cfg: DenseConfig, device: str = "cuda"):
        transformers_logging.set_verbosity_error()
        # Load your PyTorch model or DNABERT here
        bert_config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        if not hasattr(bert_config, "pad_token_id") or bert_config.pad_token_id is None:
            bert_config.pad_token_id = 3  # DNABERT Tokenizer [PAD] token id

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
            elif self.pooling == "mean":
                embeddings = (outputs * mask).sum(dim=1) / mask.sum(dim=1)
            elif self.pooling == "max":
                mask_expanded = mask.expand(outputs.size())
                outputs[mask_expanded == 0] = -1e9
                embeddings, _ = outputs.max(dim=1)
            else:
                raise ValueError(f"self.pooling got unexpected value {self.pooling}")

            batch_embeds = nn.functional.normalize(embeddings, dim=1)
            embeds_list.append(batch_embeds)

        embeddings = torch.cat(embeds_list, dim=0)
        return embeddings
