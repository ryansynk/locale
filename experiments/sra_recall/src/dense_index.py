# import base64
# import io
import math

# import os
# from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import polars as pl
import torch
from Bio import SeqIO
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, BertConfig
from transformers.utils import logging as transformers_logging

from rawbert.modeling.bert_layers import BertModel as DNABertModel
from rawbert.modeling.model import RawBERT
from rawbert.utils.patch import patch_with_flash_lib

from .base_index import BaseIndex
from .config import DenseConfig, ExperimentConfig


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
        self.cfg = cfg
        self.model_cfg = cfg.model

    def load(self, index_path: Path):
        index_file = index_path / "index.pt"
        self.accessions_tensor_map = torch.load(index_file)

    def build(self, accessions: list[Path], index_path: Path):
        for accession in tqdm(accessions, desc="Indexing accessions..."):
            sequences = [str(record.seq) for record in SeqIO.parse(accession, "fasta")]
            chunked_sequences = []
            for seq in sequences:
                if len(seq) < self.model_cfg.min_seq_len:
                    pass
                if len(seq) <= self.model_cfg.max_seq_len:
                    chunked_sequences.append(seq)
                else:
                    chunks = chunk_sequence(
                        seq,
                        self.model_cfg.max_seq_len,
                        math.ceil(
                            self.model_cfg.max_seq_len
                            * self.model_cfg.min_overlap_percent
                            / 2
                        ),
                    )
                    chunked_sequences.extend(chunks)
            embeddings = self.model.encode(chunked_sequences)
            srr_id = accession.parent.stem
            self.accessions_tensor_map[srr_id] = embeddings.cpu()
            self.indexed.append(accession)

    @torch.no_grad()
    def search(self, queries: pl.DataFrame) -> pl.DataFrame:
        query_features = self.model.encode(queries).to(self.model_cfg.device)
        all_scores = []
        accession_names = []
        for acc, acc_tensor in tqdm(
            self.accessions_tensor_map.items(),
            total=len(self.indexed),
            desc="Searching...",
        ):
            accession_names.append(acc)
            per_accession_logits = torch.matmul(
                query_features, acc_tensor.to(self.model_cfg.device).T
            )  # (num_queries, num_seqs_in_accession)
            scores = per_accession_logits.max(dim=-1)
            all_scores.append(scores)

        scores = torch.cat(all_scores, dim=1)  # (num_queries, num_accessions)
        scores_cpu = scores.cpu().numpy()
        scores_col = []
        for i in range(scores_cpu.shape[0]):
            query_scores = [
                {"accession": accession_names[j], "score": float(scores_cpu[i, j])}
                for j in range(len(accession_names))
            ]
            scores_col.append(query_scores)

        return pl.DataFrame(
            {"query_sequence": queries["query_sequence"], "scores": scores_col}
        )

    def indexed_accessions(self) -> list[Path]:
        return self.indexed

    def save(self, output_path: Path):
        output_file = output_path / "index.pt"
        cpu_map = {}
        for srr_id, embeddings in self.accessions_tensor_map.items():
            cpu_map[srr_id] = embeddings.cpu()
        print(f"Saving index to {output_file}")
        torch.save(cpu_map, output_file)


def chunk_sequence(seq, chunk_size, overlap):
    if overlap >= chunk_size:
        raise ValueError("The overlap must be strictly less than the chunk size.")
    if chunk_size <= 0:
        raise ValueError("Chunk size (c) must be greater than 0.")

    step_size = chunk_size - overlap

    # Generate chunks of exactly size c
    chunks = [
        seq[i : i + chunk_size] for i in range(0, len(seq) - chunk_size + 1, step_size)
    ]

    return chunks


class DenseEncoder:
    # def __init__(self, model_name, batch_size, pooling, checkpoint_path, device="cuda"):
    def __init__(self, cfg: DenseConfig):
        transformers_logging.set_verbosity_error()
        # Load your PyTorch model or DNABERT here
        bert_config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        if not hasattr(bert_config, "pad_token_id") or bert_config.pad_token_id is None:
            bert_config.pad_token_id = 3  # DNABERT Tokenizer [PAD] token id

        device = cfg.device
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
            batched(sequences, self.batch_size),
            total=num_batches,
            desc="Encoding...",
            leave=False,
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
