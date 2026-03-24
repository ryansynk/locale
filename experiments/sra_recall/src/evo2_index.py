from itertools import islice
from pathlib import Path

import polars as pl
import torch
from Bio import SeqIO
from torch import nn
from tqdm import tqdm

from .base_index import BaseIndex
from .config import Evo2Config, ExperimentConfig


def batched(iterable, n):
    """Batch data into lists of length n. The last batch may be shorter."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


class Evo2Index(BaseIndex):
    def __init__(self, cfg: ExperimentConfig):
        assert isinstance(cfg.model, Evo2Config)
        self.model = Evo2Encoder(cfg.model)
        self.indexed = []
        self.accessions_tensor_map: dict[str, torch.Tensor] = {}
        self.cfg = cfg
        self.model_cfg = cfg.model

    def load(self, index_path: Path):
        index_file = index_path / "index.pt"
        self.accessions_tensor_map = torch.load(index_file)

    def build(self, accessions: list[Path], index_path: Path):
        for accession in tqdm(accessions, desc="Indexing accessions..."):
            sequences = [str(record.seq) for record in SeqIO.parse(accession, "fasta")]
            embeddings = self.model.encode(sequences)
            srr_id = accession.parent.stem
            self.accessions_tensor_map[srr_id] = embeddings.cpu()
            self.indexed.append(accession)

    @torch.no_grad()
    def search(self, queries: pl.DataFrame) -> pl.DataFrame:
        queries = queries.with_row_index()
        query_features = self.model.encode(queries["query_sequence"].to_list()).to(
            self.model_cfg.device
        )
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
            scores, _ = per_accession_logits.max(dim=-1)
            all_scores.append(scores)

        scores = torch.stack(all_scores, dim=1)  # (num_queries, num_accessions)
        scores_cpu = scores.cpu().numpy()
        # scores_col = []
        scores_df = []
        for i in range(scores_cpu.shape[0]):
            for j in range(scores_cpu.shape[1]):
                scores_df.append(
                    {
                        "query_idx": i,
                        "accession": accession_names[j],
                        "score": float(scores_cpu[i, j]),
                    }
                )

        scores_df = pl.from_dicts(scores_df)
        scores_df = (
            scores_df.with_columns(pl.struct("accession", "score").alias("result"))
            .group_by("query_idx")
            .agg(pl.col("result").alias("results"))
        )
        df = queries.join(
            scores_df, left_on="index", right_on="query_idx", how="left"
        ).select("read_id", "accession", "results")

        df = df.rename({"read_id": "query_read", "accession": "query_accession"})
        assert len(df) == len(queries)
        return df

    def indexed_accessions(self) -> list[Path]:
        return self.indexed

    def save(self, output_path: Path):
        output_path.mkdir(exist_ok=True)
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


class Evo2Encoder:
    # def __init__(self, model_name, batch_size, pooling, checkpoint_path, device="cuda"):
    def __init__(self, cfg: Evo2Config):
        try:
            from evo2 import Evo2
        except ImportError:
            raise ImportError(
                "Evo2 is missing. For a light install, run: pip install evo2"
            )

        # Loads Evo2 7B. Bypassing TransformerEngine keeps the installation light.
        model = Evo2("evo2_7b")
        # self.model = model.to(self.device)
        self.model = model
        self.forward = self.model
        self.tokenizer = self.model.tokenizer
        self.device = cfg.device
        self.batch_size = cfg.batch_size
        self.pooling = cfg.pooling

    @torch.no_grad()
    def encode(self, sequences):
        embeds_list = []
        assert self.tokenizer
        for batch in batched(sequences, self.batch_size):
            # Evo2 tokenizer requires manual batching and padding
            tokenized = [self.tokenizer.tokenize(seq) for seq in batch]
            max_len = max(len(t) for t in tokenized)
            # Fallback to the 'N' token if a pad token isn't explicitly mapped
            pad_id = getattr(
                self.tokenizer, "pad_token_id", self.tokenizer.tokenize("N")[0]
            )

            input_ids = []
            masks = []
            for t in tokenized:
                pad_len = max_len - len(t)
                input_ids.append(t + [pad_id] * pad_len)
                # 1 for actual tokens, 0 for padding
                masks.append([1] * len(t) + [0] * pad_len)

            tokens_tensor = torch.tensor(
                input_ids, dtype=torch.long, device=self.device
            )
            mask = torch.tensor(
                masks, dtype=torch.float32, device=self.device
            ).unsqueeze(-1)

            # Explicitly pull from the intermediate layer highly recommended by Evo2 authors
            layer_name = "blocks.28.mlp.l3"
            _, embeddings_dict = self.forward(
                tokens_tensor, return_embeddings=True, layer_names=[layer_name]
            )
            outputs = embeddings_dict[layer_name]

            if self.pooling == "mean":
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
