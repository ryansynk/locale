import copy
import multiprocessing as mp
import os
import random
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, process
from itertools import islice
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
from Bio import SeqIO
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BertConfig,
)
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


# Global variable to hold the model instance per worker process
_worker_encoder = None


def _init_worker(cfg, gpu_queue):
    """Initializes the model once per worker process on a specific GPU."""
    global _worker_encoder
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device_id = gpu_queue.get()

    local_cfg = copy.deepcopy(cfg)
    local_cfg.model.device = f"cuda:{device_id}"

    # Initialize and keep in memory
    _worker_encoder = DenseEncoder(local_cfg.model)


def _process_sequence_batch(batch):
    """Encodes a batch of (srr_id, sequence) tuples."""
    global _worker_encoder
    assert isinstance(_worker_encoder, DenseEncoder)
    try:
        # Unzip the batch into IDs and sequences
        srr_ids = [item[0] for item in batch]
        sequences = [item[1] for item in batch]

        # Encode the batch (encoder.encode handles its own internal batching if needed,
        # but ideally this function's batch size matches your optimal GPU batch size)
        embeddings = _worker_encoder.encode(sequences).cpu()

        # Return the mapping to be re-assembled by the main process
        return srr_ids, embeddings
    except Exception as e:
        # Print the full traceback directly to the console from the worker
        print("\n--- WORKER ERROR ---\n", file=sys.stderr)
        traceback.print_exc()
        print("--------------------\n", file=sys.stderr)
        raise e  # Re-raise so the main thread knows it failed


class DenseIndex(BaseIndex):
    def __init__(self, cfg: ExperimentConfig):
        assert isinstance(cfg.model, DenseConfig)
        self.model = DenseEncoder(cfg.model)
        self.k = cfg.model.k
        self.cfg = cfg
        self.model_cfg = cfg.model
        self.chunk_type: Literal["stride", "exact_chunk"] = cfg.model.chunk_type
        self.chunk_overlap: int = cfg.model.chunk_overlap
        self.contig_align_intervals: dict[str, list[tuple[int, int]]] | None = None

    def load(self, index_path: Path):
        mmap = np.load(index_path / "embeddings.npy", mmap_mode="r")
        meta = pl.read_parquet(index_path / "meta.parquet")
        self._mmap = mmap  # keep reference to prevent GC closing the mapping
        self.acc_names_flat = meta["srr_id"].to_list()
        starts = meta["start_row"].to_list()
        counts = meta["num_rows"].to_list()
        self.acc_offsets = starts + [starts[-1] + counts[-1]] if starts else [0]
        self.all_embeddings = torch.from_numpy(mmap)
        print(
            f"Loaded {len(starts)} accessions ({mmap.shape[0]} vectors) [memory-mapped]"
        )

    def _iter_chunks(self, accessions: list[Path]):
        """Yield (srr_id, sequence_chunk) for every chunk across all accessions."""
        for accession in accessions:
            srr_id = accession.parent.stem
            for record in SeqIO.parse(accession, "fasta"):
                seq = str(record.seq)
                contig_id = str(record.id)
                if len(seq) <= self.model_cfg.max_seq_len:
                    yield srr_id, seq
                else:
                    for chunk in chunk_sequence(
                        seq,
                        contig_id,
                        self.model_cfg.max_seq_len,
                        self.chunk_overlap,
                        self.chunk_type,
                        self.contig_align_intervals,
                    ):
                        yield srr_id, chunk

    def build(self, accessions: list[Path], index_path: Path):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPUs available for building the index.")

        print("Pre-counting chunks (one-pass FASTA scan)...")
        total_chunks = sum(1 for _ in self._iter_chunks(accessions))
        print(f"Total chunks: {total_chunks:,}")

        embed_dim = self.model.encode(["ACGT"]).shape[1]

        index_path.mkdir(exist_ok=True, parents=True)
        raw_mmap = np.lib.format.open_memmap(
            index_path / "embeddings.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total_chunks, embed_dim),
        )

        print(f"Embedding across {num_gpus} GPUs...")
        ctx = mp.get_context("spawn")
        m = ctx.Manager()
        gpu_queue = m.Queue()
        for i in range(num_gpus):
            gpu_queue.put(i)

        submission_batch_size = self.model_cfg.batch_size * 4
        rows: list[dict] = []
        offset = 0

        with ProcessPoolExecutor(
            max_workers=num_gpus,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(self.cfg, gpu_queue),
        ) as executor:
            futures = [
                executor.submit(_process_sequence_batch, batch)
                for batch in batched(
                    self._iter_chunks(accessions), submission_batch_size
                )
            ]
            # Process in submission order: chunks from the same accession land
            # contiguously in the memmap, so no post-hoc compaction is needed.
            for future in tqdm(futures, total=len(futures), desc="Embedding batches"):
                try:
                    srr_ids, embeddings = future.result()
                    arr = embeddings.numpy()
                    srr_groups: dict[str, list[int]] = defaultdict(list)
                    for i, srr_id in enumerate(srr_ids):
                        srr_groups[srr_id].append(i)
                    for srr_id, indices in srr_groups.items():
                        chunk = arr[indices]
                        n = len(chunk)
                        raw_mmap[offset : offset + n] = chunk
                        rows.append(
                            {"srr_id": srr_id, "start_row": offset, "num_rows": n}
                        )
                        offset += n
                except process.BrokenProcessPool:
                    print("\n[!] A worker died abruptly. Halting.")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                except Exception as e:
                    print(f"Worker failed: {e}")

        raw_mmap.flush()
        del raw_mmap

        # Merge consecutive rows for the same accession into single entries
        # (a batch boundary may split one accession across two consecutive rows).
        final_rows: list[dict] = []
        for r in rows:
            if final_rows and final_rows[-1]["srr_id"] == r["srr_id"]:
                final_rows[-1]["num_rows"] += r["num_rows"]
            else:
                final_rows.append(dict(r))

        pl.DataFrame(final_rows).write_parquet(index_path / "meta.parquet")
        self._streamed_to = index_path
        self.load(index_path)
        print(
            f"Built: {offset:,} vectors, {len(final_rows)} accessions -> {index_path}"
        )

    @torch.no_grad()
    def search(self, queries: pl.DataFrame) -> pl.DataFrame:
        # Unified replacement for search/search_short/search_long.
        # Short queries (< max_seq_len) produce a single chunk identical to the
        # full query, so sum-of-chunk-maxima reduces to a plain max — the same
        # score search_short would produce.  Long queries are chunked without
        # overlap and scored as sum of per-chunk maxima, identical to search_long.
        queries = queries.with_row_index()
        query_chunks = []
        query_indices = []
        prev_idx = 0
        for query in queries["query_sequence"].to_list():
            chunked_query = [
                query[i : (i + self.model_cfg.max_seq_len)]
                for i in range(0, len(query), self.model_cfg.max_seq_len)
            ]
            num_chunks = len(chunked_query)
            query_indices.append((prev_idx, prev_idx + num_chunks))
            query_chunks.extend(chunked_query)
            prev_idx = prev_idx + num_chunks

        query_chunk_features = self.model.encode(query_chunks).to(self.model_cfg.device)
        n_chunks = len(query_chunks)
        n_queries = len(queries)
        device = self.model_cfg.device

        # Pre-compute once: which query each chunk belongs to
        chunk_to_query = torch.zeros(n_chunks, dtype=torch.long, device=device)
        for qi, (s, e) in enumerate(query_indices):
            chunk_to_query[s:e] = qi

        # Per-accession loop — 2 GPU ops per accession instead of n_queries
        all_scores = []
        for i in range(len(self.acc_names_flat)):
            s, e = self.acc_offsets[i], self.acc_offsets[i + 1]
            logits = (
                query_chunk_features @ self.all_embeddings[s:e].to(device).T
            )  # (n_chunks, acc_size)
            chunk_maxes = logits.max(dim=-1).values  # (n_chunks,)
            acc_scores = torch.zeros(n_queries, device=device, dtype=chunk_maxes.dtype)
            acc_scores.scatter_add_(0, chunk_to_query, chunk_maxes)  # (n_queries,)
            all_scores.append(acc_scores)

        accession_names = self.acc_names_flat
        scores = torch.stack(all_scores, dim=1)  # (n_queries, n_acc)
        scores_cpu = scores.float().cpu().numpy()
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
        ).select("query_id", "results")

        assert len(df) == len(queries)
        return df

    def indexed_accessions(self) -> list[str]:
        return self.acc_names_flat

    def save(self, output_path: Path):
        print(f"Index already written to {output_path} during build.")

    @staticmethod
    def merge_shards(index_path: Path, num_nodes: int):
        """Stream per-node shard embeddings into a single memmap file — no full load into RAM."""
        all_meta = []
        total_vectors = 0
        embed_dim = None

        for rank in range(num_nodes):
            shard_path = index_path / f"shard_{rank}"
            arr = np.load(shard_path / "embeddings.npy", mmap_mode="r")
            if embed_dim is None and arr.ndim == 2:
                embed_dim = arr.shape[1]
            meta = pl.read_parquet(shard_path / "meta.parquet")
            meta = meta.with_columns(
                (pl.col("start_row") + total_vectors).alias("start_row")
            )
            all_meta.append(meta)
            total_vectors += len(arr)
            del arr
            print(f"  Shard {rank}: {len(meta)} accessions")

        if embed_dim is None:
            raise ValueError("No valid shards found")

        merged = np.lib.format.open_memmap(
            index_path / "embeddings.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total_vectors, embed_dim),
        )
        offset = 0
        for rank in range(num_nodes):
            arr = np.load(
                index_path / f"shard_{rank}" / "embeddings.npy", mmap_mode="r"
            )
            n = len(arr)
            merged[offset : offset + n] = arr
            offset += n
            del arr
            print(f"  Streamed shard {rank} ({n} vectors)")

        merged.flush()
        del merged

        pl.concat(all_meta).write_parquet(index_path / "meta.parquet")
        print(f"Merged {num_nodes} shards -> {total_vectors} vectors")


def chunk_sequence(
    seq: str,
    contig_id: str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_type: Literal["stride", "exact_chunk"],
    contig_align_intervals: dict[str, list[tuple[int, int]]] | None,
):
    if chunk_type == "stride":
        if chunk_overlap >= chunk_size:
            raise ValueError("The overlap must be strictly less than the chunk size.")
        if chunk_size <= 0:
            raise ValueError("Chunk size (c) must be greater than 0.")

        step_size = chunk_size - chunk_overlap

        # Generate chunks of exactly size c
        chunks = [
            seq[i : i + chunk_size]
            for i in range(0, len(seq) - chunk_size + 1, step_size)
        ]
    elif chunk_type == "exact_chunk":
        assert contig_align_intervals
        align_intervals = contig_align_intervals.get(contig_id, [])
        chunks = []
        covered = []

        for iv_start, iv_end in align_intervals:
            # chunk must start early enough to reach iv_start, and late enough to cover iv_end
            lo = max(0, iv_end - chunk_size)
            hi = min(iv_start, len(seq) - chunk_size)
            chunk_start = random.randint(lo, max(lo, hi))
            chunks.append(seq[chunk_start : chunk_start + chunk_size])
            covered.append((chunk_start, chunk_start + chunk_size))

        # Merge covered intervals to find uncovered regions
        covered.sort()
        merged: list[tuple[int, int]] = []
        for s, e in covered:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        uncovered_regions: list[tuple[int, int]] = []
        prev = 0
        for s, e in merged:
            if prev < s:
                uncovered_regions.append((prev, s))
            prev = e
        if prev < len(seq):
            uncovered_regions.append((prev, len(seq)))

        # Evenly chunk each uncovered region, including any leftover
        for r_start, r_end in uncovered_regions:
            for i in range(r_start, r_end, chunk_size):
                chunks.append(seq[i : i + chunk_size])
    else:
        raise ValueError(f"Incorrect chunk_type recieved: {chunk_type}")

    return chunks


class DenseEncoder:
    def __init__(self, cfg: DenseConfig):
        if cfg.name == "dnabert":
            self._encoder = DNABertEncoder(cfg)
        elif cfg.name == "rawbert":
            self._encoder = RawBERTEncoder(cfg)
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


class RawBERTEncoder:
    # def __init__(self, model_name, batch_size, pooling, checkpoint_path, device="cuda"):
    def __init__(self, cfg: DenseConfig):
        transformers_logging.set_verbosity_error()

        device = cfg.device
        assert cfg.name == "rawbert"
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
