import copy
import math
import multiprocessing as mp
import os
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed, process
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
        print(f"\n--- WORKER ERROR ---\n", file=sys.stderr)
        traceback.print_exc()
        print(f"--------------------\n", file=sys.stderr)
        raise e  # Re-raise so the main thread knows it failed


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

    def build_serial(self, accessions: list[Path], index_path: Path):
        for accession in tqdm(accessions, desc="Indexing accessions..."):
            sequences = [str(record.seq) for record in SeqIO.parse(accession, "fasta")]
            chunked_sequences = []
            for seq in sequences:
                if len(seq) <= self.model_cfg.max_seq_len:
                    chunked_sequences.append(seq)
                else:
                    overlap = self.model_cfg.max_seq_len - 10
                    chunks = chunk_sequence(
                        seq, self.model_cfg.max_seq_len, overlap=overlap
                    )
                    chunked_sequences.extend(chunks)
            embeddings = self.model.encode(chunked_sequences)
            srr_id = accession.parent.stem
            self.accessions_tensor_map[srr_id] = embeddings.cpu()
            self.indexed.append(accession)

    def build(self, accessions: list[Path], index_path: Path):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPUs available for building the index.")

        print(f"Parallelizing sequence-level build across {num_gpus} GPUs...")

        # 1. Setup the GPU assignment queue for workers
        ctx = mp.get_context("spawn")
        m = ctx.Manager()
        gpu_queue = m.Queue()
        for i in range(num_gpus):
            gpu_queue.put(i)

        # 2. Define a generator to flatten all files into (srr_id, sequence) tuples
        def sequence_generator():
            for accession in accessions:
                self.indexed.append(accession)
                srr_id = accession.parent.stem

                # Yield parsed and chunked sequences
                for record in SeqIO.parse(accession, "fasta"):
                    seq = str(record.seq)
                    if len(seq) <= self.model_cfg.max_seq_len:
                        yield (srr_id, seq)
                    else:
                        overlap = self.model_cfg.max_seq_len - 10
                        chunks = chunk_sequence(
                            seq, self.model_cfg.max_seq_len, overlap
                        )
                        for chunk in chunks:
                            yield (srr_id, chunk)

        # 3. Create a dictionary to hold lists of tensors per accession
        temp_tensor_map = defaultdict(list)

        # We will send work to the GPUs in chunks of N sequences.
        # Make this a multiple of your model's batch_size for optimal throughput.
        submission_batch_size = self.model_cfg.batch_size * 4

        with ProcessPoolExecutor(
            max_workers=num_gpus,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(self.cfg, gpu_queue),
        ) as executor:
            # Submit batches to the workers
            futures = []
            for batch in batched(sequence_generator(), submission_batch_size):
                futures.append(executor.submit(_process_sequence_batch, batch))

            # 4. Collect results as they complete
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing Batches",
            ):
                try:
                    srr_ids, embeddings = future.result()

                    # Group the returned embeddings back by their original file
                    for i, srr_id in enumerate(srr_ids):
                        # Extract the single embedding vector and append
                        temp_tensor_map[srr_id].append(embeddings[i].unsqueeze(0))
                except process.BrokenProcessPool:
                    # Catch the specific abrupt termination error
                    print(
                        "\n[!] A worker died abruptly (likely OOM or Segfault). Halting the pool to stop error spam."
                    )
                    # Cancel all remaining futures so they don't also print errors
                    executor.shutdown(wait=False, cancel_futures=True)
                    break  # Exit the collection loop
                except Exception as e:
                    print(f"Worker failed: {e}")

        # 5. Finalize by concatenating the lists of tensors into standard matrices
        print("Finalizing tensor map...")
        for srr_id, tensor_list in temp_tensor_map.items():
            self.indexed.append(srr_id)
            self.accessions_tensor_map[srr_id] = torch.cat(tensor_list, dim=0)

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
        assert self.tokenizer
        for batch in batched(sequences, self.batch_size):
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
                outputs = outputs.clone()  # Prevent in-place modification warnings
                outputs[mask_expanded == 0] = -1e9
                embeddings, _ = outputs.max(dim=1)
            else:
                raise ValueError(f"self.pooling got unexpected value {self.pooling}")

            batch_embeds = nn.functional.normalize(embeddings, dim=1)
            embeds_list.append(batch_embeds)

        embeddings = torch.cat(embeds_list, dim=0)
        return embeddings
