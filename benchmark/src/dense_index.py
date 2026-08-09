import copy
import faulthandler
import json
import math
import multiprocessing as mp
import os
import random
import sys
import traceback
from collections import defaultdict
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    process,
)
from itertools import islice
from pathlib import Path
from typing import Literal

import diskannpy
import numpy as np
import polars as pl
import torch
from Bio import SeqIO
from huggingface_hub import hf_hub_download
from torch import nn
from tqdm import tqdm
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

from .base_index import BaseIndex
from .config import PAPER_CHECKPOINT, DenseConfig, ExperimentConfig


def batched(iterable, n):
    """Batch data into lists of length n. The last batch may be shorter."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def _create_fbin_memmap(path: Path, n: int, d: int) -> np.memmap:
    """Create an fbin file with a uint32 [n, d] header and return a writable float32 memmap."""
    with open(path, "wb") as f:
        np.array([n, d], dtype=np.uint32).tofile(f)
        f.seek(n * d * np.dtype(np.float32).itemsize - 1, 1)
        f.write(b"\x00")
    return np.memmap(path, dtype=np.float32, mode="r+", offset=8, shape=(n, d))


def _load_fbin_mmap(path: Path) -> np.memmap:
    """Memory-map an fbin file and return a read-only float32 array."""
    with open(path, "rb") as f:
        n, d = np.frombuffer(f.read(8), dtype=np.uint32)
    return np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(int(n), int(d)))


# ---------------------------------------------------------------------------
# RaBitQ 1-bit quantization index
# ---------------------------------------------------------------------------


def _build_rabitq_index(
    fbin_path: Path,
    rabitq_dir: Path,
    chunk_rows: int = 200_000,
    seed: int = 0,
) -> None:
    """Build a RaBitQ index from an existing .fbin file.

    Pass 1 computes the centroid on CPU. Pass 2 rotates and quantizes chunks
    in parallel across all available GPUs via ThreadPoolExecutor (CUDA ops
    release the GIL). Output memmaps are pre-allocated so each worker writes
    directly to its offset with no sequential bottleneck.
    """
    rabitq_dir.mkdir(parents=True, exist_ok=True)

    mmap = _load_fbin_mmap(fbin_path)
    n, d = mmap.shape
    if d % 8 != 0:
        raise ValueError(
            f"Embedding dim={d} must be a multiple of 8 for RaBitQ packing."
        )

    bytes_per_vec = d // 8
    sqrt_d = math.sqrt(d)

    n_gpus = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(n_gpus)] if n_gpus > 0 else ["cpu"]
    print(f"Building RaBitQ index using {len(devices)} device(s)...")

    # Pass 1: streaming centroid (CPU)
    centroid = np.zeros(d, dtype=np.float64)
    num_chunks = n // chunk_rows
    for start in tqdm(
        range(0, n, chunk_rows), total=num_chunks, desc="Streaming centroid..."
    ):
        centroid += mmap[start : start + chunk_rows].sum(axis=0, dtype=np.float64)
    centroid = (centroid / n).astype(np.float32)

    # Random orthogonal rotation via QR decomposition
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((d, d)).astype(np.float32)
    q, r = np.linalg.qr(g)
    rotation = (q * np.sign(np.diag(r))).astype(np.float32)

    # Move centroid and rotation to each device once
    centroid_per_dev = {dev: torch.from_numpy(centroid).to(dev) for dev in devices}
    rotation_per_dev = {dev: torch.from_numpy(rotation).to(dev) for dev in devices}

    # Pre-allocate output memmaps — random-access writes are safe across threads
    # since each chunk writes to a non-overlapping offset range.
    codes_mm = np.memmap(
        rabitq_dir / "codes.u8", dtype=np.uint8, mode="w+", shape=(n, bytes_per_vec)
    )
    norms_mm = np.memmap(
        rabitq_dir / "norms.f32", dtype=np.float32, mode="w+", shape=(n,)
    )
    dots_mm = np.memmap(
        rabitq_dir / "dots.f32", dtype=np.float32, mode="w+", shape=(n,)
    )

    def _quantize_chunk(start: int, dev: str):
        end = min(start + chunk_rows, n)
        chunk = torch.from_numpy(np.array(mmap[start:end], dtype=np.float32)).to(dev)
        xr = (chunk - centroid_per_dev[dev]) @ rotation_per_dev[dev]
        chunk_norms = torch.linalg.norm(xr, dim=1).float().cpu().numpy()
        signs = torch.where(xr >= 0, torch.ones_like(xr), -torch.ones_like(xr))
        chunk_dots = (xr * signs).sum(dim=1).float().cpu().numpy() / sqrt_d
        packed = np.packbits(
            (signs > 0).to(torch.uint8).cpu().numpy(), axis=-1, bitorder="little"
        )
        return start, end, packed, chunk_norms, chunk_dots

    # Pass 2: parallel quantization — distribute chunks round-robin across GPUs
    chunk_starts = list(range(0, n, chunk_rows))
    chunk_devs = [devices[i % len(devices)] for i in range(len(chunk_starts))]

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = {
            pool.submit(_quantize_chunk, start, dev): start
            for start, dev in zip(chunk_starts, chunk_devs)
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Quantizing chunks"
        ):
            start, end, packed, norms, dots = future.result()
            codes_mm[start:end] = packed
            norms_mm[start:end] = norms
            dots_mm[start:end] = dots

    codes_mm.flush()
    norms_mm.flush()
    dots_mm.flush()
    del codes_mm, norms_mm, dots_mm

    np.save(rabitq_dir / "centroid.npy", centroid)
    np.save(rabitq_dir / "rotation.npy", rotation)
    with open(rabitq_dir / "meta.json", "w") as f:
        json.dump({"n": n, "d": d, "bytes_per_vec": bytes_per_vec}, f)

    print(f"RaBitQ index built: {n:,} vectors -> {rabitq_dir}")


class RaBitQIndex:
    """Sharded GPU RaBitQ index for approximate inner-product search.

    Codes are stored as int8 {-1, +1} on each GPU shard. Search is asymmetric:
    fp32 rotated queries vs int8 codes via fp16 matmul, with per-vector
    norm/dot scalars to debias the estimator.
    """

    def __init__(
        self,
        n: int,
        d: int,
        centroid: torch.Tensor,
        rotation: torch.Tensor,
        shards: list[dict],
    ):
        self.n = n
        self.d = d
        self.centroid = centroid
        self.rotation = rotation
        self.shards = shards
        self.sqrt_d = math.sqrt(d)

    @classmethod
    def load(cls, rabitq_dir: Path, devices: list[str] | None = None) -> "RaBitQIndex":
        with open(rabitq_dir / "meta.json") as f:
            meta = json.load(f)
        n, d, bytes_per_vec = meta["n"], meta["d"], meta["bytes_per_vec"]

        if devices is None:
            n_gpus = torch.cuda.device_count()
            devices = [f"cuda:{i}" for i in range(n_gpus)] if n_gpus > 0 else ["cpu"]

        codes_mm = np.memmap(
            rabitq_dir / "codes.u8", dtype=np.uint8, mode="r", shape=(n, bytes_per_vec)
        )
        norms_mm = np.memmap(
            rabitq_dir / "norms.f32", dtype=np.float32, mode="r", shape=(n,)
        )
        dots_mm = np.memmap(
            rabitq_dir / "dots.f32", dtype=np.float32, mode="r", shape=(n,)
        )

        primary_dev = devices[0]
        centroid = torch.from_numpy(np.load(rabitq_dir / "centroid.npy")).to(
            primary_dev
        )
        rotation = torch.from_numpy(np.load(rabitq_dir / "rotation.npy")).to(
            primary_dev
        )

        shard_sizes = [n // len(devices)] * len(devices)
        for i in range(n % len(devices)):
            shard_sizes[i] += 1

        shards = []
        offset = 0
        for dev, size in zip(devices, shard_sizes):
            packed = torch.from_numpy(
                np.ascontiguousarray(codes_mm[offset : offset + size])
            )
            unpacked01 = torch.from_numpy(
                np.unpackbits(packed.numpy(), axis=1, bitorder="little").astype(np.int8)
            )
            codes_pm1 = (unpacked01 * 2 - 1).to(dev)
            norms = torch.from_numpy(
                np.ascontiguousarray(norms_mm[offset : offset + size])
            ).to(dev)
            dots = torch.from_numpy(
                np.ascontiguousarray(dots_mm[offset : offset + size])
            ).to(dev)
            shards.append(
                {
                    "device": dev,
                    "offset": offset,
                    "codes": codes_pm1,
                    "norms": norms,
                    "dots": dots,
                }
            )
            offset += size

        print(f"Loaded RaBitQ index: {n:,} vectors across {len(devices)} device(s)")
        return cls(n, d, centroid, rotation, shards)

    @torch.no_grad()
    def search(
        self, queries: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (scores, indices) each (q, k); indices are global row ids.

        queries may be on any device; centering and rotation run on the primary
        device (where centroid/rotation live), then q_rot is scattered to each
        shard's device for the matmul.
        """
        if queries.dtype != torch.float32:
            queries = queries.float()
        if queries.dim() == 1:
            queries = queries.unsqueeze(0)

        # Pre-rotate on the primary device (GPU if available, else CPU).
        primary_dev = self.centroid.device
        q = queries.to(primary_dev, non_blocking=True)
        q_rot = (q - self.centroid) @ self.rotation  # (n_q, d) on primary_dev
        q_dot_centroid = q @ self.centroid  # (n_q,)   on primary_dev

        n_q = q.shape[0]
        per_shard_scores: list[torch.Tensor] = []
        per_shard_indices: list[torch.Tensor] = []
        for shard in self.shards:
            dev = shard["device"]
            shard_size = shard["codes"].shape[0]
            local_k = min(k, shard_size)

            qd_fp16 = q_rot.to(dev, non_blocking=True).to(torch.float16)
            qdot = q_dot_centroid.to(dev)

            # Size sub-batches to stay within ~40% of free GPU memory, avoiding
            # a full fp16 cast of the entire shard (2x the int8 footprint).
            if dev != "cpu":
                free_mem, _ = torch.cuda.mem_get_info(torch.device(dev))
                sub_batch = max(1, int(free_mem * 0.4 / (self.d * 2 + n_q * 4)))
            else:
                sub_batch = shard_size

            # Running top-k merged across sub-batches.
            running_scores = torch.full((n_q, local_k), float("-inf"), device=dev)
            running_indices = torch.zeros((n_q, local_k), dtype=torch.long, device=dev)

            for sub_start in range(0, shard_size, sub_batch):
                sub_end = min(sub_start + sub_batch, shard_size)

                sub_codes = shard["codes"][sub_start:sub_end].to(torch.float16)
                sub_raw = (qd_fp16 @ sub_codes.T).float()
                del sub_codes

                sub_scale = shard["norms"][sub_start:sub_end] / (
                    self.sqrt_d * shard["dots"][sub_start:sub_end]
                )
                sub_est = sub_raw * sub_scale + qdot.unsqueeze(1)
                del sub_raw

                sub_local_k = min(local_k, sub_end - sub_start)
                sub_scores, sub_local_idx = sub_est.topk(sub_local_k, dim=1)
                del sub_est

                combined_scores = torch.cat([running_scores, sub_scores], dim=1)
                combined_indices = torch.cat(
                    [running_indices, sub_local_idx + sub_start], dim=1
                )
                running_scores, sel = combined_scores.topk(local_k, dim=1)
                running_indices = combined_indices.gather(1, sel)

            per_shard_scores.append(running_scores.cpu())
            per_shard_indices.append((running_indices + shard["offset"]).cpu())

        all_scores = torch.cat(per_shard_scores, dim=1)
        all_indices = torch.cat(per_shard_indices, dim=1)
        final_scores, sel = all_scores.topk(k, dim=1)
        final_indices = all_indices.gather(1, sel)
        return final_scores, final_indices

    @torch.no_grad()
    def search_cpu(
        self, queries: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (scores, indices) each (q, k); indices are global row ids.

        queries may be on any device; centering and rotation run on the primary
        device (where centroid/rotation live), then q_rot is scattered to each
        shard's device for the matmul.
        """
        if queries.dtype != torch.float32:
            queries = queries.float()
        if queries.dim() == 1:
            queries = queries.unsqueeze(0)

        # Pre-rotate on the primary device (GPU if available, else CPU).
        primary_dev = self.centroid.device
        q = queries.to(primary_dev, non_blocking=True)
        q_rot = (q - self.centroid) @ self.rotation  # (n_q, d) on primary_dev
        q_dot_centroid = q @ self.centroid  # (n_q,)   on primary_dev

        per_shard_scores: list[torch.Tensor] = []
        per_shard_indices: list[torch.Tensor] = []
        for shard in self.shards:
            dev = shard["device"]
            qd_fp16 = q_rot.to(dev, non_blocking=True).to(torch.float16)
            codes_fp16 = shard["codes"].to(torch.float16)

            raw = (qd_fp16 @ codes_fp16.T).float()
            scale = shard["norms"] / (self.sqrt_d * shard["dots"])
            est = raw * scale + q_dot_centroid.to(dev).unsqueeze(1)

            local_k = min(k, est.shape[1])
            top_scores, top_local_idx = est.topk(local_k, dim=1)
            per_shard_scores.append(top_scores.cpu())
            per_shard_indices.append((top_local_idx + shard["offset"]).cpu())

        all_scores = torch.cat(per_shard_scores, dim=1)
        all_indices = torch.cat(per_shard_indices, dim=1)
        final_scores, sel = all_scores.topk(k, dim=1)
        final_indices = all_indices.gather(1, sel)
        return final_scores, final_indices


# Global variable to hold the model instance per worker process
_worker_encoder = None


def _init_worker(cfg, gpu_queue):
    """Initializes the model once per worker process on a specific GPU."""
    global _worker_encoder
    faulthandler.enable(file=sys.stderr)  # dump traceback on SIGSEGV/SIGFPE/etc.
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
        self.no_search: bool = cfg.no_search
        self.k = cfg.model.k
        self.cfg = cfg
        self.use_ann: bool = cfg.model.use_ann
        self.exact_search: bool = cfg.model.exact_search
        self.use_rabitq: bool = cfg.model.use_rabitq
        self.model_cfg = cfg.model

        if not self.no_search:
            self.model = DenseEncoder(cfg.model)
            # chunk_type was dropped from DenseConfig during the v1 cleanup;
            # stride is the only supported mode (see the hardcoded "chunkstride"
            # index tag in config.py and the legacy column in run_benchmark.py).
            self.chunk_type: Literal["stride", "exact_chunk"] = "stride"
            self.chunk_overlap: int = cfg.model.chunk_overlap
            self.contig_align_intervals: dict[str, list[tuple[int, int]]] | None = None

    def load(self, index_path: Path):
        meta = pl.read_parquet(index_path / "meta.parquet")
        self.acc_names_flat = meta["srr_id"].to_list()
        starts = meta["start_row"].to_list()
        counts = meta["num_rows"].to_list()
        self.acc_offsets = starts + [starts[-1] + counts[-1]] if starts else [0]
        if self.use_ann:
            if not (index_path / "vamana_index").exists():
                self.construct_ann_index(index_path)

            self.index = diskannpy.StaticMemoryIndex(
                distance_metric="mips",
                vector_dtype=np.float32,
                index_directory=str(index_path / "vamana_index"),
                num_threads=0,
                initial_search_complexity=128,
            )
            self.all_embeddings = None
        elif self.use_rabitq:
            rabitq_dir = index_path / "rabitq"
            if not rabitq_dir.exists():
                print("RaBitQ index not found, building from embeddings.fbin...")
                _build_rabitq_index(index_path / "embeddings.fbin", rabitq_dir)
            self.rabitq_index = RaBitQIndex.load(rabitq_dir)
            self.all_embeddings = None
        else:
            mmap = _load_fbin_mmap(index_path / "embeddings.fbin")
            self._mmap = mmap  # keep reference to prevent GC closing the mapping
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
        raw_mmap = _create_fbin_memmap(
            index_path / "embeddings.fbin", total_chunks, embed_dim
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

        # Window size caps how many completed-but-uncollected results sit in RAM.
        # Submitting all futures at once lets workers race far ahead of the main
        # loop, causing unbounded result accumulation that triggers OOM kills.
        window_size = num_gpus * 2
        batch_iter = iter(batched(self._iter_chunks(accessions), submission_batch_size))
        total_batches = -(-total_chunks // submission_batch_size)  # ceil div

        def _next_future(ex):
            batch = next(batch_iter, None)
            return (
                ex.submit(_process_sequence_batch, batch) if batch is not None else None
            )

        with ProcessPoolExecutor(
            max_workers=num_gpus,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(self.cfg, gpu_queue),
        ) as executor:
            # Seed the window
            window = [
                f
                for _ in range(window_size)
                if (f := _next_future(executor)) is not None
            ]

            # Process in submission order so accession chunks land contiguously.
            with tqdm(total=total_batches, desc="Embedding batches") as pbar:
                while window:
                    future = window.pop(0)
                    try:
                        srr_ids, embeddings = future.result()
                    except process.BrokenProcessPool:
                        print("\n[!] A worker died abruptly. Halting.")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    except Exception as e:
                        print(f"Worker failed: {e}")
                        pbar.update(1)
                        nxt = _next_future(executor)
                        if nxt:
                            window.append(nxt)
                        continue

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

                    pbar.update(1)
                    nxt = _next_future(executor)
                    if nxt:
                        window.append(nxt)

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
        print(
            f"Built: {offset:,} vectors, {len(final_rows)} accessions -> {index_path}"
        )

        if self.use_rabitq:
            print("Building RaBitQ quantized index...")
            _build_rabitq_index(index_path / "embeddings.fbin", index_path / "rabitq")

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
        if self.use_ann:
            n_queries = len(queries)
            n_acc = len(self.acc_names_flat)
            n_chunks = len(query_chunks)
            identifiers, distances = self.index.batch_search(
                query_chunk_features.cpu().numpy(), 10, 128, num_threads=0
            )

            chunk_to_query_np = np.zeros(n_chunks, dtype=np.int64)
            for qi, (s, e) in enumerate(query_indices):
                chunk_to_query_np[s:e] = qi

            flat_ids = identifiers.ravel()
            flat_dists = distances.ravel()
            chunk_idx_flat = np.repeat(np.arange(n_chunks), 10)

            acc_offsets_arr = np.array(self.acc_offsets)
            acc_idx_flat = np.searchsorted(acc_offsets_arr, flat_ids, side="right") - 1
            query_idx_flat = chunk_to_query_np[chunk_idx_flat]

            scores_cpu = -2 * np.ones((n_queries, n_acc), dtype=np.float32)
            np.maximum.at(scores_cpu, (query_idx_flat, acc_idx_flat), flat_dists)
        elif self.use_rabitq:
            n_chunks = len(query_chunks)
            n_queries = len(queries)
            n_acc = len(self.acc_names_flat)
            top_k = 10

            chunk_to_query_np = np.zeros(n_chunks, dtype=np.int64)
            for qi, (s, e) in enumerate(query_indices):
                chunk_to_query_np[s:e] = qi

            scores_tensor, ids_tensor = self.rabitq_index.search(
                query_chunk_features, top_k
            )

            flat_ids = ids_tensor.numpy().ravel()
            flat_dists = scores_tensor.numpy().ravel()
            chunk_idx_flat = np.repeat(np.arange(n_chunks), top_k)

            acc_offsets_arr = np.array(self.acc_offsets)
            acc_idx_flat = np.searchsorted(acc_offsets_arr, flat_ids, side="right") - 1
            query_idx_flat = chunk_to_query_np[chunk_idx_flat]

            scores_cpu = -2 * np.ones((n_queries, n_acc), dtype=np.float32)
            np.maximum.at(scores_cpu, (query_idx_flat, acc_idx_flat), flat_dists)
        elif self.exact_search:
            assert self.all_embeddings is not None
            n_chunks = len(query_chunks)
            n_queries = len(queries)
            n_acc = len(self.acc_names_flat)
            top_k = 10

            # Pre-compute once: which query each chunk belongs to
            chunk_to_query_np = np.zeros(n_chunks, dtype=np.int64)
            for qi, (s, e) in enumerate(query_indices):
                chunk_to_query_np[s:e] = qi

            # Shard all_embeddings across available GPUs and compute matmul in parallel.
            # GPU ops release the GIL so threads give true parallelism.
            all_embeddings = self.all_embeddings
            n_vecs = all_embeddings.shape[0]
            n_gpus = torch.cuda.device_count()
            shard_size = (n_vecs + n_gpus - 1) // n_gpus
            qcf_cpu = query_chunk_features.cpu()

            def _matmul_shard(gpu_id: int):
                dev = torch.device(f"cuda:{gpu_id}")
                s = gpu_id * shard_size
                e = min(s + shard_size, n_vecs)
                emb = all_embeddings[s:e].to(dev)
                q = qcf_cpu.to(dev)
                logits = q @ emb.T  # (n_chunks, shard_size)
                k = min(top_k, logits.shape[1])
                vals, idx = torch.topk(logits, k, dim=-1)
                if k < top_k:
                    pad = top_k - k
                    vals = torch.nn.functional.pad(vals, (0, pad), value=float("-inf"))
                    idx = torch.nn.functional.pad(idx, (0, pad), value=0)
                return vals.cpu(), idx.cpu() + s

            import concurrent.futures as _cf

            with _cf.ThreadPoolExecutor(max_workers=n_gpus) as pool:
                shard_results = list(pool.map(_matmul_shard, range(n_gpus)))

            # Merge per-shard top-k into global top-k
            all_vals = torch.cat(
                [r[0] for r in shard_results], dim=1
            )  # (n_chunks, n_gpus*top_k)
            all_idx = torch.cat([r[1] for r in shard_results], dim=1)
            top_vals, top_pos = torch.topk(all_vals, top_k, dim=-1)
            identifiers = torch.gather(all_idx, 1, top_pos)
            distances = top_vals

            flat_ids = identifiers.numpy().ravel()
            flat_dists = distances.numpy().ravel()
            chunk_idx_flat = np.repeat(np.arange(n_chunks), top_k)

            acc_offsets_arr = np.array(self.acc_offsets)
            acc_idx_flat = np.searchsorted(acc_offsets_arr, flat_ids, side="right") - 1
            query_idx_flat = chunk_to_query_np[chunk_idx_flat]

            scores_cpu = -2 * np.ones((n_queries, n_acc), dtype=np.float32)
            np.maximum.at(scores_cpu, (query_idx_flat, acc_idx_flat), flat_dists)
        else:
            assert self.all_embeddings is not None
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
                acc_scores = torch.zeros(
                    n_queries, device=device, dtype=chunk_maxes.dtype
                )
                acc_scores.scatter_add_(0, chunk_to_query, chunk_maxes)  # (n_queries,)
                all_scores.append(acc_scores)

            scores = torch.stack(all_scores, dim=1)  # (n_queries, n_acc)
            scores_cpu = scores.float().cpu().numpy()

        accession_names = self.acc_names_flat
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

    def index_size_gb(self, index_path: Path):
        total = 0
        if self.use_rabitq:
            rabitq_dir = index_path / "rabitq"
            assert rabitq_dir.exists()
            total += sum(f.stat().st_size for f in rabitq_dir.iterdir()) / (1024**3)
        else:
            meta_gb = (index_path / "meta.parquet").stat().st_size / (1024**3)
            embeds_gb = (index_path / "embeddings.fbin").stat().st_size / (1024**3)
            total = meta_gb + embeds_gb

        return total

    def construct_ann_index(self, index_path: Path):
        print("Constructing diskANN index")
        (index_path / "vamana_index").mkdir(exist_ok=True, parents=True)
        diskannpy.build_memory_index(
            data=str(index_path / "embeddings.fbin"),
            distance_metric="mips",
            index_directory=str(index_path / "vamana_index"),
            complexity=128,
            graph_degree=64,
            num_threads=0,
            alpha=1.2,
            use_pq_build=False,
            num_pq_bytes=0,
            use_opq=False,
            vector_dtype=np.float32,
            tags="",
            filter_labels=None,
            universal_label="",
            filter_complexity=0,
        )

    @staticmethod
    def merge_shards(index_path: Path, num_nodes: int, use_rabitq: bool = False):
        """Stream per-node shard embeddings into a single memmap file — no full load into RAM."""
        all_meta = []
        total_vectors = 0
        embed_dim = None

        for rank in range(num_nodes):
            shard_path = index_path / f"shard_{rank}"
            arr = _load_fbin_mmap(shard_path / "embeddings.fbin")
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

        merged = _create_fbin_memmap(
            index_path / "embeddings.fbin", total_vectors, embed_dim
        )
        offset = 0
        for rank in range(num_nodes):
            arr = _load_fbin_mmap(index_path / f"shard_{rank}" / "embeddings.fbin")
            n = len(arr)
            merged[offset : offset + n] = arr
            offset += n
            del arr
            print(f"  Streamed shard {rank} ({n} vectors)")

        merged.flush()
        del merged

        pl.concat(all_meta).write_parquet(index_path / "meta.parquet")
        print(f"Merged {num_nodes} shards -> {total_vectors} vectors")

        if use_rabitq:
            print("Building RaBitQ quantized index from merged embeddings...")
            _build_rabitq_index(index_path / "embeddings.fbin", index_path / "rabitq")


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
