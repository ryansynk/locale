"""Exact fp32 top-k scan over embeddings.fbin, streamed in blocks across GPUs.

No artifact of its own: ``build`` is a no-op and ``load`` memory-maps the
encoder's fbin. The scan reads row blocks with pread (see fbin.pread_into for
why not memmap slicing) into a pinned buffer per device thread and folds each
block's top-k into a running (n_queries, top_k) buffer, so a multi-TB index
never has to fit anywhere. ``vec_range`` makes it shardable: a node's hits are
exact over its own row range, so node 0 merges them with
topk_regroup.merge_topk_hits.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import ExactIndex
from .fbin import _load_fbin_mmap, read_fbin_rows

BLOCK_ROWS = 2_000_000  # 2M x 768 fp32 = 6 GB per block


class FbinBlockLoader:
    """Brings row blocks of an fbin onto one device: pread into a pinned host
    buffer, then a single H2D copy. One instance per scanning thread; each
    holds its own descriptor (positional reads, no shared offset) and one
    (block_rows, d) staging buffer, so a 2M-row block is 6 GB of pinned
    memory per GPU.
    """

    def __init__(self, path: Path, d: int, block_rows: int, device: torch.device):
        self.fd = os.open(path, os.O_RDONLY)
        self.device = device
        self.buf = torch.empty(
            (block_rows, d), dtype=torch.float32, pin_memory=device.type == "cuda"
        )

    def __call__(self, bs: int, be: int) -> torch.Tensor:
        """Rows [bs, be) as a (be - bs, d) tensor on the device."""
        read_fbin_rows(self.fd, bs, be, self.buf.numpy())
        return self.buf[: be - bs].to(self.device)

    def close(self) -> None:
        os.close(self.fd)


@contextmanager
def block_loader(
    fbin_path: Path | None, all_embeddings: torch.Tensor, block_rows: int, device: torch.device
):
    """Yields load(bs, be) -> (be - bs, d) tensor of index rows on device.

    Reads through FbinBlockLoader when the index is file-backed, else slices
    ``all_embeddings`` (in-memory tests). One loader per thread: call inside
    the thread.
    """
    if fbin_path is None:
        yield lambda bs, be: all_embeddings[bs:be].to(device)
        return
    loader = FbinBlockLoader(fbin_path, all_embeddings.shape[1], block_rows, device)
    try:
        yield loader
    finally:
        loader.close()


class ExactEngine:
    SHARDABLE = True  # topk_hits accepts vec_range (multi-node row sharding)
    WORKER_EMBED = False

    def __init__(self, cfg: ExactIndex):
        self.cfg = cfg
        self.all_embeddings: torch.Tensor | None = None
        # None for in-memory embeddings (tests), which the scan then slices.
        self.fbin_path: Path | None = None
        self.devices: list[str] = ["cpu"]

    def build(self, fbin_dir: Path, index_dir: Path, shard: int, num_shards: int) -> None:
        pass

    def load(self, fbin_dir: Path, index_dir: Path, devices: list[str], encoder_cfg=None) -> None:
        mmap = _load_fbin_mmap(fbin_dir / "embeddings.fbin")
        self._mmap = mmap  # keep reference to prevent GC closing the mapping
        self.all_embeddings = torch.from_numpy(mmap)
        self.fbin_path = fbin_dir / "embeddings.fbin"
        self.devices = list(devices)
        print(f"Loaded {mmap.shape[0]:,} vectors [memory-mapped]")

    @torch.no_grad()
    def topk_hits(
        self,
        query_vecs: np.ndarray,
        top_k: int,
        vec_range: tuple[int, int] | None = None,
        block_rows: int = BLOCK_ROWS,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per query row, its top_k index vectors over rows [vec_range).

        Returns (scores, ids), each (n_queries, k') with k' <= top_k,
        score-descending; ids are global fbin rows, -1 = padding. The range is
        split evenly across the devices; each streams its sub-range in
        block_rows blocks.
        """
        assert self.all_embeddings is not None
        n_vecs = self.all_embeddings.shape[0]
        start, end = (0, n_vecs) if vec_range is None else vec_range
        if not (0 <= start <= end <= n_vecs):
            raise ValueError(f"vec_range {vec_range} outside [0, {n_vecs}]")
        qcf_cpu = torch.as_tensor(np.ascontiguousarray(query_vecs, dtype=np.float32))
        n_chunks = len(qcf_cpu)
        devices = self.devices
        n_dev = len(devices)
        per_dev = -(-(end - start) // n_dev) if end > start else 0

        def _scan(dev_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            dev = torch.device(devices[dev_idx])
            s = start + dev_idx * per_dev
            e = min(s + per_dev, end)
            q = qcf_cpu.to(dev)
            best_vals = torch.full(
                (n_chunks, top_k), float("-inf"), device=dev, dtype=q.dtype
            )
            best_ids = torch.full((n_chunks, top_k), -1, device=dev, dtype=torch.int64)
            blocks = range(s, e, block_rows)
            if dev_idx == 0:
                blocks = tqdm(
                    blocks,
                    desc=f"Exact top-{top_k} vector scan ({n_dev} device(s), "
                    f"{end - start:,} vectors)",
                )
            with block_loader(self.fbin_path, self.all_embeddings, block_rows, dev) as load_block:
                for bs in blocks:
                    be = min(bs + block_rows, e)
                    logits = q @ load_block(bs, be).T  # (n_chunks, be-bs)
                    k = min(top_k, be - bs)
                    vals, idx = torch.topk(logits, k, dim=-1)
                    # The logits tile is the big allocation (2024 chunks x 2M
                    # rows fp32 = 15 GB with 1000 both-strand queries). Drop it
                    # before the next block is loaded; otherwise the previous
                    # tile is still bound when the next matmul allocates and
                    # the peak is two tiles plus a block, which OOMs a 40 GB
                    # A100.
                    del logits
                    cand_vals = torch.cat([best_vals, vals], dim=1)
                    cand_ids = torch.cat([best_ids, idx + bs], dim=1)
                    best_vals, pos = torch.topk(cand_vals, top_k, dim=-1)
                    best_ids = torch.gather(cand_ids, 1, pos)
                    del vals, idx, cand_vals, cand_ids, pos
            return best_vals.cpu(), best_ids.cpu()

        with ThreadPoolExecutor(max_workers=n_dev) as pool:
            shard_results = list(pool.map(_scan, range(n_dev)))

        # Per-device buffers are exact over disjoint sub-ranges; fold them into
        # per-row global top-k. Padding slots (id -1, -inf) stay at the tail.
        all_vals = torch.cat([r[0] for r in shard_results], dim=1)
        all_ids = torch.cat([r[1] for r in shard_results], dim=1)
        k = min(top_k, all_vals.shape[1])
        top_vals, top_pos = torch.topk(all_vals, k, dim=-1)
        top_ids = torch.gather(all_ids, 1, top_pos)
        return top_vals.numpy(), top_ids.numpy()

    def size_gb(self, fbin_dir: Path, index_dir: Path) -> float:
        meta_gb = (fbin_dir / "meta.parquet").stat().st_size / (1024**3)
        embeds_gb = (fbin_dir / "embeddings.fbin").stat().st_size / (1024**3)
        return meta_gb + embeds_gb
