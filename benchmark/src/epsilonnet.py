"""Epsilon-net index: one greedy net per accession, searched as a smaller
exact index.

Artifact in the index directory:
    net_ids.pt       sorted int64 rows of the encoder's fbin kept in the net
    embeddings.fbin  those rows' vectors, in net_ids order

Search is ExactEngine over the net's own fbin (streamed, split across the
node's GPUs); hit rows are mapped back to encoder-fbin rows through net_ids,
so the accession regrouping downstream is unchanged. The build runs one
thread per GPU, each taking the largest remaining accession next.
"""

import os
import queue
import threading
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from .config import EpsilonNetIndex, ExactIndex
from .exact import BLOCK_ROWS, ExactEngine
from .fbin import _create_fbin_memmap, _load_fbin_mmap, read_fbin_rows

NET_IDS = "net_ids.pt"
NET_FBIN = "embeddings.fbin"
# Score tile budget in greedy_net: (batch, centers) fp32 elements per matmul.
SIM_TILE_ELEMS = 2**30  # 4 GB


@torch.no_grad()
def greedy_net(
    x: torch.Tensor,
    eps: float,
    batch_size: int = 4096,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Indices into x (unit-norm rows) of a greedy eps-net: every row has cosine
    similarity >= 1 - eps to some center. Rows are visited in one random order,
    batch_size at a time; a row becomes a center if no earlier center covers
    it. Rows in the same batch are not checked against each other, so larger
    batches run faster but can add near-duplicate centers. Memory: two (n, d)
    buffers plus a SIM_TILE_ELEMS score tile."""
    n, d = x.shape
    perm = torch.randperm(n, device=x.device, generator=generator)
    net_points = torch.empty((n, d), dtype=x.dtype, device=x.device)
    net_idx = torch.empty(n, dtype=torch.long, device=x.device)
    m = 0  # centers so far: net_points[:m], net_idx[:m]
    for i in range(0, n, batch_size):
        p_idx = perm[i : i + batch_size]
        if m == 0:
            new = p_idx
        else:
            # Max over the centers in column tiles: the full (batch, m) score
            # matrix is 33 GB for sra500's largest accession.
            p = x[p_idx]
            cols = max(1, SIM_TILE_ELEMS // len(p_idx))
            maxsim = torch.full((len(p_idx),), -2.0, device=x.device, dtype=x.dtype)
            for c in range(0, m, cols):
                tile_max = (p @ net_points[c : min(c + cols, m)].T).max(dim=1).values
                maxsim = torch.maximum(maxsim, tile_max)
            new = p_idx[maxsim < 1 - eps]
        k = len(new)
        net_points[m : m + k] = x[new]
        net_idx[m : m + k] = new
        m += k
    return net_idx[:m]


def write_net(fbin_dir: Path, index_dir: Path, ids: torch.Tensor) -> None:
    """Write net_ids.pt and the net's own fbin (rows ``ids`` of the encoder
    fbin). One sequential pass over the encoder fbin in BLOCK_ROWS blocks, so
    the gather never random-reads Lustre."""
    ids_np = np.sort(ids.numpy().astype(np.int64))
    src = _load_fbin_mmap(fbin_dir / "embeddings.fbin")
    n_src, d = src.shape
    del src
    index_dir.mkdir(parents=True, exist_ok=True)
    out = _create_fbin_memmap(index_dir / NET_FBIN, len(ids_np), d)
    buf = np.empty((BLOCK_ROWS, d), dtype=np.float32)
    fd = os.open(fbin_dir / "embeddings.fbin", os.O_RDONLY)
    try:
        blocks = range(0, n_src, BLOCK_ROWS)
        for bs in tqdm(blocks, desc=f"Writing {len(ids_np):,}-vector net fbin"):
            be = min(bs + BLOCK_ROWS, n_src)
            j0, j1 = np.searchsorted(ids_np, [bs, be])
            if j0 == j1:
                continue
            rows = read_fbin_rows(fd, bs, be, buf)
            out[j0:j1] = rows[ids_np[j0:j1] - bs]
    finally:
        os.close(fd)
    out.flush()
    del out
    torch.save(torch.from_numpy(ids_np), index_dir / NET_IDS)


class EpsilonNetEngine:
    # Multi-node row sharding would split the encoder fbin's row count
    # (run_benchmark), not the net's; one node's GPUs scan the net instead.
    SHARDABLE = False
    WORKER_EMBED = False

    def __init__(self, cfg: EpsilonNetIndex):
        self.cfg = cfg
        self.net_ids: np.ndarray | None = None
        self.exact: ExactEngine | None = None

    def build(
        self, fbin_dir: Path, index_dir: Path, shard: int, num_shards: int
    ) -> None:
        if num_shards > 1:
            raise ValueError(
                f"{type(self).__name__} build is single-process (num_shards must "
                "be 1); it already uses every visible GPU"
            )
        meta = pl.read_parquet(fbin_dir / "meta.parquet")
        ids = self._select(fbin_dir, meta)
        n = int(meta["num_rows"].sum())
        print(f"net: {len(ids):,} of {n:,} vectors ({n / max(len(ids), 1):.3f}x)")
        write_net(fbin_dir, index_dir, ids)

    def _select(self, fbin_dir: Path, meta: pl.DataFrame) -> torch.Tensor:
        """Encoder-fbin rows of every accession's net. One worker thread per
        visible GPU (CPU when none); each takes the largest remaining
        accession, so the few big accessions start first and run side by
        side. Accession i's visiting order is seeded with i, so the net does
        not depend on which GPU built it."""
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
        starts = meta["start_row"].to_list()
        counts = meta["num_rows"].to_list()
        todo: queue.Queue = queue.Queue()
        for i in sorted(range(len(counts)), key=lambda i: -counts[i]):
            todo.put(i)
        nets: list[torch.Tensor | None] = [None] * len(counts)
        errors: list[BaseException] = []
        pbar = tqdm(
            total=len(counts), desc=f"Building epsilon nets ({len(devices)} device(s))"
        )
        lock = threading.Lock()
        path = fbin_dir / "embeddings.fbin"
        d = _load_fbin_mmap(path).shape[1]

        def worker(dev: str) -> None:
            fd = os.open(path, os.O_RDONLY)
            try:
                while not errors:
                    try:
                        i = todo.get_nowait()
                    except queue.Empty:
                        return
                    host = np.empty((counts[i], d), dtype=np.float32)
                    rows = read_fbin_rows(fd, starts[i], starts[i] + counts[i], host)
                    x = torch.from_numpy(rows).to(dev)
                    gen = torch.Generator(device=dev).manual_seed(i)
                    local = greedy_net(x, self.cfg.epsilon, generator=gen)
                    nets[i] = (local + starts[i]).cpu()
                    del x, local
                    with lock:
                        pbar.update(1)
            except BaseException as e:  # surfaced after join
                errors.append(e)
            finally:
                os.close(fd)

        threads = [threading.Thread(target=worker, args=(dev,)) for dev in devices]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        pbar.close()
        if errors:
            raise errors[0]
        return torch.cat(nets)

    def load(
        self, fbin_dir: Path, index_dir: Path, devices: list[str], encoder_cfg=None
    ) -> None:
        self.net_ids = torch.load(index_dir / NET_IDS).numpy()
        self.exact = ExactEngine(ExactIndex(top_k=self.cfg.top_k))
        self.exact.load(index_dir, index_dir, devices)  # the net's own fbin
        if self.exact.all_embeddings.shape[0] != len(self.net_ids):
            raise ValueError(f"{index_dir}: {NET_FBIN} and {NET_IDS} disagree")

    def topk_hits(
        self,
        query_vecs: np.ndarray,
        top_k: int,
        vec_range: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """ExactEngine's top_k over the net, rows mapped to encoder-fbin rows
        (-1 padding kept)."""
        assert self.exact is not None and self.net_ids is not None
        scores, rows = self.exact.topk_hits(query_vecs, top_k, vec_range)
        ids = np.where(rows >= 0, self.net_ids[np.maximum(rows, 0)], -1)
        return scores, ids

    def size_gb(self, fbin_dir: Path, index_dir: Path) -> float:
        """The net's vectors plus its id map, in GiB like ExactEngine."""
        files = (index_dir / NET_FBIN, index_dir / NET_IDS)
        return sum(f.stat().st_size for f in files) / (1024**3)
