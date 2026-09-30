import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import polars as pl
from tqdm import tqdm

from .config import EpsilonNetIndex
from .fbin import _load_fbin_mmap, read_fbin_rows

BLOCK_ROWS = 2_000_000  # 2M x 768 fp32 = 6 GB per block


@torch.no_grad()
def greedy_net(x: torch.Tensor, eps: float, batch_size: int = 4096) -> torch.Tensor:
    n, d = x.shape
    perm = torch.randperm(n, device=x.device)
    net_points = torch.empty((n, d), dtype=x.dtype, device=x.device)
    net_idx = torch.empty(n, dtype=torch.long, device=x.device)
    m = 0  # centers so far: net_points[:m], net_idx[:m]
    for i in range(0, n, batch_size):
        p_idx = perm[i : i + batch_size]
        if m == 0:
            new = p_idx
        else:
            maxsim = (x[p_idx] @ net_points[:m].T).max(dim=1).values
            new = p_idx[maxsim < 1 - eps]
        k = len(new)
        net_points[m : m + k] = x[new]
        net_idx[m : m + k] = new
        m += k
    return net_idx[:m]


class EpsilonNetEngine:
    SHARDABLE = False
    WORKER_EMBED = False

    def __init__(self, cfg: EpsilonNetIndex):
        self.cfg = cfg
        self.all_embeddings: torch.Tensor | None = None
        # None for in-memory embeddings (tests), which the scan then slices.
        self.fbin_path: Path | None = None
        self.meta_path: Path | None = None
        self.devices: list[str] = ["cpu"]
        self.net_ids: torch.Tensor | None = None
        self.net: torch.Tensor | None = None

    def build(
        self, fbin_dir: Path, index_dir: Path, shard: int, num_shards: int
    ) -> None:
        # One net per accession; ids saved as global fbin rows.
        if num_shards > 1:
            raise ValueError("epsilonnet build is single-node (num_shards must be 1)")
        meta = pl.read_parquet(fbin_dir / "meta.parquet")
        mmap = _load_fbin_mmap(fbin_dir / "embeddings.fbin")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        eps = self.cfg.epsilon
        net_ids = []
        for start, n in tqdm(
            zip(meta["start_row"].to_list(), meta["num_rows"].to_list()),
            total=meta.height,
            desc="Building epsilon nets",
        ):
            x = torch.from_numpy(mmap[start : start + n]).to(device)
            local = greedy_net(x, eps)  # return indexes within x of epsilon net points
            net_ids.append((local + start).cpu())

        index_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.cat(net_ids).sort().values, index_dir / "net_ids.pt")

    def load(
        self, fbin_dir: Path, index_dir: Path, devices: list[str], encoder_cfg=None
    ) -> None:
        mmap = _load_fbin_mmap(fbin_dir / "embeddings.fbin")
        self._mmap = mmap  # keep reference to prevent GC closing the mapping
        self.all_embeddings = torch.from_numpy(mmap)
        self.fbin_path = fbin_dir / "embeddings.fbin"
        self.devices = list(devices)
        print(f"Loaded {mmap.shape[0]:,} vectors [memory-mapped]")

        ids = torch.load(index_dir / "net_ids.pt")
        self.net = torch.from_numpy(mmap[ids.numpy()]).to(self.devices[0])
        self.net_ids = ids.to(self.devices[0])
        print(f"Loaded {self.net.shape[0]:,} net vectors")

    @torch.no_grad()
    def topk_hits(
        self,
        query_vecs: np.ndarray,
        top_k: int,
        vec_range: int | None = None,
        block_rows: int = BLOCK_ROWS,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.net is not None
        assert self.net_ids is not None
        q = torch.as_tensor(np.ascontiguousarray(query_vecs, dtype=np.float32))
        k = min(top_k, self.net.shape[0])

        # Query rows per block so the (rows, n_net) fp32 score matrix stays
        # ~4 GB: a 9M-vector net (eps 0.05 on sra50) gets ~110 rows per block.
        rows = max(1, min(4096, 2**30 // self.net.shape[0]))
        scores, ids = [], []
        for i in range(0, len(q), rows):
            s, j = torch.topk(
                q[i : i + rows].to(self.net.device) @ self.net.T, k, dim=1
            )
            scores.append(s.cpu())
            ids.append(self.net_ids[j].cpu())  # net_ids on device
        return torch.cat(scores).numpy(), torch.cat(ids).numpy()

    def size_gb(self, fbin_dir: Path, index_dir: Path) -> float:
        assert self.net is not None
        return self.net.shape[0] * self.net.shape[1] * 4 / 1e9
