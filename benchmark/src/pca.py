"""PCA baseline: every encoder vector projected onto the top pca_dim
eigenvectors of a per-accession-balanced sample's second moment, then the
same exact scan as ExactEngine over the projected fbin. Rows keep their
encoder-fbin ids, so hits need no remapping and vec_range sharding works."""

import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from .config import ExactIndex, PCAIndex
from .exact import BLOCK_ROWS, ExactEngine
from .fbin import _load_fbin_mmap, read_fbin_rows
from .fbin_stream import transform_fbin

PCA_PROJECTOR = "projector.pt"
PCA_FBIN = "embeddings.fbin"


class PCAEngine:
    SHARDABLE = True  # topk_hits accepts vec_range (multi-node row sharding)
    WORKER_EMBED = False

    def __init__(self, cfg: PCAIndex):
        self.cfg = cfg
        self.exact: ExactEngine | None = None
        self.projector: torch.Tensor | None = None
        self.devices: list[str] = ["cpu"]

    def build(
        self, fbin_dir: Path, index_dir: Path, shard: int, num_shards: int
    ) -> None:
        if num_shards > 1:
            raise ValueError(
                f"{type(self).__name__} build is single-process (num_shards must be 1)"
            )
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        src_path = fbin_dir / "embeddings.fbin"
        mmap = _load_fbin_mmap(src_path)
        n_src, d = mmap.shape
        k = self.cfg.pca_dim
        if not 0 < k <= d:
            raise ValueError(f"pca_dim must be in [1, {d}], got {k}")

        # Training sample: up to N_total / n_accessions rows of every
        # accession, so the few big accessions do not dominate the basis.
        meta = pl.read_parquet(fbin_dir / "meta.parquet")
        starts = meta["start_row"].to_list()
        counts = meta["num_rows"].to_list()
        N_total = 100_000
        per_acc = max(1, N_total // len(counts))
        gen = torch.Generator().manual_seed(0)
        train_ids = torch.cat(
            [
                start + torch.randperm(c, generator=gen)[: min(c, per_acc)]
                for c, start in zip(counts, starts)
            ]
        )
        del mmap
        # One pread per row, ascending: about twice as fast on Lustre as
        # faulting the same scattered rows through the memmap (login node,
        # 2026-10-08), and it gives the gather a progress bar.
        train_ids = np.sort(train_ids.numpy())
        t0 = time.time()
        host = np.empty((len(train_ids), d), dtype=np.float32)
        fd = os.open(src_path, os.O_RDONLY)
        try:
            for j, i in enumerate(
                tqdm(
                    train_ids,
                    desc="Reading PCA training sample",
                    unit="row",
                    mininterval=5,
                )
            ):
                read_fbin_rows(fd, int(i), int(i) + 1, host[j : j + 1])
        finally:
            os.close(fd)
        sample = torch.from_numpy(host).to(device)
        print(f"Read {len(train_ids):,} sampled rows in {time.time() - t0:.0f}s")

        # Uncentered PCA: the top-k eigenvectors of the second moment X^T X
        # minimise E||x - P x||^2, which is what inner-product search needs
        # (centering would only shift every score of a query by a constant).
        # eigh of the d x d Gram matrix in fp64 instead of an SVD of the sample.
        t0 = time.time()
        gram = sample.double().T @ sample.double()
        evals, evecs = torch.linalg.eigh(gram)  # ascending
        self.projector = evecs[:, -k:].flip(1).float().contiguous()  # (d, k)
        kept = (evals[-k:].sum() / evals.sum()).item()
        print(
            f"PCA {d} -> {k} from {len(train_ids):,} sampled rows: "
            f"{kept:.4f} of the second moment kept ({time.time() - t0:.1f}s)"
        )
        del sample, host, gram

        # Stream the encoder fbin through the projection into the index's own
        # fbin; rows keep their encoder-fbin ids, so ExactEngine's hits need no
        # remapping.
        index_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.projector.cpu(), index_dir / PCA_PROJECTOR)
        projector = self.projector
        transform_fbin(
            src_path,
            index_dir / PCA_FBIN,
            lambda x: x @ projector,
            k,
            device,
            desc=f"Projecting {d} -> {k}",
        )

    def load(
        self, fbin_dir: Path, index_dir: Path, devices: list[str], encoder_cfg=None
    ) -> None:
        self.projector = torch.load(index_dir / PCA_PROJECTOR)  # (d, pca_dim)
        self.exact = ExactEngine(ExactIndex(top_k=self.cfg.top_k))
        self.exact.load(index_dir, index_dir, devices)  # the projected fbin
        self.devices = list(devices)
        n_src = _load_fbin_mmap(fbin_dir / "embeddings.fbin").shape[0]
        n_pca, k = self.exact.all_embeddings.shape
        if n_pca != n_src or k != self.projector.shape[1]:
            raise ValueError(
                f"{index_dir}: projected fbin is {n_pca:,} x {k}, expected "
                f"{n_src:,} x {self.projector.shape[1]}"
            )

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
        assert self.exact is not None
        assert self.projector is not None
        q = torch.as_tensor(np.ascontiguousarray(query_vecs, dtype=np.float32))
        lowdim_query_vecs = (q @ self.projector).numpy()  # (n_queries, pca_dim)
        return self.exact.topk_hits(lowdim_query_vecs, top_k, vec_range, block_rows)

    def size_gb(self, fbin_dir: Path, index_dir: Path) -> float:
        """The projected vectors plus the projector, in GiB like ExactEngine."""
        files = (index_dir / PCA_FBIN, index_dir / PCA_PROJECTOR)
        return sum(f.stat().st_size for f in files) / (1024**3)
