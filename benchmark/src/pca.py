"""PCA baseline: every encoder vector projected onto the top pca_dim
eigenvectors of a per-accession-balanced sample's second moment, then the
same exact scan as ExactEngine over the projected fbin. Rows keep their
encoder-fbin ids, so hits need no remapping and vec_range sharding works.

``dtype`` stores the projected values as fp32 (embeddings.fbin) or rounded
to fp16 / bf16 (embeddings.{fp16,bf16}bin, quantized_exact's format and
rounding); the scan upcasts each block to fp32 on the device either way, so
the queries are projected in fp32 and only the stored values are rounded.

The build shards rows over nodes like rabitq's (see PCAEngine.build): rank
0 fits the projector, every rank writes its row range of one preallocated
file using all of its GPUs, and every rank returns once build.json says the
index is complete. One node of N reads 1/N of the encoder fbin, which is
what makes sra4571 (6.3 TB) a short job."""

import json
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
from .fbin_stream import create_fbin_file, transform_rows_parallel
from .quantized_exact import TORCH_DTYPES, quantized_file, round_to
from .rabitq import _atomic_write_json, _devices, _wait_for, shard_bounds

PCA_PROJECTOR = "projector.pt"
PCA_BUILD_DONE = "build.json"  # rank 0 writes it once every shard is in


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp)
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    tmp.rename(path)


def pca_file(dtype: str) -> str:
    """The projected vectors' file: the plain fbin for fp32, quantized_exact's
    name for a 2-byte dtype."""
    return "embeddings.fbin" if dtype == "fp32" else quantized_file(dtype)


class PCAEngine:
    SHARDABLE = True  # topk_hits accepts vec_range (multi-node row sharding)
    WORKER_EMBED = False
    # How long a rank waits for another's files during build (as rabitq).
    WAIT_TIMEOUT = 8 * 3600.0
    POLL = 15.0

    def __init__(self, cfg: PCAIndex):
        if cfg.dtype != "fp32" and cfg.dtype not in TORCH_DTYPES:
            raise ValueError(
                f"dtype must be fp32 or one of {list(TORCH_DTYPES)}, got {cfg.dtype}"
            )
        self.cfg = cfg
        self.dtype = torch.float32 if cfg.dtype == "fp32" else TORCH_DTYPES[cfg.dtype]
        self.exact: ExactEngine | None = None
        self.projector: torch.Tensor | None = None
        self.devices: list[str] = ["cpu"]

    def build(
        self, fbin_dir: Path, index_dir: Path, shard: int, num_shards: int
    ) -> None:
        """Build (or resume) the index across ``num_shards`` ranks (nodes);
        every rank returns once build.json exists, so the index is complete
        and loadable on return (run_benchmark marks it .done after rank 0's
        build returns, in either stage).

        Rank 0 fits the projector, creates the full-size output file and then
        saves projector.pt, whose presence releases the other ranks. Each rank
        projects its contiguous row range (rabitq.shard_bounds) on all of its
        GPUs, one sub-range and read stream per GPU, writes it into the shared
        file and records shard_rank_<r>.json. Rank 0 checks every marker's
        range and writes build.json. A rank whose marker exists skips its
        range, so a timed-out build resumes; rerun with the same num_shards.
        """
        index_dir.mkdir(parents=True, exist_ok=True)
        done_p = index_dir / PCA_BUILD_DONE
        if done_p.exists():
            return
        src_path = fbin_dir / "embeddings.fbin"
        n_src, d = _load_fbin_mmap(src_path).shape
        k = self.cfg.pca_dim
        if not 0 < k <= d:
            raise ValueError(f"pca_dim must be in [1, {d}], got {k}")
        if not 0 <= shard < num_shards:
            raise ValueError(f"shard {shard} outside [0, {num_shards})")
        devices = _devices()
        out_path = index_dir / pca_file(self.cfg.dtype)
        proj_p = index_dir / PCA_PROJECTOR

        if shard == 0 and not proj_p.exists():
            projector = self._fit_projector(fbin_dir, src_path, devices[0])
            create_fbin_file(out_path, n_src, k, self.dtype)
            _atomic_torch_save(projector.cpu(), proj_p)  # last: releases others
        elif not proj_p.exists():
            print(f"[rank {shard}] Waiting for the PCA projector from rank 0...")
        _wait_for([proj_p], self.WAIT_TIMEOUT, self.POLL)
        self.projector = torch.load(proj_p)
        if tuple(self.projector.shape) != (d, k):
            raise ValueError(
                f"{proj_p} is {tuple(self.projector.shape)}, expected {(d, k)}"
            )

        bounds = shard_bounds(n_src, num_shards)
        start, end = bounds[shard], bounds[shard + 1]
        marker = index_dir / f"shard_rank_{shard}.json"
        if marker.exists():
            got = json.loads(marker.read_text())
            if (got["start"], got["end"]) != (start, end):
                raise ValueError(
                    f"{marker} covers [{got['start']}, {got['end']}), not "
                    f"[{start}, {end}): it was written by a build with a "
                    "different num_shards; delete the shard markers and rebuild"
                )
            print(f"[rank {shard}] rows [{start:,}, {end:,}) already written")
        else:
            projector, dtype = self.projector, self.cfg.dtype

            def make_transform(dev: str):
                p = projector.to(dev)
                if dtype == "fp32":
                    return lambda x: x @ p
                return lambda x: round_to(x @ p, dtype)

            transform_rows_parallel(
                src_path,
                out_path,
                make_transform,
                k,
                self.dtype,
                devices,
                start,
                end,
                desc=f"[rank {shard}/{num_shards}] Projecting {d} -> {k} ({dtype})",
            )
            _atomic_write_json(marker, {"rank": shard, "start": start, "end": end})

        if shard != 0:
            print(f"[rank {shard}] Shard done. Waiting for {PCA_BUILD_DONE}...")
            _wait_for([done_p], self.WAIT_TIMEOUT, self.POLL)
            return
        markers = [index_dir / f"shard_rank_{r}.json" for r in range(num_shards)]
        if num_shards > 1:
            print(f"[rank 0] Waiting for {num_shards - 1} other PCA shard(s)...")
        _wait_for(markers, self.WAIT_TIMEOUT, self.POLL)
        for r, m in enumerate(markers):
            got = json.loads(m.read_text())
            if (got["start"], got["end"]) != (bounds[r], bounds[r + 1]):
                raise ValueError(
                    f"{m} covers [{got['start']}, {got['end']}), not "
                    f"[{bounds[r]}, {bounds[r + 1]}): it was written by a build "
                    "with a different num_shards; delete the shard markers and "
                    "rebuild"
                )
        _atomic_write_json(
            done_p,
            {
                "n": n_src,
                "pca_dim": k,
                "dtype": self.cfg.dtype,
                "num_shards": num_shards,
            },
        )
        print(f"PCA index built: {n_src:,} x {k} {self.cfg.dtype} -> {out_path}")

    def _fit_projector(
        self, fbin_dir: Path, src_path: Path, device: str
    ) -> torch.Tensor:
        """The (d, pca_dim) projector: top eigenvectors of the second moment
        of a per-accession-balanced sample of the encoder fbin."""
        n_src, d = _load_fbin_mmap(src_path).shape
        k = self.cfg.pca_dim
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
        projector = evecs[:, -k:].flip(1).float().contiguous()  # (d, k)
        kept = (evals[-k:].sum() / evals.sum()).item()
        print(
            f"PCA {d} -> {k} from {len(train_ids):,} sampled rows: "
            f"{kept:.4f} of the second moment kept ({time.time() - t0:.1f}s)"
        )
        return projector

    def load(
        self, fbin_dir: Path, index_dir: Path, devices: list[str], encoder_cfg=None
    ) -> None:
        self.projector = torch.load(index_dir / PCA_PROJECTOR)  # (d, pca_dim)
        self.exact = ExactEngine(ExactIndex(top_k=self.cfg.top_k))
        self.exact.load_file(index_dir / pca_file(self.cfg.dtype), devices, self.dtype)
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
        files = (index_dir / pca_file(self.cfg.dtype), index_dir / PCA_PROJECTOR)
        return sum(f.stat().st_size for f in files) / (1024**3)
