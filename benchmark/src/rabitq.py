"""1-bit RaBitQ codes over an fbin: sharded build, packed-resident search.

Each index vector x is centered, rotated by a seeded random orthogonal matrix
and reduced to the sign of every coordinate (d bits = d/8 bytes). Search is
asymmetric: the fp32 rotated query against the +-1 codes, rescaled by the
per-vector norm and dot factors so the estimate of <q, x> is unbiased.

Layout of ``rabitq_dir``::

    centroid.npy, rotation.npy   transforms, written once by rank 0
    codes_rank_{r}.u8            (rows_r, d/8) packed sign bits, little bit order
    norms_rank_{r}.f32           ||rot(x - c)||
    dots_rank_{r}.f32            <rot(x - c), sign> / sqrt(d)
    shard_rank_{r}.json          written after a rank's three files are flushed
    meta.json                    written last, by rank 0; its presence means the
                                 index is complete

Every rank quantizes one contiguous, equal range of rows into its own files:
no two processes ever write the same file, so a multi-node build needs no
locking, and a rank whose marker exists is skipped on rerun. The loader reads
any global row range across shard files, so a search node holds only the
rows it scores. Codes stay packed in device memory (96 B/vector at d=768,
about 260 GB for the 2.5 B-vector 100-studies index, spread over any number
of GPUs) and are unpacked per sub-batch right before the matmul.

The centroid is estimated from a seeded random sample of contiguous row
blocks rather than a full pass: at 2 M rows the per-coordinate standard error
is ~1e-3, far below the codes' own error, and the estimator is unbiased for
any fixed centroid, so accuracy is all that is at stake -- not correctness.

The pre-shard layout (codes.u8/norms.f32/dots.f32, meta.json without
"shards") is read as one shard so existing indexes keep working.
"""

import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .fbin import _load_fbin_mmap, pread_into, read_fbin_rows

DEFAULT_CENTROID_SAMPLE_ROWS = 2_000_000
CENTROID_SAMPLE_BLOCK_ROWS = 1_000


class _FbinRows:
    """Rows of the fbin behind ``mmap`` by pread (see fbin.pread_into), falling
    back to slicing when the array is not file-backed (in-memory tests).
    Each thread gets its own descriptor and (max_rows, d) float32 buffer."""

    def __init__(self, mmap: np.ndarray, max_rows: int):
        self.mmap = mmap
        self.path = getattr(mmap, "filename", None)
        self.max_rows = max_rows
        self.d = mmap.shape[1]
        self._local = threading.local()

    def __call__(self, start: int, end: int) -> np.ndarray:
        if self.path is None:
            return np.array(self.mmap[start:end], dtype=np.float32)
        loc = self._local
        if not hasattr(loc, "fd"):
            loc.fd = os.open(self.path, os.O_RDONLY)
            loc.buf = np.empty((self.max_rows, self.d), dtype=np.float32)
        return read_fbin_rows(loc.fd, start, end, loc.buf)


def shard_bounds(n: int, num_ranks: int) -> list[int]:
    """Row boundaries of ``num_ranks`` contiguous, near-equal shards."""
    return [n * r // num_ranks for r in range(num_ranks + 1)]


def estimate_centroid(
    mmap: np.memmap,
    sample_rows: int,
    seed: int,
    block_rows: int = CENTROID_SAMPLE_BLOCK_ROWS,
) -> np.ndarray:
    """Mean vector from a seeded sample of contiguous row blocks (fp64 sum).

    Falls back to the exact streaming mean when the sample covers the whole
    file. Blocks rather than single rows keep the reads sequential-ish: 2000
    blocks of 1000 rows are 2000 x 3 MB reads instead of 2 M x 3 KB seeks.
    """
    n, d = mmap.shape
    acc = np.zeros(d, dtype=np.float64)
    if sample_rows >= n:
        step = max(block_rows, 200_000)
        rows_of = _FbinRows(mmap, step)
        for start in range(0, n, step):
            acc += rows_of(start, min(start + step, n)).sum(axis=0, dtype=np.float64)
        return (acc / n).astype(np.float32)

    num_blocks = n // block_rows
    want = min(num_blocks, -(-sample_rows // block_rows))
    rng = np.random.default_rng(seed)
    blocks = np.sort(rng.choice(num_blocks, size=want, replace=False))
    rows_of = _FbinRows(mmap, block_rows)
    rows = 0
    for b in tqdm(blocks, desc="Sampling centroid blocks", mininterval=5):
        chunk = rows_of(b * block_rows, (b + 1) * block_rows)
        acc += chunk.sum(axis=0, dtype=np.float64)
        rows += len(chunk)
    return (acc / rows).astype(np.float32)


def make_rotation(d: int, seed: int) -> np.ndarray:
    """Seeded random orthogonal matrix (QR of a Gaussian, sign-fixed)."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((d, d)).astype(np.float32)
    q, r = np.linalg.qr(g)
    return (q * np.sign(np.diag(r))).astype(np.float32)


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(".npy.tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(path)


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(path)


def _wait_for(paths: list[Path], timeout: float, poll: float) -> None:
    deadline = time.time() + timeout
    while not all(p.exists() for p in paths):
        if time.time() > deadline:
            missing = [str(p) for p in paths if not p.exists()]
            raise TimeoutError(f"Timed out after {timeout}s waiting for: {missing}")
        time.sleep(poll)


def _transform_paths(rabitq_dir: Path) -> tuple[Path, Path]:
    return rabitq_dir / "centroid.npy", rabitq_dir / "rotation.npy"


def _shard_files(rabitq_dir: Path, rank: int) -> tuple[Path, Path, Path, Path]:
    return (
        rabitq_dir / f"codes_rank_{rank}.u8",
        rabitq_dir / f"norms_rank_{rank}.f32",
        rabitq_dir / f"dots_rank_{rank}.f32",
        rabitq_dir / f"shard_rank_{rank}.json",
    )


def _devices() -> list[str]:
    n_gpus = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(n_gpus)] if n_gpus > 0 else ["cpu"]


def quantize_rows(
    mmap: np.memmap,
    rabitq_dir: Path,
    rank: int,
    start: int,
    end: int,
    centroid: np.ndarray,
    rotation: np.ndarray,
    chunk_rows: int = 200_000,
    devices: list[str] | None = None,
) -> None:
    """Quantize rows [start, end) into this rank's shard files.

    Chunks are dealt round-robin to the node's GPUs (CUDA ops and the pread
    chunk loads release the GIL); each result lands at its offset in
    pre-sized memmaps. The marker
    json is written only after the three files are flushed, so a rank is
    resumable: an existing marker for the same range skips the work.
    """
    codes_p, norms_p, dots_p, marker_p = _shard_files(rabitq_dir, rank)
    if marker_p.exists():
        marker = json.loads(marker_p.read_text())
        if marker["start"] == start and marker["end"] == end:
            print(
                f"[rank {rank}] RaBitQ shard rows {start:,}-{end:,} exists, skipping."
            )
            return
        raise ValueError(
            f"{marker_p} covers rows {marker['start']}-{marker['end']}, "
            f"not {start}-{end}: shard count changed; remove the rabitq dir"
        )

    n_rows = end - start
    _, d = mmap.shape
    bytes_per_vec = d // 8
    sqrt_d = math.sqrt(d)
    devices = devices or _devices()
    centroid_dev = {dev: torch.from_numpy(centroid).to(dev) for dev in devices}
    rotation_dev = {dev: torch.from_numpy(rotation).to(dev) for dev in devices}

    codes_mm = np.memmap(
        codes_p, dtype=np.uint8, mode="w+", shape=(n_rows, bytes_per_vec)
    )
    norms_mm = np.memmap(norms_p, dtype=np.float32, mode="w+", shape=(n_rows,))
    dots_mm = np.memmap(dots_p, dtype=np.float32, mode="w+", shape=(n_rows,))

    rows_of = _FbinRows(mmap, chunk_rows)

    @torch.no_grad()
    def _quantize_chunk(cs: int, dev: str):
        ce = min(cs + chunk_rows, end)
        chunk = torch.from_numpy(rows_of(cs, ce)).to(dev)
        xr = (chunk - centroid_dev[dev]) @ rotation_dev[dev]
        norms = torch.linalg.norm(xr, dim=1).float().cpu().numpy()
        signs = torch.where(xr >= 0, 1.0, -1.0).to(xr.dtype)
        dots = ((xr * signs).sum(dim=1) / sqrt_d).float().cpu().numpy()
        packed = np.packbits(
            (signs > 0).to(torch.uint8).cpu().numpy(), axis=-1, bitorder="little"
        )
        return cs - start, ce - start, packed, norms, dots

    starts = list(range(start, end, chunk_rows))
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [
            pool.submit(_quantize_chunk, cs, devices[i % len(devices)])
            for i, cs in enumerate(starts)
        ]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"[rank {rank}] Quantizing {n_rows:,} rows",
            mininterval=5,
        ):
            ls, le, packed, norms, dots = fut.result()
            codes_mm[ls:le] = packed
            norms_mm[ls:le] = norms
            dots_mm[ls:le] = dots

    for mm in (codes_mm, norms_mm, dots_mm):
        mm.flush()
    del codes_mm, norms_mm, dots_mm
    _atomic_write_json(
        marker_p, {"rank": rank, "start": start, "end": end, "rows": n_rows}
    )


def build_rabitq_index(
    fbin_path: Path,
    rabitq_dir: Path,
    rank: int = 0,
    num_ranks: int = 1,
    chunk_rows: int = 200_000,
    seed: int = 0,
    centroid_sample_rows: int = DEFAULT_CENTROID_SAMPLE_ROWS,
    devices: list[str] | None = None,
    wait_timeout: float = 8 * 3600,
    poll: float = 15.0,
) -> None:
    """Build (or resume) the index; every rank returns once meta.json exists.

    Rank 0 writes the transforms, quantizes its shard, waits for every other
    rank's marker and writes meta.json. Other ranks wait for the transforms,
    quantize their shard and wait for meta.json, so on return the index is
    complete and loadable on all ranks. Single node: rank 0 of 1 does it all.
    """
    rabitq_dir.mkdir(parents=True, exist_ok=True)
    meta_p = rabitq_dir / "meta.json"
    if meta_p.exists():
        return
    mmap = _load_fbin_mmap(fbin_path)
    n, d = mmap.shape
    if d % 8 != 0:
        raise ValueError(
            f"Embedding dim={d} must be a multiple of 8 for RaBitQ packing."
        )
    bounds = shard_bounds(n, num_ranks)
    centroid_p, rotation_p = _transform_paths(rabitq_dir)

    if rank == 0:
        if not (centroid_p.exists() and rotation_p.exists()):
            centroid = estimate_centroid(mmap, centroid_sample_rows, seed)
            _atomic_save_npy(rotation_p, make_rotation(d, seed))
            _atomic_save_npy(centroid_p, centroid)  # last: its presence unblocks others
    else:
        print(f"[rank {rank}] Waiting for RaBitQ transforms from rank 0...")
    _wait_for([centroid_p, rotation_p], wait_timeout, poll)
    centroid = np.load(centroid_p)
    rotation = np.load(rotation_p)

    quantize_rows(
        mmap,
        rabitq_dir,
        rank,
        bounds[rank],
        bounds[rank + 1],
        centroid,
        rotation,
        chunk_rows=chunk_rows,
        devices=devices,
    )

    if rank != 0:
        print(f"[rank {rank}] Shard done. Waiting for meta.json...")
        _wait_for([meta_p], wait_timeout, poll)
        return

    markers = [_shard_files(rabitq_dir, r)[3] for r in range(num_ranks)]
    if num_ranks > 1:
        print(f"[rank 0] Waiting for {num_ranks - 1} other RaBitQ shard(s)...")
    _wait_for(markers, wait_timeout, poll)
    shards = [json.loads(m.read_text()) for m in markers]
    for r, s in enumerate(shards):
        if (s["start"], s["end"]) != (bounds[r], bounds[r + 1]):
            raise ValueError(
                f"shard {r} bounds {s} do not match {bounds[r]}-{bounds[r + 1]}"
            )
    _atomic_write_json(
        meta_p,
        {
            "n": n,
            "d": d,
            "bytes_per_vec": d // 8,
            "seed": seed,
            "centroid_sample_rows": min(centroid_sample_rows, n),
            "shards": [
                {"rank": s["rank"], "start": s["start"], "end": s["end"]}
                for s in shards
            ],
        },
    )
    print(f"RaBitQ index built: {n:,} vectors in {num_ranks} shard(s) -> {rabitq_dir}")


def unpack_codes(packed: torch.Tensor, d: int) -> torch.Tensor:
    """(m, d/8) uint8 little-bit-order sign bits -> (m, d) float16 in {-1, +1}."""
    bits = torch.arange(8, dtype=torch.uint8, device=packed.device)
    unpacked = (packed.unsqueeze(-1) >> bits) & 1  # (m, d/8, 8)
    return unpacked.reshape(packed.shape[0], d).to(torch.float16) * 2 - 1


class RaBitQIndex:
    """Packed 1-bit codes resident on one or more devices, searched by row range.

    ``open`` reads only the metadata and transforms; ``load_rows`` brings a
    global row range onto the devices (split evenly across them), which is
    what lets one node of a multi-node search hold only its share.
    """

    def __init__(
        self, rabitq_dir: Path, meta: dict, centroid: np.ndarray, rotation: np.ndarray
    ):
        self.rabitq_dir = rabitq_dir
        self.n: int = meta["n"]
        self.d: int = meta["d"]
        self.bytes_per_vec: int = meta["bytes_per_vec"]
        self.sqrt_d = math.sqrt(self.d)
        self._centroid_np = centroid
        self._rotation_np = rotation
        self.centroid = torch.from_numpy(centroid)
        self.rotation = torch.from_numpy(rotation)
        # Disk shards as (start, end, codes_path, norms_path, dots_path)
        if "shards" in meta:
            self.disk_shards = [
                (s["start"], s["end"], *_shard_files(rabitq_dir, s["rank"])[:3])
                for s in sorted(meta["shards"], key=lambda s: s["start"])
            ]
        else:  # legacy single-file layout
            self.disk_shards = [
                (
                    0,
                    self.n,
                    rabitq_dir / "codes.u8",
                    rabitq_dir / "norms.f32",
                    rabitq_dir / "dots.f32",
                )
            ]
        self.shards: list[dict] = []
        self.loaded_range: tuple[int, int] | None = None

    @classmethod
    def open(cls, rabitq_dir: Path) -> "RaBitQIndex":
        with open(rabitq_dir / "meta.json") as f:
            meta = json.load(f)
        centroid_p, rotation_p = _transform_paths(rabitq_dir)
        return cls(rabitq_dir, meta, np.load(centroid_p), np.load(rotation_p))

    @classmethod
    def load(cls, rabitq_dir: Path, devices: list[str] | None = None) -> "RaBitQIndex":
        """Open and bring every row onto the devices (single-node use)."""
        index = cls.open(rabitq_dir)
        index.load_rows(0, index.n, devices)
        return index

    def read_rows(
        self, start: int, end: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Codes, norms and dots for global rows [start, end) from the shard files."""
        codes, norms, dots = [], [], []
        for s_start, s_end, codes_p, norms_p, dots_p in self.disk_shards:
            lo, hi = max(start, s_start), min(end, s_end)
            if lo >= hi:
                continue
            # Headerless raw arrays; pread the slice (see fbin.pread_into).
            first, n_rows = lo - s_start, hi - lo
            c = np.empty((n_rows, self.bytes_per_vec), np.uint8)
            nm = np.empty(n_rows, np.float32)
            dm = np.empty(n_rows, np.float32)
            for path, out in ((codes_p, c), (norms_p, nm), (dots_p, dm)):
                fd = os.open(path, os.O_RDONLY)
                try:
                    pread_into(
                        fd,
                        first * out.itemsize * (out.shape[1] if out.ndim > 1 else 1),
                        out,
                    )
                finally:
                    os.close(fd)
            codes.append(c)
            norms.append(nm)
            dots.append(dm)
        if not codes:
            e = np.empty((0, self.bytes_per_vec), dtype=np.uint8)
            return e, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
        return np.concatenate(codes), np.concatenate(norms), np.concatenate(dots)

    def load_rows(self, start: int, end: int, devices: list[str] | None = None) -> None:
        """Make global rows [start, end) resident, split evenly across devices."""
        if not (0 <= start <= end <= self.n):
            raise ValueError(f"row range {start}-{end} outside [0, {self.n}]")
        if self.loaded_range == (start, end):
            return
        devices = devices or _devices()
        primary = devices[0]
        self.centroid = torch.from_numpy(self._centroid_np).to(primary)
        self.rotation = torch.from_numpy(self._rotation_np).to(primary)
        bounds = shard_bounds(end - start, len(devices))
        self.shards = []
        for dev, lo, hi in zip(devices, bounds[:-1], bounds[1:]):
            codes, norms, dots = self.read_rows(start + lo, start + hi)
            self.shards.append(
                {
                    "device": dev,
                    "offset": start + lo,
                    "codes": torch.from_numpy(codes).to(dev),
                    "norms": torch.from_numpy(norms).to(dev),
                    "dots": torch.from_numpy(dots).to(dev),
                }
            )
        self.loaded_range = (start, end)
        print(
            f"Loaded RaBitQ rows {start:,}-{end:,} ({end - start:,} vectors) "
            f"across {len(devices)} device(s)"
        )

    @torch.no_grad()
    def search(
        self, queries: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k estimated inner products over the loaded rows.

        Returns (scores, indices), each (n_q, k'), k' = min(k, loaded rows),
        with indices as GLOBAL row ids. Codes are unpacked per sub-batch on
        the device, sized to the free memory, so the resident footprint stays
        at the packed size.
        """
        if self.loaded_range is None:
            raise RuntimeError("call load_rows() before search()")
        queries = queries.float()
        if queries.dim() == 1:
            queries = queries.unsqueeze(0)
        n_q = queries.shape[0]
        primary = self.centroid.device
        q = queries.to(primary)
        q_rot = (q - self.centroid) @ self.rotation
        q_dot_c = q @ self.centroid

        per_scores, per_ids = [], []
        for shard in self.shards:
            dev = shard["device"]
            m = shard["codes"].shape[0]
            if m == 0:
                continue
            local_k = min(k, m)
            qd = q_rot.to(dev).to(torch.float16)
            qdot = q_dot_c.to(dev)
            scale = shard["norms"] / (self.sqrt_d * shard["dots"])
            if dev != "cpu":
                free_mem, _ = torch.cuda.mem_get_info(torch.device(dev))
                # unpacked fp16 codes (2 B) + uint8 intermediate (1 B) per dim,
                # plus fp32 estimates and fp16 raw scores per query
                sub = max(1, int(free_mem * 0.4 / (self.d * 3 + n_q * 6)))
            else:
                sub = 65_536
            best_s = torch.full((n_q, local_k), float("-inf"), device=dev)
            best_i = torch.zeros((n_q, local_k), dtype=torch.long, device=dev)
            for s in range(0, m, sub):
                e = min(s + sub, m)
                codes = unpack_codes(shard["codes"][s:e], self.d)
                est = (qd @ codes.T).float() * scale[s:e] + qdot.unsqueeze(1)
                del codes
                kk = min(local_k, e - s)
                vals, idx = est.topk(kk, dim=1)
                del est
                cand_s = torch.cat([best_s, vals], dim=1)
                cand_i = torch.cat([best_i, idx + s], dim=1)
                best_s, sel = cand_s.topk(local_k, dim=1)
                best_i = cand_i.gather(1, sel)
            per_scores.append(best_s.cpu())
            per_ids.append((best_i + shard["offset"]).cpu())

        if not per_scores:
            return torch.empty((n_q, 0)), torch.empty((n_q, 0), dtype=torch.long)
        all_s = torch.cat(per_scores, dim=1)
        all_i = torch.cat(per_ids, dim=1)
        kk = min(k, all_s.shape[1])
        final_s, sel = all_s.topk(kk, dim=1)
        return final_s, all_i.gather(1, sel)

    def size_gb(self) -> float:
        return sum(
            p.stat().st_size for p in self.rabitq_dir.iterdir() if p.is_file()
        ) / (1024**3)
