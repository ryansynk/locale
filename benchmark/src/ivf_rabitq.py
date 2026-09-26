"""IVF + 1-bit RaBitQ (faiss) over an fbin, with an exact fp32 rerank.

The 1-bit brute-force scan (rabitq.py) touches every code for every query,
which is what keeps it multi-node on sra4571 (2.06 B vectors, 200 GB of
codes). This index partitions the vectors into ``nlist`` spherical k-means
cells, stores each vector as a faiss IVF-RaBitQ residual code in its cell and
scans only the ``nprobe`` cells nearest a query. The 1-bit estimate is a good
candidate generator but a noisy ranker (score error std ~0.017, see memory
scaling-roadmap), so the ``rerank`` best candidates of every query chunk are
re-scored with their exact fp32 vectors, read from the fbin by pread.

Pieces, in build order:

    sample_rows          seeded random rows (small blocks) for training
    spherical_kmeans     multi-GPU torch k-means under inner product
    assign_rows          nearest centroid of every row (multi-GPU)
    build_ivf_rabitq     train -> per-rank shard indexes -> merged index

Layout of ``ivf_dir``::

    train_sample.npy     the k-means training rows (kept for re-training)
    centroids_{nlist}.npy
    shard_{r}_of_{R}.faiss   one IVF-RaBitQ index per build rank, global ids
    ivf{nlist}_rabitq{b}.faiss   the merged index; its .json marker is
                                 written last and means "complete"
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .fbin import FBIN_HEADER_BYTES, _load_fbin_mmap, read_fbin_rows

TRAIN_BLOCK_ROWS = 8


def _devices() -> list[str]:
    n = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(n)] if n else ["cpu"]


def sample_rows(
    fbin_path: Path,
    n_rows: int,
    seed: int = 0,
    block_rows: int = TRAIN_BLOCK_ROWS,
    threads: int = 64,
) -> np.ndarray:
    """``n_rows`` fbin rows drawn as seeded random blocks of ``block_rows``.

    Consecutive rows are overlapping windows of one contig and nearly
    identical, so big blocks would give k-means many near-duplicates; 8-row
    blocks keep ~n_rows/8 independent draws while still reading 24 KB per
    pread. Blocks are read by a thread pool (pread releases the GIL).
    """
    mmap = _load_fbin_mmap(fbin_path)
    n, d = mmap.shape
    n_blocks = n // block_rows
    want = min(n_blocks, -(-n_rows // block_rows))
    rng = np.random.default_rng(seed)
    blocks = np.sort(rng.choice(n_blocks, size=want, replace=False))
    out = np.empty((want * block_rows, d), dtype=np.float32)
    local = threading.local()

    def _read(i_b):
        i, b = i_b
        if not hasattr(local, "fd"):
            local.fd = os.open(fbin_path, os.O_RDONLY)
        read_fbin_rows(
            local.fd,
            int(b) * block_rows,
            int(b + 1) * block_rows,
            out[i * block_rows : (i + 1) * block_rows],
        )

    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(
            tqdm(
                pool.map(_read, enumerate(blocks), chunksize=256),
                total=len(blocks),
                desc=f"Sampling {len(out):,} training rows",
                mininterval=10,
            )
        )
    return out[:n_rows]


def read_rows_by_id(
    fbin_path: Path, ids: np.ndarray, threads: int = 64
) -> np.ndarray:
    """fp32 rows ``ids`` (any order, repeats allowed) by one pread per row.

    Reads in sorted order, so rows of one Lustre stripe tend to go out
    together; returns them in the order of ``ids``.
    """
    ids = np.asarray(ids, dtype=np.int64)
    with open(fbin_path, "rb") as f:
        n, d = np.frombuffer(f.read(8), dtype=np.uint32)
    d = int(d)
    uniq, inv = np.unique(ids, return_inverse=True)
    out = np.empty((len(uniq), d), dtype=np.float32)
    row_bytes = d * 4
    fd = os.open(fbin_path, os.O_RDONLY)
    try:

        def _read(span):
            s, e = span
            for j in range(s, e):
                buf = os.pread(fd, row_bytes, FBIN_HEADER_BYTES + int(uniq[j]) * row_bytes)
                out[j] = np.frombuffer(buf, dtype=np.float32)

        step = max(1, -(-len(uniq) // (threads * 8)))
        spans = [(s, min(s + step, len(uniq))) for s in range(0, len(uniq), step)]
        with ThreadPoolExecutor(max_workers=threads) as pool:
            list(pool.map(_read, spans))
    finally:
        os.close(fd)
    return out[inv]


@torch.no_grad()
def _nearest(
    x_dev: torch.Tensor, cent_t: torch.Tensor, tile: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """argmax_c <x, c> and its value for every row of x_dev (tiled)."""
    idx = torch.empty(len(x_dev), dtype=torch.int64, device=x_dev.device)
    val = torch.empty(len(x_dev), dtype=torch.float32, device=x_dev.device)
    for s in range(0, len(x_dev), tile):
        v, i = (x_dev[s : s + tile] @ cent_t).max(dim=1)
        idx[s : s + tile] = i
        val[s : s + tile] = v
    return idx, val


@torch.no_grad()
def spherical_kmeans(
    x: np.ndarray,
    k: int,
    n_iter: int = 20,
    seed: int = 0,
    devices: list[str] | None = None,
    tile: int = 8192,
    verbose: bool = True,
) -> np.ndarray:
    """k unit-norm centroids maximizing sum_i max_c <x_i, c> (k-means under IP).

    The data is split across the GPUs and stays resident; each iteration
    every device assigns its rows (TF32 matmul, tiled) and returns per-cell
    sums, the centroids are the normalized sums. An empty cell is reseeded
    with the worst-fit row of a random device, so no cell stays dead.
    """
    devices = devices or _devices()
    torch.backends.cuda.matmul.allow_tf32 = True
    n, d = x.shape
    rng = np.random.default_rng(seed)
    cent = x[rng.choice(n, size=k, replace=False)].copy()
    parts = np.array_split(np.arange(n), len(devices))
    xs = [torch.from_numpy(x[p[0] : p[-1] + 1]).to(dev) for p, dev in zip(parts, devices)]

    def _step(i, cent_np):
        dev = devices[i]
        c = torch.from_numpy(cent_np).to(dev).T.contiguous()
        idx, val = _nearest(xs[i], c, tile)
        sums = torch.zeros((k, d), dtype=torch.float32, device=dev)
        sums.index_add_(0, idx, xs[i])
        counts = torch.bincount(idx, minlength=k)
        worst = torch.topk(val, min(k, len(val)), largest=False).indices
        return sums.cpu(), counts.cpu(), val.sum().item(), xs[i][worst].cpu()

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        for it in range(n_iter):
            t0 = time.time()
            res = list(pool.map(lambda i: _step(i, cent), range(len(devices))))
            sums = sum(r[0] for r in res)
            counts = sum(r[1] for r in res)
            obj = sum(r[2] for r in res) / n
            new = torch.nn.functional.normalize(sums, dim=1).numpy()
            empty = np.flatnonzero(counts.numpy() == 0)
            if len(empty):
                pool_rows = torch.cat([r[3] for r in res]).numpy()
                new[empty] = pool_rows[rng.choice(len(pool_rows), len(empty), replace=False)]
            cent = np.ascontiguousarray(new, dtype=np.float32)
            if verbose:
                print(
                    f"  kmeans k={k} iter {it + 1}/{n_iter}: mean max-IP {obj:.4f}, "
                    f"{len(empty)} empty, {time.time() - t0:.1f}s",
                    flush=True,
                )
    del xs
    torch.cuda.empty_cache()
    return cent


@torch.no_grad()
def assign_rows(
    x: np.ndarray, centroids: np.ndarray, devices: list[str] | None = None, tile: int = 8192
) -> np.ndarray:
    """Nearest centroid (by IP) of every row of x, split across the GPUs."""
    devices = devices or _devices()
    torch.backends.cuda.matmul.allow_tf32 = True
    parts = np.array_split(np.arange(len(x)), len(devices))
    out = np.empty(len(x), dtype=np.int64)

    def _run(i):
        p = parts[i]
        if not len(p):
            return
        dev = devices[i]
        c = torch.from_numpy(centroids).to(dev).T.contiguous()
        xd = torch.from_numpy(x[p[0] : p[-1] + 1]).to(dev)
        out[p[0] : p[-1] + 1] = _nearest(xd, c, tile)[0].cpu().numpy()

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        list(pool.map(_run, range(len(devices))))
    return out


@torch.no_grad()
def probe_ranks(
    q: np.ndarray, centroids: np.ndarray, cells: np.ndarray, device: str = "cuda:0"
) -> np.ndarray:
    """Rank (0 = nearest) of cell cells[i, j] in query i's centroid ordering.

    q: (n_q, d); cells: (n_q, m) cell ids. The rank is the number of
    centroids scoring strictly higher than the given cell, i.e. the smallest
    nprobe - 1 at which a probe of query i visits that cell.
    """
    qt = torch.from_numpy(q).to(device)
    ct = torch.from_numpy(centroids).to(device)
    cl = torch.from_numpy(cells).to(device)
    out = torch.empty(cl.shape, dtype=torch.int64, device=device)
    # the (b, m, k) comparison is the big temporary: keep it near 2 GB
    step = max(1, (2 << 30) // (cl.shape[1] * len(ct)))
    for s in range(0, len(qt), step):
        sims = qt[s : s + step] @ ct.T  # (b, k)
        target = torch.gather(sims, 1, cl[s : s + step])  # (b, m)
        out[s : s + step] = (sims.unsqueeze(1) > target.unsqueeze(2)).sum(dim=2)
    return out.cpu().numpy()


# --------------------------------------------------------------------------
# Build: centroids (rank 0) -> one faiss shard per rank -> merged index
# --------------------------------------------------------------------------

DEFAULT_TRAIN_ROWS = 6_000_000
BUILD_BLOCK_ROWS = 1_000_000


def centroids_path(ivf_dir: Path, nlist: int) -> Path:
    return ivf_dir / f"centroids_{nlist}.npy"


def index_name(nlist: int, nb_bits: int) -> str:
    return f"ivf{nlist}_rabitq{nb_bits}"


def shard_path(ivf_dir: Path, nlist: int, nb_bits: int, rank: int, num_ranks: int) -> Path:
    return ivf_dir / f"{index_name(nlist, nb_bits)}_shard_{rank}_of_{num_ranks}.faiss"


def merged_path(ivf_dir: Path, nlist: int, nb_bits: int) -> Path:
    return ivf_dir / f"{index_name(nlist, nb_bits)}.faiss"


def _atomic_write_index(index, path: Path) -> None:
    import faiss

    tmp = path.with_suffix(".faiss.tmp")
    faiss.write_index(index, str(tmp))
    tmp.rename(path)


def _wait(paths: list[Path], timeout: float = 4 * 3600, poll: float = 15) -> None:
    deadline = time.time() + timeout
    while not all(p.exists() for p in paths):
        if time.time() > deadline:
            raise TimeoutError(f"timed out waiting for {[str(p) for p in paths if not p.exists()]}")
        time.sleep(poll)


def train_centroids(
    fbin_path: Path, ivf_dir: Path, nlist: int, train_rows: int = DEFAULT_TRAIN_ROWS, seed: int = 0
) -> np.ndarray:
    """Spherical k-means centroids for nlist cells, cached in ivf_dir.

    The training sample is cached too (train_sample.npy) so a second nlist
    reuses it; with fewer than 40 rows per cell k-means is undertrained, so
    that is refused.
    """
    cp = centroids_path(ivf_dir, nlist)
    if cp.exists():
        return np.load(cp)
    if train_rows < 40 * nlist:
        raise ValueError(f"train_rows={train_rows:,} is under 40 rows per cell for nlist={nlist}")
    ivf_dir.mkdir(parents=True, exist_ok=True)
    sp = ivf_dir / "train_sample.npy"
    if sp.exists() and len(np.load(sp, mmap_mode="r")) >= train_rows:
        train = np.ascontiguousarray(np.load(sp, mmap_mode="r")[:train_rows])
    else:
        train = sample_rows(fbin_path, train_rows, seed=seed)
        np.save(sp, train)
    cent = spherical_kmeans(train, nlist, seed=seed)
    tmp = cp.with_suffix(".npy.tmp")
    with open(tmp, "wb") as f:
        np.save(f, cent)
    tmp.rename(cp)
    return cent


def new_ivf_rabitq(centroids: np.ndarray, nb_bits: int = 1):
    """An empty, trained IndexIVFRaBitQ over the given centroids (IP metric)."""
    import faiss

    nlist, d = centroids.shape
    quantizer = faiss.IndexFlatIP(d)
    quantizer.add(centroids)
    index = faiss.IndexIVFRaBitQ(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT, True, nb_bits)
    # The quantizer is already populated, so train() only fits the RaBitQ
    # side, which for residual codes needs no data beyond the centroids --
    # training on the centroids themselves keeps every shard identical.
    index.train(centroids)
    index.referenced_objects = [quantizer]  # keep the quantizer alive
    return index


def build_ivf_shard(
    fbin_path: Path,
    ivf_dir: Path,
    nlist: int,
    rank: int = 0,
    num_ranks: int = 1,
    nb_bits: int = 1,
    train_rows: int = DEFAULT_TRAIN_ROWS,
    block_rows: int = BUILD_BLOCK_ROWS,
    row_range: tuple[int, int] | None = None,
) -> Path:
    """Encode this rank's contiguous row range into its own IVF-RaBitQ shard.

    Rank 0 trains the centroids (the others wait for the file). Blocks are
    read by a prefetch thread while the GPUs assign the previous block to its
    nearest centroid and faiss (OpenMP) encodes the residual codes, with the
    fbin row number as the vector id. Resumable: an existing shard file is
    kept. ``row_range`` overrides the rank's range (used for subset benches).
    """
    from faiss.contrib.ivf_tools import add_preassigned

    out = shard_path(ivf_dir, nlist, nb_bits, rank, num_ranks)
    if row_range is not None:
        out = out.with_name(out.stem + f"_rows{row_range[0]}-{row_range[1]}.faiss")
    if out.exists():
        print(f"[rank {rank}] {out.name} exists, skipping.")
        return out
    if rank == 0:
        cent = train_centroids(fbin_path, ivf_dir, nlist, train_rows)
    else:
        _wait([centroids_path(ivf_dir, nlist)])
        cent = np.load(centroids_path(ivf_dir, nlist))
    n = int(_load_fbin_mmap(fbin_path).shape[0])
    start, end = row_range or (n * rank // num_ranks, n * (rank + 1) // num_ranks)
    index = new_ivf_rabitq(cent, nb_bits)
    d = cent.shape[1]
    fd = os.open(fbin_path, os.O_RDONLY)
    bufs = [np.empty((block_rows, d), dtype=np.float32) for _ in range(2)]
    blocks = list(range(start, end, block_rows))

    def _read(i):
        bs = blocks[i]
        return read_fbin_rows(fd, bs, min(bs + block_rows, end), bufs[i % 2])

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=1) as reader:
        fut = reader.submit(_read, 0) if blocks else None
        for i, bs in enumerate(tqdm(blocks, desc=f"[rank {rank}] IVF encode {end - start:,} rows", mininterval=30)):
            x = fut.result()
            # The reader may refill the other buffer while this one is used.
            fut = reader.submit(_read, i + 1) if i + 1 < len(blocks) else None
            a = assign_rows(x, cent)
            # Bound to a name on purpose: add_preassigned rebinds its ids
            # argument to a raw swig pointer, so a temporary array would be
            # freed before add_core reads it (corrupt ids, seen 2026-09-24).
            ids = np.arange(bs, bs + len(x), dtype=np.int64)
            add_preassigned(index, x, a, ids)
            del ids
    os.close(fd)
    print(f"[rank {rank}] encoded {index.ntotal:,} rows in {time.time() - t0:.0f}s")
    _atomic_write_index(index, out)
    return out


def to_fastscan(index):
    """IndexIVFRaBitQ -> IndexIVFRaBitQFastScan that owns the coarse quantizer.

    The FastScan constructor shares the source's quantizer pointer; the source
    owns (and on destruction frees) it, so dropping the source -- the natural
    ``index = to_fastscan(index)`` -- left the FastScan index on freed memory
    (heap corruption in the first merge, 2026-09-24). Ownership moves to the
    new index instead -- but only if the source had it: an index built in
    memory by new_ivf_rabitq has its quantizer owned by a Python object, and
    taking C++ ownership of that one frees it twice. Either way the new index
    keeps the source's referenced Python objects alive.
    """
    import faiss

    fs = faiss.IndexIVFRaBitQFastScan(index)
    if index.own_fields:
        index.own_fields = False
        fs.own_fields = True
    fs.referenced_objects = list(getattr(index, "referenced_objects", None) or []) + [index.quantizer]
    return fs


def list_shards(ivf_dir: Path, nlist: int, nb_bits: int) -> list[Path]:
    """The build's shard files, in rank order (all from one build: same R)."""
    stem = index_name(nlist, nb_bits)
    shards = sorted(
        ivf_dir.glob(f"{stem}_shard_*_of_*.faiss"),
        key=lambda p: int(p.stem.split("_shard_")[1].split("_of_")[0]),
    )
    counts = {p.stem.rsplit("_of_", 1)[1] for p in shards}
    if len(counts) > 1:
        raise ValueError(f"shards of several build sizes in {ivf_dir}: {counts}")
    if shards and len(shards) != int(counts.pop()):
        raise FileNotFoundError(f"incomplete IVF build in {ivf_dir}: {len(shards)} shards")
    return shards


def merge_ivf_shards(ivf_dir: Path, nlist: int, nb_bits: int, num_ranks: int) -> Path:
    """Fold the per-rank shards into one plain IVF-RaBitQ file (needs RAM for
    the whole index). Only for the non-FastScan layout: merge_from on
    IndexIVFRaBitQFastScan corrupts the heap at sra4571 shard sizes (faiss
    1.15.1, 2026-09-24; fine on small indexes), so FastScan search keeps the
    shards separate instead (IVFRaBitQSearcher takes a list)."""
    import faiss

    out = merged_path(ivf_dir, nlist, nb_bits)
    if out.exists():
        return out
    paths = [shard_path(ivf_dir, nlist, nb_bits, r, num_ranks) for r in range(num_ranks)]
    t0 = time.time()
    index = faiss.read_index(str(paths[0]))
    for p in tqdm(paths[1:], desc="Merging IVF shards"):
        other = faiss.read_index(str(p))
        index.merge_from(other, 0)  # ids are already global fbin rows
        del other
    print(f"merged {index.ntotal:,} vectors in {time.time() - t0:.0f}s")
    _atomic_write_index(index, out)
    return out


# --------------------------------------------------------------------------
# Search: coarse probe + 1-bit scan (faiss) -> exact fp32 rerank (pread)
# --------------------------------------------------------------------------


class IVFRaBitQSearcher:
    """IVF-RaBitQ index part(s) held in RAM plus the fbin they rerank from.

    ``index_files`` is one merged file or the build's per-rank shards. Shards
    share the centroids, so each is searched with the same nprobe and the
    per-shard candidate lists are merged: the union of the shards' probed
    lists is exactly the merged index's probed lists, so the scanned codes --
    and, after the exact rerank, the result -- are those of one index; only
    the (cheap) coarse search and LUT setup repeat per shard.

    quantizer: "flat" keeps the exact coarse search (cheap for a batch: 2k
    queries x 16k-64k centroids is one small GEMM); "hnsw" swaps in one HNSW
    graph over the centroids, shared by the shards -- the per-query-latency
    option for large nlist. fastscan converts the codes to faiss' SIMD 4-bit
    LUT layout at load time.
    """

    def __init__(
        self,
        index_files: Path | list[Path],
        fbin_path: Path,
        quantizer: str = "flat",
        fastscan: bool = False,
        hnsw_m: int = 32,
        hnsw_ef: int = 0,
    ):
        import faiss

        if isinstance(index_files, Path):
            index_files = [index_files]
        if quantizer not in ("flat", "hnsw"):
            raise ValueError(f"unknown coarse quantizer {quantizer!r}")
        self.parts = []
        t0 = time.time()
        for f in index_files:
            index = faiss.read_index(str(f))
            if fastscan and not isinstance(index, faiss.IndexIVFRaBitQFastScan):
                index = to_fastscan(index)
            self.parts.append(index)
        print(
            f"Loaded {len(self.parts)} IVF part(s), {self.ntotal:,} vectors, "
            f"nlist {self.parts[0].nlist}, fastscan={fastscan} in {time.time() - t0:.0f}s"
        )
        self.quantizer_kind = quantizer
        if quantizer == "hnsw":
            flat = faiss.downcast_index(self.parts[0].quantizer)
            cent = flat.reconstruct_n(0, flat.ntotal)
            hnsw = faiss.IndexHNSWFlat(cent.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
            hnsw.hnsw.efConstruction = 200
            t0 = time.time()
            hnsw.add(cent)
            print(f"  HNSW coarse quantizer over {len(cent):,} centroids in {time.time() - t0:.1f}s")
            for part in self.parts:
                # the part still owns (and frees) its flat quantizer; keep it
                # referenced and give the part the shared graph, unowned
                part.referenced_objects = [part.quantizer, hnsw]
                part.own_fields = False
                part.quantizer = hnsw
            self._hnsw = hnsw
            self.hnsw_ef = hnsw_ef
        self.fbin_path = fbin_path
        self.fastscan = fastscan
        self.index_files = list(index_files)

    @property
    def ntotal(self) -> int:
        return int(sum(p.ntotal for p in self.parts))

    def _scan(self, q: np.ndarray, n_cand: int, nprobe: int, qb: int):
        """Best n_cand 1-bit candidates per query over all parts."""
        import faiss

        params = faiss.IVFRaBitQSearchParameters()
        params.nprobe = nprobe
        params.qb = qb
        if self.quantizer_kind == "hnsw":
            qp = faiss.SearchParametersHNSW()
            qp.efSearch = max(self.hnsw_ef, 2 * nprobe)
            params.quantizer_params = qp
        Ds, Is = [], []
        for part in self.parts:
            D, I = part.search(q, n_cand, params=params)
            Ds.append(D)
            Is.append(I)
        if len(self.parts) == 1:
            return Ds[0], Is[0]
        D = np.concatenate(Ds, axis=1)
        I = np.concatenate(Is, axis=1)
        D[I < 0] = -np.inf
        top = np.argpartition(-D, n_cand - 1, axis=1)[:, :n_cand]
        D = np.take_along_axis(D, top, axis=1)
        I = np.take_along_axis(I, top, axis=1)
        order = np.argsort(-D, axis=1, kind="stable")
        return np.take_along_axis(D, order, axis=1), np.take_along_axis(I, order, axis=1)

    def search(
        self,
        q: np.ndarray,
        top_k: int,
        nprobe: int,
        rerank: int,
        qb: int = 8,
        io_threads: int = 128,
        timings: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(scores, ids) of shape (n_q, top_k), exact IP, score-descending.

        The 1-bit scan keeps the best max(rerank, top_k) candidates per query,
        whose fp32 rows are then read from the fbin and scored exactly.
        rerank=0 skips the rerank and returns the 1-bit estimates.
        Missing slots are id -1 / score -inf.
        """
        q = np.ascontiguousarray(q, dtype=np.float32)
        n_cand = max(rerank, top_k)
        t0 = time.time()
        D, I = self._scan(q, n_cand, nprobe, qb)
        t1 = time.time()
        if timings is not None:
            timings["scan_s"] = timings.get("scan_s", 0.0) + t1 - t0
        if rerank <= 0:
            return D[:, :top_k], I[:, :top_k]
        valid = I >= 0
        ids = I[valid]
        vecs = read_rows_by_id(self.fbin_path, ids, threads=io_threads)
        qi = np.nonzero(valid)[0]
        exact = np.full(I.shape, -np.inf, dtype=np.float32)
        exact[valid] = np.einsum("ij,ij->i", vecs, q[qi])
        order = np.argsort(-exact, axis=1, kind="stable")[:, :top_k]
        scores = np.take_along_axis(exact, order, axis=1)
        out_ids = np.take_along_axis(I, order, axis=1)
        out_ids[~np.isfinite(scores)] = -1
        if timings is not None:
            timings["rerank_s"] = timings.get("rerank_s", 0.0) + time.time() - t1
            timings["rerank_rows"] = timings.get("rerank_rows", 0) + len(np.unique(ids))
        return scores, out_ids
