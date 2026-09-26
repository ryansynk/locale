"""GPU IVF-PQ (cuVS) over an fbin: many-GPU shard build, one-node GPU search.

The faiss CPU engine (ivf_rabitq.py) reaches exact-level accuracy on
sra4571 but needs ~130 s per 1000 queries on a CPU node, against ~3.8 s for
a warm metagraph server. This keeps the whole index in the HBM of one
4 x A100-80GB node and scans it with cuVS' batched IVF-PQ kernels:

* 2.06 B vectors x (128 B PQ code + 8 B id) = 280 GB, split into
  ``num_shards`` independent IVF-PQ indexes (row ranges of the fbin, each
  with its own k-means lists and PQ codebooks) dealt round-robin to the GPUs.
  At 96 B (pq_dim 96) PQ ranks clearly worse than 1-bit RaBitQ on this data;
  pq_dim 128 x 8 bits matches it (2026-09-24 bench, logs_ivf/bench_codes.py).
* One worker process per GPU holds its shards (and a query-encoder replica);
  every shard is searched with the same ``n_probes`` (same scanned fraction),
  each worker keeps the best candidates over its shards, the parent merges.
  Processes, not threads: a cuVS search call blocks its host thread.
* An optional exact rerank re-scores each query chunk's best ``rerank``
  candidates from the fp32 fbin (pread). Lustre gives ~15-60 K random rows/s,
  so only a shallow rerank fits a metagraph-parity budget.

Build: one process per GPU (``SLURM_PROCID`` = shard, ``SLURM_NTASKS`` =
shards), each trains on a sample of its own rows and extends in blocks read by
a prefetch thread. Shards are small enough (~17.5 GB at 16 shards) to build
on 40 GB cards, but load as ~20.6 GB each (cuVS list layout), so four per
80 GB card leave ~2 GB free. Files::

    <index>/ivfpq/pq{dim}x{bits}_L{lists}/shard_{i}_of_{S}.cuvs  (+ .json)
"""

import atexit
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .fbin import _load_fbin_mmap, read_fbin_rows
from .ivf_rabitq import read_rows_by_id

TRAIN_ROWS_PER_LIST = 256
# 16384 lists x 256 rows (4.2 M x 768 fp32, 12.6 GB) trains fine on a 40 GB
# card; finer shards get fewer rows per list instead of more memory.
MAX_TRAIN_ROWS = 4_194_304
BUILD_BLOCK_ROWS = 2_000_000


def ivfpq_dir(index_path: Path, pq_dim: int, pq_bits: int, lists_per_shard: int) -> Path:
    return index_path / "ivfpq" / f"pq{pq_dim}x{pq_bits}_L{lists_per_shard}"


def shard_file(d: Path, shard: int, num_shards: int) -> Path:
    return d / f"shard_{shard}_of_{num_shards}.cuvs"


def list_shard_files(d: Path, num_shards: int) -> list[Path]:
    files = [shard_file(d, s, num_shards) for s in range(num_shards)]
    missing = [f for f in files if not f.with_suffix(".json").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {num_shards} IVF-PQ shards missing in {d}")
    return files


def _sample_range(fd: int, start: int, end: int, n_rows: int, d: int, seed: int, block: int = 8) -> np.ndarray:
    """n_rows rows of [start, end) as seeded random 8-row blocks (threaded preads)."""
    n_blocks = (end - start) // block
    want = min(n_blocks, -(-n_rows // block))
    rng = np.random.default_rng(seed)
    blocks = np.sort(rng.choice(n_blocks, size=want, replace=False))
    out = np.empty((want * block, d), dtype=np.float32)

    def _read(i):
        b = start + int(blocks[i]) * block
        read_fbin_rows(fd, b, b + block, out[i * block : (i + 1) * block])

    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(_read, range(want), chunksize=512))
    return out


def build_ivfpq_shard(
    fbin_path: Path,
    out_dir: Path,
    shard: int,
    num_shards: int,
    pq_dim: int = 128,
    pq_bits: int = 8,
    lists_per_shard: int = 4096,
    block_rows: int = BUILD_BLOCK_ROWS,
    device: str = "cuda:0",
    seed: int = 0,
) -> Path:
    """Train and fill one shard's IVF-PQ index over its contiguous row range."""
    from cuvs.neighbors import ivf_pq

    out = shard_file(out_dir, shard, num_shards)
    marker = out.with_suffix(".json")
    if marker.exists():
        print(f"[shard {shard}] {out.name} exists, skipping.")
        return out
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(device)
    n, d = _load_fbin_mmap(fbin_path).shape
    start, end = n * shard // num_shards, n * (shard + 1) // num_shards
    fd = os.open(fbin_path, os.O_RDONLY)
    t0 = time.time()
    n_train = min(TRAIN_ROWS_PER_LIST * lists_per_shard, MAX_TRAIN_ROWS)
    train = _sample_range(fd, start, end, n_train, d, seed + shard)
    params = ivf_pq.IndexParams(
        n_lists=lists_per_shard,
        metric="inner_product",
        pq_dim=pq_dim,
        pq_bits=pq_bits,
        add_data_on_build=False,
        kmeans_trainset_fraction=1.0,
        kmeans_n_iters=20,
        # lists are filled by many extend() calls; exact-size lists would
        # re-copy on every call, but the default over-allocation wastes HBM
        # that the 4-shards-per-GPU search layout needs.
        conservative_memory_allocation=True,
    )
    index = ivf_pq.build(params, torch.from_numpy(train).to(device))
    del train
    print(f"[shard {shard}] trained {lists_per_shard} lists in {time.time() - t0:.0f}s", flush=True)

    bufs = [np.empty((block_rows, d), dtype=np.float32) for _ in range(2)]
    blocks = list(range(start, end, block_rows))

    def _read(i):
        bs = blocks[i]
        return read_fbin_rows(fd, bs, min(bs + block_rows, end), bufs[i % 2])

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=1) as reader:
        fut = reader.submit(_read, 0)
        for i, bs in enumerate(tqdm(blocks, desc=f"[shard {shard}] extend {end - start:,} rows", mininterval=30)):
            x = torch.from_numpy(fut.result()).to(device)
            fut = reader.submit(_read, i + 1) if i + 1 < len(blocks) else None
            ids = torch.arange(bs, bs + len(x), dtype=torch.int64, device=device)
            index = ivf_pq.extend(index, x, ids)
            del x, ids
    os.close(fd)
    torch.cuda.synchronize()
    print(f"[shard {shard}] extended {end - start:,} rows in {time.time() - t0:.0f}s", flush=True)
    tmp = out.with_suffix(".cuvs.tmp")
    ivf_pq.save(str(tmp), index, include_dataset=True)
    tmp.rename(out)
    marker.write_text(json.dumps({"shard": shard, "num_shards": num_shards, "start": start, "end": end,
                                  "pq_dim": pq_dim, "pq_bits": pq_bits, "n_lists": lists_per_shard}))
    return out


def _gpu_worker(conn, shard_files: list[str], encoder_cfg) -> None:
    """One process per GPU (CUDA_VISIBLE_DEVICES set by the parent): holds
    this GPU's shards and, optionally, a query-encoder replica, and serves
    ("embed", chunks) / ("search", q, k, n_probes, lut) / ("close",) requests.

    Processes rather than threads: a cuVS ivf_pq.search call blocks the
    calling host thread for its whole duration, so per-GPU threads ran the
    GPUs one after another (4.3 s instead of ~1.1 s for 16 shards,
    2026-09-24)."""
    import time

    import numpy as np
    import torch
    from cuvs.neighbors import ivf_pq

    try:
        torch.cuda.set_device(0)
        indexes = [ivf_pq.load(f) for f in shard_files]
        encoder = None
        if encoder_cfg is not None:
            from .encoders import DenseEncoder

            encoder_cfg.device = "cuda:0"
            encoder = DenseEncoder(encoder_cfg)
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info(0)
        conn.send(("ready", len(indexes), int(indexes[0].n_lists) if indexes else 0, (total - free) / 2**30))
    except Exception as e:  # report instead of dying silently
        conn.send(("error", repr(e)))
        return
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):  # parent went away
            return
        try:
            if msg[0] == "close":
                return
            if msg[0] == "embed":
                with torch.no_grad():
                    f = encoder.encode(msg[1]).float().cpu().numpy()
                # hand the activations' cache back before the next search
                torch.cuda.empty_cache()
                conn.send(("ok", f))
            elif msg[0] == "search":
                _, q, k, n_probes, lut, batch, dist = msg
                # Four 128x8 shards leave ~2 GB of an 80 GB card free (cuVS
                # lays a loaded 17.5 GB shard out in ~20.6 GB); batches of
                # 4096 query rows still fit and were fastest (2026-09-24:
                # 0.62 s at 17 probes vs 0.66 at 1024, 1.22 at 128).
                sp = ivf_pq.SearchParams(
                    n_probes=n_probes,
                    lut_dtype=np.float16 if lut == "float16" else np.float32,
                    internal_distance_dtype=np.float16 if dist == "float16" else np.float32,
                    coarse_search_dtype=np.float16,
                    max_internal_batch_size=batch,
                )
                tw = time.time()
                qd = torch.from_numpy(q).cuda()
                Ds, Is = [], []
                for index in indexes:
                    D = torch.empty((len(qd), k), dtype=torch.float32, device="cuda")
                    I = torch.empty((len(qd), k), dtype=torch.int64, device="cuda")
                    ivf_pq.search(sp, index, qd, k, neighbors=I, distances=D)
                    Ds.append(D)
                    Is.append(I)
                D = torch.cat(Ds, dim=1)
                I = torch.cat(Is, dim=1)
                D = D.masked_fill(I < 0, float("-inf"))  # cuVS pads with id -1
                D, pos = torch.topk(D, min(k, D.shape[1]), dim=1)
                I = torch.gather(I, 1, pos)
                D, I = D.cpu().numpy(), I.cpu().numpy()
                conn.send(("ok", D, I, time.time() - tw))
            else:
                conn.send(("error", f"unknown request {msg[0]!r}"))
        except Exception as e:
            conn.send(("error", repr(e)))


class IVFPQGPUSearcher:
    """All shards resident on this node's GPUs, one worker process per GPU.

    Shard i lives on GPU i % n_gpus. With ``encoder_cfg`` every worker also
    holds a query-encoder replica and ``embed`` splits a chunk list across
    them, so the parent process never needs CUDA (a CUDA context there
    would take memory the full cards do not have).
    """

    def __init__(
        self,
        shard_files: list[Path],
        fbin_path: Path,
        n_gpus: int | None = None,
        encoder_cfg=None,
    ):
        import multiprocessing as mp

        n_gpus = n_gpus or torch.cuda.device_count()
        n_gpus = min(n_gpus, len(shard_files))
        self.fbin_path = fbin_path
        self.shard_files = list(shard_files)
        self.n_gpus = n_gpus
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        t0 = time.time()
        old = os.environ.get("CUDA_VISIBLE_DEVICES")
        visible = old.split(",") if old else [str(i) for i in range(n_gpus)]
        try:
            for g in range(n_gpus):
                os.environ["CUDA_VISIBLE_DEVICES"] = visible[g]
                parent, child = ctx.Pipe()
                files = [str(f) for f in self.shard_files[g::n_gpus]]
                p = ctx.Process(target=_gpu_worker, args=(child, files, encoder_cfg), daemon=True)
                p.start()
                self.conns.append(parent)
                self.procs.append(p)
        finally:
            if old is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old
        for g, c in enumerate(self.conns):
            msg = c.recv()
            if msg[0] != "ready":
                raise RuntimeError(f"IVF-PQ worker for GPU {visible[g]} failed: {msg[1]}")
            _, n_shards, n_lists, used = msg
            self.lists_per_shard = n_lists
            print(f"  GPU {visible[g]}: {n_shards} shards, {used:.1f} GiB used")
        self.has_encoder = encoder_cfg is not None
        atexit.register(self.close)
        print(
            f"Loaded {len(self.shard_files)} IVF-PQ shards onto {n_gpus} GPU worker(s) "
            f"({self.lists_per_shard} lists each) in {time.time() - t0:.0f}s"
        )

    def _gather(self):
        out = []
        for c in self.conns:
            msg = c.recv()
            if msg[0] != "ok":
                raise RuntimeError(f"IVF-PQ worker error: {msg[1]}")
            out.append(msg[1:])
        return out

    def embed(self, chunks: list[str]) -> np.ndarray:
        """Query-chunk embeddings, split contiguously across the GPU workers."""
        assert self.has_encoder, "searcher was built without encoder_cfg"
        n = self.n_gpus
        bounds = [len(chunks) * i // n for i in range(n + 1)]
        for i, c in enumerate(self.conns):
            c.send(("embed", chunks[bounds[i] : bounds[i + 1]]))
        return np.concatenate([o[0] for o in self._gather()], axis=0)

    def close(self) -> None:
        if not self.procs:
            return
        for c in self.conns:
            try:
                c.send(("close",))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=30)
        self.procs = []

    def search(
        self,
        q: np.ndarray,
        top_k: int,
        n_probes: int,
        rerank: int = 0,
        lut_dtype: str = "float16",
        batch: int = 4096,
        internal_dtype: str = "float32",
        io_threads: int = 128,
        timings: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(scores, ids) of shape (n_q, top_k), score-descending; id -1 = empty.

        Scores are PQ estimates, except for each row's best ``rerank``
        candidates, which are re-scored exactly from the fbin before the list
        is re-sorted. Both are inner-product scales; the PQ estimate is biased
        low (~-0.05 on sra4571), so a reranked hit is never pushed below an
        unreranked one it truly beats.
        """
        if not isinstance(lut_dtype, str):
            lut_dtype = np.dtype(lut_dtype).name
        q = np.ascontiguousarray(q, dtype=np.float32)
        k = max(top_k, rerank)
        t0 = time.time()
        for c in self.conns:
            c.send(("search", q, k, n_probes, lut_dtype, batch, internal_dtype))
        parts = self._gather()
        if timings is not None:
            timings["gpu_s"] = [round(p[2], 3) for p in parts]
        D = np.concatenate([p[0] for p in parts], axis=1)
        I = np.concatenate([p[1] for p in parts], axis=1)
        top = np.argpartition(-D, k - 1, axis=1)[:, :k]
        D = np.take_along_axis(D, top, axis=1)
        I = np.take_along_axis(I, top, axis=1)
        order = np.argsort(-D, axis=1, kind="stable")
        D = np.take_along_axis(D, order, axis=1)
        I = np.take_along_axis(I, order, axis=1)
        t1 = time.time()
        if timings is not None:
            timings["scan_s"] = timings.get("scan_s", 0.0) + t1 - t0
        if rerank > 0:
            head = I[:, :rerank]
            valid = head >= 0
            vecs = read_rows_by_id(self.fbin_path, head[valid], threads=io_threads)
            exact = np.full(head.shape, -np.inf, dtype=np.float32)
            exact[valid] = np.einsum("ij,ij->i", vecs, q[np.nonzero(valid)[0]])
            D = D.copy()
            D[:, :rerank] = exact
            order = np.argsort(-D, axis=1, kind="stable")
            D = np.take_along_axis(D, order, axis=1)
            I = np.take_along_axis(I, order, axis=1)
            if timings is not None:
                timings["rerank_s"] = timings.get("rerank_s", 0.0) + time.time() - t1
                timings["rerank_rows"] = timings.get("rerank_rows", 0) + int(valid.sum())
        D, I = D[:, :top_k], I[:, :top_k]
        I = np.where(np.isfinite(D), I, -1)
        return D, I
