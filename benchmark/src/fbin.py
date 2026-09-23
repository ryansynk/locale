"""The fbin embedding format: a uint32 [n, d] header followed by float32[n * d].

This is the big-ann-benchmarks convention, and it is the interchange format between
the index builders here and the published vecdb bundles. Kept in its own module so a
consumer that only needs to read vectors does not pull in torch, transformers, or
cuvs along with them.
"""

import os
from pathlib import Path

import numpy as np

FBIN_HEADER_BYTES = 8


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


def read_fbin_rows(
    fd: int, start: int, end: int, out: np.ndarray, piece_bytes: int = 64 << 20
) -> np.ndarray:
    """Fill ``out[: end - start]`` with fbin rows [start, end) and return that view.

    ``fd`` is an open descriptor on the fbin; ``out`` is a C-contiguous float32
    (rows, d) buffer with at least end - start rows. The rows are read with
    positional reads of ``piece_bytes`` each, so the call is thread-safe (no
    shared file offset) and never faults a memory map.

    Why not slice the memmap: on Lustre a page-faulted read degrades to one
    small synchronous RPC per page whenever the client's readahead heuristic
    does not recognise the access as sequential. For the multi-threaded scans
    in dense_index (four GPU threads streaming four distant regions of one
    6 TB mapping) that meant ~70 MB/s per node, against 2.5+ GB/s for 64 MB
    preads on the same node (Perlmutter, 2026-09-23).
    """
    if not out.flags.c_contiguous or out.dtype != np.float32:
        raise ValueError("out must be a C-contiguous float32 array")
    n_rows = end - start
    if n_rows < 0 or n_rows > out.shape[0]:
        raise ValueError(f"rows [{start}, {end}) do not fit a {out.shape} buffer")
    row_bytes = out.shape[1] * out.dtype.itemsize
    n_bytes = n_rows * row_bytes
    mv = memoryview(out).cast("B")[:n_bytes]
    offset = FBIN_HEADER_BYTES + start * row_bytes
    done = 0
    while done < n_bytes:
        piece = mv[done : done + min(piece_bytes, n_bytes - done)]
        got = os.preadv(fd, [piece], offset + done)
        if got <= 0:
            raise IOError(f"short read of fbin at byte {offset + done}")
        done += got
    return out[:n_rows]
