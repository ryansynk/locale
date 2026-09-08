"""The fbin embedding format: a uint32 [n, d] header followed by float32[n * d].

This is the big-ann-benchmarks convention, and it is the interchange format between
the index builders here and the published vecdb bundles. Kept in its own module so a
consumer that only needs to read vectors does not pull in torch, transformers, or
cuvs along with them.
"""

from pathlib import Path

import numpy as np


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
