"""One-off: convert a pre-fbin dense index's embeddings.npy to embeddings.fbin.

The indexes built before the fbin switch (e.g. indexes_500/rawbert/...) store
their vectors as a plain .npy; everything downstream now memmaps the
big-ann-benchmarks fbin layout instead (see src/fbin.py). This streams the .npy
into <dir>/embeddings.fbin in bounded-memory tiles, writing to a .tmp first so
an interrupted run never leaves a half-written file behind under the real name.
The .npy is left in place; delete it by hand once the fbin is in use.

Usage:
    uv run python convert_npy_to_fbin.py <index_dir> [--tile_rows 500000]
"""

from pathlib import Path

import numpy as np
from jsonargparse import CLI
from src.fbin import _create_fbin_memmap


def main(index_dir: Path, tile_rows: int = 500_000):
    src = index_dir / "embeddings.npy"
    dst = index_dir / "embeddings.fbin"
    if dst.exists():
        raise SystemExit(f"{dst} already exists; refusing to overwrite.")

    arr = np.load(src, mmap_mode="r")
    if arr.dtype != np.float32 or arr.ndim != 2:
        raise SystemExit(f"expected float32 (n, d), got {arr.dtype} {arr.shape}")
    n, d = arr.shape
    print(f"{src}: {n:,} x {d} float32 ({arr.nbytes / 1024**3:.0f} GiB)")

    tmp = index_dir / "embeddings.fbin.tmp"
    out = _create_fbin_memmap(tmp, n, d)
    for i in range(0, n, tile_rows):
        out[i : i + tile_rows] = arr[i : i + tile_rows]
        print(f"  {min(i + tile_rows, n):,}/{n:,}", flush=True)
    out.flush()
    del out

    # Spot-check a few rows through the real reader before the rename makes it live.
    from src.fbin import _load_fbin_mmap

    check = _load_fbin_mmap(tmp)
    assert check.shape == (n, d)
    for row in (0, n // 2, n - 1):
        assert np.array_equal(check[row], arr[row]), f"row {row} mismatch"
    del check

    tmp.rename(dst)
    print(f"done: {dst}")


if __name__ == "__main__":
    CLI(main, as_positional=True)
