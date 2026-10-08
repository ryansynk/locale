"""Compare two builds of the same PCA index, e.g. a 1-node and a 2-node build
(slurm_scripts/pca_shard_check.sh): projector, build.json, shard markers,
and the stored file value by value. Exits non-zero if they disagree beyond
rounding.

The fit is seeded, so the projectors must be bitwise equal. The stored rows
can differ by matmul noise (different GPUs and block shapes) and then by one
step of the stored dtype's rounding, so values are compared bitwise first and
the rest by their size in units of the dtype's spacing.

    uv run python slurm_scripts/pca_shard_check.py <index dir A> <index dir B>
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import read_config  # noqa: E402
from src.pca import PCA_BUILD_DONE, PCA_PROJECTOR, pca_file  # noqa: E402
from src.rabitq import shard_bounds  # noqa: E402

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
CHUNK_ROWS = 4_000_000


def _ulps(ra: np.ndarray, rb: np.ndarray, item: int) -> np.ndarray:
    """Representable steps between raw IEEE bit patterns (same width): map
    sign-magnitude to an ordered integer line, so adjacent floats differ by
    one, across zero too."""
    sign = np.int64(1) << (8 * item - 1)

    def ordered(r):
        v = r.astype(np.int64)
        return np.where(v & sign, -(v & (sign - 1)), v)

    return np.abs(ordered(ra) - ordered(rb))


def main(a: Path, b: Path) -> int:
    ok = True
    ca, cb = read_config(a), read_config(b)
    print(f"config A {ca}\nconfig B {cb}")
    if ca != cb:
        print("FAIL: different build identities")
        return 1
    dtype = ca["dtype"]
    tdt = DTYPES[dtype]

    pa, pb = torch.load(a / PCA_PROJECTOR), torch.load(b / PCA_PROJECTOR)
    same_proj = torch.equal(pa, pb)
    print(f"projector {tuple(pa.shape)} bitwise equal: {same_proj}")
    ok &= same_proj

    for d in (a, b):
        built = json.loads((d / PCA_BUILD_DONE).read_text())
        bounds = shard_bounds(built["n"], built["num_shards"])
        marks = [
            json.loads((d / f"shard_rank_{r}.json").read_text())
            for r in range(built["num_shards"])
        ]
        good = all(
            (m["start"], m["end"]) == (bounds[r], bounds[r + 1])
            for r, m in enumerate(marks)
        )
        print(f"{d.name}: {built}; markers cover their ranges: {good}")
        ok &= good

    fa, fb = a / pca_file(dtype), b / pca_file(dtype)
    ha = np.fromfile(fa, np.uint32, 2)
    hb = np.fromfile(fb, np.uint32, 2)
    sizes = fa.stat().st_size, fb.stat().st_size
    print(f"headers {ha.tolist()} / {hb.tolist()}, sizes {sizes[0]:,} / {sizes[1]:,}")
    if not np.array_equal(ha, hb) or sizes[0] != sizes[1]:
        print("FAIL: headers or sizes differ")
        return 1
    n, k = map(int, ha)
    item = tdt.itemsize
    raw = {1: np.uint8, 2: np.uint16, 4: np.uint32}[item]

    n_diff = 0
    max_abs = 0.0
    max_ulps = 0  # representable steps between the two values
    zero_rows = 0
    for s in range(0, n, CHUNK_ROWS):
        e = min(s + CHUNK_ROWS, n)
        off = 8 + s * k * item
        ra = np.fromfile(fa, raw, (e - s) * k, offset=off)
        rb = np.fromfile(fb, raw, (e - s) * k, offset=off)
        neq = ra != rb
        n_diff += int(neq.sum())
        va = torch.from_numpy(ra.view({2: np.int16, 4: np.int32}[item])).view(tdt)
        vb = torch.from_numpy(rb.view({2: np.int16, 4: np.int32}[item])).view(tdt)
        # an unwritten (hole) row reads as all zeros
        zero_rows += int((vb.reshape(-1, k) == 0).all(1).sum())
        if neq.any():
            sel = torch.from_numpy(neq)
            diff = (va.float()[sel] - vb.float()[sel]).abs()
            max_abs = max(max_abs, diff.max().item())
            max_ulps = max(max_ulps, int(_ulps(ra[neq], rb[neq], item).max()))
        print(f"  rows [{s:,}, {e:,}): {int(neq.sum()):,} differing values", flush=True)

    total = n * k
    print(
        f"values: {total:,}; bitwise different: {n_diff:,} ({n_diff / total:.2e}); "
        f"max |diff| {max_abs:.3e}, max {max_ulps} {dtype} ULP(s) apart; "
        f"all-zero rows in B: {zero_rows:,}"
    )
    if zero_rows:
        print("FAIL: B has all-zero rows (unwritten range?)")
        ok = False
    if max_ulps > 1:
        print("FAIL: values differ by more than one rounding step")
        ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
