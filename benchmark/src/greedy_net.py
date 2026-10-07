"""Greedy epsilon-net construction, shared by the per-accession net
(src/epsilonnet.py) and the merged net (src/merged_epsilonnet.py)."""

import torch
from tqdm import tqdm

# Score tile budget in greedy_net: (batch, centers) fp32 elements per matmul.
SIM_TILE_ELEMS = 2**30  # 4 GB
# Starting rows of greedy_net's center buffer; it doubles whenever it fills.
INIT_CENTERS = 2**20  # 3 GB at 768-d fp32


@torch.no_grad()
def greedy_net(
    x: torch.Tensor,
    eps: float,
    batch_size: int = 4096,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    progress: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedy eps-net of x (unit-norm rows): every row has cosine similarity
    >= 1 - eps to some center. Rows are visited in one random order,
    batch_size at a time; a row becomes a center if no earlier center covers
    it. Rows in the same batch are not checked against each other, so larger
    batches run faster but can add near-duplicate centers.

    x may live anywhere (e.g. host memory); the work runs on ``device``
    (default x.device), which holds the centers and one batch of x at a time.
    The visiting order is drawn on ``device``, so ``generator`` must be a
    generator for that device. progress shows a bar of rows visited and
    centers so far (off by default: the per-accession build runs many small
    nets side by side).

    Returns (ids, assign, sim), all on device:
        ids     (m,) indices into x of the centers, in the order they were added
        assign  (n,) per row of x, the position in ids of its center: the
                first-visited center that covered it (the best one existing at
                its visit, not necessarily the nearest in the final net), or
                its own position if it became a center
        sim     (n,) cosine similarity of each row to that center (1.0 for
                centers)

    Device memory: the (m, d) centers, in a buffer that starts at
    INIT_CENTERS rows and doubles when full (briefly old + new while it
    grows), two (n,) buffers, one batch and a SIM_TILE_ELEMS score tile."""
    device = x.device if device is None else torch.device(device)
    n, d = x.shape
    perm = torch.randperm(n, device=device, generator=generator)
    net_points = torch.empty((min(n, INIT_CENTERS), d), dtype=x.dtype, device=device)
    net_idx = torch.empty(len(net_points), dtype=torch.long, device=device)
    assign = torch.empty(n, dtype=torch.long, device=device)
    sim = torch.empty(n, dtype=x.dtype, device=device)
    m = 0  # centers so far: net_points[:m], net_idx[:m]
    pbar = tqdm(
        total=n, desc="Greedy net", unit="row", disable=not progress, mininterval=10
    )
    for i in range(0, n, batch_size):
        p_idx = perm[i : i + batch_size]
        p = x[p_idx.to(x.device)].to(device)
        if m == 0:
            is_new = torch.ones(len(p_idx), dtype=torch.bool, device=device)
        else:
            # Max and argmax over the centers in column tiles: the full
            # (batch, m) score matrix is 33 GB for sra500's largest accession.
            # On ties the earlier tile wins.
            cols = max(1, SIM_TILE_ELEMS // len(p_idx))
            maxsim = torch.full((len(p_idx),), -2.0, device=device, dtype=x.dtype)
            best = torch.zeros(len(p_idx), dtype=torch.long, device=device)
            for c in range(0, m, cols):
                tile = p @ net_points[c : min(c + cols, m)].T
                tile_max, tile_arg = tile.max(dim=1)
                better = tile_max > maxsim
                maxsim = torch.where(better, tile_max, maxsim)
                best = torch.where(better, tile_arg + c, best)
                del tile
            is_new = maxsim < 1 - eps
            covered = p_idx[~is_new]
            assign[covered] = best[~is_new]
            sim[covered] = maxsim[~is_new]
        new = p_idx[is_new]
        k = len(new)
        if m + k > len(net_points):
            cap = min(n, max(2 * len(net_points), m + k))
            net_points = _grow(net_points, m, cap)
            net_idx = _grow(net_idx, m, cap)
        net_points[m : m + k] = p[is_new]
        net_idx[m : m + k] = new
        assign[new] = torch.arange(m, m + k, device=device)
        sim[new] = 1.0
        m += k
        pbar.update(len(p_idx))
        pbar.set_postfix(centers=f"{m:,}", refresh=False)
    pbar.close()
    return net_idx[:m], assign, sim


def _grow(buf: torch.Tensor, used: int, rows: int) -> torch.Tensor:
    """A rows-long copy of buf holding its first ``used`` rows."""
    out = torch.empty((rows, *buf.shape[1:]), dtype=buf.dtype, device=buf.device)
    out[:used] = buf[:used]
    return out
