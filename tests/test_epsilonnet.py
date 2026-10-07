"""EpsilonNetIndex: greedy_net covers every row (also through the column-tiled
max), is reproducible from its generator, and the engine's build -> load ->
topk_hits equals a brute-force scan over the net mapped to encoder rows."""

import numpy as np
import polars as pl
import pytest
import torch
from src import greedy_net as greedy_net_module
from src.config import EpsilonNetIndex
from src.engines import make_engine
from src.epsilonnet import EpsilonNetEngine
from src.fbin import _create_fbin_memmap, _load_fbin_mmap
from src.greedy_net import greedy_net

ROWS = [400, 1, 250, 3]
D = 8


def _unit(n, d=D, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=1)


def _worst_cover(x, net):
    return (x @ x[net].T).max(dim=1).values.min().item()


@pytest.mark.parametrize("eps", [0.05, 0.3, 0.6])
def test_greedy_net_covers_every_row(eps):
    x = _unit(2000)
    net, _, _ = greedy_net(x, eps, batch_size=64)
    assert len(net.unique()) == len(net)
    assert _worst_cover(x, net) >= 1 - eps - 1e-6


def test_column_tiled_max_matches_untiled(monkeypatch):
    # Tiles of 64 x 5 elements force many column tiles per batch.
    x = _unit(1500)
    g = lambda: torch.Generator().manual_seed(3)  # noqa: E731
    full, _, _ = greedy_net(x, 0.4, batch_size=64, generator=g())
    monkeypatch.setattr(greedy_net_module, "SIM_TILE_ELEMS", 64 * 5)
    tiled, _, _ = greedy_net(x, 0.4, batch_size=64, generator=g())
    assert torch.equal(full, tiled)


def test_generator_makes_net_reproducible():
    x = _unit(1000)
    a, _, _ = greedy_net(x, 0.3, generator=torch.Generator().manual_seed(7))
    b, _, _ = greedy_net(x, 0.3, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_assign_and_sim_point_at_a_covering_center():
    x = _unit(2000)
    ids, assign, sim = greedy_net(x, 0.3, batch_size=64)
    assert torch.equal(assign[ids], torch.arange(len(ids)))  # centers own themselves
    assert torch.allclose((x * x[ids[assign]]).sum(dim=1), sim, atol=1e-5)
    assert (sim >= 1 - 0.3 - 1e-6).all()


@pytest.mark.parametrize("init", [1, 7])
def test_growing_center_buffer_matches_preallocated(monkeypatch, init):
    # A center buffer that starts tiny and doubles many times picks the same
    # centers, assignments and similarities as one sized for every row.
    x = _unit(2000)
    g = lambda: torch.Generator().manual_seed(5)  # noqa: E731
    full = greedy_net(x, 0.3, batch_size=64, generator=g())
    monkeypatch.setattr(greedy_net_module, "INIT_CENTERS", init)
    grown = greedy_net(x, 0.3, batch_size=64, generator=g())
    assert all(torch.equal(a, b) for a, b in zip(full, grown))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_host_x_matches_device_x():
    # x in host memory with the work on the GPU equals x resident on the GPU.
    x = _unit(5000)
    g = lambda: torch.Generator(device="cuda").manual_seed(2)  # noqa: E731
    on_gpu = greedy_net(x.cuda(), 0.3, batch_size=256, generator=g())
    on_host = greedy_net(x, 0.3, batch_size=256, generator=g(), device="cuda")
    assert all(torch.equal(a, b) for a, b in zip(on_gpu, on_host))


@pytest.fixture
def fbin_dir(tmp_path):
    X = _unit(sum(ROWS), seed=1).numpy()
    fb = _create_fbin_memmap(tmp_path / "embeddings.fbin", *X.shape)
    fb[:] = X
    fb.flush()
    del fb
    starts = np.cumsum([0] + ROWS[:-1]).tolist()
    pl.DataFrame(
        {"srr_id": list("ABCD"), "start_row": starts, "num_rows": ROWS}
    ).write_parquet(tmp_path / "meta.parquet")
    return tmp_path, X, starts


def test_build_load_search(fbin_dir):
    d, X, starts = fbin_dir
    eps = 0.5
    eng = make_engine(EpsilonNetIndex(epsilon=eps))
    assert isinstance(eng, EpsilonNetEngine)
    eng.build(d, d / "net", 0, 1)

    ids = torch.load(d / "net" / "net_ids.pt").numpy()
    assert np.all(np.diff(ids) > 0)
    # The net fbin holds exactly the kept encoder rows, in net_ids order.
    assert np.array_equal(_load_fbin_mmap(d / "net" / "embeddings.fbin"), X[ids])
    # Every accession keeps a center and is covered by its own centers.
    acc = np.searchsorted(starts, ids, side="right") - 1
    assert set(acc) == set(range(len(ROWS)))
    for a, (s, n) in enumerate(zip(starts, ROWS)):
        own = torch.from_numpy(ids[acc == a] - s)
        assert _worst_cover(torch.from_numpy(X[s : s + n]), own) >= 1 - eps - 1e-6

    eng.load(d, d / "net", ["cpu"])
    q = X[:15]
    scores, hits = eng.topk_hits(q, 5)
    ref = q @ X[ids].T
    order = np.argsort(-ref, axis=1)[:, :5]
    assert np.array_equal(hits, ids[order])
    assert np.allclose(scores, np.take_along_axis(ref, order, 1), atol=1e-5)


def test_top_k_beyond_net_pads_with_minus_one(fbin_dir):
    d, X, _ = fbin_dir
    eng = make_engine(EpsilonNetIndex(epsilon=0.9))
    eng.build(d, d / "net", 0, 1)
    eng.load(d, d / "net", ["cpu"])
    n_net = len(torch.load(d / "net" / "net_ids.pt"))
    _, hits = eng.topk_hits(X[:3], n_net + 10)
    assert (hits[:, n_net:] == -1).all() and (hits[:, :n_net] >= 0).all()


def test_multi_shard_build_raises(fbin_dir):
    d, _, _ = fbin_dir
    with pytest.raises(ValueError):
        make_engine(EpsilonNetIndex()).build(d, d / "net", 0, 2)
