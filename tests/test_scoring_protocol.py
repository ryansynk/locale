"""The two search-protocol changes of 2026-09-22, on CPU with toy encoders.

1. Top-k regroup is the default scoring path and the raw hits are persisted:
   a later run at a smaller k must find them (find_cached_hits) and produce
   the same results and recall as a fresh search at that k.
2. Both-strand querying for dense methods: on a synthetic set where half the
   queries are reverse-complemented, both-strand recall must be >= (and here
   strictly >) forward-only recall.

The DenseIndex is built with __new__ and hand-wired state, as in
test_dense_index.py, so no GPU or model download is involved.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch
import torch.nn.functional as F

import run_benchmark
from plot_results import recall_at_k_per_query
from src.config import DenseConfig, ExperimentConfig
from src.dense_index import DenseIndex, reverse_complement
from src.topk_regroup import MISS_SCORE, regroup_topk_hits

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _cfg(tmp_path: Path, **model_kwargs) -> ExperimentConfig:
    return ExperimentConfig(
        model=DenseConfig(name="dna2vec", device="cpu", **model_kwargs),
        dataset_name="synthetic",
        dataset_dir=None,
        index_dir=tmp_path / "index",
        results_dir=tmp_path / "results",
        mutation_rate=0.0,
    )


def _mean_recall(results: pl.DataFrame, accs: list[str], truth: dict, k: int) -> float:
    """print_results' per-query recall@k, averaged, from a results frame."""
    pos = {a: i for i, a in enumerate(accs)}
    vals = []
    for qid, rows in zip(results["query_id"], results["results"]):
        y_score = np.full(len(accs), MISS_SCORE)
        for r in rows:
            y_score[pos[r["accession"]]] = r["score"]
        y_true = np.zeros(len(accs))
        y_true[pos[truth[qid]]] = 1
        vals.append(recall_at_k_per_query(y_true, y_score, k))
    return float(np.mean(vals))


class _KmerEncoder:
    """Strand-sensitive 4-mer profile, L2-normalised: a read and the chunk it
    came from share most of their 4-mers, the read's reverse complement shares
    almost none, so it stands in for an encoder that does not see strands."""

    K = 4
    _CODE = {"A": 0, "C": 1, "G": 2, "T": 3}

    def encode(self, sequences):
        out = torch.zeros(len(sequences), 4**self.K)
        for i, s in enumerate(sequences):
            for j in range(len(s) - self.K + 1):
                idx = 0
                for ch in s[j : j + self.K]:
                    idx = idx * 4 + self._CODE[ch]
                out[i, idx] += 1
        return F.normalize(out, dim=1)


def _random_dna(rng: np.random.Generator, n: int) -> str:
    return "".join(rng.choice(list("ACGT"), size=n))


def _synthetic_index(
    seed: int = 0,
    n_acc: int = 8,
    contig_len: int = 600,
    window: int = 100,
    stride: int = 50,
    **flags,
) -> tuple[DenseIndex, list[str], dict[str, str]]:
    """A DenseIndex over stride-chunked random contigs plus the contigs."""
    rng = np.random.default_rng(seed)
    enc = _KmerEncoder()
    accs = [f"acc{i}" for i in range(n_acc)]
    contigs = {a: _random_dna(rng, contig_len) for a in accs}
    rows, offsets = [], [0]
    for a in accs:
        chunks = [
            contigs[a][i : i + window]
            for i in range(0, contig_len - window + 1, stride)
        ]
        rows.append(enc.encode(chunks))
        offsets.append(offsets[-1] + len(chunks))

    index = DenseIndex.__new__(DenseIndex)
    index.no_search = False
    index.k = 10
    index.use_ann = False
    index.use_rabitq = False
    index.exhaustive = flags.get("exhaustive", False)
    index.top_k = flags.get("top_k", 5)
    index.both_strands = flags.get("both_strands", False)
    index.all_embeddings = torch.cat(rows)
    index.acc_names_flat = accs
    index.acc_offsets = offsets
    index.n_vectors = offsets[-1]
    index.model_cfg = SimpleNamespace(device="cpu", max_seq_len=window)
    index.chunk_type = "stride"
    index.chunk_overlap = window - stride
    index.contig_align_intervals = None
    index.model = enc
    return index, accs, contigs


def _reads(
    contigs: dict[str, str], n: int, read_len: int, seed: int
) -> tuple[pl.DataFrame, dict[str, str]]:
    """n reads sampled from the contigs; every even-numbered one is
    reverse-complemented. Returns the queries frame and query_id -> source."""
    rng = np.random.default_rng(seed)
    accs = list(contigs)
    ids, seqs, truth = [], [], {}
    for i in range(n):
        a = accs[rng.integers(len(accs))]
        start = rng.integers(len(contigs[a]) - read_len + 1)
        read = contigs[a][start : start + read_len]
        if i % 2 == 0:
            read = reverse_complement(read)
        qid = f"q{i}"
        ids.append(qid)
        seqs.append(read)
        truth[qid] = a
    return pl.DataFrame({"query_id": ids, "query_sequence": seqs}), truth


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_topk_regroup_is_the_default_and_gets_its_own_id(self):
        m = DenseConfig(name="dna2vec")
        assert not m.exhaustive and m.top_k == 100
        assert m.experiment_id.endswith("_exacttop100")
        assert m.hits_id == m.experiment_id.replace("_exacttop100", "_exact")

    def test_exhaustive_keeps_the_bare_baseline_id(self):
        m = DenseConfig(name="dna2vec", exhaustive=True)
        assert m.experiment_id == "dna2vec_maxlen1024_poolmax_chunkstride"
        assert m.hits_id is None

    def test_both_strands_suffix_and_engine_suffixes(self):
        m = DenseConfig(name="dna2vec", both_strands=True, top_k=50)
        assert m.experiment_id.endswith("_exacttop50_bothstrands")
        assert m.hits_id.endswith("_exact_bothstrands")
        assert DenseConfig(name="dna2vec", use_rabitq=True).experiment_id.endswith(
            "_rabitq1bit_top100"
        )
        assert DenseConfig(name="dna2vec", use_ann=True).experiment_id.endswith(
            "_cagra_top100"
        )

    def test_legacy_exact_search_configs_still_parse_to_the_same_id(self):
        # configs/*_exact_topk1000.yaml: exact_search: true, top_k: 1000
        m = DenseConfig(name="dna2vec", exact_search=True, top_k=1000)
        assert m.experiment_id.endswith("_exacttop1000")

    def test_invalid_combinations_raise(self):
        with pytest.raises(ValueError):
            DenseConfig(name="dna2vec", exhaustive=True, use_rabitq=True)
        with pytest.raises(ValueError):
            DenseConfig(name="dna2vec", use_ann=True, use_rabitq=True)


# --------------------------------------------------------------------------- #
# reverse complement + both-strand embedding
# --------------------------------------------------------------------------- #


class TestReverseComplement:
    def test_basic_and_involution(self):
        assert reverse_complement("AACG") == "CGTT"
        assert reverse_complement(reverse_complement("ACGTTGCAN")) == "ACGTTGCAN"

    def test_iupac_and_case(self):
        assert reverse_complement("acgtn") == "nacgt"
        assert reverse_complement("RYKM") == "KMRY"


class TestBothStrandEmbedding:
    class _Recorder:
        def __init__(self):
            self.seen = []

        def encode(self, sequences):
            self.seen.extend(sequences)
            return torch.zeros(len(sequences), 4)

    def _index(self, both: bool, max_len: int):
        idx = DenseIndex.__new__(DenseIndex)
        idx.both_strands = both
        idx.model_cfg = SimpleNamespace(device="cpu", max_seq_len=max_len)
        idx.model = self._Recorder()
        return idx

    def test_forward_only_is_unchanged(self):
        idx = self._index(both=False, max_len=100)
        _, ranges, strand = idx._embed_queries(
            pl.DataFrame({"query_sequence": ["AACC", "GGGT"]})
        )
        assert idx.model.seen == ["AACC", "GGGT"]
        assert ranges == [(0, 1), (1, 2)]
        assert strand.tolist() == [0, 0]

    def test_reverse_complement_chunks_follow_the_forward_ones(self):
        idx = self._index(both=True, max_len=100)
        _, ranges, strand = idx._embed_queries(
            pl.DataFrame({"query_sequence": ["AACC", "GGGT"]})
        )
        assert idx.model.seen == ["AACC", "GGTT", "GGGT", "ACCC"]
        assert ranges == [(0, 2), (2, 4)]
        assert strand.tolist() == [0, 1, 0, 1]

    def test_long_query_chunks_both_strands_whole(self):
        # 7 nt at max_len 3: 3 forward chunks + 3 reverse chunks, the reverse
        # complement chunked from its own start (not per-chunk complemented).
        idx = self._index(both=True, max_len=3)
        _, ranges, strand = idx._embed_queries(
            pl.DataFrame({"query_sequence": ["AAACCCG"]})
        )
        assert idx.model.seen == ["AAA", "CCC", "G", "CGG", "GTT", "T"]
        assert ranges == [(0, 6)]
        assert strand.tolist() == [0, 0, 0, 1, 1, 1]


# --------------------------------------------------------------------------- #
# (1) a smaller-k run reuses the persisted hits and matches a fresh search
# --------------------------------------------------------------------------- #


def _persist(cfg, hits: pl.DataFrame, tmp_path: Path) -> Path:
    """What the runner does after a scan: annotate and write the hits file."""
    path = run_benchmark.topk_hits_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    run_benchmark._annotate_results(
        hits, cfg, SimpleNamespace(index_size_gb=lambda p: 0.0), tmp_path, -1.0
    ).write_parquet(path)
    return path


class TestCachedHitsReproduceFreshSearch:
    BIG_K = 20

    @pytest.mark.parametrize("k", [1, 2, 5, 10, 20])
    def test_smaller_k_is_served_from_the_big_file_and_matches(self, tmp_path, k):
        index, accs, contigs = _synthetic_index(top_k=self.BIG_K)
        queries, truth = _reads(contigs, n=30, read_len=60, seed=k)

        # One scan at the largest k, persisted the way the runner does it.
        big_cfg = _cfg(tmp_path, top_k=self.BIG_K)
        _persist(big_cfg, index.topk_hits(queries), tmp_path)

        # A run at k: finds the file, truncates, regroups...
        cfg = _cfg(tmp_path, top_k=k)
        assert cfg.model.hits_id == big_cfg.model.hits_id
        cached = run_benchmark.find_cached_hits(cfg, queries)
        assert cached is not None
        assert cached["query_id"].to_list() == queries["query_id"].to_list()
        assert cached["hits"].list.len().max() <= k
        from_cache = regroup_topk_hits(cached, accs)

        # ...and must equal what a fresh scan at k gives, results and recall.
        index.top_k = k
        fresh = index.search(queries)
        assert from_cache.equals(fresh)
        for at in (1, 5):
            assert _mean_recall(from_cache, accs, truth, at) == pytest.approx(
                _mean_recall(fresh, accs, truth, at)
            )

    def test_larger_k_or_missing_queries_are_not_served(self, tmp_path):
        index, accs, contigs = _synthetic_index(top_k=5)
        queries, _ = _reads(contigs, n=10, read_len=60, seed=0)
        _persist(_cfg(tmp_path, top_k=5), index.topk_hits(queries), tmp_path)
        assert run_benchmark.find_cached_hits(_cfg(tmp_path, top_k=6), queries) is None
        more, _ = _reads(contigs, n=12, read_len=60, seed=0)  # q10, q11 unseen
        assert run_benchmark.find_cached_hits(_cfg(tmp_path, top_k=5), more) is None
        fewer = queries.head(4)
        assert (
            run_benchmark.find_cached_hits(_cfg(tmp_path, top_k=5), fewer) is not None
        )

    def test_other_strand_setting_has_its_own_cache(self, tmp_path):
        index, accs, contigs = _synthetic_index(top_k=5)
        queries, _ = _reads(contigs, n=10, read_len=60, seed=0)
        _persist(_cfg(tmp_path, top_k=5), index.topk_hits(queries), tmp_path)
        both = _cfg(tmp_path, top_k=5, both_strands=True)
        assert run_benchmark.find_cached_hits(both, queries) is None


# --------------------------------------------------------------------------- #
# (2) both strands >= forward only
# --------------------------------------------------------------------------- #


class TestBothStrandsRecall:
    def _recalls(self, exhaustive: bool):
        index, accs, contigs = _synthetic_index(exhaustive=exhaustive)
        queries, truth = _reads(contigs, n=40, read_len=60, seed=1)
        index.both_strands = False
        fwd = _mean_recall(index.search(queries), accs, truth, 1)
        index.both_strands = True
        both = _mean_recall(index.search(queries), accs, truth, 1)
        return fwd, both

    def test_topk_regroup_both_strands_beats_forward_only(self):
        fwd, both = self._recalls(exhaustive=False)
        assert both >= fwd
        assert both > fwd  # half the reads are reverse-complemented
        assert both > 0.9

    def test_exhaustive_both_strands_beats_forward_only(self):
        fwd, both = self._recalls(exhaustive=True)
        assert both >= fwd
        assert both > fwd
        assert both > 0.9

    def test_both_strand_hits_stay_within_top_k(self):
        index, accs, contigs = _synthetic_index(both_strands=True, top_k=7)
        queries, _ = _reads(contigs, n=10, read_len=60, seed=2)
        hits = index.topk_hits(queries)
        assert hits["hits"].list.len().max() <= 7
        # the union of two strands' lists is deduplicated by vector
        for rows in hits["hits"]:
            ids = [r["vector_id"] for r in rows]
            assert len(ids) == len(set(ids))
