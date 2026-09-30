"""embed.py must produce exactly what dense_index.py produces.

The probe's whole claim rests on these being the same vectors the benchmark
would compute. So this compares against two things: the encoder the benchmark
constructs (``DenseEncoder``), and the query-embedding path ``DenseIndex``
actually calls (``_embed_queries``).

The model runs on CPU here. It is small (a 117M-parameter DNABERT-2 backbone)
and the test embeds a handful of short sequences, but it still loads a
checkpoint, so it is marked ``slow`` and skips cleanly when no checkpoint is
reachable (no Hub access and nothing cached).
"""

import os

import numpy as np
import polars as pl
import pytest

import embed as emb

pytestmark = pytest.mark.slow

# Short enough that dense_index's query chunking (non-overlapping max_seq_len
# windows) and this probe's tiling both yield exactly one window, which is what
# makes the two paths directly comparable.
SEQS = [
    "ATGGTTAAAAAATCACTGCGCCAGTTTACGCTGATGGCGACGGCAACCGTCACGCTGTTG",
    "GCGAGCGCTAGCGCTTAACGGTTCAGGCTGAACCGTTTAGCCAGCTGGCAGGTCAGGTTA",
    "TTTTTTTTTTTTTTTTTTTTAAAAAAAAAAAAAAAAAAAACCCCCCCCCCGGGGGGGGGG",
    "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
]


def _checkpoint() -> str | None:
    """A local checkpoint if one is configured, else None (= Hub download)."""
    return os.environ.get("PROBE_CHECKPOINT") or None


@pytest.fixture(scope="module")
def cfg():
    return emb.encoder_config(
        device="cpu", batch_size=2, checkpoint_path=_checkpoint()
    )


@pytest.fixture(scope="module")
def ours(cfg):
    try:
        encoder = emb.load_encoder(cfg)
    except Exception as exc:  # no Hub access and nothing cached
        pytest.skip(f"LOCALE checkpoint unavailable: {exc}")
    return emb.embed(encoder, SEQS)


class TestEncoderParity:
    def test_matches_dense_encoder(self, cfg, ours):
        """Same vectors as benchmark/src/encoders.py's DenseEncoder."""
        from src.encoders import DenseEncoder

        theirs = DenseEncoder(cfg).encode(SEQS).detach().cpu().numpy()
        assert ours.shape == theirs.shape
        np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-6)

    def test_matches_dense_index_query_path(self, cfg, ours, tmp_path):
        """Same vectors as DenseIndex._embed_queries, the path a benchmark
        search actually takes to turn a query string into a vector."""
        from src.config import DenseMethod, ExactIndex, ExperimentConfig
        from src.dense_index import DenseIndex

        exp = ExperimentConfig(
            model=DenseMethod(encoder=cfg, index=ExactIndex()),
            dataset_name="probe-parity",
            dataset_dir=str(tmp_path),
            index_dir=tmp_path / "index",
            results_dir=tmp_path / "results",
        )
        index = DenseIndex(exp)
        queries = pl.DataFrame({
            "query_id": [f"q{i}" for i in range(len(SEQS))],
            "query_sequence": SEQS,
        })
        feats, ranges, strands = index._embed_queries(queries)

        # both_strands is off in encoder_config, so one chunk per query.
        assert ranges == [(i, i + 1) for i in range(len(SEQS))]
        assert strands.tolist() == [0] * len(SEQS)
        np.testing.assert_allclose(
            ours, feats.detach().cpu().numpy(), rtol=0, atol=1e-6
        )

    def test_embeddings_are_l2_normalised(self, ours):
        np.testing.assert_allclose(
            np.linalg.norm(ours, axis=1), 1.0, rtol=0, atol=1e-5
        )

    def test_embedding_dim_is_the_backbone_hidden_size(self, cfg, ours):
        # 768, not the 128 of TrainConfig.dim: LOCALEEncoder pools
        # `model.bert_q(**tokens)[0]`, the backbone hidden states, and never
        # touches the MoCo projection head. A silent backbone swap would show up
        # here before it showed up as a strange AUROC.
        encoder = emb.load_encoder(cfg)
        assert ours.shape == (len(SEQS), encoder.model.bert_q.config.hidden_size)
        assert ours.shape[1] == 768

    def test_tiled_long_sequence_matches_manual_windows(self, cfg, ours):
        """embed_windows == embed(tile(...)), with owners lining up."""
        encoder = emb.load_encoder(cfg)
        seqs = [SEQS[0] * 10, SEQS[1]]          # one long, one single-window
        vec, owner = emb.embed_windows(encoder, seqs)
        manual = emb.tile(seqs[0]) + emb.tile(seqs[1])
        assert len(manual) == vec.shape[0]
        assert owner.tolist() == [0] * len(emb.tile(seqs[0])) + [1]
        np.testing.assert_allclose(
            vec, emb.embed(encoder, manual), rtol=0, atol=1e-6
        )


class TestSelfSimilarity:
    def test_identical_sequence_scores_one(self, ours):
        sim = emb.cosine(ours, ours)
        np.testing.assert_allclose(np.diag(sim), 1.0, rtol=0, atol=1e-5)

    def test_cosine_is_symmetric(self, ours):
        sim = emb.cosine(ours, ours)
        np.testing.assert_allclose(sim, sim.T, rtol=0, atol=1e-6)
