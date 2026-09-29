"""The encoder/index split: identities, labels, config.json checking, and
that every live yaml parses into the new types."""

import dataclasses
from pathlib import Path

import pytest
from jsonargparse import CLI

from src.config import (
    PAPER_CHECKPOINT,
    ConfigMismatch,
    DenseMethod,
    EncoderConfig,
    ExactIndex,
    ExperimentConfig,
    IVFPQIndex,
    MetagraphConfig,
    MMseqs2Config,
    RaBitQIndex,
    check_config,
    ensure_config,
    read_config,
    results_file_name,
    run_search_identity,
    write_config,
)

BENCH = Path(__file__).parent.parent / "benchmark"
LIVE_CONFIG_DIRS = ["sra50", "sra500", "sra4571", "sra55viral", "ablation_sweep", "backbone_sweep"]
LIVE_YAMLS = sorted(p for d in LIVE_CONFIG_DIRS for p in (BENCH / "configs" / d).glob("*.yaml"))


class TestEncoderConfig:
    def test_paper_checkpoint_is_pinned_when_path_unset(self):
        enc = EncoderConfig(name="locale")
        assert enc.checkpoint() == (PAPER_CHECKPOINT["ckpt_id"], PAPER_CHECKPOINT["step"])

    def test_local_checkpoint_identity_comes_from_the_path(self):
        enc = EncoderConfig(name="locale", checkpoint_path="/x/checkpoints/abc123/checkpoint5859.pth.tar")
        assert enc.checkpoint() == ("abc123", 5859)
        assert enc.identity()["checkpoint"] == "abc123"

    def test_identity_excludes_runtime_fields(self):
        a = EncoderConfig(name="llmed", batch_size=1, device="cpu", both_strands=False, query_embed_gpus=3)
        b = EncoderConfig(name="llmed", batch_size=1000, device="cuda", both_strands=True)
        assert a.identity() == b.identity()

    def test_bad_name_and_overlap_raise(self):
        with pytest.raises(ValueError):
            EncoderConfig(name="evo2")
        with pytest.raises(ValueError):
            EncoderConfig(name="locale", max_seq_len=100, chunk_overlap=100)


class TestIndexConfigs:
    def test_build_and_search_fields_partition_the_knobs(self):
        for cls in (ExactIndex, RaBitQIndex, IVFPQIndex):
            names = {f.name for f in dataclasses.fields(cls) if f.name != "engine"}
            assert set(cls.BUILD) | set(cls.SEARCH) == names, cls
            assert not set(cls.BUILD) & set(cls.SEARCH), cls

    def test_dense_method_identities(self):
        m = DenseMethod(
            encoder=EncoderConfig(name="locale"),
            index=IVFPQIndex(lists_per_shard=16384, nprobe=272, rerank=10),
            encoder_label="locale@8vqiabk9",
            index_label="ivfpq-pq128x8-L16384x16",
            search_label="top100-np272-rr10",
        )
        assert m.engine_identity() == {
            "engine": "ivfpq", "pq_dim": 128, "pq_bits": 8, "lists_per_shard": 16384, "num_shards": 16,
        }
        assert m.search_identity() == {
            "engine": "ivfpq", "top_k": 100, "nprobe": 272, "rerank": 10, "lut": "float16",
            "both_strands": True,
        }
        assert str(m) == "locale@8vqiabk9/ivfpq-pq128x8-L16384x16/top100-np272-rr10"
        assert m.results_path(Path("r")) == Path("r/locale@8vqiabk9/ivfpq-pq128x8-L16384x16/top100-np272-rr10")
        assert m.engine_path(Path("i")) == Path("i/locale@8vqiabk9/ivfpq-pq128x8-L16384x16")

    def test_labels_must_be_plain_directory_names(self):
        with pytest.raises(ValueError):
            DenseMethod(encoder=EncoderConfig(name="locale"), encoder_label="a/b")
        with pytest.raises(ValueError):
            MetagraphConfig(search_label="")

    def test_non_dense_methods_have_no_encoder_level(self):
        m = MetagraphConfig(k=31, server_parallel=64, search_label="p64")
        assert m.labels() == {"encoder": None, "index": "metagraph", "search": "p64"}
        assert m.index_path(Path("i")) == Path("i/metagraph")
        assert m.results_path(Path("r")) == Path("r/metagraph/p64")
        assert m.index_identity() == {"engine": "metagraph", "k": 31}
        assert MMseqs2Config().search_identity() == {"engine": "mmseqs", "max_seqs": 300}

    def test_results_file_name(self):
        assert results_file_name(0.1) == "mut0.10.parquet"
        assert results_file_name(0.0) == "mut0.00.parquet"


class TestConfigJson:
    def test_write_read_check(self, tmp_path):
        ident = {"engine": "ivfpq", "pq_dim": 128, "lut": "float16", "fastscan": False, "x": None}
        write_config(tmp_path / "d", ident)
        assert read_config(tmp_path / "d") == ident
        check_config(tmp_path / "d", ident)  # no raise
        with pytest.raises(ConfigMismatch, match="pq_dim"):
            check_config(tmp_path / "d", {**ident, "pq_dim": 96})

    def test_ensure_config_writes_checks_and_refuses_unstamped_artifacts(self, tmp_path):
        d = tmp_path / "label"
        ensure_config(d, {"a": 1}, artifact_present=False)
        assert read_config(d) == {"a": 1}
        ensure_config(d, {"a": 1}, artifact_present=True)
        with pytest.raises(ConfigMismatch):
            ensure_config(d, {"a": 2}, artifact_present=True)
        legacy = tmp_path / "legacy"
        legacy.mkdir()
        with pytest.raises(ConfigMismatch, match="not.*built under this layout"):
            ensure_config(legacy, {"a": 1}, artifact_present=True)

    def test_run_identity_adds_the_query_draw(self, tmp_path):
        cfg = ExperimentConfig(
            model=DenseMethod(encoder=EncoderConfig(name="dna2vec")),
            dataset_name="x", dataset_dir=None, index_dir=tmp_path, results_dir=tmp_path / "r",
            num_queries=500, random_seed=7,
        )
        assert run_search_identity(cfg) == {
            "engine": "exact", "top_k": 100, "both_strands": True, "num_queries": 500, "random_seed": 7,
        }


@pytest.mark.parametrize("yaml_path", LIVE_YAMLS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_live_yaml_parses_and_places_itself(yaml_path, monkeypatch, tmp_path):
    """Every live config parses into DenseMethod / MetagraphConfig / MMseqs2Config
    and names three plain labels. index_dir/results_dir are redirected so
    parsing (which mkdirs results_dir) touches nothing real."""
    cfg = CLI(
        ExperimentConfig,
        as_positional=False,
        args=["--config", str(yaml_path), "--results_dir", str(tmp_path / "r"), "--index_dir", str(tmp_path / "i")],
    )
    m = cfg.model
    assert isinstance(m, (DenseMethod, MetagraphConfig, MMseqs2Config))
    labels = m.labels()
    assert labels["index"] and labels["search"]
    if isinstance(m, DenseMethod):
        assert labels["encoder"]
        assert m.results_path(cfg.results_dir).relative_to(cfg.results_dir).parts == (
            labels["encoder"], labels["index"], labels["search"],
        )
        assert m.engine_path(cfg.index_dir).parent == m.index_path(cfg.index_dir)
    else:
        assert labels["encoder"] is None


def test_merge_stage_refuses_until_every_shard_is_done(tmp_path):
    model = DenseMethod(encoder=EncoderConfig(name="dna2vec"))
    kw = dict(model=model, dataset_name="x", dataset_dir=str(tmp_path), index_dir=tmp_path,
              results_dir=tmp_path / "r", stage="merge", num_shards=2)
    shard0 = model.index_path(tmp_path) / "shard_0"
    shard0.mkdir(parents=True)
    (shard0 / ".done").touch()
    with pytest.raises(ValueError, match=r"shards not complete: \[1\]"):
        ExperimentConfig(**kw)
    shard1 = model.index_path(tmp_path) / "shard_1"
    shard1.mkdir()
    (shard1 / ".done").touch()
    assert ExperimentConfig(**kw).stage == "merge"


def test_shard_must_be_below_num_shards(tmp_path):
    with pytest.raises(ValueError, match="shard 2"):
        ExperimentConfig(
            model=DenseMethod(encoder=EncoderConfig(name="dna2vec")), dataset_name="x",
            dataset_dir=str(tmp_path), index_dir=tmp_path, results_dir=tmp_path / "r",
            shard=2, num_shards=2,
        )
