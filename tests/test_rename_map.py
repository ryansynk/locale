"""The old-layout -> new-layout map is a bijection over everything the naming
oracle recorded, the legacy parser is lossless, and every live yaml lands
where the map sends its old id."""

import json
import sys
from pathlib import Path

import pytest
from jsonargparse import CLI

ROOT = Path(__file__).parent.parent
BENCH = ROOT / "benchmark"
sys.path.insert(0, str(BENCH / "scripts"))
import legacy_naming as ln  # noqa: E402

from src.config import DenseMethod, ExperimentConfig  # noqa: E402

ORACLE = json.loads((ROOT / "tests/fixtures/naming_oracle.json").read_text())
RENAME_MAP = json.loads((BENCH / "rename_map.json").read_text())


def test_map_was_generated_from_this_oracle():
    assert RENAME_MAP["_meta"]["oracle_commit"] == ORACLE["_meta"]["git_commit"]


class TestLegacyParserIsLossless:
    @pytest.mark.parametrize("exp_id", sorted({e["id"] for e in ORACLE["results_dirs"]} | set(ORACLE["model_values"])))
    def test_experiment_ids_round_trip(self, exp_id):
        assert ln.format_experiment_id(ln.parse(exp_id)) == exp_id

    @pytest.mark.parametrize("hits_id", sorted({e["id"] for e in ORACLE["hits_dirs"]}))
    def test_hits_ids_round_trip(self, hits_id):
        p = ln.parse(hits_id)
        assert p["top_k"] is None and p["engine"] is not None
        assert ln.format_experiment_id(p) == hits_id

    @pytest.mark.parametrize("y", ORACLE["yamls"], ids=lambda y: y["yaml"])
    def test_yaml_ids_round_trip(self, y):
        p = ln.parse(y["experiment_id"])
        assert ln.format_experiment_id(p) == y["experiment_id"]
        assert ln.format_index_suffix(p) == y["index_suffix"]
        if y["hits_id"] is not None:
            assert ln.format_experiment_id(p, with_top_k=False) == y["hits_id"]

    def test_rejects_garbage(self):
        for bad in ("locale_maxlen256_poolmean_chunkstride", "dna2vec_abc_1_maxlen256_poolmax_chunkstride",
                    "dna2vec_maxlen256_poolmax_chunkstride_top100", "ivfpq128x8", ""):
            with pytest.raises(ValueError):
                ln.parse(bad)


class TestBijection:
    def test_every_results_dir_maps_once_and_targets_are_distinct(self):
        olds = [e["old"] for e in RENAME_MAP["results"]]
        assert sorted(olds) == sorted(e["path"] for e in ORACLE["results_dirs"])
        assert len(set(olds)) == len(olds)
        news = [e["new"] for e in RENAME_MAP["results"]]
        assert len(set(news)) == len(news)
        for e in RENAME_MAP["results"]:
            assert set(e["files"]) == set(next(o for o in ORACLE["results_dirs"] if o["path"] == e["old"])["files"])
            assert len(set(e["files"].values())) == len(e["files"])
            assert e["new"].startswith(e["old"].rsplit("/", 1)[0] + "/")

    def test_every_hits_file_maps_once_into_the_matching_search_dir(self):
        expected = {f"{e['path']}/{f}" for e in ORACLE["hits_dirs"] for f in e["files"]}
        olds = [e["old"] for e in RENAME_MAP["hits"]]
        assert sorted(olds) == sorted(expected)
        news = [e["new"] for e in RENAME_MAP["hits"]]
        assert len(set(news)) == len(news)
        results_new = {e["new"]: e for e in RENAME_MAP["results"]}
        for e in RENAME_MAP["hits"]:
            assert e["new"] == f"{e['search_dir']}/hits/{Path(e['new']).name}"
            assert "_topk_hits" not in e["search_dir"]
            assert e["config"]["top_k"] == int(Path(e["old"]).name.rsplit("topk", 1)[1].split(".")[0])
            # a hits file whose results dir exists carries the same search identity
            if e["search_dir"] in results_new:
                assert results_new[e["search_dir"]]["config"] == e["config"]

    def test_every_index_dir_maps_once(self):
        oracle_paths = [e["path"] for e in ORACLE["index_dirs"]]
        mapped = [e["old"] for e in RENAME_MAP["indexes"] if e["kind"] in ("encoder", "index", "symlink")]
        assert sorted(mapped) == sorted(oracle_paths)
        engine_olds = {(e["old"], e["new"]) for e in RENAME_MAP["indexes"] if e["kind"] in ("engine", "engine_files")}
        oracle_engines = {x["path"] for e in ORACLE["index_dirs"] for x in e.get("engines", [])}
        assert {o for o, _ in engine_olds} == oracle_engines
        news = [e["new"] for e in RENAME_MAP["indexes"] if "new" in e]
        assert len(set(news)) == len(news)
        for e in RENAME_MAP["indexes"]:
            if e["kind"] in ("engine", "engine_files"):
                assert e["new"].startswith(e["encoder_new"] + "/")
                assert e["old"].startswith(e["old"].split("/" + e["old_rel"])[0])

    def test_every_model_value_maps(self):
        assert set(RENAME_MAP["models"]) == set(ORACLE["model_values"])
        for model, labels in RENAME_MAP["models"].items():
            assert set(labels) == {"encoder", "index", "search"}
            assert labels["index"] and labels["search"]

    def test_model_values_agree_with_results_dirs(self):
        """A parquet's model column and the directory it sits in were the same id."""
        for e in RENAME_MAP["results"]:
            model = Path(e["old"]).name
            assert RENAME_MAP["models"][model] == e["labels"]


@pytest.mark.parametrize("y", ORACLE["yamls"], ids=lambda y: y["yaml"])
def test_live_yaml_round_trips_through_the_map(y, tmp_path):
    """The rewritten yaml, parsed by the new types, places itself exactly where
    the map sends the id the old yaml produced."""
    entry = RENAME_MAP["yamls"][y["yaml"]]
    cfg = CLI(
        ExperimentConfig,
        as_positional=False,
        args=["--config", str(BENCH / y["yaml"]), "--results_dir", str(tmp_path / "r"), "--index_dir", str(tmp_path / "i")],
    )
    m = cfg.model
    assert m.labels() == entry["labels"]
    rel_results = m.results_path(cfg.results_dir).relative_to(cfg.results_dir)
    assert f"{y['results_dir']}/{rel_results}" == entry["results_path"]
    rel_index = m.index_path(cfg.index_dir).relative_to(cfg.index_dir)
    assert f"{y['index_dir']}/{rel_index}" == entry["index_path"]
    # ...which is where the map sends the old results directory of that id.
    old_results = f"{y['results_dir']}/{y['experiment_id']}"
    mapped = {e["old"]: e["new"] for e in RENAME_MAP["results"]}
    if old_results in mapped:  # the yaml has been run
        assert mapped[old_results] == entry["results_path"]
        assert next(e for e in RENAME_MAP["results"] if e["old"] == old_results)["config"] == {
            **m.search_identity(), "num_queries": y["num_queries"], "random_seed": y["random_seed"],
        }
    old_index = f"{y['index_dir']}/{y['index_suffix']}"
    mapped_idx = {e["old"]: e["new"] for e in RENAME_MAP["indexes"] if e["kind"] in ("encoder", "index")}
    if old_index in mapped_idx:
        assert mapped_idx[old_index] == entry["index_path"]
        idx_entry = next(e for e in RENAME_MAP["indexes"] if e["old"] == old_index)
        assert idx_entry["config"] == m.index_identity()
    if isinstance(m, DenseMethod):
        old_engine = {e["new"]: e for e in RENAME_MAP["indexes"] if e["kind"] in ("engine", "engine_files")}
        if entry["engine_path"] in old_engine:
            assert old_engine[entry["engine_path"]]["config"] == m.engine_identity()
