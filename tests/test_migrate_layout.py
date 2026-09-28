"""migrate_layout.py on a synthetic copy of the old tree: every path in the
oracle is rebuilt (empty files, tiny parquets with the old ``model`` column),
migrated, and checked against rename_map.json. Also: the dry run touches
nothing, a second run is a no-op, and a config mismatch is refused."""

import json
import re
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).parent.parent
BENCH = ROOT / "benchmark"
ORACLE = json.loads((ROOT / "tests/fixtures/naming_oracle.json").read_text())
RENAME_MAP = json.loads((BENCH / "rename_map.json").read_text())
SCRIPT = BENCH / "scripts" / "migrate_layout.py"


def _parquet(path: Path, model: str, hits: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        {"hits": [[{"accession": "a", "score": 1.0, "vector_id": 3}]]}
        if hits
        else {"results": [[{"accession": "a", "score": 1.0}]]}
    )
    pl.DataFrame(
        {"query_id": ["q0"], **payload, "index_size_gb": [1.0], "avg_time": [-1.0], "model": [model],
         "mutation_rate": [0.0], "query_type": ["raw_read"], "checkpoint": [None], "max_len": [256],
         "checkpoint_step_num": [None], "chunk_type": ["stride"]}
    ).write_parquet(path)


def build_old_tree(root: Path) -> None:
    for e in ORACLE["results_dirs"] + ORACLE["hits_dirs"]:
        for f in e["files"]:
            _parquet(root / e["path"] / f, e["id"], hits=e in ORACLE["hits_dirs"])
    for e in ORACLE["index_dirs"]:
        d = root / e["path"]
        if e["kind"] == "symlink":
            d.parent.mkdir(parents=True, exist_ok=True)
            d.symlink_to(e["target"])
            continue
        d.mkdir(parents=True, exist_ok=True)
        if e["done"]:
            (d / ".done").touch()
        if e["kind"] == "dense":
            (d / "embeddings.fbin").touch()
            (d / "meta.parquet").touch()
            for s in e["shards"]:
                (d / s).mkdir()
                (d / s / "embeddings.fbin").touch()
            for o in e["other_dirs"]:
                (d / o).mkdir()
                (d / o / "rank_0_of_4.parquet").touch()
            for eng in e["engines"]:
                ed = root / eng["path"]
                ed.mkdir(parents=True, exist_ok=True)
                if eng["engine"] == "ivfpq":
                    (ns,) = eng["num_shards"]
                    for i in range(eng["n_shard_files"]):
                        (ed / f"shard_{i}_of_{ns}.cuvs").touch()
                        (ed / f"shard_{i}_of_{ns}.json").touch()
                elif eng["engine"] == "rabitq":
                    for f in ("centroid.npy", "rotation.npy", "codes_rank_0.u8") + (("meta.json",) if eng["complete"] else ()):
                        (ed / f).touch()
                else:
                    for files in eng["files"].values():
                        for f in files:
                            (ed / f).touch()
                    for f in eng["other_files"]:
                        (ed / f).touch()
        else:
            (d / "graph_primary.dbg" if e["kind"] == "metagraph" else d / "contig_manifest.txt").touch()


def run(root: Path, *args: str) -> str:
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *args],
        capture_output=True, text=True, cwd=BENCH,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout


def snapshot(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


@pytest.fixture(scope="module")
def migrated(tmp_path_factory) -> tuple[Path, str, str]:
    root = tmp_path_factory.mktemp("tree")
    build_old_tree(root)
    before = snapshot(root)
    dry = run(root, "--dry-run")
    assert snapshot(root) == before, "dry run changed the tree"
    real = run(root)
    return root, dry, real


def test_dry_run_lists_every_oracle_directory_and_nothing_else(migrated):
    root, dry, _ = migrated
    moved = set(re.findall(r"mv (\S+) -> ", dry)) | set(re.findall(r"rm symlink (\S+);", dry))
    expected_dirs = {e["path"] for e in ORACLE["results_dirs"]}
    for d in expected_dirs:  # results move file by file
        assert any(m.startswith(d + "/") for m in moved), d
    for e in ORACLE["hits_dirs"]:
        for f in e["files"]:
            assert f"{e['path']}/{f}" in moved
    stamped = set(re.findall(r"write (\S+)/config.json", dry))
    for e in ORACLE["index_dirs"]:
        if e["kind"] == "mmseqs":  # old path == new path: only stamped
            assert e["path"] in stamped, e["path"]
        else:
            assert e["path"] in moved, e["path"]
        for eng in e.get("engines", []):
            if eng["engine"] == "ivf":
                assert any(m.startswith(eng["path"] + "/") for m in moved)
            else:
                assert eng["path"] in moved
    # nothing outside the oracle
    known = {e["path"] for e in ORACLE["results_dirs"] + ORACLE["hits_dirs"] + ORACLE["index_dirs"]} | {
        x["path"] for e in ORACLE["index_dirs"] for x in e.get("engines", [])
    }
    for m in moved:
        if m.endswith(".migrating"):
            continue
        assert any(m == k or m.startswith(k + "/") for k in known), m


def test_every_new_path_exists_and_no_old_path_remains(migrated):
    root, _, _ = migrated
    for e in RENAME_MAP["results"]:
        old = root / e["old"]
        if Path(e["new"]).is_relative_to(e["old"]):  # mmseqs -> mmseqs/default
            assert not list(old.glob("*.parquet"))
        else:
            assert not old.exists()
        for old_name, new_name in e["files"].items():
            assert (root / e["new"] / new_name).is_file()
        assert json.loads((root / e["new"] / "config.json").read_text()) == e["config"]
    for e in RENAME_MAP["hits"]:
        assert not (root / e["old"]).exists()
        assert (root / e["new"]).is_file()
        assert json.loads((root / e["search_dir"] / "config.json").read_text()) == e["config"]
    for e in RENAME_MAP["indexes"]:
        if e["kind"] == "leftover_files":
            continue
        new = root / e["new"]
        if e["kind"] == "symlink":
            assert new.is_symlink() and str(new.readlink()) == e["target"]
            assert not (root / e["old"]).exists() and not (root / e["old"]).is_symlink()
            continue
        assert new.is_dir(), e["new"]
        assert json.loads((new / "config.json").read_text()) == e["config"]
        if e.get("done"):
            assert (new / ".done").exists()
        if e["kind"] == "encoder":
            assert (new / "embeddings.fbin").exists() and (new / "meta.parquet").exists()
            for s in e["shards"]:
                assert (new / s / "embeddings.fbin").exists()
        if e["kind"] == "engine_files":
            for f in e["files"]:
                assert (new / f).exists()
    # old encoder parents are gone; the ivf probe leftovers and stale partials stayed
    assert not (root / "indexes/sra50/locale").exists()
    assert not (root / "indexes/sra50/dna2vec").exists()
    assert (root / "indexes/sra4571/locale@8vqiabk9/ivf/train_sample.npy").exists()
    assert (root / "indexes/sra55viral/locale@8vqiabk9/exact_top100_bothstrands_partials_mut0.0_n500_seed1337").is_dir()
    assert not any(p.name.endswith("_topk_hits") for p in (root / "results").iterdir())
    assert not any(p.name.endswith(".migrating") for p in root.rglob("*"))


def test_parquets_carry_label_columns_instead_of_model(migrated):
    root, _, _ = migrated
    for e in RENAME_MAP["results"]:
        for new_name in e["files"].values():
            df = pl.read_parquet(root / e["new"] / new_name)
            assert "model" not in df.columns
            assert df["dataset"][0] == e["dataset"]
            assert (df["encoder"][0], df["index"][0], df["search"][0]) == tuple(e["labels"].values())
    e = RENAME_MAP["hits"][0]
    df = pl.read_parquet(root / e["new"])
    assert "model" not in df.columns and df["hits"].dtype == pl.List


def test_second_run_is_a_no_op(migrated):
    root, _, _ = migrated
    before = snapshot(root)
    out = run(root)
    assert snapshot(root) == before
    assert "== done: 0 moves, 0 parquet rewrites" in out


def test_migrated_dirs_pass_the_loaders_config_check(migrated):
    root, _, _ = migrated
    from src.config import ConfigMismatch, ensure_config

    e = next(x for x in RENAME_MAP["indexes"] if x["kind"] == "engine")
    ensure_config(root / e["new"], e["config"], artifact_present=True)  # no raise
    with pytest.raises(ConfigMismatch):
        ensure_config(root / e["new"], {**e["config"], "pq_dim": 7}, artifact_present=True)
