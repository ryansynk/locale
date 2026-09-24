"""MetagraphIndex shard merge and result post-processing (no metagraph binary)."""

import pandas as pd
import polars as pl
import pytest

from src import metagraph_index as mg
from src.metagraph_index import MetagraphIndex


def _make_shard(index_path, rank, accessions, graph_bytes=10, anno_bytes=5):
    shard = index_path / f"shard_{rank}"
    shard.mkdir(parents=True)
    (shard / mg.GRAPH_FILE).write_bytes(b"g" * graph_bytes)
    for ext in mg.GRAPH_SIDECARS:
        (shard / (mg.GRAPH_FILE + ext)).write_bytes(b"s")
    (shard / mg.ANNOTATION_FILE).write_bytes(b"a" * anno_bytes)
    (shard / mg.MANIFEST_FILE).write_text("".join(f"/d/{a}/{a}.contigs.fa\n" for a in accessions))
    return shard


class TestMergeShards:
    def test_writes_csv_and_sorted_manifest(self, tmp_path):
        _make_shard(tmp_path, 0, ["SRR3", "SRR1"])
        _make_shard(tmp_path, 1, ["SRR2"])

        MetagraphIndex.merge_shards(tmp_path, 2)

        rows = (tmp_path / mg.GRAPHS_CSV).read_text().splitlines()
        assert len(rows) == 2
        names = {r.split(",")[0] for r in rows}
        assert names == {mg.CSV_INDEX_NAME}, "all shards must share one name"
        assert rows[0].split(",")[1] == str(tmp_path / "shard_0" / mg.GRAPH_FILE)
        assert rows[1].split(",")[2] == str(tmp_path / "shard_1" / mg.ANNOTATION_FILE)

        manifest = (tmp_path / mg.MANIFEST_FILE).read_text().splitlines()
        assert manifest == sorted(manifest) and len(manifest) == 3

        pairs = MetagraphIndex.index_files(tmp_path)
        assert [p[0].parent.name for p in pairs] == ["shard_0", "shard_1"]

    def test_index_size_sums_every_served_file(self, tmp_path):
        _make_shard(tmp_path, 0, ["A"], graph_bytes=100, anno_bytes=50)
        _make_shard(tmp_path, 1, ["B"], graph_bytes=200, anno_bytes=25)
        MetagraphIndex.merge_shards(tmp_path, 2)
        index = MetagraphIndex.__new__(MetagraphIndex)
        expected = (
            100 + 50 + 200 + 25
            + 2 * len(mg.GRAPH_SIDECARS)
            + (tmp_path / mg.MANIFEST_FILE).stat().st_size
        )
        assert index.index_size_gb(tmp_path) == pytest.approx(expected / 1024**3)

    def test_missing_sidecar_is_refused(self, tmp_path):
        shard = _make_shard(tmp_path, 0, ["A"])
        (shard / (mg.GRAPH_FILE + ".anchors")).unlink()
        with pytest.raises(FileNotFoundError, match="shard_0"):
            MetagraphIndex.merge_shards(tmp_path, 1)

    def test_duplicate_contig_is_refused(self, tmp_path):
        _make_shard(tmp_path, 0, ["A"])
        _make_shard(tmp_path, 1, ["A"])
        with pytest.raises(ValueError, match="more than one shard"):
            MetagraphIndex.merge_shards(tmp_path, 2)

    def test_single_node_layout_has_no_csv(self, tmp_path):
        pairs = MetagraphIndex.index_files(tmp_path)
        assert pairs == [(tmp_path / mg.GRAPH_FILE, tmp_path / mg.ANNOTATION_FILE)]


class _FakeClient:
    def __init__(self, frame):
        self.frame = frame
        self.calls = []

    def search(self, sequences, **kwargs):
        self.calls.append((list(sequences), kwargs))
        return self.frame


def _index_with(frame):
    index = MetagraphIndex.__new__(MetagraphIndex)
    index.graph_client = _FakeClient(frame)
    return index


class TestSearchPostprocessing:
    def test_union_is_ranked_and_cut_to_top_labels(self):
        # Query 0 gets TOP_LABELS hits from "shard A" and 20 better ones from
        # "shard B", as a CSV-backed server would append them; query 1 has none.
        n = mg.TOP_LABELS
        rows = [("0", f"/d/A{i}/A{i}.contigs.fa", 10) for i in range(n)]
        rows += [("0", f"/d/B{i}/B{i}.contigs.fa", 50 + i) for i in range(20)]
        frame = pd.DataFrame(rows, columns=["seq_description", "sample", "kmer_count"])
        queries = pl.DataFrame({"query_id": ["q0", "q1"], "query_sequence": ["ACGT" * 20] * 2})

        out = _index_with(frame).search(queries)

        assert out["query_id"].to_list() == ["q0", "q1"]
        res0 = out["results"][0].to_list()
        assert len(res0) == n
        scores = [r["score"] for r in res0]
        assert scores == sorted(scores, reverse=True)
        assert all(r["accession"].startswith("B") for r in res0[:20])
        assert res0[0] == {"accession": "B19", "score": 69.0}
        assert out["results"][1].to_list() == []
        assert out.schema["results"] == mg.RESULTS_DTYPE

    def test_client_is_asked_for_top_labels(self):
        frame = pd.DataFrame([("0", "/d/X/X.contigs.fa", 3)],
                             columns=["seq_description", "sample", "kmer_count"])
        index = _index_with(frame)
        index.search(pl.DataFrame({"query_id": ["q"], "query_sequence": ["ACGT" * 20]}))
        _, kwargs = index.graph_client.calls[0]
        assert kwargs == {"top_labels": mg.TOP_LABELS, "discovery_fraction": mg.DISCOVERY_FRACTION}

    def test_no_hits_at_all(self):
        frame = pd.DataFrame(columns=["seq_description", "sample", "kmer_count"])
        queries = pl.DataFrame({"query_id": ["q0"], "query_sequence": ["ACGT" * 20]})
        out = _index_with(frame).search(queries)
        assert out["results"].to_list() == [[]]
        assert out.schema["results"] == mg.RESULTS_DTYPE
