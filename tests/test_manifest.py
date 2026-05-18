"""
Tests for manifest validation logic.

The validation is currently inline in run_benchmark.py. These tests document
the expected contract and should remain green once the logic is extracted to a
standalone function — e.g.:

    def validate_manifest(manifest_ids: set[str], accession_paths: list[Path]):
        downloaded_ids = {p.name.removesuffix(".contigs.fa") for p in accession_paths}
        if missing := manifest_ids - downloaded_ids:
            raise RuntimeError(
                f"{len(missing)} accessions missing after download: {missing}"
            )

We define that same function here so the tests run today, and so the expected
interface is explicit for when you extract it from run_benchmark.py.
"""

from pathlib import Path
import pytest


# ---------------------------------------------------------------------------
# Reference implementation (mirrors the logic in run_benchmark.py)
# ---------------------------------------------------------------------------


def validate_manifest(manifest_ids: set[str], accession_paths: list[Path]):
    downloaded_ids = {p.name.removesuffix(".contigs.fa") for p in accession_paths}
    if missing := manifest_ids - downloaded_ids:
        raise RuntimeError(
            f"{len(missing)} accessions missing after download: {missing}"
        )


def _fake_paths(ids: list[str]) -> list[Path]:
    return [Path(f"/fake/{acc}/{acc}.contigs.fa") for acc in ids]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidateManifest:
    def test_complete_match_raises_nothing(self):
        ids = {"SRR001", "SRR002", "SRR003"}
        validate_manifest(ids, _fake_paths(list(ids)))  # must not raise

    def test_missing_accession_raises_runtime_error(self):
        manifest = {"SRR001", "SRR002", "SRR003"}
        downloaded = _fake_paths(["SRR001", "SRR002"])  # SRR003 missing
        with pytest.raises(RuntimeError, match="SRR003"):
            validate_manifest(manifest, downloaded)

    def test_error_message_reports_count(self):
        manifest = {"SRR001", "SRR002", "SRR003"}
        with pytest.raises(RuntimeError, match="2 accessions missing"):
            validate_manifest(manifest, _fake_paths(["SRR001"]))

    def test_extra_files_in_dir_are_ignored(self):
        # Extra files that aren't in the manifest should not cause an error
        manifest = {"SRR001", "SRR002"}
        downloaded = _fake_paths(["SRR001", "SRR002", "SRR999_extra"])
        validate_manifest(manifest, downloaded)  # must not raise

    def test_empty_manifest_with_empty_dir_raises_nothing(self):
        validate_manifest(set(), [])  # must not raise

    def test_all_missing_raises(self):
        manifest = {"SRR001", "SRR002"}
        with pytest.raises(RuntimeError):
            validate_manifest(manifest, [])

    def test_single_missing_accession(self):
        manifest = {"SRR001"}
        with pytest.raises(RuntimeError, match="SRR001"):
            validate_manifest(manifest, [])
