"""The baseline machinery that is this probe's own code, not MMseqs2's.

`greedy_set_cover_clusters` and `catalog_capture` decide whether the verdict
reads "MMseqs2 matches LOCALE", so they get the same scrutiny as the sweep. The
parts that are genuinely MMseqs2 (the alignments) are not tested here -- they
are its problem, and the probe runs it permissively on purpose.
"""

import numpy as np
import pytest

import baselines as bl
import noise as nz
from run_probe import match_carrier_capture
from sweep import CaptureCurves


# --------------------------------------------------------------------------- #
# k-mers
# --------------------------------------------------------------------------- #


class TestKmers:
    def test_canonical_collapses_strands(self):
        fwd = bl.canonical_kmers("ACGTAC", k=3)
        rev = bl.canonical_kmers("GTACGT", k=3)  # reverse complement
        assert fwd == rev

    def test_non_acgt_kmers_dropped(self):
        # A k-mer spanning the N is dropped, the ones either side survive.
        assert bl.canonical_kmers("AAANAAA", k=3) == bl.canonical_kmers("AAA", k=3)

    def test_too_short_yields_nothing(self):
        assert bl.canonical_kmers("ACGT", k=31) == set()

    def test_containment_of_subsequence_is_one(self):
        target = "ACGTTGCAATCGGATCCGATTACAGGCATGC" * 3
        query = target[10:70]
        sim = bl.kmer_similarity([query], [target], k=31)
        assert sim[0, 0] == pytest.approx(1.0)

    def test_unrelated_sequences_score_zero(self):
        sim = bl.kmer_similarity(["A" * 60], ["C" * 60], k=31)
        assert sim[0, 0] == 0.0

    def test_containment_is_asymmetric_and_jaccard_is_not(self):
        # Non-repetitive on purpose: a periodic sequence has the same k-mer set
        # as its own prefix, which would make containment and Jaccard agree.
        long_seq = "TTTCCTCATGCAATTCAAAACCATGTCCGTAATGTAGGCGAAATAGTAAACCATTTTACGGAGGATACCAAATTCCTCCTTATTCAGGACCTAACCTGAGGTAAACCAGGTCTCTCCGCCCCCTTATAAAAGCTGTTGCACCTAGCCAAGTTCAACGGCAGCTGCAATGGAAATAGGCAATGACGGATATATATTAAAAA"
        short = long_seq[:80]
        cont = bl.kmer_similarity([short], [long_seq], k=31)[0, 0]
        jac = bl.kmer_similarity([short], [long_seq], k=31, mode="jaccard")[0, 0]
        assert cont == pytest.approx(1.0)
        assert jac < cont

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            bl.kmer_similarity(["ACGT"], ["ACGT"], mode="cosine")


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #


class TestGreedySetCover:
    def test_two_clean_cliques(self):
        ids = ["a", "b", "c", "d"]
        adj = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ], dtype=bool)
        m = bl.greedy_set_cover_clusters(ids, adj)
        assert m["a"] == m["b"]
        assert m["c"] == m["d"]
        assert m["a"] != m["c"]

    def test_singletons_get_their_own_cluster(self):
        ids = ["a", "b", "c"]
        m = bl.greedy_set_cover_clusters(ids, np.eye(3, dtype=bool))
        assert len({m[i] for i in ids}) == 3

    def test_every_id_is_assigned(self):
        rng = np.random.default_rng(0)
        ids = [f"s{i}" for i in range(20)]
        adj = rng.random((20, 20)) < 0.2
        adj |= adj.T
        m = bl.greedy_set_cover_clusters(ids, adj)
        assert set(m) == set(ids)

    def test_highest_degree_becomes_representative(self):
        # 'a' touches everything, so greedy set cover must pick it first and
        # absorb the whole graph into one cluster.
        ids = ["a", "b", "c", "d"]
        adj = np.eye(4, dtype=bool)
        adj[0, :] = True
        adj[:, 0] = True
        m = bl.greedy_set_cover_clusters(ids, adj)
        assert len({m[i] for i in ids}) == 1
        assert m["b"] == "a"


class TestCatalogCapture:
    def _setup(self):
        target_ids = ["c0", "c1", "c2", "x0", "x1"]
        is_carrier = np.array([True, True, True, False, False])
        # catalog: the three carriers cluster together, confusables apart
        mapping = {"c0": "c0", "c1": "c0", "c2": "c0", "x0": "x0", "x1": "x0"}
        return target_ids, is_carrier, mapping

    @staticmethod
    def _units(is_carrier):
        """One unit per target, so counting is per gene."""
        return np.arange(len(is_carrier)), np.asarray(is_carrier)

    def test_query_pools_with_its_cluster_mates(self):
        target_ids, is_carrier, mapping = self._setup()
        # one query, derived from c0, aligning perfectly to it
        assign = np.array([[1.0, 0.9, 0.9, 0.1, 0.1]])
        uot, uic = self._units(is_carrier)
        out = bl.catalog_capture(
            mapping, target_ids, assign, np.array([0]), 0.9, uot, uic
        )
        # c1 and c2 are the other carriers in the cluster; c0 itself excluded
        assert out["carrier_capture"] == pytest.approx(1.0)
        assert out["confusable_capture"] == 0.0
        assert out["assigned_fraction"] == 1.0

    def test_below_threshold_is_unassigned_and_scores_zero(self):
        # The fragmentation failure: a noisy read falls out of a tight catalog
        # entirely. It must score 0, not be dropped from the average.
        target_ids, is_carrier, mapping = self._setup()
        assign = np.array([[0.80, 0.7, 0.7, 0.1, 0.1]])
        uot, uic = self._units(is_carrier)
        out = bl.catalog_capture(
            mapping, target_ids, assign, np.array([0]), 0.95, uot, uic
        )
        assert out["carrier_capture"] == 0.0
        assert out["assigned_fraction"] == 0.0

    def test_fragmented_catalog_gives_partial_capture(self):
        target_ids, is_carrier, _ = self._setup()
        mapping = {"c0": "c0", "c1": "c0", "c2": "c2", "x0": "x0", "x1": "x0"}
        assign = np.array([[1.0, 0.9, 0.5, 0.1, 0.1]])
        uot, uic = self._units(is_carrier)
        out = bl.catalog_capture(
            mapping, target_ids, assign, np.array([0]), 0.9, uot, uic
        )
        # only c1 pools with c0, out of the two other carriers
        assert out["carrier_capture"] == pytest.approx(0.5)
        assert out["n_carrier_clusters"] == 2
        assert out["largest_carrier_cluster_share"] == pytest.approx(2 / 3)

    def test_confusable_contamination_is_counted(self):
        target_ids, is_carrier, _ = self._setup()
        mapping = {t: "c0" for t in target_ids}   # everything in one cluster
        assign = np.array([[1.0, 0.9, 0.9, 0.9, 0.9]])
        uot, uic = self._units(is_carrier)
        out = bl.catalog_capture(
            mapping, target_ids, assign, np.array([0]), 0.9, uot, uic
        )
        assert out["carrier_capture"] == pytest.approx(1.0)
        assert out["confusable_capture"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# head-to-head lookup
# --------------------------------------------------------------------------- #


class TestMatchCarrierCapture:
    def _curves(self, carrier, confusable):
        radii = np.linspace(0.0, 1.0, len(carrier))
        return CaptureCurves(
            radii=radii,
            carrier_per_query=np.asarray(carrier)[None, :],
            confusable_per_query=np.asarray(confusable)[None, :],
            n_queries=1, n_carriers=1, n_confusables=1,
        )

    def test_picks_first_radius_meeting_the_confusable_budget(self):
        c = self._curves([1.0, 0.9, 0.6, 0.3, 0.0],
                         [1.0, 0.5, 0.1, 0.0, 0.0])
        # budget 0.1 -> index 2 -> carrier 0.6
        assert match_carrier_capture(c, 0.1) == pytest.approx(0.6)

    def test_zero_budget_needs_zero_confusables(self):
        c = self._curves([1.0, 0.9, 0.6, 0.3, 0.0],
                         [1.0, 0.5, 0.1, 0.0, 0.0])
        assert match_carrier_capture(c, 0.0) == pytest.approx(0.3)

    def test_unreachable_budget_is_nan(self):
        c = self._curves([1.0, 0.5], [0.9, 0.8])
        assert np.isnan(match_carrier_capture(c, 0.1))


# --------------------------------------------------------------------------- #
# noise
# --------------------------------------------------------------------------- #


class TestNoise:
    SEQ = "ACGTTGCAATCGGATCCGATTACAGGCATGCTTAGCCA" * 8

    def test_zero_rate_is_identity(self):
        for conv in nz.CONVENTIONS:
            assert nz.mutate(self.SEQ, 0.0, conv) == self.SEQ

    def test_repo_convention_hits_target_identity(self):
        out = nz.mutate_repo(self.SEQ, 0.10)
        ident = nz.realized_identity(self.SEQ, out)
        # Augmenter pins the edit count, so identity lands on 0.90 closely.
        assert ident == pytest.approx(0.90, abs=0.03)

    def test_benchmark_convention_is_roughly_1_2x_divergence(self):
        # SNP rate r plus two indel rates of r/10 each.
        out = nz.mutate_benchmark(self.SEQ, 0.10)
        ident = nz.realized_identity(self.SEQ, out)
        assert 0.80 < ident < 0.95

    def test_deterministic_for_a_given_seed(self):
        for conv in nz.CONVENTIONS:
            a = nz.mutate(self.SEQ, 0.1, conv, salt=7)
            b = nz.mutate(self.SEQ, 0.1, conv, salt=7)
            assert a == b

    def test_different_salt_gives_different_sequence(self):
        a = nz.mutate(self.SEQ, 0.1, "repo", salt=1)
        b = nz.mutate(self.SEQ, 0.1, "repo", salt=2)
        assert a != b

    def test_output_alphabet_stays_acgt(self):
        for conv in nz.CONVENTIONS:
            assert set(nz.mutate(self.SEQ, 0.2, conv)) <= set("ACGT")

    def test_mutate_all_varies_per_sequence(self):
        seqs = [self.SEQ, self.SEQ]
        out = nz.mutate_all(seqs, 0.1, "repo", salt=0)
        # same input twice, different salt per position -> different outputs
        assert out[0] != out[1]

    def test_unknown_convention_raises(self):
        with pytest.raises(ValueError):
            nz.mutate(self.SEQ, 0.1, "made-up")

    def test_realized_identity_endpoints(self):
        assert nz.realized_identity("ACGT", "ACGT") == 1.0
        assert nz.realized_identity("ACGT", "ACGA") == pytest.approx(0.75)

    def test_repo_reports_its_own_exact_identity(self):
        # The Augmenter counts its own edits, so this number is exact rather
        # than recovered by an alignment -- and it must agree with one.
        out, ident = nz.mutate_repo_with_identity(self.SEQ, 0.10)
        assert out == nz.mutate_repo(self.SEQ, 0.10)
        assert ident == pytest.approx(
            nz.realized_identity(self.SEQ, out), abs=0.02
        )

    def test_repo_identity_at_zero_rate(self):
        out, ident = nz.mutate_repo_with_identity(self.SEQ, 0.0)
        assert out == self.SEQ and ident == 1.0

    def test_realized_identity_prefix_cap_is_unbiased(self):
        # Capping the DP at a prefix must not shift the estimate much, since
        # edits are i.i.d. along the sequence.
        long = self.SEQ * 8                      # ~2.4 kb
        out = nz.mutate_repo(long, 0.10)
        short_est = nz.realized_identity(long, out, max_len=600)
        full = nz.realized_identity(long, out, max_len=10_000)
        assert short_est == pytest.approx(full, abs=0.05)
