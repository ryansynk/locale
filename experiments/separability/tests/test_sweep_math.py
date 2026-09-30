"""The sweep arithmetic, checked against hand-computed answers.

Every verdict in report.md is a function of these numbers, so they are checked
against values worked out by hand rather than against a second implementation
that could share a bug with the first.
"""

import numpy as np
import pytest

import embed as emb
import run_probe as _rp
import sweep as sw


# --------------------------------------------------------------------------- #
# grouping / max-pooling
# --------------------------------------------------------------------------- #


class TestGrouping:
    def test_group_starts_contiguous(self):
        owner = np.array([0, 0, 1, 2, 2, 2])
        assert emb.group_starts(owner).tolist() == [0, 2, 3]

    def test_group_starts_rejects_unsorted(self):
        with pytest.raises(ValueError, match="sorted"):
            emb.group_starts(np.array([0, 1, 0]))

    def test_group_starts_rejects_empty_group(self):
        # owner 1 owns nothing; a silent reduceat here would attribute owner 2's
        # windows to owner 1.
        with pytest.raises(ValueError, match="no empty groups"):
            emb.group_starts(np.array([0, 2, 2]))

    def test_max_over_groups_axis1(self):
        sim = np.array([[0.1, 0.9, 0.3], [0.5, 0.2, 0.8]])
        out = emb.max_over_groups(sim, np.array([0, 0, 1]), axis=1)
        np.testing.assert_allclose(out, [[0.9, 0.3], [0.5, 0.8]])

    def test_max_pool_matches_bruteforce(self):
        rng = np.random.default_rng(0)
        sim = rng.random((7, 5))
        row_owner = np.array([0, 0, 1, 1, 1, 2, 2])
        col_owner = np.array([0, 1, 1, 2, 2])
        got = emb.max_pool_similarity(sim, row_owner, col_owner)
        want = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                want[i, j] = sim[np.ix_(row_owner == i, col_owner == j)].max()
        np.testing.assert_allclose(got, want)

    def test_tile_many_owner_is_contiguous(self):
        seqs = ["A" * 100, "C" * 600, "G" * 300]
        windows, owner = emb.tile_many(seqs)
        assert len(windows) == len(owner)
        emb.group_starts(owner)  # must not raise
        assert owner.tolist() == sorted(owner.tolist())

    def test_tile_passthrough_and_stride(self):
        # <= one window: whole, exactly as DenseIndex._iter_chunks does.
        short = "ACGT" * 10
        assert emb.tile(short) == [short]
        # longer: stride windows of exactly WINDOW bp, ragged tail dropped
        long = "ACGT" * 200  # 800 bp
        w = emb.tile(long)
        assert all(len(x) == emb.WINDOW for x in w)
        step = emb.WINDOW - emb.OVERLAP
        assert len(w) == (len(long) - emb.WINDOW) // step + 1


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #


class TestSweep:
    def _simple(self):
        # 2 queries x 4 targets; targets 0,1 carriers, 2,3 confusables.
        sim = np.array([
            [0.9, 0.7, 0.3, 0.1],
            [0.8, 0.4, 0.6, 0.2],
        ])
        is_carrier = np.array([True, True, False, False])
        return sim, is_carrier

    def test_capture_counts_at_a_radius(self):
        sim, is_carrier = self._simple()
        radii = np.array([0.0, 0.5, 0.75, 0.95])
        c = sw.sweep(sim, is_carrier, radii=radii)
        # query 0 carriers {0.9, 0.7}: >=0 -> 2/2, >=0.5 -> 2/2, >=0.75 -> 1/2,
        # >=0.95 -> 0/2
        np.testing.assert_allclose(c.carrier_per_query[0], [1.0, 1.0, 0.5, 0.0])
        # query 1 carriers {0.8, 0.4}
        np.testing.assert_allclose(c.carrier_per_query[1], [1.0, 0.5, 0.5, 0.0])
        # query 1 confusables {0.6, 0.2}
        np.testing.assert_allclose(c.confusable_per_query[1], [1.0, 0.5, 0.0, 0.0])

    def test_radius_boundary_is_inclusive(self):
        # "within r" must mean >= r, not > r: a radius read off the curve has to
        # capture the pair that sits exactly on it.
        sim = np.array([[0.5, 0.2]])
        c = sw.sweep(sim, np.array([True, False]), radii=np.array([0.5]))
        assert c.carrier_per_query[0, 0] == 1.0

    def test_self_mask_removes_pair_and_shrinks_denominator(self):
        sim, is_carrier = self._simple()
        mask = np.zeros_like(sim, dtype=bool)
        mask[0, 0] = True  # query 0 is target 0
        c = sw.sweep(sim, is_carrier, mask, radii=np.array([0.0, 0.75]))
        # query 0 now has one valid carrier (0.7): 1/1 then 0/1
        np.testing.assert_allclose(c.carrier_per_query[0], [1.0, 0.0])
        # query 1 untouched
        np.testing.assert_allclose(c.carrier_per_query[1], [1.0, 0.5])

    def test_pooled_sims_exclude_masked_pairs(self):
        sim, is_carrier = self._simple()
        mask = np.zeros_like(sim, dtype=bool)
        mask[0, 0] = True
        c = sw.sweep(sim, is_carrier, mask)
        assert 0.9 not in set(c.carrier_sims.tolist())
        assert len(c.carrier_sims) == 3

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            sw.sweep(np.zeros((2, 4)), np.array([True, False]))


# --------------------------------------------------------------------------- #
# AUROC
# --------------------------------------------------------------------------- #


class TestAuroc:
    def test_perfect_separation(self):
        assert sw.auroc(np.array([0.9, 0.8]), np.array([0.2, 0.1])) == 1.0

    def test_reversed(self):
        assert sw.auroc(np.array([0.1, 0.2]), np.array([0.8, 0.9])) == 0.0

    def test_all_ties_is_half(self):
        # A method that returns a constant must score 0.5, not 1.0. This is the
        # case that catches a strict-inequality rank implementation.
        assert sw.auroc(np.full(5, 0.3), np.full(7, 0.3)) == 0.5

    def test_known_value(self):
        # pos {3, 1}, neg {2, 0}: pairs (3>2, 3>0, 1<2, 1>0) -> 3/4
        assert sw.auroc(np.array([3.0, 1.0]), np.array([2.0, 0.0])) == 0.75

    def test_empty_is_nan(self):
        assert np.isnan(sw.auroc(np.array([]), np.array([1.0])))


# --------------------------------------------------------------------------- #
# band
# --------------------------------------------------------------------------- #


class TestBand:
    def _curves(self, carrier, confusable, radii):
        return sw.CaptureCurves(
            radii=radii,
            carrier_per_query=np.asarray(carrier)[None, :],
            confusable_per_query=np.asarray(confusable)[None, :],
            n_queries=1, n_carriers=1, n_confusables=1,
            carrier_sims=np.array([1.0]), confusable_sims=np.array([0.0]),
        )

    def test_window_exists_when_bands_are_separated(self):
        radii = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        # carriers stay captured out to 0.75; confusables gone by 0.25
        c = self._curves([1.0, 1.0, 1.0, 0.95, 0.0],
                         [1.0, 0.0, 0.0, 0.0, 0.0], radii)
        b = sw.band(c, tau=0.9, alpha=0.01)
        assert b.hi == 0.75
        assert b.lo == 0.25
        assert b.margin == pytest.approx(0.5)
        assert b.exists

    def test_no_window_when_bands_interleave(self):
        radii = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        # confusables only vanish at 0.75, by which point carriers are gone
        c = self._curves([1.0, 0.95, 0.4, 0.0, 0.0],
                         [1.0, 0.9, 0.5, 0.0, 0.0], radii)
        b = sw.band(c, tau=0.9, alpha=0.01)
        assert b.hi == 0.25
        assert b.lo == 0.75
        assert b.margin < 0
        assert not b.exists

    def test_margin_is_nan_when_carriers_never_reach_tau(self):
        radii = np.array([0.0, 0.5, 1.0])
        c = self._curves([0.5, 0.2, 0.0], [0.0, 0.0, 0.0], radii)
        b = sw.band(c, tau=0.9, alpha=0.01)
        assert np.isnan(b.hi)
        assert not b.exists


# --------------------------------------------------------------------------- #
# f_j
# --------------------------------------------------------------------------- #


class TestFj:
    def test_fj_is_prevalence_times_capture(self):
        radii = np.array([0.0, 1.0])
        c = sw.CaptureCurves(
            radii=radii,
            carrier_per_query=np.array([[1.0, 0.5]]),
            confusable_per_query=np.array([[0.4, 0.0]]),
            n_queries=1, n_carriers=1, n_confusables=1,
        )
        a = sw.FjAssumptions(element_prevalence=0.2,
                             confusable_prevalence=0.5, source="test")
        out = sw.fj_curve(c, a)
        np.testing.assert_allclose(out["f_j"], [0.2, 0.1])
        np.testing.assert_allclose(out["f_contamination"], [0.2, 0.0])
        # purity at r=0: 0.2 / (0.2 + 0.2) = 0.5; at r=1 no contamination
        np.testing.assert_allclose(out["purity"], [0.5, 1.0])

    def test_fragmented_feature_has_small_fj(self):
        # The failure mode the probe exists to detect: a feature that captures
        # 5% of carriers has 5% of the frequency, and no test has power on it.
        radii = np.array([0.0])
        c = sw.CaptureCurves(
            radii=radii,
            carrier_per_query=np.array([[0.05]]),
            confusable_per_query=np.array([[0.0]]),
            n_queries=1, n_carriers=1, n_confusables=1,
        )
        a = sw.FjAssumptions(0.2, 0.5, "test")
        assert sw.fj_curve(c, a)["f_j"][0] == pytest.approx(0.01)

    def test_cluster_point_matches_curve_arithmetic(self):
        a = sw.FjAssumptions(0.2, 0.5, "test")
        pt = sw.fj_for_cluster(0.5, 0.0, a)
        assert pt["f_j"] == pytest.approx(0.1)
        assert pt["purity"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# carrier scopes -- the fix for multi-gene elements
# --------------------------------------------------------------------------- #


class TestAggregateToEntities:
    def test_max_over_an_entitys_columns(self):
        sim = np.array([[0.1, 0.9, 0.4], [0.5, 0.2, 0.8]])
        out = sw.aggregate_to_entities(sim, np.array([0, 0, 1]), 2)
        np.testing.assert_allclose(out, [[0.9, 0.4], [0.5, 0.8]])

    def test_dropped_columns_are_ignored(self):
        sim = np.array([[0.1, 0.99, 0.4]])
        # column 1 has entity -1 and must not contribute
        out = sw.aggregate_to_entities(sim, np.array([0, -1, 1]), 2)
        np.testing.assert_allclose(out, [[0.1, 0.4]])


class TestScopes:
    def _fixture(self):
        # 4 references: clbB, clbA (both carriers, different genes),
        # irp1, entF (confusables). Two accessions.
        gene_key = ["clbb", "clba", "irp1", "entf"]
        acc = ["ACC1", "ACC1", "ACC2", "ACC2"]
        is_carrier = np.array([True, True, False, False])
        qsrc = np.array([0])                  # one query, derived from clbB
        base = _rp.self_mask_from(qsrc, 4)
        return gene_key, acc, is_carrier, qsrc, base

    def test_same_gene_masks_other_carrier_genes(self):
        gene_key, acc, is_carrier, qsrc, base = self._fixture()
        scope = _rp._same_gene_scope(gene_key, is_carrier, qsrc, base)
        sim = np.array([[1.0, 0.8, 0.3, 0.2]])
        s, car, mask = scope(sim)
        # clbB (self) and clbA (different carrier gene) are both masked;
        # the confusables stay in scope.
        assert mask[0].tolist() == [True, True, False, False]
        assert car.tolist() == [True, True, False, False]

    def test_same_gene_capture_has_no_other_gene_penalty(self):
        # With clbA masked out, a clbB query that finds nothing else has an
        # empty carrier denominator rather than a 0/1 miss it could never fix.
        gene_key, acc, is_carrier, qsrc, base = self._fixture()
        scope = _rp._same_gene_scope(gene_key, is_carrier, qsrc, base)
        sim = np.array([[1.0, 0.8, 0.3, 0.2]])
        s, car, mask = scope(sim)
        curves = sw.sweep(s, car, mask)
        # no in-scope carriers at all -> the curve is flat zero, not a penalty
        assert curves.carrier_sims.size == 0

    def test_element_scope_counts_accessions(self):
        gene_key, acc, is_carrier, qsrc, base = self._fixture()
        scope = _rp._element_scope(acc, is_carrier, qsrc, base)
        sim = np.array([[1.0, 0.8, 0.3, 0.2]])
        s, car, mask = scope(sim)
        # ACC1 holds carriers -> carrier entity; ACC2 holds only confusables
        assert s.shape == (1, 2)
        assert car.tolist() == [True, False]
        # entity score is the max over that accession's genes
        np.testing.assert_allclose(s, [[1.0, 0.3]])

    def test_accession_with_both_is_a_carrier_not_a_confusable(self):
        # An accession carrying pks AND ybt is a true carrier: retrieving it is
        # not contamination, and counting it as such would invent a false
        # positive rate out of nothing.
        # gene_key is irrelevant to the element scope, which keys on accession.
        acc = ["ACC1", "ACC1"]
        is_carrier = np.array([True, False])
        qsrc = np.array([0])
        base = _rp.self_mask_from(qsrc, 2)
        scope = _rp._element_scope(acc, is_carrier, qsrc, base)
        s, car, mask = scope(np.array([[1.0, 0.9]]))
        assert s.shape == (1, 1)
        assert car.tolist() == [True]

    def test_element_scope_masks_the_querys_own_accession(self):
        gene_key, acc, is_carrier, qsrc, base = self._fixture()
        scope = _rp._element_scope(acc, is_carrier, qsrc, base)
        s, car, mask = scope(np.array([[1.0, 0.8, 0.3, 0.2]]))
        assert mask[0].tolist() == [True, False]


class TestLocaleWindowToGene:
    def _fixture(self, seed=0):
        rng = np.random.default_rng(seed)
        q = rng.normal(size=(23, 16)).astype(np.float32)
        t = rng.normal(size=(31, 16)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        owner = np.repeat(np.arange(7), [4, 5, 3, 6, 4, 7, 2])
        return q, t, owner

    def test_cpu_path_matches_bruteforce(self):
        q, t, owner = self._fixture()
        got = _rp.locale_window_to_gene(q, t, owner, device="cpu")
        want = np.zeros((q.shape[0], 7), dtype=np.float32)
        full = q @ t.T
        for g in range(7):
            want[:, g] = full[:, owner == g].max(axis=1)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-5)

    def test_blocking_does_not_change_the_answer(self):
        q, t, owner = self._fixture(1)
        a = _rp.locale_window_to_gene(q, t, owner, block=4, device="cpu")
        b = _rp.locale_window_to_gene(q, t, owner, block=1000, device="cpu")
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)

    @pytest.mark.skipif(
        not __import__("torch").cuda.is_available(), reason="no GPU"
    )
    def test_gpu_path_matches_cpu_path(self):
        # The GPU path uses index_reduce_ over gene ids instead of reduceat
        # over contiguous blocks; they must agree exactly.
        q, t, owner = self._fixture(2)
        cpu = _rp.locale_window_to_gene(q, t, owner, device="cpu")
        gpu = _rp.locale_window_to_gene(q, t, owner, device="cuda")
        np.testing.assert_allclose(cpu, gpu, rtol=0, atol=1e-5)
