"""Independent numerical checks for the repaired Coulomb operator."""
import unittest
from unittest.mock import patch
import numpy as np
from pyscf import df, gto, lib, scf
from hierarchical_ri import HierarchicalMatrix, HierarchicalRIJ, HierarchicalRIError


class HierarchicalMatrixTests(unittest.TestCase):
    def test_transpose_shares_compressed_factors(self):
        rng = np.random.default_rng(24)
        rows = rng.normal(size=(80, 3)); cols = rng.normal(size=(70, 3)) + 40
        u, v = rng.normal(size=(80, 3)), rng.normal(size=(3, 70))
        dense = u @ v
        matrix = HierarchicalMatrix(rows, cols, lambda i, j: dense[np.ix_(i, j)],
                                    leaf_size=12, tolerance=1e-12)
        x, y = rng.normal(size=70), rng.normal(size=80)
        self.assertGreater(matrix.stats['low_rank_blocks'], 0)
        np.testing.assert_allclose(matrix.matvec(x), dense @ x, atol=2e-11)
        np.testing.assert_allclose(matrix.matvec(y, transpose=True), dense.T @ y, atol=2e-11)
        self.assertAlmostEqual(y @ matrix.matvec(x), x @ matrix.matvec(y, transpose=True), places=10)

    def test_symmetric_hmatrix_and_transpose(self):
        xyz = np.c_[np.arange(40) * 3., np.zeros((40, 2))]
        dense = 1. / np.sqrt(1. + (xyz[:, 0, None] - xyz[None, :, 0])**2)
        matrix = HierarchicalMatrix(xyz, xyz, lambda i, j: dense[np.ix_(i, j)],
                                    symmetric=True, leaf_size=8)
        x = np.sin(np.arange(40))
        np.testing.assert_allclose(matrix.matvec(x), dense @ x, atol=2e-11)
        np.testing.assert_allclose(matrix.matvec(x, transpose=True), dense @ x, atol=2e-11)

    def test_holdout_probes_detect_rank_beyond_initial_probe_subspace(self):
        rng = np.random.default_rng(921)
        rows = rng.normal(size=(128, 3)); cols = rng.normal(size=(128, 3)) + 50
        dense = rng.normal(size=(128, 40)) @ rng.normal(size=(40, 128))
        matrix = HierarchicalMatrix(rows, cols, lambda i, j: dense[np.ix_(i, j)],
                                    tolerance=1e-12, max_rank=64)
        x = rng.normal(size=128)
        np.testing.assert_allclose(matrix.matvec(x), dense @ x, atol=2e-9, rtol=0)
        self.assertGreaterEqual(matrix.stats['maximum_rank'], 40)

    def test_budget_failure_has_no_dense_fallback(self):
        xyz = np.zeros((4, 3))
        with self.assertRaises(HierarchicalRIError):
            HierarchicalMatrix(xyz, xyz, lambda i, j: np.eye(4)[np.ix_(i, j)], memory_bytes=8)


class HierarchicalRIJTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lib.num_threads(1)

    @staticmethod
    def molecule(atom, basis='sto-3g'):
        return gto.M(atom=atom, basis=basis, verbose=0)

    @staticmethod
    def reference(mol, dm):
        return df.df_jk.get_j(df.DF(mol, auxbasis='weigend'), dm, hermi=1)

    def test_zero_density_block_does_not_delete_coulomb_element(self):
        mol = self.molecule('H 0 0 0; H 0 0 1')
        dm = np.eye(2)
        solver = HierarchicalRIJ(mol)
        j = solver.build(dm)
        self.assertAlmostEqual(j[0, 1], .618671094988539, places=11)
        np.testing.assert_allclose(j, self.reference(mol, dm), atol=1e-10, rtol=0)

    def test_random_density_linearity_reciprocity_and_reference(self):
        mol = self.molecule('O 0 0 0; H 0 .757 .587; H 0 -.757 .587', 'cc-pvdz')
        rng = np.random.default_rng(37)
        a, b = rng.normal(size=(2, mol.nao_nr(), mol.nao_nr()))
        a, b = (a + a.T) / 2, (b + b.T) / 2
        solver = HierarchicalRIJ(mol)
        ja, jb = solver.build(a), solver.build(b)
        np.testing.assert_allclose(ja, self.reference(mol, a), atol=1e-8, rtol=0)
        np.testing.assert_allclose(solver.build(2 * a - .3 * b), 2 * ja - .3 * jb, atol=1e-8, rtol=0)
        self.assertAlmostEqual(float(np.sum(a * jb)), float(np.sum(b * ja)), places=8)
        psd = a @ a.T
        self.assertGreater(float(np.sum(psd * solver.build(psd))), 0)

    def test_packed_b_and_metric_match_independent_integrals(self):
        mol = self.molecule('H 0 0 0; H 0 0 1', 'cc-pvdz')
        solver = HierarchicalRIJ(mol).prepare()
        raw = df.incore.aux_e2(mol, solver.auxmol, intor='int3c2e', aosym='s1')
        b = raw[solver.pair_i, solver.pair_j]
        metric = solver.auxmol.intor('int2c2e')
        rng = np.random.default_rng(31)
        c = rng.normal(size=solver.auxmol.nao_nr())
        d = rng.normal(size=len(solver.pair_i))
        np.testing.assert_allclose(solver.three_center.matvec(c), b @ c, atol=1e-11, rtol=0)
        np.testing.assert_allclose(solver.three_center.matvec(d, transpose=True), b.T @ d, atol=1e-11, rtol=0)
        np.testing.assert_allclose(solver.metric.matvec(c), metric @ c, atol=1e-11, rtol=0)

    def test_multidomain_ethane_solver_matches_global_df(self):
        from make_alkane_xyz import alkane_coordinates
        mol = gto.M(atom=alkane_coordinates(2), basis='sto-3g', verbose=0)
        dm = scf.hf.init_guess_by_minao(mol)
        solver = HierarchicalRIJ(mol, options={'pcg_maxiter': 100})
        j = solver.build(dm)
        self.assertGreater(solver.stats['preconditioner']['domains'], 1)
        np.testing.assert_allclose(j, self.reference(mol, dm), atol=1e-8, rtol=0)

    def test_separated_monomers_keep_long_range_coulomb(self):
        atom = 'O 0 0 0; H 0 .757 .587; H 0 -.757 .587'
        one = self.molecule(atom)
        two = self.molecule(atom + '; O 0 0 30; H 0 .757 30.587; H 0 -.757 30.587')
        d1 = scf.hf.init_guess_by_minao(one)
        dm = np.zeros((two.nao_nr(), two.nao_nr())); n = one.nao_nr()
        dm[:n, :n] = d1; dm[n:, n:] = d1
        solver = HierarchicalRIJ(two, options={'leaf_size': 16})
        j = solver.build(dm)
        np.testing.assert_allclose(j, self.reference(two, dm), atol=1e-8, rtol=0)
        self.assertGreater(np.max(abs(j[:n, :n] - self.reference(one, d1))), .01)

    def test_translation_invariance(self):
        mol = self.molecule('H 0 0 0; H 0 0 1', '6-31g')
        moved = self.molecule('H 11 -8 5; H 11 -8 6', '6-31g')
        dm = scf.hf.init_guess_by_minao(mol)
        np.testing.assert_allclose(HierarchicalRIJ(mol).build(dm),
                                   HierarchicalRIJ(moved).build(dm), atol=1e-9, rtol=0)

    def test_curvature_failure_is_rejected(self):
        mol = self.molecule('H 0 0 0; H 0 0 1')
        solver = HierarchicalRIJ(mol).prepare()
        with patch.object(solver, '_scaled_metric', side_effect=lambda x: -x):
            with self.assertRaisesRegex(HierarchicalRIError, 'curvature'):
                solver._solve(np.ones(solver.auxmol.nao_nr()))

    def test_nonfinite_density_is_rejected(self):
        solver = HierarchicalRIJ(self.molecule('H 0 0 0; H 0 0 1'))
        with self.assertRaisesRegex(ValueError, 'finite'):
            solver.build(np.full((2, 2), np.nan))

    def test_memory_budget_rejects_before_unbounded_factor_allocation(self):
        solver = HierarchicalRIJ(self.molecule('H 0 0 0; H 0 0 1'), max_memory_mb=.001)
        with self.assertRaisesRegex(HierarchicalRIError, 'budget|bytes'):
            solver.build(np.eye(2))

    def test_exact_integral_audit_detects_consistent_but_wrong_metric(self):
        # Multiplying M by a constant preserves symmetry, positive definiteness,
        # PCG convergence and d.J == gamma.c. The independent integral audit
        # must nevertheless reject this physically wrong Coulomb result.
        original = HierarchicalMatrix.matvec
        def corrupted(operator, vector, transpose=False):
            value = original(operator, vector, transpose=transpose)
            return 1.001 * value if operator.symmetric else value
        solver = HierarchicalRIJ(self.molecule('H 0 0 0; H 0 0 1'))
        with patch.object(HierarchicalMatrix, 'matvec', new=corrupted):
            with self.assertRaisesRegex(HierarchicalRIError, 'metric sample residual'):
                solver.build(np.eye(2))

    def test_rotation_invariance_for_s_orbitals(self):
        coords = np.array([[0., 0., 0.], [0., 0., 1.], [0., 2., 0.], [1., 0., 0.]])
        angle = .713
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                             [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        mol = gto.M(atom=[('H', r) for r in coords], basis='sto-3g', verbose=0)
        moved = gto.M(atom=[('H', r) for r in coords @ rotation.T], basis='sto-3g', verbose=0)
        dm = scf.hf.init_guess_by_minao(mol)
        np.testing.assert_allclose(HierarchicalRIJ(mol).build(dm),
                                   HierarchicalRIJ(moved).build(dm), atol=1e-9, rtol=0)


if __name__ == '__main__':
    unittest.main()
