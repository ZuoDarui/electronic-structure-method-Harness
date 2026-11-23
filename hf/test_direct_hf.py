import unittest
from unittest.mock import patch

import numpy as np
from pyscf import gto, scf
from scipy.linalg import eigh

import hf
from hf import Hartree_Fock


def make_mol(atom, basis):
    return gto.M(atom=atom, basis=basis, unit='Angstrom', verbose=0)


class DirectHFTests(unittest.TestCase):
    def test_schwarz_matches_pairwise_shell_quartets(self):
        mol = make_mol('H 0 0 0; H 0 0 1', '6-31g')
        solver = Hartree_Fock(mol)
        expected_pairs = mol.nbas * (mol.nbas + 1) // 2
        self.assertEqual(solver.schwarz_pairs_evaluated, expected_pairs)
        for ish in range(mol.nbas):
            for jsh in range(ish + 1):
                eri = mol.intor(
                    'int2e_sph',
                    shls_slice=(ish, ish + 1, jsh, jsh + 1,
                                ish, ish + 1, jsh, jsh + 1),
                )
                reference = np.sqrt(np.max(np.abs(eri))) if eri.size else 0.0
                self.assertAlmostEqual(solver.schwarz_ints[ish, jsh], reference, places=12)
                self.assertEqual(
                    solver.schwarz_ints[ish, jsh], solver.schwarz_ints[jsh, ish])

    def test_cosx_does_not_materialize_grid_ao_ao_tensor(self):
        mol = make_mol('H 0 0 0; H 0 0 1', 'cc-pvtz')
        dm = scf.RHF(mol).get_init_guess()
        slices = []
        real_getints = hf.moleintor.getints

        def record_getints(*args, **kwargs):
            shls = kwargs.get('shls_slice')
            if shls is None and len(args) >= 5:
                shls = args[4]
            if args and 'grids' in str(args[0]):
                slices.append(tuple(shls[:4]))
            return real_getints(*args, **kwargs)

        solver = Hartree_Fock(mol, jk_engine='hierarchical_rijcosx')
        with patch.object(hf.moleintor, 'getints', side_effect=record_getints):
            matrix = solver.build_cosx_k(
                dm, grid_rad=10, grid_ang=50, batch_size=128, cosx_tol=1e-10)
        self.assertGreater(np.max(np.abs(matrix)),0.1)
        self.assertTrue(solver.last_k_stats['shared_fused_integral_kernel'])
        self.assertEqual(slices, [])
        self.assertEqual(solver.cosx_full_tensor_calls, 0)

    def test_get_jk_matches_separate_ri_j_and_cosx(self):
        mol = make_mol('H 0 0 0; H 0 0 1', 'sto-3g')
        dm = scf.RHF(mol).get_init_guess()
        separate = Hartree_Fock(mol, jk_engine='hierarchical_rijcosx')
        reference_j = separate.build_ri_j(dm, auxbasis='weigend', df_tol=1e-9)
        reference_k = separate.build_cosx_k(
            dm, grid_rad=10, grid_ang=50, batch_size=128, cosx_tol=1e-10)
        combined = Hartree_Fock(mol, jk_engine='hierarchical_rijcosx')
        combined_j, combined_k = combined.get_jk(
            dm, jk_mode='rijcosx', auxbasis='weigend', df_tol=1e-9,
            grid_rad=10, grid_ang=50, batch_size=128, cosx_tol=1e-10)
        direct_j, direct_k = Hartree_Fock(mol).get_jk(dm, jk_mode='direct')
        analytic_j, analytic_k = Hartree_Fock(mol).build_jk(dm)
        np.testing.assert_allclose(combined_j, reference_j, atol=2e-8, rtol=2e-8)
        np.testing.assert_allclose(combined_k, reference_k, atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(direct_j, analytic_j, atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(direct_k, analytic_k, atol=2e-10, rtol=2e-10)

    def test_iteration_rijcosx_times_j_k_and_eigh_without_scf(self):
        mol = make_mol('H 0 0 0; H 0 0 1', 'sto-3g')
        dm = np.asarray(scf.RHF(mol).get_init_guess(), dtype=np.float64)
        hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        overlap = mol.intor('int1e_ovlp')
        kwargs = dict(
            auxbasis='weigend', df_tol=1e-9, grid_rad=10, grid_ang=50,
            batch_size=128, cosx_tol=1e-10, ao_tol=1e-10,
        )
        # Check attribution with a controlled clock: cluster scheduling and
        # first-call LAPACK startup must not decide a unit test's outcome.
        solver = Hartree_Fock(mol, jk_engine='hierarchical_rijcosx', dense_backend='cpu')
        real_j = solver.build_ri_j
        real_k = solver.build_cosx_k
        real_eigh = hf.eigh
        clock = [0.0]

        def slow_j(*args, **kw):
            value = real_j(*args, **kw)
            clock[0] += 0.05
            return value

        def slow_k(*args, **kw):
            value = real_k(*args, **kw)
            clock[0] += 0.15
            return value

        def timed_eigh(*args, **kw):
            value = real_eigh(*args, **kw)
            clock[0] += 0.01
            return value

        with patch.object(hf.time, 'perf_counter', side_effect=lambda: clock[0]), \
                patch.object(hf, 'eigh', side_effect=timed_eigh), \
                patch.object(solver, 'build_ri_j', side_effect=slow_j), \
                patch.object(solver, 'build_cosx_k', side_effect=slow_k), \
                patch.object(
                    Hartree_Fock, 'solve_rhf',
                    side_effect=AssertionError('solve_rhf must not run')):
            result = solver.iteration_rijcosx(
                dm, hcore, overlap, mol.nelec[0], mol.nelectron, **kwargs)
            repeated = solver.iteration_rijcosx(
                dm, hcore, overlap, mol.nelec[0], mol.nelectron, **kwargs)

        reference = Hartree_Fock(mol, jk_engine='hierarchical_rijcosx')
        timings = {}
        ref_j, ref_k = reference.get_jk(dm, jk_mode='rijcosx', timings=timings, **kwargs)
        direct_timings = {}
        reference.get_jk(dm, jk_mode='direct', timings=direct_timings)
        self.assertEqual(direct_timings, {})
        self.assertGreater(timings['j_seconds'], 0.0)
        self.assertGreater(timings['k_seconds'], 0.0)
        np.testing.assert_allclose(result['j_matrix'], ref_j, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(result['k_matrix'], ref_k, atol=1e-12, rtol=1e-12)
        for measured in (result, repeated):
            self.assertAlmostEqual(measured['j_seconds'], 0.05)
            self.assertAlmostEqual(measured['k_seconds'], 0.15)
            self.assertAlmostEqual(measured['eigh_seconds'], 0.01)
        self.assertTrue(result['all_finite'])
        self.assertLess(result['j_asymmetry'], 1e-8)
        self.assertLess(result['k_asymmetry'], 1e-8)
        self.assertLess(result['density_asymmetry'], 1e-8)
        self.assertLess(result['density_electron_abs'], 1e-8)
        self.assertEqual(result['cosx_full_tensor_calls'], 0)
        fock = hcore + result['j_matrix'] - 0.5 * result['k_matrix']
        energy = 0.5 * np.einsum('mn,mn->', hcore + fock, dm)
        self.assertAlmostEqual(result['electronic_energy'], float(energy))
        _, coeff = eigh(fock, overlap)
        occupied = coeff[:, :mol.nelec[0]]
        density = 2.0 * np.einsum('ma,na->mn', occupied, occupied.conj())
        np.testing.assert_allclose(result['density'], density, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(
            repeated['density'], result['density'], atol=1e-12, rtol=1e-12)
        with self.assertRaises(ValueError):
            solver.iteration_rijcosx(
                dm[:1], hcore, overlap, mol.nelec[0], mol.nelectron, **kwargs)

    def test_jk_matches_pyscf_for_h2_and_water(self):
        for atom, basis in [
            ('H 0 0 0; H 0 0 1', '6-31g'),
            ('O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', 'sto-3g'),
        ]:
            with self.subTest(atom=atom, basis=basis):
                mol = make_mol(atom, basis)
                rng = np.random.default_rng(7)
                x = rng.normal(size=(mol.nao_nr(), mol.nao_nr()))
                dm = x @ x.T * 0.05
                direct_j, direct_k = Hartree_Fock(mol).build_jk(dm)
                mf = scf.RHF(mol)
                ref_j, ref_k = mf.get_jk(mol, dm)
                np.testing.assert_allclose(direct_j, ref_j, atol=2e-10, rtol=2e-10)
                np.testing.assert_allclose(direct_k, ref_k, atol=2e-10, rtol=2e-10)

    def test_rhf_energy_and_density_match_pyscf(self):
        for atom, basis in [
            ('H 0 0 0; H 0 0 1', '6-31g'),
            ('O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', 'sto-3g'),
        ]:
            with self.subTest(atom=atom, basis=basis):
                mol = make_mol(atom, basis)
                hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
                overlap = mol.intor('int1e_ovlp')
                solver = Hartree_Fock(mol)
                energy, dm = solver.solve_rhf(mol.nelec, hcore, overlap, mol,
                                               mol.energy_nuc(), tol=1e-9)
                mf = scf.RHF(mol)
                mf.conv_tol = 1e-12
                mf.kernel()
                self.assertAlmostEqual(energy, mf.e_tot, places=8)
                np.testing.assert_allclose(dm, mf.make_rdm1(), atol=2e-7, rtol=2e-7)

    def test_user_density_is_used_as_initial_guess(self):
        mol = make_mol('H 0 0 0; H 0 0 1', 'sto-3g')
        hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        overlap = mol.intor('int1e_ovlp')
        supplied = np.eye(mol.nao_nr()) * 0.2

        class RecordingSolver(Hartree_Fock):
            def __init__(self, mol):
                self.seen = []
                super().__init__(mol)

            def build_jk(self, dm):
                if not self.seen:
                    self.seen.append(dm.copy())
                return super().build_jk(dm)

        solver = RecordingSolver(mol)
        solver.solve_rhf(mol.nelec, hcore, overlap, mol, mol.energy_nuc(), dm_init=supplied)
        np.testing.assert_array_equal(solver.seen[0], supplied)

    def test_open_shell_is_rejected(self):
        mol = gto.M(atom='H 0 0 0', basis='sto-3g', spin=1, verbose=0)
        solver = Hartree_Fock(mol)
        hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        overlap = mol.intor('int1e_ovlp')
        with self.assertRaisesRegex(ValueError, 'closed-shell'):
            solver.solve_rhf(mol.nelec, hcore, overlap, mol, mol.energy_nuc())


    def test_incremental_fock_matches_full_fock_for_h2_and_water(self):
        for atom, basis in [
            ('H 0 0 0; H 0 0 1', '6-31g'),
            ('O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', 'sto-3g'),
        ]:
            with self.subTest(atom=atom, basis=basis):
                mol = make_mol(atom, basis)
                solver = Hartree_Fock(mol)
                hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
                rng = np.random.default_rng(13)
                x = rng.normal(size=(mol.nao_nr(), mol.nao_nr()))
                dm_old = x @ x.T * 0.1
                y = rng.normal(size=(mol.nao_nr(), mol.nao_nr()))
                delta_dm = (y + y.T) * 0.02
                dm_new = dm_old + delta_dm

                j_old, k_old = solver.build_jk(dm_old)
                fock_old = hcore + j_old - 0.5 * k_old
                fock_incremental = solver.build_fock_incremental(fock_old, delta_dm)
                j_new, k_new = solver.build_jk(dm_new)
                fock_full = hcore + j_new - 0.5 * k_new
                np.testing.assert_allclose(fock_incremental, fock_full,
                                           atol=2e-10, rtol=2e-10)

    def test_density_screen_receives_density_increment(self):
        mol = make_mol('H 0 0 0; H 0 0 1', 'sto-3g')
        delta_dm = np.eye(mol.nao_nr()) * 0.03

        class RecordingSolver(Hartree_Fock):
            def build_jk(self, dm):
                self.last_dm = np.asarray(dm).copy()
                return super().build_jk(dm)

        solver = RecordingSolver(mol)
        fock_prev = np.zeros((mol.nao_nr(), mol.nao_nr()))
        solver.build_fock_incremental(fock_prev, delta_dm)
        np.testing.assert_array_equal(solver.last_dm, delta_dm)

    def test_periodic_full_fock_rebuild_and_default_interval(self):
        mol = make_mol('H 0 0 0; H 0 0 1', '6-31g')
        hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        overlap = mol.intor('int1e_ovlp')

        class RecordingSolver(Hartree_Fock):
            def __init__(self, mol):
                self.build_inputs = []
                super().__init__(mol)

            def build_jk(self, dm):
                self.build_inputs.append(np.asarray(dm).copy())
                return super().build_jk(dm)

        solver = RecordingSolver(mol)
        energy, dm = solver.solve_rhf(
            mol.nelec, hcore, overlap, mol, mol.energy_nuc(),
            incremental_rebuild_interval=2,
        )
        overlap_electrons = [np.trace(d @ overlap) for d in solver.build_inputs]
        full_density_calls = sum(abs(value - sum(mol.nelec)) < 1e-6
                                 for value in overlap_electrons)
        self.assertGreaterEqual(full_density_calls, 2)

        default_solver = Hartree_Fock(mol)
        default_energy, _ = default_solver.solve_rhf(
            mol.nelec, hcore, overlap, mol, mol.energy_nuc())
        mf = scf.RHF(mol)
        mf.kernel()
        self.assertAlmostEqual(default_energy, mf.e_tot, places=8)
        self.assertAlmostEqual(energy, mf.e_tot, places=8)
        np.testing.assert_allclose(dm, mf.make_rdm1(), atol=2e-7, rtol=2e-7)

    def test_incremental_rebuild_interval_must_be_positive(self):
        mol = make_mol('H 0 0 0; H 0 0 1', 'sto-3g')
        hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        overlap = mol.intor('int1e_ovlp')
        with self.assertRaisesRegex(ValueError, 'positive'):
            Hartree_Fock(mol).solve_rhf(
                mol.nelec, hcore, overlap, mol, mol.energy_nuc(),
                incremental_rebuild_interval=0,
            )


    def test_ri_j_matches_pyscf_density_fitting(self):
        from pyscf import df
        for atom, basis in [
            ('H 0 0 0; H 0 0 1', 'sto-3g'),
            ('O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', 'sto-3g'),
        ]:
            with self.subTest(atom=atom):
                mol = make_mol(atom, basis)
                rng = np.random.default_rng(21)
                x = rng.normal(size=(mol.nao_nr(), mol.nao_nr()))
                dm = x @ x.T * 0.05
                direct = Hartree_Fock(mol, jk_engine='hierarchical_rijcosx')
                j = direct.build_ri_j(dm, auxbasis='weigend', df_tol=1e-8)
                ref, _ = df.DF(mol, auxbasis='weigend').get_jk(
                    dm, with_j=True, with_k=False)
                np.testing.assert_allclose(j, ref, atol=2e-8, rtol=2e-8)

    def test_ri_j_reuses_prepared_auxiliary_data(self):
        mol = make_mol('H 0 0 0; H 0 0 1', 'sto-3g')
        solver = Hartree_Fock(mol, jk_engine='hierarchical_rijcosx')
        dm = scf.RHF(mol).get_init_guess()
        solver.build_ri_j(dm, auxbasis='weigend')
        prepared = solver._hierarchical_j_engines['weigend']
        self.assertIsNotNone(prepared)
        solver.build_ri_j(1.1 * dm, auxbasis='weigend')
        self.assertIs(solver._hierarchical_j_engines['weigend'], prepared)

    def test_rijcosx_requires_explicit_auxbasis(self):
        mol = make_mol('H 0 0 0; H 0 0 1', 'sto-3g')
        hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        overlap = mol.intor('int1e_ovlp')
        with self.assertRaisesRegex(ValueError, 'auxbasis'):
            Hartree_Fock(mol).solve_rhf(
                mol.nelec, hcore, overlap, mol, mol.energy_nuc(),
                jk_mode='rijcosx')

    def test_cosx_k_is_symmetric_and_finite(self):
        mol = make_mol('H 0 0 0; H 0 0 1', 'sto-3g')
        dm = scf.RHF(mol).get_init_guess()
        solver = Hartree_Fock(mol, jk_engine='hierarchical_rijcosx')
        k = solver.build_cosx_k(dm, grid_rad=10, grid_ang=50,
                                batch_size=128, cosx_tol=1e-10)
        self.assertTrue(np.isfinite(k).all())
        np.testing.assert_allclose(k, k.T, atol=1e-10)


    def test_cosx_k_tracks_analytic_k_on_small_h2(self):
        mol = make_mol('H 0 0 0; H 0 0 1', 'sto-3g')
        dm = scf.RHF(mol).get_init_guess()
        solver = Hartree_Fock(mol, jk_engine='hierarchical_rijcosx')
        k = solver.build_cosx_k(dm, grid_rad=20, grid_ang=110,
                                batch_size=256, cosx_tol=1e-10)
        _, ref = scf.RHF(mol).get_jk(mol, dm, with_j=False, with_k=True)
        np.testing.assert_allclose(k, ref, atol=1e-4, rtol=1e-4)

    def test_cpu_parallel_jk_matches_serial_and_pyscf(self):
        mol = make_mol(
            'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', 'sto-3g')
        dm = scf.RHF(mol).get_init_guess()
        serial_j, serial_k = Hartree_Fock(mol, parallel_backend='serial').build_jk(dm)
        ref_j, ref_k = scf.RHF(mol).get_jk(mol, dm)
        np.testing.assert_allclose(serial_j, ref_j, atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(serial_k, ref_k, atol=2e-10, rtol=2e-10)
        for workers in (2, 4):
            with self.subTest(workers=workers):
                solver = Hartree_Fock(
                    mol, parallel_backend='cpu', parallel_workers=workers)
                parallel_j, parallel_k = solver.build_jk(dm)
                self.assertEqual(parallel_j.shape, (mol.nao_nr(), mol.nao_nr()))
                self.assertTrue(np.isfinite(parallel_j).all())
                self.assertTrue(np.isfinite(parallel_k).all())
                np.testing.assert_allclose(parallel_j, serial_j, atol=2e-10, rtol=2e-10)
                np.testing.assert_allclose(parallel_k, serial_k, atol=2e-10, rtol=2e-10)

    def test_parallel_backend_and_worker_validation(self):
        mol = make_mol('H 0 0 0; H 0 0 1', 'sto-3g')
        with self.assertRaisesRegex(ValueError, 'parallel_backend'):
            Hartree_Fock(mol, parallel_backend='unknown')
        with self.assertRaisesRegex(ValueError, 'parallel_workers'):
            Hartree_Fock(mol, parallel_backend='cpu', parallel_workers=0)
        with self.assertRaisesRegex(ValueError, 'parallel_workers'):
            Hartree_Fock(mol, parallel_backend='cpu', parallel_workers=True)
        with self.assertRaisesRegex(ValueError, 'parallel_chunk_size'):
            Hartree_Fock(mol, parallel_backend='cpu', parallel_chunk_size=0)
        with self.assertRaisesRegex(ValueError, 'parallel_min_shells'):
            Hartree_Fock(mol, parallel_backend='cpu', parallel_min_shells=0)

    def test_small_cpu_system_falls_back_to_serial_map(self):
        mol = make_mol('H 0 0 0; H 0 0 1', 'sto-3g')
        solver = Hartree_Fock(
            mol, parallel_backend='cpu', parallel_workers=4,
            parallel_min_shells=12)
        self.assertFalse(solver.parallel_active)

    def test_chunked_cpu_map_matches_unchunked_result(self):
        mol = make_mol(
            'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', 'sto-3g')
        dm = scf.RHF(mol).get_init_guess()
        reference = Hartree_Fock(
            mol, parallel_backend='cpu', parallel_workers=2,
            parallel_chunk_size=1, parallel_min_shells=1).build_jk(dm)
        chunked = Hartree_Fock(
            mol, parallel_backend='cpu', parallel_workers=2,
            parallel_chunk_size=8, parallel_min_shells=1).build_jk(dm)
        np.testing.assert_allclose(chunked[0], reference[0], atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(chunked[1], reference[1], atol=2e-10, rtol=2e-10)

    def test_cpu_parallel_ri_j_and_cosx_match_serial(self):
        mol = make_mol(
            'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', 'sto-3g')
        dm = scf.RHF(mol).get_init_guess()
        serial = Hartree_Fock(
            mol, parallel_backend='serial', jk_engine='hierarchical_rijcosx')
        serial_j = serial.build_ri_j(dm, auxbasis='weigend')
        serial_k = serial.build_cosx_k(dm, grid_rad=10, grid_ang=50,
                                       batch_size=256)
        for workers in (2, 4):
            with self.subTest(workers=workers):
                solver = Hartree_Fock(
                    mol, parallel_backend='cpu', parallel_workers=workers,
                    parallel_min_shells=1, parallel_chunk_size=4,
                    jk_engine='hierarchical_rijcosx')
                parallel_j = solver.build_ri_j(dm, auxbasis='weigend')
                parallel_k = solver.build_cosx_k(dm, grid_rad=10, grid_ang=50,
                                                 batch_size=256)
                np.testing.assert_allclose(parallel_j, serial_j, atol=2e-8, rtol=2e-8)
                np.testing.assert_allclose(parallel_k, serial_k, atol=2e-10, rtol=2e-10)

    def test_cosx_k_tracks_analytic_k_for_water(self):
        mol = make_mol(
            'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', 'sto-3g')
        dm = scf.RHF(mol).get_init_guess()
        solver = Hartree_Fock(mol, jk_engine='hierarchical_rijcosx')
        k = solver.build_cosx_k(dm, grid_rad=30, grid_ang=110,
                                batch_size=512)
        _, ref = scf.RHF(mol).get_jk(mol, dm, with_j=False, with_k=True)
        np.testing.assert_allclose(k, ref, atol=5e-5, rtol=5e-5)

    def test_distant_pairs_leave_the_neighbor_list(self):
        mol = make_mol('H 0 0 0; H 0 0 80', 'sto-3g')
        solver = Hartree_Fock(mol, jk_engine='hierarchical_rijcosx')
        self.assertFalse(solver.coverage_complete)
        self.assertEqual(sorted(map(tuple, solver.neighbor_pairs)), [(0, 0), (1, 1)])
        self.assertEqual(solver.schwarz_ints[0, 1], 0.0)
        self.assertGreater(solver.schwarz_ints[0, 0], 0.0)
        dm = scf.RHF(mol).get_init_guess()
        j_matrix, k_matrix = solver.get_jk(
            dm, jk_mode='rijcosx', auxbasis='weigend', df_tol=1e-9,
            grid_rad=15, grid_ang=50, batch_size=64, cosx_tol=1e-10)
        self.assertTrue(np.isfinite(j_matrix).all())
        self.assertTrue(np.isfinite(k_matrix).all())
        np.testing.assert_allclose(j_matrix, j_matrix.T, atol=1e-10)
        np.testing.assert_allclose(k_matrix, k_matrix.T, atol=1e-10)
        self.assertEqual(solver.cosx_full_tensor_calls, 0)

    def test_separated_water_ri_j_includes_other_monomer(self):
        monomer = 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587'
        dimer = (
            monomer + '; O 0 0 30; H 0 -0.757 30.587; H 0 0.757 30.587')
        one = make_mol(monomer, 'sto-3g')
        mol = make_mol(dimer, 'sto-3g')
        dm_one = np.asarray(scf.RHF(one).get_init_guess())
        dm = np.asarray(scf.RHF(mol).get_init_guess())
        monomer_j = Hartree_Fock(
            one, jk_engine='hierarchical_rijcosx').build_ri_j(dm_one, 'weigend', 1e-9)
        solver = Hartree_Fock(mol, jk_engine='hierarchical_rijcosx')
        self.assertFalse(solver.coverage_complete)
        dimer_j = solver.build_ri_j(dm, 'weigend', 1e-9)
        nao = one.nao_nr()
        from pyscf.df import df_jk
        from pyscf.df.df import DF
        reference = df_jk.get_j(DF(mol, auxbasis='weigend'), dm, hermi=1)
        np.testing.assert_allclose(dimer_j, reference, atol=1e-8, rtol=0)
        self.assertGreater(np.max(np.abs(dimer_j[:nao, :nao] - monomer_j)), 0.01)
        np.testing.assert_allclose(dimer_j[:nao, nao:], 0.0, atol=1e-8)
        self.assertTrue(np.isfinite(dimer_j).all())

    def test_rijcosx_scf_runs_and_matches_df_reference(self):
        mol = make_mol('H 0 0 0; H 0 0 1', 'sto-3g')
        hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        overlap = mol.intor('int1e_ovlp')
        for engine in ('hierarchical_rijcosx', 'pyscf_reference'):
            with self.subTest(engine=engine):
                energy, dm = Hartree_Fock(mol, jk_engine=engine).solve_rhf(
                    mol.nelec, hcore, overlap, mol, mol.energy_nuc(), tol=1e-8,
                    jk_mode='rijcosx', auxbasis='weigend',
                    grid_rad=10, grid_ang=50, final_grid_rad=12, final_grid_ang=74,
                    batch_size=128)
                self.assertTrue(np.isfinite(energy))
                self.assertTrue(np.isfinite(dm).all())

    def test_df_j_and_sgx_k_match_pyscf_kernels(self):
        from pyscf.df import df_jk
        from pyscf.sgx import sgx, sgx_jk

        mol = make_mol(
            'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', 'cc-pvdz')
        dm = np.asarray(scf.hf.init_guess_by_minao(mol), dtype=np.float64)
        solver = Hartree_Fock(mol)
        j_matrix = solver.build_df_j(dm, 'weigend')
        k_matrix = solver.build_sgx_k(dm, 'weigend', 20, 86)
        engine = sgx.SGX(mol, auxbasis='weigend')
        engine.verbose = 0
        engine.build()
        engine.grids = hf.matching_cosx_grids(mol, 20, 86)
        ref_j = np.asarray(df_jk.get_j(engine, dm, hermi=1))
        ref_k = np.asarray(sgx_jk.get_k_only(engine, dm, hermi=1))
        np.testing.assert_allclose(j_matrix, ref_j, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(k_matrix, ref_k, atol=1e-8, rtol=1e-8)
        first = solver._sgx_engines[('weigend', 20, 86, True)]
        again = solver.build_sgx_k(dm, 'weigend', 20, 86)
        self.assertIs(solver._sgx_engines[('weigend', 20, 86, True)], first)
        np.testing.assert_allclose(again, k_matrix, atol=1e-12, rtol=1e-12)

    def test_sgx_engine_and_schwarz_are_built_once(self):
        mol = make_mol('H 0 0 0; H 0 0 1', 'sto-3g')
        dm = np.asarray(scf.hf.init_guess_by_minao(mol), dtype=np.float64)
        solver = Hartree_Fock(mol)
        solver.build_sgx_k(dm, 'weigend', 10, 50)
        engine = solver._sgx_engines[('weigend', 10, 50, True)]
        grids = engine.grids
        bounds = engine._pjs_data
        solver.build_sgx_k(1.1 * dm, 'weigend', 10, 50)
        self.assertIs(solver._sgx_engines[('weigend', 10, 50, True)], engine)
        self.assertIs(engine.grids, grids)
        self.assertIs(engine._pjs_data, bounds)

    def test_incremental_sgx_k_matches_full_k(self):
        mol = make_mol(
            'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', 'sto-3g')
        dm = np.asarray(scf.hf.init_guess_by_minao(mol), dtype=np.float64)
        delta = 0.05 * dm
        solver = Hartree_Fock(mol)
        full = solver.build_sgx_k(dm + delta, 'weigend', 15, 50)
        incremental = solver.build_sgx_k(dm, 'weigend', 15, 50)
        incremental = incremental + solver.build_sgx_k(
            delta, 'weigend', 15, 50, screen_dm=dm + delta)
        np.testing.assert_allclose(incremental, full, atol=1e-8, rtol=1e-8)

    def test_hierarchical_ri_j_matches_global_df_for_distant_fragments(self):
        monomer = 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587'
        dimer = (
            monomer + '; O 0 0 15; H 0 -0.757 15.587; H 0 0.757 15.587')
        mol = make_mol(dimer, 'sto-3g')
        dm = np.asarray(scf.hf.init_guess_by_minao(mol), dtype=np.float64)
        solver = Hartree_Fock(mol, jk_engine='hierarchical_rijcosx')
        self.assertFalse(solver.coverage_complete)
        from pyscf.df import df_jk
        from pyscf.df.df import DF
        local = solver.build_ri_j(dm, 'weigend', 1e-8)
        ref = np.asarray(df_jk.get_j(DF(mol, auxbasis='weigend'), dm, hermi=1))
        np.testing.assert_allclose(local, ref, atol=1e-8, rtol=0)
        np.testing.assert_allclose(
            solver.build_df_j(dm, 'weigend'), ref, atol=1e-8, rtol=1e-8)

    def test_rijcosx_stops_on_energy_and_gradient(self):
        mol = make_mol('H 0 0 0; H 0 0 1', 'sto-3g')
        hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        overlap = mol.intor('int1e_ovlp')
        results = {}
        for engine in ('hierarchical_rijcosx', 'pyscf_reference'):
            results[engine] = Hartree_Fock(mol, jk_engine=engine).solve_rhf(
                mol.nelec, hcore, overlap, mol, mol.energy_nuc(), tol=1e-8,
                jk_mode='rijcosx', auxbasis='weigend', incremental_k=True,
                grid_rad=15, grid_ang=50, final_grid_rad=15, final_grid_ang=50,
                kfull_every=5,
            )
        mf = scf.RHF(mol).COSX(auxbasis='weigend')
        mf.conv_tol = 1e-8
        mf.conv_tol_grad = 1e-8
        mf.direct_scf = False
        mf.max_cycle = 40
        mf.init_guess = 'minao'
        mf.diis_space = 8
        mf.verbose = 0
        mf.with_df.dfj = True
        mf.with_df.grids_level_i = 1
        mf.with_df.grids_level_f = 1
        original_build = mf.with_df.build

        def build_same_grid(level=None):
            original_build(level)
            mf.with_df.grids = hf.matching_cosx_grids(mol, 15, 50)

        mf.with_df.build = build_same_grid
        self.assertTrue(mf.kernel() is not None and mf.converged)
        for engine, (energy, dm) in results.items():
            with self.subTest(engine=engine):
                self.assertAlmostEqual(energy, mf.e_tot, places=8)
                self.assertTrue(np.isfinite(dm).all())

    def test_thread_budget_rejects_oversubscribe(self):
        self.assertEqual(hf._check_thread_budget(1, 8, 48), 8)
        with self.assertRaisesRegex(ValueError, 'cpu_cores'):
            hf._check_thread_budget(4, 16, 48)

    def test_available_log_path_keeps_the_existing_file(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / 'water8_ccpvtz'
            first = hf._available_log_path(prefix)
            self.assertEqual(first, prefix.with_suffix('.log'))
            first.write_text('old\n')
            second = hf._available_log_path(prefix)
            self.assertEqual(second.name, 'water8_ccpvtz_2.log')
            second.write_text('new\n')
            third = hf._available_log_path(prefix)
            self.assertEqual(third.name, 'water8_ccpvtz_3.log')
            self.assertEqual(first.read_text(), 'old\n')

    def test_job_memory_budget_uses_a_quarter_of_the_node(self):
        self.assertEqual(hf.job_max_memory_mb({'memory_gib': 256}), 65536.0)
        self.assertEqual(hf.job_max_memory_mb({}), hf._DEFAULT_MAX_MEMORY_MB)
        self.assertGreater(hf._DEFAULT_MAX_MEMORY_MB, 2000)
        with self.assertRaisesRegex(ValueError, 'memory_gib'):
            hf.job_max_memory_mb({'memory_gib': 0})

    def test_sgx_engine_uses_solver_memory_and_target_grid(self):
        from pyscf.sgx import sgx_jk

        mol = make_mol('H 0 0 0; H 0 0 1', 'sto-3g')
        dm = np.asarray(scf.hf.init_guess_by_minao(mol), dtype=np.float64)
        solver = Hartree_Fock(mol, max_memory=12345)
        solver.build_df_j(dm, 'weigend')
        solver.build_sgx_k(dm, 'weigend', 10, 50)
        df_engine = solver._df_engines['weigend']
        engine = solver._sgx_engines[('weigend', 10, 50, True)]
        self.assertEqual(df_engine.max_memory, 12345)
        self.assertEqual(engine.max_memory, 12345)
        self.assertEqual(engine.grids.atom_grid['H'], (10, 50))
        self.assertIsNotNone(engine._opt)
        self.assertIsNotNone(engine._pjs_data)
        ngrids, blksize = solver._last_k_grid
        self.assertEqual(ngrids, engine.grids.coords.shape[0])
        self.assertGreaterEqual(blksize, int(sgx_jk.SGX_BLKSIZE))
        self.assertIn(f'ngrids={ngrids} blksize={blksize}', solver._k_grid_note())

    def test_rms_gradient_accepts_logged_alkane_within_twenty_iterations(self):
        """Replay the C21 and C86 logs through the RMS cutoff.

        The recorded Frobenius norms are the old convergence measure.
        Dividing by the occupied-virtual count is what solve_rhf now
        compares with 1e-8. The coarse energy at that iteration is
        already within 1e-6 Ha of the converged coarse energy.
        """

        def first_accepting(records, nocc, nvir, converged_energy):
            scale = np.sqrt(nocc * nvir)
            for iteration, frobenius, energy in records:
                vector = np.full(nocc * nvir, frobenius / scale)
                rms = hf.orbital_gradient_rms(vector, nocc, nvir)
                if rms < 1e-8 and abs(energy - converged_energy) < 1e-6:
                    return iteration
            return None

        c21 = [
            (8, 1.711e-5, -821.160333446179),
            (9, 5.254e-6, -821.160333446279),
            (10, 1.286e-6, -821.160333446286),
            (15, 9.475e-9, -821.160333446284),
        ]
        c86 = [
            (8, 2.630e-5, -3359.261815096056),
            (9, 8.819e-6, -3359.261815096300),
            (14, 2.623e-8, -3359.261815096303),
            (15, 2.037e-8, -3359.261815096330),
        ]
        c21_iteration = first_accepting(c21, 85, 1161, -821.160333446284)
        c86_iteration = first_accepting(c86, 345, 4671, -3359.261815096330)
        self.assertIsNotNone(c21_iteration)
        self.assertLessEqual(c21_iteration, 20)
        self.assertIsNotNone(c86_iteration)
        self.assertLessEqual(c86_iteration, 20)
        self.assertAlmostEqual(
            hf.orbital_gradient_rms(np.ones(85 * 1161), 85, 1161),
            1.0,
        )


if __name__ == '__main__':
    unittest.main()
