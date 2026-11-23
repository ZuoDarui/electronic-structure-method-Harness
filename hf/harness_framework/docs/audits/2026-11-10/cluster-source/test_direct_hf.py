import unittest
from unittest.mock import patch

import numpy as np
from pyscf import gto, scf

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

    def test_cosx_integrals_do_not_cover_every_shell(self):
        mol = make_mol('H 0 0 0; H 0 0 1', 'sto-3g')
        self.assertGreater(mol.nbas, 1)
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

        solver = Hartree_Fock(mol)
        with patch.object(hf.moleintor, 'getints', side_effect=record_getints):
            solver.build_cosx_k(
                dm, grid_rad=10, grid_ang=50, batch_size=128, cosx_tol=1e-10)
        self.assertTrue(slices)
        full = (0, mol.nbas, 0, mol.nbas)
        self.assertNotIn(full, slices)
        self.assertEqual(solver.cosx_full_tensor_calls, 0)

    def test_get_jk_matches_separate_ri_j_and_cosx(self):
        mol = make_mol('H 0 0 0; H 0 0 1', 'sto-3g')
        dm = scf.RHF(mol).get_init_guess()
        separate = Hartree_Fock(mol)
        reference_j = separate.build_ri_j(dm, auxbasis='weigend', df_tol=1e-9)
        reference_k = separate.build_cosx_k(
            dm, grid_rad=10, grid_ang=50, batch_size=128, cosx_tol=1e-10)
        combined_j, combined_k = Hartree_Fock(mol).get_jk(
            dm, jk_mode='rijcosx', auxbasis='weigend', df_tol=1e-9,
            grid_rad=10, grid_ang=50, batch_size=128, cosx_tol=1e-10)
        direct_j, direct_k = Hartree_Fock(mol).get_jk(dm, jk_mode='direct')
        analytic_j, analytic_k = Hartree_Fock(mol).build_jk(dm)
        np.testing.assert_allclose(combined_j, reference_j, atol=2e-8, rtol=2e-8)
        np.testing.assert_allclose(combined_k, reference_k, atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(direct_j, analytic_j, atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(direct_k, analytic_k, atol=2e-10, rtol=2e-10)

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
                direct = Hartree_Fock(mol)
                j = direct.build_ri_j(dm, auxbasis='weigend', df_tol=1e-8)
                ref, _ = df.DF(mol, auxbasis='weigend').get_jk(
                    dm, with_j=True, with_k=False)
                np.testing.assert_allclose(j, ref, atol=2e-8, rtol=2e-8)

    def test_ri_j_reuses_prepared_auxiliary_data(self):
        mol = make_mol('H 0 0 0; H 0 0 1', 'sto-3g')
        solver = Hartree_Fock(mol)
        dm = scf.RHF(mol).get_init_guess()
        solver.build_ri_j(dm, auxbasis='weigend')
        prepared = solver._ri_cache
        solver.build_ri_j(1.1 * dm, auxbasis='weigend')
        self.assertIs(solver._ri_cache, prepared)

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
        solver = Hartree_Fock(mol)
        k = solver.build_cosx_k(dm, grid_rad=10, grid_ang=50,
                                batch_size=128, cosx_tol=1e-10)
        self.assertTrue(np.isfinite(k).all())
        np.testing.assert_allclose(k, k.T, atol=1e-10)


    def test_cosx_k_tracks_analytic_k_on_small_h2(self):
        mol = make_mol('H 0 0 0; H 0 0 1', 'sto-3g')
        dm = scf.RHF(mol).get_init_guess()
        solver = Hartree_Fock(mol)
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
        serial = Hartree_Fock(mol, parallel_backend='serial')
        serial_j = serial.build_ri_j(dm, auxbasis='weigend')
        serial_k = serial.build_cosx_k(dm, grid_rad=10, grid_ang=50,
                                       batch_size=256)
        for workers in (2, 4):
            with self.subTest(workers=workers):
                solver = Hartree_Fock(
                    mol, parallel_backend='cpu', parallel_workers=workers,
                    parallel_min_shells=1, parallel_chunk_size=4)
                parallel_j = solver.build_ri_j(dm, auxbasis='weigend')
                parallel_k = solver.build_cosx_k(dm, grid_rad=10, grid_ang=50,
                                                 batch_size=256)
                np.testing.assert_allclose(parallel_j, serial_j, atol=2e-8, rtol=2e-8)
                np.testing.assert_allclose(parallel_k, serial_k, atol=2e-10, rtol=2e-10)

    def test_cosx_k_tracks_analytic_k_for_water(self):
        mol = make_mol(
            'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', 'sto-3g')
        dm = scf.RHF(mol).get_init_guess()
        solver = Hartree_Fock(mol)
        k = solver.build_cosx_k(dm, grid_rad=30, grid_ang=110,
                                batch_size=512)
        _, ref = scf.RHF(mol).get_jk(mol, dm, with_j=False, with_k=True)
        np.testing.assert_allclose(k, ref, atol=5e-5, rtol=5e-5)

    def test_rijcosx_scf_runs_and_matches_df_reference(self):
        mol = make_mol('H 0 0 0; H 0 0 1', 'sto-3g')
        hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        overlap = mol.intor('int1e_ovlp')
        energy, dm = Hartree_Fock(mol).solve_rhf(
            mol.nelec, hcore, overlap, mol, mol.energy_nuc(), tol=1e-8,
            jk_mode='rijcosx', auxbasis='weigend',
            grid_rad=10, grid_ang=50, final_grid_rad=12, final_grid_ang=74,
            batch_size=128)
        self.assertTrue(np.isfinite(energy))
        self.assertTrue(np.isfinite(dm).all())


if __name__ == '__main__':
    unittest.main()
