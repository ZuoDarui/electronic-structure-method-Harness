"""Physical and routing regressions for the C172 RIJCOSX repair."""
import unittest
from unittest.mock import patch
import contextlib
import io
import json

import numpy as np
from pyscf import df, gto, lib, scf
from pyscf.df import df_jk

import hf
from hf import Hartree_Fock

lib.num_threads(2)


class RepairRegressionTests(unittest.TestCase):
    def test_zero_density_does_not_zero_coulomb_output(self):
        # The remote He makes the old automatic local branch active. The
        # H-H Coulomb element is nonzero although D[0,1] is exactly zero.
        mol = gto.M(atom='H 0 0 0; H 0 0 1; He 0 0 30', basis='sto-3g', verbose=0)
        density = np.eye(mol.nao_nr())
        expected = df_jk.get_j(df.DF(mol, auxbasis='weigend'), density, hermi=1)
        actual = Hartree_Fock(
            mol, jk_engine='hierarchical_rijcosx').build_ri_j(density, 'weigend')
        self.assertGreater(abs(expected[0, 1]), 0.1)
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)

    def test_distant_density_has_nonzero_coulomb_effect(self):
        mol = gto.M(atom='He 0 0 0; He 0 0 30', basis='sto-3g', verbose=0)
        density = np.diag([0., 2.])
        expected = df_jk.get_j(df.DF(mol, auxbasis='weigend'), density, hermi=1)
        actual = Hartree_Fock(
            mol, jk_engine='hierarchical_rijcosx').build_ri_j(density, 'weigend')
        self.assertGreater(expected[0, 0], 0.01)
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)

    def test_fine_grid_uses_selected_engine_and_reconverges(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='sto-3g', verbose=0)
        hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        overlap = mol.intor('int1e_ovlp')
        energies=[]
        for engine in ('hierarchical_rijcosx','pyscf_reference'):
            solver=Hartree_Fock(mol,jk_engine=engine)
            with contextlib.redirect_stdout(io.StringIO()):
                energy,density=solver.solve_rhf(
                    mol.nelec,hcore,overlap,mol,mol.energy_nuc(),
                    jk_mode='rijcosx',auxbasis='weigend',
                    grid_rad=10,grid_ang=50,final_grid_rad=15,final_grid_ang=74,
                    incremental_k=True,max_iteration=20,tol=1e-8)
            details=solver.last_scf_result
            self.assertEqual(details['jk_engine'],engine)
            self.assertEqual([s['grid'] for s in details['grid_stages']],[[10,50],[15,74]])
            self.assertTrue(all(s['iterations']>=2 for s in details['grid_stages']))
            self.assertLess(details['gradient_rms'],1e-8)
            self.assertAlmostEqual(float(np.einsum('ij,ji',density,overlap)),2.,places=8)
            energies.append(energy)
        self.assertLess(abs(energies[0]-energies[1]),1e-8)

    def test_fine_grid_cannot_reset_the_total_iteration_budget(self):
        mol=gto.M(atom='H 0 0 0; H 0 0 1',basis='sto-3g',verbose=0)
        solver=Hartree_Fock(mol,jk_engine='pyscf_reference')
        h=mol.intor('int1e_kin')+mol.intor('int1e_nuc')
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'budget'):
                solver.solve_rhf(mol.nelec,h,mol.intor('int1e_ovlp'),mol,mol.energy_nuc(),
                    jk_mode='rijcosx',auxbasis='weigend',max_iteration=2,
                    grid_rad=10,grid_ang=50,final_grid_rad=15,final_grid_ang=74)
        self.assertIsNone(solver.last_scf_result)

    def test_final_rebuild_rejects_nan_gradient(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='sto-3g', verbose=0)
        solver = Hartree_Fock(mol, jk_engine='pyscf_reference')
        hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        state = {'final': False}
        original_gradient = hf.orbital_gradient_rms

        def observe_stage(message, *args, **kwargs):
            if message == 'SCF final energy':
                state['final'] = True

        def gradient(*args, **kwargs):
            if state['final']:
                return float('nan')
            return original_gradient(*args, **kwargs)

        # All SCF iterations are physical. Only the final independent
        # verification reports an invalid residual: NaN must not count as
        # "less than the threshold" through a false >= comparison.
        with patch.object(hf, '_emit_progress', side_effect=observe_stage), \
                patch.object(hf, 'orbital_gradient_rms', side_effect=gradient):
            with self.assertRaisesRegex(RuntimeError, 'Final full J/K rebuild'):
                solver.solve_rhf(
                    mol.nelec, hcore, mol.intor('int1e_ovlp'), mol,
                    mol.energy_nuc(), jk_mode='rijcosx', auxbasis='weigend',
                    grid_rad=10, grid_ang=50, final_grid_rad=10,
                    final_grid_ang=50, max_iteration=30, tol=1e-8)
        self.assertTrue(state['final'])
        self.assertIsNone(solver.last_scf_result)

    def test_final_rebuild_rejects_energy_jump_with_zero_gradient(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='sto-3g', verbose=0)
        solver = Hartree_Fock(mol, jk_engine='pyscf_reference')
        hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        overlap = mol.intor('int1e_ovlp')
        state = {'final': False, 'gradient': None}
        original_j = solver._coulomb_matrix
        original_gradient = hf.orbital_gradient_rms

        def observe_stage(message, *args, **kwargs):
            if message == 'SCF final energy':
                state['final'] = True

        def coulomb(*args, **kwargs):
            result = original_j(*args, **kwargs)
            # A shift proportional to S leaves occupied/virtual mixing and
            # the orbital gradient unchanged but changes E by 0.5*shift*Ne.
            return result + 1e-3 * overlap if state['final'] else result

        def gradient(*args, **kwargs):
            value = original_gradient(*args, **kwargs)
            if state['final']:
                state['gradient'] = value
            return value

        with patch.object(hf, '_emit_progress', side_effect=observe_stage), \
                patch.object(solver, '_coulomb_matrix', side_effect=coulomb), \
                patch.object(hf, 'orbital_gradient_rms', side_effect=gradient):
            with self.assertRaisesRegex(RuntimeError, 'Final full J/K rebuild'):
                solver.solve_rhf(
                    mol.nelec, hcore, overlap, mol, mol.energy_nuc(),
                    jk_mode='rijcosx', auxbasis='weigend',
                    grid_rad=10, grid_ang=50, final_grid_rad=10,
                    final_grid_ang=50, max_iteration=30, tol=1e-8)
        self.assertTrue(state['final'])
        self.assertTrue(np.isfinite(state['gradient']))
        self.assertLess(state['gradient'], 1e-8)
        self.assertIsNone(solver.last_scf_result)


class PhysicalGuardTests(unittest.TestCase):
    def setUp(self):
        self.mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='sto-3g', verbose=0)
        self.hcore = self.mol.intor('int1e_kin') + self.mol.intor('int1e_nuc')
        self.overlap = self.mol.intor('int1e_ovlp')
        self.messages = []

    def solve(self, solver, **options):
        settings = dict(jk_mode='rijcosx', auxbasis='weigend', grid_rad=10,
                        grid_ang=50, final_grid_rad=10, final_grid_ang=50,
                        max_iteration=30, tol=1e-8)
        settings.update(options)
        record = lambda message, *args, **kwargs: self.messages.append(message)
        with patch.object(hf, '_emit_progress', side_effect=record):
            return solver.solve_rhf(self.mol.nelec, self.hcore, self.overlap,
                                    self.mol, self.mol.energy_nuc(), **settings)

    def energy_lines(self):
        return [m for m in self.messages if m.startswith('SCF iteration') and 'energy=' in m]

    def test_scaled_coulomb_is_rejected_at_initial_fock(self):
        solver = Hartree_Fock(self.mol)
        original = solver._coulomb_matrix
        calls = []

        def scaled(*args, **kwargs):
            calls.append(1)
            return 10.0 * original(*args, **kwargs)

        with patch.object(solver, '_coulomb_matrix', side_effect=scaled):
            with self.assertRaisesRegex(RuntimeError, 'at initial Fock: total energy E='):
                self.solve(solver)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.energy_lines(), [])
        self.assertIsNone(solver.last_scf_result)

    def test_scaled_coulomb_is_rejected_at_first_iteration(self):
        solver = Hartree_Fock(self.mol)
        original = solver._coulomb_matrix
        calls = []

        def scaled_after_guess(*args, **kwargs):
            calls.append(1)
            matrix = original(*args, **kwargs)
            return matrix if len(calls) == 1 else 10.0 * matrix

        with patch.object(solver, '_coulomb_matrix', side_effect=scaled_after_guess):
            with self.assertRaisesRegex(RuntimeError, 'at SCF iteration 1: total energy E='):
                self.solve(solver)
        self.assertEqual(len(calls), 2)
        self.assertTrue(any(m.startswith('SCF initial diagnostics') for m in self.messages))
        self.assertEqual(self.energy_lines(), [])

    def test_wrong_sign_exchange_is_rejected_at_initial_fock(self):
        solver = Hartree_Fock(self.mol)
        original = solver._exchange_matrix
        flipped = lambda *args, **kwargs: -original(*args, **kwargs)
        with patch.object(solver, '_exchange_matrix', side_effect=flipped):
            with self.assertRaisesRegex(RuntimeError, 'at initial Fock: exchange energy E_K='):
                self.solve(solver)
        self.assertEqual(self.energy_lines(), [])

    def test_wrong_electron_count_is_rejected_at_first_iteration(self):
        solver = Hartree_Fock(self.mol)
        real_solve = hf.DenseLinearAlgebra.solve

        def stretched(dense, fock, nocc):
            values, vectors, density = real_solve(dense, fock, nocc)
            return values, 1.001 * vectors, 1.001 ** 2 * density

        # Inject at the shared boundary so CPU LAPACK and CUDA receive the
        # same non-normalized density, regardless of their eigenvalue library.
        with patch.object(hf.DenseLinearAlgebra, 'solve', new=stretched):
            with self.assertRaisesRegex(RuntimeError, 'at SCF iteration 1: electron count'):
                self.solve(solver)
        self.assertEqual(self.energy_lines(), [])

    def test_wrong_exchange_on_final_grid_is_rejected(self):
        solver = Hartree_Fock(self.mol)
        original = solver._exchange_matrix

        def flipped_on_final_grid(dm, auxbasis, grid_rad, grid_ang, *args, **kwargs):
            matrix = original(dm, auxbasis, grid_rad, grid_ang, *args, **kwargs)
            return -matrix if (grid_rad, grid_ang) == (15, 74) else matrix

        with patch.object(solver, '_exchange_matrix', side_effect=flipped_on_final_grid):
            with self.assertRaisesRegex(RuntimeError, 'at initial Fock: exchange energy E_K='):
                self.solve(solver, final_grid_rad=15, final_grid_ang=74)
        self.assertTrue(any('switching to final grid' in m for m in self.messages))
        self.assertIsNone(solver.last_scf_result)

    def test_unnormalized_guess_passes_and_logs_initial_energy(self):
        guess = 0.9 * scf.hf.init_guess_by_minao(self.mol)
        count = float(np.einsum('ij,ji', guess, self.overlap))
        self.assertGreater(abs(count - self.mol.nelectron), 0.1)
        solver = Hartree_Fock(self.mol)
        self.solve(solver, dm_init=guess)
        initial = [json.loads(m[m.index('{'):]) for m in self.messages
                   if m.startswith('SCF initial diagnostics')]
        self.assertEqual(len(initial), 1)
        self.assertEqual(initial[0]['physical_guard'], 'passed')
        self.assertAlmostEqual(initial[0]['electron_count'], count, places=10)
        j = Hartree_Fock(self.mol).build_ri_j(guess, 'weigend')
        k = Hartree_Fock(self.mol).build_cosx_k(guess, 10, 50, cosx_tol=1e-13)
        expected = self.mol.energy_nuc() + float(np.einsum(
            'ij,ji', guess, self.hcore + 0.5 * j - 0.25 * k))
        self.assertAlmostEqual(initial[0]['energy_hartree'], expected, places=10)
        self.assertTrue(solver.last_scf_result['converged'])
        self.assertAlmostEqual(solver.last_scf_result['electron_count'], 2.0, places=8)


if __name__ == '__main__':
    unittest.main()
