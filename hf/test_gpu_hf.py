"""Independent physical and failure checks for the portable RHF solver.

End-to-end reference coverage belongs to the harness candidate; these tests
exercise failure modes, a tabulated energy, and coordinate/block invariance.
"""

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
from pyscf import gto, lib, scf

from gpu_hf import GPUFriendlyRHF


lib.num_threads(1)


def molecule(atom, basis="sto-3g", **kwargs):
    return gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0, **kwargs)


class GPUFriendlyRHFTests(unittest.TestCase):
    def test_h2_known_sto3g_energy(self):
        mol = molecule("H 0 0 0; H 0 0 0.74")
        result = GPUFriendlyRHF(mol, backend="numpy").kernel()
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.energy, -1.1167593073964255, places=10)

    def test_fully_occupied_helium_has_no_virtual_orbitals(self):
        mol = molecule("He 0 0 0")
        result = GPUFriendlyRHF(mol, backend="numpy").kernel()
        reference = scf.RHF(mol).run(conv_tol=1e-12)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.energy, reference.e_tot, places=10)
        overlap = mol.intor("int1e_ovlp")
        self.assertAlmostEqual(np.trace(result.density @ overlap), 2.0, places=10)

    def test_exact_energy_is_invariant_under_rigid_motion(self):
        atoms = [("O", (0.0, 0.0, 0.0)),
                 ("H", (0.0, -0.757, 0.587)),
                 ("H", (0.0, 0.757, 0.587))]
        theta = 0.713
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                             [np.sin(theta), np.cos(theta), 0.0],
                             [0.0, 0.0, 1.0]])
        transformed = [(symbol, rotation @ np.asarray(coords) + [1.2, -0.8, 0.6])
                       for symbol, coords in atoms]
        first = GPUFriendlyRHF(molecule(atoms), backend="numpy").kernel()
        second = GPUFriendlyRHF(molecule(transformed), backend="numpy").kernel()
        self.assertTrue(first.converged and second.converged)
        self.assertAlmostEqual(first.energy, second.energy, places=9)

    def test_jk_accepts_indefinite_symmetric_density(self):
        mol = molecule("O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587")
        rng = np.random.default_rng(219)
        raw = rng.normal(size=(mol.nao_nr(), mol.nao_nr()))
        density = (raw + raw.T) * 0.07
        self.assertLess(np.linalg.eigvalsh(density).min(), 0.0)
        solver = GPUFriendlyRHF(mol, backend="numpy")
        actual = solver.build_jk(density)
        expected = scf.RHF(mol).get_jk(mol, density)
        for matrix, reference in zip(actual, expected):
            np.testing.assert_allclose(solver.to_numpy(matrix), reference,
                                       atol=2e-11, rtol=2e-11)

    def test_df_blocking_does_not_change_the_method(self):
        mol = molecule("Li 0 0 0; H 0 0 1.6")
        first = GPUFriendlyRHF(mol, backend="numpy", integral_mode="df",
                               df_block_size=1)
        second = GPUFriendlyRHF(mol, backend="numpy", integral_mode="df",
                                df_block_size=7)
        density = scf.RHF(mol).get_init_guess()
        for a, b in zip(first.build_jk(density), second.build_jk(density)):
            np.testing.assert_allclose(first.to_numpy(a), second.to_numpy(b),
                                       atol=2e-11, rtol=2e-11)
        first_result, second_result = first.kernel(), second.kernel()
        self.assertTrue(first_result.converged and second_result.converged)
        self.assertAlmostEqual(first_result.energy, second_result.energy, places=10)

    def test_one_cycle_does_not_claim_convergence_and_reports_its_density(self):
        mol = molecule("O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587")
        solver = GPUFriendlyRHF(mol, backend="numpy", max_cycle=1)
        result = solver.kernel()
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 1)
        hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        j, k = scf.RHF(mol).get_jk(mol, result.density)
        fock = hcore + j - 0.5 * k
        energy = 0.5 * np.sum(result.density * (hcore + fock)) + mol.energy_nuc()
        self.assertAlmostEqual(result.energy, energy, places=10)
        self.assertTrue(np.isfinite(result.energy))

    def test_rejected_final_canonicalization_keeps_the_last_consistent_state(self):
        mol = molecule("O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587")
        solver = GPUFriendlyRHF(mol, backend="numpy", max_cycle=1,
                                conv_tol=1e3, residual_tol=1e3)
        physical_error = solver._error
        errors_seen = []

        def reject_final_confirmation(fock, density):
            error = physical_error(fock, density)
            errors_seen.append(error)
            # Inject a failed final confirmation, after the ordinary DIIS
            # error and the newly constructed density have been evaluated.
            return np.full_like(error, 1e6) if len(errors_seen) == 3 else error

        with patch.object(solver, "_error", side_effect=reject_final_confirmation):
            result = solver.kernel()
        self.assertEqual(len(errors_seen), 3)
        self.assertFalse(result.converged)
        reconstructed = (result.mo_coeff * result.orbital_occupations) @ result.mo_coeff.T
        np.testing.assert_allclose(result.density, reconstructed, atol=1e-12, rtol=1e-12)
        j, k = scf.RHF(mol).get_jk(mol, result.density)
        hcore = scf.RHF(mol).get_hcore()
        physical_energy = np.sum(result.density * (hcore + 0.5 * j - 0.25 * k))
        self.assertAlmostEqual(result.energy, physical_energy + mol.energy_nuc(), places=10)

    def test_open_shell_is_rejected(self):
        with self.assertRaises(ValueError):
            GPUFriendlyRHF(molecule("H 0 0 0", spin=1), backend="numpy")

    def test_bad_initial_density_is_rejected(self):
        mol = molecule("H 0 0 0; H 0 0 0.74")
        solver = GPUFriendlyRHF(mol, backend="numpy")
        for density in (np.ones((1, 1)), np.full((2, 2), np.nan),
                        np.full((2, 2), np.inf)):
            with self.subTest(density=density):
                with self.assertRaises(ValueError):
                    solver.kernel(dm0=density)

    def test_exact_memory_budget_is_checked_before_integrals(self):
        mol = molecule("O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", "6-31g")
        with self.assertRaises(MemoryError):
            GPUFriendlyRHF(mol, backend="numpy", integral_mode="exact",
                           memory_limit_mb=0.001)

    @unittest.skipIf(importlib.util.find_spec("cupy") is not None,
                     "This check needs an environment without CuPy")
    def test_explicit_gpu_request_does_not_silently_run_on_cpu(self):
        mol = molecule("H 0 0 0; H 0 0 0.74")
        with self.assertRaises(RuntimeError):
            GPUFriendlyRHF(mol, backend="cupy")


if __name__ == "__main__":
    unittest.main()
