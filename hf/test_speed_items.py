"""Regressions for the C172 speed items: the fixed SGX block budget, the
coarse-state release and early switch to the final grid, and the CPU and
CuPy dense linear algebra.

CuPy tests skip when no CUDA device is visible, unless HF_REQUIRE_CUPY=1.
"""
import contextlib
import io
import os
import unittest
import weakref
from unittest.mock import patch

import numpy as np
import scipy.linalg
from pyscf import gto, lib, scf
from pyscf.sgx import sgx_jk

import hf
from hf import DenseLinearAlgebra, Hartree_Fock

lib.num_threads(2)

WATER = 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'


def require_cupy(test):
    try:
        cupy = hf.import_cupy()
        if cupy.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA device is visible")
    except Exception:
        if os.environ.get("HF_REQUIRE_CUPY") == "1":
            raise
        test.skipTest("CuPy with a CUDA device is not available")
    return cupy


def phase1_extrapolate(errors, focks):
    """The Phase 1 DIIS step, kept verbatim as the reference."""
    space = len(errors)
    if space < 2:
        return None
    B = np.zeros((space + 1, space + 1))
    B[-1, :-1] = B[:-1, -1] = 1
    B[-1, -1] = 0
    g = np.zeros(space + 1)
    g[-1] = 1
    for i in range(space):
        for j in range(space):
            B[i, j] = B[j, i] = np.vdot(errors[i], errors[j]).real
    if not np.isfinite(np.linalg.cond(B)) or np.linalg.cond(B) > 1e12:
        return None
    try:
        c = scipy.linalg.solve(B, g, assume_a='sym')[:-1]
    except scipy.linalg.LinAlgError:
        return None
    return np.einsum("i,ijk->jk", c, focks)


def random_problem(n=60, nocc=7, seed=3):
    """A well-conditioned overlap, a symmetric Fock and an occupied density."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    overlap = np.eye(n) + 0.02 * (a @ a.T) / n
    b = rng.standard_normal((n, n))
    fock = 0.5 * (b + b.T) + np.diag(np.linspace(-5.0, 5.0, n))
    _, coeff = scipy.linalg.eigh(fock, overlap)
    density = 2.0 * coeff[:, :nocc] @ coeff[:, :nocc].T
    return overlap, fock, density, nocc


def rijcosx_settings(**options):
    settings = dict(jk_mode='rijcosx', auxbasis='weigend', grid_rad=10, grid_ang=50,
                    final_grid_rad=15, final_grid_ang=74, max_iteration=60, tol=1e-8,
                    incremental_k=True, kfull_every=5)
    settings.update(options)
    return settings


def solve(solver, mol, messages=None, **options):
    hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    record = (lambda message, *args, **kwargs: messages.append(message)) \
        if messages is not None else (lambda *args, **kwargs: None)
    with patch.object(hf, '_emit_progress', side_effect=record):
        return solver.solve_rhf(mol.nelec, hcore, mol.intor('int1e_ovlp'), mol,
                                mol.energy_nuc(), **rijcosx_settings(**options))


class KBudgetTests(unittest.TestCase):
    def setUp(self):
        self.mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
        self.dm = scf.hf.init_guess_by_minao(self.mol)
        self.unit = int(sgx_jk.SGX_BLKSIZE)
        self.per_point_mb = 4 * self.mol.nao_nr() * 8 / 1e6

    def test_fixed_budget_pins_the_block_pyscf_loops_over(self):
        reference = Hartree_Fock(self.mol).build_sgx_k(self.dm, 'weigend', 20, 110)
        solver = Hartree_Fock(self.mol, k_max_memory_mb=2.01 * self.unit * self.per_point_mb)
        first = solver.build_sgx_k(self.dm, 'weigend', 20, 110)
        self.assertEqual(solver._last_k_grid[1], 2 * self.unit)
        # The screening data keeps the max_memory of its build, which alone
        # would give two units again; only the override yields three.
        solver.k_max_memory_mb = 3.01 * self.unit * self.per_point_mb
        calls = []
        original = gto.Mole.eval_gto

        def record(mol, eval_name, coords, *args, **kwargs):
            calls.append(len(coords))
            return original(mol, eval_name, coords, *args, **kwargs)

        with patch.object(gto.Mole, 'eval_gto', record):
            second = solver.build_sgx_k(self.dm, 'weigend', 20, 110)
        ngrids, blksize = solver._last_k_grid
        self.assertEqual(blksize, 3 * self.unit)
        blocks = -(-ngrids // blksize)
        self.assertGreaterEqual(blocks, 2)
        self.assertEqual(len(calls), blocks)
        self.assertEqual(calls[:-1], [blksize] * (blocks - 1))
        self.assertEqual(sum(calls), ngrids)
        self.assertIn('k_budget_mb=', solver._k_grid_note())
        np.testing.assert_allclose(first, reference, atol=1e-12, rtol=0)
        np.testing.assert_allclose(second, reference, atol=1e-12, rtol=0)

    def test_budget_ignores_the_resident_set_and_respects_the_ceiling(self):
        budget = 2.01 * self.unit * self.per_point_mb
        huge_rss = 1.0e6
        with patch('pyscf.lib.current_memory', return_value=(huge_rss, huge_rss)):
            legacy = Hartree_Fock(self.mol, max_memory=4000)
            legacy.build_sgx_k(self.dm, 'weigend', 10, 50)
            fixed = Hartree_Fock(self.mol, k_max_memory_mb=budget)
            fixed.build_sgx_k(self.dm, 'weigend', 10, 50)
            capped = Hartree_Fock(self.mol, k_max_memory_mb=budget, memory_ceiling_mb=huge_rss)
            capped.build_sgx_k(self.dm, 'weigend', 10, 50)
        self.assertEqual(legacy._last_k_grid[1], self.unit)
        self.assertIsNone(legacy._sgx_engines[('weigend', 10, 50, True)]._hf_blksize)
        self.assertEqual(fixed._last_k_grid[1], min(2 * self.unit, fixed._last_k_grid[0]))
        self.assertEqual(capped._last_k_grid[1], self.unit)
        self.assertLess(capped._last_k_budget_mb, 0.0)

    def test_job_k_memory_reads_resources(self):
        self.assertEqual(hf.job_k_memory_mb({'memory_gib': 256}), (None, None))
        self.assertEqual(hf.job_k_memory_mb({'memory_gib': 256, 'k_memory_gib': 64}),
                         (65536.0, 256 * 1024.0 * 0.75))
        self.assertEqual(hf.job_k_memory_mb({'k_memory_gib': 8}), (8192.0, None))
        with self.assertRaisesRegex(ValueError, 'ceiling'):
            hf.job_k_memory_mb({'memory_gib': 256, 'k_memory_gib': 200})
        with self.assertRaisesRegex(ValueError, 'k_memory_gib'):
            hf.job_k_memory_mb({'memory_gib': 256, 'k_memory_gib': 0})

    def test_invalid_budgets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'k_max_memory_mb'):
            Hartree_Fock(self.mol, k_max_memory_mb=0)
        with self.assertRaisesRegex(ValueError, 'memory_ceiling_mb'):
            Hartree_Fock(self.mol, memory_ceiling_mb=-1)


class GridSwitchTests(unittest.TestCase):
    def setUp(self):
        self.mol = gto.M(atom=WATER, basis='6-31g', verbose=0)

    def test_early_switch_stops_the_coarse_grid_and_keeps_the_fine_energy(self):
        full = Hartree_Fock(self.mol)
        energy_full, _ = solve(full, self.mol)
        messages = []
        early = Hartree_Fock(self.mol, k_max_memory_mb=64.0)
        energy_early, density = solve(early, self.mol, messages, grid_switch_energy_tol=1e-5)
        stages_full = full.last_scf_result['grid_stages']
        stages_early = early.last_scf_result['grid_stages']
        self.assertEqual(stages_full[0]['exit'], 'converged')
        self.assertEqual(stages_early[0]['exit'], 'energy_switch')
        self.assertEqual([s['grid'] for s in stages_early], [[10, 50], [15, 74]])
        self.assertLess(stages_early[0]['iterations'], stages_full[0]['iterations'])
        self.assertTrue(any('reached grid_switch_energy_tol' in m for m in messages))
        self.assertLess(early.last_scf_result['gradient_rms'], 1e-8)
        self.assertLess(abs(energy_early - energy_full), 1e-8)
        overlap = self.mol.intor('int1e_ovlp')
        self.assertAlmostEqual(float(np.einsum('ij,ji', density, overlap)),
                               self.mol.nelectron, places=8)

    def test_switch_tolerance_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, 'grid_switch_energy_tol'):
            solve(Hartree_Fock(self.mol), self.mol, grid_switch_energy_tol=0.0)

    def test_switch_releases_the_coarse_grid_engine(self):
        solver = Hartree_Fock(self.mol)
        created = {}
        original = solver._sgx_engine

        def record(auxbasis, grid_rad, grid_ang, overlap_fit):
            engine = original(auxbasis, grid_rad, grid_ang, overlap_fit)
            created.setdefault((grid_rad, grid_ang), weakref.ref(engine))
            return engine

        messages = []
        with patch.object(solver, '_sgx_engine', side_effect=record):
            solve(solver, self.mol, messages)
        self.assertEqual(list(solver._sgx_engines), [('weigend', 15, 74, True)])
        self.assertIsNone(created[(10, 50)]())
        self.assertIsNotNone(created[(15, 74)]())
        self.assertTrue(any(m.startswith('SCF switching to final grid')
                            and 'released coarse-grid state' in m for m in messages))


class CpuDenseTests(unittest.TestCase):
    def test_cpu_dense_matches_the_phase1_operations(self):
        overlap, fock, density, nocc = random_problem()
        dense = DenseLinearAlgebra(overlap, backend='cpu')
        error = dense.commutator(fock, density)
        np.testing.assert_allclose(
            error, fock @ density @ overlap - overlap @ density @ fock, atol=1e-12, rtol=0)
        np.testing.assert_allclose(error, -error.T, atol=0, rtol=0)
        energies, coeff = scipy.linalg.eigh(fock, overlap)
        mo_energy, mo_coeff, solved = dense.solve(fock, nocc)
        occupied = coeff[:, :nocc]
        np.testing.assert_allclose(mo_energy, energies, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            solved, 2.0 * np.einsum('ma,na->mn', occupied, occupied.conj()), atol=1e-12, rtol=0)

    def test_incremental_gram_matches_the_phase1_extrapolation_after_eviction(self):
        overlap, fock, density, _ = random_problem()
        rng = np.random.default_rng(11)
        dense = DenseLinearAlgebra(overlap, backend='cpu', max_diis=4)
        errors, focks = [], []
        for step in range(7):
            perturbation = rng.standard_normal(fock.shape)
            trial = fock + 0.1 * (perturbation + perturbation.T) / (step + 1)
            error = dense.commutator(trial, density)
            dense.push(error, trial)
            errors.append(error)
            focks.append(trial.copy())
            errors, focks = errors[-4:], focks[-4:]
            expected = phase1_extrapolate(errors, focks)
            actual = dense.extrapolate()
            if expected is None:
                self.assertIsNone(actual)
            else:
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        self.assertEqual(len(dense.errors), 4)
        dense.clear()
        self.assertEqual((len(dense.errors), dense._gram.shape), (0, (0, 0)))

    def test_unknown_dense_backend_is_rejected(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='sto-3g', verbose=0)
        with self.assertRaisesRegex(ValueError, 'dense_backend'):
            Hartree_Fock(mol, dense_backend='jax')

    def test_cupy_backend_fails_closed_without_cupy(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='sto-3g', verbose=0)
        with patch.object(hf, 'import_cupy', side_effect=ImportError('no cupy')):
            with self.assertRaisesRegex(RuntimeError, 'requires CuPy'):
                Hartree_Fock(mol, dense_backend='cupy')


class CupyDenseTests(unittest.TestCase):
    def test_cupy_kernels_match_cpu(self):
        cupy = require_cupy(self)
        overlap, fock, density, nocc = random_problem(n=200, nocc=20)
        cpu = DenseLinearAlgebra(overlap, backend='cpu')
        gpu = DenseLinearAlgebra(overlap, backend='cupy', check_calls=1000)
        self.assertIn('cupy', gpu.device)
        np.testing.assert_allclose(cupy.asnumpy(gpu.commutator(fock, density)),
                                   cpu.commutator(fock, density), atol=1e-10, rtol=0)
        cpu_energy, _, cpu_density = cpu.solve(fock, nocc)
        gpu_energy, gpu_coeff, gpu_density = gpu.solve(fock, nocc)
        self.assertIsInstance(gpu_density, np.ndarray)
        np.testing.assert_allclose(gpu_density, cpu_density, atol=1e-10, rtol=0)
        np.testing.assert_allclose(gpu_energy, cpu_energy, atol=1e-10, rtol=0)
        rng = np.random.default_rng(5)
        for step in range(5):
            perturbation = rng.standard_normal(fock.shape)
            trial = fock + 0.1 * (perturbation + perturbation.T) / (step + 1)
            for dense in (cpu, gpu):
                dense.push(dense.commutator(trial, density), trial)
            reference = cpu.extrapolate()
            result = gpu.extrapolate()
            if reference is None:
                self.assertIsNone(result)
                continue
            np.testing.assert_allclose(cupy.asnumpy(result), reference, atol=1e-10, rtol=0)
            gpu.solve(result, nocc)
        self.assertLessEqual(max(gpu.check_max.values()), 1e-10)
        self.assertGreater(gpu.pool_gib(), 0.0)
        gpu.clear()

    def test_cupy_check_mode_rejects_a_wrong_eigensolve(self):
        cupy = require_cupy(self)
        overlap, fock, _, nocc = random_problem(n=80, nocc=9)
        gpu = DenseLinearAlgebra(overlap, backend='cupy', check_calls=1)
        real_eigh = cupy.linalg.eigh

        def shifted(matrix):
            values, vectors = real_eigh(matrix)
            return values + 1e-6, vectors

        with patch.object(cupy.linalg, 'eigh', side_effect=shifted):
            with self.assertRaisesRegex(RuntimeError, 'orbital_energy'):
                gpu.solve(fock, nocc)

    def test_cupy_scf_matches_cpu_scf(self):
        require_cupy(self)
        mol = gto.M(atom=WATER, basis='6-31g', verbose=0)
        energy_cpu, density_cpu = solve(Hartree_Fock(mol), mol, grid_switch_energy_tol=1e-5)
        gpu = Hartree_Fock(mol, dense_backend='cupy', dense_check_calls=100)
        messages = []
        energy_gpu, density_gpu = solve(gpu, mol, messages, grid_switch_energy_tol=1e-5)
        self.assertLess(abs(energy_gpu - energy_cpu), 1e-10)
        np.testing.assert_allclose(density_gpu, density_cpu, atol=1e-8, rtol=0)
        details = gpu.last_scf_result['dense']
        self.assertEqual(details['backend'], 'cupy')
        self.assertLessEqual(max(details['check_max_abs_diff'].values()), 1e-10)
        self.assertTrue(any(m.startswith('SCF dense linear algebra backend=cupy') for m in messages))
        self.assertTrue(any('gpu_pool_gib=' in m for m in messages))


if __name__ == '__main__':
    unittest.main()
