import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pyscf import gto, lib, scf

import hf_torch
from hf_torch import DenseLinearAlgebra

try:
    hf_torch.import_torch()
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

DEVICE = os.environ.get("HF_TEST_TORCH_DEVICE", "cpu")


def _problem(n=60, nocc=11, seed=4):
    rng = np.random.RandomState(seed)
    a = rng.randn(n, n)
    overlap = a @ a.T / n + np.eye(n)
    focks = []
    for _ in range(5):
        f = rng.randn(n, n)
        focks.append(0.5 * (f + f.T))
    d = rng.randn(n, n)
    return overlap, focks, 0.5 * (d + d.T), nocc


@unittest.skipUnless(HAVE_TORCH, "PyTorch is not installed")
class TorchDenseTest(unittest.TestCase):
    def test_matches_numpy_backend(self):
        overlap, focks, dm, nocc = _problem()
        cpu = DenseLinearAlgebra(overlap, backend="cpu")
        dev = DenseLinearAlgebra(overlap, backend="torch", torch_device=DEVICE)
        for fock in focks:
            err_cpu = cpu.commutator(fock, dm)
            err_dev = dev.commutator(fock, dm)
            np.testing.assert_allclose(dev._host(err_dev), err_cpu, atol=1e-12)
            self.assertAlmostEqual(dev.norm(err_dev), cpu.norm(err_cpu), delta=1e-10)
            cpu.push(err_cpu, fock)
            dev.push(err_dev, fock)
            np.testing.assert_allclose(dev._gram, cpu._gram, rtol=1e-12, atol=1e-12)
        f_cpu = cpu.extrapolate()
        f_dev = dev.extrapolate()
        np.testing.assert_allclose(dev._host(f_dev), f_cpu, atol=1e-10)
        e_cpu, _, d_cpu = cpu.solve(f_cpu, nocc)
        e_dev, c_dev, d_dev = dev.solve(f_dev, nocc)
        self.assertIsInstance(c_dev, np.ndarray)
        np.testing.assert_allclose(e_dev, e_cpu, atol=1e-10)
        np.testing.assert_allclose(d_dev, d_cpu, atol=1e-9)
        np.testing.assert_allclose(c_dev.T @ overlap @ c_dev, np.eye(len(e_dev)), atol=1e-10)
        dev.clear()
        self.assertEqual(dev.errors, [])

    def test_check_calls_compare_against_scipy(self):
        overlap, focks, dm, nocc = _problem(seed=9)
        dev = DenseLinearAlgebra(overlap, backend="torch", torch_device=DEVICE,
                                 check_calls=2)
        for fock in focks[:3]:
            err = dev.commutator(fock, dm)
            dev.push(err, fock)
            extrapolated = dev.extrapolate()
            dev.solve(fock if extrapolated is None else extrapolated, nocc)
        self.assertEqual(dev._checks_done, 2)
        self.assertLess(max(dev.check_max.values()), 1e-10)

    def test_rejects_unknown_device(self):
        overlap, _, _, _ = _problem(n=8)
        with self.assertRaises(ValueError):
            DenseLinearAlgebra(overlap, backend="torch", torch_device="mps")

    def test_explicit_cuda_fails_without_cuda(self):
        torch = hf_torch.import_torch()
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "visible CUDA"):
                DenseLinearAlgebra(np.eye(3), backend="torch", torch_device="cuda")

    def test_rejects_indefinite_overlap(self):
        with self.assertRaises(RuntimeError):
            DenseLinearAlgebra(-np.eye(6), backend="torch", torch_device=DEVICE)

    def test_iteration_rijcosx_honors_torch_device(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g", verbose=0)
        hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        overlap = mol.intor("int1e_ovlp")
        dm = scf.hf.init_guess_by_minao(mol)
        options = dict(auxbasis="weigend", grid_rad=10, grid_ang=50)
        reference = hf_torch.Hartree_Fock(mol, dense_backend="cpu").iteration_rijcosx(
            dm, hcore, overlap, mol.nelectron // 2, mol.nelectron, **options)
        solver = hf_torch.Hartree_Fock(mol, dense_backend="torch", torch_device=DEVICE)
        actual = solver.iteration_rijcosx(
            dm, hcore, overlap, mol.nelectron // 2, mol.nelectron, **options)
        self.assertEqual(actual["dense_backend"], "torch")
        self.assertIn(DEVICE, actual["dense_device"])
        self.assertTrue(actual["all_finite"])
        self.assertLess(actual["density_electron_abs"], 1e-10)
        np.testing.assert_allclose(actual["density"], reference["density"], atol=1e-10, rtol=0)

    def test_benchmark_config_honors_device_without_hiding_cuda(self):
        previous_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
        previous_threads = lib.num_threads()
        try:
            with tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                (directory / "h2.xyz").write_text("2\nH2\nH 0 0 0\nH 0 0 .74\n")
                config = {
                    "xyz": "h2.xyz", "basis": "sto-3g", "benchmark_one_iteration": True,
                    "output_prefix": "benchmark", "output_dir": ".",
                    "resources": {"threads": 2, "cpu_cores": 2,
                                  "dense_backend": "torch", "torch_device": DEVICE},
                    "solver_options": {"auxbasis": "weigend", "grid_rad": 10, "grid_ang": 50},
                }
                path = directory / "config.json"
                path.write_text(json.dumps(config))
                with contextlib.redirect_stdout(io.StringIO()):
                    result = hf_torch.run_from_config(path)
                self.assertEqual(result["dense_backend"], "torch")
                self.assertEqual(result["torch_device"], DEVICE)
                self.assertIn(DEVICE, result["dense_device"])
                self.assertTrue(result["fock_finite"])
                self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), previous_visibility)
        finally:
            lib.num_threads(previous_threads)

    def test_default_solver_uses_torch_and_repeated_cpu_call_keeps_cuda_visibility(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g", verbose=0)
        hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        overlap = mol.intor("int1e_ovlp")
        visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
        solver = hf_torch.Hartree_Fock(mol, torch_device="cpu")
        self.assertEqual(solver.dense_backend, "torch")
        with contextlib.redirect_stdout(io.StringIO()):
            energy1, dm1 = solver.solve_rhf(mol.nelec, hcore, overlap, mol, mol.energy_nuc(), tol=1e-10)
            energy2, dm2 = solver.solve_rhf(mol.nelec, hcore, overlap, mol, mol.energy_nuc(), tol=1e-10)
        self.assertAlmostEqual(energy1, energy2, places=12)
        np.testing.assert_allclose(dm1, dm2, atol=1e-12, rtol=0)
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), visibility)
        if DEVICE == "cuda":
            dense = DenseLinearAlgebra(overlap, backend="torch", torch_device="cuda")
            self.assertIn("cuda", dense.device)
            dense.clear()


class DenseBackendNameTest(unittest.TestCase):
    def test_unknown_backend(self):
        with self.assertRaises(ValueError):
            DenseLinearAlgebra(np.eye(3), backend="jax")


if __name__ == "__main__":
    unittest.main()
