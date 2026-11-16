"""Closed-shell restricted Hartree–Fock baseline.

Replace this file with the cluster program when that program exposes
``solve(mol) -> {"e_tot": float, "converged": bool}``. The harness times and
scores whatever file is configured as the baseline; it does not import this
module by name.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def _jk(eri: np.ndarray, dm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coulomb = np.einsum("ijkl,kl->ij", eri, dm, optimize=True)
    exchange = np.einsum("ikjl,kl->ij", eri, dm, optimize=True)
    return coulomb, exchange


def _core_density(hcore: np.ndarray, overlap: np.ndarray, nocc: int) -> np.ndarray:
    _, coeff = eigh(hcore, overlap)
    occupied = coeff[:, :nocc]
    return 2.0 * occupied @ occupied.T


def _atomic_density(mol, hcore: np.ndarray, overlap: np.ndarray, nocc: int) -> np.ndarray:
    """MINAO atomic density, with a core-Hamiltonian density if it is unavailable."""
    try:
        from pyscf.scf.hf import init_guess_by_minao

        density = np.asarray(init_guess_by_minao(mol), dtype=float)
    except Exception:
        return _core_density(hcore, overlap, nocc)
    if density.shape != hcore.shape or not np.isfinite(density).all():
        return _core_density(hcore, overlap, nocc)
    return 0.5 * (density + density.T)


def _diis(errors: list[np.ndarray], focks: list[np.ndarray]) -> np.ndarray:
    size = len(errors)
    matrix = np.zeros((size + 1, size + 1))
    matrix[-1, :-1] = 1.0
    matrix[:-1, -1] = 1.0
    rhs = np.zeros(size + 1)
    rhs[-1] = 1.0
    for i in range(size):
        for j in range(i, size):
            value = float(np.dot(errors[i].ravel(), errors[j].ravel()))
            matrix[i, j] = matrix[j, i] = value
    try:
        coeff = np.linalg.solve(matrix, rhs)[:-1]
    except np.linalg.LinAlgError:
        return focks[-1]
    mixed = np.zeros_like(focks[-1])
    for weight, fock in zip(coeff, focks):
        mixed += weight * fock
    return mixed


def solve(mol, max_iteration: int = 80, tol: float = 1e-10, max_diis: int = 8) -> dict:
    """Run closed-shell RHF on a built PySCF ``Mole`` and return the total energy."""
    if mol.nelectron % 2 != 0 or mol.spin != 0:
        raise ValueError("baselines/hf.py implements closed-shell RHF only")
    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    overlap = mol.intor("int1e_ovlp")
    eri = mol.intor("int2e")
    nuclear = float(mol.energy_nuc())
    nocc = mol.nelectron // 2

    density = _atomic_density(mol, hcore, overlap, nocc)
    errors: list[np.ndarray] = []
    focks: list[np.ndarray] = []
    previous = None
    energy = None
    converged = False

    for _ in range(max_iteration):
        coulomb, exchange = _jk(eri, density)
        fock = hcore + coulomb - 0.5 * exchange
        error = fock @ density @ overlap - overlap @ density @ fock
        errors.append(error)
        focks.append(fock)
        if len(errors) > max_diis:
            errors.pop(0)
            focks.pop(0)
        if len(errors) >= 2:
            fock = _diis(errors, focks)

        _, coeff = eigh(fock, overlap)
        occupied = coeff[:, :nocc]
        density = 2.0 * occupied @ occupied.T
        coulomb, exchange = _jk(eri, density)
        fock_new = hcore + coulomb - 0.5 * exchange
        energy = nuclear + 0.5 * float(np.einsum("ij,ij->", hcore + fock_new, density))
        commutator = fock_new @ density @ overlap - overlap @ density @ fock_new
        energy_change = None if previous is None else abs(energy - previous)
        previous = energy
        if (
            energy_change is not None
            and energy_change < tol
            and float(np.linalg.norm(commutator)) < tol
        ):
            converged = True
            break

    if energy is None:
        raise RuntimeError("SCF did not evaluate an energy")
    return {"e_tot": float(energy), "converged": bool(converged)}
