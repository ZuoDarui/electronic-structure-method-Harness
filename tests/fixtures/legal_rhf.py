"""Closed-shell RHF that stays inside the generated-code import rules."""

import numpy as np
from scipy.linalg import eigh


def _jk(eri, dm):
    coulomb = np.einsum("ijkl,kl->ij", eri, dm, optimize=True)
    exchange = np.einsum("ikjl,kl->ij", eri, dm, optimize=True)
    return coulomb, exchange


def _diis(errors, focks):
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


def solve(mol, max_iteration=80, tol=1e-10, max_diis=8):
    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    overlap = mol.intor("int1e_ovlp")
    eri = mol.intor("int2e")
    nuclear = float(mol.energy_nuc())
    nocc = mol.nelectron // 2
    _, coeff = eigh(hcore, overlap)
    density = 2.0 * coeff[:, :nocc] @ coeff[:, :nocc].T
    errors = []
    focks = []
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
        density = 2.0 * coeff[:, :nocc] @ coeff[:, :nocc].T
        coulomb, exchange = _jk(eri, density)
        fock_new = hcore + coulomb - 0.5 * exchange
        energy = nuclear + 0.5 * float(np.einsum("ij,ij->", hcore + fock_new, density))
        commutator = fock_new @ density @ overlap - overlap @ density @ fock_new
        energy_change = None if previous is None else abs(energy - previous)
        previous = energy
        if energy_change is not None and energy_change < tol and float(np.linalg.norm(commutator)) < tol:
            converged = True
            break
    return {"e_tot": float(energy), "converged": bool(converged)}
