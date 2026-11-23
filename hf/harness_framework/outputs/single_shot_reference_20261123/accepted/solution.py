"""Reference NumPy RHF candidate used to validate the single-shot accuracy gate.

It is a test fixture, not a generated program. It obeys the same contract and
AST guard as generated code (only numpy/scipy/math imports).
"""
import math

import numpy as np
import scipy.linalg


def build_jk(problem, density, backend):
    density = np.asarray(density, dtype=np.float64)
    if "eri" in problem:
        eri = problem["eri"]
        j = np.einsum("pqrs,rs->pq", eri, density, optimize=True)
        k = np.einsum("prqs,rs->pq", eri, density, optimize=True)
        return j, k
    factors = problem["df_factors"]
    n = density.shape[0]
    j = np.einsum("Ppq,P->pq", factors, np.einsum("Prs,rs->P", factors, density, optimize=True), optimize=True)
    k = np.zeros((n, n))
    block = 32
    for start in range(0, factors.shape[0], block):
        chunk = factors[start:start + block]
        half = np.einsum("Ppr,rs->Pps", chunk, density, optimize=True)
        k += np.einsum("Pps,Pqs->pq", half, chunk, optimize=True)
    return j, k


def _fock(problem, density):
    j, k = build_jk(problem, density, "numpy")
    return problem["hcore"] + j - 0.5 * k


def _energy(problem, density, fock):
    return float(0.5 * np.einsum("pq,qp->", density, problem["hcore"] + fock) + problem["enuc"])


def _orbitals(fock, overlap, nocc):
    eps, coeff = scipy.linalg.eigh(fock, overlap)
    density = 2.0 * coeff[:, :nocc] @ coeff[:, :nocc].T
    return eps, coeff, 0.5 * (density + density.T)


def solve(problem, options, backend):
    overlap, hcore = problem["overlap"], problem["hcore"]
    n = overlap.shape[0]
    nocc = int(problem["nelectron"]) // 2
    conv_tol = float(options.get("conv_tol", 1e-10))
    residual_tol = float(options.get("residual_tol", 1e-8))
    max_cycle = int(options.get("max_cycle", 100))
    diis_space = int(options.get("diis_space", 8))
    s_eval, s_vec = np.linalg.eigh(overlap)
    x = s_vec / np.sqrt(s_eval)
    eps, coeff, density = _orbitals(hcore, overlap, nocc)
    fock = _fock(problem, density)
    energy = _energy(problem, density, fock)
    focks, errors, history = [], [], []
    converged = False
    for cycle in range(1, max_cycle + 1):
        residual = fock @ density @ overlap - overlap @ density @ fock
        residual_norm = float(np.max(np.abs(residual)))
        if diis_space > 0:
            focks.append(fock)
            errors.append(x.T @ residual @ x)
            focks, errors = focks[-diis_space:], errors[-diis_space:]
            m = len(focks)
            b = -np.ones((m + 1, m + 1))
            b[m, m] = 0.0
            for i in range(m):
                for k in range(i + 1):
                    b[i, k] = b[k, i] = float(np.vdot(errors[i], errors[k]))
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                weights = np.linalg.solve(b, rhs)[:m]
                extrapolated = sum(w * f for w, f in zip(weights, focks))
            except np.linalg.LinAlgError:
                extrapolated = fock
        else:
            extrapolated = fock
        eps, coeff, new_density = _orbitals(extrapolated, overlap, nocc)
        new_fock = _fock(problem, new_density)
        new_energy = _energy(problem, new_density, new_fock)
        delta = abs(new_energy - energy)
        density, fock, energy = new_density, new_fock, new_energy
        new_residual = float(np.max(np.abs(fock @ density @ overlap - overlap @ density @ fock)))
        history.append({"cycle": cycle, "energy": energy, "residual_norm": new_residual, "delta_e": delta,
                        "pre_residual_norm": residual_norm})
        if delta < conv_tol and new_residual < residual_tol:
            converged = True
            break
    # Final orbitals from the physical Fock of the converged density.
    eps, coeff, density = _orbitals(fock, overlap, nocc)
    fock = _fock(problem, density)
    energy = _energy(problem, density, fock)
    occupations = np.zeros(n)
    occupations[:nocc] = 2.0
    density = (coeff * occupations) @ coeff.T
    if not math.isfinite(energy):
        converged = False
    return {"energy": energy, "density": density, "mo_coeff": coeff, "mo_energy": eps,
            "occupations": occupations, "converged": converged, "history": history}
