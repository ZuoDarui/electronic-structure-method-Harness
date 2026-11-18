Write one self-contained Python program for closed-shell restricted Hartree–Fock (RHF).
Define solve(mol) and return {"e_tot": float, "converged": bool}.
mol is a built pyscf.gto.Mole with an even electron count and spin 0.
You may import numpy, scipy, pyscf.gto, and the pure standard-library modules
math, cmath, itertools, functools, collections, typing, and dataclasses; any
other import is rejected. Use mol.intor(...) for integrals and mol.energy_nuc()
for the nuclear repulsion.
Do not import pyscf.scf or pyscf.df, and do not call kernel(). File I/O, exec,
and eval are rejected. Each timed call runs in a freshly loaded module with a
fresh Mole, so nothing can be cached between calls.
Reply with the complete program in one fenced python block.


Use chemist integrals eri[i,j,k,l] = (ij|kl) from mol.intor("int2e").
h = mol.intor("int1e_kin") + mol.intor("int1e_nuc"), S = mol.intor("int1e_ovlp").
nocc = mol.nelectron // 2. Build D from the occupied eigenvectors of the
generalized problem F C = S C epsilon. With D the total closed-shell density,
J_ij = sum_kl (ij|kl) D_kl and K_ij = sum_kl (ik|jl) D_kl, F = h + J - 0.5 K,
E = mol.energy_nuc() + 0.5 * sum_ij D_ij (h_ij + F_ij).
Accelerate with DIIS on the commutator F D S - S D F.
Stop when the energy change and the commutator norm are both under 1e-10.


Scoring: every molecule below must converge, and the energy is compared with an
independent RHF reference (target error under 1e-8 Hartree). solve(mol) wall
time is compared with a reference program on the same host; lower is better.
Integral evaluation through mol.intor counts toward your time.


Molecules, with coordinates in the stated unit:
- h2_sto3g: basis sto-3g, charge 0, spin 0, unit Angstrom, atom H 0 0 0; H 0 0 0.74
- heh_sto3g: basis sto-3g, charge 1, spin 0, unit Angstrom, atom He 0 0 0; H 0 0 0.774
- h2o_sto3g: basis sto-3g, charge 0, spin 0, unit Angstrom, atom O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692
- h2_631g: basis 6-31g, charge 0, spin 0, unit Angstrom, atom H 0 0 0; H 0 0 0.74
- h2o_631g: basis 6-31g, charge 0, spin 0, unit Angstrom, atom O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692
- nh3_631gs: basis 6-31g*, charge 0, spin 0, unit Angstrom, atom N 0 0 0.1162; H 0 0.9377 -0.2711; H 0.8121 -0.4689 -0.2711; H -0.8121 -0.4689 -0.2711
- ch4_ccpvdz: basis cc-pvdz, charge 0, spin 0, unit Angstrom, atom C 0 0 0; H 0.6291 0.6291 0.6291; H -0.6291 -0.6291 0.6291; H -0.6291 0.6291 -0.6291; H 0.6291 -0.6291 -0.6291

Reference program on the scoring host (shown for this ablation only; it uses pyscf.scf for its initial guess, which your program may not import):
```python
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

```

Best program so far and its scores. time_ratio is your time divided by the reference program's time; under 1 means faster.

status: ok
- h2_sto3g: converged=True signed_err=2.220446049250313e-16 time_ratio=0.055
- heh_sto3g: converged=True signed_err=-8.881784197001252e-16 time_ratio=0.055
- h2o_sto3g: converged=True signed_err=7.105427357601002e-14 time_ratio=0.062
- h2_631g: converged=True signed_err=-1.0436096431476471e-14 time_ratio=0.104
- h2o_631g: converged=True signed_err=4.263256414560601e-14 time_ratio=0.098
- nh3_631gs: converged=True signed_err=9.947598300641403e-14 time_ratio=0.155
- ch4_ccpvdz: converged=True signed_err=1.4210854715202004e-13 time_ratio=0.034
max_abs_err=1.421e-13 geomean_time_ratio=0.072

```python
import numpy as np


def _packed_g(mol, n):
    """Return the closed-shell two-electron operator in packed (i>=j) pair space.

    F_p = h_p + sum_q G[p, q] * Dpacked_q, where p=(i,j), q=(k,l) with i>=j, k>=l.
    """
    s8 = np.asarray(mol.intor("int2e", aosym="s8")).ravel()
    npair = n * (n + 1) // 2
    rows, cols = np.tril_indices(npair)
    m = np.empty((npair, npair))
    m[rows, cols] = s8
    m[cols, rows] = s8

    ii, jj = np.indices((n, n))
    hi = np.maximum(ii, jj)
    lo = np.minimum(ii, jj)
    pidx = hi * (hi + 1) // 2 + lo

    I, J = np.tril_indices(n)
    pik = pidx[np.ix_(I, I)]
    pjl = pidx[np.ix_(J, J)]
    kx = m[pik, pjl]
    del pik, pjl
    pil = pidx[np.ix_(I, J)]
    pjk = pidx[np.ix_(J, I)]
    kx += m[pil, pjk]
    del pil, pjk

    g = m
    g -= 0.25 * kx
    w = np.where(I == J, 1.0, 2.0)
    g *= w[None, :]
    return g, pidx, I, J


def solve(mol, max_iteration=200, tol=1e-10, max_diis=8):
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    s = mol.intor("int1e_ovlp")
    n = h.shape[0]
    nocc = mol.nelectron // 2
    enuc = float(mol.energy_nuc())

    g, pidx, I, J = _packed_g(mol, n)

    w, v = np.linalg.eigh(s)
    keep = w > 1e-10
    x = v[:, keep] / np.sqrt(w[keep])
    xt = x.T

    def density_from_fock(f):
        _, c = np.linalg.eigh(xt @ f @ x)
        co = x @ c[:, :nocc]
        return 2.0 * (co @ co.T)

    def build_fock(d):
        return h + (g @ d[I, J])[pidx]

    d = density_from_fock(h)

    errs = []
    focks = []
    e_prev = None
    energy = 0.0
    converged = False
    f = h

    for _ in range(max_iteration):
        f = build_fock(d)
        energy = enuc + 0.5 * float(np.vdot(d, h + f))
        fds = f @ d @ s
        comm = fds - fds.T
        cnorm = float(np.linalg.norm(comm))
        if e_prev is not None and abs(energy - e_prev) < tol and cnorm < tol:
            converged = True
            break
        e_prev = energy

        errs.append((xt @ comm @ x).ravel())
        focks.append(f)
        if len(errs) > max_diis:
            errs.pop(0)
            focks.pop(0)
        k = len(errs)
        if k >= 2:
            e_arr = np.array(errs)
            b = np.empty((k + 1, k + 1))
            b[:k, :k] = e_arr @ e_arr.T
            b[k, :k] = 1.0
            b[:k, k] = 1.0
            b[k, k] = 0.0
            rhs = np.zeros(k + 1)
            rhs[k] = 1.0
            try:
                coef = np.linalg.solve(b, rhs)[:k]
                if not np.all(np.isfinite(coef)):
                    raise np.linalg.LinAlgError
                f_use = np.tensordot(coef, np.array(focks), axes=1)
            except np.linalg.LinAlgError:
                errs = errs[-1:]
                focks = focks[-1:]
                f_use = f
        else:
            f_use = f
        d = density_from_fock(f_use)

    if not converged:
        d = density_from_fock(f)
        for _ in range(100):
            f = build_fock(d)
            energy = enuc + 0.5 * float(np.vdot(d, h + f))
            fds = f @ d @ s
            cnorm = float(np.linalg.norm(fds - fds.T))
            if e_prev is not None and abs(energy - e_prev) < tol and cnorm < tol:
                converged = True
                break
            e_prev = energy
            d = 0.5 * d + 0.5 * density_from_fock(f)

    return {"e_tot": float(energy), "converged": bool(converged)}
```

Write an improved program: keep every molecule converged with energy error under 1e-8 Hartree and reduce solve(mol) time. Reply with the full program.
