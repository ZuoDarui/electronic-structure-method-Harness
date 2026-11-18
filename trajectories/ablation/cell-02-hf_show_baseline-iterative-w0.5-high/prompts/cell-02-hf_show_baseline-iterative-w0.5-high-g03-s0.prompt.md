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
- h2_sto3g: converged=True signed_err=-2.220446049250313e-16 time_ratio=0.047
- heh_sto3g: converged=True signed_err=0.0 time_ratio=0.036
- h2o_sto3g: converged=True signed_err=0.0 time_ratio=0.050
- h2_631g: converged=True signed_err=-1.021405182655144e-14 time_ratio=0.066
- h2o_631g: converged=True signed_err=2.842170943040401e-14 time_ratio=0.079
- nh3_631gs: converged=True signed_err=8.526512829121202e-14 time_ratio=0.114
- ch4_ccpvdz: converged=True signed_err=9.947598300641403e-14 time_ratio=0.021
max_abs_err=9.948e-14 geomean_time_ratio=0.052

```python
import math

import numpy as np


def _pair_index(n):
    ar = np.arange(n)
    hi = np.maximum.outer(ar, ar)
    pidx = hi * (hi + 1)
    pidx >>= 1
    pidx += np.minimum.outer(ar, ar)
    return pidx


def _fock_builder(mol, h, n):
    """Return fock(d) -> h + J(d) - 0.5 K(d) built from s8 integrals."""
    s8 = np.asarray(mol.intor("int2e", aosym="s8")).ravel()
    pidx = _pair_index(n)

    if n <= 9:
        n2 = n * n
        p = pidx.ravel()
        idx = np.maximum.outer(p, p)
        idx = idx * (idx + 1)
        idx >>= 1
        idx += np.minimum.outer(p, p)
        eri = np.take(s8, idx)
        g = eri - 0.5 * eri.reshape(n, n, n, n).transpose(0, 2, 1, 3).reshape(n2, n2)

        def fock(d):
            f = (g @ d.ravel()).reshape(n, n)
            f += h
            return f

        return fock

    npair = n * (n + 1) // 2
    r, c = np.tril_indices(npair)
    m = np.empty(npair * npair)
    m[r * npair + c] = s8
    m[c * npair + r] = s8
    del r, c

    I, J = np.tril_indices(n)
    pin = pidx[I] * npair
    pj = pidx[J]
    a = np.take(m, pin[:, :, None] + pj[:, None, :])
    a2 = a + a.transpose(0, 2, 1)
    del a
    trif = I * n + J
    kx = np.take(a2.reshape(npair, n * n), trif, axis=1)
    del a2

    g = m.reshape(npair, npair)
    kx *= 0.25
    g -= kx
    del kx
    g *= np.where(I == J, 1.0, 2.0)[None, :]

    def fock(d):
        f = np.take(g @ np.take(d, trif), pidx)
        f += h
        return f

    return fock


def solve(mol, max_iteration=100, tol=1e-10, max_diis=8):
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    s = mol.intor("int1e_ovlp")
    n = h.shape[0]
    nocc = mol.nelectron // 2
    enuc = float(mol.energy_nuc())
    fock = _fock_builder(mol, h, n)

    w, v = np.linalg.eigh(s)
    keep = w > 1e-10
    if keep.all():
        x = v / np.sqrt(w)
    else:
        w = w[keep]
        x = v[:, keep] / np.sqrt(w)
    wmin = float(w.min())
    xt = np.ascontiguousarray(x.T)
    mo = x.shape[1]
    sq2 = math.sqrt(2.0)

    # ||X^T C X||_F <= ||C||_F / wmin, so the AO commutator can only be below
    # tol once the orthogonal-basis error is below this gate.
    gate = tol / wmin * 1.01
    egate = max(1e-4, gate * 1e3)

    fs = np.empty((max_diis, mo * mo))
    es = np.empty((max_diis, mo * mo))
    b = np.zeros((max_diis + 1, max_diis + 1))
    b[0, 1:] = 1.0
    b[1:, 0] = 1.0
    rhs = np.zeros(max_diis + 1)
    rhs[0] = 1.0
    cnt = 0
    ptr = 0

    fo_use = xt @ h @ x
    e_prev = None
    energy = None
    converged = False
    d = None
    f = None

    for _ in range(max_iteration):
        c = np.linalg.eigh(fo_use)[1]
        co = c[:, :nocc] * sq2
        do = co @ co.T
        cx = x @ co
        d = cx @ cx.T
        f = fock(d)
        fo = xt @ f @ x
        fd = fo @ do
        ev = fd - fd.T
        evr = ev.ravel()
        nrm = math.sqrt(float(evr @ evr))

        if nrm < egate:
            energy = enuc + 0.5 * float(np.vdot(d, h) + np.vdot(d, f))
            if nrm < gate and e_prev is not None and abs(energy - e_prev) < tol:
                fds = f @ d @ s
                if float(np.linalg.norm(fds - fds.T)) < tol:
                    converged = True
                    break
            e_prev = energy
        else:
            e_prev = None

        fs[ptr] = fo.ravel()
        es[ptr] = evr
        if cnt < max_diis:
            cnt += 1
        row = es[:cnt] @ evr
        b[ptr + 1, 1:cnt + 1] = row
        b[1:cnt + 1, ptr + 1] = row
        ptr += 1
        if ptr == max_diis:
            ptr = 0

        if cnt > 1:
            try:
                coef = np.linalg.solve(b[:cnt + 1, :cnt + 1], rhs[:cnt + 1])[1:]
                if not math.isfinite(float(coef.sum())):
                    raise np.linalg.LinAlgError
                fo_use = (coef @ fs[:cnt]).reshape(mo, mo)
            except np.linalg.LinAlgError:
                fs[0] = fo.ravel()
                es[0] = evr
                b[1, 1] = float(evr @ evr)
                cnt = 1
                ptr = 1 % max_diis
                fo_use = fo
        else:
            fo_use = fo

    if not converged:
        e_prev = None
        for _ in range(200):
            f = fock(d)
            energy = enuc + 0.5 * float(np.vdot(d, h) + np.vdot(d, f))
            fds = f @ d @ s
            cnorm = float(np.linalg.norm(fds - fds.T))
            if e_prev is not None and abs(energy - e_prev) < tol and cnorm < tol:
                converged = True
                break
            e_prev = energy
            c = np.linalg.eigh(xt @ f @ x)[1]
            cx = x @ (c[:, :nocc] * sq2)
            d = 0.5 * d + 0.5 * (cx @ cx.T)

    if energy is None:
        energy = enuc + 0.5 * float(np.vdot(d, h) + np.vdot(d, f))

    return {"e_tot": float(energy), "converged": bool(converged)}
```

Write an improved program: keep every molecule converged with energy error under 1e-8 Hartree and reduce solve(mol) time. Reply with the full program.
