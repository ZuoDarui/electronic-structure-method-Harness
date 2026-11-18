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

Best program so far and its scores. time_ratio is your time divided by the reference program's time; under 1 means faster.

status: ok
- h2_sto3g: converged=True signed_err=0.0 time_ratio=0.037
- heh_sto3g: converged=True signed_err=2.6645352591003757e-15 time_ratio=0.078
- h2o_sto3g: converged=True signed_err=2.842170943040401e-14 time_ratio=0.046
- h2_631g: converged=True signed_err=-8.881784197001252e-15 time_ratio=0.103
- h2o_631g: converged=True signed_err=8.526512829121202e-14 time_ratio=0.095
- nh3_631gs: converged=True signed_err=4.263256414560601e-14 time_ratio=0.168
- ch4_ccpvdz: converged=True signed_err=1.1368683772161603e-13 time_ratio=2.040
max_abs_err=1.137e-13 geomean_time_ratio=0.124

```python
import numpy as np
import scipy.linalg


def _full_eri(mol, n):
    eri8 = mol.intor("int2e", aosym="s8")
    npair = n * (n + 1) // 2
    ii, jj = np.tril_indices(n)
    pidx = np.empty((n, n), dtype=np.intp)
    pidx[ii, jj] = np.arange(npair)
    pidx[jj, ii] = np.arange(npair)
    a = np.arange(npair)
    hi = np.maximum(a[:, None], a[None, :])
    lo = np.minimum(a[:, None], a[None, :])
    s4 = eri8[hi * (hi + 1) // 2 + lo]
    p = pidx.ravel()
    return s4[p[:, None], p[None, :]].reshape(n, n, n, n)


def solve(mol):
    n = mol.nao_nr()
    nocc = mol.nelectron // 2
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri = _full_eri(mol, n)
    nn = n * n
    V = eri.reshape(nn, nn) - 0.5 * eri.transpose(0, 2, 1, 3).reshape(nn, nn)
    V = np.ascontiguousarray(V)
    enuc = mol.energy_nuc()

    w, U = np.linalg.eigh(S)
    keep = w > 1e-10
    X = U[:, keep] / np.sqrt(w[keep])

    def diag(F):
        e, Cp = np.linalg.eigh(X.T @ F @ X)
        return X @ Cp

    C = diag(H)
    Co = C[:, :nocc]
    D = 2.0 * Co @ Co.T

    focks, errs = [], []
    e_old = None
    converged = False
    e_tot = 0.0
    max_diis = 8
    for it in range(200):
        G = (V @ D.ravel()).reshape(n, n)
        F = H + G
        e_tot = 0.5 * np.einsum("ij,ij->", D, H + F) + enuc
        FDS = F @ D @ S
        err = X.T @ (FDS - FDS.T) @ X
        enorm = np.max(np.abs(err))
        if e_old is not None and abs(e_tot - e_old) < 1e-11 and enorm < 1e-7:
            converged = True
            break
        e_old = e_tot
        focks.append(F)
        errs.append(err)
        if len(focks) > max_diis:
            focks.pop(0)
            errs.pop(0)
        m = len(focks)
        if m >= 2:
            B = np.empty((m + 1, m + 1))
            B[-1, :] = -1.0
            B[:, -1] = -1.0
            B[-1, -1] = 0.0
            for i in range(m):
                for j in range(i + 1):
                    v = np.vdot(errs[i], errs[j])
                    B[i, j] = v
                    B[j, i] = v
            rhs = np.zeros(m + 1)
            rhs[-1] = -1.0
            try:
                c = scipy.linalg.solve(B, rhs, assume_a="sym")[:m]
            except Exception:
                c = np.linalg.lstsq(B, rhs, rcond=None)[0][:m]
            Fd = sum(ci * Fi for ci, Fi in zip(c, focks))
        else:
            Fd = F
        C = diag(Fd)
        Co = C[:, :nocc]
        D = 2.0 * Co @ Co.T

    return {"e_tot": float(e_tot), "converged": bool(converged)}
```

Write an improved program: keep every molecule converged with energy error under 1e-8 Hartree and reduce solve(mol) time. Reply with the full program.
