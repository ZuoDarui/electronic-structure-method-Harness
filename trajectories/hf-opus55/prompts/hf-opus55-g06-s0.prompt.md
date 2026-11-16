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
- h2_sto3g: converged=True signed_err=0.0 time_ratio=0.047
- heh_sto3g: converged=True signed_err=7.105427357601002e-15 time_ratio=0.046
- h2o_sto3g: converged=True signed_err=1.1368683772161603e-13 time_ratio=0.054
- h2_631g: converged=True signed_err=-1.021405182655144e-14 time_ratio=0.083
- h2o_631g: converged=True signed_err=7.105427357601002e-14 time_ratio=0.079
- nh3_631gs: converged=True signed_err=1.4210854715202004e-13 time_ratio=0.138
- ch4_ccpvdz: converged=True signed_err=4.405364961712621e-13 time_ratio=0.026
max_abs_err=4.405e-13 geomean_time_ratio=0.060

```python
import numpy as np


def solve(mol):
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin")
    H += mol.intor("int1e_nuc")
    v = mol.intor("int2e", aosym="s8")
    n = S.shape[0]
    nocc = mol.nelectron // 2
    enuc = mol.energy_nuc()

    ii, jj = np.tril_indices(n)
    npair = ii.size
    ar = np.arange(npair)
    Pm = np.empty((n, n), dtype=np.intp)
    Pm[ii, jj] = ar
    Pm[jj, ii] = ar

    ta, tb = np.tril_indices(npair)
    Jm = np.empty((npair, npair))
    Jm[ta, tb] = v
    Jm[tb, ta] = v
    del ta, tb, v

    iC = ii[:, None]
    jC = jj[:, None]
    iR = ii[None, :]
    jR = jj[None, :]
    K = Jm[Pm[iC, iR], Pm[jC, jR]]
    K += Jm[Pm[iC, jR], Pm[jC, iR]]
    K *= -0.25
    K += Jm
    W = K
    del Jm
    W *= np.where(ii == jj, 1.0, 2.0)[None, :]

    s, U = np.linalg.eigh(S)
    keep = s > 1e-9
    X = U[:, keep] / np.sqrt(s[keep])
    XT = X.T

    def density(F):
        e, C = np.linalg.eigh(XT @ F @ X)
        Co = X @ C[:, :nocc]
        return 2.0 * (Co @ Co.T)

    D = density(H)
    focks, errs = [], []
    converged = False
    F = H
    maxdiis = 8
    for it in range(200):
        F = H + (W @ D[ii, jj])[Pm]
        FDS = F @ D @ S
        err = XT @ (FDS - FDS.T) @ X
        errn = np.max(np.abs(err))
        if errn < 2e-6:
            converged = True
            break
        focks.append(F)
        errs.append(err.ravel())
        if len(focks) > maxdiis:
            focks.pop(0)
            errs.pop(0)
        m = len(focks)
        if m >= 2:
            E = np.array(errs)
            B = np.empty((m + 1, m + 1))
            B[:m, :m] = E @ E.T
            B[m, :] = -1.0
            B[:, m] = -1.0
            B[m, m] = 0.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                c = np.linalg.solve(B, rhs)[:m]
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(B, rhs, rcond=None)[0][:m]
            Fd = np.tensordot(c, np.array(focks), axes=1)
        else:
            Fd = F
        D = density(Fd)

    e_tot = 0.5 * np.sum(D * (H + F)) + enuc
    return {"e_tot": float(e_tot), "converged": bool(converged)}
```

Most recent attempt, which did not beat the best program:

status: ok
- h2_sto3g: converged=True signed_err=2.220446049250313e-16 time_ratio=0.047
- heh_sto3g: converged=True signed_err=1.3509193763638905e-12 time_ratio=0.033
- h2o_sto3g: converged=True signed_err=8.526512829121202e-14 time_ratio=0.051
- h2_631g: converged=True signed_err=-7.549516567451064e-15 time_ratio=0.078
- h2o_631g: converged=True signed_err=1.311661890213145e-11 time_ratio=0.071
- nh3_631gs: converged=True signed_err=1.0018652574217413e-11 time_ratio=0.110
- ch4_ccpvdz: converged=True signed_err=4.547473508864641e-13 time_ratio=0.019
max_abs_err=1.312e-11 geomean_time_ratio=0.051

```python
import numpy as np


def solve(mol):
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin")
    H += mol.intor("int1e_nuc")
    v = mol.intor("int2e", aosym="s8")
    n = S.shape[0]
    nocc = mol.nelectron // 2
    enuc = mol.energy_nuc()

    ii, jj = np.tril_indices(n)
    npair = ii.size
    ar = np.arange(npair)
    Pm = np.empty((n, n), dtype=np.intp)
    Pm[ii, jj] = ar
    Pm[jj, ii] = ar
    flat = ii * n + jj

    L = np.zeros((npair, npair))
    L[np.tri(npair, dtype=bool)] = v
    del v
    Jm = L + L.T
    Jm.flat[::npair + 1] = np.diagonal(L)
    del L

    Pi = Pm.take(ii, axis=0)
    Pj = Pm.take(jj, axis=0)
    A = Pi.take(ii, axis=1)
    A *= npair
    A += Pj.take(jj, axis=1)
    Cx = Pi.take(jj, axis=1)
    B = Cx * npair
    B += Cx.T
    del Pi, Pj, Cx
    Jf = Jm.ravel()
    W = Jf.take(A)
    W += Jf.take(B)
    del A, B, Jf
    W *= -0.25
    W += Jm
    del Jm
    W *= np.where(ii == jj, 1.0, 2.0)[None, :]

    s, U = np.linalg.eigh(S)
    keep = s > 1e-9
    X = U[:, keep] / np.sqrt(s[keep])
    XT = np.ascontiguousarray(X.T)

    e, C = np.linalg.eigh(XT @ H @ X)
    Cocc = C[:, :nocc]

    focks, errs = [], []
    converged = False
    F = H
    D = None
    maxdiis = 8
    for it in range(200):
        Co = X @ Cocc
        D = (2.0 * Co) @ Co.T
        F = H + (W @ D.ravel().take(flat)).take(Pm)
        Fp = XT @ F @ X
        FD = Fp @ (Cocc @ Cocc.T)
        err = FD - FD.T
        if np.abs(err).max() < 2.5e-6:
            converged = True
            break
        focks.append(Fp)
        errs.append(err.ravel())
        if len(focks) > maxdiis:
            focks.pop(0)
            errs.pop(0)
        m = len(focks)
        if m >= 2:
            E = np.array(errs)
            Bm = np.empty((m + 1, m + 1))
            Bm[:m, :m] = E @ E.T
            Bm[m, :] = -1.0
            Bm[:, m] = -1.0
            Bm[m, m] = 0.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                c = np.linalg.solve(Bm, rhs)[:m]
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(Bm, rhs, rcond=None)[0][:m]
            Fd = np.tensordot(c, np.array(focks), axes=1)
        else:
            Fd = Fp
        e, C = np.linalg.eigh(Fd)
        Cocc = C[:, :nocc]

    e_tot = 0.5 * np.vdot(D, H + F) + enuc
    return {"e_tot": float(e_tot), "converged": bool(converged)}
```

Write an improved program: keep every molecule converged with energy error under 1e-8 Hartree and reduce solve(mol) time. Reply with the full program.
