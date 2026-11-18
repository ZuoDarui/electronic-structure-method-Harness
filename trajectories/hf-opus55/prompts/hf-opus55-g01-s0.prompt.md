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
- h2_sto3g: converged=True signed_err=0.0 time_ratio=0.058
- heh_sto3g: converged=True signed_err=0.0 time_ratio=0.075
- h2o_sto3g: converged=True signed_err=5.684341886080802e-14 time_ratio=0.064
- h2_631g: converged=True signed_err=-1.0658141036401503e-14 time_ratio=0.113
- h2o_631g: converged=True signed_err=5.684341886080802e-14 time_ratio=0.120
- nh3_631gs: converged=True signed_err=1.1368683772161603e-13 time_ratio=0.215
- ch4_ccpvdz: converged=True signed_err=9.947598300641403e-14 time_ratio=0.121
max_abs_err=1.137e-13 geomean_time_ratio=0.100

```python
import numpy as np


def _full_eri(mol, nao):
    v = mol.intor("int2e", aosym="s8")
    npair = nao * (nao + 1) // 2
    M = np.empty((npair, npair))
    r, c = np.tril_indices(npair)
    M[r, c] = v
    M[c, r] = v
    i, j = np.tril_indices(nao)
    P = np.empty((nao, nao), dtype=np.intp)
    P[i, j] = np.arange(npair)
    P[j, i] = np.arange(npair)
    eri = M[P.reshape(-1)[:, None], P.reshape(-1)[None, :]]
    return eri  # (nao*nao, nao*nao), (ij|kl)


def solve(mol):
    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    H = T + V
    nao = S.shape[0]
    nocc = mol.nelectron // 2
    enuc = mol.energy_nuc()

    eri = _full_eri(mol, nao)
    n2 = nao * nao
    eriK = eri.reshape(nao, nao, nao, nao).transpose(0, 2, 1, 3).reshape(n2, n2).copy()

    s, U = np.linalg.eigh(S)
    keep = s > 1e-9
    X = U[:, keep] / np.sqrt(s[keep])

    def density(F):
        e, C = np.linalg.eigh(X.T @ F @ X)
        Co = X @ C[:, :nocc]
        return 2.0 * Co @ Co.T

    def fock(D):
        d = D.reshape(-1)
        J = (eri @ d).reshape(nao, nao)
        K = (eriK @ d).reshape(nao, nao)
        return H + J - 0.5 * K

    D = density(H)
    focks, errs = [], []
    e_old = 0.0
    converged = False
    e_tot = 0.0
    for it in range(200):
        F = fock(D)
        e_tot = 0.5 * np.sum(D * (H + F)) + enuc
        FDS = F @ D @ S
        err = X.T @ (FDS - FDS.T) @ X
        errn = np.max(np.abs(err))
        if it > 0 and abs(e_tot - e_old) < 1e-11 and errn < 1e-8:
            converged = True
            break
        e_old = e_tot
        focks.append(F)
        errs.append(err)
        if len(focks) > 8:
            focks.pop(0)
            errs.pop(0)
        m = len(focks)
        if m >= 2:
            B = np.empty((m + 1, m + 1))
            for a in range(m):
                for b in range(a, m):
                    B[a, b] = B[b, a] = np.sum(errs[a] * errs[b])
            B[m, :] = -1.0
            B[:, m] = -1.0
            B[m, m] = 0.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                c = np.linalg.solve(B, rhs)[:m]
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(B, rhs, rcond=None)[0][:m]
            Fd = sum(ci * Fi for ci, Fi in zip(c, focks))
        else:
            Fd = F
        D = density(Fd)

    return {"e_tot": float(e_tot), "converged": bool(converged)}
```

Write an improved program: keep every molecule converged with energy error under 1e-8 Hartree and reduce solve(mol) time. Reply with the full program.
