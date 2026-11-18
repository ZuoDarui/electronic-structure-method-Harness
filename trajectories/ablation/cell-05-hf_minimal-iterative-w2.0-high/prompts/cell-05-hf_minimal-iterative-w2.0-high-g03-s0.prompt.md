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
- h2_sto3g: converged=True signed_err=2.220446049250313e-16 time_ratio=0.051
- heh_sto3g: converged=True signed_err=0.0 time_ratio=0.042
- h2o_sto3g: converged=True signed_err=2.842170943040401e-14 time_ratio=0.053
- h2_631g: converged=True signed_err=6.061817714453355e-14 time_ratio=0.059
- h2o_631g: converged=True signed_err=5.684341886080802e-14 time_ratio=0.090
- nh3_631gs: converged=True signed_err=1.8474111129762605e-13 time_ratio=0.147
- ch4_ccpvdz: converged=True signed_err=4.547473508864641e-13 time_ratio=0.028
max_abs_err=4.547e-13 geomean_time_ratio=0.059

```python
import numpy as np


def solve(mol):
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    n = S.shape[0]
    flat = mol.intor("int2e", aosym="s8")
    enuc = mol.energy_nuc()
    nocc = mol.nelectron // 2

    npair = n * (n + 1) // 2
    r, c = np.tril_indices(npair)
    eri4 = np.empty((npair, npair))
    eri4[r, c] = flat
    eri4[c, r] = flat
    del r, c, flat
    ii, jj = np.tril_indices(n)
    P = np.empty((n, n), dtype=np.intp)
    ar = np.arange(npair)
    P[ii, jj] = ar
    P[jj, ii] = ar
    eri = eri4[P][:, :, P]
    del eri4
    n2 = n * n
    M = eri.reshape(n2, n2) - 0.5 * np.ascontiguousarray(
        eri.transpose(0, 2, 1, 3)).reshape(n2, n2)
    del eri

    s, U = np.linalg.eigh(S)
    keep = s > 1e-10
    X = U[:, keep] / np.sqrt(s[keep])
    XT = X.T

    def density(F):
        e, C = np.linalg.eigh(XT @ F @ X)
        Co = X @ C[:, :nocc]
        return 2.0 * (Co @ Co.T)

    def fock(D):
        return H + (M @ D.ravel()).reshape(n, n)

    D = density(H)
    e_old = 0.0
    focks, errs = [], []
    Bfull = np.zeros((9, 9))
    converged = False
    E = 0.0
    for it in range(200):
        F = fock(D)
        E = 0.5 * np.vdot(D, H + F) + enuc
        FDS = F @ D @ S
        err = XT @ (FDS - FDS.T) @ X
        enorm = np.max(np.abs(err))
        if abs(E - e_old) < 1e-9 and enorm < 1e-6:
            converged = True
            break
        e_old = E
        focks.append(F)
        errs.append(err)
        if len(focks) > 8:
            focks.pop(0)
            errs.pop(0)
            Bfull[:7, :7] = Bfull[1:8, 1:8]
        m = len(focks)
        for i in range(m):
            v = np.vdot(errs[i], err)
            Bfull[i, m - 1] = Bfull[m - 1, i] = v
        if m > 1:
            B = -np.ones((m + 1, m + 1))
            B[m, m] = 0.0
            B[:m, :m] = Bfull[:m, :m]
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                cc = np.linalg.solve(B, rhs)[:m]
                Fd = cc[0] * focks[0]
                for ci, Fi in zip(cc[1:], focks[1:]):
                    Fd = Fd + ci * Fi
            except np.linalg.LinAlgError:
                Fd = F
        else:
            Fd = F
        D = density(Fd)

    return {"e_tot": float(E), "converged": bool(converged)}
```

Write an improved program: keep every molecule converged with energy error under 1e-8 Hartree and reduce solve(mol) time. Reply with the full program.
