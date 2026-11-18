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
- h2_sto3g: converged=True signed_err=-2.220446049250313e-16 time_ratio=0.046
- heh_sto3g: converged=True signed_err=0.0 time_ratio=0.061
- h2o_sto3g: converged=True signed_err=4.263256414560601e-14 time_ratio=0.059
- h2_631g: converged=True signed_err=-1.0658141036401503e-14 time_ratio=0.123
- h2o_631g: converged=True signed_err=0.0 time_ratio=0.129
- nh3_631gs: converged=True signed_err=9.947598300641403e-14 time_ratio=0.212
- ch4_ccpvdz: converged=True signed_err=1.2789769243681803e-13 time_ratio=0.036
max_abs_err=1.279e-13 geomean_time_ratio=0.080

```python
import numpy as np


def solve(mol):
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    n = S.shape[0]
    flat = mol.intor("int2e", aosym="s8")
    enuc = mol.energy_nuc()
    nocc = mol.nelectron // 2

    ii, jj = np.indices((n, n))
    hi = np.maximum(ii, jj)
    lo = np.minimum(ii, jj)
    P = hi * (hi + 1) // 2 + lo
    a = P[:, :, None, None]
    b = P[None, None, :, :]
    h4 = np.maximum(a, b)
    l4 = np.minimum(a, b)
    idx = h4 * (h4 + 1) // 2 + l4
    eri = flat[idx]
    n2 = n * n
    M = eri.reshape(n2, n2) - 0.5 * np.ascontiguousarray(
        eri.transpose(0, 2, 1, 3)).reshape(n2, n2)
    del eri, idx, h4, l4

    s, U = np.linalg.eigh(S)
    keep = s > 1e-10
    X = U[:, keep] / np.sqrt(s[keep])

    def density(F):
        e, C = np.linalg.eigh(X.T @ F @ X)
        Co = X @ C[:, :nocc]
        return 2.0 * Co @ Co.T

    def fock(D):
        return H + (M @ D.ravel()).reshape(n, n)

    D = density(H)
    e_old = 0.0
    focks, errs = [], []
    converged = False
    E = 0.0
    for it in range(200):
        F = fock(D)
        E = 0.5 * np.sum(D * (H + F)) + enuc
        FDS = F @ D @ S
        err = X.T @ (FDS - FDS.T) @ X
        enorm = np.max(np.abs(err))
        if abs(E - e_old) < 1e-10 and enorm < 1e-7:
            converged = True
            break
        e_old = E
        focks.append(F)
        errs.append(err)
        if len(focks) > 8:
            focks.pop(0)
            errs.pop(0)
        m = len(focks)
        if m > 1:
            B = -np.ones((m + 1, m + 1))
            B[m, m] = 0.0
            for i in range(m):
                for j in range(i, m):
                    v = np.sum(errs[i] * errs[j])
                    B[i, j] = B[j, i] = v
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                c = np.linalg.solve(B, rhs)[:m]
                Fd = sum(ci * Fi for ci, Fi in zip(c, focks))
            except np.linalg.LinAlgError:
                Fd = F
        else:
            Fd = F
        D = density(Fd)

    return {"e_tot": float(E), "converged": bool(converged)}
```

Write an improved program: keep every molecule converged with energy error under 1e-8 Hartree and reduce solve(mol) time. Reply with the full program.
