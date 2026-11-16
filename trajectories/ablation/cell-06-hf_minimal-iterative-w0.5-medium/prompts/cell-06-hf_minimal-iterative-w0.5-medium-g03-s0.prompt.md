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
- h2_sto3g: converged=True signed_err=2.220446049250313e-16 time_ratio=0.050
- heh_sto3g: converged=True signed_err=4.440892098500626e-15 time_ratio=0.096
- h2o_sto3g: converged=True signed_err=1.4210854715202004e-14 time_ratio=0.069
- h2_631g: converged=True signed_err=-1.021405182655144e-14 time_ratio=0.150
- h2o_631g: converged=True signed_err=5.684341886080802e-14 time_ratio=0.113
- nh3_631gs: converged=True signed_err=9.947598300641403e-14 time_ratio=0.161
- ch4_ccpvdz: converged=True signed_err=1.2789769243681803e-13 time_ratio=0.085
max_abs_err=1.279e-13 geomean_time_ratio=0.096

```python
import numpy as np
import scipy.linalg


def solve(mol):
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri4 = mol.intor("int2e", aosym="s4")
    enuc = mol.energy_nuc()
    n = S.shape[0]
    nocc = mol.nelectron // 2
    n2 = n * n

    ti, tj = np.tril_indices(n)
    P = np.empty((n, n), dtype=np.intp)
    P[ti, tj] = np.arange(ti.size)
    P[tj, ti] = P[ti, tj]
    eriK = eri4[P[:, None, :, None], P[None, :, None, :]].reshape(n2, n2)
    offw = np.where(ti == tj, 1.0, 2.0)

    w, U = np.linalg.eigh(S)
    keep = w > 1e-9
    X = U[:, keep] / np.sqrt(w[keep])

    def diag(F):
        e, C = np.linalg.eigh(X.T @ F @ X)
        return X @ C[:, :nocc]

    Co = diag(H)
    D = 2.0 * Co @ Co.T

    focks, errs = [], []
    e_old = 0.0
    converged = False
    e_tot = 0.0
    for it in range(200):
        Dv = D.ravel()
        jt = eri4 @ (D[ti, tj] * offw)
        J = np.empty((n, n))
        J[ti, tj] = jt
        J[tj, ti] = jt
        K = (eriK @ Dv).reshape(n, n)
        F = H + J - 0.5 * K
        e_tot = 0.5 * np.dot(Dv, (H + F).ravel()) + enuc
        FDS = F @ D @ S
        err = X.T @ (FDS - FDS.T) @ X
        enorm = np.abs(err).max()
        if it > 0 and abs(e_tot - e_old) < 1e-10 and enorm < 1e-7:
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
            for i in range(m):
                for j in range(i, m):
                    B[i, j] = B[j, i] = np.vdot(errs[i], errs[j])
            B[m, :] = -1.0
            B[:, m] = -1.0
            B[m, m] = 0.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                c = scipy.linalg.solve(B, rhs, assume_a="sym")[:m]
            except Exception:
                c = np.linalg.lstsq(B, rhs, rcond=None)[0][:m]
            Fd = c[0] * focks[0]
            for ci, Fi in zip(c[1:], focks[1:]):
                Fd = Fd + ci * Fi
        else:
            Fd = F
        Co = diag(Fd)
        D = 2.0 * Co @ Co.T

    return {"e_tot": float(e_tot), "converged": bool(converged)}
```

Most recent attempt, which did not beat the best program:

status: ok
- h2_sto3g: converged=True signed_err=2.220446049250313e-16 time_ratio=0.050
- heh_sto3g: converged=True signed_err=1.7763568394002505e-15 time_ratio=0.082
- h2o_sto3g: converged=True signed_err=4.263256414560601e-14 time_ratio=0.072
- h2_631g: converged=True signed_err=-9.769962616701378e-15 time_ratio=0.150
- h2o_631g: converged=True signed_err=5.684341886080802e-14 time_ratio=0.119
- nh3_631gs: converged=True signed_err=1.5631940186722204e-13 time_ratio=0.156
- ch4_ccpvdz: converged=True signed_err=4.263256414560601e-13 time_ratio=0.121
max_abs_err=4.263e-13 geomean_time_ratio=0.100

```python
import numpy as np
import scipy.linalg


def solve(mol):
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri4 = mol.intor("int2e", aosym="s4")
    enuc = mol.energy_nuc()
    n = S.shape[0]
    nocc = mol.nelectron // 2
    n2 = n * n

    ti, tj = np.tril_indices(n)
    P = np.empty((n, n), dtype=np.intp)
    P[ti, tj] = np.arange(ti.size)
    P[tj, ti] = P[ti, tj]
    eriK = eri4[P[:, None, :, None], P[None, :, None, :]].reshape(n2, n2)
    offw = np.where(ti == tj, 1.0, 2.0)

    w, U = np.linalg.eigh(S)
    keep = w > 1e-9
    X = U[:, keep] / np.sqrt(w[keep])
    XT = X.T

    def diag(F):
        e, C = np.linalg.eigh(XT @ F @ X)
        return X @ C[:, :nocc]

    Co = diag(H)
    D = 2.0 * Co @ Co.T

    focks, errs = [], []
    B = np.zeros((10, 10))
    e_old = 0.0
    converged = False
    e_tot = 0.0
    J = np.empty((n, n))
    for it in range(200):
        Dv = D.ravel()
        jt = eri4 @ (D[ti, tj] * offw)
        J[ti, tj] = jt
        J[tj, ti] = jt
        K = (eriK @ Dv).reshape(n, n)
        F = H + J - 0.5 * K
        e_tot = 0.5 * np.dot(Dv, (H + F).ravel()) + enuc
        FDS = F @ D @ S
        err = XT @ (FDS - FDS.T) @ X
        enorm = np.abs(err).max()
        if it > 0 and abs(e_tot - e_old) < 1e-9 and enorm < 1e-6:
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
            Bm = np.empty((m + 1, m + 1))
            E = np.array([e.ravel() for e in errs])
            Bm[:m, :m] = E @ E.T
            Bm[m, :] = -1.0
            Bm[:, m] = -1.0
            Bm[m, m] = 0.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                c = scipy.linalg.solve(Bm, rhs, assume_a="sym")[:m]
            except Exception:
                c = np.linalg.lstsq(Bm, rhs, rcond=None)[0][:m]
            Fd = np.tensordot(c, np.array(focks), axes=1)
        else:
            Fd = F
        Co = diag(Fd)
        D = 2.0 * Co @ Co.T

    return {"e_tot": float(e_tot), "converged": bool(converged)}
```

Write an improved program: keep every molecule converged with energy error under 1e-8 Hartree and reduce solve(mol) time. Reply with the full program.
