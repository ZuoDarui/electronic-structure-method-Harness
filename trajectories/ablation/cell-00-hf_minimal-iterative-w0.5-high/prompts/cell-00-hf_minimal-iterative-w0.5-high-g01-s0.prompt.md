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
- h2_sto3g: converged=True signed_err=2.220446049250313e-16 time_ratio=0.045
- heh_sto3g: converged=True signed_err=8.881784197001252e-16 time_ratio=0.112
- h2o_sto3g: converged=True signed_err=7.105427357601002e-14 time_ratio=0.078
- h2_631g: converged=True signed_err=-9.992007221626409e-15 time_ratio=0.151
- h2o_631g: converged=True signed_err=1.4210854715202004e-14 time_ratio=0.195
- nh3_631gs: converged=True signed_err=9.947598300641403e-14 time_ratio=0.315
- ch4_ccpvdz: converged=True signed_err=1.1368683772161603e-13 time_ratio=0.128
max_abs_err=1.137e-13 geomean_time_ratio=0.125

```python
import numpy as np
import scipy.linalg


def solve(mol):
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri = mol.intor("int2e", aosym="s1")
    n = S.shape[0]
    nocc = mol.nelectron // 2
    enuc = mol.energy_nuc()

    Jmat = eri.reshape(n * n, n * n)
    Kmat = np.ascontiguousarray(eri.transpose(0, 2, 1, 3)).reshape(n * n, n * n)
    W = Jmat - 0.5 * Kmat
    del eri, Jmat, Kmat

    s, U = np.linalg.eigh(S)
    keep = s > 1e-9
    X = U[:, keep] / np.sqrt(s[keep])

    def diag(F):
        e, C = np.linalg.eigh(X.T @ F @ X)
        C = X @ C
        Co = C[:, :nocc]
        return 2.0 * Co @ Co.T

    D = diag(H)
    focks, errs = [], []
    e_old = 0.0
    converged = False
    for it in range(200):
        G = (W @ D.ravel()).reshape(n, n)
        F = H + G
        e_el = np.vdot(D, H + 0.5 * G)
        FDS = F @ D @ S
        err = X.T @ (FDS - FDS.T) @ X
        err_norm = np.abs(err).max()
        if it > 0 and abs(e_el - e_old) < 1e-11 and err_norm < 1e-8:
            converged = True
            break
        e_old = e_el
        focks.append(F)
        errs.append(err)
        if len(focks) > 8:
            focks.pop(0)
            errs.pop(0)
        m = len(focks)
        if m > 1:
            B = np.empty((m + 1, m + 1))
            for i in range(m):
                for j in range(i + 1):
                    B[i, j] = B[j, i] = np.vdot(errs[i], errs[j])
            B[m, :] = -1.0
            B[:, m] = -1.0
            B[m, m] = 0.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                c = scipy.linalg.solve(B, rhs, assume_a="sym")
            except Exception:
                c = np.linalg.lstsq(B, rhs, rcond=None)[0]
            Fd = sum(c[i] * focks[i] for i in range(m))
        else:
            Fd = F
        D = diag(Fd)

    return {"e_tot": float(e_el + enuc), "converged": bool(converged)}
```

Write an improved program: keep every molecule converged with energy error under 1e-8 Hartree and reduce solve(mol) time. Reply with the full program.
