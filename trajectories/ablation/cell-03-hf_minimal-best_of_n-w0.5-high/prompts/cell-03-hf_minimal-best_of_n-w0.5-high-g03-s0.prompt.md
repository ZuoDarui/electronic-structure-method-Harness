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
- heh_sto3g: converged=True signed_err=1.687538997430238e-13 time_ratio=0.044
- h2o_sto3g: converged=True signed_err=1.2789769243681803e-13 time_ratio=0.054
- h2_631g: converged=True signed_err=1.1324274851176597e-14 time_ratio=0.073
- h2o_631g: converged=True signed_err=7.105427357601002e-14 time_ratio=0.079
- nh3_631gs: converged=True signed_err=1.2789769243681803e-13 time_ratio=0.105
- ch4_ccpvdz: converged=True signed_err=4.405364961712621e-13 time_ratio=0.022
max_abs_err=4.405e-13 geomean_time_ratio=0.056

```python
import numpy as np


def _build_packed_g(mol, n):
    # Packed two-electron operator over symmetric pair indices (i>=j, k>=l):
    # F_pair[ij] = sum_{kl} G[ij, kl] * Dpair[kl]  (Dpair already weighted)
    eri8 = mol.intor("int2e", aosym="s8")
    I, J = np.tril_indices(n)
    npair = I.size
    tri = np.empty((n, n), dtype=np.intp)
    pidx = np.arange(npair, dtype=np.intp)
    tri[I, J] = pidx
    tri[J, I] = pidx

    r, c = np.tril_indices(npair)
    eri4 = np.empty((npair, npair))
    eri4[r, c] = eri8
    eri4[c, r] = eri8
    flat = eri4.ravel()

    tI = tri[I]
    tJ = tri[J]
    G = flat[tI[:, I] * npair + tJ[:, J]]
    G += flat[tI[:, J] * npair + tJ[:, I]]
    G *= -0.25
    G += eri4
    w = np.where(I == J, 1.0, 2.0)
    G *= w[None, :]
    return G, I * n + J, tri


def solve(mol):
    n = mol.nao_nr()
    nocc = mol.nelectron // 2
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin")
    H += mol.intor("int1e_nuc")
    enuc = mol.energy_nuc()
    G, pidx, tri = _build_packed_g(mol, n)

    s, U = np.linalg.eigh(S)
    keep = s > 1e-9
    X = U[:, keep] / np.sqrt(s[keep])
    XT = X.T

    def density(F):
        e, C = np.linalg.eigh(XT @ F @ X)
        Co = X @ C[:, :nocc]
        return 2.0 * (Co @ Co.T)

    D = density(H)
    maxvec = 8
    focks = []
    errs = []
    converged = False
    F = H
    for it in range(200):
        F = H + (G @ D.take(pidx)).take(tri)
        FDS = F @ D @ S
        err = XT @ (FDS - FDS.T) @ X
        enorm = np.abs(err).max()
        if enorm < 1e-6:
            converged = True
            break
        focks.append(F)
        errs.append(err.ravel())
        if len(focks) > maxvec:
            focks.pop(0)
            errs.pop(0)
        m = len(focks)
        if m > 1:
            E = np.array(errs)
            B = np.empty((m + 1, m + 1))
            B[:m, :m] = E @ E.T
            B[m, :] = -1.0
            B[:, m] = -1.0
            B[m, m] = 0.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                cvec = np.linalg.solve(B, rhs)[:m]
                Fd = np.tensordot(cvec, np.array(focks), axes=1)
            except np.linalg.LinAlgError:
                Fd = F
        else:
            Fd = F
        D = density(Fd)
    e_tot = 0.5 * float(np.vdot(D, H + F)) + enuc
    return {"e_tot": float(e_tot), "converged": bool(converged)}
```

Write an improved program: keep every molecule converged with energy error under 1e-8 Hartree and reduce solve(mol) time. Reply with the full program.
