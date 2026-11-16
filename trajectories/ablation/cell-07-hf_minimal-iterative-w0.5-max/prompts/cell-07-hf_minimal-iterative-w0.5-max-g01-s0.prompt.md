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
- heh_sto3g: converged=True signed_err=8.881784197001252e-16 time_ratio=0.036
- h2o_sto3g: converged=True signed_err=7.105427357601002e-14 time_ratio=0.049
- h2_631g: converged=True signed_err=-9.769962616701378e-15 time_ratio=0.079
- h2o_631g: converged=True signed_err=9.947598300641403e-14 time_ratio=0.069
- nh3_631gs: converged=True signed_err=1.3500311979441904e-13 time_ratio=0.111
- ch4_ccpvdz: converged=True signed_err=4.263256414560601e-13 time_ratio=0.019
max_abs_err=4.263e-13 geomean_time_ratio=0.052

```python
import numpy as np


def solve(mol):
    n = mol.nao_nr()
    nocc = mol.nelectron // 2
    S = mol.intor('int1e_ovlp')
    H = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    eri = mol.intor('int2e', aosym='s8')
    enuc = float(mol.energy_nuc())

    npr = n * (n + 1) // 2
    tr, tc = np.tril_indices(n)
    ap = np.arange(npr)
    P = np.empty((n, n), dtype=np.intp)
    P[tr, tc] = ap
    P[tc, tr] = ap

    # M[q, p] = (q|p) over packed pairs, unpacked from the s8 lower-triangular layout
    tri = ap * (ap + 1) // 2
    M = eri.take(np.maximum(tri[:, None] + ap, ap[:, None] + tri))

    # Kf[(a, b), p] = (a i_p | b j_p)
    Kf = M.ravel().take(
        (P[:, tr][:, None, :] * npr + P[:, tc][None, :, :]).reshape(n * n, npr))
    # A[q, p] = (ij|kl) - 1/4 [(ik|jl) + (il|jk)],  q = (k, l), p = (i, j)
    A = Kf[tr * n + tc]
    A += Kf[tc * n + tr]
    A *= -0.25
    A += M

    # G_packed = A @ (w * D_packed) with D = 2 * Dh; off-diagonal pairs count twice
    w2 = np.where(tr == tc, 2.0, 4.0)
    tri_flat = tr * n + tc
    hp = H[tr, tc]

    s, U = np.linalg.eigh(S)
    X = U / np.sqrt(s)
    Xt = X.T
    Hp = Xt @ H @ X
    Cp = np.linalg.eigh(Hp)[1]

    maxd = 8
    nn = n * n
    Fs = np.empty((maxd, nn))
    Es = np.empty((maxd, nn))
    Bb = np.zeros((maxd + 1, maxd + 1))
    Bb[0, 1:] = 1.0
    Bb[1:, 0] = 1.0
    rhs = np.zeros(maxd + 1)
    rhs[0] = 1.0
    nd = 0
    head = 0
    E = 0.0
    converged = False
    for it in range(150):
        Cpo = Cp[:, :nocc]
        Co = X @ Cpo
        d = (Co @ Co.T).ravel().take(tri_flat)
        d *= w2
        g = A @ d
        E = enuc + d @ hp + 0.5 * (d @ g)
        Fp = Hp + Xt @ g.take(P) @ X
        FD = (Fp @ Cpo) @ Cpo.T
        err = FD - FD.T
        ev = err.ravel()

        k = head
        Es[k] = ev
        Fs[k] = Fp.ravel()
        if nd < maxd:
            nd += 1
        head = k + 1 if k + 1 < maxd else 0
        row = Es[:nd] @ ev
        if row[k] < 1e-12:
            converged = True
            break
        Bb[k + 1, 1:nd + 1] = row
        Bb[1:nd + 1, k + 1] = row

        Fext = Fp
        if nd > 1:
            try:
                c = np.linalg.solve(Bb[:nd + 1, :nd + 1], rhs[:nd + 1])[1:]
                ok = np.abs(c).max() < 1e4
            except np.linalg.LinAlgError:
                ok = False
            if ok:
                Fext = (c @ Fs[:nd]).reshape(n, n)
            else:
                Es[0] = ev
                Fs[0] = Fs[k]
                Bb[1, 1] = row[k]
                nd = 1
                head = 1
        if it >= 60:
            Fext = Fext + 0.5 * (np.eye(n) - Cpo @ Cpo.T)
        Cp = np.linalg.eigh(Fext)[1]

    return {"e_tot": float(E), "converged": bool(converged)}
```

Write an improved program: keep every molecule converged with energy error under 1e-8 Hartree and reduce solve(mol) time. Reply with the full program.
