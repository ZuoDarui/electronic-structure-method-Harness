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

Best program so far and its scores. time_ratio is your time divided by the reference program's time; under 1 means faster.

status: ok
- h2_sto3g: converged=True signed_err=-2.220446049250313e-16 time_ratio=0.044
- heh_sto3g: converged=True signed_err=0.0 time_ratio=0.074
- h2o_sto3g: converged=True signed_err=0.0 time_ratio=0.082
- h2_631g: converged=True signed_err=-1.021405182655144e-14 time_ratio=0.127
- h2o_631g: converged=True signed_err=1.4210854715202004e-14 time_ratio=0.198
- nh3_631gs: converged=True signed_err=9.947598300641403e-14 time_ratio=0.318
- ch4_ccpvdz: converged=True signed_err=1.4210854715202004e-13 time_ratio=0.073
max_abs_err=1.421e-13 geomean_time_ratio=0.106

```python
import numpy as np


def solve(mol):
    S = mol.intor("int1e_ovlp")
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    n = S.shape[0]
    eri = mol.intor("int2e").reshape(n, n, n, n)
    enuc = mol.energy_nuc()
    nocc = mol.nelectron // 2

    eJ = eri.reshape(n * n, n * n)
    eK = np.ascontiguousarray(eri.transpose(0, 2, 1, 3)).reshape(n * n, n * n)
    # Combined two-electron operator: G(D) = (J - 0.5 K)(D)
    G = eJ - 0.5 * eK

    s, U = np.linalg.eigh(S)
    keep = s > 1e-10
    X = U[:, keep] / np.sqrt(s[keep])

    def density(F):
        e, Cp = np.linalg.eigh(X.T @ F @ X)
        C = X @ Cp[:, :nocc]
        return 2.0 * (C @ C.T)

    D = density(h)
    E_old = None
    converged = False
    E = 0.0
    fock_list = []
    err_list = []
    max_diis = 8

    for _ in range(300):
        F = h + (G @ D.reshape(-1)).reshape(n, n)
        E = enuc + 0.5 * float(np.sum(D * (h + F)))
        FDS = F @ D @ S
        comm = FDS - FDS.T
        cnorm = np.linalg.norm(comm)
        if E_old is not None and abs(E - E_old) < 1e-10 and cnorm < 1e-10:
            converged = True
            break
        E_old = E

        err = X.T @ comm @ X
        fock_list.append(F)
        err_list.append(err)
        if len(fock_list) > max_diis:
            fock_list.pop(0)
            err_list.pop(0)
        m = len(fock_list)
        if m >= 2:
            B = np.empty((m + 1, m + 1))
            B[-1, :] = -1.0
            B[:, -1] = -1.0
            B[-1, -1] = 0.0
            for i in range(m):
                for j in range(i, m):
                    v = float(np.sum(err_list[i] * err_list[j]))
                    B[i, j] = v
                    B[j, i] = v
            rhs = np.zeros(m + 1)
            rhs[-1] = -1.0
            try:
                c = np.linalg.solve(B, rhs)[:m]
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(B, rhs, rcond=None)[0][:m]
            Fd = np.zeros_like(F)
            for ci, Fi in zip(c, fock_list):
                Fd += ci * Fi
        else:
            Fd = F
        D = density(Fd)

    return {"e_tot": float(E), "converged": bool(converged)}
```

Write an improved program: keep every molecule converged with energy error under 1e-8 Hartree and reduce solve(mol) time. Reply with the full program.
