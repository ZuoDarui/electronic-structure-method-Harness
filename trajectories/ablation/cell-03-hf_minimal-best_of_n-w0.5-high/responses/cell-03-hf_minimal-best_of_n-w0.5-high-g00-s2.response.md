```python
import numpy as np


def solve(mol):
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    n = S.shape[0]
    eri = mol.intor("int2e", aosym="s1").reshape(n, n, n, n)
    n2 = n * n
    eri_J = eri.reshape(n2, n2)
    eri_K = np.ascontiguousarray(eri.transpose(0, 2, 1, 3)).reshape(n2, n2)
    enuc = mol.energy_nuc()
    nocc = mol.nelectron // 2

    w, U = np.linalg.eigh(S)
    keep = w > 1e-9
    X = U[:, keep] / np.sqrt(w[keep])

    def density(F):
        e, C = np.linalg.eigh(X.T @ F @ X)
        Co = X @ C[:, :nocc]
        return 2.0 * (Co @ Co.T)

    def fock(D):
        d = D.ravel()
        J = (eri_J @ d).reshape(n, n)
        K = (eri_K @ d).reshape(n, n)
        return H + J - 0.5 * K

    D = density(H)
    e_old = 0.0
    focks, errs = [], []
    converged = False
    max_diis = 8
    for it in range(200):
        F = fock(D)
        e = 0.5 * np.vdot(D, H + F) + enuc
        FDS = F @ D @ S
        err = X.T @ (FDS - FDS.T) @ X
        enorm = np.max(np.abs(err))
        if it > 0 and abs(e - e_old) < 1e-11 and enorm < 1e-8:
            converged = True
            e_old = e
            break
        e_old = e
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
                for j in range(i, m):
                    v = np.vdot(errs[i], errs[j])
                    B[i, j] = B[j, i] = v
            rhs = np.zeros(m + 1)
            rhs[-1] = -1.0
            try:
                c = np.linalg.solve(B, rhs)[:m]
                Fd = sum(ci * Fi for ci, Fi in zip(c, focks))
            except np.linalg.LinAlgError:
                Fd = F
        else:
            Fd = F
        D = density(Fd)

    if converged:
        F = fock(D)
        e_old = 0.5 * np.vdot(D, H + F) + enuc
    return {"e_tot": float(e_old), "converged": bool(converged)}
```
