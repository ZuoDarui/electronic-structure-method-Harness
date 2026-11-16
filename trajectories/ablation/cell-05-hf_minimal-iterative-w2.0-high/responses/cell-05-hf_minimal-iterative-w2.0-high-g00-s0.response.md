```python
import numpy as np


def solve(mol):
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    n = S.shape[0]
    eri = mol.intor("int2e").reshape(n, n, n, n)
    n2 = n * n
    eri_j = eri.reshape(n2, n2)
    eri_k = np.ascontiguousarray(eri.transpose(0, 2, 1, 3)).reshape(n2, n2)
    enuc = mol.energy_nuc()
    nocc = mol.nelectron // 2

    s, U = np.linalg.eigh(S)
    keep = s > 1e-10
    X = U[:, keep] / np.sqrt(s[keep])

    def density(F):
        e, C = np.linalg.eigh(X.T @ F @ X)
        Co = X @ C[:, :nocc]
        return 2.0 * Co @ Co.T

    def fock(D):
        d = D.ravel()
        J = (eri_j @ d).reshape(n, n)
        K = (eri_k @ d).reshape(n, n)
        return H + J - 0.5 * K

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
        if abs(E - e_old) < 1e-11 and enorm < 1e-8:
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
