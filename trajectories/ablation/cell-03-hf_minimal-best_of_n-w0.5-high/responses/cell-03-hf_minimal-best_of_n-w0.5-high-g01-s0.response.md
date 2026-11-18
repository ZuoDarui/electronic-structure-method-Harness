```python
import numpy as np


def _build_g(mol, n):
    eri8 = mol.intor("int2e", aosym="s8")
    i, j = np.tril_indices(n)
    tri = np.empty((n, n), dtype=np.int64)
    tri[i, j] = i * (i + 1) // 2 + j
    tri[j, i] = tri[i, j]
    p = tri.ravel()
    big = np.maximum(p[:, None], p[None, :])
    small = np.minimum(p[:, None], p[None, :])
    big *= big + 1
    big >>= 1
    big += small
    eri = eri8[big]  # (ij|kl) as (n2, n2)
    k = eri.reshape(n, n, n, n).transpose(0, 2, 1, 3).reshape(n * n, n * n)
    eri -= 0.5 * k
    return eri


def solve(mol):
    n = mol.nao_nr()
    nocc = mol.nelectron // 2
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin")
    H += mol.intor("int1e_nuc")
    enuc = mol.energy_nuc()
    G = _build_g(mol, n)

    s, U = np.linalg.eigh(S)
    keep = s > 1e-9
    X = U[:, keep] / np.sqrt(s[keep])
    XT = X.T

    def density(F):
        e, C = np.linalg.eigh(XT @ F @ X)
        Co = X @ C[:, :nocc]
        return 2.0 * (Co @ Co.T)

    D = density(H)
    e_old = 0.0
    maxv = 8
    focks, errs = [], []
    B = np.zeros((maxv, maxv))
    converged = False
    e_tot = 0.0
    for it in range(200):
        F = H + (G @ D.ravel()).reshape(n, n)
        e_tot = 0.5 * np.vdot(D, H + F) + enuc
        FDS = F @ D @ S
        err = XT @ (FDS - FDS.T) @ X
        enorm = np.max(np.abs(err))
        if it > 0 and enorm < 1e-7 and abs(e_tot - e_old) < 1e-9:
            converged = True
            break
        e_old = e_tot
        if len(focks) == maxv:
            focks.pop(0)
            errs.pop(0)
            B[:-1, :-1] = B[1:, 1:]
        focks.append(F)
        errs.append(err)
        m = len(focks)
        ev = err.ravel()
        for a in range(m):
            v = np.dot(errs[a].ravel(), ev)
            B[a, m - 1] = v
            B[m - 1, a] = v
        if m > 1:
            A = np.empty((m + 1, m + 1))
            A[:m, :m] = B[:m, :m]
            A[m, :] = -1.0
            A[:, m] = -1.0
            A[m, m] = 0.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                c = np.linalg.solve(A, rhs)[:m]
                Fd = c[0] * focks[0]
                for ci, Fi in zip(c[1:], focks[1:]):
                    Fd = Fd + ci * Fi
            except np.linalg.LinAlgError:
                Fd = F
        else:
            Fd = F
        D = density(Fd)
    return {"e_tot": float(e_tot), "converged": bool(converged)}
```
