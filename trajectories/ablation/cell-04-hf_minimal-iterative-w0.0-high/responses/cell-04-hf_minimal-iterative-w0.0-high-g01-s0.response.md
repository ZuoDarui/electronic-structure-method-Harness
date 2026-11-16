```python
import numpy as np


def solve(mol):
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    n = S.shape[0]
    enuc = mol.energy_nuc()
    nocc = mol.nelectron // 2

    eri8 = mol.intor("int2e", aosym="s8")
    ti, tj = np.tril_indices(n)
    pidx = np.empty((n, n), dtype=np.int64)
    pidx[ti, tj] = np.arange(ti.size)
    pidx[tj, ti] = pidx[ti, tj]
    pp = pidx.ravel()
    a = np.maximum(pp[:, None], pp[None, :])
    b = np.minimum(pp[:, None], pp[None, :])
    eri = eri8[a * (a + 1) // 2 + b]
    del a, b, eri8
    G = eri.reshape(n, n, n, n).transpose(0, 2, 1, 3).reshape(n * n, n * n)
    G = eri - 0.5 * G
    del eri

    s, U = np.linalg.eigh(S)
    keep = s > 1e-10
    X = U[:, keep] / np.sqrt(s[keep])
    XT = X.T

    def density(F):
        e, C = np.linalg.eigh(XT @ F @ X)
        Co = X @ C[:, :nocc]
        return 2.0 * (Co @ Co.T)

    D = density(H)
    E_old = 0.0
    fs, es = [], []
    maxdiis = 8
    B = np.zeros((maxdiis + 1, maxdiis + 1))
    converged = False
    E = 0.0
    for it in range(200):
        F = H + (G @ D.ravel()).reshape(n, n)
        E = 0.5 * np.vdot(D, H + F) + enuc
        FDS = F @ D @ S
        err = XT @ (FDS - FDS.T) @ X
        errn = np.max(np.abs(err))
        if abs(E - E_old) < 1e-10 and errn < 1e-7:
            converged = True
            break
        E_old = E
        if len(fs) == maxdiis:
            fs.pop(0)
            es.pop(0)
            B[:maxdiis - 1, :maxdiis - 1] = B[1:maxdiis, 1:maxdiis].copy()
        fs.append(F)
        es.append(err)
        m = len(fs)
        for i in range(m):
            v = np.vdot(es[i], err)
            B[i, m - 1] = v
            B[m - 1, i] = v
        if m >= 2:
            A = np.empty((m + 1, m + 1))
            A[:m, :m] = B[:m, :m]
            A[m, :] = -1.0
            A[:, m] = -1.0
            A[m, m] = 0.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                c = np.linalg.solve(A, rhs)[:m]
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(A, rhs, rcond=None)[0][:m]
            Fd = c[0] * fs[0]
            for ci, fi in zip(c[1:], fs[1:]):
                Fd = Fd + ci * fi
        else:
            Fd = F
        D = density(Fd)

    return {"e_tot": float(E), "converged": bool(converged)}
```
