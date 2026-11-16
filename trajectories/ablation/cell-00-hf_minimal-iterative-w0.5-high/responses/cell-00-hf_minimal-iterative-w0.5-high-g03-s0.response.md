```python
import numpy as np


def solve(mol):
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin")
    H += mol.intor("int1e_nuc")
    eri8 = mol.intor("int2e", aosym="s8")
    n = S.shape[0]
    nocc = mol.nelectron // 2
    enuc = mol.energy_nuc()

    npair = n * (n + 1) // 2
    ip = np.arange(npair)
    A = np.maximum(ip[:, None], ip[None, :])
    B = np.minimum(ip[:, None], ip[None, :])
    eri4 = eri8[A * (A + 1) // 2 + B]
    del eri8, A, B

    r = np.arange(n)
    a = np.maximum(r[:, None], r[None, :])
    b = np.minimum(r[:, None], r[None, :])
    Pf = (a * (a + 1) // 2 + b).ravel()
    Jm = eri4[:, Pf][Pf]
    del eri4
    W = Jm - 0.5 * Jm.reshape(n, n, n, n).transpose(0, 2, 1, 3).reshape(n * n, n * n)
    del Jm

    s, U = np.linalg.eigh(S)
    keep = s > 1e-9
    X = U[:, keep] / np.sqrt(s[keep])
    XT = X.T

    def diag(F):
        e, C = np.linalg.eigh(XT @ F @ X)
        Co = X @ C[:, :nocc]
        return 2.0 * (Co @ Co.T)

    D = diag(H)
    focks, errs = [], []
    e_old = 0.0
    e_el = 0.0
    converged = False
    maxd = 8
    for it in range(200):
        G = (W @ D.ravel()).reshape(n, n)
        F = H + G
        e_el = np.vdot(D, H + 0.5 * G)
        FDS = F @ D @ S
        err = XT @ (FDS - FDS.T) @ X
        err_norm = np.abs(err).max()
        if err_norm < 2e-6 and (it == 0 or abs(e_el - e_old) < 1e-8):
            converged = True
            break
        e_old = e_el
        focks.append(F)
        errs.append(err.ravel())
        if len(focks) > maxd:
            focks.pop(0)
            errs.pop(0)
        m = len(focks)
        if m > 1:
            E = np.array(errs)
            Bm = np.empty((m + 1, m + 1))
            Bm[:m, :m] = E @ E.T
            Bm[m, :] = -1.0
            Bm[:, m] = -1.0
            Bm[m, m] = 0.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                c = np.linalg.solve(Bm, rhs)
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(Bm, rhs, rcond=None)[0]
            Fd = np.tensordot(c[:m], np.array(focks), axes=1)
        else:
            Fd = F
        D = diag(Fd)

    return {"e_tot": float(e_el + enuc), "converged": bool(converged)}
```
