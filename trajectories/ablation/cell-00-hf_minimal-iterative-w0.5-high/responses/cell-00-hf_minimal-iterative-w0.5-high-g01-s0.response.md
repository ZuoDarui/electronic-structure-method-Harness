```python
import numpy as np
import scipy.linalg


def solve(mol):
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri8 = mol.intor("int2e", aosym="s8")
    n = S.shape[0]
    nocc = mol.nelectron // 2
    enuc = mol.energy_nuc()

    r = np.arange(n)
    a = np.maximum(r[:, None], r[None, :])
    b = np.minimum(r[:, None], r[None, :])
    Pf = (a * (a + 1) // 2 + b).ravel()
    npair = n * (n + 1) // 2
    q = np.arange(npair)
    qa = np.maximum(q[:, None], q[None, :])
    qb = np.minimum(q[:, None], q[None, :])
    eri4 = eri8[qa * (qa + 1) // 2 + qb]
    del eri8, qa, qb
    Jm = eri4[Pf][:, Pf]
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
        return 2.0 * Co @ Co.T

    D = diag(H)
    focks, errs = [], []
    e_old = 0.0
    e_el = 0.0
    converged = False
    for it in range(200):
        G = (W @ D.ravel()).reshape(n, n)
        F = H + G
        e_el = np.vdot(D, H + 0.5 * G)
        FDS = F @ D @ S
        err = XT @ (FDS - FDS.T) @ X
        err_norm = np.abs(err).max()
        if it > 0 and abs(e_el - e_old) < 1e-10 and err_norm < 1e-7:
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
            Fd = c[0] * focks[0]
            for i in range(1, m):
                Fd = Fd + c[i] * focks[i]
        else:
            Fd = F
        D = diag(Fd)

    return {"e_tot": float(e_el + enuc), "converged": bool(converged)}
```
