```python
import numpy as np
import scipy.linalg


def solve(mol):
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri4 = mol.intor("int2e", aosym="s4")
    enuc = mol.energy_nuc()
    n = S.shape[0]
    nocc = mol.nelectron // 2
    n2 = n * n

    ti, tj = np.tril_indices(n)
    P = np.empty((n, n), dtype=np.intp)
    P[ti, tj] = np.arange(ti.size)
    P[tj, ti] = P[ti, tj]
    eriK = eri4[P[:, None, :, None], P[None, :, None, :]].reshape(n2, n2)
    offw = np.where(ti == tj, 1.0, 2.0)

    w, U = np.linalg.eigh(S)
    keep = w > 1e-9
    X = U[:, keep] / np.sqrt(w[keep])
    XT = X.T

    def diag(F):
        e, C = np.linalg.eigh(XT @ F @ X)
        return X @ C[:, :nocc]

    Co = diag(H)
    D = 2.0 * Co @ Co.T

    focks, errs = [], []
    B = np.zeros((10, 10))
    e_old = 0.0
    converged = False
    e_tot = 0.0
    J = np.empty((n, n))
    for it in range(200):
        Dv = D.ravel()
        jt = eri4 @ (D[ti, tj] * offw)
        J[ti, tj] = jt
        J[tj, ti] = jt
        K = (eriK @ Dv).reshape(n, n)
        F = H + J - 0.5 * K
        e_tot = 0.5 * np.dot(Dv, (H + F).ravel()) + enuc
        FDS = F @ D @ S
        err = XT @ (FDS - FDS.T) @ X
        enorm = np.abs(err).max()
        if it > 0 and abs(e_tot - e_old) < 1e-9 and enorm < 1e-6:
            converged = True
            break
        e_old = e_tot
        focks.append(F)
        errs.append(err)
        if len(focks) > 8:
            focks.pop(0)
            errs.pop(0)
        m = len(focks)
        if m >= 2:
            Bm = np.empty((m + 1, m + 1))
            E = np.array([e.ravel() for e in errs])
            Bm[:m, :m] = E @ E.T
            Bm[m, :] = -1.0
            Bm[:, m] = -1.0
            Bm[m, m] = 0.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                c = scipy.linalg.solve(Bm, rhs, assume_a="sym")[:m]
            except Exception:
                c = np.linalg.lstsq(Bm, rhs, rcond=None)[0][:m]
            Fd = np.tensordot(c, np.array(focks), axes=1)
        else:
            Fd = F
        Co = diag(Fd)
        D = 2.0 * Co @ Co.T

    return {"e_tot": float(e_tot), "converged": bool(converged)}
```
