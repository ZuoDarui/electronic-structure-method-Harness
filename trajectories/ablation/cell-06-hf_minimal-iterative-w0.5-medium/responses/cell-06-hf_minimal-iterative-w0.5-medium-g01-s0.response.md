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

    def diag(F):
        e, C = np.linalg.eigh(X.T @ F @ X)
        return X @ C[:, :nocc]

    Co = diag(H)
    D = 2.0 * Co @ Co.T

    focks, errs = [], []
    e_old = 0.0
    converged = False
    e_tot = 0.0
    for it in range(200):
        Dv = D.ravel()
        jt = eri4 @ (D[ti, tj] * offw)
        J = np.empty((n, n))
        J[ti, tj] = jt
        J[tj, ti] = jt
        K = (eriK @ Dv).reshape(n, n)
        F = H + J - 0.5 * K
        e_tot = 0.5 * np.dot(Dv, (H + F).ravel()) + enuc
        FDS = F @ D @ S
        err = X.T @ (FDS - FDS.T) @ X
        enorm = np.abs(err).max()
        if it > 0 and abs(e_tot - e_old) < 1e-10 and enorm < 1e-7:
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
            B = np.empty((m + 1, m + 1))
            for i in range(m):
                for j in range(i, m):
                    B[i, j] = B[j, i] = np.vdot(errs[i], errs[j])
            B[m, :] = -1.0
            B[:, m] = -1.0
            B[m, m] = 0.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                c = scipy.linalg.solve(B, rhs, assume_a="sym")[:m]
            except Exception:
                c = np.linalg.lstsq(B, rhs, rcond=None)[0][:m]
            Fd = c[0] * focks[0]
            for ci, Fi in zip(c[1:], focks[1:]):
                Fd = Fd + ci * Fi
        else:
            Fd = F
        Co = diag(Fd)
        D = 2.0 * Co @ Co.T

    return {"e_tot": float(e_tot), "converged": bool(converged)}
```
