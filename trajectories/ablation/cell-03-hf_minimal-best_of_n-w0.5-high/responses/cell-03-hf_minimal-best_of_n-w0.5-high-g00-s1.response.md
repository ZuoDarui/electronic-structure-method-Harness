```python
import numpy as np
import scipy.linalg


def solve(mol):
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    n = S.shape[0]
    eri = mol.intor("int2e").reshape(n, n, n, n)
    nn = n * n
    eri_J = eri.reshape(nn, nn)
    eri_K = np.ascontiguousarray(eri.transpose(0, 2, 1, 3)).reshape(nn, nn)
    enuc = mol.energy_nuc()
    nocc = mol.nelectron // 2

    s, U = np.linalg.eigh(S)
    keep = s > 1e-9
    X = U[:, keep] / np.sqrt(s[keep])

    def diag(F):
        e, C = np.linalg.eigh(X.T @ F @ X)
        return e, X @ C

    def density(C):
        Co = C[:, :nocc]
        return 2.0 * Co @ Co.T

    def fock(D):
        d = D.ravel()
        J = (eri_J @ d).reshape(n, n)
        K = (eri_K @ d).reshape(n, n)
        return H + J - 0.5 * K

    _, C = diag(H)
    D = density(C)

    fs, errs = [], []
    e_old = 0.0
    converged = False
    max_diis = 8
    for it in range(200):
        F = fock(D)
        e_elec = 0.5 * np.sum(D * (H + F))
        FDS = F @ D @ S
        err = X.T @ (FDS - FDS.T) @ X
        err_norm = np.max(np.abs(err))
        de = abs(e_elec - e_old)
        e_old = e_elec
        if it > 0 and de < 1e-11 and err_norm < 1e-8:
            converged = True
            break
        fs.append(F)
        errs.append(err)
        if len(fs) > max_diis:
            fs.pop(0)
            errs.pop(0)
        m = len(fs)
        if m >= 2:
            B = -np.ones((m + 1, m + 1))
            B[m, m] = 0.0
            for i in range(m):
                for j in range(i, m):
                    v = np.vdot(errs[i], errs[j])
                    B[i, j] = B[j, i] = v
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                c = scipy.linalg.solve(B, rhs, assume_a="sym")[:m]
                Fd = sum(ci * fi for ci, fi in zip(c, fs))
            except Exception:
                Fd = F
        else:
            Fd = F
        _, C = diag(Fd)
        D = density(C)

    F = fock(D)
    e_tot = 0.5 * np.sum(D * (H + F)) + enuc
    return {"e_tot": float(e_tot), "converged": bool(converged)}
```
