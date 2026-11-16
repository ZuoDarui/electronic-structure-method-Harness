```python
import numpy as np


def _full_eri(mol, nao):
    v = mol.intor("int2e", aosym="s8")
    npair = nao * (nao + 1) // 2
    M = np.empty((npair, npair))
    r, c = np.tril_indices(npair)
    M[r, c] = v
    M[c, r] = v
    i, j = np.tril_indices(nao)
    P = np.empty((nao, nao), dtype=np.intp)
    P[i, j] = np.arange(npair)
    P[j, i] = np.arange(npair)
    eri = M[P.reshape(-1)[:, None], P.reshape(-1)[None, :]]
    return eri  # (nao*nao, nao*nao), (ij|kl)


def solve(mol):
    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    H = T + V
    nao = S.shape[0]
    nocc = mol.nelectron // 2
    enuc = mol.energy_nuc()

    eri = _full_eri(mol, nao)
    n2 = nao * nao
    eriK = eri.reshape(nao, nao, nao, nao).transpose(0, 2, 1, 3).reshape(n2, n2).copy()

    s, U = np.linalg.eigh(S)
    keep = s > 1e-9
    X = U[:, keep] / np.sqrt(s[keep])

    def density(F):
        e, C = np.linalg.eigh(X.T @ F @ X)
        Co = X @ C[:, :nocc]
        return 2.0 * Co @ Co.T

    def fock(D):
        d = D.reshape(-1)
        J = (eri @ d).reshape(nao, nao)
        K = (eriK @ d).reshape(nao, nao)
        return H + J - 0.5 * K

    D = density(H)
    focks, errs = [], []
    e_old = 0.0
    converged = False
    e_tot = 0.0
    for it in range(200):
        F = fock(D)
        e_tot = 0.5 * np.sum(D * (H + F)) + enuc
        FDS = F @ D @ S
        err = X.T @ (FDS - FDS.T) @ X
        errn = np.max(np.abs(err))
        if it > 0 and abs(e_tot - e_old) < 1e-11 and errn < 1e-8:
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
            for a in range(m):
                for b in range(a, m):
                    B[a, b] = B[b, a] = np.sum(errs[a] * errs[b])
            B[m, :] = -1.0
            B[:, m] = -1.0
            B[m, m] = 0.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                c = np.linalg.solve(B, rhs)[:m]
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(B, rhs, rcond=None)[0][:m]
            Fd = sum(ci * Fi for ci, Fi in zip(c, focks))
        else:
            Fd = F
        D = density(Fd)

    return {"e_tot": float(e_tot), "converged": bool(converged)}
```
