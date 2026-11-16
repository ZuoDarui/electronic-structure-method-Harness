```python
import numpy as np


def _build_packed_g(mol, n):
    # Packed two-electron operator over symmetric pair indices (i>=j, k>=l):
    # F_pair[ij] = sum_{kl} G[ij, kl] * D[k, l]
    eri8 = mol.intor("int2e", aosym="s8")
    I, J = np.tril_indices(n)
    npair = I.size
    tri = np.empty((n, n), dtype=np.intp)
    pidx = np.arange(npair, dtype=np.intp)
    tri[I, J] = pidx
    tri[J, I] = pidx

    r, c = np.tril_indices(npair)
    eri4 = np.empty((npair, npair))
    eri4[r, c] = eri8
    eri4[c, r] = eri8
    flat = eri4.ravel()

    tI = tri[I]
    tJ = tri[J]
    G = flat[tI[:, I] * npair + tJ[:, J]]
    G += flat[tI[:, J] * npair + tJ[:, I]]
    G *= -0.25
    G += eri4
    w = np.where(I == J, 1.0, 2.0)
    G *= w[None, :]
    return G, I, J, tri


def solve(mol):
    n = mol.nao_nr()
    nocc = mol.nelectron // 2
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    enuc = mol.energy_nuc()
    G, I, J, tri = _build_packed_g(mol, n)

    s, U = np.linalg.eigh(S)
    keep = s > 1e-9
    X = U[:, keep] / np.sqrt(s[keep])
    XT = X.T

    def density(F):
        e, C = np.linalg.eigh(XT @ F @ X)
        Co = X @ C[:, :nocc]
        return 2.0 * (Co @ Co.T)

    D = density(H)
    focks, errs = [], []
    converged = False
    e_tot = 0.0
    for it in range(200):
        F = H + (G @ D[I, J])[tri]
        e_tot = 0.5 * float(np.sum(D * (H + F))) + enuc
        FDS = F @ D @ S
        err = XT @ (FDS - FDS.T) @ X
        enorm = np.max(np.abs(err))
        if enorm < 1e-7:
            converged = True
            break
        focks.append(F)
        errs.append(err)
        if len(focks) > 8:
            focks.pop(0)
            errs.pop(0)
        m = len(focks)
        if m > 1:
            B = np.empty((m + 1, m + 1))
            B[-1, :] = -1.0
            B[:, -1] = -1.0
            B[-1, -1] = 0.0
            for a in range(m):
                for b in range(a + 1):
                    v = np.sum(errs[a] * errs[b])
                    B[a, b] = B[b, a] = v
            rhs = np.zeros(m + 1)
            rhs[-1] = -1.0
            try:
                cvec = np.linalg.solve(B, rhs)[:m]
                Fd = cvec[0] * focks[0]
                for ci, Fi in zip(cvec[1:], focks[1:]):
                    Fd = Fd + ci * Fi
            except np.linalg.LinAlgError:
                Fd = F
        else:
            Fd = F
        D = density(Fd)
    return {"e_tot": float(e_tot), "converged": bool(converged)}
```
