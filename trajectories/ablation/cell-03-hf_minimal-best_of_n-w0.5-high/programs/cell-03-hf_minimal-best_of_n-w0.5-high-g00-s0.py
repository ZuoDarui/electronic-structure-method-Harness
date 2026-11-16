import numpy as np


def _build_g(mol, n):
    eri8 = mol.intor("int2e", aosym="s8")
    i, j = np.tril_indices(n)
    tri = np.empty((n, n), dtype=np.int64)
    tri[i, j] = i * (i + 1) // 2 + j
    tri[j, i] = tri[i, j]
    p = tri.ravel()
    a = p[:, None]
    b = p[None, :]
    big = np.maximum(a, b)
    small = np.minimum(a, b)
    eri = eri8[big * (big + 1) // 2 + small]  # (ij|kl) as (n2, n2)
    k = eri.reshape(n, n, n, n).transpose(0, 2, 1, 3).reshape(n * n, n * n)
    return eri - 0.5 * k


def solve(mol):
    n = mol.nao_nr()
    nocc = mol.nelectron // 2
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    enuc = mol.energy_nuc()
    G = _build_g(mol, n)

    s, U = np.linalg.eigh(S)
    keep = s > 1e-9
    X = U[:, keep] / np.sqrt(s[keep])

    def density(F):
        e, C = np.linalg.eigh(X.T @ F @ X)
        Co = X @ C[:, :nocc]
        return 2.0 * Co @ Co.T

    D = density(H)
    e_old = 0.0
    focks, errs = [], []
    converged = False
    e_tot = 0.0
    for it in range(200):
        F = H + (G @ D.ravel()).reshape(n, n)
        e_tot = 0.5 * np.sum(D * (H + F)) + enuc
        FDS = F @ D @ S
        err = X.T @ (FDS - FDS.T) @ X
        enorm = np.max(np.abs(err))
        if it > 0 and abs(e_tot - e_old) < 1e-11 and enorm < 1e-8:
            converged = True
            break
        e_old = e_tot
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
                c = np.linalg.solve(B, rhs)[:m]
                Fd = sum(ci * Fi for ci, Fi in zip(c, focks))
            except np.linalg.LinAlgError:
                Fd = F
        else:
            Fd = F
        D = density(Fd)
    return {"e_tot": float(e_tot), "converged": bool(converged)}
