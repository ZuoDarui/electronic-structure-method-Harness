import numpy as np


def solve(mol):
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    n = S.shape[0]
    flat = mol.intor("int2e", aosym="s8")
    enuc = mol.energy_nuc()
    nocc = mol.nelectron // 2

    ii, jj = np.indices((n, n))
    hi = np.maximum(ii, jj)
    lo = np.minimum(ii, jj)
    P = hi * (hi + 1) // 2 + lo
    a = P[:, :, None, None]
    b = P[None, None, :, :]
    h4 = np.maximum(a, b)
    l4 = np.minimum(a, b)
    idx = h4 * (h4 + 1) // 2 + l4
    eri = flat[idx]
    n2 = n * n
    M = eri.reshape(n2, n2) - 0.5 * np.ascontiguousarray(
        eri.transpose(0, 2, 1, 3)).reshape(n2, n2)
    del eri, idx, h4, l4

    s, U = np.linalg.eigh(S)
    keep = s > 1e-10
    X = U[:, keep] / np.sqrt(s[keep])

    def density(F):
        e, C = np.linalg.eigh(X.T @ F @ X)
        Co = X @ C[:, :nocc]
        return 2.0 * Co @ Co.T

    def fock(D):
        return H + (M @ D.ravel()).reshape(n, n)

    D = density(H)
    e_old = 0.0
    focks, errs = [], []
    converged = False
    E = 0.0
    for it in range(200):
        F = fock(D)
        E = 0.5 * np.sum(D * (H + F)) + enuc
        FDS = F @ D @ S
        err = X.T @ (FDS - FDS.T) @ X
        enorm = np.max(np.abs(err))
        if abs(E - e_old) < 1e-10 and enorm < 1e-7:
            converged = True
            break
        e_old = E
        focks.append(F)
        errs.append(err)
        if len(focks) > 8:
            focks.pop(0)
            errs.pop(0)
        m = len(focks)
        if m > 1:
            B = -np.ones((m + 1, m + 1))
            B[m, m] = 0.0
            for i in range(m):
                for j in range(i, m):
                    v = np.sum(errs[i] * errs[j])
                    B[i, j] = B[j, i] = v
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                c = np.linalg.solve(B, rhs)[:m]
                Fd = sum(ci * Fi for ci, Fi in zip(c, focks))
            except np.linalg.LinAlgError:
                Fd = F
        else:
            Fd = F
        D = density(Fd)

    return {"e_tot": float(E), "converged": bool(converged)}
