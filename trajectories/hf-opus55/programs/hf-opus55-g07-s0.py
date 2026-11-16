import numpy as np


def solve(mol):
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin")
    H += mol.intor("int1e_nuc")
    v = mol.intor("int2e", aosym="s8")
    n = S.shape[0]
    nocc = mol.nelectron // 2
    enuc = mol.energy_nuc()

    ii, jj = np.tril_indices(n)
    npair = ii.size
    ar = np.arange(npair)
    Pm = np.empty((n, n), dtype=np.intp)
    Pm[ii, jj] = ar
    Pm[jj, ii] = ar
    flat = ii * n + jj

    L = np.zeros((npair, npair))
    L[np.tri(npair, dtype=bool)] = v
    del v
    Jm = L + L.T
    Jm.flat[::npair + 1] = np.diagonal(L)
    del L

    Pi = Pm.take(ii, axis=0)
    Pj = Pm.take(jj, axis=0)
    A = Pi.take(ii, axis=1)
    A *= npair
    A += Pj.take(jj, axis=1)
    Cx = Pi.take(jj, axis=1)
    B = Cx * npair
    B += Cx.T
    del Pi, Pj, Cx
    Jf = Jm.ravel()
    W = Jf.take(A)
    W += Jf.take(B)
    del A, B, Jf
    W *= -0.25
    W += Jm
    del Jm
    W *= np.where(ii == jj, 1.0, 2.0)[None, :]

    s, U = np.linalg.eigh(S)
    keep = s > 1e-9
    X = U[:, keep] / np.sqrt(s[keep])
    XT = np.ascontiguousarray(X.T)
    nm = X.shape[1]

    e, C = np.linalg.eigh(XT @ H @ X)
    Cocc = C[:, :nocc]

    maxdiis = 8
    Fs = np.empty((maxdiis, nm * nm))
    Es = np.empty((maxdiis, nm * nm))
    Bf = np.empty((maxdiis, maxdiis))
    converged = False
    F = H
    D = None
    for it in range(200):
        Co = X @ Cocc
        D = (2.0 * Co) @ Co.T
        F = H + (W @ D.ravel().take(flat)).take(Pm)
        Fp = XT @ F @ X
        FD = Fp @ (Cocc @ Cocc.T)
        err = FD - FD.T
        if np.abs(err).max() < 2e-6:
            converged = True
            break
        slot = it % maxdiis
        m = it + 1 if it < maxdiis else maxdiis
        Fs[slot] = Fp.ravel()
        ev = err.ravel()
        Es[slot] = ev
        row = Es[:m] @ ev
        Bf[slot, :m] = row
        Bf[:m, slot] = row
        if m >= 2:
            Bm = np.full((m + 1, m + 1), -1.0)
            Bm[:m, :m] = Bf[:m, :m]
            Bm[m, m] = 0.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                c = np.linalg.solve(Bm, rhs)[:m]
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(Bm, rhs, rcond=None)[0][:m]
            Fd = (c @ Fs[:m]).reshape(nm, nm)
        else:
            Fd = Fp
        e, C = np.linalg.eigh(Fd)
        Cocc = C[:, :nocc]

    e_tot = 0.5 * np.vdot(D, H + F) + enuc
    return {"e_tot": float(e_tot), "converged": bool(converged)}
