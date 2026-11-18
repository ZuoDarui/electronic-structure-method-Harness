import numpy as np


def solve(mol):
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
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

    ta, tb = np.tril_indices(npair)
    Jm = np.empty((npair, npair))
    Jm[ta, tb] = v
    Jm[tb, ta] = v
    del ta, tb

    iC = ii[:, None]
    jC = jj[:, None]
    iR = ii[None, :]
    jR = jj[None, :]
    K = Jm[Pm[iC, iR], Pm[jC, jR]]
    K += Jm[Pm[iC, jR], Pm[jC, iR]]
    W = Jm - 0.25 * K
    del Jm, K
    W *= np.where(ii == jj, 1.0, 2.0)[None, :]

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
    e_old = 0.0
    converged = False
    e_tot = 0.0
    maxdiis = 8
    for it in range(200):
        F = H + (W @ D[ii, jj])[Pm]
        e_tot = 0.5 * np.sum(D * (H + F)) + enuc
        FDS = F @ D @ S
        err = XT @ (FDS - FDS.T) @ X
        errn = np.max(np.abs(err))
        if it > 0 and errn < 1e-6 and abs(e_tot - e_old) < 1e-8:
            converged = True
            break
        e_old = e_tot
        focks.append(F)
        errs.append(err.ravel())
        if len(focks) > maxdiis:
            focks.pop(0)
            errs.pop(0)
        m = len(focks)
        if m >= 2:
            E = np.array(errs)
            B = np.empty((m + 1, m + 1))
            B[:m, :m] = E @ E.T
            B[m, :] = -1.0
            B[:, m] = -1.0
            B[m, m] = 0.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                c = np.linalg.solve(B, rhs)[:m]
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(B, rhs, rcond=None)[0][:m]
            Fd = np.tensordot(c, np.array(focks), axes=1)
        else:
            Fd = F
        D = density(Fd)

    return {"e_tot": float(e_tot), "converged": bool(converged)}
