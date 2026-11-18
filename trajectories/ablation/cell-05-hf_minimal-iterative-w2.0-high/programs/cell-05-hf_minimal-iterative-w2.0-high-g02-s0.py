import numpy as np


def solve(mol):
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    n = S.shape[0]
    flat = mol.intor("int2e", aosym="s8")
    enuc = mol.energy_nuc()
    nocc = mol.nelectron // 2

    npair = n * (n + 1) // 2
    r, c = np.tril_indices(npair)
    eri4 = np.empty((npair, npair))
    eri4[r, c] = flat
    eri4[c, r] = flat
    del r, c, flat
    ii, jj = np.tril_indices(n)
    P = np.empty((n, n), dtype=np.intp)
    ar = np.arange(npair)
    P[ii, jj] = ar
    P[jj, ii] = ar
    eri = eri4[P][:, :, P]
    del eri4
    n2 = n * n
    M = eri.reshape(n2, n2) - 0.5 * np.ascontiguousarray(
        eri.transpose(0, 2, 1, 3)).reshape(n2, n2)
    del eri

    s, U = np.linalg.eigh(S)
    keep = s > 1e-10
    X = U[:, keep] / np.sqrt(s[keep])
    XT = X.T

    def density(F):
        e, C = np.linalg.eigh(XT @ F @ X)
        Co = X @ C[:, :nocc]
        return 2.0 * (Co @ Co.T)

    def fock(D):
        return H + (M @ D.ravel()).reshape(n, n)

    D = density(H)
    e_old = 0.0
    focks, errs = [], []
    Bfull = np.zeros((9, 9))
    converged = False
    E = 0.0
    for it in range(200):
        F = fock(D)
        E = 0.5 * np.vdot(D, H + F) + enuc
        FDS = F @ D @ S
        err = XT @ (FDS - FDS.T) @ X
        enorm = np.max(np.abs(err))
        if abs(E - e_old) < 1e-9 and enorm < 1e-6:
            converged = True
            break
        e_old = E
        focks.append(F)
        errs.append(err)
        if len(focks) > 8:
            focks.pop(0)
            errs.pop(0)
            Bfull[:7, :7] = Bfull[1:8, 1:8]
        m = len(focks)
        for i in range(m):
            v = np.vdot(errs[i], err)
            Bfull[i, m - 1] = Bfull[m - 1, i] = v
        if m > 1:
            B = -np.ones((m + 1, m + 1))
            B[m, m] = 0.0
            B[:m, :m] = Bfull[:m, :m]
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                cc = np.linalg.solve(B, rhs)[:m]
                Fd = cc[0] * focks[0]
                for ci, Fi in zip(cc[1:], focks[1:]):
                    Fd = Fd + ci * Fi
            except np.linalg.LinAlgError:
                Fd = F
        else:
            Fd = F
        D = density(Fd)

    return {"e_tot": float(E), "converged": bool(converged)}
