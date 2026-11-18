import numpy as np


def solve(mol):
    S = mol.intor("int1e_ovlp")
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    n = S.shape[0]
    enuc = mol.energy_nuc()
    nocc = mol.nelectron // 2

    I, J = np.tril_indices(n)
    npair = I.size
    ar = np.arange(npair)
    pidx = np.empty((n, n), dtype=np.intp)
    pidx[I, J] = ar
    pidx[J, I] = ar
    pflat = pidx.ravel()
    lowflat = I * n + J

    e8 = np.asarray(mol.intor("int2e", aosym="s8")).ravel()
    if e8.size == npair * (npair + 1) // 2:
        s4 = np.empty((npair, npair))
        ti, tj = np.tril_indices(npair)
        s4[ti, tj] = e8
        s4[tj, ti] = e8
        del ti, tj
    else:
        full = e8.reshape(n * n, n * n)
        s4 = full[np.ix_(lowflat, lowflat)]
    flat = s4.ravel()

    pI = pidx[I]
    pJ = pidx[J]
    idx = pI[:, I] * npair
    idx += pJ[:, J]
    G = np.take(flat, idx)
    idx = pI[:, J] * npair
    idx += pJ[:, I]
    G += np.take(flat, idx)
    del idx, pI, pJ

    w = np.where(I == J, 1.0, 2.0)
    G *= -0.25
    G += s4
    G *= w[None, :]

    hp = h.ravel()[lowflat]
    whp = w * hp

    s, U = np.linalg.eigh(S)
    keep = s > 1e-10
    X = U[:, keep] / np.sqrt(s[keep])
    XT = X.T.copy()
    SX = S @ X

    def coeffs(F):
        e, Cp = np.linalg.eigh(XT @ F @ X)
        return X @ Cp[:, :nocc]

    C = coeffs(h)
    D = 2.0 * (C @ C.T)
    E_old = None
    converged = False
    E = 0.0
    max_diis = 8
    nx = X.shape[1]
    Fhist = np.empty((max_diis, npair))
    Ehist = np.empty((max_diis, nx * nx))
    Bfull = np.empty((max_diis, max_diis))
    cnt = 0

    for _ in range(300):
        Dp = D.ravel()[lowflat]
        Fp = hp + G @ Dp
        F = Fp[pflat].reshape(n, n)
        E = enuc + 0.5 * float(np.dot(w * Dp, hp + Fp))
        # X^T (F D S - S D F) X = A - A^T with A = X^T F D S X
        A = (XT @ F) @ (D @ SX)
        errm = A - A.T
        cnorm = np.linalg.norm(S @ (D @ F) - F @ (D @ S)) if False else None
        FDS = F @ D @ S
        comm = FDS - FDS.T
        cnorm = np.linalg.norm(comm)
        if E_old is not None and abs(E - E_old) < 1e-10 and cnorm < 1e-10:
            converged = True
            break
        E_old = E

        slot = cnt % max_diis
        Fhist[slot] = Fp
        ev = errm.ravel()
        Ehist[slot] = ev
        cnt += 1
        m = min(cnt, max_diis)
        row = Ehist[:m] @ ev
        Bfull[slot, :m] = row
        Bfull[:m, slot] = row
        if m >= 2:
            B = np.empty((m + 1, m + 1))
            B[:m, :m] = Bfull[:m, :m]
            B[-1, :] = -1.0
            B[:, -1] = -1.0
            B[-1, -1] = 0.0
            rhs = np.zeros(m + 1)
            rhs[-1] = -1.0
            try:
                c = np.linalg.solve(B, rhs)[:m]
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(B, rhs, rcond=None)[0][:m]
            Fd = (c @ Fhist[:m])[pflat].reshape(n, n)
        else:
            Fd = F
        C = coeffs(Fd)
        D = 2.0 * (C @ C.T)

    return {"e_tot": float(E), "converged": bool(converged)}
