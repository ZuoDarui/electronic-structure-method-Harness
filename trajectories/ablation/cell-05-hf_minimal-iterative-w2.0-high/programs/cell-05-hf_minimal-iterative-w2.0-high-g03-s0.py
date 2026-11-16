import numpy as np


def solve(mol):
    S = mol.intor("int1e_ovlp", hermi=1)
    H = mol.intor("int1e_kin", hermi=1) + mol.intor("int1e_nuc", hermi=1)
    flat = mol.intor("int2e", aosym="s8")
    enuc = mol.energy_nuc()
    nocc = mol.nelectron // 2
    n = S.shape[0]
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
    iic = ii[:, None]
    jjc = jj[:, None]
    iir = ii[None, :]
    jjr = jj[None, :]
    K = eri4[P[iic, iir], P[jjc, jjr]]
    K += eri4[P[iic, jjr], P[jjc, iir]]
    w = np.where(ii == jj, 1.0, 2.0)
    # Packed J - K/2 operator acting on the lower-triangle of D (off-diagonals doubled).
    Mpw = (eri4 - 0.25 * K) * w[None, :]
    del eri4, K
    tri = ii * n + jj

    s, U = np.linalg.eigh(S)
    keep = s > 1e-10
    X = U[:, keep] / np.sqrt(s[keep])
    XT = X.T
    nb = X.shape[1]

    maxd = 8
    Fs = np.empty((maxd, nb * nb))
    Es = np.empty((maxd, nb * nb))
    Bm = np.zeros((maxd, maxd))
    k = 0

    e, C = np.linalg.eigh(XT @ H @ X)
    converged = False
    D = None
    F = None
    for it in range(200):
        Co = C[:, :nocc]
        Cao = X @ Co
        D = 2.0 * (Cao @ Cao.T)
        F = H + (Mpw @ D.take(tri))[P]
        Fp = XT @ F @ X
        A = (Fp @ Co) @ Co.T
        err = 2.0 * (A - A.T)
        enorm = np.abs(err).max()
        if enorm < 2e-6:
            converged = True
            break

        slot = k % maxd
        ev = err.ravel()
        Fs[slot] = Fp.ravel()
        Es[slot] = ev
        k += 1
        m = k if k < maxd else maxd
        row = Es[:m] @ ev
        Bm[slot, :m] = row
        Bm[:m, slot] = row
        if m > 1:
            B = np.empty((m + 1, m + 1))
            sc = Bm[:m, :m].diagonal().max()
            if sc <= 0.0:
                sc = 1.0
            B[:m, :m] = Bm[:m, :m] / sc
            B[m, :m] = 1.0
            B[:m, m] = 1.0
            B[m, m] = 0.0
            rhs = np.zeros(m + 1)
            rhs[m] = 1.0
            try:
                cc = np.linalg.solve(B, rhs)[:m]
                Fd = (cc @ Fs[:m]).reshape(nb, nb)
            except np.linalg.LinAlgError:
                Fd = Fp
        else:
            Fd = Fp
        e, C = np.linalg.eigh(Fd)

    E = 0.5 * np.vdot(D, H + F) + enuc
    return {"e_tot": float(E), "converged": bool(converged)}
