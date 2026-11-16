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

    tI = tri[I] * npair
    tJ = tri[J]
    G = flat[tI[:, I] + tJ[:, J]]
    G += flat[tI[:, J] + tJ[:, I]]
    G *= -0.25
    G += eri4
    w = np.where(I == J, 1.0, 2.0)
    G *= w[None, :]
    return G, I, J, tri


def solve(mol):
    n = mol.nao_nr()
    nocc = mol.nelectron // 2
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin")
    H += mol.intor("int1e_nuc")
    enuc = mol.energy_nuc()
    G, I, J, tri = _build_packed_g(mol, n)

    s, U = np.linalg.eigh(S)
    keep = s > 1e-9
    X = U[:, keep] / np.sqrt(s[keep])
    XT = X.T
    eigh = np.linalg.eigh

    def density(F):
        e, C = eigh(XT @ F @ X)
        Co = X @ C[:, :nocc]
        return 2.0 * (Co @ Co.T)

    D = density(H)
    maxdiis = 8
    focks, errs = [], []
    Bfull = np.zeros((maxdiis, maxdiis))
    converged = False
    F = H
    for it in range(200):
        F = H + (G @ D[I, J])[tri]
        FDS = F @ D @ S
        err = (XT @ (FDS - FDS.T) @ X).ravel()
        enorm = np.max(np.abs(err))
        if enorm < 1e-6:
            converged = True
            break
        if len(focks) == maxdiis:
            focks.pop(0)
            errs.pop(0)
            Bfull[:-1, :-1] = Bfull[1:, 1:]
        focks.append(F)
        errs.append(err)
        m = len(focks)
        for a in range(m):
            v = float(errs[a] @ err)
            Bfull[a, m - 1] = v
            Bfull[m - 1, a] = v
        if m > 1:
            B = np.empty((m + 1, m + 1))
            B[:m, :m] = Bfull[:m, :m]
            B[m, :] = -1.0
            B[:, m] = -1.0
            B[m, m] = 0.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                cvec = np.linalg.solve(B, rhs)[:m]
                Fd = cvec[0] * focks[0]
                for ci, Fi in zip(cvec[1:], focks[1:]):
                    Fd += ci * Fi
            except np.linalg.LinAlgError:
                Fd = F
        else:
            Fd = F
        D = density(Fd)
    e_tot = 0.5 * float(np.sum(D * (H + F))) + enuc
    return {"e_tot": float(e_tot), "converged": bool(converged)}
