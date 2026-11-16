import numpy as np


def _build_packed_g(mol, n):
    # Packed two-electron operator over symmetric pair indices (i>=j, k>=l):
    # F_pair[ij] = sum_{kl} G[ij, kl] * D_pair[kl]
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
    G[:, I != J] *= 2.0
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

    def density(F):
        _, C = np.linalg.eigh(XT @ F @ X)
        Co = X @ C[:, :nocc]
        return 2.0 * (Co @ Co.T)

    D = density(H)
    maxvec = 8
    focks, errs = [], []
    B = np.zeros((maxvec, maxvec))
    converged = False
    e_tot = 0.0
    for it in range(200):
        F = H + (G @ D[I, J])[tri]
        e_tot = 0.5 * float(np.vdot(D, H + F)) + enuc
        FDS = F @ D @ S
        err = XT @ (FDS - FDS.T) @ X
        if np.max(np.abs(err)) < 1e-7:
            converged = True
            break
        if len(focks) == maxvec:
            focks.pop(0)
            errs.pop(0)
            B[:-1, :-1] = B[1:, 1:]
        focks.append(F)
        errs.append(err)
        m = len(focks)
        for a in range(m):
            v = float(np.vdot(errs[a], err))
            B[a, m - 1] = v
            B[m - 1, a] = v
        if m > 1:
            A = np.empty((m + 1, m + 1))
            A[:m, :m] = B[:m, :m]
            A[m, :] = -1.0
            A[:, m] = -1.0
            A[m, m] = 0.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                cvec = np.linalg.solve(A, rhs)[:m]
                Fd = cvec[0] * focks[0]
                for ci, Fi in zip(cvec[1:], focks[1:]):
                    Fd += ci * Fi
            except np.linalg.LinAlgError:
                Fd = F
        else:
            Fd = F
        D = density(Fd)
    return {"e_tot": float(e_tot), "converged": bool(converged)}
