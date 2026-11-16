import numpy as np


def _build_packed_g(mol, n):
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
    return G, I * n + J, tri


def solve(mol):
    n = mol.nao_nr()
    nocc = mol.nelectron // 2
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    enuc = mol.energy_nuc()
    G, pidx, tri = _build_packed_g(mol, n)

    s, U = np.linalg.eigh(S)
    keep = s > 1e-9
    X = U[:, keep] / np.sqrt(s[keep])
    XT = X.T

    def density(F):
        e, C = np.linalg.eigh(XT @ F @ X)
        Co = X @ C[:, :nocc]
        return 2.0 * (Co @ Co.T)

    D = density(H)
    maxvec = 8
    focks = []
    errs = []
    converged = False
    F = H
    for it in range(200):
        F = H + (G @ D.take(pidx)).take(tri)
        FDS = F @ D @ S
        err = XT @ (FDS - FDS.T) @ X
        enorm = np.abs(err).max()
        if enorm < 1e-6:
            converged = True
            break
        focks.append(F)
        errs.append(err.ravel())
        if len(focks) > maxvec:
            focks.pop(0)
            errs.pop(0)
        m = len(focks)
        if m > 1:
            E = np.array(errs)
            B = np.empty((m + 1, m + 1))
            B[:m, :m] = E @ E.T
            B[m, :] = -1.0
            B[:, m] = -1.0
            B[m, m] = 0.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                cvec = np.linalg.solve(B, rhs)[:m]
                Fd = np.tensordot(cvec, np.array(focks), axes=1)
            except np.linalg.LinAlgError:
                Fd = F
        else:
            Fd = F
        D = density(Fd)
    e_tot = 0.5 * float(np.vdot(D, H + F)) + enuc
    return {"e_tot": float(e_tot), "converged": bool(converged)}
