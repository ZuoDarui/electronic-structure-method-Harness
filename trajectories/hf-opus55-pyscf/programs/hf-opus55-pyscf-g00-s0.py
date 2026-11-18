import numpy as np
import scipy.linalg


def _full_eri(mol, n):
    eri8 = mol.intor("int2e", aosym="s8")
    npair = n * (n + 1) // 2
    ii, jj = np.tril_indices(n)
    pidx = np.empty((n, n), dtype=np.intp)
    pidx[ii, jj] = np.arange(npair)
    pidx[jj, ii] = np.arange(npair)
    a = np.arange(npair)
    hi = np.maximum(a[:, None], a[None, :])
    lo = np.minimum(a[:, None], a[None, :])
    s4 = eri8[hi * (hi + 1) // 2 + lo]
    p = pidx.ravel()
    return s4[p[:, None], p[None, :]].reshape(n, n, n, n)


def solve(mol):
    n = mol.nao_nr()
    nocc = mol.nelectron // 2
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri = _full_eri(mol, n)
    nn = n * n
    V = eri.reshape(nn, nn) - 0.5 * eri.transpose(0, 2, 1, 3).reshape(nn, nn)
    V = np.ascontiguousarray(V)
    enuc = mol.energy_nuc()

    w, U = np.linalg.eigh(S)
    keep = w > 1e-10
    X = U[:, keep] / np.sqrt(w[keep])

    def diag(F):
        e, Cp = np.linalg.eigh(X.T @ F @ X)
        return X @ Cp

    C = diag(H)
    Co = C[:, :nocc]
    D = 2.0 * Co @ Co.T

    focks, errs = [], []
    e_old = None
    converged = False
    e_tot = 0.0
    max_diis = 8
    for it in range(200):
        G = (V @ D.ravel()).reshape(n, n)
        F = H + G
        e_tot = 0.5 * np.einsum("ij,ij->", D, H + F) + enuc
        FDS = F @ D @ S
        err = X.T @ (FDS - FDS.T) @ X
        enorm = np.max(np.abs(err))
        if e_old is not None and abs(e_tot - e_old) < 1e-11 and enorm < 1e-7:
            converged = True
            break
        e_old = e_tot
        focks.append(F)
        errs.append(err)
        if len(focks) > max_diis:
            focks.pop(0)
            errs.pop(0)
        m = len(focks)
        if m >= 2:
            B = np.empty((m + 1, m + 1))
            B[-1, :] = -1.0
            B[:, -1] = -1.0
            B[-1, -1] = 0.0
            for i in range(m):
                for j in range(i + 1):
                    v = np.vdot(errs[i], errs[j])
                    B[i, j] = v
                    B[j, i] = v
            rhs = np.zeros(m + 1)
            rhs[-1] = -1.0
            try:
                c = scipy.linalg.solve(B, rhs, assume_a="sym")[:m]
            except Exception:
                c = np.linalg.lstsq(B, rhs, rcond=None)[0][:m]
            Fd = sum(ci * Fi for ci, Fi in zip(c, focks))
        else:
            Fd = F
        C = diag(Fd)
        Co = C[:, :nocc]
        D = 2.0 * Co @ Co.T

    return {"e_tot": float(e_tot), "converged": bool(converged)}
