import numpy as np
import scipy.linalg


def solve(mol):
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri = mol.intor("int2e", aosym="s1")
    n = S.shape[0]
    nocc = mol.nelectron // 2
    enuc = mol.energy_nuc()

    Jmat = eri.reshape(n * n, n * n)
    Kmat = np.ascontiguousarray(eri.transpose(0, 2, 1, 3)).reshape(n * n, n * n)
    W = Jmat - 0.5 * Kmat
    del eri, Jmat, Kmat

    s, U = np.linalg.eigh(S)
    keep = s > 1e-9
    X = U[:, keep] / np.sqrt(s[keep])

    def diag(F):
        e, C = np.linalg.eigh(X.T @ F @ X)
        C = X @ C
        Co = C[:, :nocc]
        return 2.0 * Co @ Co.T

    D = diag(H)
    focks, errs = [], []
    e_old = 0.0
    converged = False
    for it in range(200):
        G = (W @ D.ravel()).reshape(n, n)
        F = H + G
        e_el = np.vdot(D, H + 0.5 * G)
        FDS = F @ D @ S
        err = X.T @ (FDS - FDS.T) @ X
        err_norm = np.abs(err).max()
        if it > 0 and abs(e_el - e_old) < 1e-11 and err_norm < 1e-8:
            converged = True
            break
        e_old = e_el
        focks.append(F)
        errs.append(err)
        if len(focks) > 8:
            focks.pop(0)
            errs.pop(0)
        m = len(focks)
        if m > 1:
            B = np.empty((m + 1, m + 1))
            for i in range(m):
                for j in range(i + 1):
                    B[i, j] = B[j, i] = np.vdot(errs[i], errs[j])
            B[m, :] = -1.0
            B[:, m] = -1.0
            B[m, m] = 0.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                c = scipy.linalg.solve(B, rhs, assume_a="sym")
            except Exception:
                c = np.linalg.lstsq(B, rhs, rcond=None)[0]
            Fd = sum(c[i] * focks[i] for i in range(m))
        else:
            Fd = F
        D = diag(Fd)

    return {"e_tot": float(e_el + enuc), "converged": bool(converged)}
