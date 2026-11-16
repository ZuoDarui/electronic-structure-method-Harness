import numpy as np
import scipy.linalg


def solve(mol):
    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    H = T + V
    n = S.shape[0]
    eri = mol.intor("int2e", aosym="s1").reshape(n, n, n, n)
    eri_J = eri.reshape(n * n, n * n)
    eri_K = np.ascontiguousarray(eri.transpose(0, 2, 1, 3)).reshape(n * n, n * n)
    del eri
    enuc = mol.energy_nuc()
    nocc = mol.nelectron // 2

    s, U = np.linalg.eigh(S)
    keep = s > 1e-10
    X = U[:, keep] / np.sqrt(s[keep])

    def density(F):
        e, C = np.linalg.eigh(X.T @ F @ X)
        C = X @ C
        Co = C[:, :nocc]
        return 2.0 * Co @ Co.T

    def fock(D):
        d = D.ravel()
        J = (eri_J @ d).reshape(n, n)
        K = (eri_K @ d).reshape(n, n)
        return H + J - 0.5 * K

    D = density(H)
    E_old = 0.0
    fs, es = [], []
    converged = False
    maxdiis = 8
    for it in range(200):
        F = fock(D)
        E = 0.5 * np.sum(D * (H + F)) + enuc
        FDS = F @ D @ S
        err = X.T @ (FDS - FDS.T) @ X
        errn = np.max(np.abs(err))
        if abs(E - E_old) < 1e-12 and errn < 1e-9:
            converged = True
            break
        E_old = E
        fs.append(F)
        es.append(err)
        if len(fs) > maxdiis:
            fs.pop(0)
            es.pop(0)
        m = len(fs)
        if m >= 2:
            B = np.empty((m + 1, m + 1))
            B[-1, :] = -1.0
            B[:, -1] = -1.0
            B[-1, -1] = 0.0
            for i in range(m):
                for j in range(i + 1):
                    v = np.vdot(es[i], es[j])
                    B[i, j] = v
                    B[j, i] = v
            rhs = np.zeros(m + 1)
            rhs[-1] = -1.0
            try:
                c = np.linalg.solve(B, rhs)[:m]
            except np.linalg.LinAlgError:
                c = scipy.linalg.lstsq(B, rhs)[0][:m]
            Fd = sum(ci * fi for ci, fi in zip(c, fs))
        else:
            Fd = F
        D = density(Fd)

    if converged:
        F = fock(D)
        E = 0.5 * np.sum(D * (H + F)) + enuc
    return {"e_tot": float(E), "converged": bool(converged)}
