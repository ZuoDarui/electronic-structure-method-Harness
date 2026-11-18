import numpy as np


def solve(mol):
    S = mol.intor("int1e_ovlp")
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    n = S.shape[0]
    eri = mol.intor("int2e").reshape(n, n, n, n)
    enuc = mol.energy_nuc()
    nocc = mol.nelectron // 2

    eJ = eri.reshape(n * n, n * n)
    eK = np.ascontiguousarray(eri.transpose(0, 2, 1, 3)).reshape(n * n, n * n)
    # Combined two-electron operator: G(D) = (J - 0.5 K)(D)
    G = eJ - 0.5 * eK

    s, U = np.linalg.eigh(S)
    keep = s > 1e-10
    X = U[:, keep] / np.sqrt(s[keep])

    def density(F):
        e, Cp = np.linalg.eigh(X.T @ F @ X)
        C = X @ Cp[:, :nocc]
        return 2.0 * (C @ C.T)

    D = density(h)
    E_old = None
    converged = False
    E = 0.0
    fock_list = []
    err_list = []
    max_diis = 8

    for _ in range(300):
        F = h + (G @ D.reshape(-1)).reshape(n, n)
        E = enuc + 0.5 * float(np.sum(D * (h + F)))
        FDS = F @ D @ S
        comm = FDS - FDS.T
        cnorm = np.linalg.norm(comm)
        if E_old is not None and abs(E - E_old) < 1e-10 and cnorm < 1e-10:
            converged = True
            break
        E_old = E

        err = X.T @ comm @ X
        fock_list.append(F)
        err_list.append(err)
        if len(fock_list) > max_diis:
            fock_list.pop(0)
            err_list.pop(0)
        m = len(fock_list)
        if m >= 2:
            B = np.empty((m + 1, m + 1))
            B[-1, :] = -1.0
            B[:, -1] = -1.0
            B[-1, -1] = 0.0
            for i in range(m):
                for j in range(i, m):
                    v = float(np.sum(err_list[i] * err_list[j]))
                    B[i, j] = v
                    B[j, i] = v
            rhs = np.zeros(m + 1)
            rhs[-1] = -1.0
            try:
                c = np.linalg.solve(B, rhs)[:m]
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(B, rhs, rcond=None)[0][:m]
            Fd = np.zeros_like(F)
            for ci, Fi in zip(c, fock_list):
                Fd += ci * Fi
        else:
            Fd = F
        D = density(Fd)

    return {"e_tot": float(E), "converged": bool(converged)}
