```python
import numpy as np


def solve(mol):
    S = mol.intor("int1e_ovlp")
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    n = S.shape[0]
    enuc = mol.energy_nuc()
    nocc = mol.nelectron // 2

    # Packed pair indexing: pair(i,j) = i*(i+1)/2 + j for i >= j
    I, J = np.tril_indices(n)
    npair = I.size
    ar = np.arange(npair)
    pidx = np.empty((n, n), dtype=np.intp)
    pidx[I, J] = ar
    pidx[J, I] = ar

    # 8-fold symmetric integrals, unpacked to (npair, npair)
    e8 = np.asarray(mol.intor("int2e", aosym="s8")).ravel()
    if e8.size == npair * (npair + 1) // 2:
        s4 = np.empty((npair, npair))
        ti, tj = np.tril_indices(npair)
        s4[ti, tj] = e8
        s4[tj, ti] = e8
        del ti, tj
    else:
        full = e8.reshape(n * n, n * n)
        pf = I * n + J
        s4 = full[np.ix_(pf, pf)]
    flat = s4.ravel()

    # Exchange part in pair space: (ik|jl) + (il|jk)
    pI = pidx[I]
    pJ = pidx[J]
    idx1 = pI[:, I] * npair
    idx1 += pJ[:, J]
    Kt = np.take(flat, idx1)
    idx2 = pI[:, J] * npair
    idx2 += pJ[:, I]
    del idx1
    Kt += np.take(flat, idx2)
    del idx2

    w = np.where(I == J, 1.0, 2.0)
    Kt *= -0.25
    Kt += s4
    Kt *= w[None, :]
    G = Kt  # F_pack = h_pack + G @ D_pack

    hp = h[I, J]

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
        Dp = D[I, J]
        Fp = hp + G @ Dp
        F = Fp[pidx]
        E = enuc + 0.5 * float(np.dot(w * Dp, hp + Fp))
        FDS = F @ D @ S
        comm = FDS - FDS.T
        cnorm = np.linalg.norm(comm)
        if E_old is not None and abs(E - E_old) < 1e-10 and cnorm < 1e-10:
            converged = True
            break
        E_old = E

        err = (X.T @ comm @ X).ravel()
        fock_list.append(Fp)
        err_list.append(err)
        if len(fock_list) > max_diis:
            fock_list.pop(0)
            err_list.pop(0)
        m = len(fock_list)
        if m >= 2:
            Em = np.array(err_list)
            B = np.empty((m + 1, m + 1))
            B[:m, :m] = Em @ Em.T
            B[-1, :] = -1.0
            B[:, -1] = -1.0
            B[-1, -1] = 0.0
            rhs = np.zeros(m + 1)
            rhs[-1] = -1.0
            try:
                c = np.linalg.solve(B, rhs)[:m]
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(B, rhs, rcond=None)[0][:m]
            Fd = (c @ np.array(fock_list))[pidx]
        else:
            Fd = F
        D = density(Fd)

    return {"e_tot": float(E), "converged": bool(converged)}
```
