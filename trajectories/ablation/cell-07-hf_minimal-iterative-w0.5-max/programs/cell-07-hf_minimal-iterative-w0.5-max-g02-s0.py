import numpy as np


def solve(mol):
    n = mol.nao_nr()
    nocc = mol.nelectron // 2
    S = mol.intor('int1e_ovlp')
    H = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    eri = mol.intor('int2e', aosym='s8')
    enuc = float(mol.energy_nuc())

    npr = n * (n + 1) // 2
    tr, tc = np.tril_indices(n)
    ap = np.arange(npr)
    P = np.empty((n, n), dtype=np.intp)
    P[tr, tc] = ap
    P[tc, tr] = ap

    # M[p, q] = (i_p j_p | i_q j_q), full pair matrix unpacked from the s8 layout
    tri = ap * (ap + 1) // 2
    M = eri.take(np.maximum(tri[:, None] + ap, ap[:, None] + tri))

    # A[q, p] = (ij|kl) - 1/4 [(ik|jl) + (il|jk)],  q = (k, l), p = (i, j)
    Mr = M.ravel()
    Pk = P[tr]
    Pl = P[tc]
    A = Mr.take(Pk[:, tr] * npr + Pl[:, tc])
    A += Mr.take(Pl[:, tr] * npr + Pk[:, tc])
    A *= -0.25
    A += M

    # D = 2 * Dh, packed over lower-triangular pairs; off-diagonal pairs count twice
    w2 = np.where(tr == tc, 2.0, 4.0)
    tri_flat = tr * n + tc
    hp = H[tr, tc]

    s, U = np.linalg.eigh(S)
    X = U / np.sqrt(s)
    Xt = X.T
    Hp = Xt @ H @ X
    Cp = np.linalg.eigh(Hp)[1]

    maxd = 8
    nn = n * n
    Fs = np.empty((maxd, nn))
    Es = np.empty((maxd, nn))
    Bb = np.zeros((maxd + 1, maxd + 1))
    Bb[0, 1:] = 1.0
    Bb[1:, 0] = 1.0
    rhs = np.zeros(maxd + 1)
    rhs[0] = 1.0
    nd = 0
    head = 0
    E = 0.0
    converged = False
    for it in range(150):
        Cpo = Cp[:, :nocc]
        Co = X @ Cpo
        d = (Co @ Co.T).ravel().take(tri_flat)
        d *= w2
        g = A @ d
        E = enuc + d @ hp + 0.5 * (d @ g)
        Fp = Hp + Xt @ g.take(P) @ X
        FD = (Fp @ Cpo) @ Cpo.T
        err = FD - FD.T
        ev = err.ravel()

        k = head
        Es[k] = ev
        Fs[k] = Fp.ravel()
        if nd < maxd:
            nd += 1
        head = k + 1 if k + 1 < maxd else 0
        row = Es[:nd] @ ev
        if row[k] < 1e-12:
            converged = True
            break
        Bb[k + 1, 1:nd + 1] = row
        Bb[1:nd + 1, k + 1] = row

        Fext = Fp
        if nd > 1:
            try:
                c = np.linalg.solve(Bb[:nd + 1, :nd + 1], rhs[:nd + 1])[1:]
                ok = np.abs(c).max() < 1e4
            except np.linalg.LinAlgError:
                ok = False
            if ok:
                Fext = (c @ Fs[:nd]).reshape(n, n)
            else:
                Es[0] = ev
                Fs[0] = Fs[k]
                Bb[1, 1] = row[k]
                nd = 1
                head = 1
        if it >= 60:
            Fext = Fext + 0.5 * (np.eye(n) - Cpo @ Cpo.T)
        Cp = np.linalg.eigh(Fext)[1]

    return {"e_tot": float(E), "converged": bool(converged)}
