import numpy as np


def _pick_linalg():
    # The raw LAPACK gufuncs behind numpy.linalg skip its per-call wrapper
    # overhead; fall back to the public API if they are missing or misbehave.
    try:
        ul = np.linalg._umath_linalg
        eigh, solve1 = ul.eigh_lo, ul.solve1
        a = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])
        w, v = eigh(a)
        x = solve1(a, b)
        if (np.abs(a @ v - v * w).max() < 1e-12
                and np.abs(a @ x - b).max() < 1e-12):
            return eigh, solve1
    except Exception:
        pass
    return np.linalg.eigh, np.linalg.solve


_EIGH, _SOLVE = _pick_linalg()


def solve(mol):
    with np.errstate(all='ignore'):
        return _rhf(mol)


def _rhf(mol):
    eigh = _EIGH
    solve1 = _SOLVE
    nocc = mol.nelectron // 2
    S = mol.intor('int1e_ovlp', hermi=1)
    H = mol.intor('int1e_kin', hermi=1)
    H += mol.intor('int1e_nuc', hermi=1)
    eri = mol.intor('int2e', aosym='s8')
    enuc = float(mol.energy_nuc())

    n = S.shape[0]
    npr = n * (n + 1) // 2
    ar = np.arange(n)
    tn = np.cumsum(ar)
    tr = np.repeat(ar, ar + 1)
    ap = np.arange(npr)
    tc = ap - tn[tr]
    # P[i, j] = packed index of the pair (max(i, j), min(i, j))
    P = np.maximum(tn[:, None] + ar, tn + ar[:, None])

    # M[q, p] = (q|p) over packed pairs, unpacked from the s8 lower-triangular layout
    tri = np.cumsum(ap)
    I0 = tri[:, None] + ap
    np.maximum(I0, ap[:, None] + tri, out=I0)
    M = eri.take(I0)
    del I0

    # Kf[(a, b), p] = (a i_p | b j_p)
    Pr = P.take(tr, axis=1)
    Pr *= npr
    Kf = M.take((Pr[:, None, :] + P.take(tc, axis=1)).reshape(n * n, npr))
    # A[q, p] = (ij|kl) - 1/4 [(ik|jl) + (il|jk)],  q = (k, l), p = (i, j)
    tri_flat = tr * n + tc
    A = Kf.take(tri_flat, axis=0)
    A += Kf.take(tc * n + tr, axis=0)
    del Kf
    A *= -0.25
    A += M
    del M

    # G_packed = A @ (w2 * Dh_packed) with D = 2 * Dh; off-diagonal pairs count twice
    w2 = np.where(tr == tc, 2.0, 4.0)

    s, U = eigh(S)
    X = U / np.sqrt(s)
    Xt = X.T
    Hp = Xt @ H @ X
    Cp = eigh(Hp)[1]

    maxd = 8
    nn = n * n
    Fs = np.empty((maxd, n, n))
    Es = np.empty((maxd, n, n))
    Fs2 = Fs.reshape(maxd, nn)
    Es2 = Es.reshape(maxd, nn)
    Bb = np.zeros((maxd + 1, maxd + 1))
    Bb[0, 1:] = 1.0
    Bb[1:, 0] = 1.0
    rhs = np.zeros(maxd + 1)
    rhs[0] = 1.0
    add = np.add
    sub = np.subtract
    nd = 0
    head = 0
    converged = False
    for it in range(150):
        Cpo = Cp[:, :nocc]
        Co = X @ Cpo
        d = (Co @ Co.T).take(tri_flat)
        d *= w2
        g = A @ d
        k = head
        Fp = Fs[k]
        add(Hp, Xt @ g.take(P) @ X, out=Fp)
        FD = Fp @ Cpo @ Cpo.T
        sub(FD, FD.T, out=Es[k])
        if nd < maxd:
            nd += 1
        head = k + 1 if k + 1 < maxd else 0
        row = Es2[:nd] @ Es2[k]
        # ||[F, D]||^2; the variational energy error is quadratic in the commutator
        if row[k] < 1e-10:
            converged = True
            break
        Bb[k + 1, 1:nd + 1] = row
        Bb[1:nd + 1, k + 1] = row

        Fext = Fp
        if nd > 1:
            try:
                x = solve1(Bb[:nd + 1, :nd + 1], rhs[:nd + 1])
                ok = x @ x < 1e8
            except Exception:
                ok = False
            if ok:
                Fext = (x[1:] @ Fs2[:nd]).reshape(n, n)
            else:
                Es[0] = Es[k]
                Fs[0] = Fp
                Bb[1, 1] = row[k]
                nd = 1
                head = 1
                Fext = Fs[0]
        if it >= 60:
            Fext = Fext + 0.5 * (np.eye(n) - Cpo @ Cpo.T)
        Cp = eigh(Fext)[1]

    E = enuc + float(d @ H[tr, tc]) + 0.5 * float(d @ g)
    return {"e_tot": E, "converged": bool(converged)}
