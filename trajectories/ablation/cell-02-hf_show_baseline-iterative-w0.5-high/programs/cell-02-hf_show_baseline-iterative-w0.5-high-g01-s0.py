import numpy as np


def _packed_g(mol, n):
    """Return the closed-shell two-electron operator in packed (i>=j) pair space.

    F_p = h_p + sum_q G[p, q] * Dpacked_q, where p=(i,j), q=(k,l) with i>=j, k>=l.
    """
    s8 = np.asarray(mol.intor("int2e", aosym="s8")).ravel()
    npair = n * (n + 1) // 2
    rows, cols = np.tril_indices(npair)
    m = np.empty((npair, npair))
    m[rows, cols] = s8
    m[cols, rows] = s8

    ii, jj = np.indices((n, n))
    hi = np.maximum(ii, jj)
    lo = np.minimum(ii, jj)
    pidx = hi * (hi + 1) // 2 + lo

    I, J = np.tril_indices(n)
    pik = pidx[np.ix_(I, I)]
    pjl = pidx[np.ix_(J, J)]
    kx = m[pik, pjl]
    del pik, pjl
    pil = pidx[np.ix_(I, J)]
    pjk = pidx[np.ix_(J, I)]
    kx += m[pil, pjk]
    del pil, pjk

    g = m
    g -= 0.25 * kx
    w = np.where(I == J, 1.0, 2.0)
    g *= w[None, :]
    return g, pidx, I, J


def solve(mol, max_iteration=200, tol=1e-10, max_diis=8):
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    s = mol.intor("int1e_ovlp")
    n = h.shape[0]
    nocc = mol.nelectron // 2
    enuc = float(mol.energy_nuc())

    g, pidx, I, J = _packed_g(mol, n)

    w, v = np.linalg.eigh(s)
    keep = w > 1e-10
    x = v[:, keep] / np.sqrt(w[keep])
    xt = x.T

    def density_from_fock(f):
        _, c = np.linalg.eigh(xt @ f @ x)
        co = x @ c[:, :nocc]
        return 2.0 * (co @ co.T)

    def build_fock(d):
        return h + (g @ d[I, J])[pidx]

    d = density_from_fock(h)

    errs = []
    focks = []
    e_prev = None
    energy = 0.0
    converged = False
    f = h

    for _ in range(max_iteration):
        f = build_fock(d)
        energy = enuc + 0.5 * float(np.vdot(d, h + f))
        fds = f @ d @ s
        comm = fds - fds.T
        cnorm = float(np.linalg.norm(comm))
        if e_prev is not None and abs(energy - e_prev) < tol and cnorm < tol:
            converged = True
            break
        e_prev = energy

        errs.append((xt @ comm @ x).ravel())
        focks.append(f)
        if len(errs) > max_diis:
            errs.pop(0)
            focks.pop(0)
        k = len(errs)
        if k >= 2:
            e_arr = np.array(errs)
            b = np.empty((k + 1, k + 1))
            b[:k, :k] = e_arr @ e_arr.T
            b[k, :k] = 1.0
            b[:k, k] = 1.0
            b[k, k] = 0.0
            rhs = np.zeros(k + 1)
            rhs[k] = 1.0
            try:
                coef = np.linalg.solve(b, rhs)[:k]
                if not np.all(np.isfinite(coef)):
                    raise np.linalg.LinAlgError
                f_use = np.tensordot(coef, np.array(focks), axes=1)
            except np.linalg.LinAlgError:
                errs = errs[-1:]
                focks = focks[-1:]
                f_use = f
        else:
            f_use = f
        d = density_from_fock(f_use)

    if not converged:
        d = density_from_fock(f)
        for _ in range(100):
            f = build_fock(d)
            energy = enuc + 0.5 * float(np.vdot(d, h + f))
            fds = f @ d @ s
            cnorm = float(np.linalg.norm(fds - fds.T))
            if e_prev is not None and abs(energy - e_prev) < tol and cnorm < tol:
                converged = True
                break
            e_prev = energy
            d = 0.5 * d + 0.5 * density_from_fock(f)

    return {"e_tot": float(energy), "converged": bool(converged)}
