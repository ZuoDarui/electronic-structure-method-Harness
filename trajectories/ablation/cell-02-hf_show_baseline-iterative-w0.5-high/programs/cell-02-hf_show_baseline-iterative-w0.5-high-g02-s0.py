import math

import numpy as np


def _pair_index(n):
    ar = np.arange(n)
    hi = np.maximum.outer(ar, ar)
    pidx = hi * (hi + 1)
    pidx >>= 1
    pidx += np.minimum.outer(ar, ar)
    return pidx


def _fock_builder(mol, h, n):
    """Return fock(d) -> h + J(d) - 0.5 K(d) built from s8 integrals."""
    s8 = np.asarray(mol.intor("int2e", aosym="s8")).ravel()
    pidx = _pair_index(n)

    if n <= 9:
        n2 = n * n
        p = pidx.ravel()
        idx = np.maximum.outer(p, p)
        idx = idx * (idx + 1)
        idx >>= 1
        idx += np.minimum.outer(p, p)
        eri = np.take(s8, idx)
        g = eri - 0.5 * eri.reshape(n, n, n, n).transpose(0, 2, 1, 3).reshape(n2, n2)

        def fock(d):
            f = (g @ d.ravel()).reshape(n, n)
            f += h
            return f

        return fock

    npair = n * (n + 1) // 2
    r, c = np.tril_indices(npair)
    m = np.empty(npair * npair)
    m[r * npair + c] = s8
    m[c * npair + r] = s8
    del r, c

    I, J = np.tril_indices(n)
    pin = pidx[I] * npair
    pj = pidx[J]
    a = np.take(m, pin[:, :, None] + pj[:, None, :])
    a2 = a + a.transpose(0, 2, 1)
    del a
    trif = I * n + J
    kx = np.take(a2.reshape(npair, n * n), trif, axis=1)
    del a2

    g = m.reshape(npair, npair)
    kx *= 0.25
    g -= kx
    del kx
    g *= np.where(I == J, 1.0, 2.0)[None, :]

    def fock(d):
        f = np.take(g @ np.take(d, trif), pidx)
        f += h
        return f

    return fock


def solve(mol, max_iteration=100, tol=1e-10, max_diis=8):
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    s = mol.intor("int1e_ovlp")
    n = h.shape[0]
    nocc = mol.nelectron // 2
    enuc = float(mol.energy_nuc())
    fock = _fock_builder(mol, h, n)

    w, v = np.linalg.eigh(s)
    keep = w > 1e-10
    if keep.all():
        x = v / np.sqrt(w)
    else:
        w = w[keep]
        x = v[:, keep] / np.sqrt(w)
    wmin = float(w.min())
    xt = np.ascontiguousarray(x.T)
    mo = x.shape[1]
    sq2 = math.sqrt(2.0)

    # ||X^T C X||_F <= ||C||_F / wmin, so the AO commutator can only be below
    # tol once the orthogonal-basis error is below this gate.
    gate = tol / wmin * 1.01
    egate = max(1e-4, gate * 1e3)

    fs = np.empty((max_diis, mo * mo))
    es = np.empty((max_diis, mo * mo))
    b = np.zeros((max_diis + 1, max_diis + 1))
    b[0, 1:] = 1.0
    b[1:, 0] = 1.0
    rhs = np.zeros(max_diis + 1)
    rhs[0] = 1.0
    cnt = 0
    ptr = 0

    fo_use = xt @ h @ x
    e_prev = None
    energy = None
    converged = False
    d = None
    f = None

    for _ in range(max_iteration):
        c = np.linalg.eigh(fo_use)[1]
        co = c[:, :nocc] * sq2
        do = co @ co.T
        cx = x @ co
        d = cx @ cx.T
        f = fock(d)
        fo = xt @ f @ x
        fd = fo @ do
        ev = fd - fd.T
        evr = ev.ravel()
        nrm = math.sqrt(float(evr @ evr))

        if nrm < egate:
            energy = enuc + 0.5 * float(np.vdot(d, h) + np.vdot(d, f))
            if nrm < gate and e_prev is not None and abs(energy - e_prev) < tol:
                fds = f @ d @ s
                if float(np.linalg.norm(fds - fds.T)) < tol:
                    converged = True
                    break
            e_prev = energy
        else:
            e_prev = None

        fs[ptr] = fo.ravel()
        es[ptr] = evr
        if cnt < max_diis:
            cnt += 1
        row = es[:cnt] @ evr
        b[ptr + 1, 1:cnt + 1] = row
        b[1:cnt + 1, ptr + 1] = row
        ptr += 1
        if ptr == max_diis:
            ptr = 0

        if cnt > 1:
            try:
                coef = np.linalg.solve(b[:cnt + 1, :cnt + 1], rhs[:cnt + 1])[1:]
                if not math.isfinite(float(coef.sum())):
                    raise np.linalg.LinAlgError
                fo_use = (coef @ fs[:cnt]).reshape(mo, mo)
            except np.linalg.LinAlgError:
                fs[0] = fo.ravel()
                es[0] = evr
                b[1, 1] = float(evr @ evr)
                cnt = 1
                ptr = 1 % max_diis
                fo_use = fo
        else:
            fo_use = fo

    if not converged:
        e_prev = None
        for _ in range(200):
            f = fock(d)
            energy = enuc + 0.5 * float(np.vdot(d, h) + np.vdot(d, f))
            fds = f @ d @ s
            cnorm = float(np.linalg.norm(fds - fds.T))
            if e_prev is not None and abs(energy - e_prev) < tol and cnorm < tol:
                converged = True
                break
            e_prev = energy
            c = np.linalg.eigh(xt @ f @ x)[1]
            cx = x @ (c[:, :nocc] * sq2)
            d = 0.5 * d + 0.5 * (cx @ cx.T)

    if energy is None:
        energy = enuc + 0.5 * float(np.vdot(d, h) + np.vdot(d, f))

    return {"e_tot": float(energy), "converged": bool(converged)}
