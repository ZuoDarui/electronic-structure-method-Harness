import numpy as np
import scipy.linalg
from pyscf import gto
from pyscf.dft import gen_grid, radi

BLK = 128                    # screening block (grid points)
SC_BLOCKS = 128              # blocks per superchunk
SGX_TOL_ENERGY = 6.0e-13     # per (block, shell pair) energy tolerance (Hartree)
SGX_TOL_LOOSE = 1.0e-8       # used only for the early SCF iterations
LOOSE_SWITCH_GRAD = 1.0e-4
LOOSE_MAX_ITER = 40
PRESCREEN_SAFETY = 10.0
AO_CUT = 1.0e-14
COL_CUT = 1.0e-17
PAIR_CUT = 1.0e-16
J_STORE_CAP = 90.0e9
J_BLOCK_CAP = 1.0e9
AO_STORE_CAP = 50.0e9

_AUXBASIS_ALIASES = {"weigend": "def2-universal-jfit"}


def _build_auxmol(mol, auxbasis):
    name = _AUXBASIS_ALIASES.get(str(auxbasis).lower(), str(auxbasis))
    last_err = None
    for cand in (name, "def2-universal-jfit", str(auxbasis)):
        try:
            return gto.M(atom=mol._atom, basis=cand, unit="Bohr", charge=0,
                         spin=0, cart=mol.cart, verbose=0)
        except Exception as err:
            last_err = err
    raise last_err


def _build_grid(mol, rad_ang):
    rad, ang = int(rad_ang[0]), int(rad_ang[1])
    grids = gen_grid.Grids(mol)
    grids.atom_grid = {mol.atom_pure_symbol(i): (rad, ang) for i in range(mol.natm)}
    grids.radi_method = radi.treutler_ahlrichs
    grids.prune = gen_grid.nwchem_prune
    grids.build(with_non0tab=True)
    return grids


def _sub(M, rows, cols):
    if rows is None and cols is None:
        return M
    if rows is None:
        return M[:, cols]
    if cols is None:
        return M[rows]
    return M[np.ix_(rows, cols)]


def _add_sub(M, rows, cols, B):
    if rows is None and cols is None:
        M += B
    elif rows is None:
        M[:, cols] += B
    elif cols is None:
        M[rows] += B
    else:
        M[np.ix_(rows, cols)] += B


class _Ctx:
    def __init__(self, mol):
        self.mol = mol
        self.natm = mol.natm
        self.nao = mol.nao_nr()
        self.ao_loc = mol.ao_loc_nr()
        aos = np.asarray(mol.aoslice_by_atom())
        self.sh0 = aos[:, 0].astype(int)
        self.sh1 = aos[:, 1].astype(int)
        self.ao0 = aos[:, 2].astype(int)
        self.ao1 = aos[:, 3].astype(int)
        self.nfun = self.ao1 - self.ao0
        self.nsh = self.sh1 - self.sh0
        self.R = np.asarray(mol.atom_coords(), dtype=float)
        self.shoff = []
        self.shidx = []
        for a in range(self.natm):
            loc = self.ao_loc[self.sh0[a]:self.sh1[a] + 1]
            self.shoff.append((loc[:-1] - self.ao0[a]).astype(int))
            self.shidx.append(np.repeat(np.arange(self.nsh[a]), np.diff(loc)))
        self.rcut = self._cutoffs()
        self.asamp = self._pair_samples()
        self._full_layout = None

    def _cutoffs(self):
        mol = self.mol
        rs = np.arange(0.0, 60.0, 0.1)
        d = np.array([0.62, 0.48, 0.36])
        d /= np.linalg.norm(d)
        cache = {}
        rc = np.zeros(self.natm)
        for a in range(self.natm):
            key = mol.atom_symbol(a)
            if key not in cache:
                pts = np.ascontiguousarray(self.R[a] + rs[:, None] * d)
                vals = mol.eval_gto("GTOval", pts,
                                    shls_slice=(int(self.sh0[a]), int(self.sh1[a])))
                m = np.abs(np.asarray(vals)).max(axis=1)
                idx = np.nonzero(m > AO_CUT)[0]
                cache[key] = (rs[idx[-1]] if idx.size else 0.0) + 0.5
            rc[a] = cache[key]
        return rc

    def _pair_samples(self):
        mol = self.mol
        R = self.R
        natm = self.natm
        offs = [np.zeros(3)]
        for dlt in (0.4, 1.2):
            for ax in range(3):
                for sg in (1.0, -1.0):
                    e = np.zeros(3)
                    e[ax] = sg * dlt
                    offs.append(e)
        offs = np.array(offs)
        tline = np.array([0.25, 0.5, 0.75])[:, None]
        dist = np.sqrt(((R[:, None, :] - R[None, :, :]) ** 2).sum(-1))
        asamp = np.zeros((natm, natm))
        for a in range(natm):
            for b in range(a + 1):
                if dist[a, b] > self.rcut[a] + self.rcut[b]:
                    continue
                if a == b:
                    pts = R[a] + offs
                else:
                    line = R[b][None, :] + tline * (R[a] - R[b])[None, :]
                    pts = np.vstack([R[a] + offs, R[b] + offs, line])
                v = mol.intor("int1e_grids", grids=np.ascontiguousarray(pts),
                              shls_slice=(int(self.sh0[a]), int(self.sh1[a]),
                                          int(self.sh0[b]), int(self.sh1[b])))
                val = float(np.abs(np.asarray(v)).max())
                asamp[a, b] = asamp[b, a] = val
        return asamp

    def layout(self, atoms):
        full = len(atoms) == self.natm
        if full and self._full_layout is not None:
            return self._full_layout
        nf = self.nfun[atoms]
        col_off = np.zeros(len(atoms) + 1, dtype=int)
        col_off[1:] = np.cumsum(nf)
        if full:
            cols = None
        else:
            cols = np.concatenate([np.arange(self.ao0[a], self.ao1[a]) for a in atoms])
        sh_starts = np.concatenate([self.shoff[a] + col_off[k]
                                    for k, a in enumerate(atoms)])
        sh_off = np.zeros(len(atoms) + 1, dtype=int)
        sh_off[1:] = np.cumsum(self.nsh[atoms])
        out = (cols, col_off, sh_starts, sh_off)
        if full:
            self._full_layout = out
        return out

    def eval_ao(self, c, atoms):
        parts = []
        i = 0
        n = len(atoms)
        while i < n:
            j = i
            while j + 1 < n and atoms[j + 1] == atoms[j] + 1:
                j += 1
            parts.append(np.asarray(self.mol.eval_gto(
                "GTOval", c, shls_slice=(int(self.sh0[atoms[i]]), int(self.sh1[atoms[j]])))))
            i = j + 1
        if len(parts) == 1:
            return np.ascontiguousarray(parts[0])
        return np.hstack(parts)

    def atom_absmax(self, M):
        r = np.add.reduceat(np.abs(M), self.ao0, axis=0)
        return np.maximum.reduceat(r, self.ao0, axis=1)


class _JBuilder:
    """J_mn = sum_PQ (mn|P) [(P|Q)^-1] (Q|ls) D_ls over atom-pair blocks."""

    def __init__(self, ctx, auxmol):
        self.ctx = ctx
        mol = ctx.mol
        self.pmol = mol + auxmol
        self.nbas = mol.nbas
        self.aux_nbas = auxmol.nbas
        self.naux = auxmol.nao_nr()
        self.cho = scipy.linalg.cho_factor(auxmol.intor("int2c2e"))
        tasks = []
        for a in range(ctx.natm):
            na = int(ctx.nfun[a])
            lim = max(1, int(J_BLOCK_CAP / (8.0 * na * self.naux)))
            run = []
            for b in range(a + 1):
                if ctx.asamp[a, b] <= PAIR_CUT:
                    continue
                if run and (b != run[-1] + 1 or (ctx.ao1[b] - ctx.ao0[run[0]]) > lim):
                    tasks.append((a, run[0], run[-1] + 1))
                    run = []
                run.append(b)
            if run:
                tasks.append((a, run[0], run[-1] + 1))
        self.tasks = tasks
        self.store = []
        used = 0.0
        for t in tasks:
            a, b0, b1 = t
            size = 8.0 * ctx.nfun[a] * (ctx.ao1[b1 - 1] - ctx.ao0[b0]) * self.naux
            if used + size <= J_STORE_CAP:
                self.store.append(self._compute(t))
                used += size
            else:
                self.store.append(None)

    def _compute(self, t):
        a, b0, b1 = t
        c = self.ctx
        T = self.pmol.intor("int3c2e", shls_slice=(
            int(c.sh0[a]), int(c.sh1[a]), int(c.sh0[b0]), int(c.sh1[b1 - 1]),
            self.nbas, self.nbas + self.aux_nbas), aosym="s1")
        T = np.asarray(T)
        return np.ascontiguousarray(T.transpose(2, 1, 0)).reshape(self.naux, -1)

    def get_j(self, D):
        c = self.ctx
        gamma = np.zeros(self.naux)
        for t, M in zip(self.tasks, self.store):
            if M is None:
                M = self._compute(t)
            a, b0, b1 = t
            A0, A1 = c.ao0[a], c.ao1[a]
            B0, B1 = c.ao0[b0], c.ao1[b1 - 1]
            W = 2.0 * D[B0:B1, A0:A1]
            if b1 - 1 == a:
                W[A0 - B0:A1 - B0, :] *= 0.5
            gamma += M @ W.ravel()
        x = scipy.linalg.cho_solve(self.cho, gamma)
        J = np.zeros((c.nao, c.nao))
        for t, M in zip(self.tasks, self.store):
            if M is None:
                M = self._compute(t)
            a, b0, b1 = t
            A0, A1 = c.ao0[a], c.ao1[a]
            B0, B1 = c.ao0[b0], c.ao1[b1 - 1]
            Jt = (x @ M).reshape(B1 - B0, A1 - A0)
            J[B0:B1, A0:A1] = Jt
            J[A0:A1, B0:B1] = Jt.T
        return J


class _Chunk:
    pass


class _KGrid:
    """Overlap-fitted seminumerical exchange with energy-based screening."""

    def __init__(self, ctx, rad_ang, S):
        self.ctx = ctx
        mol = ctx.mol
        nao = ctx.nao
        grids = _build_grid(mol, rad_ang)
        coords = np.asarray(grids.coords, dtype=float)
        weights = np.asarray(grids.weights, dtype=float)
        del grids
        ng = weights.size
        R = ctx.R
        RR = (R ** 2).sum(1)
        rc2 = ctx.rcut ** 2
        Snum = np.zeros((nao, nao))
        self.chunks = []
        used = 0.0
        step = BLK * SC_BLOCKS
        for p0 in range(0, ng, step):
            p1 = min(ng, p0 + step)
            n = p1 - p0
            nb = (n + BLK - 1) // BLK
            npad = nb * BLK
            c = np.empty((npad, 3))
            c[:n] = coords[p0:p1]
            c[n:] = coords[p1 - 1]
            w = np.zeros(npad)
            w[:n] = weights[p0:p1]
            d2 = (c ** 2).sum(1)[:, None] + RR[None, :] - 2.0 * (c @ R.T)
            atoms = np.nonzero(d2.min(axis=0) < rc2)[0]
            if atoms.size == 0:
                continue
            cols, col_off, _, _ = ctx.layout(atoms)
            ao = ctx.eval_ao(c, atoms)
            ch = _Chunk()
            ch.c = c
            ch.w = w
            ch.wmax = float(w.max())
            ch.nb = nb
            ch.npad = npad
            ch.atoms = atoms
            ch.cols = cols
            ch.aomax = np.maximum.reduceat(np.abs(ao).max(axis=0), col_off[:-1])
            wao = ao * w[:, None]
            _add_sub(Snum, cols, cols, ao.T @ wao)
            size = 8.0 * ao.size
            if used + size <= AO_STORE_CAP:
                ch.ao = ao
                used += size
            else:
                ch.ao = None
            self.chunks.append(ch)
        self.Q = scipy.linalg.solve(Snum, S)

    def get_k(self, D, tol):
        ctx = self.ctx
        mol = ctx.mol
        nao = ctx.nao
        QD = self.Q @ D
        Dat = ctx.atom_absmax(D)
        QDat = ctx.atom_absmax(QD)
        K = np.zeros((nao, nao))
        ar = np.arange(BLK)
        pre = 0.25 * PRESCREEN_SAFETY
        for ch in self.chunks:
            Fb = ch.aomax @ QDat[ch.atoms]
            Gb = (ch.aomax @ Dat[ch.atoms]) * ch.wmax
            Fatoms = np.nonzero(Fb > COL_CUT)[0]
            Gatoms = np.nonzero(Gb > COL_CUT)[0]
            if Fatoms.size == 0 or Gatoms.size == 0:
                continue
            ao = ch.ao if ch.ao is not None else ctx.eval_ao(ch.c, ch.atoms)
            w = ch.w
            fcols, fcol_off, fsh_starts, fsh_off = ctx.layout(Fatoms)
            gcols, gcol_off, gsh_starts, gsh_off = ctx.layout(Gatoms)
            rows = ch.cols
            F = ao @ _sub(QD, rows, fcols)
            Gw = ao @ _sub(D, rows, gcols)
            Gw *= w[:, None]
            Fsh = np.maximum.reduceat(np.abs(F), fsh_starts, axis=1)
            Gsh = np.maximum.reduceat(np.abs(Gw), gsh_starts, axis=1)
            Fat = np.maximum.reduceat(Fsh, fsh_off[:-1], axis=1)
            Gat = np.maximum.reduceat(Gsh, gsh_off[:-1], axis=1)
            nb = ch.nb
            Fsh_r = Fsh.reshape(nb, BLK, -1)
            Gsh_r = Gsh.reshape(nb, BLK, -1)
            WGFat = np.matmul(Gat.reshape(nb, BLK, -1).transpose(0, 2, 1),
                              Fat.reshape(nb, BLK, -1))
            As = pre * ctx.asamp[np.ix_(Gatoms, Fatoms)]
            Pmax = As * WGFat.max(axis=0)
            v = np.zeros((ch.npad, Gw.shape[1]))
            for i, j in np.argwhere(Pmax > tol):
                cand = np.nonzero(WGFat[:, i, j] * As[i, j] > tol)[0]
                if cand.size == 0:
                    continue
                N = Gatoms[i]
                Sa = Fatoms[j]
                nN = int(ctx.nfun[N])
                nS = int(ctx.nfun[Sa])
                nc = cand.size
                pts = (cand[:, None] * BLK + ar[None, :]).ravel()
                A = mol.intor("int1e_grids", grids=ch.c[pts],
                              shls_slice=(int(ctx.sh0[N]), int(ctx.sh1[N]),
                                          int(ctx.sh0[Sa]), int(ctx.sh1[Sa])))
                A = np.asarray(A).reshape(nc, BLK, nN, nS)
                Ab = np.abs(A).max(axis=1)
                Ash = np.maximum.reduceat(
                    np.maximum.reduceat(Ab, ctx.shoff[N], axis=1), ctx.shoff[Sa], axis=2)
                gs0, gs1 = gsh_off[i], gsh_off[i + 1]
                fs0, fs1 = fsh_off[j], fsh_off[j + 1]
                WGF = np.matmul(Gsh_r[cand, :, gs0:gs1].transpose(0, 2, 1),
                                Fsh_r[cand, :, fs0:fs1])
                keep = (0.25 * Ash * WGF) > tol
                if not keep.any():
                    continue
                if not keep.all():
                    km = keep[:, ctx.shidx[N]][:, :, ctx.shidx[Sa]]
                    A = A * km[:, None, :, :]
                fc0, fc1 = fcol_off[j], fcol_off[j + 1]
                gc0, gc1 = gcol_off[i], gcol_off[i + 1]
                Fsub = F[pts, fc0:fc1].reshape(nc, BLK, nS)
                vg = np.matmul(A, Fsub[..., None])[..., 0]
                v[pts, gc0:gc1] += vg.reshape(-1, nN)
            Kb = (ao * w[:, None]).T @ v
            _add_sub(K, rows, gcols, Kb)
        return 0.5 * (K + K.T)


class _DIIS:
    def __init__(self, space=8):
        self.space = space
        self.F = []
        self.E = []
        self.B = np.zeros((0, 0))

    def update(self, F, err):
        e = err.ravel()
        if len(self.F) == self.space:
            self.F.pop(0)
            self.E.pop(0)
            self.B = self.B[1:, 1:]
        self.F.append(F)
        self.E.append(e)
        n = len(self.F)
        Bn = np.zeros((n, n))
        Bn[:n - 1, :n - 1] = self.B
        for i in range(n):
            val = float(np.dot(self.E[i], e))
            Bn[i, n - 1] = Bn[n - 1, i] = val
        self.B = Bn
        if n == 1:
            return F
        scale = max(np.abs(np.diag(Bn)).max(), 1e-300)
        H = np.zeros((n + 1, n + 1))
        H[:n, :n] = Bn / scale
        H[n, :n] = 1.0
        H[:n, n] = 1.0
        rhs = np.zeros(n + 1)
        rhs[n] = 1.0
        try:
            c = np.linalg.solve(H, rhs)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(H, rhs, rcond=None)[0]
        Fn = c[0] * self.F[0]
        for i in range(1, n):
            Fn = Fn + c[i] * self.F[i]
        return Fn


def _scf(jb, kg, hcore, X, enuc, nocc, Co, conv_tol, loose, max_cycle=200):
    tol = SGX_TOL_LOOSE if loose else SGX_TOL_ENERGY
    loose_phase = loose
    n_loose = 0
    diis = _DIIS(8)
    e_old = None
    converged = False
    E = 0.0
    D = J = None
    for cyc in range(max_cycle):
        Cao = X @ Co
        D = 2.0 * (Cao @ Cao.T)
        J = jb.get_j(D)
        while True:
            K = kg.get_k(D, tol)
            F = hcore + J - 0.5 * K
            E = enuc + 0.5 * float(np.vdot(D, hcore + F))
            Fo = X @ F @ X
            FC = Fo @ Co
            Gr = FC - Co @ (Co.T @ FC)
            gnorm = 2.0 * float(np.linalg.norm(Gr))
            if loose_phase:
                n_loose += 1
                if gnorm < LOOSE_SWITCH_GRAD or n_loose >= LOOSE_MAX_ITER:
                    loose_phase = False
                    tol = SGX_TOL_ENERGY
                    diis = _DIIS(8)
                    e_old = None
                    continue
            break
        if (not loose_phase) and e_old is not None and \
                abs(E - e_old) < conv_tol and gnorm < conv_tol:
            converged = True
            break
        e_old = E
        err = FC @ Co.T
        err = err - err.T
        Fd = diis.update(Fo, err)
        _, C = np.linalg.eigh(Fd)
        Co = np.ascontiguousarray(C[:, :nocc])
    return E, D, Co, J, converged


def solve(mol, method):
    S = mol.intor_symmetric("int1e_ovlp")
    hcore = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
    enuc = mol.energy_nuc()
    nocc = mol.nelectron // 2
    conv_tol = float(method.get("conv_tol", 1e-8))

    sval, svec = np.linalg.eigh(S)
    X = (svec * (1.0 / np.sqrt(sval))) @ svec.T

    ctx = _Ctx(mol)
    auxmol = _build_auxmol(mol, method.get("auxbasis", "weigend"))
    jb = _JBuilder(ctx, auxmol)

    kg = _KGrid(ctx, method.get("grid", [30, 110]), S)
    _, C0 = np.linalg.eigh(X @ hcore @ X)
    Co = np.ascontiguousarray(C0[:, :nocc])

    E, D, Co, J, converged = _scf(jb, kg, hcore, X, enuc, nocc, Co,
                                  conv_tol, loose=True)
    del kg

    final_grid = method.get("final_grid", method.get("grid", [30, 110]))
    kg2 = _KGrid(ctx, final_grid, S)
    if method.get("final_energy", "single_point") == "reconverge":
        E, D, Co, J, converged = _scf(jb, kg2, hcore, X, enuc, nocc, Co,
                                      conv_tol, loose=False)
    else:
        K = kg2.get_k(D, SGX_TOL_ENERGY)
        E = (enuc + float(np.vdot(D, hcore)) + 0.5 * float(np.vdot(D, J))
             - 0.25 * float(np.vdot(D, K)))

    return {"e_tot": float(E), "converged": bool(converged)}
