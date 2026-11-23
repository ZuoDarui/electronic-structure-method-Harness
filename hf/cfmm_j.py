"""Multipole-accelerated RI-J (CFMM near/far split of the three-index integrals).

RI-J needs two contractions with the three-index integrals (ab|P):

    gamma_P = sum_ab (P|ab) D_ab,   c = V^-1 gamma,   J_ab = sum_P (ab|P) c_P.

Shells are grouped by spatial box and by their most diffuse exponent.  For
every pair block (AO group I, AO group J) and auxiliary group K the block of
(ab|P) is either computed exactly with int3c2e (near) or replaced by a
multipole interaction of total order <= ``order`` (far).  A pair is far only
when the two charge distributions do not penetrate (erfc bound ``eps``) and
the geometric radii are below ``theta`` times the distance.  Both passes use
the same split, so the operator stays symmetric.

Moments are real solid-harmonic moments from libcint (a ghost shell with a
vanishing exponent), so the far field only sees the harmonic part of every
moment, which is all a Coulomb interaction depends on.
"""

import math
import time

import numpy as np
import scipy.linalg
import scipy.sparse
from scipy.special import erfcinv
from pyscf import gto
from pyscf.df import addons

_GHOST_EXPONENT = 1e-20
_MAX_ORDER = 12


def cartesian_indices(order):
    """(i, j, k) with i + j + k <= order, by degree, PySCF order inside one degree."""

    rows = []
    for l in range(order + 1):
        for i in range(l, -1, -1):
            for j in range(l - i, -1, -1):
                rows.append((i, j, l - i - j))
    return np.array(rows, dtype=np.int64)


def ncart_upto(order):
    return (order + 1) * (order + 2) * (order + 3) // 6


class CoulombDerivatives:
    """All d^t/dx^t d^u/dy^u d^v/dz^v (1/|R|) with t + u + v <= order.

    McMurchie-Davidson recursion with the Boys function replaced by its
    large-argument limit: R^n_000 = (-1)^n (2n-1)!! / R^(2n+1).
    """

    def __init__(self, order):
        self.order = int(order)
        self.index = cartesian_indices(self.order)
        lookup = {tuple(v): k for k, v in enumerate(self.index)}
        self.levels = []
        for n in range(self.order - 1, -1, -1):
            size = ncart_upto(self.order - n)
            per_axis = []
            for axis in range(3):
                tgt, coef, src2, src1 = [], [], [], []
                for k in range(1, size):
                    t = self.index[k]
                    if int(np.flatnonzero(t)[0]) != axis:
                        continue
                    lower = t.copy()
                    lower[axis] -= 1
                    src1.append(lookup[tuple(lower)])
                    if lower[axis] > 0:
                        lower2 = lower.copy()
                        lower2[axis] -= 1
                        src2.append(lookup[tuple(lower2)])
                        coef.append(float(lower[axis]))
                    else:
                        src2.append(0)
                        coef.append(0.0)
                    tgt.append(k)
                per_axis.append((np.array(tgt, dtype=np.int64),
                                 np.array(coef), np.array(src2, dtype=np.int64),
                                 np.array(src1, dtype=np.int64)))
            self.levels.append((n, size, per_axis))

    def __call__(self, vectors):
        vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
        r2 = np.einsum("ij,ij->i", vectors, vectors)
        inv_r2 = 1.0 / r2
        g = np.empty((self.order + 1, len(vectors)))
        g[0] = np.sqrt(inv_r2)
        for n in range(1, self.order + 1):
            g[n] = -(2 * n - 1) * g[n - 1] * inv_r2
        prev = g[self.order][:, None]
        for n, size, per_axis in self.levels:
            cur = np.empty((len(vectors), size))
            cur[:, 0] = g[n]
            for axis, (tgt, coef, src2, src1) in enumerate(per_axis):
                if len(tgt):
                    cur[:, tgt] = (prev[:, src2] * coef
                                   + vectors[:, axis, None] * prev[:, src1])
            prev = cur
        return prev


class MultipoleTranslation:
    """Cartesian multipole-to-local operator of total order <= ``order``.

    For a source with moments M about A and a target with moments N about B,
    E = sum_{a,b} (-1)^|a| / (a! b!) M_a N_b T_{a+b}(B - A).
    """

    def __init__(self, order):
        self.order = int(order)
        index = cartesian_indices(self.order)
        lookup = {tuple(v): k for k, v in enumerate(index)}
        self.size = n = len(index)
        degree = index.sum(axis=1)
        factorial = np.array([math.factorial(i) * math.factorial(j) * math.factorial(k)
                              for i, j, k in index], dtype=np.float64)
        self.source_scale = (-1.0) ** degree / factorial
        self.target_scale = 1.0 / factorial
        flat, rows = [], []
        for a in range(n):
            for b in range(n):
                if degree[a] + degree[b] <= self.order:
                    flat.append(a * n + lookup[tuple(index[a] + index[b])])
                    rows.append(b)
        self.flat = np.array(flat, dtype=np.int64)
        self.gather = scipy.sparse.csr_matrix(
            (np.ones(len(rows)), (np.arange(len(rows)), np.array(rows))),
            shape=(len(rows), n))

    def local(self, source_moments, tensors):
        """``source_moments`` (ns, n) already scaled, ``tensors`` (nt, ns, n)."""

        y = np.matmul(source_moments.T[None], tensors)
        values = y.reshape(len(tensors), -1)[:, self.flat]
        return np.asarray(self.gather.T @ values.T).T


def _ghost_mol(centers, order):
    """One ghost atom per center carrying shells l = 0..order, exponent -> 0."""

    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    basis = {"ghost": [[l, [_GHOST_EXPONENT, 1.0]] for l in range(order + 1)]}
    mol = gto.M(atom=[["ghost", tuple(c)] for c in centers], basis=basis,
                unit="Bohr", verbose=0, spin=0, charge=0)
    for ib in range(mol.nbas):
        mol._env[mol._bas[ib, gto.PTR_COEFF]] = 1.0
    return mol


def harmonic_to_cartesian(order):
    """Matrix G (nsph, ncart): M_eff = Q @ G maps solid-harmonic moments to
    Cartesian moments that give the same result against any harmonic
    polynomial, and w = q @ G.T maps harmonic Cartesian coefficients q to
    solid-harmonic weights."""

    ghost = _ghost_mol(np.zeros((1, 3)), order)
    rng = np.random.RandomState(7)
    points = rng.uniform(-1.0, 1.0, size=(6 * ncart_upto(order), 3))
    values = ghost.eval_gto("GTOval_sph", points)
    index = cartesian_indices(order)
    nsph = (order + 1) ** 2
    g = np.zeros((nsph, len(index)))
    worst = 0.0
    for l in range(order + 1):
        cart = np.flatnonzero(index.sum(axis=1) == l)
        mono = np.prod(points[:, None, :] ** index[cart][None], axis=2)
        sph = values[:, l * l:(l + 1) * (l + 1)]
        coeff, *_ = np.linalg.lstsq(mono, sph, rcond=None)
        worst = max(worst, float(np.abs(mono @ coeff - sph).max()
                                 / max(np.abs(sph).max(), 1e-300)))
        g[l * l:(l + 1) * (l + 1), cart] = np.linalg.pinv(coeff)
    if worst > 1e-10:
        raise RuntimeError(f"ghost harmonic fit residual {worst:.2e}")
    return g


class _NearField:
    """Screened int3c2e contractions on shell slices, PySCF's DF-J kernels.

    The optimizer mirrors ``pyscf.df.df_jk.get_j``: Schwarz bounds of the AO
    pairs and of the auxiliary shells, a density bound set once per pass,
    and ``direct_scf_tol`` as the cutoff.
    """

    def __init__(self, ao_mol, aux_mol, aux_diag, direct_scf_tol):
        from pyscf.scf import _vhf

        opt = _vhf._VHFOpt(ao_mol, "int3c2e", "CVHFnr3c2e_schwarz_cond",
                           dmcondname="CVHFnr_dm_cond",
                           direct_scf_tol=direct_scf_tol)
        opt.init_cvhf_direct(ao_mol, "int2e", "CVHFnr_int2e_q_cond")
        aux_root = np.sqrt(np.abs(aux_diag))
        aux_loc = aux_mol.ao_loc
        aux_q = [aux_root[i0:i1].max() for i0, i1 in zip(aux_loc[:-1], aux_loc[1:])]
        opt.q_cond = np.hstack((opt.q_cond.ravel(), aux_q))
        placeholder = np.array([0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int32)
        fake = ao_mol + aux_mol
        fake._bas = np.vstack((fake._bas, placeholder))
        self.opt = opt
        self.fake = fake
        self.nbas = ao_mol.nbas
        self.slot = ao_mol.nbas + aux_mol.nbas
        self.aux_loc = aux_loc

    def _slice(self, si, ei, sj, ej, run):
        nb = self.nbas
        return (si, ei, sj, ej, nb + run[0], nb + run[1], self.slot, self.slot + 1)

    def begin_pass1(self, dm):
        from pyscf.scf import jk  # noqa: F401  (loads libcvhf symbols)

        opt = self.opt
        opt.prescreen = "CVHFnr3c2e_vj_pass1_prescreen"
        fake = self.fake
        opt.set_dm(dm[None], fake._atm, fake._bas, fake._env)

    def gamma(self, dm_block_t, si, ei, sj, ej, run, diagonal):
        """sum_{a in I, b in J} (ab|P) D_ab for P in the run; ``dm_block_t`` is D[J, I]."""

        from pyscf import lib
        from pyscf.scf import jk

        with lib.temporary_env(self.opt, _dmcondname=None):
            out = jk.get_jk(self.fake, dm_block_t, "ijkl,ji->kl", "int3c2e",
                            aosym="s2ij" if diagonal else "s1", hermi=0,
                            shls_slice=self._slice(si, ei, sj, ej, run),
                            vhfopt=self.opt)
        return np.asarray(out)[:, 0]

    def begin_pass2(self, coeff):
        opt = self.opt
        opt.prescreen = "CVHFnr3c2e_vj_pass2_prescreen"
        loc = self.aux_loc
        with_none = opt._dmcondname
        opt._dmcondname = None
        opt.dm_cond = np.array([np.abs(coeff[i0:i1]).max()
                                for i0, i1 in zip(loc[:-1], loc[1:])])
        opt._dmcondname = with_none

    def coulomb(self, coeff_run, si, ei, sj, ej, run, diagonal):
        """sum_P (ab|P) c_P over the run, for a in I, b in J."""

        from pyscf import lib
        from pyscf.scf import jk

        with lib.temporary_env(self.opt, _dmcondname=None):
            out = jk.get_jk(self.fake, coeff_run[None, :], "ijkl,lk->ij", "int3c2e",
                            aosym="s2ij" if diagonal else "s1",
                            hermi=1 if diagonal else 0,
                            shls_slice=self._slice(si, ei, sj, ej, run),
                            vhfopt=self.opt)
        return np.asarray(out)


def _bounding_center(points):
    points = np.asarray(points).reshape(-1, 3)
    return 0.5 * (points.min(axis=0) + points.max(axis=0))


class _Groups:
    """Shells of one molecule sorted by (exponent class, box)."""

    def __init__(self, mol, box_of_atom, edges):
        nbas = mol.nbas
        atoms = np.array([mol.bas_atom(i) for i in range(nbas)])
        min_exp = np.array([mol.bas_exp(i).min() for i in range(nbas)])
        klass = np.searchsorted(np.asarray(edges, dtype=np.float64), min_exp)
        box = box_of_atom[atoms]
        order = np.lexsort((np.arange(nbas), box, klass))
        self.order = order
        self.mol = mol.copy()
        self.mol._bas = np.asarray(mol._bas[order], order="C")
        ao_loc = mol.ao_loc_nr()
        self.ao_perm = np.concatenate(
            [np.arange(ao_loc[s], ao_loc[s + 1]) for s in order])
        new_loc = self.mol.ao_loc_nr()
        coords = mol.atom_coords()
        self.shell_xyz = coords[atoms[order]]
        self.shell_min_exp = min_exp[order]
        keys = np.stack((klass[order], box[order]), axis=1)
        starts = np.flatnonzero(np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)])
        ends = np.r_[starts[1:], nbas]
        self.shell_range = np.stack((starts, ends), axis=1)
        self.ao_range = np.stack((new_loc[starts], new_loc[ends]), axis=1)
        self.klass = keys[starts, 0]
        self.center = np.array([_bounding_center(self.shell_xyz[s:e])
                                for s, e in self.shell_range])
        self.radius = np.array([
            np.linalg.norm(self.shell_xyz[s:e] - c, axis=1).max()
            for (s, e), c in zip(self.shell_range, self.center)])
        self.min_exp = np.array([self.shell_min_exp[s:e].min()
                                 for s, e in self.shell_range])

    def __len__(self):
        return len(self.shell_range)


class CFMMRIJ:
    """RI-J with exact near field and multipole far field."""

    def __init__(self, mol, auxbasis="weigend", order=10, box_bohr=6.0,
                 theta=0.5, eps=1e-12, pair_tol=1e-14,
                 ao_class_edges=(0.2, 1.0), aux_class_edges=(0.5, 2.0),
                 direct_scf_tol=1e-13, target_chunk=32):
        order = int(order)
        if not 0 <= order <= _MAX_ORDER:
            raise ValueError(f"order must lie in 0..{_MAX_ORDER} (libcint l limit)")
        if not 0.0 < theta < 1.0:
            raise ValueError("theta must lie in (0, 1)")
        if box_bohr <= 0 or eps <= 0 or pair_tol <= 0:
            raise ValueError("box_bohr, eps and pair_tol must be positive")
        self.mol = mol
        self.auxbasis = auxbasis
        self.order = order
        self.box_bohr = float(box_bohr)
        self.theta = float(theta)
        self.eps = float(eps)
        self.pair_tol = float(pair_tol)
        self.ao_class_edges = tuple(float(x) for x in ao_class_edges)
        self.aux_class_edges = tuple(float(x) for x in aux_class_edges)
        self.direct_scf_tol = float(direct_scf_tol)
        self.target_chunk = int(target_chunk)
        self.nao = mol.nao_nr()
        self.stats = {}
        self._prepared = False

    def options(self):
        return {"order": self.order, "box_bohr": self.box_bohr,
                "theta": self.theta, "eps": self.eps, "pair_tol": self.pair_tol,
                "ao_class_edges": list(self.ao_class_edges),
                "aux_class_edges": list(self.aux_class_edges),
                "direct_scf_tol": self.direct_scf_tol}

    def prepare(self):
        if self._prepared:
            return self
        started = time.perf_counter()
        mol = self.mol
        coords = mol.atom_coords()
        cells = np.floor((coords - coords.min(axis=0)) / self.box_bohr).astype(np.int64)
        _, box_of_atom = np.unique(cells, axis=0, return_inverse=True)
        box_of_atom = box_of_atom.ravel()

        self.ao = _Groups(mol, box_of_atom, self.ao_class_edges)
        auxmol = addons.make_auxmol(mol, self.auxbasis)
        self.aux = _Groups(auxmol, box_of_atom, self.aux_class_edges)
        self.naux = self.aux.mol.nao_nr()

        self._build_pair_blocks()
        self._classify()
        self._build_moments()

        t0 = time.perf_counter()
        j2c = self.aux.mol.intor("int2c2e", hermi=1)
        self._nearfield = _NearField(self.ao.mol, self.aux.mol, j2c.diagonal().copy(),
                                     self.direct_scf_tol)
        try:
            self._metric = ("cd", scipy.linalg.cho_factor(j2c, lower=True))
        except scipy.linalg.LinAlgError:
            self._metric = ("lu", scipy.linalg.lu_factor(j2c))
        del j2c
        self.stats["metric_s"] = time.perf_counter() - t0
        self.stats["prepare_s"] = time.perf_counter() - started
        self._prepared = True
        return self

    def _build_pair_blocks(self):
        ao = self.ao
        cut = -math.log(self.pair_tol)
        spread = erfcinv(self.eps)
        n = len(ao)
        dist = np.linalg.norm(ao.center[:, None] - ao.center[None], axis=2)
        mu_min = (ao.min_exp[:, None] * ao.min_exp[None]
                  / (ao.min_exp[:, None] + ao.min_exp[None]))
        gap = np.maximum(dist - ao.radius[:, None] - ao.radius[None], 0.0)
        candidates = np.argwhere(np.triu(mu_min * gap * gap < cut))
        blocks = []
        for i, j in candidates:
            si, ei = ao.shell_range[i]
            sj, ej = ao.shell_range[j]
            a, b = ao.shell_xyz[si:ei], ao.shell_xyz[sj:ej]
            ea, eb = ao.shell_min_exp[si:ei], ao.shell_min_exp[sj:ej]
            d2 = ((a[:, None] - b[None]) ** 2).sum(axis=2)
            mu = ea[:, None] * eb[None] / (ea[:, None] + eb[None])
            significant = mu * d2 < cut
            if not significant.any():
                continue
            center = 0.5 * (ao.center[i] + ao.center[j])
            ra = np.linalg.norm(a - center, axis=1)
            rb = np.linalg.norm(b - center, axis=1)
            reach = np.maximum(ra[:, None], rb[None])
            width = spread / np.sqrt(ea[:, None] + eb[None])
            blocks.append((int(i), int(j), center,
                           float(reach[significant].max()),
                           float((reach + width)[significant].max())))
        self.pair_i = np.array([b[0] for b in blocks], dtype=np.int64)
        self.pair_j = np.array([b[1] for b in blocks], dtype=np.int64)
        self.pair_center = np.array([b[2] for b in blocks]).reshape(-1, 3)
        self.pair_radius = np.array([b[3] for b in blocks])
        self.pair_extent = np.array([b[4] for b in blocks])
        self.stats["ao_groups"] = n
        self.stats["pair_blocks"] = len(blocks)

    def _classify(self):
        aux = self.aux
        spread = erfcinv(self.eps)
        aux_extent = np.array([
            (np.linalg.norm(aux.shell_xyz[s:e] - c, axis=1)
             + spread / np.sqrt(aux.shell_min_exp[s:e])).max()
            for (s, e), c in zip(aux.shell_range, aux.center)])
        vec = aux.center[None] - self.pair_center[:, None]
        dist = np.linalg.norm(vec, axis=2)
        far = ((dist > self.pair_extent[:, None] + aux_extent[None])
               & (self.pair_radius[:, None] + aux.radius[None] < self.theta * dist))
        self.far_mask = far
        self.near_runs = []
        near_elements = 0
        ao_size = (self.ao.ao_range[:, 1] - self.ao.ao_range[:, 0]).astype(np.int64)
        aux_size = (aux.ao_range[:, 1] - aux.ao_range[:, 0]).astype(np.int64)
        for p in range(len(self.pair_i)):
            groups = np.flatnonzero(~far[p])
            runs = []
            if len(groups):
                breaks = np.flatnonzero(np.diff(groups) != 1)
                first = np.r_[groups[0], groups[breaks + 1]]
                last = np.r_[groups[breaks], groups[-1]]
                for g0, g1 in zip(first, last):
                    runs.append((int(aux.shell_range[g0, 0]), int(aux.shell_range[g1, 1]),
                                 int(aux.ao_range[g0, 0]), int(aux.ao_range[g1, 1])))
            self.near_runs.append(runs)
            near_elements += (ao_size[self.pair_i[p]] * ao_size[self.pair_j[p]]
                              * aux_size[groups].sum())
        self.far_pairs = np.argwhere(far)
        self.stats["aux_groups"] = len(aux)
        self.stats["far_interactions"] = int(far.sum())
        self.stats["near_interactions"] = int((~far).sum())
        self.stats["near_3c_elements"] = int(near_elements)
        full = 0
        for i, j in zip(self.pair_i, self.pair_j):
            full += ao_size[i] * ao_size[j]
        self.stats["pair_block_ao_pairs"] = int(full)
        self.stats["near_fraction"] = float(near_elements / max(full * self.naux, 1))

    def _build_moments(self):
        t0 = time.perf_counter()
        order = self.order
        nsph = (order + 1) ** 2
        self._g = harmonic_to_cartesian(order)
        self._tensor = CoulombDerivatives(order)
        self._m2l = MultipoleTranslation(order)

        nb = self.ao.mol.nbas
        ghost = _ghost_mol(self.pair_center, order)
        fake = gto.conc_mol(self.ao.mol, ghost)
        shells = order + 1
        self.pair_moments = []
        for p, (i, j) in enumerate(zip(self.pair_i, self.pair_j)):
            si, ei = self.ao.shell_range[i]
            sj, ej = self.ao.shell_range[j]
            g0 = nb + p * shells
            self.pair_moments.append(np.ascontiguousarray(fake.intor(
                "int3c1e_sph", shls_slice=(si, ei, sj, ej, g0, g0 + shells))))

        nba = self.aux.mol.nbas
        ghost = _ghost_mol(self.aux.center, order)
        fake = gto.conc_mol(self.aux.mol, ghost)
        self.aux_moments = []
        for k, (s, e) in enumerate(self.aux.shell_range):
            g0 = nba + k * shells
            self.aux_moments.append(np.ascontiguousarray(fake.intor(
                "int1e_ovlp_sph", shls_slice=(s, e, g0, g0 + shells))))

        pairs = self.far_pairs
        vectors = self.aux.center[pairs[:, 1]] - self.pair_center[pairs[:, 0]]
        self._far_tensor = self._tensor(vectors) if len(pairs) else np.zeros((0, 1))
        moment_bytes = sum(m.nbytes for m in self.pair_moments)
        moment_bytes += sum(m.nbytes for m in self.aux_moments)
        self.stats["moment_mb"] = moment_bytes / 1e6
        self.stats["far_tensor_mb"] = self._far_tensor.nbytes / 1e6
        self.stats["nsph"] = nsph
        self.stats["moments_s"] = time.perf_counter() - t0

    def _far_local(self, source, target_is_aux):
        """Local expansions at all targets from scaled Cartesian source moments."""

        pairs = self.far_pairs
        if target_is_aux:
            ntarget = len(self.aux)
            tgt, src = pairs[:, 1], pairs[:, 0]
            sign = 1.0
        else:
            ntarget = len(self.pair_i)
            tgt, src = pairs[:, 0], pairs[:, 1]
            sign = (-1.0) ** cartesian_indices(self.order).sum(axis=1)
        nsrc = len(source)
        ncart = self._m2l.size
        local = np.zeros((ntarget, ncart))
        if not len(pairs):
            return local
        order = np.argsort(tgt, kind="stable")
        tgt_sorted = tgt[order]
        bounds = np.searchsorted(tgt_sorted, np.arange(ntarget + 1))
        active = np.flatnonzero(np.diff(bounds))
        for c0 in range(0, len(active), self.target_chunk):
            chunk = active[c0:c0 + self.target_chunk]
            dense = np.zeros((len(chunk), nsrc, ncart))
            for row, t in enumerate(chunk):
                sel = order[bounds[t]:bounds[t + 1]]
                dense[row, src[sel]] = self._far_tensor[sel] * sign
            local[chunk] = self._m2l.local(source, dense)
        return local

    def build(self, dm):
        self.prepare()
        dm = np.asarray(dm, dtype=np.float64)
        if dm.shape != (self.nao, self.nao):
            raise ValueError("dm must have shape (nao, nao)")
        perm = self.ao.ao_perm
        dmp = dm[np.ix_(perm, perm)]
        dmp = 0.5 * (dmp + dmp.T)
        ao_rng = self.ao.ao_range
        aux_rng = self.aux.ao_range
        g = self._g
        m2l = self._m2l

        near = self._nearfield
        shell_rng = self.ao.shell_range
        t0 = time.perf_counter()
        gamma = np.zeros(self.naux)
        npair = len(self.pair_i)
        pair_q = np.zeros((npair, g.shape[0]))
        near.begin_pass1(dmp)
        for p in range(npair):
            i, j = self.pair_i[p], self.pair_j[p]
            block = dmp[ao_rng[i, 0]:ao_rng[i, 1], ao_rng[j, 0]:ao_rng[j, 1]]
            weight = 1.0 if i == j else 2.0
            pair_q[p] = weight * np.tensordot(block, self.pair_moments[p], axes=2)
            if self.near_runs[p]:
                block_t = np.ascontiguousarray(block.T)
                si, ei = shell_rng[i]
                sj, ej = shell_rng[j]
                for run in self.near_runs[p]:
                    gamma[run[2]:run[3]] += weight * near.gamma(
                        block_t, si, ei, sj, ej, run, i == j)
        t_near1 = time.perf_counter() - t0
        t1 = time.perf_counter()
        source = (pair_q @ g) * m2l.source_scale
        local = self._far_local(source, target_is_aux=True)
        weights = (local * m2l.target_scale) @ g.T
        for k in range(len(self.aux)):
            gamma[aux_rng[k, 0]:aux_rng[k, 1]] += self.aux_moments[k] @ weights[k]
        t_far1 = time.perf_counter() - t1

        t2 = time.perf_counter()
        kind, factor = self._metric
        if kind == "cd":
            coeff = scipy.linalg.cho_solve(factor, gamma)
        else:
            coeff = scipy.linalg.lu_solve(factor, gamma)
        t_solve = time.perf_counter() - t2

        t3 = time.perf_counter()
        aux_q = np.array([coeff[aux_rng[k, 0]:aux_rng[k, 1]] @ self.aux_moments[k]
                          for k in range(len(self.aux))])
        source = (aux_q @ g) * m2l.source_scale
        local = self._far_local(source, target_is_aux=False)
        weights = (local * m2l.target_scale) @ g.T
        t_far2 = time.perf_counter() - t3
        t4 = time.perf_counter()
        near.begin_pass2(coeff)
        vj = np.zeros((self.nao, self.nao))
        for p in range(npair):
            i, j = self.pair_i[p], self.pair_j[p]
            block = self.pair_moments[p] @ weights[p]
            si, ei = shell_rng[i]
            sj, ej = shell_rng[j]
            for run in self.near_runs[p]:
                block += near.coulomb(coeff[run[2]:run[3]], si, ei, sj, ej, run, i == j)
            a0, a1 = ao_rng[i]
            b0, b1 = ao_rng[j]
            vj[a0:a1, b0:b1] += block
            if i != j:
                vj[b0:b1, a0:a1] += block.T
        t_near2 = time.perf_counter() - t4

        out = np.empty_like(vj)
        out[np.ix_(perm, perm)] = vj
        self.stats.update({
            "near_pass1_s": t_near1, "far_pass1_s": t_far1, "solve_s": t_solve,
            "far_pass2_s": t_far2, "near_pass2_s": t_near2,
        })
        self.last_coeff = coeff
        self.last_gamma = gamma
        return out
