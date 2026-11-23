"""Consistent, matrix-free density fitting with hierarchical Coulomb operators.

The production calculation is B M^-1 B.T, never a sum of independently fitted
owned coefficients.  AO pairs are packed once, independently of the density;
the same stored blocks implement B and B.T.  Near blocks use libcint exactly.
Far blocks use deterministic ACA and residual probes.  Probe residuals are
empirical error controls, not certified spectral error bounds: independent
DF references remain necessary for scientific acceptance.

No PySCF J builder or dense three-index tensor is used by this module.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import time

import numpy as np
import scipy.linalg
from scipy.spatial import cKDTree
from pyscf import df, gto
from pyscf.gto import moleintor


class HierarchicalRIError(RuntimeError):
    """A numerical or resource gate rejected the approximate Coulomb operator."""


@dataclass
class Cluster:
    indices: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    children: tuple = ()

    @property
    def diameter(self):
        return float(np.linalg.norm(self.upper - self.lower))


def cluster_tree(points, indices=None, leaf_size=128):
    points = np.asarray(points, dtype=float)
    if indices is None:
        indices = np.arange(len(points), dtype=np.int32)
    if not len(indices):
        return Cluster(indices, np.zeros(3), np.zeros(3))
    selected = points[indices]
    lower, upper = selected.min(axis=0), selected.max(axis=0)
    children = ()
    if len(indices) > leaf_size:
        axis = int(np.argmax(upper - lower))
        order = np.argsort(selected[:, axis], kind="stable")
        middle = len(order) // 2
        children = (cluster_tree(points, indices[order[:middle]], leaf_size),
                    cluster_tree(points, indices[order[middle:]], leaf_size))
    return Cluster(indices, lower, upper, children)


def leaves(node):
    if not node.children:
        yield node
    else:
        for child in node.children:
            yield from leaves(child)


class HierarchicalMatrix:
    """Rectangular or symmetric H matrix with shared transpose factors."""

    def __init__(self, row_points, col_points, provider, *, symmetric=False,
                 leaf_size=128, separation=2.0, max_rank=64, tolerance=1e-12,
                 memory_bytes=1 << 30, xp=np, stream_near=False):
        self.shape = (len(row_points), len(col_points))
        self.provider, self.symmetric = provider, bool(symmetric)
        self.max_rank, self.tolerance = int(max_rank), float(tolerance)
        self.separation, self.memory_bytes = float(separation), int(memory_bytes)
        self.xp = xp
        self.stream_near = bool(stream_near)
        self.blocks = []
        self.stored_bytes = 0
        self.stats = {"dense_blocks": 0, "low_rank_blocks": 0,
                      "streamed_near_blocks": 0,
                      "maximum_rank": 0, "sampled_entries": 0,
                      "probe_relative_residual_max": 0.0,
                      "validation_probe_relative_residual_max": 0.0}
        self.row_tree = cluster_tree(row_points, leaf_size=leaf_size)
        self.col_tree = (self.row_tree if symmetric else
                         cluster_tree(col_points, leaf_size=leaf_size))
        self._partition(self.row_tree, self.col_tree)
        self.stats["stored_bytes"] = self.stored_bytes

    def _entries(self, rows, cols):
        result = np.asarray(self.provider(rows, cols), dtype=np.float64)
        if result.shape != (len(rows), len(cols)) or not np.isfinite(result).all():
            raise HierarchicalRIError("Nonfinite or incorrectly shaped integral block")
        self.stats["sampled_entries"] += result.size
        return result

    def _admissible(self, a, b):
        gap = np.maximum(0., np.maximum(a.lower - b.upper, b.lower - a.upper))
        distance = float(np.linalg.norm(gap))
        return distance > self.separation * max(a.diameter, b.diameter, 1e-12)

    def _store(self, a, b, dense=None, u=None, v=None):
        arrays = [dense] if dense is not None else [u, v]
        extra = sum(x.nbytes for x in arrays)
        if self.stored_bytes + extra > self.memory_bytes:
            raise HierarchicalRIError(
                f"Hierarchical operator needs more than {self.memory_bytes} bytes; "
                "no dense fallback is permitted")
        self.stored_bytes += extra
        if self.xp is not np:
            arrays = [self.xp.asarray(x) for x in arrays]
        if dense is not None:
            self.stats["dense_blocks"] += 1
            block = (a.indices, b.indices, arrays[0], None, None)
        else:
            self.stats["low_rank_blocks"] += 1
            self.stats["maximum_rank"] = max(self.stats["maximum_rank"], u.shape[1])
            block = (a.indices, b.indices, None, arrays[0], arrays[1])
        self.blocks.append(block)

    def _partition(self, a, b):
        if not len(a.indices) or not len(b.indices):
            return
        if self._admissible(a, b) and min(len(a.indices), len(b.indices)) > 3:
            factors = self._aca(a.indices, b.indices)
            if factors is not None:
                self._store(a, b, u=factors[0], v=factors[1])
                return
        if not a.children and not b.children:
            if self.stream_near:
                self.blocks.append((a.indices, b.indices, None, None, None))
                self.stats['streamed_near_blocks'] += 1
                return
            peak_bytes = 2 * len(a.indices) * len(b.indices) * 8
            if self.stored_bytes + peak_bytes > self.memory_bytes:
                raise HierarchicalRIError('Exact near-block temporary exceeds memory budget')
            matrix = self._entries(a.indices, b.indices)
            if self.symmetric and a is b:
                matrix = (matrix + matrix.T) * .5
            self._store(a, b, dense=matrix)
        elif self.symmetric and a is b:
            left, right = a.children
            self._partition(left, left)
            self._partition(left, right)
            self._partition(right, right)
        elif a.children and (not b.children or len(a.indices) >= len(b.indices)):
            for child in a.children:
                self._partition(child, b)
        else:
            for child in b.children:
                self._partition(a, child)

    def _aca(self, rows, cols):
        nr, nc = len(rows), len(cols)
        # If workspace would exceed the allocation budget, partition the
        # block further before asking the integral provider for any probes.
        if self.stored_bytes + 4 * (nr + nc) * 64 * 8 > self.memory_bytes:
            return None
        # Independent deterministic probes include both boundaries and interior.
        seed = (int(rows[0]) * 1000003 + int(cols[0]) * 9176 + nr + nc) % (2**32)
        rng = np.random.default_rng(seed)
        pr = np.unique(np.r_[0, nr - 1, np.linspace(0, nr - 1, min(8, nr), dtype=int),
                             rng.integers(nr, size=min(8, nr))])
        pc = np.unique(np.r_[0, nc - 1, np.linspace(0, nc - 1, min(8, nc), dtype=int),
                             rng.integers(nc, size=min(8, nc))])
        probe_rows = self._entries(rows[pr], cols)
        probe_cols = self._entries(rows, cols[pc])
        norm = max(float(np.linalg.norm(probe_rows)), float(np.linalg.norm(probe_cols)))
        if norm == 0.:
            # Zero angular blocks are common. Verify all small blocks exactly;
            # for a large block, refuse a zero approximation based on probes.
            return None
        u = np.empty((nr, 0)); v = np.empty((0, nc))

        def validate_factors():
            nonlocal pr, pc, probe_rows, probe_cols, norm
            unused_r = np.setdiff1d(np.arange(nr), pr, assume_unique=False)
            unused_c = np.setdiff1d(np.arange(nc), pc, assume_unique=False)
            hr = np.sort(rng.choice(unused_r, min(16, len(unused_r)), replace=False))
            hc = np.sort(rng.choice(unused_c, min(16, len(unused_c)), replace=False))
            vr = self._entries(rows[hr], cols)
            vc = self._entries(rows, cols[hc])
            norm = max(norm, float(np.linalg.norm(vr)), float(np.linalg.norm(vc)))
            error = max(float(np.linalg.norm(vr - u[hr] @ v)),
                        float(np.linalg.norm(vc - u @ v[:, hc]))) / norm
            if error <= self.tolerance:
                self.stats['validation_probe_relative_residual_max'] = max(
                    self.stats['validation_probe_relative_residual_max'], error)
                return True
            # A fixed probe subspace can be fit exactly while unsampled
            # multipole channels remain wrong. Failed independent probes become
            # new pivots; simply tightening the old stopping tolerance cannot
            # repair that false convergence.
            pr = np.r_[pr, hr]; pc = np.r_[pc, hc]
            probe_rows = np.vstack((probe_rows, vr))
            probe_cols = np.column_stack((probe_cols, vc))
            return False

        for _ in range(min(self.max_rank, nr, nc)):
            peak_bytes = 4 * (nr + nc) * (u.shape[1] + len(pr) + len(pc) + 32) * 8
            if self.stored_bytes + peak_bytes > self.memory_bytes:
                return None
            rr = probe_rows - u[pr] @ v
            rc = probe_cols - u @ v[:, pc]
            rel = max(float(np.linalg.norm(rr)), float(np.linalg.norm(rc))) / norm
            if u.shape[1] and rel <= self.tolerance:
                if validate_factors():
                    self.stats["probe_relative_residual_max"] = max(
                        self.stats["probe_relative_residual_max"], rel)
                    return u, v
                rr = probe_rows - u[pr] @ v
                rc = probe_cols - u @ v[:, pc]
            if np.max(np.abs(rr)) >= np.max(np.abs(rc)):
                probe_i, j = np.unravel_index(np.argmax(np.abs(rr)), rr.shape)
                i = int(pr[probe_i])
                row = rr[probe_i].copy()
                column = self._entries(rows, cols[j:j + 1])[:, 0] - u @ v[:, j]
            else:
                i, probe_j = np.unravel_index(np.argmax(np.abs(rc)), rc.shape)
                j = int(pc[probe_j])
                column = rc[:, probe_j].copy()
                row = self._entries(rows[i:i + 1], cols)[0] - u[i] @ v
            pivot = row[j]
            if abs(pivot) < np.finfo(float).eps * norm:
                return None
            u = np.column_stack((u, column / pivot))
            v = np.vstack((v, row))
            if u.size + v.size >= nr * nc:
                return None
        rr = probe_rows - u[pr] @ v
        rc = probe_cols - u @ v[:, pc]
        rel = max(float(np.linalg.norm(rr)), float(np.linalg.norm(rc))) / norm
        if (rel <= self.tolerance and u.size + v.size < nr * nc
                and validate_factors()):
            self.stats["probe_relative_residual_max"] = max(
                self.stats["probe_relative_residual_max"], rel)
            return u, v
        return None

    def matvec(self, vector, transpose=False):
        if self.symmetric:
            transpose = False
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (self.shape[0 if transpose else 1],):
            raise ValueError("Operator vector has wrong shape")
        xp = self.xp
        x = xp.asarray(vector) if xp is not np else vector
        y = xp.zeros(self.shape[1 if transpose else 0], dtype=xp.float64)
        for rows, cols, dense, u, v in self.blocks:
            if dense is None and u is None:
                dense = self._entries(rows, cols)
                if xp is not np:
                    dense = xp.asarray(dense)
            if transpose:
                val = dense.T @ x[rows] if dense is not None else v.T @ (u.T @ x[rows])
                y[cols] += val
            else:
                val = dense @ x[cols] if dense is not None else u @ (v @ x[cols])
                y[rows] += val
            if self.symmetric and not np.array_equal(rows, cols):
                val = dense.T @ x[rows] if dense is not None else v.T @ (u.T @ x[rows])
                y[cols] += val
        if xp is not np:
            return xp.asnumpy(y)
        return y


class _IntegralProvider:
    def __init__(self, mol, auxmol, pair_i, pair_j, cache_bytes):
        self.mol, self.auxmol = mol, auxmol
        self.pair_i, self.pair_j = pair_i, pair_j
        self.ao_loc, self.aux_loc = mol.ao_loc_nr(), auxmol.ao_loc_nr()
        self.ao_shell = np.repeat(np.arange(mol.nbas), np.diff(self.ao_loc))
        self.aux_shell = np.repeat(np.arange(auxmol.nbas), np.diff(self.aux_loc))
        self.ri_atm, self.ri_bas, self.ri_env = gto.mole.conc_env(
            mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env)
        self.intor = mol._add_suffix('int3c2e')
        self.ri_loc = moleintor.make_loc(self.ri_bas, self.intor)
        self.cintopt = moleintor.make_cintopt(self.ri_atm, self.ri_bas, self.ri_env, self.intor)
        self.cache, self.cache_bytes = OrderedDict(), int(cache_bytes)
        self.cache_size = 0
        self.integral_calls = 0

    def _cache(self, key, make):
        value = self.cache.pop(key, None)
        if value is None:
            value = make()
            self.integral_calls += 1
            if value.nbytes > self.cache_bytes:
                return value
            while self.cache and self.cache_size + value.nbytes > self.cache_bytes:
                _, old = self.cache.popitem(last=False)
                self.cache_size -= old.nbytes
            self.cache_size += value.nbytes
        self.cache[key] = value
        return value

    def metric(self, rows, cols):
        result = np.empty((len(rows), len(cols)))
        rs, cs = self.aux_shell[rows] // 16, self.aux_shell[cols] // 16
        for ib in np.unique(rs):
            ir = np.flatnonzero(rs == ib)
            i0, i1 = int(ib) * 16, min((int(ib) + 1) * 16, self.auxmol.nbas)
            for jb in np.unique(cs):
                jc = np.flatnonzero(cs == jb)
                j0, j1 = int(jb) * 16, min((int(jb) + 1) * 16, self.auxmol.nbas)
                key = ('m', i0, j0)
                block = self._cache(key, lambda a=i0, b=i1, c=j0, d=j1:
                    self.auxmol.intor('int2c2e', shls_slice=(a, b, c, d)))
                result[np.ix_(ir, jc)] = block[np.ix_(rows[ir] - self.aux_loc[i0],
                                                       cols[jc] - self.aux_loc[j0])]
        return result

    def three_center(self, rows, cols):
        result = np.empty((len(rows), len(cols)))
        ii, jj = self.pair_i[rows], self.pair_j[rows]
        si, sj = self.ao_shell[ii], self.ao_shell[jj]
        key_pairs = si.astype(np.int64) * self.mol.nbas + sj
        col_blocks = self.aux_shell[cols] // 16
        for code in np.unique(key_pairs):
            ir = np.flatnonzero(key_pairs == code)
            ish, jsh = divmod(int(code), self.mol.nbas)
            li, lj = ii[ir] - self.ao_loc[ish], jj[ir] - self.ao_loc[jsh]
            for blk in np.unique(col_blocks):
                jc = np.flatnonzero(col_blocks == blk)
                p0, p1 = int(blk) * 16, min((int(blk) + 1) * 16, self.auxmol.nbas)
                key = ('b', ish, jsh, p0)
                def make(i=ish, j=jsh, p=p0, q=p1):
                    return moleintor.getints(
                        self.intor, self.ri_atm, self.ri_bas, self.ri_env,
                        (i, i + 1, j, j + 1, self.mol.nbas + p, self.mol.nbas + q),
                        aosym='s1', ao_loc=self.ri_loc, cintopt=self.cintopt)
                value = self._cache(key, make)
                result[np.ix_(ir, jc)] = value[li[:, None], lj[:, None],
                                              (cols[jc] - self.aux_loc[p0])[None, :]]
        return result


def _pair_geometry(mol, tail_tolerance, schwarz_tolerance):
    loc = mol.ao_loc_nr()
    centers = mol.atom_coords()[mol._bas[:, gto.ATOM_OF]]
    radii = []
    for shell in range(mol.nbas):
        alpha = float(np.min(mol.bas_exp(shell)))
        weight = max(1., float(np.max(np.abs(mol.bas_ctr_coeff(shell)))))
        # Conservative operational support, checked against full references;
        # this estimate is not presented as a theorem for contraction tails.
        radii.append(2. + np.sqrt(max(0., (np.log(weight / tail_tolerance)
                                            + mol.bas_angular(shell)) / alpha)))
    radii = np.asarray(radii)
    tree = cKDTree(centers)
    pair_i, pair_j, points = [], [], []
    candidates, retained = 0, 0
    intor = mol._add_suffix('int2e')
    opt = moleintor.make_cintopt(mol._atm, mol._bas, mol._env, intor)
    for ish, point in enumerate(centers):
        neighbors = tree.query_ball_point(point, radii[ish] + np.max(radii))
        for jsh in sorted(j for j in neighbors if j <= ish):
            if np.linalg.norm(point - centers[jsh]) > radii[ish] + radii[jsh]:
                continue
            candidates += 1
            value = moleintor.getints(intor, mol._atm, mol._bas, mol._env,
                (ish, ish + 1, jsh, jsh + 1, ish, ish + 1, jsh, jsh + 1),
                aosym='s1', cintopt=opt)
            if np.sqrt(np.max(np.abs(value))) < schwarz_tolerance:
                continue
            retained += 1
            ii, jj = np.meshgrid(np.arange(loc[ish], loc[ish + 1]),
                                 np.arange(loc[jsh], loc[jsh + 1]), indexing='ij')
            keep = ii >= jj
            ii, jj = ii[keep], jj[keep]
            pair_i.extend(ii); pair_j.extend(jj)
            ai = float(np.min(mol.bas_exp(ish)))
            aj = float(np.min(mol.bas_exp(jsh)))
            center = (ai * point + aj * centers[jsh]) / (ai + aj)
            points.extend([center] * len(ii))
    return (np.asarray(pair_i, dtype=np.int32), np.asarray(pair_j, dtype=np.int32),
            np.asarray(points), {"candidate_shell_pairs": candidates,
                                 "retained_shell_pairs": retained})


class HierarchicalRIJ:
    """Global-consistent RI-J using local exact and compressed far-field blocks."""

    def __init__(self, mol, auxbasis='weigend', backend='serial', max_memory_mb=4000,
                 options=None):
        self.mol, self.auxbasis = mol, auxbasis
        self.backend = str(backend)
        if self.backend not in ('serial', 'cpu', 'gpu'):
            raise ValueError('backend must be serial, cpu or gpu')
        self.xp = np
        if self.backend == 'gpu':
            import cupy
            self.xp = cupy
        self.options = dict(leaf_size=128, separation=2., max_rank=64,
                            tolerance=1e-12, pair_tolerance=1e-14,
                            tail_tolerance=1e-18, pcg_tolerance=2e-13,
                            pcg_maxiter=500, coarse_limit=128, audit_samples=8)
        if options:
            unknown = set(options) - set(self.options)
            if unknown:
                raise ValueError(f'Unknown hierarchical RI options: {sorted(unknown)}')
            self.options.update(options)
        if any(float(self.options[k]) <= 0 for k in self.options):
            raise ValueError('All hierarchical RI options must be positive')
        self.max_memory_mb = float(max_memory_mb)
        if not np.isfinite(self.max_memory_mb) or self.max_memory_mb <= 0:
            raise ValueError('max_memory_mb must be finite and positive')
        self._prepared = False
        self.stats = {"engine": "hierarchical_ri", "backend": self.backend,
                      "compression_error_certification": "deterministic probes; reference validation required",
                      "asymptotic_scaling_certified": False}

    def prepare(self):
        if self._prepared:
            return self
        started = time.perf_counter()
        self.auxmol = df.addons.make_auxmol(self.mol, self.auxbasis)
        if self.auxmol.cart != self.mol.cart:
            raise NotImplementedError('Mixed Cartesian/spherical auxiliary bases')
        self.pair_i, self.pair_j, self.pair_points, pair_stats = _pair_geometry(
            self.mol, self.options['tail_tolerance'], self.options['pair_tolerance'])
        self.weights = np.where(self.pair_i == self.pair_j, 1., 2.)
        self.stats.update(pair_stats)
        self.stats['packed_ao_pairs'] = len(self.pair_i)
        aux_centers = self.auxmol.atom_coords()[self.auxmol._bas[:, gto.ATOM_OF]]
        self.aux_points = np.repeat(aux_centers, np.diff(self.auxmol.ao_loc_nr()), axis=0)
        total_bytes = int(self.max_memory_mb * 1e6)
        cache_bytes = min(64 << 20, total_bytes // 16)
        self.provider = _IntegralProvider(self.mol, self.auxmol,
                                          self.pair_i, self.pair_j, cache_bytes)
        self._total_bytes = total_bytes
        geometry_bytes = sum(a.nbytes for a in (self.pair_i, self.pair_j,
                             self.pair_points, self.weights, self.aux_points))
        # Reserve headroom for integral temporaries, NumPy views and tree index
        # metadata. This is an allocation budget, not a process RSS hard limit.
        self._allocation_budget = int(total_bytes * .8) - geometry_bytes - cache_bytes
        if self._allocation_budget <= 0:
            raise HierarchicalRIError('Pair geometry exceeds the memory allocation budget')
        failures = []
        for tolerance in self._refinements():
            try:
                self._build_operators(tolerance)
                break
            except HierarchicalRIError as exc:
                failures.append(str(exc))
                if 'budget' in str(exc) or 'bytes' in str(exc):
                    raise
        else:
            raise HierarchicalRIError('Operator preparation failed: ' + '; '.join(failures))
        self.stats['preparation_refinement_failures'] = failures
        self.stats['prepare_seconds'] = time.perf_counter() - started
        self._prepared = True
        return self

    def _refinements(self):
        start = float(self.options['tolerance'])
        return sorted({start, min(start, 1e-13), min(start, 1e-14)}, reverse=True)

    def _build_operators(self, tolerance):
        # Release old operator allocations before a tighter rebuild.
        self.metric = self.three_center = None
        self._local_factors = []
        self._coarse_basis = self._coarse_factor = None
        kwargs = dict(leaf_size=int(self.options['leaf_size']),
                      separation=float(self.options['separation']),
                      max_rank=int(self.options['max_rank']), tolerance=tolerance,
                      memory_bytes=int(self._allocation_budget * .4), xp=self.xp)
        self.metric = HierarchicalMatrix(self.aux_points, self.aux_points,
                                         self.provider.metric, symmetric=True, **kwargs)
        kwargs['memory_bytes'] = self._allocation_budget - self.metric.stored_bytes
        self.three_center = HierarchicalMatrix(self.pair_points, self.aux_points,
                                               self.provider.three_center, stream_near=True, **kwargs)
        self._factor_budget = (self._allocation_budget - self.metric.stored_bytes
                               - self.three_center.stored_bytes)
        diagonal = np.empty(self.auxmol.nao_nr())
        for shell in range(self.auxmol.nbas):
            i, j = self.auxmol.ao_loc_nr()[shell:shell + 2]
            diagonal[i:j] = np.diag(self.provider.metric(np.arange(i, j), np.arange(i, j)))
        if np.any(diagonal <= 0) or not np.isfinite(diagonal).all():
            raise HierarchicalRIError('Coulomb metric has a nonpositive diagonal')
        self.scale = 1. / np.sqrt(diagonal)
        self._make_preconditioner()
        self.stats.update(metric=self.metric.stats, three_center=self.three_center.stats,
                          compression_tolerance=tolerance,
                          integral_calls=self.provider.integral_calls)

    def _scaled_metric(self, vector):
        return self.scale * self.metric.matvec(self.scale * vector)

    def _make_preconditioner(self):
        naux = len(self.scale)
        own = [leaf.indices for leaf in leaves(self.metric.row_tree)]
        domains = []
        # Each leaf overlaps all AO centers within 4 Bohr of its owned support.
        tree = cKDTree(self.aux_points)
        for indices in own:
            expanded = set(map(int, indices))
            for xyz in np.unique(self.aux_points[indices], axis=0):
                expanded.update(tree.query_ball_point(xyz, 4.))
            domains.append(np.asarray(sorted(expanded), dtype=np.int32))
        counts = np.zeros(naux)
        for indices in domains:
            counts[indices] += 1
        if np.any(counts == 0):
            raise HierarchicalRIError('Schwarz domains do not cover auxiliary space')
        self._local_factors = []
        memory = 0
        for indices in domains:
            temporary_bytes = 3 * len(indices)**2 * 8
            if memory + temporary_bytes > self._factor_budget:
                raise HierarchicalRIError('Schwarz factor temporary exceeds aggregate memory budget')
            matrix = self.provider.metric(indices, indices)
            matrix *= self.scale[indices, None] * self.scale[None, indices]
            try:
                factor = scipy.linalg.cho_factor(matrix, lower=True, check_finite=False)
            except scipy.linalg.LinAlgError as exc:
                raise HierarchicalRIError('Local Coulomb metric is not positive definite') from exc
            # Classical symmetric additive Schwarz (W_s = I). Pointwise
            # partition weights spoil the cancellation of nearly dependent
            # auxiliary radial functions at domain boundaries; using identity
            # weights preserves their local inverse. Overlap changes only the
            # preconditioner, never the Coulomb operator or fitted energy.
            weight = np.ones(len(indices))
            self._local_factors.append((indices, weight, factor))
            memory += matrix.nbytes
            if memory > self._factor_budget:
                raise HierarchicalRIError('Schwarz factors exceed their memory budget')
        # A bounded coarse solve is explicit. For larger spaces, tree groups
        # aggregate leaves; no unbounded dense coarse operator is constructed.
        ngroups = min(int(self.options['coarse_limit']), len(own))
        coarse_peak = 3 * naux * ngroups * 8 + 3 * ngroups**2 * 8
        if memory + coarse_peak > self._factor_budget:
            raise HierarchicalRIError('Coarse preconditioner exceeds aggregate memory budget')
        groups = np.array_split(np.arange(len(own)), ngroups)
        coarse = np.zeros((naux, ngroups))
        for column, group in enumerate(groups):
            indices = np.concatenate([own[int(i)] for i in group])
            coarse[indices, column] = 1. / np.sqrt(len(indices))
        applied = np.column_stack([self._scaled_metric(coarse[:, i]) for i in range(ngroups)])
        coarse_metric = coarse.T @ applied
        try:
            self._coarse_factor = scipy.linalg.cho_factor(
                (coarse_metric + coarse_metric.T) * .5, lower=True, check_finite=False)
        except scipy.linalg.LinAlgError as exc:
            raise HierarchicalRIError('Coarse Coulomb metric is not positive definite') from exc
        self._coarse_basis = coarse
        self.stats['preconditioner'] = {'domains': len(domains), 'coarse_dimension': ngroups,
                                       'factor_bytes': memory,
                                       'symmetric_weights': 'identity',
                                       'coarse_aggregation_active': len(own) > ngroups}

    def _audit(self, packed, gamma, coeff, packed_j):
        """Independent exact-entry probes, not a full spectral certification."""
        naux, npair = len(gamma), len(packed)
        count = int(self.options['audit_samples'])
        rng = np.random.default_rng(190731)
        ai = np.sort(rng.choice(naux, min(count, naux), replace=False))
        pi = np.sort(rng.choice(npair, min(count, npair), replace=False))
        aux_all = np.arange(naux, dtype=np.int32)
        metric_rows = self.provider.metric(ai, aux_all)
        metric_error = metric_rows @ coeff - gamma[ai]
        j_exact = self.provider.three_center(pi, aux_all) @ coeff
        gamma_exact = np.zeros(len(ai))
        for start in range(0, npair, 128):
            rows = np.arange(start, min(start + 128, npair), dtype=np.int32)
            gamma_exact += self.provider.three_center(rows, ai).T @ packed[rows]
        gamma_error = gamma_exact - gamma[ai]
        j_error = j_exact - packed_j[pi]
        values = {
            'exact_metric_sample_relative_residual': float(np.linalg.norm(metric_error) / max(np.linalg.norm(gamma[ai]), 1e-30)),
            'exact_b_transpose_sample_max_error': float(np.max(abs(gamma_error))),
            'exact_b_sample_max_error': float(np.max(abs(j_error))),
            'auxiliary_samples': len(ai), 'ao_pair_samples': len(pi),
            'full_operator_certification': False,
        }
        self.stats['independent_integral_audit'] = values
        if values['exact_metric_sample_relative_residual'] > 1e-12:
            raise HierarchicalRIError('Independent exact metric sample residual exceeded 1e-12')
        if (values['exact_b_sample_max_error'] > 1e-10 or
            values['exact_b_transpose_sample_max_error'] > 1e-12 * max(1., np.max(abs(gamma)))):
            raise HierarchicalRIError('Independent exact B/B.T sample residual exceeded tolerance')

    def _precondition(self, residual):
        value = np.zeros_like(residual)
        for indices, weight, factor in self._local_factors:
            value[indices] += weight * scipy.linalg.cho_solve(
                factor, weight * residual[indices], check_finite=False)
        z = self._coarse_basis
        value += z @ scipy.linalg.cho_solve(self._coarse_factor, z.T @ residual,
                                           check_finite=False)
        return value

    def _solve(self, rhs):
        rhs = self.scale * rhs
        x = np.zeros_like(rhs)
        residual = rhs.copy()
        norm = float(np.linalg.norm(rhs))
        if norm == 0:
            return x, 0, 0.
        z = self._precondition(residual)
        p = z.copy()
        rz = float(residual @ z)
        if rz <= 0:
            raise HierarchicalRIError('Nonpositive preconditioned residual')
        target = float(self.options['pcg_tolerance']) * norm
        best_true_residual = 1.
        for iteration in range(1, int(self.options['pcg_maxiter']) + 1):
            ap = self._scaled_metric(p)
            pap = float(p @ ap)
            if pap <= 0 or not np.isfinite(pap):
                raise HierarchicalRIError('PCG detected nonpositive Coulomb curvature')
            alpha = rz / pap
            x += alpha * p
            residual -= alpha * ap
            if iteration % 25 == 0 or np.linalg.norm(residual) <= target:
                true_residual = rhs - self._scaled_metric(x)
                true_norm = float(np.linalg.norm(true_residual))
                best_true_residual = min(best_true_residual, true_norm / norm)
                if true_norm <= target:
                    return self.scale * x, iteration, true_norm / norm
                # Periodic observation must not destroy Krylov conjugacy.
                # Restart only when recursive roundoff is materially different
                # from the independently recomputed residual.
                if np.linalg.norm(true_residual - residual) > .1 * true_norm:
                    residual = true_residual
                    z = self._precondition(residual)
                    p = z.copy(); rz = float(residual @ z)
                    continue
            z = self._precondition(residual)
            next_rz = float(residual @ z)
            if next_rz <= 0 or not np.isfinite(next_rz):
                raise HierarchicalRIError('PCG preconditioner lost positive definiteness')
            p = z + (next_rz / rz) * p
            rz = next_rz
        raise HierarchicalRIError(
            f'PCG did not reach its true-residual tolerance; best={best_true_residual:.3e}')

    def build(self, dm):
        dm = np.asarray(dm, dtype=np.float64)
        if dm.shape != (self.mol.nao_nr(), self.mol.nao_nr()):
            raise ValueError('dm must have shape (nao, nao)')
        if not np.isfinite(dm).all():
            raise ValueError('dm must be finite')
        if np.max(np.abs(dm - dm.T)) > 1e-10:
            raise ValueError('RI-J currently requires a real symmetric density')
        self.prepare()
        started = time.perf_counter()
        packed = self.weights * dm[self.pair_i, self.pair_j]
        failures = []
        for tolerance in self._refinements():
            if tolerance > self.stats['compression_tolerance']:
                continue
            try:
                if tolerance < self.stats['compression_tolerance']:
                    self._build_operators(tolerance)
                gamma = self.three_center.matvec(packed, transpose=True)
                coeff, iterations, relative_residual = self._solve(gamma)
                packed_j = self.three_center.matvec(coeff)
                audit_started = time.perf_counter()
                self._audit(packed, gamma, coeff, packed_j)
                self.stats['independent_audit_seconds'] = time.perf_counter() - audit_started
                break
            except HierarchicalRIError as exc:
                failures.append(str(exc))
        else:
            raise HierarchicalRIError('All fixed-operator refinements failed: ' + '; '.join(failures))
        if not np.isfinite(packed_j).all():
            raise HierarchicalRIError('Nonfinite Coulomb result')
        energy = .5 * float(packed @ packed_j)
        identity = abs(energy - .5 * float(gamma @ coeff))
        if identity > 5e-11 * max(1., abs(energy)):
            raise HierarchicalRIError('B/B.T Coulomb energy identity failed')
        kernel_seconds = time.perf_counter() - started
        expand_started = time.perf_counter()
        result = np.zeros_like(dm)
        result[self.pair_i, self.pair_j] = packed_j
        result[self.pair_j, self.pair_i] = packed_j
        self.stats.update(pcg_iterations=iterations, pcg_relative_residual=relative_residual,
                          coulomb_energy=energy, energy_identity_error=identity,
                          build_seconds=kernel_seconds,
                          dense_expansion_seconds=time.perf_counter() - expand_started,
                          integral_calls=self.provider.integral_calls,
                          refinement_failures=failures)
        return result
