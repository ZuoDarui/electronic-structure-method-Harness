"""Local, projected-density COSX with explicit full-equation checks.

Gaussian AO/integral evaluation and the optional fused G=V*F integral primitive
come from PySCF. This module does not call a PySCF J/K builder or SCF solver.
The overlap projection is applied to
the density *before* the grid contraction.  The default symmetric-overlap
variant also corrects the output on the left, matching PySCF SGX's default.
Local linear solves are accepted only after checking the complete sparse
overlap equation, and expanded when that residual is too large.

Spatial tail thresholds are numerical approximations, not proven global
energy bounds.  Independent same-grid SGX comparisons and scaling measurements
are therefore required before declaring a molecule/accuracy range validated.
"""

from __future__ import annotations

from collections import defaultdict, OrderedDict
import math
from types import SimpleNamespace
import time
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from scipy.spatial import cKDTree
from pyscf import gto
from pyscf.dft import gen_grid, radi, numint
from pyscf.gto import moleintor


def _runs(indices, limit=0):
    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        return []
    result, start, last = [], int(indices[0]), int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == last + 1 and (limit <= 0 or value - start < limit):
            last = value
        else:
            result.append((start, last + 1))
            start = last = value
    result.append((start, last + 1))
    return result


class _AtomBlocks:
    """Accumulate bounded AO blocks without allocating a dense global matrix."""

    def __init__(self, atom_slices, nao, max_bytes=None):
        self.slices = atom_slices
        self.ends = atom_slices[:, 3]
        self.nao = int(nao)
        self.blocks = {}
        self.max_bytes=max_bytes
        self.bytes=0

    def add(self, rows, cols, values):
        rows, cols = np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)
        if not len(rows) or not len(cols):
            return
        ra = np.searchsorted(self.ends, rows, side="right")
        ca = np.searchsorted(self.ends, cols, side="right")
        row_groups = [(a, np.flatnonzero(ra == a)) for a in np.unique(ra)]
        col_groups = [(a, np.flatnonzero(ca == a)) for a in np.unique(ca)]
        for a, ri in row_groups:
            a0, a1 = self.slices[a, 2:]
            for b, ci in col_groups:
                b0, b1 = self.slices[b, 2:]
                value = values[np.ix_(ri, ci)]
                if not np.any(value):
                    continue
                key = (int(a), int(b))
                target = self.blocks.get(key)
                if target is None:
                    size=int(a1-a0)*int(b1-b0)*8
                    if self.max_bytes is not None and self.bytes+size>self.max_bytes:
                        raise MemoryError("COSX sparse block storage exceeds memory budget")
                    target = self.blocks[key] = np.zeros((a1-a0, b1-b0))
                    self.bytes+=size
                target[np.ix_(rows[ri]-a0, cols[ci]-b0)] += value

    def sparse(self, threshold=0.0):
        rr, cc, vv = [], [], []
        for (a, b), value in self.blocks.items():
            r, c = np.nonzero(np.abs(value) > threshold)
            if len(r):
                rr.append(r+self.slices[a, 2])
                cc.append(c+self.slices[b, 2])
                vv.append(value[r, c])
        if not vv:
            return scipy.sparse.csr_matrix((self.nao, self.nao))
        return scipy.sparse.coo_matrix(
            (np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
            shape=(self.nao, self.nao)).tocsr()


class LocalCOSX:
    def __init__(self, mol, grid_rad=30, grid_ang=110, batch_size=512,
                 backend="serial", overlap_fit=True, max_memory_mb=8000,
                 sym_ovlp=True,
                 projection_rtol=1e-12, projection_atol=1e-13,
                 spatial_precision=1e-16, fused_integrals=True):
        if min(grid_rad, grid_ang, batch_size, max_memory_mb) <= 0:
            raise ValueError("grid sizes, batch size and memory must be positive")
        if backend not in ("serial", "cpu", "gpu"):
            raise ValueError("backend must be serial, cpu, or gpu")
        if not 0 < projection_rtol < 1 or not 0 < projection_atol < 1:
            raise ValueError("projection tolerances must be positive and below one")
        if not 0 < spatial_precision < 1:
            raise ValueError("spatial_precision must be positive and below one")
        self.mol, self.backend = mol, backend
        self.nao = mol.nao_nr()
        self.nshell = mol.nbas
        self.ao_loc = mol.ao_loc_nr()
        self.atom_slices = np.asarray(mol.aoslice_by_atom(), dtype=int)
        self.atom_coords = np.asarray(mol.atom_coords())
        self.shell_atoms = np.asarray(mol._bas[:, gto.ATOM_OF], dtype=int)
        self.shell_coords = self.atom_coords[self.shell_atoms]
        self.shell_tree = cKDTree(self.shell_coords)
        self.grid_rad, self.grid_ang = int(grid_rad), int(grid_ang)
        self.blksize = int(batch_size)
        self.overlap_fit = bool(overlap_fit)
        self.sym_ovlp = bool(sym_ovlp)
        self.max_memory_mb = float(max_memory_mb)
        if self.nao*self.nao*8 > self.max_memory_mb*1e6:
            raise MemoryError("dense K output exceeds configured memory budget")
        self.projection_rtol = float(projection_rtol)
        self.projection_atol = float(projection_atol)
        self.spatial_precision = float(spatial_precision)
        self.fused_integrals=bool(fused_integrals)
        self.stats = {}
        self._prepared = False
        self._cupy = None
        if backend == "gpu":
            import cupy
            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("GPU COSX requested but CUDA device is unavailable")
            self._cupy = cupy
        self.shell_radii = self._radii(spatial_precision)
        self._largest_radius = float(np.max(self.shell_radii))
        # For f=chi_mu*chi_nu, ||f||_1 <= sqrt(S_mm S_nn).
        # Splitting the Coulomb integral into a radius-a ball and its exterior
        # gives 2*pi*a^2*||f||_inf + ||f||_1/a.  Minimization gives the uniform
        # point-potential bound used below; no point-charge singularity is
        # applied to a Gaussian AO product.
        amplitude=0.
        for sh in range(self.nshell):
            alpha=np.asarray(mol.bas_exp(sh)); angular=mol.bas_angular(sh)
            radial=np.ones_like(alpha) if angular==0 else (angular/(2*np.e*alpha))**(angular/2)
            amplitude=max(amplitude,float(10*np.max(
                radial @ np.abs(mol._libcint_ctr_coeff(sh)))))
        diagonal=np.diag(mol.intor('int1e_ovlp')) if self.nao<256 else np.ones(self.nao)
        # Spherical AOs are normalized. Cartesian contractions can carry a
        # different component norm, so obtain their diagonal shell by shell.
        if mol.cart and self.nao>=256:
            diagonal=np.concatenate([np.diag(mol.intor('int1e_ovlp',
                shls_slice=(sh,sh+1,sh,sh+1))) for sh in range(self.nshell)])
        l1=float(np.max(diagonal))
        self._potential_bound=1.5*(4*np.pi*amplitude**2*l1**2)**(1/3)
        self._factor_cache = {}
        self._factor_bytes = 0
        self._input_scan_seconds=0.
        self._potential_block_bounds={}
        self._fused_cache=OrderedDict()
        self._shell_pair_bounds={}
        self._chemical_pair_bounds={}
        self._subset_bounds=OrderedDict()
        self._pair_models=[(np.asarray(mol.bas_exp(sh)),
            np.abs(mol._libcint_ctr_coeff(sh)).max(axis=1),mol.bas_angular(sh))
            for sh in range(self.nshell)]
        signatures={}
        self._basis_types=[]
        for alpha,coeff,angular in self._pair_models:
            key=(alpha.tobytes(),coeff.tobytes(),angular)
            if key not in signatures:signatures[key]=len(signatures)
            self._basis_types.append(signatures[key])
        self._potential_neighbors=[]
        for sh in range(self.nshell):
            candidates=np.asarray(self.shell_tree.query_ball_point(
                self.shell_coords[sh],self.shell_radii[sh]+self._largest_radius),dtype=int)
            distances=np.linalg.norm(self.shell_coords[candidates]-self.shell_coords[sh],axis=1)
            self._potential_neighbors.append(candidates[
                distances<=self.shell_radii[sh]+self.shell_radii[candidates]])

    def _dot(self, left, right):
        if self._cupy is None:
            return left @ right
        cp = self._cupy
        return cp.asnumpy(cp.asarray(left) @ cp.asarray(right))

    def _radii(self, precision):
        radii = np.zeros(self.nshell)
        for sh in range(self.nshell):
            alpha = np.asarray(self.mol.bas_exp(sh))
            # Actual libcint radial coefficients include primitive normalization.
            coeff = np.abs(self.mol._libcint_ctr_coeff(sh)).sum(axis=1)
            angular = self.mol.bas_angular(sh)
            radius = max(1., np.sqrt(angular / (2*float(alpha.min()))))
            for _ in range(160):
                bound = 10 * (radius**angular) * np.dot(coeff, np.exp(-alpha*radius*radius))
                if bound < precision:
                    break
                radius *= 1.15
            else:
                raise RuntimeError("could not establish a finite Gaussian support radius")
            radii[sh] = radius
        return radii

    def _near(self, xyz):
        midpoint = .5*(xyz.min(axis=0)+xyz.max(axis=0))
        extent = float(np.linalg.norm(xyz-midpoint, axis=1).max())
        candidates = np.asarray(self.shell_tree.query_ball_point(
            midpoint, self._largest_radius+extent), dtype=int)
        if not len(candidates):
            return candidates
        delta = self.shell_coords[candidates]-midpoint
        keep = np.linalg.norm(delta, axis=1) <= self.shell_radii[candidates]+extent
        return np.sort(candidates[keep])

    def _ao(self, xyz):
        shells = self._near(xyz)
        columns, pieces = [], []
        for start, stop in _runs(shells):
            columns.append(np.arange(self.ao_loc[start], self.ao_loc[stop]))
            pieces.append(numint.eval_ao(self.mol, xyz, shls_slice=(start, stop)))
        if not columns:
            return np.zeros(0, dtype=int), np.empty((len(xyz), 0))
        return np.concatenate(columns), np.concatenate(pieces, axis=1)

    def prepare(self):
        if self._prepared:
            return self
        started = time.perf_counter()
        grids = gen_grid.Grids(self.mol)
        grids.atom_grid = {self.mol.atom_pure_symbol(i):
                          (self.grid_rad, self.grid_ang) for i in range(self.mol.natm)}
        grids.radi_method = radi.treutler_ahlrichs
        grids.prune = gen_grid.nwchem_prune
        grids.build(with_non0tab=True)
        self.coords, self.weights = grids.coords, grids.weights
        self.ngrids = len(self.weights)
        bins = defaultdict(list)
        for i, key in enumerate(np.floor(self.coords / 6.).astype(int)):
            bins[tuple(key)].append(i)
        self.batches = [np.asarray(indices[j:j+self.blksize], dtype=int)
                        for indices in bins.values()
                        for j in range(0, len(indices), self.blksize)]
        accum = _AtomBlocks(self.atom_slices, self.nao,self.max_memory_mb*1e6*.3)
        for index in self.batches:
            columns, ao = self._ao(self.coords[index])
            if len(columns):
                accum.add(columns, columns,
                          self._dot(ao.T, self.weights[index, None]*ao))
        self.s_num = accum.sparse(threshold=1e-18)
        self.s_num = ((self.s_num+self.s_num.T)*.5).tocsr()
        exact = _AtomBlocks(self.atom_slices, self.nao,self.max_memory_mb*1e6*.3)
        atom_radii = np.array([self.shell_radii[self.shell_atoms == a].max()
                               for a in range(self.mol.natm)])
        tree = cKDTree(self.atom_coords)
        for a, (a0, a1, i0, i1) in enumerate(self.atom_slices):
            neighbors = tree.query_ball_point(self.atom_coords[a],
                                              atom_radii[a]+atom_radii.max())
            for b in neighbors:
                if b > a or np.linalg.norm(self.atom_coords[a]-self.atom_coords[b]) > atom_radii[a]+atom_radii[b]:
                    continue
                b0, b1, j0, j1 = self.atom_slices[b]
                block = self.mol.intor('int1e_ovlp', shls_slice=(a0,a1,b0,b1))
                exact.add(np.arange(i0,i1),np.arange(j0,j1),block)
                if a != b:
                    exact.add(np.arange(j0,j1),np.arange(i0,i1),block.T)
        self.overlap = exact.sparse(threshold=1e-18)
        self._prepared = True
        self.stats.update(prepare_seconds=time.perf_counter()-started,
                          numerical_overlap_nnz=self.s_num.nnz,
                          analytic_overlap_nnz=self.overlap.nnz,
                          ngrids=self.ngrids, projection_method="expanded-local-full-residual",
                          device=self.backend, scaling_validated=False,
                          projection_formula=("P=I; K=sym(Kraw(D))" if not self.overlap_fit else
                              ("P=(Sn^-1 S+I)/2; K=sym(P.T Kraw(P D))"
                              if self.sym_ovlp else "P=Sn^-1 S; K=sym(Kraw(P D))")),
                          integral_device="cpu",projection_device="cpu",
                          contraction_device="gpu" if self._cupy is not None else "cpu")
        return self

    def _solve_overlap_rhs(self, rhs, prefix="projection"):
        rhs = scipy.sparse.csc_matrix(rhs)
        graph = self.s_num.copy()
        graph.data = np.ones_like(graph.data)
        values, rows, cols = [], [], []
        max_residual, largest_domain, expansions = 0., 0, 0
        for _, _, first, last in self.atom_slices:
            block_rhs = rhs[:, first:last].toarray()
            norm = float(np.linalg.norm(block_rhs))
            if norm == 0:
                continue
            support = np.flatnonzero(np.max(np.abs(block_rhs), axis=1) >
                                     self.projection_atol/max(1.,np.sqrt(self.nao)))
            if not len(support):
                support = np.array([int(np.argmax(np.max(np.abs(block_rhs),axis=1)))])
            solution = None
            for attempt in range(30):
                support = np.unique(support)
                key = tuple(support)
                factor = self._factor_cache.get(key)
                if factor is None:
                    if len(support)**2*8 > self.max_memory_mb*1e6*.3:
                        raise MemoryError("local overlap factor exceeds memory budget")
                    local = self.s_num[support][:,support].toarray()
                    try:
                        factor = scipy.linalg.cho_factor(local, lower=True, check_finite=False)
                    except scipy.linalg.LinAlgError as exc:
                        raise RuntimeError("numerical overlap local block is not positive definite") from exc
                    size = local.nbytes
                    if self._factor_bytes+size <= self.max_memory_mb*1e6*.2:
                        self._factor_cache[key] = factor
                        self._factor_bytes += size
                candidate = scipy.linalg.cho_solve(factor, block_rhs[support], check_finite=False)
                residual = self.s_num[:,support] @ candidate-block_rhs
                residual_norm = float(np.linalg.norm(residual))
                if residual_norm <= self.projection_atol+self.projection_rtol*norm:
                    solution = candidate
                    max_residual = max(max_residual, residual_norm/max(norm,1e-300))
                    break
                if len(support) == self.nao:
                    raise RuntimeError("full numerical overlap projection residual exceeds tolerance")
                touch = graph[:,support].nonzero()[0]
                bad = np.flatnonzero(np.max(np.abs(residual),axis=1) >
                                      (self.projection_atol+self.projection_rtol*norm)/np.sqrt(self.nao))
                grown = np.union1d(support, np.union1d(touch,bad))
                if len(grown) == len(support):
                    grown = np.arange(self.nao)
                support = grown
                expansions += 1
            if solution is None:
                raise RuntimeError("local projection did not meet the complete overlap equation")
            largest_domain = max(largest_domain,len(support))
            rr,cc = np.nonzero(solution)
            rows.append(support[rr]);cols.append(cc+first);values.append(solution[rr,cc])
        if values:
            projected = scipy.sparse.coo_matrix((np.concatenate(values),
                (np.concatenate(rows),np.concatenate(cols))),shape=(self.nao,self.nao)).tocsr()
        else:
            projected = scipy.sparse.csr_matrix((self.nao,self.nao))
        self.stats.update({prefix+"_residual_relative":max_residual,
                          prefix+"_largest_domain":largest_domain,
                          prefix+"_expansions":expansions,
                          prefix+"_nnz":projected.nnz,
                          "projection_factor_bytes":self._factor_bytes})
        return projected

    def _solve_projection(self, density):
        scan_started=time.perf_counter()
        if not scipy.sparse.issparse(density):
            # This input-interface scan is O(nao^2), like accepting the dense
            # SCF density itself.  Interior sparse operations must not inherit
            # numerically insignificant dense tails from that interface.
            density = np.array(density,copy=True)
            threshold = min(1e-18,self.projection_atol/(100*max(1,self.nao)))
            density[np.abs(density)<threshold] = 0.
        density = scipy.sparse.csr_matrix(density)
        self._input_scan_seconds+=time.perf_counter()-scan_started
        if not self.overlap_fit:
            return density
        projected = self._solve_overlap_rhs(self.overlap @ density)
        if self.sym_ovlp:
            projected = .5*(projected+density)
        return projected.tocsr()

    def _pair_potential_bound(self, a, b):
        """Uniform Coulomb bound from absolute Gaussian product moments."""
        aa,ca,la=self._pair_models[a];bb,cb,lb=self._pair_models[b]
        distance=float(np.linalg.norm(self.shell_coords[a]-self.shell_coords[b]))
        # Repeated atomic basis functions at identical separations have the
        # same analytic bound.  Round only the cache label; bracket the actual
        # separation when evaluating the monotone exponential and polynomial.
        bucket=round(distance,10)
        chemical_key=(self._basis_types[a],self._basis_types[b],bucket)
        if chemical_key in self._chemical_pair_bounds:
            return self._chemical_pair_bounds[chemical_key]
        dlow=max(0.,bucket-1e-9);dhigh=bucket+1e-9
        aa=aa[:,None];bb=bb[None,:];p=aa+bb
        prefactor=(ca[:,None]*cb[None,:])*np.exp(-(aa*bb/p)*dlow*dlow)
        shift_a=bb/p*dhigh;shift_b=aa/p*dhigh
        mass=np.zeros_like(p);peak=np.zeros_like(p)
        for i in range(la+1):
            for j in range(lb+1):
                n=i+j
                poly=math.comb(la,i)*math.comb(lb,j)*shift_a**(la-i)*shift_b**(lb-j)
                mass+=poly*(2*np.pi*math.gamma((n+3)/2))*p**(-(n+3)/2)
                peak+=poly*(1. if n==0 else (n/(2*np.e*p))**(n/2))
        l1=100*float(np.sum(prefactor*mass))
        linf=100*float(np.sum(prefactor*peak))
        result=min(self._potential_bound,1.5*(4*np.pi*linf*l1*l1)**(1/3))
        self._chemical_pair_bounds[chemical_key]=result
        return result

    def _subset_potential_bounds(self,shells):
        key=tuple(map(int,shells))
        value=self._subset_bounds.get(key)
        if value is not None:
            self._subset_bounds.move_to_end(key)
            return value
        value=np.empty((len(shells),len(shells)))
        for i,a in enumerate(shells):
            for j,b in enumerate(shells[:i+1]):
                pair=(int(a),int(b))
                bound=self._shell_pair_bounds.get(pair)
                if bound is None:
                    bound=self._pair_potential_bound(a,b)
                    self._shell_pair_bounds[pair]=bound
                value[i,j]=value[j,i]=bound
        self._subset_bounds[key]=value
        if len(self._subset_bounds)>8:self._subset_bounds.popitem(last=False)
        return value

    def _run_potential_bound(self,i0,i1,j0,j1):
        key=(i0,i1,j0,j1)
        if key not in self._potential_block_bounds:
            distances=np.linalg.norm(self.shell_coords[i0:i1,None]-
                                     self.shell_coords[None,j0:j1],axis=2)
            if np.min(distances)<2.:
                bound=self._potential_bound
            else:
                bound=max(self._pair_potential_bound(i,j)
                          for i in range(i0,i1) for j in range(j0,j1))
            self._potential_block_bounds[key]=bound
        return self._potential_block_bounds[key]

    def _fused_integral_contraction(self,shells,xyz,f,weights,shell_mask):
        """Use only the compiled Gaussian-integral/F contraction primitive.

        Projection, AO evaluation, support, screening, K assembly and SCF
        stay outside this driver. The same C integral engine as PySCF is
        shared, and therefore is not independently verified by SGX tests.
        """
        from pyscf.sgx import sgx_jk
        key=tuple(map(int,shells))
        cached=self._fused_cache.get(key)
        if cached is None:
            local=self.mol.copy(deep=False)
            local._bas=np.asarray(self.mol._bas[shells],dtype=np.int32,order='C')
            intor=local._add_suffix('int1e_grids')
            optimizer=SimpleNamespace(_cintopt=moleintor.make_cintopt(
                local._atm,local._bas,local._env,intor),_intor=intor)
            grid=SimpleNamespace(weights=np.ones(1),coords=np.zeros((1,3)))
            data=sgx_jk.SGXData(local,grid,fit_ovlp=False,sym_ovlp=False,
                sgxopt=optimizer,bound_algo='sample',etol=None,vtol=None,
                direct_scf_tol=.5)
            data._ovlp_proj=None
            data._mbar_ij=np.ones((len(shells),len(shells)),order='C')
            data._mmax_i=np.ones(len(shells))
            data._msum_i=np.ones(len(shells))*len(shells)
            data._setup_opt()
            driver=sgx_jk._gen_k_direct(local,'s2',.5,data)
            cached=(local,data,driver)
            self._fused_cache[key]=cached
            # Bounded number of local integral optimizers; no growing global
            # shell-pair object is retained for each grid batch.
            if len(self._fused_cache)>8:
                self._fused_cache.popitem(last=False)
        else:
            self._fused_cache.move_to_end(key)
        local,data,driver=cached
        data._mbar_ij[:]=shell_mask
        fg=np.ascontiguousarray(f.T[None,:,:])
        return driver(local,np.asarray(xyz,order='C'),fg,
                      np.asarray(weights,order='C'),0)[0].T

    def _potential(self, xyz, active_f_shells, f_columns, f,
                   weighted_ao, error_budget, screen_f=None,weights=None):
        """Contract V_mn(g) f_n(g); output support differs from AO(g) support."""
        shells=np.unique(np.concatenate([self._potential_neighbors[sh]
                                         for sh in active_f_shells]))
        columns = np.concatenate([np.arange(self.ao_loc[s],self.ao_loc[s+1]) for s in shells])
        f_full = np.zeros((len(xyz),len(columns)))
        f_full[:,np.searchsorted(columns,f_columns)] = f
        screen_full=f_full
        if screen_f is not None:
            screen_full=np.zeros_like(f_full)
            screen_full[:,np.searchsorted(columns,f_columns)]=screen_f
        if self.fused_integrals:
            per_column=weighted_ao @ np.maximum(np.abs(f_full),np.abs(screen_full))
            local_ao=np.concatenate(([0],np.cumsum(self.ao_loc[shells+1]-self.ao_loc[shells])))
            shell_scores=np.array([per_column[local_ao[i]:local_ao[i+1]].sum()
                                   for i in range(len(shells))])
            bounds=self._subset_potential_bounds(shells)
            contributions=bounds*np.maximum(shell_scores[:,None],shell_scores[None,:])
            pair_budget=error_budget/max(1,len(shells)*(len(shells)+1)//2)
            shell_mask=np.asarray(contributions>pair_budget,dtype=float)
            omitted=float(np.triu(np.where(shell_mask,0.,contributions)).sum())
            result=self._fused_integral_contraction(shells,xyz,f_full,
                np.ones(len(xyz)) if weights is None else weights,shell_mask)
            return columns,result,omitted
        result = np.zeros_like(f_full)
        intor = self.mol._add_suffix('int1e_grids')
        env = np.concatenate((self.mol._env,np.asarray(xyz).ravel()))
        env[moleintor.NGRIDS] = len(xyz)
        env[moleintor.PTR_GRIDS] = len(self.mol._env)
        optimizer = moleintor.make_cintopt(self.mol._atm,self.mol._bas,env,intor)
        runs = _runs(shells,32)
        per_pair_budget=error_budget/max(1,len(runs)*(len(runs)+1)//2)
        omitted_bound=0.
        for left,(i0,i1) in enumerate(runs):
            ip = np.searchsorted(columns,np.arange(self.ao_loc[i0],self.ao_loc[i1]))
            fi = f_full[:,ip]
            for j0,j1 in runs[:left+1]:
                jp = np.searchsorted(columns,np.arange(self.ao_loc[j0],self.ao_loc[j1]))
                fj = f_full[:,jp]
                if not np.any(fi) and not np.any(fj):
                    continue
                bound=self._run_potential_bound(i0,i1,j0,j1)
                fi_bound=weighted_ao @ np.maximum(np.abs(fi),np.abs(screen_full[:,ip])).sum(axis=1)
                fj_bound=weighted_ao @ np.maximum(np.abs(fj),np.abs(screen_full[:,jp])).sum(axis=1)
                contribution=float(bound*max(fi_bound,fj_bound))
                if contribution<=per_pair_budget:
                    omitted_bound+=contribution
                    continue
                integral = moleintor.getints(intor,self.mol._atm,self.mol._bas,env,
                                             (i0,i1,j0,j1),cintopt=optimizer)
                result[:,ip] += np.einsum('gij,gj->gi',integral,fj,optimize=True)
                if (i0,i1) != (j0,j1):
                    result[:,jp] += np.einsum('gij,gi->gj',integral,fi,optimize=True)
        return columns,result,omitted_bound

    def build(self, dm, screen_dm=None, direct_scf_tol=1e-13, ao_tol=1e-10):
        dm = np.asarray(dm,dtype=np.float64)
        if dm.shape != (self.nao,self.nao) or not np.isfinite(dm).all():
            raise ValueError("density must be a finite (nao,nao) matrix")
        if np.max(np.abs(dm-dm.T)) > 1e-10:
            raise ValueError("LocalCOSX supports real symmetric densities")
        if direct_scf_tol <= 0 or ao_tol <= 0:
            raise ValueError("screening tolerances must be positive")
        if ao_tol<self.spatial_precision:
            raise ValueError("ao_tol is tighter than configured spatial_precision")
        if screen_dm is not None:
            screen_dm = np.asarray(screen_dm,dtype=np.float64)
            if screen_dm.shape != dm.shape or not np.isfinite(screen_dm).all():
                raise ValueError("screen_dm must be finite and have the density shape")
        self.prepare()
        started=time.perf_counter()
        self._input_scan_seconds=0.
        projected=self._solve_projection(dm)
        # Full-density stencil can only add work to an incremental build.
        # Geometry and unthresholded projected coefficients remain identical.
        screening=projected if screen_dm is None else self._solve_projection(screen_dm)
        projection_seconds=time.perf_counter()-started
        accum=_AtomBlocks(self.atom_slices,self.nao,self.max_memory_mb*1e6*.3)
        max_fcols=0
        raw_discarded_bound=0.
        raw_budget=.5*min(1e-12,float(direct_scf_tol))/max(1,len(self.batches))
        for index in self.batches:
            columns,ao=self._ao(self.coords[index])
            if not len(columns):continue
            support=np.union1d(projected[columns].indices,screening[columns].indices)
            if not len(support):continue
            f=self._dot(ao,projected[columns][:,support].toarray())
            screen_f=f if screen_dm is None else self._dot(
                ao,screening[columns][:,support].toarray())
            weighted_ao=np.abs(self.weights[index])*np.max(np.abs(ao),axis=1)
            scores=self._potential_bound*(weighted_ao @ np.maximum(np.abs(f),np.abs(screen_f)))
            order=np.argsort(scores,kind='stable')
            count=int(np.searchsorted(np.cumsum(scores[order]),raw_budget,side='right'))
            if count:
                keep=np.ones(len(support),dtype=bool);keep[order[:count]]=False
                raw_discarded_bound+=float(scores[order[:count]].sum())
                support=support[keep];f=f[:,keep];screen_f=screen_f[:,keep]
            if not len(support):continue
            max_fcols=max(max_fcols,len(support))
            fshell=np.unique(np.searchsorted(self.ao_loc[1:],support,side='right'))
            out,g,integral_error=self._potential(self.coords[index],fshell,support,f,
                                                weighted_ao,raw_budget,screen_f,self.weights[index])
            raw_discarded_bound+=integral_error
            # Preserve the unsymmetrized orientation: AO.T @ W @ G.
            # Transposing before the left overlap correction changes the
            # symmetric-fit method, even though a final symmetrization hides
            # that transpose when overlap fitting is disabled.
            accum.add(columns,out,self._dot((self.weights[index,None]*ao).T,g))
        raw=accum.sparse()
        contraction_seconds=time.perf_counter()-started-projection_seconds
        output_projection_started=time.perf_counter()
        if self.overlap_fit and self.sym_ovlp:
            # P.T Kraw = 1/2 [S Sn^-1 Kraw + Kraw].  Projecting on
            # both sides is the default PySCF SGX symmetric-overlap variant.
            raw=.5*(self.overlap @ self._solve_overlap_rhs(raw,"output_projection")+raw)
        sparse_k=(raw+raw.T)*.5
        output_projection_seconds=time.perf_counter()-output_projection_started
        kernel_seconds=time.perf_counter()-started-self._input_scan_seconds
        expand=time.perf_counter()
        required=self.nao*self.nao*8
        if required > self.max_memory_mb*1e6:
            raise MemoryError("dense K output exceeds configured memory budget")
        k=sparse_k.toarray()
        self.stats.update(build_seconds=kernel_seconds,kernel_seconds=kernel_seconds,
            total_seconds=time.perf_counter()-started,
            projection_seconds=projection_seconds,contraction_seconds=contraction_seconds,
            output_projection_seconds=output_projection_seconds,
            dense_input_scan_seconds=self._input_scan_seconds,
            dense_expansion_seconds=time.perf_counter()-expand,k_nnz=sparse_k.nnz,
            maximum_density_output_support=max_fcols,screening_policy="conservative-geometry-and-full-density-support",
            direct_scf_tol=float(direct_scf_tol),requested_ao_tol=float(ao_tol),
            actual_spatial_precision=self.spatial_precision,sym_ovlp=self.sym_ovlp)
        self.stats.update(raw_exchange_screen_bound=raw_discarded_bound,
                          uniform_gaussian_potential_bound=self._potential_bound,
                          shared_fused_integral_kernel=self.fused_integrals,
                          potential_integral_kernel=("pyscf.libcvhf.SGXnr_direct_k_drv"
                              if self.fused_integrals else "libcint.int1e_grids"))
        if not np.isfinite(k).all():
            raise RuntimeError("COSX produced nonfinite exchange")
        return k
