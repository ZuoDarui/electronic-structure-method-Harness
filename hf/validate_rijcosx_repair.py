"""Independent, fail-closed numerical acceptance for repaired RIJCOSX.

The reference uses PySCF DF/SGX kernels and its own SCF solver. Timing includes
all streamed integral work; dense input/output and initial preparation are
also reported, never presented as part of a linear-scaling entire SCF.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh
from pyscf import df, gto, lib, scf
from pyscf.df import df_jk
from pyscf.sgx import sgx, sgx_jk

from hf import Hartree_Fock, matching_cosx_grids, orbital_gradient_rms, read_xyz
from make_alkane_xyz import alkane_coordinates


def reference_exchange(mol, radial, angular, memory):
    engine = sgx.SGX(mol, auxbasis='weigend')
    engine.verbose = 0
    engine.max_memory = memory
    engine.build()
    engine.grids = matching_cosx_grids(mol, radial, angular)
    return engine


def run(args, result):
    lib.num_threads(args.threads)
    xyz=Path(__file__).with_name(f'c{args.carbons}h{2*args.carbons+2}.xyz')
    atoms=read_xyz(xyz)[0] if xyz.exists() else alkane_coordinates(args.carbons)
    result['geometry_source']=str(xyz.name) if xyz.exists() else 'make_alkane_xyz.alkane_coordinates'
    if xyz.exists():result['geometry_sha256']=hashlib.sha256(xyz.read_bytes()).hexdigest()
    mol = gto.M(atom=atoms, basis=args.basis,
                unit='Angstrom', verbose=0, max_memory=args.memory_mb)
    overlap = mol.intor('int1e_ovlp')
    hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    density = scf.hf.init_guess_by_minao(mol)
    result.update(carbons=args.carbons, nao=mol.nao_nr(), nelectron=mol.nelectron,
                  basis=args.basis, backend=args.backend, engine=args.engine,
                  threads=args.threads, memory_mb=args.memory_mb,
                  k_memory_mb=args.k_memory_mb, memory_ceiling_mb=args.memory_ceiling_mb,
                  dense_backend=args.dense_backend,
                  grid_switch_energy_tol=args.grid_switch_energy_tol, fixed_density=[])
    start = time.perf_counter()
    solver = Hartree_Fock(mol, max_memory=args.memory_mb,
                         parallel_backend=args.backend, jk_engine=args.engine,
                         k_max_memory_mb=args.k_memory_mb,
                         memory_ceiling_mb=args.memory_ceiling_mb,
                         dense_backend=args.dense_backend)
    result['solver_setup_seconds'] = time.perf_counter()-start
    ref_df = df.DF(mol, auxbasis='weigend')
    ref_df.max_memory = args.memory_mb
    ref_k = reference_exchange(mol,30,110,args.memory_mb)
    for label in ('MINAO','first_updated_density'):
        started = time.perf_counter()
        j,k = solver.get_jk(density, grid_rad=30,grid_ang=110,batch_size=512,
                            cosx_tol=1e-13)
        elapsed = time.perf_counter()-started
        jref = df_jk.get_j(ref_df,density,hermi=1,direct_scf_tol=1e-13)
        kref = sgx_jk.get_k_only(ref_k,density,hermi=1,direct_scf_tol=1e-13)
        metrics = {
            'density':label,'candidate_seconds':elapsed,
            'j_error_max_abs':float(np.max(np.abs(j-jref))),
            'k_error_max_abs':float(np.max(np.abs(k-kref))),
            'coulomb_energy_error':abs(float(.5*np.einsum('ij,ji',density,j-jref))),
            'exchange_energy_error':abs(float(-.25*np.einsum('ij,ji',density,k-kref))),
            'coulomb_energy':float(.5*np.einsum('ij,ji',density,j)),
            'exchange_energy':float(-.25*np.einsum('ij,ji',density,k)),
            'j_stats':dict(solver.last_j_stats),'k_stats':dict(solver.last_k_stats),
            'k_grid':solver._k_grid_note(),
        }
        result['fixed_density'].append(metrics)
        print(json.dumps(metrics,default=lambda v:v.item()),flush=True)
        if (metrics['j_error_max_abs']>1e-8 or metrics['k_error_max_abs']>1e-8
                or metrics['coulomb_energy_error']>1e-7
                or metrics['exchange_energy_error']>1e-7):
            raise RuntimeError('Independent fixed-density J/K accuracy gate failed')
        if label=='MINAO':
            _,coeff=eigh(hcore+j-.5*k,overlap)
            occupied=coeff[:,:mol.nelec[0]]
            density=2*occupied@occupied.T
            result['first_density_electron_error']=abs(float(np.einsum('ij,ji',density,overlap))-mol.nelectron)
            if result['first_density_electron_error']>1e-6:
                raise RuntimeError('Electron count failed after first diagonalization')
    if args.benchmark:
        result['timing_repetitions']=[]
        solver.get_jk(density,grid_rad=30,grid_ang=110,cosx_tol=1e-13)
        for repetition in range(3):
            timings={}
            solver.get_jk(density,grid_rad=30,grid_ang=110,cosx_tol=1e-13,timings=timings)
            result['timing_repetitions'].append({**timings,'j_stats':dict(solver.last_j_stats),
                                                'k_stats':dict(solver.last_k_stats)})
        result['timing_median']={key:float(np.median([r[key] for r in result['timing_repetitions']]))
                                 for key in ('j_seconds','k_seconds')}
        result['kernel_timing_median']={
            'j_seconds':float(np.median([r['j_stats']['build_seconds'] for r in result['timing_repetitions']])),
            'k_seconds':float(np.median([r['k_stats'].get('kernel_seconds',r['k_stats']['build_seconds'])
                                         for r in result['timing_repetitions']]))}
        result['timing_scope']='kernel medians include integral work/audit/projection; dense interface costs reported separately; wall medians retain all get_jk costs'
    if args.scf:
        energy,density=solver.solve_rhf(
            mol.nelec,hcore,overlap,mol,mol.energy_nuc(),tol=1e-8,
            jk_mode='rijcosx',auxbasis='weigend',grid_rad=30,grid_ang=110,
            final_grid_rad=35,final_grid_ang=302,incremental_k=True,
            kfull_every=5,max_iteration=200,
            grid_switch_energy_tol=args.grid_switch_energy_tol)
        reference=scf.RHF(mol).COSX(auxbasis='weigend')
        reference.verbose=4
        reference.max_memory=args.memory_mb
        reference.conv_tol=1e-8
        reference.conv_tol_grad=1e-8*np.sqrt(mol.nelec[0]*(mol.nao_nr()-mol.nelec[0]))
        reference.max_cycle=200
        reference.with_df.dfj=True
        reference.with_df.grids_level_i=reference.with_df.grids_level_f=1
        original_build=reference.with_df.build
        def same_grid(level=None):
            original_build(level)
            reference.with_df.grids=matching_cosx_grids(mol,35,302)
        reference.with_df.build=same_grid
        reference_energy=reference.kernel()
        final_engine=reference_exchange(mol,35,302,args.memory_mb)
        final_j=df_jk.get_j(ref_df,density,hermi=1,direct_scf_tol=1e-13)
        final_k=sgx_jk.get_k_only(final_engine,density,hermi=1,direct_scf_tol=1e-13)
        independent_fock=hcore+final_j-.5*final_k
        recomputed=mol.energy_nuc()+.5*np.einsum('ij,ji',density,hcore+independent_fock)
        # An occupied subspace reconstructed from DS gives an independent
        # orbital-gradient check in the original AO metric.
        values,vectors=eigh(overlap)
        root=(vectors*np.sqrt(values))@vectors.T
        invroot=(vectors/np.sqrt(values))@vectors.T
        occupations,orbitals=eigh(root@density@root)
        coefficients=invroot@orbitals[:,::-1]
        mo_occ=np.zeros(mol.nao_nr());mo_occ[:mol.nelec[0]]=2
        gradient=orbital_gradient_rms(scf.hf.get_grad(coefficients,mo_occ,independent_fock),
                                      mol.nelec[0],mol.nao_nr()-mol.nelec[0])
        result['scf']={'candidate_energy':float(energy),'reference_energy':float(reference_energy),
            'reference_converged':bool(reference.converged),'energy_error':abs(float(energy-reference_energy)),
            'independent_recomputed_energy_error':abs(float(energy-recomputed)),
            'independent_gradient_rms':float(gradient),
            'electron_error':abs(float(np.einsum('ij,ji',density,overlap))-mol.nelectron),
            'candidate_details':solver.last_scf_result}
        m=result['scf']
        if (not m['reference_converged'] or m['energy_error']>1e-6 or
            m['independent_recomputed_energy_error']>1e-6 or
            m['independent_gradient_rms']>=1e-8 or m['electron_error']>1e-6):
            raise RuntimeError('Independent final SCF acceptance failed')
    result['accepted']=True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--carbons',type=int,default=21)
    parser.add_argument('--basis',default='cc-pvtz')
    parser.add_argument('--backend',choices=['serial','cpu','gpu'],default='cpu')
    parser.add_argument('--engine',choices=['pyscf_reference','hierarchical_rijcosx'],
                        default='pyscf_reference')
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--memory-mb',type=float,default=65536)
    parser.add_argument('--k-memory-mb',type=float,default=None,
                        help='fixed SGX block budget; default sizes blocks from memory-mb minus RSS')
    parser.add_argument('--memory-ceiling-mb',type=float,default=None)
    parser.add_argument('--dense-backend',choices=['cpu','cupy'],default='cpu')
    parser.add_argument('--grid-switch-energy-tol',type=float,default=None)
    parser.add_argument('--scf',action='store_true')
    parser.add_argument('--benchmark',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result={'accepted':False,'thresholds':{'matrix':1e-8,'jk_energy':1e-7,'final_energy':1e-6},
            'source_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ['hf.py','hierarchical_ri.py','local_cosx.py','validate_rijcosx_repair.py']}}
    started=time.perf_counter()
    code=0
    try:
        run(args,result)
    except Exception as exc:
        result['error']=type(exc).__name__+': '+str(exc)
        code=1
        print(result['error'],flush=True)
    finally:
        result['elapsed_seconds']=time.perf_counter()-started
        args.output.write_text(json.dumps(result,indent=2,default=lambda v:v.item()))
    raise SystemExit(code)


if __name__=='__main__':
    main()
