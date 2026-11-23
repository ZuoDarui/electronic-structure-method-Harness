"""Independent same-grid SGX checks for local projected-density COSX."""
import unittest
import numpy as np
from pyscf import gto,scf,lib
from pyscf.dft import gen_grid,radi
from pyscf.sgx import sgx,sgx_jk
from local_cosx import LocalCOSX


def reference(mol,density,grid=(15,50),fit=True,symmetric=True):
    engine=sgx.SGX(mol,auxbasis='weigend')
    engine.verbose=0
    engine.fit_ovlp=fit
    engine._symm_ovlp_fit=symmetric
    engine.build()
    grids=gen_grid.Grids(mol)
    grids.atom_grid={mol.atom_pure_symbol(i):grid for i in range(mol.natm)}
    grids.radi_method=radi.treutler_ahlrichs
    grids.prune=gen_grid.nwchem_prune
    grids.build(with_non0tab=True)
    grids.non0tab=grids.make_mask(mol,grids.coords)
    grids.screen_index=grids.non0tab
    engine.grids=grids
    return sgx_jk.get_k_only(engine,density,hermi=1,direct_scf_tol=1e-13)


class LocalCOSXTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lib.num_threads(1)
        cls.water=gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                        basis='cc-pvdz',verbose=0)

    def test_symmetric_projection_matches_independent_sgx(self):
        mol=self.water; dm=scf.hf.init_guess_by_minao(mol)
        solver=LocalCOSX(mol,15,50,batch_size=128)
        actual=solver.build(dm); expected=reference(mol,dm)
        np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=0)
        self.assertLess(abs(.25*np.einsum('ij,ji',dm,actual-expected)),1e-9)
        self.assertLess(solver.stats['projection_residual_relative'],1e-12)
        self.assertLess(solver.stats['output_projection_residual_relative'],1e-12)

    def test_unsymmetric_projection_option_matches_reference(self):
        mol=self.water;dm=scf.hf.init_guess_by_minao(mol)
        actual=LocalCOSX(mol,15,50,batch_size=128,sym_ovlp=False).build(dm)
        np.testing.assert_allclose(actual,reference(mol,dm,symmetric=False),atol=1e-9,rtol=0)

    def test_random_density_and_zero_blocks(self):
        mol=self.water;rng=np.random.default_rng(20261113)
        dm=rng.normal(size=(mol.nao,mol.nao));dm=(dm+dm.T)*.025
        dm[0:5,10:15]=0;dm[10:15,0:5]=0
        actual=LocalCOSX(mol,15,50,batch_size=128).build(dm)
        np.testing.assert_allclose(actual,reference(mol,dm),atol=1e-9,rtol=0)

    def test_incremental_density_uses_full_support(self):
        mol=self.water;dm=scf.hf.init_guess_by_minao(mol)
        solver=LocalCOSX(mol,15,50,batch_size=128)
        initial=solver.build(dm)
        delta=.07*dm
        updated=solver.build(dm+delta)
        incremental=initial+solver.build(delta,screen_dm=dm+delta)
        np.testing.assert_allclose(incremental,updated,atol=1e-10,rtol=0)

    def test_separated_molecules_match_entire_reference(self):
        mol=gto.M(atom='H 0 0 0; H 0 0 1; H 0 0 35; H 0 0 36',
                  basis='sto-3g',verbose=0)
        dm=scf.hf.init_guess_by_minao(mol)
        actual=LocalCOSX(mol,15,50,batch_size=128).build(dm)
        np.testing.assert_allclose(actual,reference(mol,dm),atol=1e-9,rtol=0)

    def test_no_overlap_fit_and_input_failures(self):
        mol=self.water;dm=scf.hf.init_guess_by_minao(mol)
        solver=LocalCOSX(mol,15,50,batch_size=128,overlap_fit=False)
        np.testing.assert_allclose(solver.build(dm),reference(mol,dm,fit=False),atol=1e-9,rtol=0)
        with self.assertRaises(ValueError):solver.build(np.full_like(dm,np.nan))
        with self.assertRaises(ValueError):solver.build(dm,screen_dm=np.eye(2))
        with self.assertRaises(ValueError):solver.build(dm,ao_tol=0)

    def test_memory_and_indefinite_projection_fail_explicitly(self):
        with self.assertRaises(MemoryError):
            LocalCOSX(self.water,15,50,max_memory_mb=.001)
        mol=gto.M(atom='H 0 0 0; H 0 0 1',basis='sto-3g',verbose=0)
        solver=LocalCOSX(mol,15,50);solver.prepare()
        solver.s_num=-solver.s_num
        with self.assertRaisesRegex(RuntimeError,'not positive definite'):
            solver.build(scf.hf.init_guess_by_minao(mol))

    def test_gaussian_product_potential_bound(self):
        mol=gto.M(atom='H 0 0 0; H 0 0 5',basis='cc-pvdz',verbose=0)
        solver=LocalCOSX(mol,15,50)
        points=np.array([[0.,0.,0.],[0.,0.,4.7],[1.,0.,4.7],[0.,0.,9.45]])
        integral=mol.intor('int1e_grids',grids=points)
        for i in range(mol.nbas):
            for j in range(mol.nbas):
                bound=solver._pair_potential_bound(i,j)
                block=integral[:,solver.ao_loc[i]:solver.ao_loc[i+1],
                                 solver.ao_loc[j]:solver.ao_loc[j+1]]
                self.assertLessEqual(np.max(np.abs(block)),bound*(1+1e-12)+1e-30)

    def test_fused_integral_primitive_matches_explicit_tensor(self):
        mol=self.water;solver=LocalCOSX(mol,15,50)
        shells=np.array([0,2,4,6,8])
        columns=np.concatenate([np.arange(solver.ao_loc[i],solver.ao_loc[i+1]) for i in shells])
        rng=np.random.default_rng(784)
        points=rng.normal(size=(13,3));f=rng.normal(size=(13,len(columns)))
        tensor=mol.intor('int1e_grids',grids=points)[:,columns][:,:,columns]
        expected=np.einsum('gij,gj->gi',tensor,f)
        actual=solver._fused_integral_contraction(shells,points,f,
            rng.uniform(-.3,1.,size=13),np.ones((len(shells),len(shells))))
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)

if __name__=='__main__':unittest.main()
