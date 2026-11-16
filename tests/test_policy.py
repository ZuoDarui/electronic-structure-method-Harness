from esm_harness.eval.policy import rejection_reason


LEGAL = "def solve(mol):\n    return {'e_tot': -1.0, 'converged': True}\n"


def test_rejects_scf_df_and_kernel_calls():
    assert rejection_reason(LEGAL, ["numpy", "pyscf.gto"]) is None
    assert "pyscf.scf" in rejection_reason("import pyscf.scf\n" + LEGAL, ["pyscf.gto"])
    assert rejection_reason("from pyscf import gto, scf\n" + LEGAL, ["pyscf.gto"])
    assert rejection_reason("import pyscf.df\n" + LEGAL, ["pyscf.gto"])
    assert "kernel" in rejection_reason(LEGAL + "\nmf.kernel()\n", ["pyscf.gto"])
    assert rejection_reason("import pyscf.scf\n" + LEGAL, ["pyscf.scf", "pyscf.gto"]) is None
