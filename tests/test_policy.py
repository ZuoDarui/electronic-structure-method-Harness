from esm_harness.eval.policy import rejection_reason


LEGAL = "def solve(mol):\n    return {'e_tot': -1.0, 'converged': True}\n"


def test_rejects_scf_df_and_kernel_calls():
    assert rejection_reason(LEGAL, ["numpy", "pyscf.gto"]) is None
    assert "pyscf.scf" in rejection_reason("import pyscf.scf\n" + LEGAL, ["pyscf.gto"])
    assert rejection_reason("from pyscf import gto, scf\n" + LEGAL, ["pyscf.gto"])
    assert rejection_reason("import pyscf.df\n" + LEGAL, ["pyscf.gto"])
    assert "kernel" in rejection_reason(LEGAL + "\nmf.kernel()\n", ["pyscf.gto"])
    assert rejection_reason("import pyscf.scf\n" + LEGAL, ["pyscf.scf", "pyscf.gto"]) is None


ALLOWED = ["numpy", "scipy", "pyscf.gto"]


def test_rejects_imports_outside_the_allowed_set():
    assert rejection_reason("import numpy as np\nimport scipy.linalg\nimport math\n" + LEGAL, ALLOWED) is None
    assert rejection_reason("from pyscf import gto\n" + LEGAL, ALLOWED) is None
    assert "os" in rejection_reason("import os\n" + LEGAL, ALLOWED)
    assert rejection_reason("from pyscf import lib\n" + LEGAL, ALLOWED)
    assert rejection_reason("import pickle\n" + LEGAL, ALLOWED)


def test_rejects_state_that_survives_between_calls():
    assert rejection_reason(LEGAL + "\nopen('cache', 'w')\n", ALLOWED)
    assert rejection_reason("import numpy as np\n" + LEGAL + "\nnp.save('x', 1)\n", ALLOWED)
    assert rejection_reason(LEGAL + "\neval('1')\n", ALLOWED)
    assert rejection_reason("def solve(mol:\n", ALLOWED).startswith("syntax error")
