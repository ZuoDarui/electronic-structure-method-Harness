import pytest

from esm_harness.cli import main
from esm_harness.config import REPO_ROOT


def test_baseline_matches_pyscf_reference():
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "check-baseline",
                "--suite",
                str(REPO_ROOT / "benchmarks" / "hf_rhf.yaml"),
                "--baseline",
                str(REPO_ROOT / "baselines" / "hf.py"),
                "--references",
                str(REPO_ROOT / "benchmarks" / "reference_energies.json"),
                "--tol",
                "1e-8",
            ]
        )
    assert caught.value.code == 0
