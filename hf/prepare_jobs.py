"""Read alkane XYZ files and write hf.py inputs plus cluster run/submit scripts."""

import json
import re
from pathlib import Path


MINIFORGE = "/home/tione/notebook/home/zuodarui/miniforge3"
PROJECT = "/home/tione/notebook/home/zuodarui/fieldquantum/hf"

MOLECULES = (
    {"xyz": "c86h174.xyz", "benchmark_one_iteration": True, "max_iteration": 1},
    {"xyz": "c172h346.xyz", "benchmark_one_iteration": True, "max_iteration": 1},
    {"xyz": "alk375.xyz", "benchmark_one_iteration": False, "max_iteration": 200},
)

RESOURCES = {
    "cpu_cores": 48,
    "threads": 8,
    "parallel_backend": "cpu",
    "parallel_workers": 4,
    "parallel_chunk_size": 16,
    "parallel_min_shells": 12,
}

SOLVER_OPTIONS = {
    "jk_mode": "rijcosx",
    "auxbasis": "weigend",
    "tol": 1e-8,
    "grid_rad": 30,
    "grid_ang": 110,
    "final_grid_rad": 35,
    "final_grid_ang": 302,
    "batch_size": 512,
}


def expected_basis_functions(xyz_path):
    lines = Path(xyz_path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"{xyz_path} has no XYZ comment line")
    match = re.search(r"nao[_-]?ccpvtz\s*=\s*(\d+)", lines[1], re.IGNORECASE)
    if match is None:
        raise ValueError(f"{xyz_path} comment has no nao_ccpvtz count")
    return int(match.group(1))


def config_for(molecule, basis_functions):
    stem = Path(molecule["xyz"]).stem
    config = {
        "xyz": molecule["xyz"],
        "basis": "cc-pvtz",
        "unit": "Angstrom",
        "charge": 0,
        "spin": 0,
        "expected_basis_functions": basis_functions,
        "resources": RESOURCES,
        "output_dir": "outputs",
        "output_prefix": f"{stem}_ccpvtz",
        "solver_options": {**SOLVER_OPTIONS, "max_iteration": molecule["max_iteration"]},
    }
    if molecule["benchmark_one_iteration"]:
        config["benchmark_one_iteration"] = True
    return config


def run_script(stem):
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${{BASH_SOURCE[0]}}")"
source {MINIFORGE}/bin/activate pyscf
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""
mkdir -p outputs
stdout_log="outputs/{stem}_ccpvtz.stdout.log"
if [[ -e "$stdout_log" ]]; then
  n=2
  base="${{stdout_log%.stdout.log}}"
  while [[ -e "${{base}}_${{n}}.stdout.log" ]]; do
    n=$((n + 1))
  done
  stdout_log="${{base}}_${{n}}.stdout.log"
fi
echo "stdout_log: $stdout_log"
python hf.py --config {stem}_ccpvtz.json 2>&1 | tee "$stdout_log"
"""


def submit_script(stem):
    return f"""#!/usr/bin/env bash
set -euo pipefail
echo "{stem} $(date -Is)"
source {MINIFORGE}/bin/activate pyscf
exec bash {PROJECT}/run_{stem}.sh
"""


def write_text(path, text):
    path.write_text(text)
    path.chmod(0o755) if path.suffix == ".sh" else None


def main():
    root = Path(__file__).resolve().parent
    written = []
    for molecule in MOLECULES:
        xyz_path = root / molecule["xyz"]
        stem = xyz_path.stem
        basis_functions = expected_basis_functions(xyz_path)
        config_path = root / f"{stem}_ccpvtz.json"
        config_path.write_text(json.dumps(config_for(molecule, basis_functions), indent=2) + "\n")
        run_path = root / f"run_{stem}.sh"
        submit_path = root / f"submit_{stem}.sh"
        write_text(run_path, run_script(stem))
        write_text(submit_path, submit_script(stem))
        written.extend((config_path.name, run_path.name, submit_path.name))
    print("wrote " + " ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
