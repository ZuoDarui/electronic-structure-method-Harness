#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /home/tione/notebook/home/zuodarui/miniforge3/bin/activate pyscf
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
mkdir -p results
python -u validate_rijcosx_repair.py \
  --carbons 21 \
  --basis cc-pvtz \
  --backend gpu \
  --threads 8 \
  --memory-mb 65536 \
  --scf \
  --output results/c21_gate.json
python -u - <<'PY'
import json
from pathlib import Path
report = json.loads(Path("results/c21_gate.json").read_text())
scf = report["scf"]
energy = float(scf["candidate_energy"])
coulomb = float(report["fixed_density"][0]["coulomb_energy"])
print("C21_CANDIDATE", energy, flush=True)
print("C21_REFERENCE", scf["reference_energy"], flush=True)
print("C21_ERROR", scf["energy_error"], flush=True)
print("C21_MINAO_COULOMB", coulomb, flush=True)
print("C21_ACCEPTED", report["accepted"], flush=True)
if not report["accepted"] or energy >= 0 or scf["energy_error"] > 1e-6:
    raise SystemExit("C21 energy gate failed")
if not (500 < coulomb < 5000):
    raise SystemExit("C21 MINAO coulomb energy is outside the expected range")
PY
