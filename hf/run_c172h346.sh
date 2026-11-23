#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /home/tione/notebook/home/zuodarui/miniforge3/bin/activate pyscf
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p outputs
stdout_log="outputs/c172h346_ccpvtz.stdout.log"
if [[ -e "$stdout_log" ]]; then
  n=2
  base="${stdout_log%.stdout.log}"
  while [[ -e "${base}_${n}.stdout.log" ]]; do
    n=$((n + 1))
  done
  stdout_log="${base}_${n}.stdout.log"
fi
echo "stdout_log: $stdout_log"
python hf.py --config c172h346_ccpvtz.json 2>&1 | tee "$stdout_log"
