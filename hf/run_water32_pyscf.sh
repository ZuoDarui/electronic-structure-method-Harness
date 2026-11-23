#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /home/tione/notebook/home/zuodarui/miniforge3/bin/activate pyscf
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""
mkdir -p outputs
python pyscf_water32.py --config water32_ccpvtz.json 2>&1 | tee "outputs/water32_ccpvtz_pyscf.stdout.log"
