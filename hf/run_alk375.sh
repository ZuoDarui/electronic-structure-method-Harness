#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /home/tione/notebook/home/zuodarui/miniforge3/bin/activate pyscf
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""
mkdir -p outputs
python hf.py --config alk375_ccpvtz.json 2>&1 | tee "outputs/alk375_ccpvtz.stdout.log"
