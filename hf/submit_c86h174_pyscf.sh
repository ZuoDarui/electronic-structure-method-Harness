#!/usr/bin/env bash
set -euo pipefail
echo "c86h174 pyscf $(date -Is)"
source /home/tione/notebook/home/zuodarui/miniforge3/bin/activate pyscf
exec bash /home/tione/notebook/home/zuodarui/fieldquantum/hf/run_c86h174_pyscf.sh
