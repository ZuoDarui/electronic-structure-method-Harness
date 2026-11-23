#!/usr/bin/env bash
set -euo pipefail
echo "c21h44 $(date -Is)"
source /home/tione/notebook/home/zuodarui/miniforge3/bin/activate pyscf
exec bash /home/tione/notebook/home/zuodarui/fieldquantum/hf/run_c21h44.sh
