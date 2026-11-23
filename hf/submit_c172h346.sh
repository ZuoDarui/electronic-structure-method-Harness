#!/usr/bin/env bash
set -euo pipefail
echo "c172h346 $(date -Is)"
source /home/tione/notebook/home/zuodarui/miniforge3/bin/activate pyscf
exec bash /home/tione/notebook/home/zuodarui/fieldquantum/hf/run_c172h346.sh
