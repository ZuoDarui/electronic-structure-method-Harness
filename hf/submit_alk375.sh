#!/usr/bin/env bash
set -euo pipefail
echo "alk375 $(date -Is)"
source /home/tione/notebook/home/zuodarui/miniforge3/bin/activate pyscf
exec bash /home/tione/notebook/home/zuodarui/fieldquantum/hf/run_alk375.sh
