#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /home/tione/notebook/home/zuodarui/miniforge3/bin/activate pyscf
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p results
python -u - <<'PY'
import hashlib,json,platform,sys
from pathlib import Path
import numpy,scipy,pyscf
from hf import import_cupy
cp=import_cupy()
props=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
result={'python':sys.version,'platform':platform.platform(),'numpy':numpy.__version__,
 'scipy':scipy.__version__,'pyscf':pyscf.__version__,'cupy':cp.__version__,
 'gpu_name':props['name'].decode() if isinstance(props['name'],bytes) else props['name'],
 'device_count':cp.cuda.runtime.getDeviceCount(),
 'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('.').glob('*.py')}}
Path('results/environment.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
PY
python -m unittest test_hierarchical_ri test_local_cosx test_direct_hf test_rijcosx_repair -v > results/unit_tests.log 2>&1
python -u validate_rijcosx_repair.py --carbons 2 --basis sto-3g --backend gpu --threads 8 --scf --output results/c2_gpu_smoke.json > results/c2_gpu_smoke.log 2>&1
python -u validate_rijcosx_repair.py --carbons 12 --backend gpu --threads 8 --output results/c12_fixed.json > results/c12_fixed.log 2>&1
for carbon_count in 21 43 86 172; do
  options=()
  if [[ "$carbon_count" != 86 ]]; then options+=(--scf); fi
  python -u validate_rijcosx_repair.py --carbons "$carbon_count" --backend gpu --threads 8 --benchmark "${options[@]}" --output "results/c${carbon_count}_acceptance.json" > "results/c${carbon_count}_acceptance.log" 2>&1
done
python -u - <<'PY'
import json,math
from pathlib import Path
reports=[json.loads(Path(f'results/c{n}_acceptance.json').read_text()) for n in (21,43,86,172)]
assert all(r['accepted'] for r in reports)
rows=[]
for left,right in zip(reports,reports[1:]):
 row={'carbons':[left['carbons'],right['carbons']],'nao':[left['nao'],right['nao']]}
 for kernel in ('j_seconds','k_seconds'):
  row[kernel+'_exponent']=math.log(right['kernel_timing_median'][kernel]/left['kernel_timing_median'][kernel])/math.log(right['nao']/left['nao'])
 rows.append(row)
accepted=all(r[k+'_exponent']<=1.5 for r in rows[-2:] for k in ('j_seconds','k_seconds'))
result={'accepted':accepted,'threshold':1.5,'pairs':rows,
        'scope':'J/K kernels on this neutral alkane sequence, same GPU/resources/precision; not entire SCF scaling'}
Path('results/scaling_acceptance.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
if not accepted:raise SystemExit('Near-linear empirical scaling gate failed')
PY
