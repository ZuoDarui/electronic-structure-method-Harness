# RHF single-shot accuracy gate

Status: **accepted**  
Task: `rhf-small-accuracy-v1`  
Candidate: supplied_candidate; model `None`, reasoning `None`  
Scope: Accuracy only: one program, CPU/NumPy, small molecules, PySCF references. No speed, GPU or large-system assessment.

Cases passed: 30 / 30

| case | nao | E candidate (Eh) | E PySCF (Eh) | abs dE | max dD | max dJ/K | FDS-SDF | cycles | result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| h2-sto3g-exact | 2 | -1.1167593074 | -1.1167593074 | 0.00e+00 | 6.66e-16 | 1.11e-16 | 2.22e-16 | 1 | passed |
| h2-sto3g-df | 2 | -1.1167936337 | -1.1167936337 | 0.00e+00 | 2.22e-16 | 2.78e-17 | 1.11e-16 | 1 | passed |
| h2-stretched-631g-exact | 4 | -0.9974972943 | -0.9974972943 | 2.00e-15 | 5.60e-13 | 2.22e-16 | 5.52e-13 | 10 | passed |
| h2-stretched-631g-df | 4 | -0.9975855761 | -0.9975855761 | 1.78e-15 | 3.72e-15 | 2.78e-17 | 4.44e-16 | 10 | passed |
| heh-cation-631g-exact | 4 | -2.9098416314 | -2.9098416314 | 0.00e+00 | 1.26e-13 | 2.22e-16 | 1.13e-14 | 6 | passed |
| heh-cation-631g-df | 4 | -2.9098864626 | -2.9098864626 | 1.78e-15 | 2.91e-12 | 2.22e-16 | 1.09e-14 | 6 | passed |
| he-ccpvdz-exact | 5 | -2.8551604772 | -2.8551604772 | 0.00e+00 | 4.00e-11 | 2.22e-16 | 4.73e-11 | 10 | passed |
| he-ccpvdz-df | 5 | -2.8551621890 | -2.8551621890 | 4.44e-16 | 5.65e-12 | 6.94e-17 | 5.60e-12 | 10 | passed |
| be-631g-exact | 9 | -14.5667640335 | -14.5667640335 | 3.55e-15 | 4.17e-11 | 1.60e-16 | 4.04e-12 | 9 | passed |
| be-631g-df | 9 | -14.5667340869 | -14.5667340869 | 0.00e+00 | 9.64e-14 | 1.11e-16 | 4.44e-15 | 10 | passed |
| lih-631g-exact | 11 | -7.9793215650 | -7.9793215650 | 3.55e-15 | 1.54e-09 | 4.44e-16 | 2.24e-10 | 9 | passed |
| lih-631g-df | 11 | -7.9793201450 | -7.9793201450 | 7.11e-15 | 1.57e-09 | 2.22e-16 | 2.26e-10 | 9 | passed |
| hf-631g-exact | 11 | -99.9834089333 | -99.9834089333 | 8.53e-14 | 1.15e-10 | 6.66e-16 | 7.31e-11 | 12 | passed |
| hf-631g-df | 11 | -99.9786560600 | -99.9786560600 | 1.42e-14 | 1.17e-10 | 6.66e-16 | 7.80e-11 | 12 | passed |
| water-631gs-exact | 18 | -76.0091080324 | -76.0091080324 | 9.95e-14 | 3.10e-10 | 1.55e-15 | 1.76e-10 | 13 | passed |
| water-631gs-df | 18 | -76.0083061218 | -76.0083061218 | 7.11e-14 | 3.01e-10 | 6.66e-16 | 1.83e-10 | 13 | passed |
| water-ccpvdz-exact | 24 | -76.0267656731 | -76.0267656731 | 1.14e-13 | 1.30e-10 | 1.11e-15 | 6.58e-11 | 14 | passed |
| water-ccpvdz-df | 24 | -76.0259362997 | -76.0259362997 | 2.84e-14 | 1.29e-10 | 1.11e-15 | 6.76e-11 | 14 | passed |
| ammonia-sto3g-exact | 8 | -55.4540856696 | -55.4540856696 | 4.26e-14 | 7.59e-10 | 3.33e-16 | 2.49e-10 | 8 | passed |
| ammonia-sto3g-df | 8 | -55.4524391342 | -55.4524391342 | 1.42e-14 | 8.00e-10 | 3.33e-16 | 2.49e-10 | 8 | passed |
| methane-631g-exact | 17 | -40.1804877661 | -40.1804877661 | 4.26e-14 | 1.31e-10 | 1.11e-15 | 1.11e-10 | 10 | passed |
| methane-631g-df | 17 | -40.1803495068 | -40.1803495068 | 3.55e-14 | 1.33e-10 | 8.88e-16 | 1.12e-10 | 10 | passed |
| n2-ccpvdz-exact | 28 | -108.9541280137 | -108.9541280137 | 1.56e-12 | 2.00e-10 | 1.26e-15 | 1.50e-10 | 11 | passed |
| n2-ccpvdz-df | 28 | -108.9508991106 | -108.9508991106 | 1.42e-12 | 1.90e-10 | 4.44e-16 | 1.47e-10 | 11 | passed |
| co-631g-exact | 18 | -112.6672045589 | -112.6672045589 | 0.00e+00 | 3.39e-10 | 6.66e-16 | 1.30e-10 | 14 | passed |
| co-631g-df | 18 | -112.6662414054 | -112.6662414054 | 5.68e-14 | 3.23e-10 | 6.66e-16 | 1.26e-10 | 14 | passed |
| ethylene-sto3g-exact | 14 | -77.0720877977 | -77.0720877977 | 1.14e-13 | 2.24e-10 | 5.55e-16 | 8.29e-11 | 9 | passed |
| ethylene-sto3g-df | 14 | -77.0720364419 | -77.0720364419 | 5.68e-14 | 2.38e-10 | 2.22e-16 | 8.27e-11 | 9 | passed |
| formaldehyde-631g-exact | 22 | -113.8072564136 | -113.8072564136 | 1.71e-13 | 9.41e-10 | 2.66e-15 | 7.14e-10 | 15 | passed |
| formaldehyde-631g-df | 22 | -113.8063484055 | -113.8063484055 | 5.68e-14 | 9.32e-10 | 8.88e-16 | 7.05e-10 | 15 | passed |

Worst-case metrics vs tolerance:

- `energy_max_abs`: 1.563e-12 (tol 1e-08) ok
- `jk_max_abs`: 2.665e-15 (tol 1e-09) ok
- `jk_symmetry_max_abs`: 1.110e-15 (tol 1e-11) ok
- `jk_linearity_max_abs`: 1.776e-15 (tol 1e-09) ok
- `density_max_abs`: 1.571e-09 (tol 1e-06) ok
- `density_symmetry_max_abs`: 0.000e+00 (tol 1e-11) ok
- `electron_count_abs`: 8.882e-15 (tol 1e-09) ok
- `idempotency_max_abs`: 6.217e-15 (tol 1e-08) ok
- `residual_max_abs`: 7.144e-10 (tol 1e-07) ok
- `orthogonality_max_abs`: 8.882e-15 (tol 1e-10) ok
- `orbital_equation_max_abs`: 1.672e-09 (tol 1e-07) ok
- `density_reconstruction_max_abs`: 0.000e+00 (tol 1e-10) ok
- `energy_reconstruction_abs`: 2.842e-14 (tol 1e-09) ok
- `occupation_max_abs`: 0.000e+00 (tol 1e-12) ok
- `orbital_order_error`: 0.000e+00 (tol 1e-12) ok
- `anchor_energy_abs`: 6.661e-16 (tol 1e-09) ok

Accepted source: `accepted/solution.py`.

Correctness is limited to the listed small molecules, exact/DF modes and CPU float64. References are PySCF RHF/DF-RHF from the same integrals; no speed claim is made.
