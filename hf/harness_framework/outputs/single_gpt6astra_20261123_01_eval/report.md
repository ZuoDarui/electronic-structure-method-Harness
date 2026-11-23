# RHF single-shot accuracy gate

Status: **accepted**  
Task: `rhf-small-accuracy-v1`  
Candidate: supplied_candidate; model `None`, reasoning `None`  
Scope: Accuracy only: one program, CPU/NumPy, small molecules, PySCF references. No speed, GPU or large-system assessment.

Cases passed: 30 / 30

| case | nao | E candidate (Eh) | E PySCF (Eh) | abs dE | max dD | max dJ/K | FDS-SDF | cycles | result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| h2-sto3g-exact | 2 | -1.1167593074 | -1.1167593074 | 4.44e-16 | 4.44e-16 | 1.11e-16 | 2.22e-16 | 2 | passed |
| h2-sto3g-df | 2 | -1.1167936337 | -1.1167936337 | 4.44e-16 | 2.22e-16 | 5.55e-17 | 3.33e-16 | 2 | passed |
| h2-stretched-631g-exact | 4 | -0.9974972943 | -0.9974972943 | 1.55e-15 | 1.33e-10 | 1.11e-16 | 1.32e-10 | 5 | passed |
| h2-stretched-631g-df | 4 | -0.9975855761 | -0.9975855761 | 1.33e-15 | 1.44e-10 | 1.11e-16 | 1.42e-10 | 5 | passed |
| heh-cation-631g-exact | 4 | -2.9098416314 | -2.9098416314 | 1.78e-15 | 7.21e-14 | 2.22e-16 | 5.48e-14 | 7 | passed |
| heh-cation-631g-df | 4 | -2.9098864626 | -2.9098864626 | 8.88e-16 | 2.84e-12 | 1.67e-16 | 5.31e-14 | 7 | passed |
| he-ccpvdz-exact | 5 | -2.8551604772 | -2.8551604772 | 8.88e-16 | 6.12e-11 | 1.94e-16 | 7.30e-11 | 6 | passed |
| he-ccpvdz-df | 5 | -2.8551621890 | -2.8551621890 | 1.33e-15 | 6.12e-11 | 1.11e-16 | 7.29e-11 | 6 | passed |
| be-631g-exact | 9 | -14.5667640335 | -14.5667640335 | 1.78e-15 | 2.65e-10 | 1.25e-16 | 2.51e-11 | 7 | passed |
| be-631g-df | 9 | -14.5667340869 | -14.5667340869 | 1.24e-14 | 2.57e-10 | 2.22e-16 | 2.43e-11 | 7 | passed |
| lih-631g-exact | 11 | -7.9793215650 | -7.9793215650 | 8.88e-15 | 6.65e-10 | 2.22e-16 | 3.51e-11 | 11 | passed |
| lih-631g-df | 11 | -7.9793201450 | -7.9793201450 | 3.55e-15 | 6.88e-10 | 8.88e-16 | 3.57e-11 | 11 | passed |
| hf-631g-exact | 11 | -99.9834089333 | -99.9834089333 | 4.26e-14 | 2.80e-10 | 6.66e-16 | 2.23e-10 | 13 | passed |
| hf-631g-df | 11 | -99.9786560600 | -99.9786560600 | 7.11e-14 | 2.95e-10 | 1.33e-15 | 2.29e-10 | 13 | passed |
| water-631gs-exact | 18 | -76.0091080324 | -76.0091080324 | 4.26e-14 | 2.18e-10 | 1.55e-15 | 1.03e-11 | 15 | passed |
| water-631gs-df | 18 | -76.0083061218 | -76.0083061218 | 4.26e-14 | 1.69e-10 | 9.99e-16 | 1.27e-11 | 15 | passed |
| water-ccpvdz-exact | 24 | -76.0267656731 | -76.0267656731 | 5.68e-14 | 1.54e-10 | 1.33e-15 | 1.82e-10 | 15 | passed |
| water-ccpvdz-df | 24 | -76.0259362997 | -76.0259362997 | 2.84e-14 | 1.56e-10 | 1.55e-15 | 1.84e-10 | 15 | passed |
| ammonia-sto3g-exact | 8 | -55.4540856696 | -55.4540856696 | 4.26e-14 | 3.88e-10 | 3.33e-16 | 2.46e-11 | 10 | passed |
| ammonia-sto3g-df | 8 | -55.4524391342 | -55.4524391342 | 1.42e-14 | 3.88e-10 | 5.55e-16 | 2.47e-11 | 10 | passed |
| methane-631g-exact | 17 | -40.1804877661 | -40.1804877661 | 3.55e-14 | 2.67e-11 | 1.22e-15 | 1.80e-12 | 12 | passed |
| methane-631g-df | 17 | -40.1803495068 | -40.1803495068 | 4.26e-14 | 2.63e-11 | 1.33e-15 | 1.76e-12 | 12 | passed |
| n2-ccpvdz-exact | 28 | -108.9541280137 | -108.9541280137 | 1.51e-12 | 2.45e-10 | 9.99e-16 | 4.91e-11 | 13 | passed |
| n2-ccpvdz-df | 28 | -108.9508991106 | -108.9508991106 | 1.39e-12 | 2.40e-10 | 1.55e-15 | 4.91e-11 | 13 | passed |
| co-631g-exact | 18 | -112.6672045589 | -112.6672045589 | 2.84e-14 | 4.59e-10 | 8.88e-16 | 3.39e-11 | 16 | passed |
| co-631g-df | 18 | -112.6662414054 | -112.6662414054 | 2.84e-14 | 4.44e-10 | 6.11e-16 | 3.68e-11 | 16 | passed |
| ethylene-sto3g-exact | 14 | -77.0720877977 | -77.0720877977 | 8.53e-14 | 1.43e-10 | 1.11e-15 | 5.21e-11 | 11 | passed |
| ethylene-sto3g-df | 14 | -77.0720364419 | -77.0720364419 | 8.53e-14 | 1.57e-10 | 8.88e-16 | 5.20e-11 | 11 | passed |
| formaldehyde-631g-exact | 22 | -113.8072564136 | -113.8072564136 | 2.84e-13 | 3.77e-10 | 2.22e-15 | 8.65e-11 | 17 | passed |
| formaldehyde-631g-df | 22 | -113.8063484055 | -113.8063484055 | 1.42e-13 | 3.78e-10 | 1.78e-15 | 8.66e-11 | 17 | passed |

Worst-case metrics vs tolerance:

- `energy_max_abs`: 1.506e-12 (tol 1e-08) ok
- `jk_max_abs`: 2.220e-15 (tol 1e-09) ok
- `jk_symmetry_max_abs`: 0.000e+00 (tol 1e-11) ok
- `jk_linearity_max_abs`: 3.109e-15 (tol 1e-09) ok
- `density_max_abs`: 6.879e-10 (tol 1e-06) ok
- `density_symmetry_max_abs`: 0.000e+00 (tol 1e-11) ok
- `electron_count_abs`: 1.421e-14 (tol 1e-09) ok
- `idempotency_max_abs`: 7.994e-15 (tol 1e-08) ok
- `residual_max_abs`: 2.289e-10 (tol 1e-07) ok
- `orthogonality_max_abs`: 8.171e-14 (tol 1e-10) ok
- `orbital_equation_max_abs`: 1.542e-10 (tol 1e-07) ok
- `density_reconstruction_max_abs`: 0.000e+00 (tol 1e-10) ok
- `energy_reconstruction_abs`: 2.842e-14 (tol 1e-09) ok
- `occupation_max_abs`: 0.000e+00 (tol 1e-12) ok
- `orbital_order_error`: 0.000e+00 (tol 1e-12) ok
- `anchor_energy_abs`: 2.220e-16 (tol 1e-09) ok

Accepted source: `accepted/solution.py`.

Correctness is limited to the listed small molecules, exact/DF modes and CPU float64. References are PySCF RHF/DF-RHF from the same integrals; no speed claim is made.
