# RHF single-shot accuracy gate

Status: **accepted**  
Task: `rhf-small-accuracy-v1`  
Candidate: generated; model `gpt-6-astra`, reasoning `max`  
Scope: Accuracy only: one program, CPU/NumPy, small molecules, PySCF references. No speed, GPU or large-system assessment.

Cases passed: 30 / 30

| case | nao | E candidate (Eh) | E PySCF (Eh) | abs dE | max dD | max dJ/K | FDS-SDF | cycles | result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| h2-sto3g-exact | 2 | -1.1167593074 | -1.1167593074 | 4.44e-16 | 2.22e-16 | 1.11e-16 | 1.11e-16 | 1 | passed |
| h2-sto3g-df | 2 | -1.1167936337 | -1.1167936337 | 2.22e-16 | 2.22e-16 | 2.78e-17 | 3.33e-16 | 1 | passed |
| h2-stretched-631g-exact | 4 | -0.9974972943 | -0.9974972943 | 1.55e-15 | 3.09e-12 | 2.22e-16 | 3.05e-12 | 10 | passed |
| h2-stretched-631g-df | 4 | -0.9975855761 | -0.9975855761 | 1.55e-15 | 3.35e-12 | 8.33e-17 | 3.31e-12 | 10 | passed |
| heh-cation-631g-exact | 4 | -2.9098416314 | -2.9098416314 | 8.88e-16 | 2.45e-13 | 2.22e-16 | 2.83e-13 | 9 | passed |
| heh-cation-631g-df | 4 | -2.9098864626 | -2.9098864626 | 0.00e+00 | 2.96e-12 | 2.22e-16 | 3.31e-13 | 9 | passed |
| he-ccpvdz-exact | 5 | -2.8551604772 | -2.8551604772 | 8.88e-16 | 9.10e-11 | 2.22e-16 | 1.12e-10 | 10 | passed |
| he-ccpvdz-df | 5 | -2.8551621890 | -2.8551621890 | 8.88e-16 | 9.10e-11 | 1.11e-16 | 1.12e-10 | 10 | passed |
| be-631g-exact | 9 | -14.5667640335 | -14.5667640335 | 0.00e+00 | 4.48e-11 | 1.28e-16 | 4.63e-12 | 10 | passed |
| be-631g-df | 9 | -14.5667340869 | -14.5667340869 | 8.88e-15 | 4.78e-11 | 1.11e-16 | 4.91e-12 | 10 | passed |
| lih-631g-exact | 11 | -7.9793215650 | -7.9793215650 | 5.33e-15 | 7.42e-10 | 4.44e-16 | 7.48e-11 | 11 | passed |
| lih-631g-df | 11 | -7.9793201450 | -7.9793201450 | 5.33e-15 | 7.64e-10 | 4.44e-16 | 7.60e-11 | 11 | passed |
| hf-631g-exact | 11 | -99.9834089333 | -99.9834089333 | 8.53e-14 | 3.76e-10 | 6.66e-16 | 2.14e-10 | 13 | passed |
| hf-631g-df | 11 | -99.9786560600 | -99.9786560600 | 7.11e-14 | 3.96e-10 | 6.66e-16 | 2.25e-10 | 13 | passed |
| water-631gs-exact | 18 | -76.0091080324 | -76.0091080324 | 2.84e-14 | 2.18e-10 | 1.55e-15 | 1.03e-11 | 14 | passed |
| water-631gs-df | 18 | -76.0083061218 | -76.0083061218 | 1.14e-13 | 1.69e-10 | 6.66e-16 | 1.27e-11 | 14 | passed |
| water-ccpvdz-exact | 24 | -76.0267656731 | -76.0267656731 | 1.14e-13 | 1.54e-10 | 1.11e-15 | 1.82e-10 | 14 | passed |
| water-ccpvdz-df | 24 | -76.0259362997 | -76.0259362997 | 8.53e-14 | 1.56e-10 | 8.88e-16 | 1.84e-10 | 14 | passed |
| ammonia-sto3g-exact | 8 | -55.4540856696 | -55.4540856696 | 5.68e-14 | 3.12e-10 | 3.33e-16 | 6.56e-11 | 13 | passed |
| ammonia-sto3g-df | 8 | -55.4524391342 | -55.4524391342 | 1.42e-14 | 3.47e-10 | 4.44e-16 | 6.34e-11 | 13 | passed |
| methane-631g-exact | 17 | -40.1804877661 | -40.1804877661 | 2.84e-14 | 2.14e-10 | 1.11e-15 | 8.82e-11 | 11 | passed |
| methane-631g-df | 17 | -40.1803495068 | -40.1803495068 | 2.13e-14 | 2.21e-10 | 1.11e-15 | 9.09e-11 | 11 | passed |
| n2-ccpvdz-exact | 28 | -108.9541280137 | -108.9541280137 | 1.51e-12 | 4.73e-10 | 9.58e-16 | 6.93e-11 | 13 | passed |
| n2-ccpvdz-df | 28 | -108.9508991106 | -108.9508991106 | 1.39e-12 | 4.71e-10 | 5.55e-16 | 6.97e-11 | 13 | passed |
| co-631g-exact | 18 | -112.6672045589 | -112.6672045589 | 8.53e-14 | 1.43e-10 | 6.66e-16 | 9.72e-11 | 16 | passed |
| co-631g-df | 18 | -112.6662414054 | -112.6662414054 | 8.53e-14 | 1.48e-10 | 6.66e-16 | 9.62e-11 | 16 | passed |
| ethylene-sto3g-exact | 14 | -77.0720877977 | -77.0720877977 | 1.14e-13 | 1.35e-10 | 5.55e-16 | 5.04e-11 | 12 | passed |
| ethylene-sto3g-df | 14 | -77.0720364419 | -77.0720364419 | 5.68e-14 | 1.49e-10 | 8.88e-16 | 5.03e-11 | 12 | passed |
| formaldehyde-631g-exact | 22 | -113.8072564136 | -113.8072564136 | 2.84e-13 | 3.02e-10 | 2.66e-15 | 2.16e-11 | 17 | passed |
| formaldehyde-631g-df | 22 | -113.8063484055 | -113.8063484055 | 1.71e-13 | 3.03e-10 | 8.88e-16 | 2.14e-11 | 17 | passed |

Worst-case metrics vs tolerance:

- `energy_max_abs`: 1.506e-12 (tol 1e-08) ok
- `jk_max_abs`: 2.665e-15 (tol 1e-09) ok
- `jk_symmetry_max_abs`: 0.000e+00 (tol 1e-11) ok
- `jk_linearity_max_abs`: 1.776e-15 (tol 1e-09) ok
- `density_max_abs`: 7.641e-10 (tol 1e-06) ok
- `density_symmetry_max_abs`: 0.000e+00 (tol 1e-11) ok
- `electron_count_abs`: 1.243e-14 (tol 1e-09) ok
- `idempotency_max_abs`: 1.688e-14 (tol 1e-08) ok
- `residual_max_abs`: 2.248e-10 (tol 1e-07) ok
- `orthogonality_max_abs`: 7.705e-14 (tol 1e-10) ok
- `orbital_equation_max_abs`: 1.180e-10 (tol 1e-07) ok
- `density_reconstruction_max_abs`: 0.000e+00 (tol 1e-10) ok
- `energy_reconstruction_abs`: 2.842e-14 (tol 1e-09) ok
- `occupation_max_abs`: 0.000e+00 (tol 1e-12) ok
- `orbital_order_error`: 0.000e+00 (tol 1e-12) ok
- `anchor_energy_abs`: 2.220e-16 (tol 1e-09) ok

Accepted source: `accepted/solution.py`.

Correctness is limited to the listed small molecules, exact/DF modes and CPU float64. References are PySCF RHF/DF-RHF from the same integrals; no speed claim is made.
