# RHF single-shot accuracy gate

Status: **generation_failed**  
Task: `rhf-small-accuracy-v1`  
Candidate: generated; model `gpt-6-astra`, reasoning `max`  
Scope: Accuracy only: one program, CPU/NumPy, small molecules, PySCF references. No speed, GPU or large-system assessment.


Reasons:

- PatchBundle fields must be files, rationale and expected_effect

Correctness is limited to the listed small molecules, exact/DF modes and CPU float64. References are PySCF RHF/DF-RHF from the same integrals; no speed claim is made.
