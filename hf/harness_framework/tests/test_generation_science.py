"""Meaningful positive and adversarial tests for trusted scientific validation.

Oracle-derived fixtures test the verifier itself; they are not generated
candidates and must never be counted as a successful generation campaign.
"""
from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import numpy as np

from harness_engineering.generation.science import (
    load_candidate_output, load_problem, make_challenge_cases, prepare_case,
    verify_case, verify_nonconverged_case, verify_rigid_motion, verify_suite,
)
from harness_engineering.generation.spec import (
    DEFAULT_TOLERANCES, expand_cases, load_spec, spec_digest, validate_spec,
)


ROOT = Path(__file__).resolve().parents[1]


class BoundedCandidateOutputTests(unittest.TestCase):
    """These archive attacks must fail before NumPy allocates any payload."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "output.npz"
        self.arrays = {key: np.zeros((2, 2), dtype=np.float64) for key in (
            "density", "mo_coeff", "j0", "k0", "j1", "k1", "jsum", "ksum")}
        self.arrays.update(energy=np.asarray(0.0), converged=np.asarray(True),
                           mo_energy=np.zeros(2), occupations=np.zeros(2),
                           history_json=np.asarray('[{"cycle":1,"energy":0,"residual_norm":0}]'))

    def npy(self, value):
        stream = io.BytesIO(); np.save(stream, value, allow_pickle=False)
        return stream.getvalue()

    def archive(self, replacements=None, duplicate=False):
        with zipfile.ZipFile(self.path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for key, value in self.arrays.items():
                archive.writestr(key + ".npy", (replacements or {}).get(key, self.npy(value)))
            if duplicate:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    archive.writestr("energy.npy", self.npy(np.asarray(0.0)))

    def assert_rejected_before_numpy_load(self):
        with patch("harness_engineering.generation.science.np.load", side_effect=AssertionError("must not allocate")):
            with self.assertRaises(ValueError):
                load_candidate_output(self.path, 2)

    def test_valid_raw_output_decodes_with_bounded_loader(self):
        self.archive()
        output = load_candidate_output(self.path, 2)
        self.assertEqual(set(output), set(self.arrays))
        np.testing.assert_array_equal(output["density"], self.arrays["density"])

    def test_compressed_zero_zipbomb_is_rejected_before_decompression(self):
        # A small compressed file represents megabytes of attacker payload.
        self.archive({"history_json": b"\x00" * 2_000_000})
        self.assertLess(self.path.stat().st_size, 20000)
        self.assert_rejected_before_numpy_load()

    def test_huge_declared_shape_without_payload_is_rejected_before_allocation(self):
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(stream, {
            "descr": "<f8", "fortran_order": False, "shape": (10**9, 10**9)})
        self.archive({"density": stream.getvalue()})
        self.assert_rejected_before_numpy_load()

    def test_huge_unicode_scalar_and_header_length_are_rejected(self):
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(stream, {
            "descr": "<U1000000000", "fortran_order": False, "shape": ()})
        self.archive({"history_json": stream.getvalue()})
        self.assert_rejected_before_numpy_load()
        self.archive({"history_json": b"\x93NUMPY\x02\x00" + (10**8).to_bytes(4, "little")})
        self.assert_rejected_before_numpy_load()

    def test_duplicate_members_and_object_payload_are_rejected(self):
        self.archive(duplicate=True)
        self.assert_rejected_before_numpy_load()
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(stream, {"descr": "|O", "fortran_order": False, "shape": ()})
        self.archive({"converged": stream.getvalue()})
        self.assert_rejected_before_numpy_load()


class GenerationSpecTests(unittest.TestCase):
    def test_development_and_holdout_have_explicit_complete_mode_coverage(self):
        dev = load_spec(ROOT / "harness/generation/rhf.json")
        holdout = load_spec(ROOT / "harness/generation/holdout_rhf_v1.json")
        self.assertEqual(len(expand_cases(dev)), 12)
        self.assertEqual(len(expand_cases(holdout)), 8)
        self.assertTrue(holdout["private_evaluation"])
        self.assertTrue(dev["execution"]["require_external_trace"])
        self.assertTrue(set(c["id"] for c in expand_cases(dev)).isdisjoint(c["id"] for c in expand_cases(holdout)))
        self.assertEqual(spec_digest(dev), spec_digest(copy.deepcopy(dev)))

    def test_invalid_spec_is_rejected_before_generation(self):
        base = load_spec(ROOT / "harness/generation/rhf.json")
        variants = []
        wrong = copy.deepcopy(base); wrong["cases"] = []; variants.append(wrong)
        wrong = copy.deepcopy(base); wrong["cases"].append(wrong["cases"][0]); variants.append(wrong)
        wrong = copy.deepcopy(base); wrong["tolerances"]["energy_max_abs"] = float("nan"); variants.append(wrong)
        wrong = copy.deepcopy(base); del wrong["tolerances"]["residual_max_abs"]; variants.append(wrong)
        wrong = copy.deepcopy(base); wrong["method"] = "UHF"; variants.append(wrong)
        wrong = copy.deepcopy(base); wrong["cases"][0]["spin"] = 1; variants.append(wrong)
        wrong = copy.deepcopy(base); wrong["seed"] = True; variants.append(wrong)
        wrong = copy.deepcopy(base); wrong["solver"]["conv_tol"] = float("inf"); variants.append(wrong)
        wrong = copy.deepcopy(base); wrong["solver"]["max_cycle"] = False; variants.append(wrong)
        wrong = copy.deepcopy(base); wrong["generation"]["max_rounds"] = 9; variants.append(wrong)
        wrong = copy.deepcopy(base); wrong["generation"]["max_rounds"] = 0; variants.append(wrong)
        wrong = copy.deepcopy(base); wrong["generation"]["candidate_file"] = "../escape.py"; variants.append(wrong)
        wrong = copy.deepcopy(base); wrong["generation"]["provider"] = "other"; variants.append(wrong)
        for wrong in variants:
            with self.subTest(wrong=wrong):
                with self.assertRaises(ValueError):
                    validate_spec(wrong)


class TrustedScienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.prepared = []
        for mode in ("exact", "df"):
            cls.prepared.append(prepare_case(
                {"id": f"verifier-water-{mode}",
                 "atom": "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", "basis": "6-31g", "auxbasis": "weigend"},
                mode, cls.directory / "public", cls.directory / "private", seed=37,
            ))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def fixture(self, prepared):
        with np.load(prepared["reference_path"], allow_pickle=False) as archive:
            output = {name: np.array(archive[name]) for name in (
                "energy", "density", "mo_coeff", "mo_energy", "occupations",
                "j0", "k0", "j1", "k1", "jsum", "ksum", "converged",
            )}
        output["history_json"] = np.asarray(json.dumps([{
            "cycle": 1, "energy": float(output["energy"]), "residual_norm": 1e-12,
        }]))
        return output

    def verify(self, prepared, output):
        output_path = self.directory / "result.npz"
        np.savez_compressed(output_path, **output)
        return verify_case(prepared, output_path, DEFAULT_TOLERANCES,
                           {"status": "passed", "returncode": 0,
                            "input_sha256": prepared["input_sha256"]})

    def test_independent_real_reference_fixtures_pass_exact_and_df(self):
        records = []
        for prepared in self.prepared:
            with self.subTest(mode=prepared["identity"]["mode"]):
                result = self.verify(prepared, self.fixture(prepared))
                self.assertEqual(result["status"], "passed", result["reasons"])
                self.assertLess(result["metrics"]["energy_reconstruction_abs"], 1e-10)
                self.assertNotEqual(result["details"]["reference_provenance"]["process_id"], os.getpid())
                loaded = result["details"]["reference_provenance"]["actual_loaded"]
                for name in ("numpy", "numpy._core._multiarray_umath", "scipy",
                             "scipy.linalg._flapack", "pyscf", "pyscf.gto.moleintor.libcgto", "python_executable"):
                    artifact = loaded[name]
                    self.assertTrue(Path(artifact["path"]).is_absolute())
                    self.assertTrue(Path(artifact["path"]).is_file())
                    self.assertRegex(artifact["sha256"], r"^[a-f0-9]{64}$")
                    self.assertGreater(artifact["bytes"], 0)
                records.append(result)
        self.assertTrue(verify_suite([p["id"] for p in self.prepared], records)["passed"])

    def test_public_problem_contains_integrals_without_reference_answers(self):
        for prepared in self.prepared:
            problem = load_problem(prepared["problem_path"])
            self.assertEqual(problem["metadata"]["schema_version"], 1)
            self.assertEqual(problem["metadata"]["identity_sha256"], prepared["identity"]["scientific_input_sha256"])
            self.assertNotIn("density", problem)
            self.assertNotIn("energy", problem)
            self.assertEqual(problem["trial_densities"].shape[0], 2)
            self.assertTrue("eri" in problem if prepared["identity"]["mode"] == "exact" else "df_factors" in problem)

    def test_incorrect_exchange_is_detected_even_when_final_energy_is_correct(self):
        for prepared in self.prepared:
            output = self.fixture(prepared)
            output["k0"] *= 1.001
            result = self.verify(prepared, output)
            self.assertEqual(result["status"], "failed")
            self.assertGreater(result["metrics"]["jk_max_abs"], DEFAULT_TOLERANCES["jk_max_abs"])
            self.assertEqual(result["metrics"]["energy_max_abs"], 0.0)

    def test_correct_tensors_cannot_hide_wrong_energy_or_false_convergence(self):
        prepared = self.prepared[0]
        wrong_energy = self.fixture(prepared)
        wrong_energy["energy"] += 0.01
        result = self.verify(prepared, wrong_energy)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("energy_max_abs" in reason for reason in result["reasons"]))
        output = self.fixture(prepared); output["converged"] = np.asarray(False)
        result = self.verify(prepared, output)
        self.assertEqual(result["status"], "failed")
        self.assertIn("candidate did not converge", result["reasons"])

    def test_wrong_density_and_occupations_are_independently_checked(self):
        prepared = self.prepared[0]
        output = self.fixture(prepared); output["density"] *= 0.9
        result = self.verify(prepared, output)
        self.assertEqual(result["status"], "failed")
        self.assertGreater(result["metrics"]["electron_count_abs"], 0.1)
        output = self.fixture(prepared); output["occupations"][:] = 0.0
        result = self.verify(prepared, output)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["metrics"]["occupation_max_abs"], 2.0)

    def test_nonfinite_wrong_dtype_and_malformed_output_are_invalid(self):
        prepared = self.prepared[0]
        outputs = []
        output = self.fixture(prepared); output["density"][0, 0] = np.nan; outputs.append(output)
        output = self.fixture(prepared); output["converged"] = np.asarray("false"); outputs.append(output)
        output = self.fixture(prepared); output["density"] = output["density"].astype(np.float32); outputs.append(output)
        output = self.fixture(prepared); del output["j0"]; outputs.append(output)
        output = self.fixture(prepared); output["metrics"] = np.asarray("forged"); outputs.append(output)
        output = self.fixture(prepared); output["history_json"] = np.asarray('[{"cycle":1,"energy":NaN,"residual_norm":0}]'); outputs.append(output)
        output = self.fixture(prepared); output["history_json"] = np.asarray('[]'); outputs.append(output)
        for output in outputs:
            with self.subTest(keys=output.keys()):
                self.assertEqual(self.verify(prepared, output)["status"], "invalid_runtime")

    def test_wrong_execution_identity_or_failed_runtime_is_invalid(self):
        prepared = self.prepared[0]
        for execution in ({"status": "passed", "returncode": 1, "input_sha256": prepared["input_sha256"]},
                          {"status": "completed", "returncode": 0, "input_sha256": prepared["input_sha256"]},
                          {"status": "passed", "returncode": 0, "input_sha256": "0" * 64}):
            result = verify_case(prepared, self.directory / "result.npz", DEFAULT_TOLERANCES, execution)
            self.assertEqual(result["status"], "invalid_runtime")

    def test_input_and_reference_integrity_are_checked(self):
        prepared = self.prepared[0]
        for field, hash_field in (("problem_path", "input_sha256"), ("reference_path", "reference_sha256")):
            tampered = dict(prepared)
            path = self.directory / f"tampered-{field}.npz"
            path.write_bytes(Path(prepared[field]).read_bytes() + b"changed")
            tampered[field] = str(path)
            # Preserve the original manifest digest; a changed file must fail.
            result = verify_case(tampered, self.directory / "result.npz", DEFAULT_TOLERANCES,
                                 {"status": "passed", "returncode": 0,
                                  "input_sha256": prepared["input_sha256"]})
            self.assertEqual(result["status"], "invalid_runtime")
            self.assertTrue(any("hash changed" in reason for reason in result["reasons"]))

    def test_suite_rejects_empty_missing_duplicate_unknown_and_failed_cases(self):
        expected = [p["id"] for p in self.prepared]
        records = [self.verify(p, self.fixture(p)) for p in self.prepared]
        unknown = copy.deepcopy(records); unknown[0]["case_id"] = "unknown"
        failed = copy.deepcopy(records); failed[0]["status"] = "failed"
        invalid = copy.deepcopy(records); invalid[0]["metrics"]["energy_max_abs"] = "false"
        for ids, values in (([], []), (expected, []), (expected, records[:1]),
                            (expected, records + records[:1]), (expected, unknown),
                            (expected, failed), (expected, invalid)):
            self.assertFalse(verify_suite(ids, values)["passed"])

    def test_memory_preflight_rejects_before_large_integral_allocation(self):
        with self.assertRaisesRegex(RuntimeError, "MemoryError"):
            prepare_case({"id": "too-large-budget", "atom": "O 0 0 0; H 0 0 1; H 0 1 0", "basis": "6-31g"},
                         "exact", self.directory / "public", self.directory / "private", memory_limit_mb=0.001)

    def test_rigid_challenges_preserve_real_reference_energies_in_both_modes(self):
        spec = load_spec(ROOT / "harness/generation/rhf.json")
        challenges = make_challenge_cases(spec)
        self.assertEqual(len(challenges), 2)
        self.assertEqual({case["mode"] for case in challenges}, {"exact", "df"})
        for case in challenges:
            moved = prepare_case(case, case["mode"], self.directory / "public", self.directory / "private", seed=73)
            base = next(p for p in self.prepared if p["identity"]["mode"] == case["mode"])
            base_record = self.verify(base, self.fixture(base))
            moved_record = self.verify(moved, self.fixture(moved))
            property_result = verify_rigid_motion(base_record, moved_record)
            self.assertTrue(property_result["passed"], property_result["reasons"])
            self.assertLess(property_result["metrics"]["reference_energy_invariance_abs"], 1e-10)
            corrupted = copy.deepcopy(moved_record)
            corrupted["details"]["candidate_energy"] += 0.01
            self.assertFalse(verify_rigid_motion(base_record, corrupted)["passed"])

    def test_physical_identity_ignores_case_id_trial_seed_and_coordinate_units(self):
        base = self.prepared[0]
        problem = load_problem(base["problem_path"])
        atoms = [[row[0], row[1:]] for row in problem["metadata"]["geometry_bohr"]]
        duplicate = prepare_case({"id": "same-physics-different-name", "atom": list(reversed(atoms)),
                                  "basis": "6-31G", "unit": "Bohr"}, "exact",
                                 self.directory / "public", self.directory / "private", seed=1041)
        self.assertNotEqual(base["input_sha256"], duplicate["input_sha256"])
        self.assertNotEqual(base["identity"]["scientific_input_sha256"], duplicate["identity"]["scientific_input_sha256"])
        self.assertEqual(base["identity"]["physical_input_sha256"], duplicate["identity"]["physical_input_sha256"])

    def test_nonconvergence_contract_preserves_physics_and_requires_false_status(self):
        prepared = self.prepared[0]
        output = self.fixture(prepared)
        path = self.directory / "nonconverged.npz"
        execution = {"status": "passed", "returncode": 0, "input_sha256": prepared["input_sha256"]}
        np.savez_compressed(path, **output)
        record = verify_nonconverged_case(prepared, path, DEFAULT_TOLERANCES, execution)
        self.assertEqual(record["status"], "failed")
        self.assertIn("max-cycle challenge requires converged=false", record["reasons"])
        output["converged"] = np.asarray(False)
        np.savez_compressed(path, **output)
        record = verify_nonconverged_case(prepared, path, DEFAULT_TOLERANCES, execution)
        self.assertEqual(record["status"], "passed", record["reasons"])
        self.assertEqual(record["kind"], "nonconvergence_contract")
        self.assertEqual(verify_case(prepared, path, DEFAULT_TOLERANCES, execution)["status"], "failed")
        output["density"] *= 0.9
        np.savez_compressed(path, **output)
        self.assertEqual(verify_nonconverged_case(prepared, path, DEFAULT_TOLERANCES, execution)["status"], "failed")

    def test_real_one_cycle_density_is_valid_exhaustion_but_not_a_converged_solution(self):
        # This is a trusted verifier test on one actual numerical iteration.
        # It supplies no baseline implementation to the generated candidate.
        prepared = self.prepared[0]
        problem = load_problem(prepared["problem_path"])
        s, h, eri = (problem[key] for key in ("overlap", "hcore", "eri"))
        eigenvalues, eigenvectors = np.linalg.eigh(s)
        x = eigenvectors / np.sqrt(eigenvalues)
        def orbitals(fock):
            values, vectors = np.linalg.eigh(x.T @ fock @ x)
            return values, x @ vectors
        def fock(density):
            j = np.einsum("pqrs,rs->pq", eri, density)
            k = np.einsum("prqs,rs->pq", eri, density)
            return h + j - 0.5 * k
        nocc = int(problem["nelectron"]) // 2
        _, initial_c = orbitals(h)
        initial_d = 2 * initial_c[:, :nocc] @ initial_c[:, :nocc].T
        eps, coeff = orbitals(fock(initial_d))
        dm = 2 * coeff[:, :nocc] @ coeff[:, :nocc].T
        physical_fock = fock(dm)
        energy = 0.5 * np.einsum("pq,qp", dm, h + physical_fock) + problem["enuc"]
        output = self.fixture(prepared)
        output.update(energy=np.asarray(energy), density=dm, mo_coeff=coeff, mo_energy=eps,
                      converged=np.asarray(False), history_json=np.asarray(json.dumps([{
                          "cycle": 1, "energy": float(energy),
                          "residual_norm": float(np.linalg.norm(physical_fock @ dm @ s - s @ dm @ physical_fock)),
                      }])))
        path = self.directory / "actual-one-cycle.npz"
        np.savez_compressed(path, **output)
        execution = {"status": "passed", "returncode": 0, "input_sha256": prepared["input_sha256"]}
        contract = verify_nonconverged_case(prepared, path, DEFAULT_TOLERANCES, execution)
        self.assertEqual(contract["status"], "passed", contract["reasons"])
        self.assertGreater(contract["metrics"]["energy_max_abs"], DEFAULT_TOLERANCES["energy_max_abs"])
        self.assertEqual(verify_case(prepared, path, DEFAULT_TOLERANCES, execution)["status"], "failed")
        # At an unfinished iterate, canonicalize occupied and virtual blocks
        # separately. Their Rayleigh spectra need not obey global Aufbau order.
        # Interchange one occupied/virtual orbital to construct a valid
        # nonstationary determinant with an actual block-spectrum crossing.
        crossing_c = coeff.copy()
        crossing_c[:, [0, nocc]] = crossing_c[:, [nocc, 0]]
        crossing_d = 2 * crossing_c[:, :nocc] @ crossing_c[:, :nocc].T
        crossing_f = fock(crossing_d)
        crossing_e = np.empty(len(eps))
        for block in (slice(0, nocc), slice(nocc, len(eps))):
            values, rotation = np.linalg.eigh(crossing_c[:, block].T @ crossing_f @ crossing_c[:, block])
            crossing_c[:, block] = crossing_c[:, block] @ rotation
            crossing_e[block] = values
        crossing_energy = float(0.5 * np.einsum("pq,qp", crossing_d, h + crossing_f) + problem["enuc"])
        self.assertGreater(crossing_e[nocc - 1], crossing_e[nocc])
        crossing_output = dict(output)
        crossing_output.update(density=crossing_d, mo_coeff=crossing_c, mo_energy=crossing_e,
                               energy=np.asarray(crossing_energy), history_json=np.asarray(json.dumps([{
                                   "cycle": 1, "energy": crossing_energy,
                                   "residual_norm": float(np.linalg.norm(crossing_f @ crossing_d @ s - s @ crossing_d @ crossing_f)),
                               }])))
        np.savez_compressed(path, **crossing_output)
        crossing_contract = verify_nonconverged_case(prepared, path, DEFAULT_TOLERANCES, execution)
        self.assertEqual(crossing_contract["status"], "passed", crossing_contract["reasons"])
        self.assertGreater(crossing_contract["metrics"]["orbital_order_error"], DEFAULT_TOLERANCES["orbital_order_error"])
        ordinary = verify_case(prepared, path, DEFAULT_TOLERANCES, execution)
        self.assertEqual(ordinary["status"], "failed")
        self.assertTrue(any("orbital_order_error" in reason for reason in ordinary["reasons"]))
        output["history_json"] = np.asarray(json.dumps([{"cycle": 2, "energy": float(energy), "residual_norm": 1.0}]))
        np.savez_compressed(path, **output)
        self.assertEqual(verify_nonconverged_case(prepared, path, DEFAULT_TOLERANCES, execution)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
