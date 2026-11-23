"""Network-free tests of bounded generation and atomic source application."""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import traceback
import unittest
from unittest.mock import patch
import urllib.error

from harness_engineering.generation.generator import (
    GenerationError, ResponsesGenerator, apply_bundle, initial_source, validate_bundle,
)


SECRET = "unit-test-private-api-credential"


def bundle(source=None):
    return {"files": [{"path": "solution.py", "content": source or initial_source()}],
            "rationale": "Implement the public numerical contract.",
            "expected_effect": "Development validation can run the implementation."}


def response(value=None, **overrides):
    result = {"id": "resp-mock", "status": "completed", "usage": {"input_tokens": 37, "output_tokens": 83},
              "output": [{"type": "message", "status": "completed", "content": [
                  {"type": "output_text", "text": json.dumps(value or bundle())}]}]}
    result.update(overrides)
    return result


class GenerationGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.environment = patch.dict(os.environ, {"OPENAI_API_KEY": SECRET}, clear=True)
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.directory.cleanup()

    def generator(self, transport, **kwargs):
        return ResponsesGenerator({"model": "test-explicit-model", "transport": transport,
                                   "sleep": lambda seconds: None, **kwargs})

    def test_flat_output_schema_is_converted_and_still_guarded(self):
        from harness_engineering.generation.generator import FLAT_SCHEMA
        requests = []
        flat = {"solution_py": initial_source(), "rationale": "r", "expected_effect": "e"}
        def transport(request, **kwargs):
            requests.append(request)
            return response(flat)
        result = self.generator(transport, output_schema="flat").generate({"round": 1}, self.root / "flat")
        self.assertEqual(result["files"][0], {"path": "solution.py", "content": initial_source()})
        self.assertEqual(requests[0]["text"]["format"]["schema"], FLAT_SCHEMA)
        self.assertIn("solution_py", requests[0]["input"][0]["content"])
        # A relay that ignores the schema (observed: "patches") is rejected, not aliased.
        drifted = {"patches": [{"path": "solution.py", "content": initial_source()}], "rationale": "r", "expected_effect": "e"}
        with self.assertRaises(GenerationError) as caught:
            self.generator(lambda request, **kwargs: response(drifted), output_schema="flat").generate(
                {"round": 1}, self.root / "drift")
        self.assertEqual(caught.exception.status, "invalid_bundle")
        guarded = {"solution_py": "import os\n" + initial_source(), "rationale": "r", "expected_effect": "e"}
        with self.assertRaises(GenerationError) as caught:
            self.generator(lambda request, **kwargs: response(guarded), output_schema="flat").generate(
                {"round": 1}, self.root / "guard")
        self.assertEqual(caught.exception.status, "forbidden_import")
        with self.assertRaises(GenerationError):
            self.generator(transport, output_schema="xml")

    def test_success_records_strict_request_response_usage_and_public_context(self):
        requests = []

        def transport(request, **kwargs):
            self.assertEqual(kwargs["api_key"], SECRET)
            self.assertEqual(kwargs["url"], "https://api.openai.com/v1/responses")
            requests.append(request)
            return response()

        generated = self.generator(transport).generate({
            "current_source": initial_source(), "public_prompt": {"goal": "RHF"},
            "dev_feedback": {"passed": False}, "holdout": "DO NOT SEND HIDDEN INPUT"}, self.root)
        self.assertEqual(generated, bundle())
        self.assertFalse(requests[0]["store"])
        self.assertTrue(requests[0]["text"]["format"]["strict"])
        self.assertEqual(requests[0]["max_output_tokens"], 16384)
        self.assertNotIn("DO NOT SEND HIDDEN INPUT", json.dumps(requests))
        self.assertTrue((self.root / "bundle.json").exists())
        status = json.loads((self.root / "attempt-01/status.json").read_text())
        self.assertEqual(status["usage"], {"input_tokens": 37, "output_tokens": 83})
        self.assertEqual(status["requested_max_output_tokens"], 16384)
        self.assertEqual(status["reported_output_tokens"], 83)
        self.assertTrue(status["budget_within_limit"])
        for path in self.root.rglob("*"):
            if path.is_file():
                self.assertNotIn(SECRET, path.read_text())

    def test_actual_token_cap_is_in_system_prompt_without_changing_model_or_effort(self):
        requests = []
        def transport(request, **kwargs):
            requests.append(request)
            return response(usage={"input_tokens": 37, "output_tokens": 83})
        self.generator(transport, model="gpt-6.1-sol", reasoning_effort="max",
                       max_output_tokens=1000).generate({}, self.root)
        request = requests[0]
        prompt = request["input"][0]["content"]
        self.assertIn("1000 tokens", prompt)
        self.assertIn("including all reasoning tokens", prompt)
        self.assertEqual(request["model"], "gpt-6.1-sol")
        self.assertEqual(request["reasoning"]["effort"], "max")
        self.assertEqual(request["max_output_tokens"], 1000)

    def test_reported_total_output_budget_includes_reasoning_and_is_not_retried(self):
        calls = []
        accounting = {"input_tokens": 1497, "output_tokens": 23663, "total_tokens": 25160,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 20179}}
        raw = response(usage=accounting)

        def transport(*args, **kwargs):
            calls.append(1)
            return raw

        with self.assertRaises(GenerationError) as error:
            self.generator(transport).generate({}, self.root)
        self.assertEqual(error.exception.status, "output_budget_exceeded")
        self.assertEqual(error.exception.usage, accounting)
        self.assertEqual(len(calls), 1)
        self.assertFalse((self.root / "bundle.json").exists())
        self.assertFalse((self.root / "solution.py").exists())
        archived = json.loads((self.root / "attempt-01/response.json").read_text())
        self.assertEqual(archived, raw)
        status = json.loads((self.root / "attempt-01/status.json").read_text())
        self.assertEqual(status["requested_max_output_tokens"], 16384)
        self.assertEqual(status["reported_output_tokens"], 23663)
        self.assertFalse(status["budget_within_limit"])
        self.assertFalse(status["retry"])

    def test_invalid_usage_types_and_nonfinite_or_negative_counts_are_rejected(self):
        invalid = [None, [], {}, {"input_tokens": 10},
                   *({"output_tokens": value} for value in
                     (True, False, -1, 1.0, "83", None, float("nan"), float("inf"), [])),
                   {"output_tokens": 83, "input_tokens": -1},
                   {"output_tokens": 83, "output_tokens_details": {"reasoning_tokens": "11"}},
                   {"output_tokens": 83, "output_tokens_details": {"reasoning_tokens": float("nan")}}]
        for index, usage in enumerate(invalid):
            out = self.root / f"invalid-usage-{index}"
            calls = []

            def transport(*args, **kwargs):
                calls.append(1)
                return response(usage=usage)

            with self.subTest(usage=usage), self.assertRaises(GenerationError) as error:
                self.generator(transport).generate({}, out)
            self.assertEqual(error.exception.status, "invalid_usage")
            self.assertEqual(len(calls), 1)
            self.assertFalse((out / "bundle.json").exists())
            self.assertTrue((out / "attempt-01/response.json").exists())
            self.assertEqual(json.loads((out / "attempt-01/status.json").read_text())["status"], "invalid_usage")

    def test_exact_output_cap_boundary_with_nested_integer_accounting_is_accepted(self):
        accounting = {"input_tokens": 37, "output_tokens": 16384, "total_tokens": 16421,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 16000}}
        raw = response(usage=accounting)
        self.assertEqual(self.generator(lambda *a, **kw: raw).generate({}, self.root), bundle())
        self.assertTrue(json.loads((self.root / "attempt-01/status.json").read_text())["budget_within_limit"])

    def test_captured_attribution_content_lists_preserve_usage_and_enforce_total_budget(self):
        # Reduced structural fixture from the observed provider extension. The
        # visible message's 3484 tokens do not replace total output accounting.
        accounting = {"input_tokens": 6084, "output_tokens": 23663, "total_tokens": 29747,
                      "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 20179},
                      "attribution": {"items": {
                          "msg_visible": {"input_tokens": 0, "output_tokens": 3484,
                                          "content": [{"input_tokens": 0, "output_tokens": 3484,
                                                       "cached_tokens": 0, "cache_write_tokens": 0}]}},
                          "request_fields": {"text.format": {"input_tokens": 69, "output_tokens": 0}}}}
        raw = response(usage=accounting)
        with self.assertRaises(GenerationError) as error:
            self.generator(lambda *a, **kw: raw).generate({}, self.root / "over-budget")
        self.assertEqual(error.exception.status, "output_budget_exceeded")
        self.assertEqual(error.exception.usage, accounting)
        archived = json.loads((self.root / "over-budget/attempt-01/response.json").read_text())
        self.assertEqual(archived["usage"], accounting)
        self.assertFalse((self.root / "over-budget/bundle.json").exists())

        accounting["output_tokens"] = 3484
        accounting["total_tokens"] = 9568
        accounting["output_tokens_details"]["reasoning_tokens"] = 0
        self.assertEqual(self.generator(lambda *a, **kw: raw).generate({}, self.root / "valid-extension"), bundle())
        accounting["attribution"]["items"]["msg_visible"]["content"][0]["output_tokens"] = True
        with self.assertRaises(GenerationError) as invalid:
            self.generator(lambda *a, **kw: raw).generate({}, self.root / "invalid-extension")
        self.assertEqual(invalid.exception.status, "invalid_usage")

    def test_configuration_requires_explicit_model_and_api_key(self):
        with self.assertRaises(GenerationError) as error:
            ResponsesGenerator({})
        self.assertEqual(error.exception.status, "config_error")
        with patch.dict(os.environ, {"OPENAI_MODEL": "model-from-environment"}):
            self.assertEqual(ResponsesGenerator({}).model, "model-from-environment")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(GenerationError):
                ResponsesGenerator({"model": "chosen"})

    def test_configuration_caps_tokens_retries_timeout_and_rejects_credentials_in_url(self):
        for item in ({"max_output_tokens": 16385}, {"max_output_tokens": True},
                     {"max_retries": 3}, {"timeout_s": float("nan")},
                     {"base_url": "https://user:pass@api.example/v1"},
                     {"base_url": "http://api.example/v1"}):
            with self.subTest(item=item), self.assertRaises(GenerationError):
                self.generator(lambda *a, **kw: response(), **item)

    def test_extended_budget_is_explicit_and_standard_defaults_do_not_increase(self):
        for config in ({"max_output_tokens": 32768}, {"timeout_s": 1200},
                       {"budget_profile": "unknown"}, {"budget_profile": []},
                       {"budget_profile": "extended", "max_output_tokens": 32769},
                       {"budget_profile": "extended", "timeout_s": 1201}):
            with self.subTest(config=config), self.assertRaises(GenerationError):
                self.generator(lambda *a, **kw: response(), **config)
        standard = self.generator(lambda *a, **kw: response())
        self.assertEqual((standard.budget_profile, standard.max_output_tokens, standard.timeout_s),
                         ("standard", 16384, 120.0))
        explicit = self.generator(lambda *a, **kw: response(), budget_profile="extended")
        self.assertEqual((explicit.max_output_tokens, explicit.timeout_s), (16384, 120.0))

    def test_extended_budget_preserves_model_and_rejects_its_own_cap_excess(self):
        calls = []
        def transport(request, **kwargs):
            calls.append((request, kwargs))
            return response(usage={"input_tokens": 100, "output_tokens": 23663,
                                   "output_tokens_details": {"reasoning_tokens": 20179}})
        self.generator(transport, model="gpt-6.1-sol", reasoning_effort="max", streaming=True,
                       budget_profile="extended", max_output_tokens=32768, timeout_s=1200).generate({}, self.root)
        request, options = calls[0]
        self.assertEqual(request["max_output_tokens"], 32768)
        self.assertEqual(request["model"], "gpt-6.1-sol")
        self.assertEqual(request["reasoning"], {"effort": "max"})
        self.assertEqual(options["timeout_s"], 1200)
        self.assertIn("32768 tokens", request["input"][0]["content"])
        status = json.loads((self.root / "attempt-01/status.json").read_text())
        self.assertEqual(status["budget_profile"], "extended")
        self.assertEqual(status["requested_timeout_seconds"], 1200)
        self.assertTrue(status["budget_within_limit"])
        for count in (32768, 32769):
            directory = self.root / str(count)
            generator = self.generator(lambda *a, **kw: response(usage={"output_tokens": count}),
                                       budget_profile="extended", max_output_tokens=32768, timeout_s=1200)
            if count == 32768:
                generator.generate({}, directory)
            else:
                with self.assertRaises(GenerationError) as raised:
                    generator.generate({}, directory)
                self.assertEqual(raised.exception.status, "output_budget_exceeded")
                self.assertFalse((directory / "bundle.json").exists())

    def test_streaming_option_is_a_strict_boolean_and_default_request_is_unchanged(self):
        for value in (0, 1, "true", "false", None, []):
            with self.subTest(value=value), self.assertRaises(GenerationError):
                self.generator(lambda *a, **kw: response(), streaming=value)
        default = self.generator(lambda *a, **kw: response())._request({})
        disabled = self.generator(lambda *a, **kw: response(), streaming=False)._request({})
        self.assertNotIn("stream", default)
        self.assertEqual(default, disabled)

    def test_streaming_wire_request_and_completed_evidence_keep_model_effort_and_budget(self):
        from harness_engineering.generation.responses_streaming import parse_sse
        calls = []
        raw_response = response()
        raw_event = ("event: response.completed\ndata: " + json.dumps({
            "type": "response.completed", "sequence_number": 0,
            "response": raw_response}) + "\n\n").encode()

        def streaming_transport(request, **kwargs):
            calls.append(request)
            self.assertTrue(request["stream"])
            return parse_sse([raw_event], deadline=1e100)

        generator = ResponsesGenerator({"model": "gpt-6.1-sol", "reasoning_effort": "max",
                                        "streaming": True, "timeout_s": 600})
        with patch("harness_engineering.generation.responses_streaming.streaming_transport",
                   side_effect=streaming_transport):
            generated = generator.generate({}, self.root)
        self.assertEqual(generated, bundle())
        request = json.loads((self.root / "attempt-01/request.json").read_text())
        self.assertEqual(request, calls[0])
        self.assertTrue(request["stream"])
        self.assertEqual(request["model"], "gpt-6.1-sol")
        self.assertEqual(request["reasoning"], {"effort": "max"})
        self.assertEqual(request["max_output_tokens"], 16384)
        self.assertFalse(request["store"])
        self.assertTrue(request["text"]["format"]["strict"])
        archived = json.loads((self.root / "attempt-01/response.json").read_text())
        metadata = archived.pop("_transport_metadata")
        self.assertEqual(archived, raw_response)
        status = json.loads((self.root / "attempt-01/status.json").read_text())
        self.assertEqual(status["transport_metadata"], metadata)

    def test_streaming_completed_total_usage_over_cap_is_rejected_without_bundle(self):
        from harness_engineering.generation.responses_streaming import parse_sse
        raw = response(usage={"input_tokens": 6084, "output_tokens": 23663,
                              "output_tokens_details": {"reasoning_tokens": 20179}, "total_tokens": 29747})
        event = ("data: " + json.dumps({"type": "response.completed", "response": raw}) + "\n\n").encode()
        generator = ResponsesGenerator({"model": "gpt-6.1-sol", "reasoning_effort": "max", "streaming": True})
        with patch("harness_engineering.generation.responses_streaming.streaming_transport",
                   side_effect=lambda *a, **kw: parse_sse([event], deadline=1e100)) as mocked:
            with self.assertRaises(GenerationError) as error:
                generator.generate({}, self.root)
        self.assertEqual(error.exception.status, "output_budget_exceeded")
        self.assertEqual(mocked.call_count, 1)
        self.assertFalse((self.root / "bundle.json").exists())
        archived = json.loads((self.root / "attempt-01/response.json").read_text())
        self.assertEqual(archived["usage"]["output_tokens"], 23663)
        self.assertIn("_transport_metadata", archived)
        self.assertFalse(json.loads((self.root / "attempt-01/status.json").read_text())["budget_within_limit"])

    def test_streaming_eof_or_protocol_error_does_not_produce_bundle_and_keeps_safe_metadata(self):
        from harness_engineering.generation.responses_streaming import parse_sse
        raws = [b'data: {"type":"response.output_text.delta","sequence_number":0,"delta":"PRIVATE_NOT_EVIDENCE"}\n\n',
                b'data: {invalid}\n\n']
        for index, raw in enumerate(raws):
            target = self.root / str(index)
            generator = ResponsesGenerator({"model": "gpt-6.1-sol", "streaming": True})
            with patch("harness_engineering.generation.responses_streaming.streaming_transport",
                       side_effect=lambda *a, **kw: parse_sse([raw], deadline=1e100)) as mocked:
                with self.assertRaises(GenerationError) as error:
                    generator.generate({}, target)
            self.assertIn(error.exception.status, {"incomplete", "invalid_stream"})
            self.assertEqual(mocked.call_count, 1)
            self.assertFalse((target / "bundle.json").exists())
            status = json.loads((target / "attempt-01/status.json").read_text())
            self.assertEqual(status["transport_metadata"]["mode"], "responses_sse")
            self.assertNotIn("PRIVATE_NOT_EVIDENCE", json.dumps(status))
            self.assertNotIn("PRIVATE_NOT_EVIDENCE", (target / "attempt-01/response.json").read_text())

    def test_requested_model_and_reasoning_effort_are_preserved_without_fallback(self):
        requests = []

        def transport(request, **kwargs):
            requests.append(request)
            raise urllib.error.HTTPError("https://api.openai.com/v1/responses", 400,
                                         "unsupported effort", {}, io.BytesIO(b'{"error":"unsupported ultra"}'))

        generator = ResponsesGenerator({"model": "gpt-6.1-sol", "reasoning_effort": "ultra",
                                        "transport": transport, "sleep": lambda _: None})
        with self.assertRaises(GenerationError) as error:
            generator.generate({}, self.root)
        self.assertEqual(error.exception.status, "http_error")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["model"], "gpt-6.1-sol")
        self.assertEqual(requests[0]["reasoning"], {"effort": "ultra"})
        saved = json.loads((self.root / "attempt-01/request.json").read_text())
        self.assertEqual(saved["reasoning"], {"effort": "ultra"})

    def test_network_errors_have_three_attempt_budget(self):
        calls = []

        def transport(*args, **kwargs):
            calls.append(1)
            raise urllib.error.URLError("temporary transport failure")

        with self.assertRaises(GenerationError) as error:
            self.generator(transport).generate({}, self.root)
        self.assertEqual(len(calls), 3)
        self.assertEqual(error.exception.status, "network_error")
        self.assertEqual(len(list(self.root.glob("attempt-*/status.json"))), 3)
        self.assertFalse((self.root / "bundle.json").exists())

    def test_429_and_server_errors_retry_then_succeed(self):
        failures = iter([429, 503, None])

        def transport(*args, **kwargs):
            code = next(failures)
            if code:
                raise urllib.error.HTTPError("https://api.openai.com/v1/responses", code,
                                             "transient", {}, io.BytesIO(b'{"error":"busy"}'))
            return response()

        self.assertEqual(self.generator(transport).generate({}, self.root), bundle())
        self.assertTrue((self.root / "attempt-03/response.json").exists())

    def test_non_retryable_http_error_is_recorded_without_credentials(self):
        calls = []

        def transport(*args, **kwargs):
            calls.append(1)
            raise urllib.error.HTTPError("https://api.openai.com/v1/responses", 401,
                                         "bad key " + SECRET, {}, io.BytesIO(SECRET.encode()))

        with self.assertRaises(GenerationError) as error:
            self.generator(transport).generate({}, self.root)
        self.assertEqual(len(calls), 1)
        self.assertEqual(error.exception.status, "http_error")
        self.assertNotIn(SECRET, str(error.exception))
        self.assertNotIn(SECRET, (self.root / "attempt-01/response.json").read_text())

    def test_refusal_incomplete_and_truncated_json_never_retry_or_apply(self):
        incomplete = response(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        refusal = response(output=[{"type": "message", "content": [{"type": "refusal", "refusal": "No."}]}])
        truncated = response(output=[{"type": "message", "content": [{"type": "output_text", "text": '{"files":['}]}])
        for index, (raw, expected) in enumerate(((incomplete, "incomplete"), (refusal, "refusal"),
                                                (truncated, "invalid_json"))):
            calls = []
            out = self.root / str(index)

            def transport(*args, **kwargs):
                calls.append(1)
                return raw

            with self.subTest(expected=expected), self.assertRaises(GenerationError) as error:
                self.generator(transport).generate({}, out)
            self.assertEqual(error.exception.status, expected)
            self.assertEqual(error.exception.usage["output_tokens"], 83)
            self.assertEqual(len(calls), 1)
            self.assertFalse((out / "bundle.json").exists())
            self.assertFalse((out / "solution.py").exists())
            self.assertTrue((out / "attempt-01/response.json").exists())

    def test_duplicate_json_fields_are_rejected(self):
        raw = response(output=[{"type": "message", "content": [{"type": "output_text", "text":
            '{"files":[],"files":[],"rationale":"x","expected_effect":"y"}'}]}])
        with self.assertRaises(GenerationError) as error:
            self.generator(lambda *a, **kw: raw).generate({}, self.root)
        self.assertEqual(error.exception.status, "invalid_json")

    def test_invalid_bundle_from_model_never_produces_an_applyable_artifact(self):
        invalid = bundle()
        invalid["files"][0]["path"] = "../validator.py"
        calls = []

        def transport(*args, **kwargs):
            calls.append(1)
            return response(invalid)

        with self.assertRaises(GenerationError) as error:
            self.generator(transport).generate({}, self.root)
        self.assertEqual(error.exception.status, "invalid_path")
        self.assertEqual(len(calls), 1)
        self.assertFalse((self.root / "bundle.json").exists())
        self.assertFalse((self.root.parent / "validator.py").exists())

    def test_oversized_response_is_bounded_recorded_and_not_retried(self):
        raw = response()
        raw["unused"] = "x" * 2_000_001
        calls = []

        def transport(*args, **kwargs):
            calls.append(1)
            return raw

        with self.assertRaises(GenerationError) as error:
            self.generator(transport).generate({}, self.root)
        self.assertEqual(error.exception.status, "response_too_large")
        self.assertEqual(len(calls), 1)
        self.assertTrue(json.loads((self.root / "attempt-01/response.json").read_text())["truncated"])
        self.assertFalse((self.root / "bundle.json").exists())

    def test_credential_in_context_is_not_sent_or_written(self):
        def forbidden_transport(*args, **kwargs):
            self.fail("A credential-bearing context must not be sent")
        with self.assertRaises(GenerationError):
            self.generator(forbidden_transport).generate({"public_prompt": SECRET}, self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_credential_echo_in_response_is_redacted_and_rejected(self):
        raw = response(bundle(initial_source() + "\n# " + SECRET))
        with self.assertRaises(GenerationError) as error:
            self.generator(lambda *a, **kw: raw).generate({}, self.root)
        self.assertEqual(error.exception.status, "credential_content")
        self.assertNotIn(SECRET, (self.root / "attempt-01/response.json").read_text())
        self.assertFalse((self.root / "bundle.json").exists())

    def test_network_diagnostics_survive_retry_and_error_conversion(self):
        calls = []
        audit = {"mode": "responses_sse", "phase": "body", "http_status": 200,
                 "stream_bytes": 47, "event_count": 1,
                 "last_event_type": "response.created"}
        def transport(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                error = urllib.error.URLError("no complete response")
                error.transport_metadata = audit
                raise error
            return response()
        self.generator(transport, max_retries=1).generate({}, self.root)
        first = json.loads((self.root / "attempt-01/status.json").read_text())
        self.assertEqual(first["status"], "network_error")
        self.assertTrue(first["retry"])
        self.assertEqual(first["transport_metadata"], audit)
        receipt = json.loads((self.root / "attempt-01/response.json").read_text())
        self.assertFalse(receipt["received"])
        self.assertEqual(receipt["transport_metadata"]["stream_bytes"], 47)
        self.assertEqual(len(calls), 2)

    def test_http_diagnostics_survive_without_changing_retry_policy(self):
        audit = {"mode": "responses_sse", "phase": "headers", "http_status": 400,
                 "content_type": "application/json"}
        calls = []
        def transport(*args, **kwargs):
            calls.append(1)
            error = urllib.error.HTTPError("https://offline.invalid", 400, "mock", {}, io.BytesIO())
            error.transport_metadata = audit
            raise error
        with self.assertRaises(GenerationError) as raised:
            self.generator(transport).generate({}, self.root)
        self.assertEqual(raised.exception.status, "http_error")
        self.assertEqual(raised.exception.transport_metadata, audit)
        status = json.loads((self.root / "attempt-01/status.json").read_text())
        self.assertEqual(status["http_status"], 400)
        self.assertEqual(status["transport_metadata"], audit)
        self.assertFalse(status["retry"])
        self.assertEqual(len(calls), 1)

    def test_cancellation_archives_progress_without_swallowing_the_interrupt(self):
        cancellation = KeyboardInterrupt()
        cancellation.transport_metadata = {"phase": "body", "http_status": 200,
                                           "stream_bytes": 83, "event_count": 2,
                                           "redaction_check": SECRET}
        def transport(*args, **kwargs):
            raise cancellation
        with self.assertRaises(KeyboardInterrupt) as raised:
            self.generator(transport).generate({}, self.root)
        self.assertIs(raised.exception, cancellation)
        status = json.loads((self.root / "attempt-01/status.json").read_text())
        self.assertEqual(status["status"], "cancelled")
        self.assertFalse(status["retry"])
        self.assertEqual(status["http_status"], 200)
        self.assertEqual(status["transport_metadata"]["stream_bytes"], 83)
        self.assertIsNone(status["reported_output_tokens"])
        self.assertFalse((self.root / "bundle.json").exists())
        for path in self.root.rglob("*.json"):
            self.assertNotIn(SECRET, path.read_text())

    def test_cancel_archive_io_failure_preserves_the_original_interrupt(self):
        from harness_engineering.generation import generator as module
        original_write = module._atomic_json
        cancellation = KeyboardInterrupt()
        cancellation.transport_metadata = {"phase": "body", "stream_bytes": 51}
        def transport(*args, **kwargs):
            raise cancellation
        def failing_write(path, value):
            if Path(path).name != "request.json":
                raise OSError("evidence disk full")
            return original_write(path, value)
        with patch.object(module, "_atomic_json", side_effect=failing_write), \
                self.assertRaises(KeyboardInterrupt) as raised:
            self.generator(transport).generate({}, self.root)
        self.assertIs(raised.exception, cancellation)
        self.assertFalse((self.root / "bundle.json").exists())

    def test_complete_traceback_hides_provider_exception_credentials(self):
        def transport(*args, **kwargs):
            raise urllib.error.HTTPError("https://offline.invalid", 401,
                                         "bad key " + SECRET, {}, io.BytesIO(SECRET.encode()))
        try:
            self.generator(transport).generate({}, self.root)
        except GenerationError as error:
            rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            self.assertNotIn(SECRET, rendered)
            self.assertIn("Responses API HTTP 401", rendered)
        else:
            self.fail("HTTP error must be rejected")

    def test_default_json_transport_uses_isolated_deadline_without_changing_request(self):
        from harness_engineering.generation import responses_streaming as wire
        raw = response()
        raw["_transport_metadata"] = {"mode": "responses_json", "wait_stage": "completed"}
        with patch.object(wire, "json_transport", return_value=raw) as isolated, \
                patch.object(wire, "streaming_transport") as streaming:
            result = ResponsesGenerator({"model": "gpt-6.1-sol", "reasoning_effort": "max",
                                         "timeout_s": 600, "max_retries": 0}).generate({}, self.root)
        self.assertEqual(result, bundle())
        streaming.assert_not_called()
        request = isolated.call_args.args[0]
        self.assertNotIn("stream", request)
        self.assertEqual(request["model"], "gpt-6.1-sol")
        self.assertEqual(request["reasoning"]["effort"], "max")
        self.assertEqual(request["max_output_tokens"], 16384)
        self.assertEqual(isolated.call_args.kwargs["timeout_s"], 600)
        self.assertEqual(json.loads((self.root / "attempt-01/request.json").read_text()), request)
        status = json.loads((self.root / "attempt-01/status.json").read_text())
        self.assertEqual(status["transport_metadata"]["mode"], "responses_json")

    def test_default_json_deadline_retains_diagnostics_without_candidate_or_hidden_retry(self):
        from harness_engineering.generation import responses_streaming as wire
        error = urllib.error.URLError("Absolute response deadline exceeded")
        error.transport_metadata = {"mode": "responses_json", "wait_stage": "response_body",
                                    "stream_bytes": 10, "http_status": 200}
        with patch.object(wire, "json_transport", side_effect=error) as isolated, \
                self.assertRaises(GenerationError) as raised:
            ResponsesGenerator({"model": "gpt-6.1-sol", "max_retries": 0}).generate({}, self.root)
        self.assertEqual(raised.exception.status, "network_error")
        self.assertEqual(isolated.call_count, 1)
        self.assertFalse((self.root / "bundle.json").exists())
        status = json.loads((self.root / "attempt-01/status.json").read_text())
        self.assertEqual(status["transport_metadata"], error.transport_metadata)
        self.assertFalse(status["retry"])
        self.assertIsNone(status["reported_output_tokens"])

    def test_invalid_paths_and_extra_files_never_modify_existing_source(self):
        source = self.root / "solution.py"
        source.write_text(initial_source())
        for path in ("../solution.py", "/tmp/solution.py", "sub/solution.py", "./solution.py", "solution.py/../x"):
            bad = bundle()
            bad["files"][0]["path"] = path
            with self.subTest(path=path), self.assertRaises(GenerationError):
                apply_bundle(bad, self.root)
            self.assertEqual(source.read_text(), initial_source())
        bad = bundle()
        bad["files"].append({"path": "validator.py", "content": "pass"})
        with self.assertRaises(GenerationError):
            apply_bundle(bad, self.root)

    def test_syntax_interfaces_and_forbidden_imports_are_rejected(self):
        for content in ("def broken(:", "def build_jk(problem,density,backend): pass",
                        initial_source().replace("density, backend", "density, options"),
                        "import pyscf\n" + initial_source(), "import os\n" + initial_source(),
                        "from importlib import import_module\n" + initial_source(),
                        initial_source() + '\neval("1")\n',
                        initial_source() + '\n().__class__\n'):
            with self.subTest(content=content), self.assertRaises(GenerationError):
                validate_bundle(bundle(content))
        validate_bundle(bundle("import numpy as np\nfrom scipy.linalg import eigh\n" + initial_source()))

    def test_atomic_apply_returns_hashes_and_reviewable_diff(self):
        source = self.root / "solution.py"
        source.write_text(initial_source())
        changed = initial_source().replace("has not been generated", "must be implemented")
        result = apply_bundle(bundle(changed), self.root)
        self.assertEqual(source.read_text(), changed)
        self.assertEqual(result["files"][0]["parent_sha256"], hashlib.sha256(initial_source().encode()).hexdigest())
        self.assertEqual(result["files"][0]["sha256"], hashlib.sha256(changed.encode()).hexdigest())
        self.assertIn("--- previous/solution.py", result["diff"])
        self.assertEqual([p.name for p in self.root.iterdir()], ["solution.py"])

    def test_failed_atomic_replace_preserves_previous_source_and_cleans_temporary(self):
        source = self.root / "solution.py"
        source.write_text(initial_source())
        with patch("harness_engineering.generation.generator.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(GenerationError) as error:
                apply_bundle(bundle(initial_source() + "\n# new\n"), self.root)
        self.assertEqual(error.exception.status, "apply_error")
        self.assertEqual(source.read_text(), initial_source())
        self.assertEqual([p.name for p in self.root.iterdir()], ["solution.py"])

    def test_invalid_previous_encoding_is_rejected_before_source_changes(self):
        source = self.root / "solution.py"
        source.write_bytes(b"\xff\xfe")
        with self.assertRaises(GenerationError):
            apply_bundle(bundle(), self.root)
        self.assertEqual(source.read_bytes(), b"\xff\xfe")

    def test_symlink_target_and_candidate_root_are_rejected(self):
        outside = self.root / "outside.py"
        outside.write_text("untouched")
        candidate = self.root / "candidate"
        candidate.mkdir()
        (candidate / "solution.py").symlink_to(outside)
        with self.assertRaises(GenerationError):
            apply_bundle(bundle(), candidate)
        link = self.root / "linked-candidate"
        link.symlink_to(candidate, target_is_directory=True)
        with self.assertRaises(GenerationError):
            apply_bundle(bundle(), link)
        self.assertEqual(outside.read_text(), "untouched")


if __name__ == "__main__":
    unittest.main()
