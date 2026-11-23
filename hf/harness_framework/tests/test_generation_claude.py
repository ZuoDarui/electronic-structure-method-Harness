"""Claude generator: request shape, response handling and evidence, with a fake SDK client."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from harness_engineering.generation.claude_generator import CLAUDE_SCHEMA, ClaudeGenerator
from harness_engineering.generation.generator import GenerationError

ROOT = Path(__file__).resolve().parents[1]
GOOD = (ROOT / "tests/fixtures/rhf_numpy_solution.py").read_text()
KEY = "sk-ant-test-secret-value"


class FakeMessage:
    def __init__(self, data):
        self.data = data

    def to_dict(self):
        return self.data


class FakeStream:
    def __init__(self, owner, kwargs):
        self.owner, self.kwargs = owner, kwargs

    def __enter__(self):
        self.owner.requests.append(self.kwargs)
        if self.owner.error:
            raise self.owner.error
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return FakeMessage(self.owner.reply)


class FakeMessages:
    def __init__(self, owner):
        self.owner = owner

    def stream(self, **kwargs):
        return FakeStream(self.owner, kwargs)


class FakeClient:
    def __init__(self, reply=None, error=None):
        self.requests, self.reply, self.error = [], reply, error
        self.messages = FakeMessages(self)
        self.beta = type("Beta", (), {"messages": FakeMessages(self)})()


def reply(source=GOOD, stop="end_turn"):
    text = json.dumps({"solution_py": source, "rationale": "Roothaan SCF + DIIS", "expected_effect": "correct RHF"})
    return {"id": "msg_test", "model": "claude-opus-5-5", "stop_reason": stop, "stop_details": None,
            "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": text}],
            "usage": {"input_tokens": 1200, "output_tokens": 3400}}


CONTEXT = {"public_prompt": {"name": "t"}, "contract": "c", "current_source": "", "round": 1,
           "private_reference": "must not be sent"}


class ClaudeGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": KEY})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def make(self, client, **config):
        return ClaudeGenerator({"model": "claude-opus-5-5", "reasoning_effort": "xhigh", "client": client, **config})

    def test_request_shape_and_successful_bundle(self):
        client = FakeClient(reply())
        bundle = self.make(client).generate(CONTEXT, self.dir / "gen")
        self.assertEqual(bundle["files"][0]["path"], "solution.py")
        self.assertIn("def solve", bundle["files"][0]["content"])
        request = client.requests[0]
        self.assertEqual(request["model"], "claude-opus-5-5")
        self.assertEqual(request["thinking"], {"type": "adaptive"})
        self.assertEqual(request["output_config"]["effort"], "xhigh")
        self.assertEqual(request["output_config"]["format"], {"type": "json_schema", "schema": CLAUDE_SCHEMA})
        self.assertEqual(request["fallbacks"], "default")
        self.assertEqual(request["betas"], ["server-side-fallback-2026-07-01"])
        self.assertNotIn("temperature", request)
        self.assertNotIn("private_reference", request["messages"][0]["content"])
        evidence = "".join(p.read_text() for p in (self.dir / "gen").rglob("*.json"))
        self.assertNotIn(KEY, evidence)
        status = json.loads((self.dir / "gen/attempt-01/status.json").read_text())
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["usage"]["output_tokens"], 3400)

    def test_fallback_can_be_disabled_for_proxies(self):
        client = FakeClient(reply())
        self.make(client, refusal_fallback=False).generate(CONTEXT, self.dir / "gen")
        self.assertNotIn("fallbacks", client.requests[0])
        self.assertNotIn("betas", client.requests[0])

    def test_refusal_and_truncation_are_not_applied(self):
        for stop, status in (("refusal", "refusal"), ("max_tokens", "output_budget_exceeded")):
            with self.subTest(stop=stop):
                with self.assertRaises(GenerationError) as caught:
                    self.make(FakeClient(reply(stop=stop))).generate(CONTEXT, self.dir / stop)
                self.assertEqual(caught.exception.status, status)

    def test_generated_source_still_passes_ast_guard(self):
        with self.assertRaises(GenerationError) as caught:
            self.make(FakeClient(reply("import os\n" + GOOD))).generate(CONTEXT, self.dir / "guard")
        self.assertEqual(caught.exception.status, "forbidden_import")

    def test_sdk_errors_map_to_blocking_statuses(self):
        import anthropic
        import httpx2 as httpx
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(401, request=request)
        error = anthropic.AuthenticationError("bad key " + KEY, response=response, body=None)
        with self.assertRaises(GenerationError) as caught:
            self.make(FakeClient(error=error)).generate(CONTEXT, self.dir / "auth")
        self.assertEqual(caught.exception.status, "authentication_error")
        self.assertNotIn(KEY, (self.dir / "auth/attempt-01/status.json").read_text())

    def test_missing_credential_and_bad_effort_are_config_errors(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(GenerationError):
                ClaudeGenerator({"model": "claude-opus-5-5"})
        with self.assertRaises(GenerationError):
            self.make(FakeClient(reply()), reasoning_effort="ultra")


if __name__ == "__main__":
    unittest.main()
