"""Offline JSON transport deadlines and compatibility; no sockets are opened."""
import base64
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error

from harness_engineering.generation import responses_streaming as transport


SECRET = "offline-json-test-credential"
URL = "https://offline.invalid/v1/responses"


def response():
    return {"id": "offline-json", "status": "completed", "output": [],
            "usage": {"input_tokens": 37, "output_tokens": 83}}


class JSONReaderTests(unittest.TestCase):
    def test_complete_json_is_unchanged_including_incomplete_refusal_and_usage(self):
        variants = [response(), {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
                    {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal"}]}]},
                    {**response(), "usage": {"output_tokens": 23663}}]
        for value in variants:
            wire = json.dumps(value).encode()
            with self.subTest(value=value):
                self.assertEqual(transport.parse_json([wire[:9], wire[9:]], deadline=time.monotonic() + 2), value)

    def test_invalid_json_keeps_bounded_raw_failure_evidence(self):
        for raw in (b"not JSON", b'{"output_tokens": NaN}', b'{"a":1,"a":2}', b"\xff"):
            with self.subTest(raw=raw):
                result = transport.parse_json([raw], deadline=time.monotonic() + 2)
                self.assertEqual(result["_transport_error"], "invalid_response")
                self.assertEqual(result["raw_prefix"], raw.decode("utf-8", errors="replace"))

    def test_oversized_json_returns_only_the_bounded_prefix(self):
        with patch.object(transport, "MAX_RESPONSE_BYTES", 32):
            result = transport.parse_json([b"x" * 16, b"y" * 20], deadline=time.monotonic() + 2)
        self.assertEqual(result["_transport_error"], "response_too_large")
        self.assertEqual(result["raw_prefix"], "x" * 16 + "y" * 16)

    def test_progress_cannot_reset_the_absolute_parser_deadline(self):
        ticks = iter([0.0, 0.0, .2, .6, .7, 1.1])
        with self.assertRaises(transport.SSETransportError) as error:
            transport.parse_json([b" "] * 5, deadline=1.0, clock=lambda: next(ticks))
        self.assertEqual(error.exception.status, "network_error")


class JSONWorkerTests(unittest.TestCase):
    def worker(self, *, chunks=None, open_error=None, limit=None, credential=SECRET):
        chunks = iter(chunks if chunks is not None else [json.dumps(response()).encode(), b""])
        class Source:
            status = 200
            headers = SimpleNamespace(get_content_type=lambda: "application/json")
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read1(self, amount): return next(chunks)
        opener = Mock()
        if open_error is not None:
            opener.open.side_effect = open_error
        else:
            opener.open.return_value = Source()
        config = {"api_key": credential, "request": {"model": "gpt-6.1-sol", "reasoning": {"effort": "max"},
                  "max_output_tokens": 16384}, "url": URL, "timeout_s": 600, "transport_mode": "json"}
        captured = io.StringIO()
        with tempfile.TemporaryFile(mode="w+b") as diagnostics, \
                patch.object(transport.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(json.dumps(config).encode()))), \
                patch.object(transport.sys, "stdout", captured), \
                patch.object(transport.urllib.request, "build_opener", return_value=opener), \
                patch.object(transport, "MAX_RESPONSE_BYTES", limit or transport.MAX_RESPONSE_BYTES):
            transport._worker(diagnostics.fileno())
            metadata = transport._read_diagnostics(diagnostics)
        self.assertNotIn(credential, captured.getvalue())
        return json.loads(captured.getvalue()), metadata, opener

    def test_actual_worker_mock_json_keeps_request_body_and_response(self):
        result, metadata, opener = self.worker()
        self.assertEqual(result["kind"], "completed")
        self.assertEqual(result["response"], response())
        request = opener.open.call_args.args[0]
        self.assertNotIn("stream", json.loads(request.data))
        self.assertEqual(request.get_header("Accept"), "application/json")
        self.assertEqual(metadata["http_status"], 200)
        self.assertEqual(metadata["content_type"], "application/json")
        self.assertIn("body_complete", metadata["phase_timings"])

    def test_actual_worker_invalid_and_oversize_response_are_failure_evidence(self):
        invalid, _, _ = self.worker(chunks=[b"not JSON", b""])
        self.assertEqual(invalid["response"]["_transport_error"], "invalid_response")
        oversized, _, _ = self.worker(chunks=[b"x" * 65], limit=64)
        self.assertEqual(oversized["response"]["_transport_error"], "response_too_large")
        self.assertEqual(len(oversized["response"]["raw_prefix"]), 64)

    def test_http_error_body_is_bounded_and_credential_redacted_before_ipc(self):
        error = urllib.error.HTTPError(URL, 503, "busy", {}, io.BytesIO(("body " + SECRET).encode()))
        result, metadata, _ = self.worker(open_error=error)
        self.assertEqual(result["kind"], "http_error")
        self.assertEqual(result["code"], 503)
        self.assertEqual(base64.b64decode(result["body_base64"]), b"body [REDACTED]")
        self.assertEqual(metadata["http_status"], 503)

    def test_unreadable_http_error_body_does_not_change_http_status(self):
        for code in (400, 503):
            body = Mock()
            body.read.side_effect = TimeoutError("mock error body unavailable")
            error = urllib.error.HTTPError(URL, code, "offline error", {}, body)
            with self.subTest(code=code):
                result, _, _ = self.worker(open_error=error)
                self.assertEqual(result["kind"], "http_error")
                self.assertEqual(result["code"], code)
                self.assertEqual(base64.b64decode(result["body_base64"]), b"")

    def test_redaction_expansion_keeps_error_body_and_failure_prefix_bounded(self):
        error = urllib.error.HTTPError(URL, 503, "busy", {}, io.BytesIO(b"#" * 65))
        result, _, _ = self.worker(open_error=error, limit=64, credential="#")
        self.assertEqual(result["code"], 503)
        self.assertEqual(len(base64.b64decode(result["body_base64"])), 65)
        result, _, _ = self.worker(chunks=[b"#" * 64, b""], limit=64, credential="#")
        self.assertEqual(result["response"]["_transport_error"], "invalid_response")
        self.assertEqual(len(result["response"]["raw_prefix"]), 64)


class JSONProcessDeadlineTests(unittest.TestCase):
    def call(self, request=None, timeout_s=2):
        return transport.json_transport(request or {"model": "gpt-6.1-sol", "reasoning": {"effort": "max"},
            "max_output_tokens": 16384, "store": False, "text": {"format": {"strict": True}}},
            api_key=SECRET, timeout_s=timeout_s, url=URL)

    def child_command(self, code):
        return lambda fd: [sys.executable, "-I", "-c", code, str(fd)]

    def test_json_request_does_not_gain_stream_true_or_change_existing_constraints(self):
        child = ("import sys,json,os,hashlib;c=json.load(sys.stdin);"
                 "r={'status':'completed','output':[],'usage':{'output_tokens':1},'request':c['request'],"
                 "'env_has_key':any('API' in k or k=='SSH_AUTH_SOCK' for k in os.environ),"
                 "'key_sha256':hashlib.sha256(c['api_key'].encode()).hexdigest()};"
                 "print(json.dumps({'kind':'completed','response':r}))")
        request = {"model": "gpt-6.1-sol", "reasoning": {"effort": "max"}, "max_output_tokens": 16384,
                   "store": False, "text": {"format": {"type": "json_schema", "strict": True}}}
        before = copy.deepcopy(request)
        with patch.object(transport, "_worker_command", side_effect=self.child_command(child)), \
                patch.dict(os.environ, {"OPENAI_API_KEY": "do-not-inherit", "SSH_AUTH_SOCK": "/secret/socket"}):
            result = self.call(request)
        self.assertEqual(request, before)
        self.assertEqual(result["request"], before)
        self.assertNotIn("stream", result["request"])
        self.assertFalse(result["env_has_key"])
        self.assertEqual(result["_transport_metadata"]["mode"], "responses_json")

    def test_json_http_status_and_error_body_keep_standard_exception_types(self):
        for code in (400, 429, 503):
            serialized = json.dumps({"kind": "http_error", "code": code,
                                     "body_base64": base64.b64encode(b"offline failure body").decode()})
            child = "import sys;sys.stdin.buffer.read();print(" + repr(serialized) + ")"
            with self.subTest(code=code), patch.object(transport, "_worker_command", side_effect=self.child_command(child)), \
                    self.assertRaises(urllib.error.HTTPError) as error:
                self.call()
            self.assertEqual(error.exception.code, code)
            self.assertEqual(error.exception.read(), b"offline failure body")
            self.assertEqual(error.exception.transport_metadata["mode"], "responses_json")

    def test_explicit_1200_second_deadline_reaches_mock_worker_without_waiting(self):
        process = Mock(returncode=0)
        process.poll.return_value = 0
        process.communicate.return_value = (json.dumps({"kind": "completed", "response": response()}).encode(), b"")
        with patch.object(transport.subprocess, "Popen", return_value=process):
            result = self.call(timeout_s=1200)
        call = process.communicate.call_args.kwargs
        envelope = json.loads(call["input"])
        self.assertEqual(envelope["timeout_s"], 1200)
        self.assertGreater(call["timeout"], 1199)
        self.assertLessEqual(call["timeout"], 1200)
        self.assertEqual(envelope["request"]["model"], "gpt-6.1-sol")
        self.assertEqual(envelope["request"]["reasoning"], {"effort": "max"})
        self.assertEqual(envelope["request"]["max_output_tokens"], 16384)
        self.assertNotIn("stream", envelope["request"])
        self.assertEqual(result["_transport_metadata"]["mode"], "responses_json")

    def test_parent_deadline_stops_both_slow_json_drip_and_blocked_headers(self):
        module_path = str(Path(transport.__file__).resolve())
        for slow_body in (False, True):
            # Execute the real HTTP worker with an in-memory opener. Neither
            # mode creates a socket: one blocks before headers, the other
            # yields bytes more often than any socket idle timeout would fire.
            child = ("import importlib.util,sys,time,types;"
                "s=importlib.util.spec_from_file_location('offline_json_worker'," + repr(module_path) + ");"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                "source=types.SimpleNamespace(status=200,headers=types.SimpleNamespace(get_content_type=lambda:'application/json'));"
                "source.read1=lambda n:(time.sleep(.02) or b' ');"
                "Source=type('Source',(),{'__enter__':lambda self:source,'__exit__':lambda *a:False});"
                "opener=types.SimpleNamespace(open=lambda *a,**k:" +
                ("Source()" if slow_body else "(time.sleep(10) or Source())") + ");"
                "m.urllib.request.build_opener=lambda *a,**k:opener;"
                "m._worker(int(sys.argv[1]))")
            started = time.monotonic()
            processes = []
            real_start = subprocess.Popen
            def start(*args, **kwargs):
                process = real_start(*args, **kwargs); processes.append(process); return process
            with self.subTest(slow_body=slow_body), \
                    patch.object(transport, "_worker_command", side_effect=self.child_command(child)), \
                    patch.object(transport.subprocess, "Popen", side_effect=start), \
                    self.assertRaises(urllib.error.URLError) as error:
                self.call(timeout_s=.2)
            self.assertLess(time.monotonic() - started, 2)
            self.assertIsNotNone(processes[0].poll())
            self.assertEqual(error.exception.transport_metadata["mode"], "responses_json")
            if slow_body:
                self.assertGreater(error.exception.transport_metadata["stream_bytes"], 0)
                self.assertIn("response_headers", error.exception.transport_metadata["phase_timings"])

    def test_json_cancellation_uses_shared_killgroup_and_closes_diagnostic_fd(self):
        process = Mock(pid=123456789)
        process.poll.return_value = None
        descriptor = None
        original = KeyboardInterrupt()
        calls = 0
        def start(*args, **kwargs):
            nonlocal descriptor
            descriptor = kwargs["pass_fds"][0]
            return process
        def communicate(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                os.write(descriptor, b'{"phase":"request_sent","elapsed_seconds":0.1}\n')
                raise original
            return b"", b""
        process.communicate.side_effect = communicate
        with patch.object(transport.subprocess, "Popen", side_effect=start), \
                patch.object(transport.os, "killpg") as killed, self.assertRaises(KeyboardInterrupt) as error:
            self.call()
        self.assertIs(error.exception, original)
        self.assertEqual(error.exception.transport_metadata["mode"], "responses_json")
        killed.assert_called_once_with(process.pid, transport.signal.SIGKILL)
        with self.assertRaises(OSError):
            os.fstat(descriptor)

    def test_json_transport_rejects_streaming_request_without_launching(self):
        with patch.object(transport.subprocess, "Popen") as start, self.assertRaises(transport.SSETransportError):
            self.call({"stream": True})
        start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
