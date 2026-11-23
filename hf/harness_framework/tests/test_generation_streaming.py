"""Offline parser/process tests: no API endpoint, credential or model is called."""
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import urllib.error
from types import SimpleNamespace
import io

from harness_engineering.generation import responses_streaming as transport
from harness_engineering.generation.generator import ResponsesGenerator, GenerationError


def complete_response():
    return {"id": "resp-offline", "status": "completed", "error": None,
            "incomplete_details": None, "model": "gpt-6.1-sol",
            "output": [{"type": "message", "status": "completed", "content": [
                {"type": "output_text", "text": '{"files":[]}'}]}],
            "usage": {"input_tokens": 400, "output_tokens": 83,
                      "output_tokens_details": {"reasoning_tokens": 40}, "total_tokens": 483}}


def event(kind, sequence=0, response=None, **fields):
    value = {"type": kind, "sequence_number": sequence, **fields}
    if response is not None:
        value["response"] = response
    return ("event: " + kind + "\ndata: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode()


def parse(parts, **kwargs):
    return transport.parse_sse(parts, deadline=time.monotonic() + 10, **kwargs)


class SSEParserTests(unittest.TestCase):
    def test_completed_response_is_returned_intact_with_only_safe_event_metadata(self):
        response = complete_response()
        raw = (event("response.created", 0) +
               event("response.reasoning_text.delta", 1, delta="DO NOT RECORD PRIVATE TEXT") +
               event("response.output_text.delta", 2, delta="partial source") +
               event("future.progress.event", 3, payload="also unrecorded") +
               event("response.completed", 4, response))
        result = parse([raw[:13], raw[13:73], raw[73:]])
        metadata = result.pop("_transport_metadata")
        self.assertEqual(result, response)
        self.assertEqual(metadata["event_count"], 5)
        self.assertNotIn("DO NOT RECORD", json.dumps(metadata))
        self.assertNotIn("partial source", json.dumps(metadata))
        self.assertEqual(set(metadata["events"][0]), {"type", "sequence_number", "elapsed_seconds"})

    def test_utf8_byte_boundaries_crlf_comments_and_multi_data_are_supported(self):
        response = complete_response()
        response["output"][0]["content"][0]["text"] = "完整输出"
        raw = (b": heartbeat\r\n\r\n" + event("response.completed", 0, response).replace(b"\n", b"\r\n"))
        self.assertEqual(parse([bytes([byte]) for byte in raw])["output"], response["output"])
        encoded = json.dumps({"type": "response.completed", "response": response})
        split = encoded.index(', "response"') + 1
        multiline = ("data: " + encoded[:split] + "\ndata: " + encoded[split:] + "\n\n").encode()
        self.assertEqual(parse([multiline])["usage"], response["usage"])

    def test_text_done_and_done_sentinel_cannot_substitute_for_completion(self):
        for raw in (event("response.output_text.done", 0, text="complete looking JSON"),
                    b"data: [DONE]\n\n", b"", b": heartbeat\n\n"):
            with self.subTest(raw=raw), self.assertRaises(transport.SSETransportError) as error:
                parse([raw])
            self.assertEqual(error.exception.status, "incomplete")

    def test_errors_failures_incomplete_and_refusals_are_rejected(self):
        for kind in ("error", "response.failed", "response.incomplete", "response.refusal.delta", "response.refusal.done"):
            with self.subTest(kind=kind), self.assertRaises(transport.SSETransportError):
                parse([event(kind, 0, complete_response())])
        refused = complete_response()
        refused["output"][0]["content"] = [{"type": "refusal", "refusal": "no"}]
        with self.assertRaises(transport.SSETransportError) as error:
            parse([event("response.completed", 0, refused)])
        self.assertEqual(error.exception.status, "refusal")

    def test_malformed_json_type_mismatch_duplicate_keys_and_sequence_are_rejected(self):
        invalid = [b"data: {bad}\n\n", b"data: []\n\n",
                   b'event: response.completed\ndata: {"type":"response.created"}\n\n',
                   b'data: {"type":"error","type":"response.created"}\n\n',
                   b'data: {"type":"x","sequence_number":true}\n\n',
                   event("response.created", 2) + event("response.output_text.delta", 1)]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(transport.SSETransportError) as error:
                parse([raw])
            self.assertEqual(error.exception.status, "invalid_stream")

    def test_eof_inside_completed_frame_is_not_accepted(self):
        raw = event("response.completed", 0, complete_response())
        with self.assertRaises(transport.SSETransportError) as error:
            parse([raw[:-1]])
        self.assertEqual(error.exception.status, "incomplete")

    def test_completed_requires_full_output_success_and_usage(self):
        variants = []
        for field in ("output", "usage"):
            value = complete_response(); del value[field]; variants.append(value)
        value = complete_response(); value["status"] = "incomplete"; variants.append(value)
        value = complete_response(); value["error"] = {"code": "server_error"}; variants.append(value)
        value = complete_response(); value["usage"] = {}; variants.append(value)
        for response in variants:
            with self.subTest(response=response), self.assertRaises(transport.SSETransportError):
                parse([event("response.completed", 0, response)])

    def test_total_bytes_frame_bytes_and_event_count_are_bounded(self):
        limits = [("MAX_STREAM_BYTES", 10, [b"x" * 11]),
                  ("MAX_FRAME_BYTES", 10, [b"data: " + b"x" * 11]),
                  ("MAX_EVENTS", 1, [event("response.created", 0) + event("progress", 1)])]
        for name, limit, parts in limits:
            with self.subTest(name=name), patch.object(transport, name, limit), self.assertRaises(transport.SSETransportError) as error:
                parse(parts)
            self.assertEqual(error.exception.status, "stream_limit")

    def test_heartbeat_does_not_reset_absolute_deadline(self):
        ticks = iter([0.0, 0.0, 0.5, 1.0, 2.0, 2.1, 2.2])
        with self.assertRaises(transport.SSETransportError) as error:
            transport.parse_sse([b": beat\n\n"] * 5, deadline=1.0, clock=lambda: next(ticks))
        self.assertEqual(error.exception.status, "network_error")

    def test_over_budget_usage_is_preserved_for_existing_generator_rejection(self):
        response = complete_response()
        response["usage"]["output_tokens"] = 23663
        response["usage"]["output_tokens_details"]["reasoning_tokens"] = 20179
        result = parse([event("response.completed", 0, response)])
        self.assertEqual(result["usage"], response["usage"])
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test-credential"}):
            generator = ResponsesGenerator({"model": "gpt-6.1-sol", "reasoning_effort": "max"})
            with self.assertRaises(GenerationError) as error:
                generator._validate_usage(result)
        self.assertEqual(error.exception.status, "output_budget_exceeded")

    def test_completed_response_byte_limit_is_enforced(self):
        with patch.object(transport, "MAX_RESPONSE_BYTES", 10), self.assertRaises(transport.SSETransportError) as error:
            parse([event("response.completed", 0, complete_response())])
        self.assertEqual(error.exception.status, "response_too_large")


class TransportProcessTests(unittest.TestCase):
    def call(self, request=None, timeout_s=2):
        return transport.streaming_transport(request or {"model": "gpt-6.1-sol", "reasoning": {"effort": "max"},
                    "max_output_tokens": 16384, "store": False, "text": {"format": {"strict": True}}},
                    api_key="offline-test-credential", timeout_s=timeout_s,
                    url="https://offline.invalid/v1/responses")

    def test_key_uses_stdin_only_child_environment_is_clean_and_request_changes_only_stream(self):
        child = ("import json,sys,os,hashlib; c=json.load(sys.stdin); "
                 "r={'status':'completed','output':[],'usage':{'output_tokens':1},"
                 "'echo_request':c['request'],'key_sha256':hashlib.sha256(c['api_key'].encode()).hexdigest(),"
                 "'env_api':any('API' in k or k=='SSH_AUTH_SOCK' for k in os.environ)}; "
                 "print(json.dumps({'kind':'completed','response':r}))")
        original = {"model": "gpt-6.1-sol", "reasoning": {"effort": "max"}, "max_output_tokens": 16384,
                    "store": False, "text": {"format": {"type": "json_schema", "strict": True}}}
        before = copy.deepcopy(original)
        command = [sys.executable, "-I", "-c", child]
        with patch.object(transport, "_worker_command", return_value=command), \
                patch.dict(os.environ, {"OPENAI_API_KEY": "must-not-inherit", "SSH_AUTH_SOCK": "/secret/socket"}):
            result = self.call(original)
        self.assertEqual(original, before)
        self.assertEqual(result["echo_request"], {**before, "stream": True})
        self.assertEqual(result["key_sha256"], hashlib.sha256(b"offline-test-credential").hexdigest())
        self.assertFalse(result["env_api"])
        self.assertNotIn("offline-test-credential", " ".join(command))

    def test_absolute_watchdog_kills_hung_worker_without_network(self):
        child = "import sys,time;sys.stdin.buffer.read();time.sleep(10)"
        real_popen = subprocess.Popen
        processes = []
        def start(*args, **kwargs):
            process = real_popen(*args, **kwargs); processes.append(process); return process
        started = time.monotonic()
        with patch.object(transport, "_worker_command", return_value=[sys.executable, "-I", "-c", child]), \
                patch.object(transport.subprocess, "Popen", side_effect=start), \
                self.assertRaises(urllib.error.URLError):
            self.call(timeout_s=.1)
        self.assertLess(time.monotonic() - started, 2)
        self.assertIsNotNone(processes[0].poll())

    def test_explicit_1200_second_deadline_reaches_mock_worker_without_waiting(self):
        process = unittest.mock.Mock(returncode=0)
        process.poll.return_value = 0
        process.communicate.return_value = (json.dumps({"kind": "completed", "response": complete_response()}).encode(), b"")
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
        self.assertIs(envelope["request"]["stream"], True)
        self.assertEqual(result["_transport_metadata"]["mode"], "responses_sse")

    def test_http_status_is_preserved_for_existing_retry_policy(self):
        for code in (400, 429, 503):
            child = "import sys;sys.stdin.buffer.read();print(" + repr(json.dumps({"kind": "http_error", "code": code})) + ")"
            with self.subTest(code=code), patch.object(transport, "_worker_command", return_value=[sys.executable, "-I", "-c", child]), \
                    self.assertRaises(urllib.error.HTTPError) as error:
                self.call()
            self.assertEqual(error.exception.code, code)

    def test_failed_or_malformed_worker_receipt_is_rejected(self):
        for child in ("import sys;sys.stdin.buffer.read();sys.exit(2)",
                      "import sys;sys.stdin.buffer.read();print('not JSON')"):
            with self.subTest(child=child), patch.object(transport, "_worker_command", return_value=[sys.executable, "-I", "-c", child]), \
                    self.assertRaises(transport.SSETransportError):
                self.call()

    def test_invalid_deadline_or_stream_false_is_rejected_without_starting_a_worker(self):
        for value in (False, 0, -1, 1201, float("inf")):
            with self.subTest(value=value), self.assertRaises(transport.SSETransportError):
                self.call(timeout_s=value)
        with self.assertRaises(transport.SSETransportError):
            self.call({"stream": False})

    def test_caller_cancellation_kills_the_detached_worker_group(self):
        process = unittest.mock.Mock()
        process.pid = 123456789
        process.poll.return_value = None
        process.communicate.side_effect = [KeyboardInterrupt(), (b"", b"")]
        with patch.object(transport.subprocess, "Popen", return_value=process), \
                patch.object(transport.os, "killpg") as killed, self.assertRaises(KeyboardInterrupt):
            self.call()
        killed.assert_called_once_with(process.pid, transport.signal.SIGKILL)

    def test_http_worker_reads_mock_sse_without_network_and_preserves_full_final_response(self):
        raw_response = complete_response()
        pieces = iter([event("response.created", 0), event("response.completed", 1, raw_response), b""])
        class Source:
            status = 200
            headers = SimpleNamespace(get_content_type=lambda: "text/event-stream")
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read1(self, count): return next(pieces)
        opener = unittest.mock.Mock()
        opener.open.return_value = Source()
        config = {"api_key": "offline-test-credential", "request": {"stream": True},
                  "url": "https://offline.invalid/v1/responses", "timeout_s": 2}
        captured = io.StringIO()
        with patch.object(transport.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(json.dumps(config).encode()))), \
                patch.object(transport.sys, "stdout", captured), \
                patch.object(transport.urllib.request, "build_opener", return_value=opener):
            transport._worker()
        result = json.loads(captured.getvalue())
        self.assertEqual(result["kind"], "completed")
        result["response"].pop("_transport_metadata")
        self.assertEqual(result["response"], raw_response)
        self.assertNotIn("offline-test-credential", captured.getvalue())

    def test_timeout_recovers_independently_flushed_headers_bytes_and_events(self):
        records = [
            {"phase": "connected", "elapsed_seconds": .01},
            {"phase": "request_sent", "elapsed_seconds": .02},
            {"phase": "response_headers", "elapsed_seconds": .03, "http_status": 200, "content_type": "text/event-stream"},
            {"phase": "first_byte", "elapsed_seconds": .04, "stream_bytes": 145},
            {"phase": "sse_event", "elapsed_seconds": .05, "type": "response.created",
             "sequence_number": 0, "event_count": 1, "stream_bytes": 145},
        ]
        encoded = "".join(json.dumps(row) + "\n" for row in records)
        child = ("import os,sys,time;sys.stdin.buffer.read();"
                 "os.write(int(sys.argv[1])," + repr(encoded.encode()) + ");time.sleep(10)")
        command = lambda fd: [sys.executable, "-I", "-c", child, str(fd)]
        with patch.object(transport, "_worker_command", side_effect=command), self.assertRaises(urllib.error.URLError) as error:
            self.call(timeout_s=.25)
        metadata = error.exception.transport_metadata
        self.assertEqual(metadata["http_status"], 200)
        self.assertEqual(metadata["content_type"], "text/event-stream")
        self.assertEqual(metadata["event_count"], 1)
        self.assertEqual(metadata["stream_bytes"], 145)
        self.assertEqual(metadata["wait_stage"], "response_body")
        self.assertEqual(metadata["first_event"]["type"], "response.created")
        self.assertEqual(metadata["last_event"]["sequence_number"], 0)
        self.assertNotIn("offline-test-credential", json.dumps(metadata))

    def test_cancellation_metadata_survives_partial_diagnostics_and_collection_failure(self):
        for read_failure in (False, True):
            original = KeyboardInterrupt()
            process = unittest.mock.Mock(pid=123456789)
            process.poll.return_value = None
            descriptor = None
            calls = 0
            def start(*args, **kwargs):
                nonlocal descriptor
                descriptor = kwargs["pass_fds"][0]
                return process
            def communicate(**kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    os.write(descriptor, b'{"phase":"connect_started","elapsed_seconds":0.1}\n{"partial"')
                    raise original
                return b"", b""
            process.communicate.side_effect = communicate
            reader = transport._read_diagnostics
            with self.subTest(read_failure=read_failure), \
                    patch.object(transport.subprocess, "Popen", side_effect=start), \
                    patch.object(transport.os, "killpg"), \
                    patch.object(transport, "_read_diagnostics", side_effect=OSError("diagnostic failure") if read_failure else reader), \
                    self.assertRaises(KeyboardInterrupt) as error:
                self.call()
            self.assertIs(error.exception, original)
            metadata = error.exception.transport_metadata
            if read_failure:
                self.assertTrue(metadata["diagnostic_collection_failed"])
            else:
                self.assertEqual(metadata["wait_stage"], "connect_dns_tcp_tls")
                self.assertTrue(metadata["discarded_partial_record"])
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_sparse_channel_checkpoints_cannot_reduce_final_response_counts(self):
        response = complete_response()
        response["_transport_metadata"] = {"stream_bytes": 5000, "event_count": 5}
        serialized = json.dumps({"kind": "completed", "response": response})
        child = ("import os,sys;sys.stdin.buffer.read();os.write(int(sys.argv[1]),"
                 "b'{\"phase\":\"first_byte\",\"elapsed_seconds\":0.1,\"stream_bytes\":100}\\n');"
                 "print(" + repr(serialized) + ")")
        with patch.object(transport, "_worker_command", side_effect=lambda fd: [sys.executable, "-I", "-c", child, str(fd)]):
            result = self.call()
        self.assertEqual(result["_transport_metadata"]["stream_bytes"], 5000)
        self.assertEqual(result["_transport_metadata"]["event_count"], 5)


class DiagnosticChannelTests(unittest.TestCase):
    def test_only_whitelisted_metadata_is_emitted_and_channel_size_is_bounded(self):
        with tempfile.TemporaryFile(mode="w+b") as channel, patch.object(transport, "MAX_DIAGNOSTIC_BYTES", 1024):
            writer = transport._Diagnostics(channel.fileno())
            writer.emit("response_headers", http_status=200, content_type="PRIVATE_HEADER_CONTENT",
                        Authorization="SECRET", url="https://private.example")
            writer.emit("sse_event", type="PRIVATE_REASONING_BODY", event_count=1, sequence_number=0)
            for index in range(1000):
                writer.emit("sse_event", type="response.output_text.delta", event_count=index + 2, sequence_number=index + 1)
            channel.seek(0)
            raw = channel.read()
            self.assertLessEqual(len(raw), 1024)
            self.assertNotIn(b"PRIVATE", raw)
            self.assertNotIn(b"SECRET", raw)
            self.assertNotIn(b"private.example", raw)
            metadata = transport._read_diagnostics(channel)
            self.assertTrue(metadata["diagnostics_truncated"])
            self.assertEqual(metadata["content_type"], "other")
            self.assertIn("unknown_progress", metadata["event_types"])

    def test_malformed_diagnostic_values_and_truncated_tail_are_tolerated(self):
        rows = [{"phase": []}, {"phase": {}}, {"phase": "sse_event", "elapsed_seconds": .1,
                "type": [], "event_count": 3, "stream_bytes": 100},
                {"phase": "response_headers", "elapsed_seconds": .2, "content_type": {}}]
        with tempfile.TemporaryFile(mode="w+b") as channel:
            channel.write(("".join(json.dumps(row) + "\n" for row in rows) + '{"partial"').encode())
            metadata = transport._read_diagnostics(channel)
        self.assertEqual(metadata["event_count"], 3)
        self.assertEqual(metadata["stream_bytes"], 100)
        self.assertTrue(metadata["discarded_partial_record"])
        self.assertEqual(metadata["content_type"], "other")
        self.assertEqual(metadata["last_event"]["type"], "unknown_progress")

    def test_mock_https_connection_reports_connect_and_send_without_network(self):
        with tempfile.TemporaryFile(mode="w+b") as channel:
            diagnostics = transport._Diagnostics(channel.fileno())
            handler = transport._ObservedHTTPSHandler(diagnostics)
            def do_open(factory, request, **kwargs):
                connection = factory("offline.invalid")
                connection.connect()
                connection.request("POST", "/responses", body=b"not-real-request")
                return "mock-response"
            with patch.object(handler, "do_open", side_effect=do_open), \
                    patch.object(transport.http.client.HTTPSConnection, "connect"), \
                    patch.object(transport.http.client.HTTPSConnection, "request"):
                self.assertEqual(handler.https_open(object()), "mock-response")
            metadata = transport._read_diagnostics(channel)
        self.assertEqual(set(metadata["phase_timings"]), {"connect_started", "connected", "request_sent"})
        self.assertEqual(metadata["wait_stage"], "response_headers")


if __name__ == "__main__":
    unittest.main()
