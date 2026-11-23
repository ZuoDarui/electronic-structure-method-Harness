"""Bounded Responses JSON/SSE transport. No model or budget substitution.

For SSE, only response.completed supplies the complete Response. JSON passes
the bounded object to the generator for status, usage, cap and bundle checks.
Network I/O runs in a disposable subprocess: the parent enforces one absolute
deadline across DNS, headers and body, and kills that process group on expiry.
Credentials travel to the worker through stdin, never argv, files or its env.
"""
from __future__ import annotations

import io
import http.client
import base64
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

MAX_STREAM_BYTES = 4_000_000
MAX_RESPONSE_BYTES = 2_000_000
MAX_FRAME_BYTES = 2_100_000
MAX_EVENTS = 10_000
MAX_REQUEST_BYTES = 1_100_000
# A bounded 2 MB failure prefix may expand sixfold when JSON-escaped. This is
# only the IPC envelope bound; raw/final response limits remain 2 MB.
MAX_WORKER_BYTES = 16_000_000
MAX_DIAGNOSTIC_BYTES = 2 * 1024 * 1024

SAFE_EVENT_TYPES = frozenset({
    "error", "response.created", "response.queued", "response.in_progress",
    "response.completed", "response.failed", "response.incomplete",
    "response.output_item.added", "response.output_item.done",
    "response.content_part.added", "response.content_part.done",
    "response.output_text.delta", "response.output_text.done",
    "response.refusal.delta", "response.refusal.done",
    "response.reasoning_text.delta", "response.reasoning_text.done",
    "response.reasoning_summary_part.added", "response.reasoning_summary_part.done",
    "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done",
})
SAFE_PHASES = frozenset({
    "worker_started", "request_ready", "connect_started", "connected",
    "request_sent", "response_headers", "first_byte", "bytes_progress",
    "sse_event", "response_complete", "body_complete", "transport_error", "diagnostics_truncated",
})


def _safe_event_type(value):
    return value if isinstance(value, str) and value in SAFE_EVENT_TYPES else "unknown_progress"


class _Diagnostics:
    """Append-only anonymous FD channel; never accept arbitrary payload fields."""

    def __init__(self, descriptor=None):
        self.descriptor = descriptor
        self.started, self.written, self.stopped = time.monotonic(), 0, False

    def emit(self, phase, **fields):
        if self.descriptor is None or self.stopped or not isinstance(phase, str) or phase not in SAFE_PHASES:
            return
        record = {"phase": phase, "elapsed_seconds": round(max(0.0, time.monotonic() - self.started), 6)}
        for key in ("stream_bytes", "event_count", "sequence_number", "http_status"):
            value = fields.get(key)
            if type(value) is int and 0 <= value <= 2**63 - 1:
                record[key] = value
        if "type" in fields:
            record["type"] = _safe_event_type(fields["type"])
        if "content_type" in fields:
            value = fields["content_type"]
            record["content_type"] = value if isinstance(value, str) and value in {"text/event-stream", "application/json"} else "other"
        data = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
        if self.written + len(data) > MAX_DIAGNOSTIC_BYTES - 256:
            data = b'{"phase":"diagnostics_truncated"}\n'
            self.stopped = True
        try:
            while data:
                written = os.write(self.descriptor, data)
                if written <= 0:
                    self.stopped = True
                    return
                self.written += written
                data = data[written:]
        except OSError:
            self.stopped = True


def _read_diagnostics(stream):
    """Recover complete safe records after success, timeout or cancellation."""
    stream.seek(0)
    data = stream.read(MAX_DIAGNOSTIC_BYTES + 1)
    summary = {"mode": "responses_sse", "diagnostic_records": 0,
               "diagnostics_truncated": len(data) > MAX_DIAGNOSTIC_BYTES,
               "discarded_partial_record": bool(data and not data.endswith(b"\n")),
               "phase": "worker_start_pending", "phase_timings": {},
               "stream_bytes": 0, "event_count": 0, "event_types": {},
               "first_event": None, "last_event": None}
    for raw in data[:MAX_DIAGNOSTIC_BYTES].splitlines(keepends=True):
        if not raw.endswith(b"\n"):
            continue
        try:
            record = _json_loads(raw.decode("utf-8"))
        except (ValueError, UnicodeError, RecursionError):
            continue
        if not isinstance(record, dict) or not isinstance(record.get("phase"), str) or record["phase"] not in SAFE_PHASES:
            continue
        phase = record["phase"]
        summary["diagnostic_records"] += 1
        if phase == "diagnostics_truncated":
            summary["diagnostics_truncated"] = True
            continue
        elapsed = record.get("elapsed_seconds")
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            continue
        summary["phase"] = phase
        summary["phase_timings"].setdefault(phase, elapsed)
        for key in ("stream_bytes", "event_count"):
            value = record.get(key)
            if type(value) is int and 0 <= value <= 2**63 - 1:
                summary[key] = max(summary[key], value)
        status = record.get("http_status")
        if type(status) is int and 100 <= status <= 599:
            summary["http_status"] = status
        if "content_type" in record:
            content_type = record["content_type"]
            summary["content_type"] = content_type if isinstance(content_type, str) and content_type in {"text/event-stream", "application/json"} else "other"
        if phase == "sse_event":
            kind = _safe_event_type(record.get("type"))
            sequence = record.get("sequence_number")
            observed = {"type": kind, "elapsed_seconds": elapsed,
                        "sequence_number": sequence if type(sequence) is int and 0 <= sequence <= 2**63 - 1 else None}
            summary["event_types"][kind] = summary["event_types"].get(kind, 0) + 1
            if summary["first_event"] is None:
                summary["first_event"] = observed
            summary["last_event"] = observed
    timings = summary["phase_timings"]
    summary["wait_stage"] = (
        "completed" if "response_complete" in timings or "body_complete" in timings else
        "response_body" if "first_byte" in timings else
        "first_response_byte" if "response_headers" in timings else
        "response_headers" if "request_sent" in timings else
        "request_send" if "connected" in timings else
        "connect_dns_tcp_tls" if "connect_started" in timings else
        "worker_initialization"
    )
    return summary


class SSETransportError(RuntimeError):
    def __init__(self, status, reason, usage=None, metadata=None):
        super().__init__(reason)
        self.status, self.reason = status, reason
        self.usage, self.metadata = usage or {}, metadata or {}


def _json_loads(value):
    def pairs(items):
        result = {}
        for key, child in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = child
        return result
    def reject_constant(value):
        raise ValueError("nonfinite JSON number")
    return json.loads(value, object_pairs_hook=pairs, parse_constant=reject_constant)


def parse_sse(chunks, *, deadline, clock=time.monotonic, diagnostics=None):
    """Parse bounded byte chunks, returning only the final complete Response.

    Progress audit contains event type, sequence and elapsed time, never deltas,
    reasoning text, prompts, or headers. Unknown progress types are ignored.
    Text-done, [DONE], EOF and HTTP success are insufficient for completion.
    """
    started = clock()
    total, frames, last_sequence = 0, 0, None
    last_byte_report = started
    buffer, lines, frame_bytes = b"", [], 0
    events = []

    def metadata():
        return {"mode": "responses_sse", "stream_bytes": total, "event_count": frames,
                "elapsed_seconds": max(0.0, clock() - started), "events": events}

    def fail(status, reason, usage=None):
        raise SSETransportError(status, reason, usage, metadata())

    def frame(rows):
        nonlocal frames, last_sequence
        event_name, data = None, []
        for raw in rows:
            if raw.startswith(b":"):
                continue
            field, separator, value = raw.partition(b":")
            if separator and value.startswith(b" "):
                value = value[1:]
            if field == b"event":
                if event_name is not None:
                    fail("invalid_stream", "Duplicate SSE event name")
                try:
                    event_name = value.decode("utf-8")
                except UnicodeError:
                    fail("invalid_stream", "SSE event name is not UTF-8")
            elif field == b"data":
                data.append(value)
        if not data:
            return None
        frames += 1
        if frames > MAX_EVENTS:
            fail("stream_limit", "SSE event count exceeds limit")
        joined = b"\n".join(data)
        if joined == b"[DONE]":
            fail("incomplete", "SSE ended without response.completed")
        try:
            value = _json_loads(joined.decode("utf-8"))
        except (ValueError, UnicodeError, RecursionError):
            fail("invalid_stream", "SSE event data is not valid finite JSON")
        if not isinstance(value, dict):
            fail("invalid_stream", "SSE event data must be an object")
        kind = value.get("type")
        if not isinstance(kind, str) or not 1 <= len(kind) <= 128:
            fail("invalid_stream", "SSE event lacks a bounded type")
        if event_name is not None and event_name != kind:
            fail("invalid_stream", "SSE event name and JSON type disagree")
        sequence = value.get("sequence_number")
        if sequence is not None:
            if type(sequence) is not int or not 0 <= sequence <= 2**63 - 1 or (last_sequence is not None and sequence <= last_sequence):
                fail("invalid_stream", "SSE sequence number is invalid or non-increasing")
            last_sequence = sequence
        events.append({"type": _safe_event_type(kind), "sequence_number": sequence,
                       "elapsed_seconds": max(0.0, clock() - started)})
        if diagnostics is not None:
            diagnostics.emit("sse_event", type=kind, sequence_number=sequence,
                             event_count=frames, stream_bytes=total)
        terminal = value.get("response")
        usage = terminal.get("usage", {}) if isinstance(terminal, dict) else {}
        if kind == "error":
            fail("api_error", "Provider emitted a streaming error event", usage)
        if kind == "response.failed":
            fail("api_error", "Provider emitted response.failed", usage)
        if kind == "response.incomplete":
            fail("incomplete", "Provider emitted response.incomplete", usage)
        if "refusal" in kind:
            fail("refusal", "Provider emitted a refusal event", usage)
        if kind != "response.completed":
            return None
        if (not isinstance(terminal, dict) or terminal.get("status") != "completed"
                or terminal.get("error") or terminal.get("incomplete_details")):
            fail("incomplete", "Completed event lacks a successful complete Response", usage)
        if not isinstance(terminal.get("output"), list):
            fail("invalid_response", "Completed Response lacks its full output")
        if not isinstance(usage, dict) or "output_tokens" not in usage:
            fail("invalid_usage", "Completed Response lacks full usage accounting")
        for item in terminal["output"]:
            if not isinstance(item, dict):
                fail("invalid_response", "Completed output items must be objects", usage)
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("type") == "refusal":
                    fail("refusal", "Completed Response contains refusal", usage)
        try:
            encoded = json.dumps(terminal, allow_nan=False).encode("utf-8")
        except (ValueError, TypeError, RecursionError):
            fail("invalid_response", "Completed Response is not finite JSON", usage)
        if len(encoded) > MAX_RESPONSE_BYTES:
            fail("response_too_large", "Completed Response exceeds byte limit", usage)
        if "_transport_metadata" in terminal:
            fail("invalid_response", "Provider used a reserved transport field", usage)
        # Complete output and accounting are unchanged. The existing generator
        # owns output-cap and schema decisions, including output_budget_exceeded.
        result = dict(terminal)
        result["_transport_metadata"] = metadata()
        return result

    for chunk in chunks:
        if clock() >= deadline:
            fail("network_error", "Absolute streaming deadline exceeded")
        if not isinstance(chunk, bytes):
            fail("invalid_stream", "SSE reader must supply byte chunks")
        total += len(chunk)
        if total > MAX_STREAM_BYTES:
            fail("stream_limit", "SSE stream exceeds byte limit")
        now = clock()
        if diagnostics is not None and chunk:
            if total == len(chunk):
                diagnostics.emit("first_byte", stream_bytes=total)
            elif now - last_byte_report >= .5:
                diagnostics.emit("bytes_progress", stream_bytes=total, event_count=frames)
                last_byte_report = now
        buffer += chunk
        while b"\n" in buffer:
            raw, buffer = buffer.split(b"\n", 1)
            if raw.endswith(b"\r"):
                raw = raw[:-1]
            frame_bytes += len(raw) + 1
            if frame_bytes > MAX_FRAME_BYTES:
                fail("stream_limit", "SSE frame exceeds byte limit")
            if not raw:
                complete = frame(lines)
                lines, frame_bytes = [], 0
                if complete is not None:
                    if clock() >= deadline:
                        fail("network_error", "Absolute streaming deadline exceeded")
                    return complete
            else:
                lines.append(raw)
        if len(buffer) + frame_bytes > MAX_FRAME_BYTES:
            fail("stream_limit", "Unterminated SSE frame exceeds byte limit")
    fail("incomplete", "SSE EOF before a complete response.completed event")


def parse_json(chunks, *, deadline, clock=time.monotonic, diagnostics=None):
    """Read bounded JSON without treating HTTP success as model completion.

    Preserve the previous non-streaming transport's bounded raw failure prefix.
    The generator owns status, refusal, usage, output-cap and bundle validation.
    """
    parts, total, last_report = [], 0, clock()
    for chunk in chunks:
        if clock() >= deadline:
            raise SSETransportError("network_error", "Absolute JSON response deadline exceeded")
        if not isinstance(chunk, bytes):
            raise SSETransportError("invalid_response", "JSON reader must supply byte chunks")
        previous = total
        total += len(chunk)
        now = clock()
        if diagnostics is not None and chunk:
            if previous == 0:
                diagnostics.emit("first_byte", stream_bytes=total)
            elif now - last_report >= .5:
                diagnostics.emit("bytes_progress", stream_bytes=total)
                last_report = now
        parts.append(chunk[:max(0, MAX_RESPONSE_BYTES - previous)])
        if total > MAX_RESPONSE_BYTES:
            return {"_transport_error": "response_too_large",
                    "raw_prefix": b"".join(parts).decode("utf-8", errors="replace")}
    if clock() >= deadline:
        raise SSETransportError("network_error", "Absolute JSON response deadline exceeded")
    raw = b"".join(parts)
    if diagnostics is not None:
        diagnostics.emit("body_complete", stream_bytes=total)
    try:
        response = _json_loads(raw.decode("utf-8"))
        if not isinstance(response, dict) or "_transport_metadata" in response:
            raise ValueError("response must be an object without reserved metadata")
        return response
    except (ValueError, UnicodeError, RecursionError):
        return {"_transport_error": "invalid_response",
                "raw_prefix": raw.decode("utf-8", errors="replace")}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _ObservedHTTPSHandler(urllib.request.HTTPSHandler):
    """Observe connect (DNS/TCP/TLS) and request completion, without headers."""

    def __init__(self, diagnostics):
        super().__init__()
        self.diagnostics = diagnostics

    def https_open(self, request):
        diagnostics = self.diagnostics
        class Connection(http.client.HTTPSConnection):
            def connect(self):
                diagnostics.emit("connect_started")
                super().connect()
                diagnostics.emit("connected")
            def request(self, *args, **kwargs):
                super().request(*args, **kwargs)
                diagnostics.emit("request_sent")
        return self.do_open(Connection, request, context=self._context)


def _worker(diagnostic_fd=None):
    """Trusted disposable HTTP worker. Its only secret input is stdin JSON."""
    credential = ""
    mode = "sse"
    diagnostics = _Diagnostics(diagnostic_fd)
    diagnostics.emit("worker_started")
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise SSETransportError("context_error", "Streaming request exceeds byte limit")
        config = _json_loads(raw.decode("utf-8"))
        credential = config["api_key"]
        mode = config.get("transport_mode", "sse")
        if mode not in {"json", "sse"}:
            raise SSETransportError("config_error", "Unsupported Responses transport mode")
        diagnostics.emit("request_ready")
        started = time.monotonic()
        request = urllib.request.Request(config["url"], method="POST",
                    data=json.dumps(config["request"], allow_nan=False).encode("utf-8"),
                    headers={"Authorization": "Bearer " + credential,
                             "Content-Type": "application/json",
                             "Accept": "text/event-stream" if mode == "sse" else "application/json"})
        opener = urllib.request.build_opener(_NoRedirect(), urllib.request.ProxyHandler({}),
                                             _ObservedHTTPSHandler(diagnostics))
        with opener.open(request, timeout=config["timeout_s"]) as source:
            diagnostics.emit("response_headers", http_status=source.status,
                             content_type=source.headers.get_content_type())
            if mode == "sse" and source.headers.get_content_type() != "text/event-stream":
                raise SSETransportError("invalid_stream", "Streaming endpoint returned a non-SSE content type")
            def chunks():
                while True:
                    chunk = source.read1(16384)
                    if not chunk:
                        return
                    yield chunk
            reader = parse_sse if mode == "sse" else parse_json
            response = reader(chunks(), deadline=started + config["timeout_s"], diagnostics=diagnostics)
        if mode == "sse":
            diagnostics.emit("response_complete")
        # A failure prefix is diagnostic evidence, not executable source. The
        # generator would redact it before archiving; do so before IPC as well.
        if response.get("_transport_error") and isinstance(response.get("raw_prefix"), str):
            response["raw_prefix"] = response["raw_prefix"].replace(credential, "[REDACTED]")[:MAX_RESPONSE_BYTES]
        result = {"kind": "completed", "response": response}
    except urllib.error.HTTPError as exc:
        # SSE returns status only; JSON preserves bounded, redacted error text.
        diagnostics.emit("response_headers", http_status=exc.code,
                         content_type=exc.headers.get_content_type() if hasattr(exc.headers, "get_content_type") else "other")
        result = {"kind": "http_error", "code": exc.code}
        if mode == "json":
            # Keep the previous JSON path's bounded HTTP error-body evidence.
            try:
                body = exc.read(MAX_RESPONSE_BYTES + 1).replace(credential.encode("utf-8"), b"[REDACTED]")[:MAX_RESPONSE_BYTES + 1]
            except (OSError, ValueError):
                # The HTTP status already exists; an unavailable error body
                # must not erase it or change the established retry policy.
                body = b""
            result["body_base64"] = base64.b64encode(body).decode("ascii")
    except SSETransportError as exc:
        result = {"kind": "error", "status": exc.status, "reason": exc.reason,
                  "usage": exc.usage, "metadata": exc.metadata}
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        result = {"kind": "error", "status": "network_error",
                  "reason": "Streaming network operation failed: " + type(exc).__name__}
    except Exception as exc:
        result = {"kind": "error", "status": "invalid_stream",
                  "reason": "Streaming worker failed: " + type(exc).__name__}
    if result["kind"] != "completed":
        diagnostics.emit("transport_error")
    serialized = json.dumps(result, allow_nan=False, separators=(",", ":"))
    if credential and credential in serialized:
        serialized = json.dumps({"kind": "error", "status": "credential_content",
                                 "reason": "Provider response echoed a credential"})
    sys.stdout.write(serialized)
    sys.stdout.flush()


def _worker_command(diagnostic_fd):
    return [sys.executable, "-I", str(Path(__file__).resolve()), "--transport-worker", "--diagnostic-fd", str(diagnostic_fd)]


def _response_transport(request, *, api_key, timeout_s, url, mode):
    """Same call shape as the generator's injected transport; no retry here.

    Adapter integration should translate SSETransportError into GenerationError
    with the same status/reason/usage. HTTPError/URLError remain standard retry
    signals. The existing generator must validate the returned final usage.
    Callers own standard versus explicitly authorized extended timeout policy.
    """
    if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s) or not 0 < timeout_s <= 1200):
        raise SSETransportError("config_error", "Responses deadline must be in (0,1200] seconds")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SSETransportError("config_error", "Streaming URL must be HTTPS without embedded credentials or query")
    if not isinstance(api_key, str) or not api_key:
        raise SSETransportError("config_error", "Streaming API credential is missing")
    if not isinstance(request, dict):
        raise SSETransportError("context_error", "Streaming request must be an object")
    wire = dict(request)
    if mode == "sse":
        if "stream" in request and request["stream"] is not True:
            raise SSETransportError("config_error", "Streaming transport cannot accept stream=false")
        wire["stream"] = True
    elif mode == "json":
        if "stream" in request and request["stream"] is not False:
            raise SSETransportError("config_error", "JSON transport cannot accept a streaming request")
    else:
        raise SSETransportError("config_error", "Unsupported Responses transport mode")
    payload = json.dumps({"request": wire, "url": url, "api_key": api_key,
                          "timeout_s": timeout_s, "transport_mode": mode}, allow_nan=False).encode("utf-8")
    if len(payload) > MAX_REQUEST_BYTES:
        raise SSETransportError("context_error", "Streaming request exceeds byte limit")
    started = time.monotonic()
    environment = {"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}
    with tempfile.TemporaryFile(mode="w+b") as diagnostic_channel:
        descriptor = diagnostic_channel.fileno()
        process, interrupted = None, None
        try:
            process = subprocess.Popen(_worker_command(descriptor), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, close_fds=True, start_new_session=True,
                                       pass_fds=(descriptor,), env=environment)
            stdout, stderr = process.communicate(input=payload, timeout=max(0.001, timeout_s - (time.monotonic() - started)))
        except BaseException as exc:
            interrupted = exc
        finally:
            # The independent FD survives both timeout and caller cancellation;
            # kill first, then recover only complete diagnostic JSONL records.
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except BaseException as cleanup_error:
                    if interrupted is None:
                        interrupted = cleanup_error
                    try:
                        process.kill()
                    except OSError:
                        pass
                try:
                    process.communicate(timeout=5)
                except BaseException as cleanup_error:
                    if interrupted is None:
                        interrupted = cleanup_error
            try:
                diagnostic_metadata = _read_diagnostics(diagnostic_channel)
            except BaseException as diagnostic_error:
                diagnostic_metadata = {"mode": "responses_sse", "diagnostic_collection_failed": True}
                if interrupted is None and isinstance(diagnostic_error, (KeyboardInterrupt, SystemExit)):
                    interrupted = diagnostic_error
    diagnostic_metadata["transport_elapsed_seconds"] = time.monotonic() - started
    diagnostic_metadata["mode"] = "responses_" + mode

    def with_diagnostics(error):
        base = getattr(error, "metadata", {})
        combined = {**base, **diagnostic_metadata}
        for key in ("stream_bytes", "event_count"):
            values = [v for v in (base.get(key), diagnostic_metadata.get(key)) if type(v) is int and v >= 0]
            if values:
                combined[key] = max(values)
        error.transport_metadata = combined
        if isinstance(error, SSETransportError):
            error.metadata = combined
        return error

    if interrupted is not None:
        if isinstance(interrupted, subprocess.TimeoutExpired):
            raise with_diagnostics(urllib.error.URLError("Absolute streaming deadline exceeded")) from interrupted
        # Keep KeyboardInterrupt/SystemExit themselves, with safe diagnostics.
        raise with_diagnostics(interrupted)
    if time.monotonic() - started >= timeout_s:
        raise with_diagnostics(urllib.error.URLError("Absolute streaming deadline exceeded"))
    if process.returncode != 0 or len(stdout) > MAX_WORKER_BYTES:
        raise with_diagnostics(SSETransportError("invalid_stream", "Streaming worker exited without a bounded successful receipt"))
    try:
        result = _json_loads(stdout.decode("utf-8"))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise with_diagnostics(SSETransportError("invalid_stream", "Streaming worker returned invalid JSON")) from exc
    if not isinstance(result, dict):
        raise with_diagnostics(SSETransportError("invalid_stream", "Streaming worker returned a non-object receipt"))
    if result.get("kind") == "http_error":
        try:
            body = base64.b64decode(result.get("body_base64", ""), validate=True)
        except (TypeError, ValueError) as exc:
            raise with_diagnostics(SSETransportError("invalid_response", "HTTP error body receipt is malformed")) from exc
        if len(body) > MAX_RESPONSE_BYTES + 1:
            raise with_diagnostics(SSETransportError("invalid_response", "HTTP error body exceeds its bounded receipt"))
        raise with_diagnostics(urllib.error.HTTPError(url, result["code"], "Responses HTTP response error", {}, io.BytesIO(body)))
    if result.get("kind") == "error":
        if result.get("status") == "network_error":
            raise with_diagnostics(urllib.error.URLError(result["reason"]))
        raise with_diagnostics(SSETransportError(result["status"], result["reason"], result.get("usage"), result.get("metadata")))
    if result.get("kind") != "completed" or not isinstance(result.get("response"), dict):
        raise with_diagnostics(SSETransportError("invalid_stream", "Streaming worker did not return a complete Response"))
    response_metadata = result["response"].get("_transport_metadata", {})
    combined = {**response_metadata, **diagnostic_metadata}
    for key in ("stream_bytes", "event_count"):
        values = [v for v in (response_metadata.get(key), diagnostic_metadata.get(key)) if type(v) is int and v >= 0]
        if values:
            combined[key] = max(values)
    result["response"]["_transport_metadata"] = combined
    return result["response"]


def streaming_transport(request, *, api_key, timeout_s, url):
    """Receive SSE under the shared absolute deadline; return only completed."""
    return _response_transport(request, api_key=api_key, timeout_s=timeout_s, url=url, mode="sse")


def json_transport(request, *, api_key, timeout_s, url):
    """Receive ordinary JSON under the same absolute deadline and cleanup."""
    return _response_transport(request, api_key=api_key, timeout_s=timeout_s, url=url, mode="json")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1:3] != ["--transport-worker", "--diagnostic-fd"]:
        raise SystemExit("This module is an internal Responses streaming worker")
    descriptor = int(sys.argv[3])
    if descriptor < 3:
        raise SystemExit("Invalid diagnostic channel descriptor")
    _worker(descriptor)
