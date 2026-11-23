"""Responses API code generation with recorded, bounded, validated patches.

The AST policy is an early error and import guard, not a security sandbox.
Generated code must still run in the separately restricted candidate worker.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable
import urllib.error
import urllib.parse


MAX_OUTPUT_TOKENS = 16384
BUDGET_PROFILES = {"standard": (MAX_OUTPUT_TOKENS, 600), "extended": (32768, 1200)}
MAX_SOURCE_BYTES = 512_000
MAX_RESPONSE_BYTES = 2_000_000
MAX_CONTEXT_BYTES = 1_000_000
ALLOWED_IMPORTS = frozenset({
    "numpy", "scipy", "cupy", "math", "collections", "functools", "itertools",
    "dataclasses", "typing", "numbers", "operator", "warnings", "__future__",
})
FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "__import__", "open", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
})
FORBIDDEN_ATTRIBUTES = frozenset({
    "load", "save", "savez", "savez_compressed", "loadtxt", "savetxt",
    "fromfile", "tofile", "memmap", "ctypeslib", "ctypes", "f2py",
    "RawKernel", "RawModule", "ElementwiseKernel", "ReductionKernel",
})

CONTRACT = """Implement exactly these public callables in solution.py:
build_jk(problem, density, backend) -> (J, K)
solve(problem, options, backend) -> dict
backend is the string 'numpy' or 'cupy'; select the corresponding array library.
problem contains overlap and hcore (float64 AO matrices), enuc (float),
nelectron (positive even int), metadata (dict including integral mode), and
either eri (exact four-index tensor) or df_factors (naux,nao,nao).
The density is real symmetric and spin-summed: D=2*Cocc*Cocc.T.
F=H+J-K/2; E=Tr[D(H+F)]/2+enuc. Respect nonorthogonal overlap.
options contains conv_tol, residual_tol, max_cycle, diis_space, df_block_size,
and memory_limit_mb. Return energy, density, mo_coeff, mo_energy, occupations,
converged, and history; each history row has cycle, energy, residual_norm.
Implement SCF and contractions yourself. No other QC solver may be called.
Keep large numerical arrays on the requested backend. Return honest convergence.
"""

SYSTEM_PROMPT = """You implement a restricted Hartree--Fock numerical candidate.
Produce one complete solution.py as a strict JSON patch bundle. Only solution.py
may change; do not alter tests, tasks, validators, references or acceptance rules.
Use the public specification, current source and development feedback supplied.
Do not infer hidden tests or claim measurements that were not run.
Explain the change briefly using rationale and expected_effect, not private reasoning.
A compact standard Roothaan SCF with Pulay DIIS is sufficient. The supplied
integrals already define the basis and Hamiltonian: do not implement parsers,
optional algorithms or an unrelated framework. Prioritize complete working code.
Allowed imports: numpy, scipy, cupy, math, collections, functools, itertools,
dataclasses, typing, numbers, operator, warnings and __future__.
No file, process, network, dynamic import, external QC, or environment access.
The import/AST checks are guards; the worker separately restricts execution.
"""

PATCH_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "files": {"type": "array", "minItems": 1, "maxItems": 1,
                  "items": {"type": "object", "additionalProperties": False,
                            "properties": {"path": {"type": "string", "enum": ["solution.py"]},
                                           "content": {"type": "string"}},
                            "required": ["path", "content"]}},
        "rationale": {"type": "string"},
        "expected_effect": {"type": "string"},
    },
    "required": ["files", "rationale", "expected_effect"],
}


class GenerationError(RuntimeError):
    def __init__(self, status: str, reason: str, usage: dict | None = None):
        super().__init__(reason)
        self.status, self.reason, self.usage = status, reason, usage or {}


def validate_budget(config: dict) -> tuple[str, int, float]:
    """Keep original defaults; larger limits require an explicit frozen profile."""
    profile = config.get("budget_profile", "standard")
    if not isinstance(profile, str) or profile not in BUDGET_PROFILES:
        raise GenerationError("config_error", "budget_profile must be standard or extended")
    token_cap, timeout_cap = BUDGET_PROFILES[profile]
    tokens = config.get("max_output_tokens", MAX_OUTPUT_TOKENS)
    timeout = config.get("timeout_s", 120.0)
    if type(tokens) is not int or not 1 <= tokens <= token_cap:
        raise GenerationError("config_error", f"max_output_tokens must be in [1,{token_cap}] for {profile}")
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or not 0 < timeout <= timeout_cap):
        raise GenerationError("config_error", f"timeout_s must be finite and within (0,{timeout_cap}] for {profile}")
    return profile, tokens, timeout


def initial_source() -> str:
    return ('"""Generated RHF candidate: implement the two public functions."""\n'
            '\n\ndef build_jk(problem, density, backend):\n'
            '    raise NotImplementedError("build_jk has not been generated")\n'
            '\n\ndef solve(problem, options, backend):\n'
            '    raise NotImplementedError("solve has not been generated")\n')


def _json_object(text: str):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def validate_bundle(bundle: Any) -> dict:
    """Validate schema, the sole relative target, syntax and public interfaces."""
    if not isinstance(bundle, dict) or set(bundle) != {"files", "rationale", "expected_effect"}:
        raise GenerationError("invalid_bundle", "PatchBundle fields must be files, rationale and expected_effect")
    for key in ("rationale", "expected_effect"):
        if not isinstance(bundle[key], str) or len(bundle[key].encode("utf-8")) > 32_000:
            raise GenerationError("invalid_bundle", f"{key} must be a bounded string")
    files = bundle["files"]
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
        raise GenerationError("invalid_bundle", "Exactly one source file is required")
    file = files[0]
    if set(file) != {"path", "content"} or file.get("path") != "solution.py":
        raise GenerationError("invalid_path", "Only the relative path solution.py is allowed")
    source = file["content"]
    if not isinstance(source, str) or not source.strip() or len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise GenerationError("invalid_source", "Source must be nonempty and within the byte limit")
    try:
        tree = ast.parse(source, filename="solution.py")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise GenerationError("syntax_error", f"Invalid Python syntax: {exc}") from exc
    definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name, arguments in (("build_jk", ["problem", "density", "backend"]),
                            ("solve", ["problem", "options", "backend"])):
        node = definitions.get(name)
        if node is None:
            raise GenerationError("interface_error", f"Missing top-level function {name}")
        names = [arg.arg for arg in (*node.args.posonlyargs, *node.args.args)]
        if names != arguments or node.args.vararg or node.args.kwarg or node.args.kwonlyargs:
            raise GenerationError("interface_error", f"{name} requires arguments {arguments}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom):
                roots = [str(node.module).split(".")[0]]
                if node.level:
                    raise GenerationError("forbidden_import", "Relative imports are not allowed")
            else:
                roots = [alias.name.split(".")[0] for alias in node.names]
            if any(root not in ALLOWED_IMPORTS for root in roots):
                raise GenerationError("forbidden_import", "Source imports a module outside the numerical allowlist")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise GenerationError("forbidden_operation", f"Forbidden dynamic or external operation: {node.id}")
        if isinstance(node, ast.Attribute) and (
                node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRIBUTES):
            raise GenerationError("forbidden_operation", f"Forbidden attribute: {node.attr}")
    return {"files": [{"path": "solution.py", "content": source}],
            "rationale": bundle["rationale"], "expected_effect": bundle["expected_effect"]}


def _atomic_json(path: Path, value: dict):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def apply_bundle(bundle: dict, candidate_dir: Path) -> dict:
    """Atomically replace the sole allowed source after all validation passes.

    A failed validation or replacement preserves the previous source. Root and
    target symlinks are refused. The controller owns snapshots and rollback
    between completed rounds, and must not share this directory with a writer.
    """
    checked = validate_bundle(bundle)
    root = Path(candidate_dir)
    if root.is_symlink():
        raise GenerationError("invalid_path", "Candidate directory must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    target = root / "solution.py"
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise GenerationError("invalid_path", "solution.py must be a regular file, not a symlink")
    before = target.read_bytes() if target.exists() else None
    after = checked["files"][0]["content"].encode("utf-8")
    try:
        diff = "".join(difflib.unified_diff(
            (before or b"").decode("utf-8").splitlines(keepends=True),
            after.decode("utf-8").splitlines(keepends=True),
            fromfile="previous/solution.py", tofile="candidate/solution.py"))
    except UnicodeError as exc:
        raise GenerationError("apply_error", "Previous source is not valid UTF-8") from exc
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=root, prefix=".solution.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(after)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except OSError as exc:
        raise GenerationError("apply_error", f"Atomic source replacement failed: {type(exc).__name__}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"applied": True, "files": [{"path": "solution.py",
            "parent_sha256": hashlib.sha256(before).hexdigest() if before is not None else None,
            "sha256": hashlib.sha256(after).hexdigest()}], "diff": diff}


def _http_transport(request: dict, *, api_key: str, timeout_s: float,
                    url: str) -> dict:
    from .responses_streaming import SSETransportError, json_transport, streaming_transport
    # Both wire formats need a parent-enforced total deadline. A socket
    # operation timeout alone permits a slowly progressing body to overrun it.
    transport = streaming_transport if request.get("stream") is True else json_transport
    try:
        return transport(request, api_key=api_key, timeout_s=timeout_s, url=url)
    except SSETransportError as exc:
        error = GenerationError(exc.status, exc.reason, exc.usage)
        error.transport_metadata = exc.metadata
        raise error from None


class ResponsesGenerator:
    """Bounded Responses calls; no credentials are written to evidence files."""

    def __init__(self, config: dict):
        self.config = dict(config)
        self.model = config.get("model") or os.environ.get("OPENAI_MODEL")
        key_env = config.get("api_key_env", "OPENAI_API_KEY")
        if not isinstance(self.model, str) or not self.model.strip():
            raise GenerationError("config_error", "Set a model explicitly or provide OPENAI_MODEL")
        if not isinstance(key_env, str) or not key_env:
            raise GenerationError("config_error", "api_key_env must name an environment variable")
        self._api_key = os.environ.get(key_env)
        if not self._api_key:
            raise GenerationError("config_error", f"Required API credential environment variable {key_env} is missing")
        self.budget_profile, self.max_output_tokens, self.timeout_s = validate_budget(config)
        self.max_retries = config.get("max_retries", 2)
        self.streaming = config.get("streaming", False)
        if type(self.streaming) is not bool:
            raise GenerationError("config_error", "streaming must be a boolean")
        self.reasoning_effort = config.get("reasoning_effort")
        if self.reasoning_effort is not None and (
                not isinstance(self.reasoning_effort, str) or not self.reasoning_effort.strip()):
            raise GenerationError("config_error", "reasoning_effort must be an explicit nonempty string")
        if (isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int)
                or not 0 <= self.max_retries <= 2):
            raise GenerationError("config_error", "max_retries must be an integer in [0,2]")
        base_url = config.get("base_url", "https://api.openai.com/v1")
        if not isinstance(base_url, str):
            raise GenerationError("config_error", "base_url must be an HTTPS URL")
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise GenerationError("config_error", "base_url must be an HTTPS URL without credentials/query")
        self.url = base_url.rstrip("/") + "/responses"
        self.transport: Callable = config.get("transport") or _http_transport
        self.sleep: Callable = config.get("sleep") or time.sleep

    def _safe(self, value):
        if isinstance(value, str):
            return value.replace(self._api_key, "[REDACTED]")
        if isinstance(value, list):
            return [self._safe(v) for v in value]
        if isinstance(value, dict):
            return {self._safe(str(k)): self._safe(v) for k, v in value.items()}
        if isinstance(value, float) and not math.isfinite(value):
            # Real HTTP JSON rejects these already. Preserve an explicit marker
            # for malformed injected responses without writing invalid JSON.
            return f"[INVALID_NONFINITE:{value!r}]"
        return value

    def _validate_usage(self, response: dict) -> dict:
        """A provider's completed response cannot override the output budget.

        Output tokens include reasoning tokens. A smaller visible answer is
        insufficient when the reported total output exceeds the request cap.
        Missing or malformed accounting is never accepted as an in-budget call.
        """
        usage = response.get("usage")
        if usage is None and response.get("status") != "completed":
            return {}
        if not isinstance(usage, dict) or "output_tokens" not in usage:
            raise GenerationError("invalid_usage", "Response usage must include output_tokens",
                                  usage if isinstance(usage, dict) else {})

        def validate_counts(value, path="usage"):
            if isinstance(value, dict):
                for name, child in value.items():
                    if not isinstance(name, str):
                        raise GenerationError("invalid_usage", "Usage field names must be strings", usage)
                    validate_counts(child, f"{path}.{name}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    validate_counts(child, f"{path}[{index}]")
            elif type(value) is not int or value < 0:
                raise GenerationError("invalid_usage", f"{path} must be a nonnegative integer", usage)

        validate_counts(usage)
        reported = usage["output_tokens"]
        if type(reported) is not int:
            raise GenerationError("invalid_usage", "usage.output_tokens must be a nonnegative integer", usage)
        if reported > self.max_output_tokens:
            raise GenerationError(
                "output_budget_exceeded",
                f"Provider reported {reported} output tokens, exceeding requested max_output_tokens={self.max_output_tokens}",
                usage,
            )
        return usage

    def _budget_status(self, usage: dict) -> dict:
        reported = usage.get("output_tokens") if isinstance(usage, dict) else None
        valid = type(reported) is int and reported >= 0
        return {"budget_profile": self.budget_profile,
                "requested_timeout_seconds": self.timeout_s,
                "requested_max_output_tokens": self.max_output_tokens,
                "reported_output_tokens": self._safe(reported),
                "budget_within_limit": reported <= self.max_output_tokens if valid else None}

    def _request(self, context: dict) -> dict:
        if not isinstance(context, dict):
            raise GenerationError("context_error", "Generation context must be a mapping")
        # Only explicit public fields cross the model boundary. Unknown fields,
        # including accidentally supplied holdout results, are never serialized.
        public = {key: context[key] for key in (
            "public_prompt", "current_source", "dev_feedback", "contract", "round") if key in context}
        try:
            content = json.dumps(public, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise GenerationError("context_error", "Public context must be finite JSON data") from exc
        if len(content.encode("utf-8")) > MAX_CONTEXT_BYTES:
            raise GenerationError("context_error", "Public generation context exceeds byte limit")
        if self._api_key in content:
            raise GenerationError("context_error", "Credential detected in public context")
        budget_instruction = (
            f"\nThe total output budget for this request is {self.max_output_tokens} tokens, "
            "including all reasoning tokens and the visible JSON/code. Finish within "
            "that total budget. Prefer a compact, complete implementation with brief "
            "rationale and expected_effect; do not spend the budget exploring optional "
            "extensions or producing a verbose explanation. The controller rejects "
            "responses whose reported total output exceeds this limit.\n"
        )
        request = {"model": self.model, "store": False,
                "max_output_tokens": self.max_output_tokens,
                "input": [{"role": "system", "content": SYSTEM_PROMPT + "\n" + CONTRACT + budget_instruction},
                          {"role": "user", "content": content}],
                "text": {"format": {"type": "json_schema", "name": "rhf_patch_bundle",
                                    "strict": True, "schema": PATCH_SCHEMA}}}
        if self.reasoning_effort is not None:
            # Preserve the requested model/effort; provider rejections are
            # reported verbatim by status, never worked around by downgrading.
            request["reasoning"] = {"effort": self.reasoning_effort}
        if self.streaming:
            # Persist the actual wire flag before transport or evidence writes.
            request["stream"] = True
        return request

    def _extract(self, response: dict) -> dict:
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        if response.get("status") != "completed":
            raise GenerationError("incomplete", f"Response did not complete: {response.get('status')!r}", usage)
        if response.get("error"):
            raise GenerationError("api_error", "Response contains an API error", usage)
        chunks = []
        for item in response.get("output", []):
            if not isinstance(item, dict):
                raise GenerationError("invalid_response", "Response output item is not an object", usage)
            if item.get("type") == "message":
                if item.get("status", "completed") != "completed":
                    raise GenerationError("incomplete", "Output message is incomplete", usage)
                for part in item.get("content", []):
                    if part.get("type") == "refusal":
                        raise GenerationError("refusal", "Model refused to produce a patch", usage)
                    if part.get("type") == "output_text":
                        if not isinstance(part.get("text"), str):
                            raise GenerationError("invalid_response", "Output text must be a string", usage)
                        chunks.append(part["text"])
        if len(chunks) != 1:
            raise GenerationError("invalid_response", "Expected exactly one structured output text", usage)
        if self._api_key in chunks[0]:
            raise GenerationError("credential_content", "Credential detected in generated output", usage)
        try:
            bundle = _json_object(chunks[0])
        except (ValueError, TypeError) as exc:
            raise GenerationError("invalid_json", "Structured output is not valid JSON", usage) from exc
        try:
            return validate_bundle(bundle)
        except GenerationError as exc:
            exc.usage = usage
            raise

    def generate(self, context: dict, output_dir: Path) -> dict:
        request = self._request(context)
        root = Path(output_dir)
        if root.is_symlink():
            raise GenerationError("invalid_path", "Evidence output directory must not be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, self.max_retries + 2):
            directory = root / f"attempt-{attempt:02d}"
            directory.mkdir(exist_ok=False)
            _atomic_json(directory / "request.json", self._safe(request))
            started = time.monotonic()
            response = None
            usage = {}
            try:
                response = self.transport(request, api_key=self._api_key,
                                          timeout_s=self.timeout_s, url=self.url)
                if not isinstance(response, dict):
                    raise GenerationError("invalid_response", "Transport did not return a response object")
                encoded = json.dumps(response, allow_nan=True).encode("utf-8")
                if len(encoded) > MAX_RESPONSE_BYTES:
                    _atomic_json(directory / "response.json", self._safe({
                        "truncated": True,
                        "raw_prefix": encoded[:MAX_RESPONSE_BYTES].decode("utf-8", errors="replace")}))
                    raise GenerationError("response_too_large", "API response exceeds byte limit")
                _atomic_json(directory / "response.json", self._safe(response))
                if response.get("_transport_error"):
                    raise GenerationError(response["_transport_error"], "API returned invalid or oversized response data")
                usage = self._validate_usage(response)
                bundle = self._extract(response)
                _atomic_json(directory / "status.json", self._safe({
                    "status": "completed", "attempt": attempt, "usage": usage,
                    "response_id": response.get("id"), "elapsed_seconds": time.monotonic() - started,
                    "transport_metadata": response.get("_transport_metadata", {}),
                    **self._budget_status(usage)}))
                _atomic_json(root / "bundle.json", self._safe(bundle))
                return bundle
            except Exception as exc:
                retry = isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError))
                http_status = exc.code if isinstance(exc, urllib.error.HTTPError) else None
                if http_status is not None:
                    retry = http_status == 429 or 500 <= http_status <= 599
                if isinstance(exc, GenerationError):
                    error = exc
                    usage = exc.usage
                    retry = False
                elif isinstance(exc, urllib.error.HTTPError):
                    error = GenerationError("http_error", f"Responses API HTTP {http_status}", usage)
                    try:
                        raw_error = exc.read(MAX_RESPONSE_BYTES + 1)
                        _atomic_json(directory / "response.json", self._safe({
                            "http_status": http_status,
                            "body": raw_error[:MAX_RESPONSE_BYTES].decode("utf-8", errors="replace"),
                            "truncated": len(raw_error) > MAX_RESPONSE_BYTES}))
                    except (OSError, ValueError):
                        pass
                elif retry:
                    error = GenerationError("network_error", f"Responses transport failed: {type(exc).__name__}", usage)
                else:
                    error = GenerationError("invalid_response", f"Responses processing failed: {type(exc).__name__}", usage)
                # Streaming errors keep their normal retry classes, while the
                # disposable transport attaches a bounded, content-free audit.
                # Preserve it when converting the exception for the campaign.
                if hasattr(exc, "transport_metadata"):
                    error.transport_metadata = self._safe(exc.transport_metadata)
                error.reason = self._safe(error.reason)
                error.args = (error.reason,)
                error.usage = self._safe(usage)
                if not (directory / "response.json").exists():
                    _atomic_json(directory / "response.json", self._safe({
                        "received": False, "reason": error.status,
                        "transport_metadata": getattr(error, "transport_metadata", {})}))
                _atomic_json(directory / "status.json", self._safe({
                    "status": error.status, "reason": error.reason, "attempt": attempt,
                    "retry": bool(retry and attempt <= self.max_retries), "usage": usage,
                    "http_status": http_status, "elapsed_seconds": time.monotonic() - started,
                    "transport_metadata": getattr(error, "transport_metadata", {}),
                    **self._budget_status(usage)}))
                if not retry or attempt > self.max_retries:
                    # Provider exception text may contain its URL, headers or
                    # an echoed credential. Only the sanitized error is public.
                    raise error from None
                self.sleep(min(2 ** (attempt - 1), 4))
            except (KeyboardInterrupt, SystemExit) as exc:
                # The transport has already stopped its child and collected
                # diagnostic progress. Archive only that safe audit, then keep
                # the caller's cancellation semantics intact.
                try:
                    metadata = self._safe(getattr(exc, "transport_metadata", {}))
                    if not (directory / "response.json").exists():
                        _atomic_json(directory / "response.json", {
                            "received": False, "reason": "cancelled",
                            "transport_metadata": metadata})
                    _atomic_json(directory / "status.json", {
                        "status": "cancelled", "reason": type(exc).__name__,
                        "attempt": attempt, "retry": False, "usage": {},
                        "http_status": metadata.get("http_status") if isinstance(metadata, dict) else None,
                        "elapsed_seconds": time.monotonic() - started,
                        "transport_metadata": metadata,
                        **self._budget_status({})})
                except Exception:
                    # A full/unwritable evidence disk cannot turn a cancellation
                    # into a processing error or trigger an automatic retry.
                    pass
                raise
        raise AssertionError("unreachable retry state")
