"""Claude (Anthropic Messages API) candidate generator.

Drop-in alternative to ``ResponsesGenerator``: same ``generate(context, dir)``
contract, same evidence layout (``attempt-NN/{request,response,status}.json``
plus ``bundle.json``), same ``validate_bundle`` AST/import guard. One call to
``generate`` produces at most one program; the SDK's own transport retries
(429/5xx/connection) re-send the same request and never create a second one.

Credentials come only from an environment variable (default
``ANTHROPIC_API_KEY``) and are never written to evidence files.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time

from .generator import (CONTRACT, MAX_CONTEXT_BYTES, SYSTEM_PROMPT, GenerationError,
                        _atomic_json, validate_bundle)


DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_MAX_TOKENS = 64000
EFFORTS = ("low", "medium", "high", "xhigh", "max")
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Structured-output schema. Claude's JSON outputs reject array-length
# constraints, so the single file is a plain string field; validate_bundle
# then enforces the same one-file / AST rules as the Responses path.
CLAUDE_SCHEMA = {
    "type": "object",
    "properties": {
        "solution_py": {"type": "string", "description": "Complete contents of solution.py"},
        "rationale": {"type": "string"},
        "expected_effect": {"type": "string"},
    },
    "required": ["solution_py", "rationale", "expected_effect"],
    "additionalProperties": False,
}


class ClaudeGenerator:
    """Bounded single-request generation through the official Anthropic SDK."""

    def __init__(self, config: dict):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise GenerationError("config_error", "Install the 'anthropic' package to use provider=anthropic") from exc
        self._anthropic = anthropic
        self.config = dict(config)
        self.model = config.get("model") or DEFAULT_MODEL
        if not isinstance(self.model, str) or not self.model.strip():
            raise GenerationError("config_error", "model must be a nonempty string")
        key_env = config.get("api_key_env") or "ANTHROPIC_API_KEY"
        self._api_key = os.environ.get(key_env)
        if not self._api_key:
            raise GenerationError("config_error", f"Required API credential environment variable {key_env} is missing")
        self.max_tokens = config.get("max_output_tokens", DEFAULT_MAX_TOKENS)
        if type(self.max_tokens) is not int or not 1024 <= self.max_tokens <= 128000:
            raise GenerationError("config_error", "max_output_tokens must be an integer in [1024, 128000]")
        self.effort = config.get("reasoning_effort") or "high"
        if self.effort not in EFFORTS:
            raise GenerationError("config_error", f"reasoning_effort must be one of {EFFORTS}")
        self.timeout_s = config.get("timeout_s", 1200)
        if isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, (int, float)) \
                or not math.isfinite(self.timeout_s) or not 0 < self.timeout_s <= 3600:
            raise GenerationError("config_error", "timeout_s must be in (0, 3600]")
        self.max_retries = config.get("max_retries", 2)
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= 5:
            raise GenerationError("config_error", "max_retries must be an integer in [0, 5]")
        self.fallbacks = config.get("refusal_fallback", True)
        if type(self.fallbacks) is not bool:
            raise GenerationError("config_error", "refusal_fallback must be boolean")
        base_url = config.get("base_url")
        if base_url is not None and (not isinstance(base_url, str) or not base_url.startswith("https://")):
            raise GenerationError("config_error", "base_url must be an HTTPS URL")
        client_kwargs = {"api_key": self._api_key, "timeout": float(self.timeout_s), "max_retries": self.max_retries}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = config.get("client") or anthropic.Anthropic(**client_kwargs)

    def _safe(self, value):
        if isinstance(value, str):
            return value.replace(self._api_key, "[REDACTED]")
        if isinstance(value, list):
            return [self._safe(v) for v in value]
        if isinstance(value, dict):
            return {self._safe(str(k)): self._safe(v) for k, v in value.items()}
        return value

    def _request(self, context: dict) -> dict:
        if not isinstance(context, dict):
            raise GenerationError("context_error", "Generation context must be a mapping")
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
        system = (SYSTEM_PROMPT + "\n" + CONTRACT +
                  "\nReturn a JSON object with solution_py (the complete solution.py source), "
                  "rationale and expected_effect.\n")
        request = {
            "model": self.model, "max_tokens": self.max_tokens, "system": system,
            "messages": [{"role": "user", "content": content}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort,
                              "format": {"type": "json_schema", "schema": CLAUDE_SCHEMA}},
        }
        if self.fallbacks:
            # Re-run a policy decline server-side on Anthropic's recommended model.
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"
        return request

    def _extract(self, message: dict) -> dict:
        usage = message.get("usage") or {}
        stop = message.get("stop_reason")
        if stop == "refusal":
            details = message.get("stop_details") or {}
            raise GenerationError("refusal", f"Model declined (category={details.get('category')})", usage)
        if stop == "max_tokens":
            raise GenerationError("output_budget_exceeded",
                                  f"Response hit max_tokens={self.max_tokens} before completing", usage)
        if stop not in {"end_turn", "stop_sequence"}:
            raise GenerationError("incomplete", f"Unexpected stop_reason {stop!r}", usage)
        texts = [block.get("text") for block in message.get("content", []) if block.get("type") == "text"]
        if len(texts) != 1 or not isinstance(texts[0], str):
            raise GenerationError("invalid_response", "Expected exactly one JSON text block", usage)
        if self._api_key in texts[0]:
            raise GenerationError("credential_content", "Credential detected in generated output", usage)
        try:
            data = json.loads(texts[0])
        except ValueError as exc:
            raise GenerationError("invalid_json", "Structured output is not valid JSON", usage) from exc
        if not isinstance(data, dict) or set(data) != {"solution_py", "rationale", "expected_effect"}:
            raise GenerationError("invalid_bundle", "Structured output fields do not match the schema", usage)
        bundle = {"files": [{"path": "solution.py", "content": data["solution_py"]}],
                  "rationale": data["rationale"], "expected_effect": data["expected_effect"]}
        try:
            return validate_bundle(bundle)
        except GenerationError as exc:
            exc.usage = usage
            raise

    def _call(self, request: dict) -> dict:
        kwargs = dict(request)
        api = self.client.beta.messages if "betas" in kwargs else self.client.messages
        # Streaming keeps long adaptive-thinking turns under HTTP timeouts.
        with api.stream(**kwargs) as stream:
            return stream.get_final_message().to_dict()

    def generate(self, context: dict, output_dir: Path) -> dict:
        anthropic = self._anthropic
        request = self._request(context)
        root = Path(output_dir)
        if root.is_symlink():
            raise GenerationError("invalid_path", "Evidence output directory must not be a symlink")
        directory = root / "attempt-01"
        directory.mkdir(parents=True, exist_ok=False)
        _atomic_json(directory / "request.json", self._safe({**request, "provider": "anthropic"}))
        started = time.monotonic()
        message, usage = None, {}
        try:
            try:
                message = self._call(request)
            except anthropic.AuthenticationError as exc:
                raise GenerationError("authentication_error", "Anthropic API rejected the credential") from exc
            except anthropic.PermissionDeniedError as exc:
                raise GenerationError("authentication_error", f"Permission denied: {exc.message}") from exc
            except anthropic.NotFoundError as exc:
                raise GenerationError("model_unavailable", f"Model or endpoint not found: {exc.message}") from exc
            except anthropic.BadRequestError as exc:
                raise GenerationError("api_error", f"Bad request: {exc.message}") from exc
            except anthropic.RateLimitError as exc:
                raise GenerationError("http_error", "Rate limited after SDK retries") from exc
            except anthropic.APIStatusError as exc:
                raise GenerationError("http_error", f"Anthropic API HTTP {exc.status_code}") from exc
            except anthropic.APIConnectionError as exc:
                raise GenerationError("network_error", f"Connection failed: {type(exc).__name__}") from exc
            _atomic_json(directory / "response.json", self._safe(message))
            usage = message.get("usage") or {}
            bundle = self._extract(message)
        except GenerationError as error:
            error.reason = self._safe(error.reason)
            error.args = (error.reason,)
            if message is None and not (directory / "response.json").exists():
                _atomic_json(directory / "response.json", {"received": False, "reason": error.status})
            _atomic_json(directory / "status.json", self._safe({
                "status": error.status, "reason": error.reason, "attempt": 1, "usage": error.usage or usage,
                "model": self.model, "effort": self.effort, "elapsed_seconds": time.monotonic() - started}))
            raise
        _atomic_json(directory / "status.json", self._safe({
            "status": "completed", "attempt": 1, "usage": usage, "model": message.get("model"),
            "requested_model": self.model, "effort": self.effort, "response_id": message.get("id"),
            "stop_reason": message.get("stop_reason"), "elapsed_seconds": time.monotonic() - started}))
        _atomic_json(root / "bundle.json", bundle)
        return bundle
