"""Model clients and response parsing.

The search loop does not call a model API directly. Each candidate is a pair of
files: ``<candidate>.prompt.md`` written by the harness and
``<candidate>.response.md`` written by the model. ``<candidate>.meta.json``
records which model answered and the token usage that call reported. An
in-process client fills those files itself; a Cursor subagent session fills
them from outside the process.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)


@dataclass
class Completion:
    text: str
    provider: str
    model: str
    usage: dict | None = None
    usage_raw: dict = field(default_factory=dict)
    latency_s: float | None = None
    extra: dict = field(default_factory=dict)

    def meta(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "usage": self.usage,
            "usage_raw": self.usage_raw,
            "latency_s": self.latency_s,
            **self.extra,
        }


def split_program(text: str) -> tuple[str, str]:
    """Return ``(code, error)``. ``error`` is empty when a program was found."""
    blocks = [(tag.strip().lower(), body.strip()) for tag, body in _FENCE.findall(text)]
    preferred = [body for tag, body in blocks if tag in {"", "python", "py"}]
    ordered = preferred or [body for _, body in blocks]
    for body in ordered:
        if "def solve" in body:
            return body, ""
    if ordered:
        return ordered[-1], ""
    if "def solve" in text:
        return text.strip(), ""
    return "", "model response did not contain a Python program"


def normalize_usage(raw) -> dict | None:
    """Map a provider usage object onto the ledger fields, or None if absent."""
    if raw is None:
        return None
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    if not isinstance(raw, dict) or not raw:
        return None
    details_in = raw.get("input_tokens_details") or {}
    details_out = raw.get("output_tokens_details") or {}
    input_tokens = int(raw.get("input_tokens") or raw.get("prompt_tokens") or 0)
    output_tokens = int(raw.get("output_tokens") or raw.get("completion_tokens") or 0)
    cached = int(
        details_in.get("cached_tokens")
        or raw.get("cached_tokens")
        or raw.get("cache_read_input_tokens")
        or 0
    )
    reasoning = int(details_out.get("reasoning_tokens") or raw.get("reasoning_tokens") or 0)
    total = int(raw.get("total_tokens") or (input_tokens + output_tokens))
    return {
        "input_tokens": input_tokens,
        "cached_tokens": cached,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning,
        "total_tokens": total,
    }


def read_meta(path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return data if isinstance(data, dict) else {}


class StubClient:
    """Deterministic stand-in for tests. Its traces are not model data."""

    provider = "stub"

    def __init__(self, programs: list[str], usage: bool = True):
        if not programs:
            raise ValueError("stub client needs at least one program")
        self.programs = programs
        self.with_usage = usage
        self.calls = 0
        self.prompts: list[str] = []

    def complete(self, *, prompt: str, factors: dict) -> Completion:
        self.calls += 1
        self.prompts.append(prompt)
        program = self.programs[min(self.calls, len(self.programs)) - 1]
        usage = None
        if self.with_usage:
            usage = {
                "input_tokens": 100 + self.calls,
                "cached_tokens": 0,
                "output_tokens": 50,
                "reasoning_tokens": 10,
                "total_tokens": 150 + self.calls,
            }
        return Completion(
            text=f"```python\n{program}\n```",
            provider=self.provider,
            model=f"stub-{factors['reasoning_effort']}",
            usage=usage,
            usage_raw=dict(usage or {}),
            latency_s=0.0,
        )
