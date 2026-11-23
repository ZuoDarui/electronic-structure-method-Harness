"""Atomic campaign state and immutable scientific-evidence inventories."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def load_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


@contextmanager
def campaign_lock(root):
    """An OS-released lock survives controller crashes without a stale PID file."""
    import fcntl

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".controller.lock").open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another controller owns this campaign") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def event(root, kind, **fields):
    record = {"time": utcnow(), "event": kind, **fields}
    content = json.dumps(record, sort_keys=True, allow_nan=False, ensure_ascii=False)
    with (Path(root) / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(content + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(f"[{record['time']}] {kind}: " + json.dumps(fields, ensure_ascii=False), flush=True)


def inventory(paths):
    return {str(Path(path).resolve()): digest(path) for path in paths}


def verify_inventory(expected):
    mismatches = []
    for name, sha in expected.items():
        path = Path(name)
        if not path.is_file() or path.is_symlink() or digest(path) != sha:
            mismatches.append(str(name))
    if mismatches:
        raise RuntimeError("Evidence identity changed: " + ", ".join(mismatches))


def write_manifest(root):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Unexpected evidence symlink: {path}")
        if path.is_file() and path.name not in {"manifest.json", ".controller.lock"}:
            files[str(path.relative_to(root))] = {"sha256": digest(path), "bytes": path.stat().st_size}
    result = {"schema_version": "campaign-evidence/1", "created_at": utcnow(),
              "meaning": "Integrity inventory; trust requires storage outside candidate write access.",
              "files": files}
    atomic_json(root / "manifest.json", result)
    return result
