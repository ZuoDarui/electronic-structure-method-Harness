"""Locate framework resources separately from the existing HF programs.

Tasks, containers, tests and framework code live below ``FRAMEWORK_ROOT``.
The production solvers and molecular geometry helpers stay in ``HF_ROOT``;
examples import those files directly rather than copying an implementation.
"""
from __future__ import annotations

from pathlib import Path
import sys


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
HF_ROOT = FRAMEWORK_ROOT.parent
_PACKAGE_ROOT = FRAMEWORK_ROOT / "harness_engineering"


def configure_import_paths() -> None:
    """Prefer this framework and its parent HF files, rejecting stale imports.

    Moving a path to the front cannot replace a module already cached by
    Python. Fail explicitly when a previous checkout's package or production
    module is loaded instead of silently mixing two copies of the project.
    """
    for name, module in tuple(sys.modules.items()):
        if name == "harness_engineering" or name.startswith("harness_engineering."):
            origin = getattr(module, "__file__", None)
            if origin is None or not Path(origin).resolve().is_relative_to(_PACKAGE_ROOT):
                raise ImportError(f"{name} is already loaded outside this harness framework")
            if name == "harness_engineering":
                locations = getattr(module, "__path__", ())
                if tuple(Path(value).resolve() for value in locations) != (_PACKAGE_ROOT,):
                    raise ImportError("harness_engineering has an unexpected package search path")
    for name in ("hf", "gpu_hf", "make_alkane_xyz"):
        module = sys.modules.get(name)
        if module is not None:
            origin = getattr(module, "__file__", None)
            if origin is None or Path(origin).resolve() != HF_ROOT / f"{name}.py":
                raise ImportError(f"{name} is already loaded outside the parent HF directory")
    roots = (str(FRAMEWORK_ROOT), str(HF_ROOT))
    sys.path[:] = [*roots, *(value for value in sys.path if value not in roots)]
