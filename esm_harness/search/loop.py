"""Compatibility entry point for an in-process search."""

from esm_harness.search.session import run_with_client as run_search

__all__ = ["run_search"]
