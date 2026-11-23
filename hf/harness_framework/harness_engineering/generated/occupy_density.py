"""Replaceable occupied-density update.

The first candidate calls the dense reference. Later candidates may change
only this function. ``hf.py`` and ``occupy_density_reference`` stay fixed.
"""

from __future__ import annotations

from harness_engineering.training_hf import occupy_density_reference


def occupy_density(fock, overlap, nocc):
    """Return the occupied density for one generalized eigenproblem."""

    return occupy_density_reference(fock, overlap, nocc)
