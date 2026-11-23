"""Deterministic water-cluster geometries for the scale ladder.

The monomer matches the water geometry used by the existing RHF tasks.
Oxygen atoms sit on a cubic lattice with a 4.0 angstrom spacing, and both
hydrogen atoms keep the monomer displacement from their oxygen.
"""

from __future__ import annotations

CLUSTER_COUNTS = (16, 32, 64, 128)
SPACING_ANGSTROM = 4.0
MONOMER = (
    ("O", 0.0, 0.0, 0.0),
    ("H", 0.0, -0.757, 0.587),
    ("H", 0.0, 0.757, 0.587),
)
_SHAPES = {
    16: (2, 2, 4),
    32: (4, 2, 4),
    64: (4, 4, 4),
    128: (4, 4, 8),
}


def cluster_shape(count: int) -> tuple[int, int, int]:
    """Return the lattice dimensions for a supported cluster size."""

    try:
        return _SHAPES[int(count)]
    except KeyError as exc:
        supported = ", ".join(str(item) for item in CLUSTER_COUNTS)
        raise ValueError(f"water cluster size must be one of {supported}") from exc


def cluster_atoms(count: int) -> list[tuple[str, float, float, float]]:
    """Return symbol and Cartesian coordinates in angstroms."""

    nx, ny, nz = cluster_shape(count)
    atoms = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                origin = (
                    ix * SPACING_ANGSTROM,
                    iy * SPACING_ANGSTROM,
                    iz * SPACING_ANGSTROM,
                )
                for symbol, x, y, z in MONOMER:
                    atoms.append((symbol, x + origin[0], y + origin[1], z + origin[2]))
    return atoms


def cluster_atom_string(count: int) -> str:
    """Return a PySCF atom string for one water cluster."""

    return "; ".join(
        f"{symbol} {x:.6f} {y:.6f} {z:.6f}"
        for symbol, x, y, z in cluster_atoms(count)
    )


def nao_mismatch_max(nao_by_count, monomer_nao: int) -> int:
    """Largest absolute deviation from monomer basis functions times cluster size."""

    if int(monomer_nao) <= 0:
        raise ValueError("monomer_nao must be positive")
    mismatches = []
    for count in CLUSTER_COUNTS:
        nao = int(nao_by_count[count])
        mismatches.append(abs(nao - int(monomer_nao) * count))
    return max(mismatches)
