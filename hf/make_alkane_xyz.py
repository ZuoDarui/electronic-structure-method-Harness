"""Write a zigzag alkane XYZ using the same geometry as alk375.xyz."""

from pathlib import Path


DX = 1.2703003855
DY = 0.8455843722

START_METHYL = (
    (-0.8722270682, 0.6536971329, 0.0),
    (-0.0182096609, -0.6292719916, -0.8898231110),
    (-0.0182096609, -0.6292719916, 0.8898231110),
)
EVEN_METHYLENE = (
    (0.0, -0.6498851137, 0.8750710480),
    (0.0, -0.6498851137, -0.8750710480),
)
ODD_METHYLENE = (
    (0.0, 0.6498851137, -0.8750710480),
    (0.0, 0.6498851137, 0.8750710480),
)
END_EVEN = (
    (-0.2664628082, -1.0569283665, 0.0),
    (0.5875545990, 0.2260407581, -0.8898231110),
    (0.5875545990, 0.2260407581, 0.8898231110),
)
END_ODD = tuple((x, -y, z) for x, y, z in END_EVEN)


def ccpvtz_basis_functions(carbon_count):
    """cc-pVTZ AO count for CnH2n+2: 30 per C and 14 per H."""
    return 58 * carbon_count + 28


def electron_count(carbon_count):
    return 8 * carbon_count + 2


def atom_count(carbon_count):
    return 3 * carbon_count + 2


def _carbon_position(index):
    y = DY if index % 2 else 0.0
    return (index * DX, y, 0.0)


def _add(origin, offset):
    return tuple(origin[i] + offset[i] for i in range(3))


def alkane_coordinates(carbon_count):
    """Return element, x, y, z rows. Carbons first, then hydrogens."""
    if carbon_count < 2:
        raise ValueError("carbon_count must be at least 2")
    carbons = [_carbon_position(i) for i in range(carbon_count)]
    hydrogens = []
    last = carbon_count - 1
    end_cap = END_EVEN if last % 2 == 0 else END_ODD
    for index, origin in enumerate(carbons):
        if index == 0:
            offsets = START_METHYL
        elif index == last:
            offsets = end_cap
        elif index % 2:
            offsets = ODD_METHYLENE
        else:
            offsets = EVEN_METHYLENE
        hydrogens.extend(_add(origin, offset) for offset in offsets)
    rows = [("C", xyz) for xyz in carbons]
    rows.extend(("H", xyz) for xyz in hydrogens)
    return rows


def format_xyz(carbon_count):
    rows = alkane_coordinates(carbon_count)
    hydrogen_count = 2 * carbon_count + 2
    comment = (
        f"C{carbon_count} H{hydrogen_count}  "
        f"nao_ccpvtz={ccpvtz_basis_functions(carbon_count)}  "
        f"nel={electron_count(carbon_count)}  dmin=1.090"
    )
    lines = [str(len(rows)), comment]
    for element, (x, y, z) in rows:
        lines.append(f"{element:<2s}{x:16.10f}{y:16.10f}{z:16.10f}")
    return "\n".join(lines) + "\n"


def write_xyz(carbon_count, path):
    path = Path(path)
    path.write_text(format_xyz(carbon_count))
    return path


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    write_xyz(86, root / "c86h174.xyz")
    write_xyz(172, root / "c172h346.xyz")
