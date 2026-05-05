from __future__ import annotations

from pathlib import Path

import torch


_ELEMENT_TO_Z = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "CL": 17,
    "BR": 35,
    "I": 53,
}


def _atom_type_to_atomic_number(atom_type: str) -> int:
    token = atom_type.split(".")[0].strip()
    token = token.upper()
    if token in _ELEMENT_TO_Z:
        return _ELEMENT_TO_Z[token]

    # Fallback for malformed or uncommon type names.
    if token:
        first = token[0]
        return _ELEMENT_TO_Z.get(first, 6)
    return 6


def parse_mol2_structure(path: str | Path) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Parse atom coordinates, inferred atomic numbers, and bonds from a mol2 file."""
    path_obj = Path(path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"mol2 file not found: {path_obj}")

    lines = path_obj.read_text().splitlines()

    mode: str | None = None
    coords: list[list[float]] = []
    z_values: list[int] = []
    bonds: set[tuple[int, int]] = set()

    for line in lines:
        if line.startswith("@<TRIPOS>"):
            if line.startswith("@<TRIPOS>ATOM"):
                mode = "atom"
            elif line.startswith("@<TRIPOS>BOND"):
                mode = "bond"
            else:
                mode = None
            continue
        if not line.strip() or mode is None:
            continue

        parts = line.split()
        if mode == "atom":
            if len(parts) < 6:
                continue

            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
            atom_type = parts[5]

            coords.append([x, y, z])
            z_values.append(_atom_type_to_atomic_number(atom_type))
            continue

        if mode == "bond":
            if len(parts) < 4:
                continue

            # mol2 stores 1-based atom indices.
            try:
                a = int(parts[1]) - 1
                b = int(parts[2]) - 1
            except ValueError:
                continue
            if a == b:
                continue

            i, j = (a, b) if a < b else (b, a)
            bonds.add((i, j))

    if not coords:
        raise ValueError(f"No atom coordinates parsed from mol2: {path_obj}")

    n_atoms = len(coords)
    filtered_bonds = [(i, j) for (i, j) in sorted(bonds) if 0 <= i < n_atoms and 0 <= j < n_atoms]
    return (
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(z_values, dtype=torch.float32),
        filtered_bonds,
    )


def parse_mol2_atoms(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible atom-only parser."""
    coords, z_values, _ = parse_mol2_structure(path)
    return coords, z_values
