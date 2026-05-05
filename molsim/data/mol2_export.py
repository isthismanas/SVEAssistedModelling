from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


_Z_TO_ELEMENT = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}


@dataclass(frozen=True)
class Mol2ExportConfig:
    output_dir: str
    overwrite: bool = False


class Mol2Exporter:
    """Export PyG molecular graphs into Tripos mol2 files for spatial supervision."""

    def __init__(self, config: Mol2ExportConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _name_for_data(data: Any, index: int) -> str:
        name = getattr(data, "name", None)
        if name is None:
            return f"sample_{index:06d}"

        # PyG QM9 can return name as str or tensor-ish.
        if isinstance(name, str):
            return name

        if hasattr(name, "item"):
            try:
                return str(name.item())
            except Exception:
                pass

        return str(name)

    @staticmethod
    def _infer_bonds(edge_index: torch.Tensor | None, n_atoms: int) -> list[tuple[int, int]]:
        if edge_index is None or edge_index.numel() == 0:
            return []

        unique: set[tuple[int, int]] = set()
        for col in range(edge_index.shape[1]):
            a = int(edge_index[0, col])
            b = int(edge_index[1, col])
            if a == b:
                continue
            i, j = (a, b) if a < b else (b, a)
            if 0 <= i < n_atoms and 0 <= j < n_atoms:
                unique.add((i, j))

        return sorted(unique)

    @staticmethod
    def _atom_type(z_value: float, atom_idx: int) -> tuple[str, str]:
        z_int = int(round(float(z_value)))
        elem = _Z_TO_ELEMENT.get(z_int, "C")
        atom_name = f"{elem}{atom_idx + 1}"
        atom_type = f"{elem}.3"
        return atom_name, atom_type

    def export_one(self, data: Any, index: int) -> Path:
        pos = getattr(data, "pos", None)
        z = getattr(data, "z", None)

        if pos is None or z is None:
            raise ValueError("Data sample must contain both 'pos' and 'z' to export mol2")

        pos_t = pos.detach().cpu().float()
        z_t = z.detach().cpu().float().view(-1)

        n_atoms = int(pos_t.shape[0])
        if z_t.shape[0] != n_atoms:
            raise ValueError("z and pos must have matching number of atoms")

        bonds = self._infer_bonds(getattr(data, "edge_index", None), n_atoms)
        mol_name = self._name_for_data(data, index)
        out_path = self.output_dir / f"{mol_name}.mol2"

        if out_path.exists() and not self.config.overwrite:
            return out_path

        lines: list[str] = []
        lines.append("@<TRIPOS>MOLECULE")
        lines.append(mol_name)
        lines.append(f"{n_atoms} {len(bonds)} 0 0 0")
        lines.append("SMALL")
        lines.append("NO_CHARGES")
        lines.append("")

        lines.append("@<TRIPOS>ATOM")
        for i in range(n_atoms):
            x, y, zc = [float(v) for v in pos_t[i].tolist()]
            atom_name, atom_type = self._atom_type(float(z_t[i].item()), i)
            lines.append(
                f"{i + 1:>6} {atom_name:<6} {x:>10.4f} {y:>10.4f} {zc:>10.4f} {atom_type:<6} 1 MOL 0.000"
            )

        lines.append("@<TRIPOS>BOND")
        for bond_idx, (a, b) in enumerate(bonds, start=1):
            # mol2 atom indices are 1-based.
            lines.append(f"{bond_idx:>6} {a + 1:>4} {b + 1:>4} 1")

        out_path.write_text("\n".join(lines) + "\n")
        return out_path

    def export_dataset(self, dataset: Any, max_samples: int | None = None) -> list[Path]:
        total = len(dataset)
        n = total if max_samples is None else min(total, int(max_samples))

        out: list[Path] = []
        for idx in range(n):
            out.append(self.export_one(dataset[idx], idx))
        return out
