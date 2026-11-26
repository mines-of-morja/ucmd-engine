"""Outputs in the spec's formats: thermo table, H5MD trajectory, extended-XYZ viewer trajectory.

Each writer can reopen its file after a restart: rows and frames after the checkpoint step are
cut off, new ones are appended (spec/checkpoint.md §7, trajectory-h5md.md §2).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .model import ModelError

THERMO_COLUMNS = (
    "step",
    "time_ps",
    "potential_kjmol",
    "kinetic_kjmol",
    "total_kjmol",
    "conserved_energy_kjmol",
    "temperature_K",
    "pressure_bar",
    "volume_nm3",
)
ANGSTROM_PER_NM = 10.0


def _num(x: float | None) -> str:
    if x is None or not math.isfinite(x):
        return "nan"
    return f"{x:.10e}"


class ThermoWriter:
    """The thermo table of spec/run-config.md (tab-separated, one flushed line per row)."""

    def __init__(self, path: Path, restart_step: int | None) -> None:
        self.path = path
        if restart_step is not None and path.is_file():
            kept = []
            for line in path.read_text().splitlines(keepends=True):
                if line.startswith("#") or int(float(line.split("\t", 1)[0])) <= restart_step:
                    kept.append(line)
            path.write_text("".join(kept))
            self.fh = path.open("a")
        else:
            self.fh = path.open("w")
            self.fh.write("# " + "\t".join(THERMO_COLUMNS) + "\n")
        self.fh.flush()

    def write(self, row: dict) -> None:
        cells = [str(row["step"]), f"{row['time_ps']:.6f}"]
        for key in THERMO_COLUMNS[2:]:
            value = row.get(key)
            cells.append(f"{value:.6f}" if key == "temperature_K" and value is not None
                         else _num(value))  # fmt: skip
        self.fh.write("\t".join(cells) + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


class XyzWriter:
    """Extended XYZ in Å, one flushed frame at a time (the engine's viewer format)."""

    def __init__(self, path: Path, symbols: list[str], restart_step: int | None) -> None:
        self.symbols = symbols
        if restart_step is not None and path.is_file():
            lines = path.read_text().splitlines(keepends=True)
            kept, at = [], 0
            while at < len(lines):
                n = int(lines[at])
                step = next(
                    (int(w[5:]) for w in lines[at + 1].split() if w.startswith("step=")), None
                )
                if step is None or step > restart_step or at + 2 + n > len(lines):
                    break
                kept += lines[at : at + 2 + n]
                at += 2 + n
            path.write_text("".join(kept))
            self.fh = path.open("a")
        else:
            self.fh = path.open("w")

    def write(self, step: int, time_ps: float, positions_nm: np.ndarray, cell_nm,
              row: dict | None = None) -> None:  # fmt: skip
        """One frame; the thermo `row` of the H5MD writer has no place in this format."""
        out = [f"{len(positions_nm)}\n"]
        comment = "Properties=species:S:1:pos:R:3"
        if cell_nm is not None:
            lattice = " ".join(f"{x * ANGSTROM_PER_NM:.8f}" for x in np.asarray(cell_nm).ravel())
            comment = f'Lattice="{lattice}" {comment} pbc="T T T"'
        out.append(f"{comment} step={step} time_ps={time_ps}\n")
        for sym, r in zip(self.symbols, positions_nm * ANGSTROM_PER_NM, strict=True):
            out.append(f"{sym} {r[0]:.8f} {r[1]:.8f} {r[2]:.8f}\n")
        self.fh.write("".join(out))
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


class H5mdWriter:
    """H5MD 1.1 with the UnitcellMD profile (spec/trajectory-h5md.md)."""

    OBSERVABLES = (
        ("potential_energy", "potential_kjmol", "kJ mol-1"),
        ("kinetic_energy", "kinetic_kjmol", "kJ mol-1"),
        ("conserved_energy", "conserved_energy_kjmol", "kJ mol-1"),
        ("temperature", "temperature_K", "K"),
        ("volume", "volume_nm3", "nm3"),
    )

    VECTORS = (("velocity", "nm ps-1"), ("force", "kJ mol-1 nm-1"))  # optional, per atom

    def __init__(self, path: Path, top, *, precision: str, provenance: dict,
                 restart_step: int | None, vectors: tuple[str, ...] = ()) -> None:  # fmt: skip
        """`vectors`: which of "velocity" and "force" every frame also holds."""
        import h5py

        self.dtype = np.float32 if precision == "float32" else np.float64
        self.vectors = [name for name, _unit in self.VECTORS if name in vectors]
        if restart_step is not None and path.is_file():
            self.fh = h5py.File(path, "a")
            missing = [n for n in self.vectors if n not in self.fh["particles/all"]]
            if missing:
                self.fh.close()
                raise ModelError(
                    f"{path}: the trajectory to continue holds no {' and no '.join(missing)} series"
                )
            self._truncate(restart_step)
            return
        self.fh = fh = h5py.File(path, "w")
        h5md = fh.create_group("h5md")
        h5md.attrs["version"] = np.array([1, 1], dtype=np.int32)
        h5md.create_group("author").attrs["name"] = provenance.get("author", "anonymous")
        creator = h5md.create_group("creator")
        creator.attrs["name"] = "ucmd"
        creator.attrs["version"] = provenance["engine_version"]
        h5md.create_group("modules/units").attrs["version"] = np.array([1, 0], dtype=np.int32)
        part = fh.create_group("particles/all")
        box = part.create_group("box")
        box.attrs["dimension"] = 3
        periodic = top.cell_nm is not None
        box.attrs["boundary"] = ["periodic" if periodic else "none"] * 3
        self._series(box, "edges", (3, 3), "nm")
        self._series(part, "position", (top.n_atoms, 3), "nm")
        for name, unit in self.VECTORS:
            if name in self.vectors:
                self._series(part, name, (top.n_atoms, 3), unit)
        for name, data, unit in (("mass", top.masses_amu, "amu"), ("charge", top.charges_e, "e")):
            d = part.create_dataset(name, data=np.asarray(data, dtype=np.float64))
            d.attrs["unit"] = unit
        part.create_dataset("species", data=top.species)
        part.create_dataset("molecule_index", data=top.molecule_index)
        obs = fh.create_group("observables")
        for name, _key, unit in self.OBSERVABLES:
            self._series(obs, name, (), unit)
        params = fh.create_group("parameters/ucmd")
        for key in ("spec_version", "engine_version", "engine_git_hash", "system_sha256"):
            params.attrs[key] = provenance[key]
        params.attrs["positions"] = "unwrapped"
        params.create_dataset("system_json", data=top.system_json)
        params.create_dataset("run_config_toml", data=provenance["run_config_toml"])
        backend = params.create_group("backend")
        for key, value in provenance["backend"].items():
            backend.attrs[key] = value

    def _series(self, parent, name: str, shape: tuple, unit: str) -> None:
        g = parent.create_group(name)
        g.create_dataset("step", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(1,))
        t = g.create_dataset("time", shape=(0,), maxshape=(None,), dtype=np.float64, chunks=(1,))
        t.attrs["unit"] = "ps"
        v = g.create_dataset(
            "value", shape=(0, *shape), maxshape=(None, *shape), dtype=self.dtype,
            chunks=(1, *shape) if shape else (1,),
        )  # fmt: skip
        v.attrs["unit"] = unit

    def _groups(self):
        yield self.fh["particles/all/box/edges"]
        yield self.fh["particles/all/position"]
        for name in self.vectors:
            yield self.fh[f"particles/all/{name}"]
        for name, _key, _unit in self.OBSERVABLES:
            yield self.fh[f"observables/{name}"]

    def _truncate(self, restart_step: int) -> None:
        for g in self._groups():
            n = int(np.searchsorted(g["step"][()], restart_step, side="right"))
            for ds in ("step", "time", "value"):
                g[ds].resize(n, axis=0)
        self.fh.flush()

    @staticmethod
    def _append(g, step: int, time_ps: float, value) -> None:
        n = g["step"].shape[0]
        for ds in ("step", "time", "value"):
            g[ds].resize(n + 1, axis=0)
        g["step"][n] = step
        g["time"][n] = time_ps
        g["value"][n] = value

    def write(self, step: int, time_ps: float, positions_nm, cell_nm, row: dict,
              vectors: dict | None = None) -> None:  # fmt: skip
        """One frame. `vectors` holds the series named at creation, in canonical atom order."""
        edges = np.zeros((3, 3)) if cell_nm is None else np.asarray(cell_nm)
        self._append(self.fh["particles/all/box/edges"], step, time_ps, edges)
        self._append(self.fh["particles/all/position"], step, time_ps, positions_nm)
        for name in self.vectors:
            self._append(self.fh[f"particles/all/{name}"], step, time_ps, vectors[name])
        for name, key, _unit in self.OBSERVABLES:
            value = row.get(key)
            self._append(self.fh[f"observables/{name}"], step, time_ps,
                         np.nan if value is None else value)  # fmt: skip
        self.fh.flush()

    def record_restart(self, record: dict) -> None:
        restarts = self.fh.require_group("parameters/ucmd/restarts")
        g = restarts.create_group(f"{len(restarts):03d}")
        for key, value in record.items():
            g.attrs[key] = value

    def close(self) -> None:
        self.fh.close()
