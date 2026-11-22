"""A spec v0.1 system package expanded to global atom indices (spec/system-package.md).

Exclusions and 1-4 pairs follow the shortest bond path within a molecule (path length 1-2
excluded, 3 scaled), as in the engine's force-field provider. Units are the spec's.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SYMBOLS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge "
    "As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe"
)
ATOMIC_NUMBER = {s: z for z, s in enumerate(_SYMBOLS.split())}


@dataclass(frozen=True)
class PairTable:
    """LJ σ_ij (nm) and ε_ij (kJ/mol) per type pair, for packages with `pair_overrides`.

    Spec 0.1.2, system-package.md §5.2: the override where one is given, else the combination
    rule. `type_index[i]` is atom i's row; the tables are symmetric.
    """

    type_index: np.ndarray
    sigma_nm: np.ndarray
    epsilon_kjmol: np.ndarray


def combine(rule: str, s1: float, e1: float, s2: float, e2: float) -> tuple[float, float]:
    """σ_ij, ε_ij by the combination rule (spec §5.2)."""
    sigma = math.sqrt(s1 * s2) if rule == "geometric" else 0.5 * (s1 + s2)
    return sigma, math.sqrt(e1 * e2)


def pair_table(ff: dict, atom_types: list[str]) -> PairTable | None:
    """The type-pair table of a force-field block, or None without overrides."""
    overrides = ff.get("pair_overrides", [])
    if not overrides:
        return None
    names = sorted(ff["atom_types"])
    row = {t: k for k, t in enumerate(names)}
    n = len(names)
    sigma, epsilon = np.zeros((n, n)), np.zeros((n, n))
    for a, ta in enumerate(names):
        for b, tb in enumerate(names):
            la, lb = ff["atom_types"][ta], ff["atom_types"][tb]
            sigma[a, b], epsilon[a, b] = combine(
                ff["combination_rule"],
                la["sigma_nm"],
                la["epsilon_kjmol"],
                lb["sigma_nm"],
                lb["epsilon_kjmol"],
            )
    for k, o in enumerate(overrides):
        if o["types"][0] not in row or o["types"][1] not in row:
            raise ValueError(f"pair override {k} names an unknown atom type: {o['types']}")
        a, b = (row[t] for t in o["types"])
        sigma[a, b] = sigma[b, a] = o["sigma_nm"]
        epsilon[a, b] = epsilon[b, a] = o["epsilon_kjmol"]
    return PairTable(np.array([row[t] for t in atom_types]), sigma, epsilon)


@dataclass(frozen=True)
class Topology:
    name: str
    system_json: str
    system_sha256: str
    positions_nm: np.ndarray
    velocities_nm_ps: np.ndarray | None
    cell_nm: np.ndarray | None
    masses_amu: np.ndarray
    charges_e: np.ndarray
    sigma_nm: np.ndarray
    epsilon_kjmol: np.ndarray
    species: np.ndarray
    molecule_index: np.ndarray
    combination_rule: str
    scale_14_lj: float
    scale_14_coulomb: float
    bonds: list
    angles: list
    periodic: list
    rb: list
    improper_periodic: list
    improper_harmonic: list
    excluded: list
    pairs14: list
    constraints: list
    pairs: PairTable | None = None  # only with pair overrides (spec 0.1.2)

    @property
    def n_atoms(self) -> int:
        return len(self.masses_amu)


def _graph_distances(n: int, bonds: list[list[int]], limit: int = 3) -> dict[tuple[int, int], int]:
    neighbours: list[list[int]] = [[] for _ in range(n)]
    for i, j in bonds:
        neighbours[i].append(j)
        neighbours[j].append(i)
    out: dict[tuple[int, int], int] = {}
    for start in range(n):
        depth = {start: 0}
        frontier = [start]
        for d in range(1, limit + 1):
            nxt = []
            for a in frontier:
                for b in neighbours[a]:
                    if b not in depth:
                        depth[b] = d
                        nxt.append(b)
            frontier = nxt
        for b, d in depth.items():
            if b > start:
                out[(start, b)] = d
    return out


def load(path: Path | str) -> Topology:
    """Read a parameterized spec 0.1 package with all its bonded terms (0.1.2: pair overrides)."""
    path = Path(path)
    raw = (path / "system.json").read_bytes()
    doc = json.loads(raw)
    if doc.get("spec_version", "").split(".")[:2] != ["0", "1"]:
        raise ValueError(f"the bridge reads spec 0.1 packages, got {doc.get('spec_version')}")
    ff = doc.get("forcefield")
    if ff is None:
        raise ValueError("the OpenMM backend needs a parameterized package")
    masses, charges, sigma, epsilon, species, molecule, types = ([] for _ in range(7))
    bonds, angles, periodic, rb, imp_p, imp_h, excluded, pairs14, constraints = (
        [] for _ in range(9)
    )
    offset = 0
    molecule_count = 0
    for component in doc["components"]:
        mtype = doc["molecule_types"][component["molecule_type"]]
        n = len(mtype["atoms"])
        distances = _graph_distances(n, [b["atoms"] for b in mtype["bonds"]])
        for _ in range(component["count"]):
            for atom in mtype["atoms"]:
                lj = ff["atom_types"][atom["type"]]
                types.append(atom["type"])
                masses.append(atom["mass_amu"])
                charges.append(atom.get("charge_e") or 0.0)
                sigma.append(lj["sigma_nm"])
                epsilon.append(lj["epsilon_kjmol"])
                species.append(ATOMIC_NUMBER.get(atom.get("element") or "X", 0))
                molecule.append(molecule_count)
            o = offset
            for b in mtype["bonds"]:
                if b["form"] != "harmonic":
                    raise ValueError(f"bond form {b['form']}")
                i, j = b["atoms"]
                bonds.append((o + i, o + j, b["params"]["r0_nm"], b["params"]["k_kjmol_nm2"]))
            for a in mtype.get("angles", []):
                i, j, k = a["atoms"]
                p = a["params"]
                angles.append(
                    (o + i, o + j, o + k, math.radians(p["theta0_deg"]), p["k_kjmol_rad2"])
                )
            for d in mtype.get("dihedrals", []):
                i, j, k, m = (o + x for x in d["atoms"])
                p = d["params"]
                if d["form"] == "periodic":
                    periodic.append(
                        (i, j, k, m, int(p["n"]), math.radians(p["phi0_deg"]), p["k_kjmol"])
                    )
                elif d["form"] == "ryckaert-bellemans":
                    rb.append((i, j, k, m, tuple(p[f"c{c}_kjmol"] for c in range(6))))
                elif d["form"] == "opls-fourier":
                    for nn, key, phase in (
                        (1, "v1_kjmol", 0.0),
                        (2, "v2_kjmol", math.pi),
                        (3, "v3_kjmol", 0.0),
                        (4, "v4_kjmol", math.pi),
                    ):
                        periodic.append((i, j, k, m, nn, phase, 0.5 * p[key]))
                else:
                    raise ValueError(f"dihedral form {d['form']}")
            for d in mtype.get("impropers", []):
                i, j, k, m = (o + x for x in d["atoms"])
                p = d["params"]
                if d["form"] == "periodic":
                    imp_p.append(
                        (i, j, k, m, int(p["n"]), math.radians(p["phi0_deg"]), p["k_kjmol"])
                    )
                elif d["form"] == "harmonic":
                    imp_h.append((i, j, k, m, math.radians(p["xi0_deg"]), p["k_kjmol_rad2"]))
                else:
                    raise ValueError(f"improper form {d['form']}")
            for (i, j), dist in distances.items():
                (excluded if dist <= 2 else pairs14).append((o + i, o + j))
            for c in mtype.get("constraints", []):
                i, j = c["atoms"]
                constraints.append((o + i, o + j, c["length_nm"]))
            offset += n
            molecule_count += 1
    cell = None if doc["cell_nm"] is None else np.array(doc["cell_nm"], dtype=float)
    vfile = path / "velocities.npy"
    return Topology(
        doc["name"],
        raw.decode("utf-8"),
        hashlib.sha256(raw).hexdigest(),
        np.load(path / "positions.npy"),
        np.load(vfile) if vfile.is_file() else None,
        cell,
        np.array(masses),
        np.array(charges),
        np.array(sigma),
        np.array(epsilon),
        np.array(species, dtype=np.int32),
        np.array(molecule, dtype=np.int64),
        ff["combination_rule"],
        float(ff["scale_14_lj"]),
        float(ff["scale_14_coulomb"]),
        bonds,
        angles,
        periodic,
        rb,
        imp_p,
        imp_h,
        excluded,
        pairs14,
        constraints,
        pair_table(ff, types),
    )
