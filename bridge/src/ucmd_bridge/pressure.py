"""The pressure column, evaluated in a context that never steps.

OpenMM has no virial. The bridge writes the molecular pressure of `MonteCarloBarostat
.computeCurrentPressure` (kinetic part of the molecule centres plus -dU/dV from a numerical
volume change). Asked of the *stepping* context on the GPU, that value drifts upward at fixed
volume: study 2026-11-23_water-pressure-vs-density, 512 SPC/Fw molecules in NVT, HIP, double
precision, +28, +68, +80, +79 and +98 bar over the five 20 ps blocks of a 100 ps run and +1060 bar
after 2 ns, with stationary energy and temperature. The same positions, velocities and box
given to a fresh context, or to the stepping context after a checkpoint reload, gave a
stationary value that agrees with an energy-only calculation (scaling the molecule centres and
differencing the energy). The cause inside OpenMM is not identified; the bias builds up with the
history of the stepping context (the GPU reorders atoms), so the estimate is taken in a second
context that is only ever given a state and never integrates.

Units: nm, ps, bar. Positions and velocities are those of the stepping context in its own atom
labels (OpenMM's reordering of identical molecules does not matter: the probe uses the same
system, so the labels agree by construction).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openmm as mm
from openmm import unit

from .model import ModelError


class PressureProbe:
    """Molecular pressure [bar] of a given state, from a context that never steps."""

    def __init__(self, system, barostat, platform, properties: dict) -> None:
        """`system` and `barostat` are those of the stepping context (the barostat is part of
        the system, with frequency 0 for NVT)."""
        self.barostat = barostat
        self.context = mm.Context(system, mm.VerletIntegrator(0.001), platform, properties)

    def __call__(self, positions, velocities, box_vectors) -> float:
        """Pressure of the state: `positions` [nm] (unwrapped is fine), `velocities` [nm/ps] and
        the periodic `box_vectors` as `State.getPeriodicBoxVectors()` returns them."""
        ctx = self.context
        ctx.setPeriodicBoxVectors(*box_vectors)
        ctx.setPositions(positions)
        ctx.setVelocities(velocities)
        # a state read first: computeCurrentPressure takes the energy of the unscaled state from
        # the context, and a second call without a read in between returned +40 000 bar instead
        # of the first value (OpenMM 8.5, HIP, 2026-11-23)
        ctx.getState(getEnergy=True)
        value = self.barostat.computeCurrentPressure(ctx)
        pressure = float(value.value_in_unit(unit.bar) if hasattr(value, "unit") else value)
        if not np.isfinite(pressure):
            raise ModelError("the pressure estimate is not finite")
        return pressure


# -- the pressure tensor (spec 0.1.1) -------------------------------------------------------------

BAR_PER_KJMOL_NM3 = 1e25 / 6.02214076e23  # CODATA 2018 (Avogadro exact)
# The strains of the six finite differences, (a, b): r_b → r_b + ε r_a (units.md §4). The diagonal
# and the lower components keep OpenMM's reduced (lower-triangular) cell lower-triangular.
STRAINS = ((0, 0), (1, 1), (2, 2), (1, 0), (2, 0), (2, 1))
# Relative strain of the central differences. Decided in docs/design/V-1-viscosity-gpu.md from
# the truncation error and the precision of the energies (validation/v1-pressure-tensor).
TENSOR_STRAIN = 1e-3


def bonded_groups(n_atoms: int, links) -> np.ndarray:
    """Group index per atom: atoms connected by bonds or constraints (units.md §4, since 0.1.3)."""
    parent = np.arange(n_atoms)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in links:
        a, b = root(int(i)), root(int(j))
        if a != b:
            parent[max(a, b)] = min(a, b)
    roots = np.array([root(i) for i in range(n_atoms)])
    _, index = np.unique(roots, return_inverse=True)
    return index


class TensorProbe:
    """Atomic pressure tensor [bar] of a given state, P = (Σ m v⊗v + W)/V (units.md §4).

    W_ab = −∂E/∂ε_ab by central differences of the energy under the six affine strains of
    STRAINS, applied to every atom and to the cell, in a context that never steps. The system
    must be the probe system of `model.build_system(..., for_virial=True)`: its LJ energy is
    continuous at the cutoff (the derivative is then the virial of the forces, as the engine's)
    and it has no tail correction; `tail_virial_v` (kJ/mol·nm³, `model.tail_virial_v`) adds the
    engine's virial-route tail W_tail = I·tail_virial_v/V. The energy is invariant under
    rotations of atoms and cell together, so W is symmetric and its lower components give the
    upper ones. Constraint forces are not in the energy: systems with constraints are refused.
    Positions may be unwrapped; an affine strain of atoms and cell does not depend on the image.
    """

    def __init__(self, system, platform, properties: dict, tail_virial_v: float = 0.0,
                 strain: float = TENSOR_STRAIN) -> None:  # fmt: skip
        if system.getNumConstraints():
            raise ModelError("the pressure tensor of a system with constraints needs their virial")
        self.masses = np.array(
            [system.getParticleMass(i).value_in_unit(unit.dalton)
             for i in range(system.getNumParticles())]
        )  # fmt: skip
        self.tail_virial_v = float(tail_virial_v)
        self.strain = float(strain)
        self.context = mm.Context(system, mm.VerletIntegrator(0.001), platform, properties)

    def _energy(self, positions: np.ndarray, cell: np.ndarray) -> float:
        ctx = self.context
        ctx.setPeriodicBoxVectors(*(mm.Vec3(*row) for row in cell))
        ctx.setPositions(positions)
        e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        if not np.isfinite(e):
            raise ModelError("the energy of a strained state is not finite")
        return float(e)

    def virial(self, positions, cell) -> np.ndarray:
        """W (kJ/mol, 3×3, symmetric) of positions [nm] (N, 3) in the cell [nm] (rows a, b, c)."""
        x = np.asarray(positions, dtype=float)
        h = np.asarray(cell, dtype=float)
        eps = self.strain
        w = np.zeros((3, 3))
        for a, b in STRAINS:
            e = []
            for sign in (1.0, -1.0):
                d = np.eye(3)
                d[a, b] += sign * eps
                e.append(self._energy(x @ d, h @ d))
            w[a, b] = w[b, a] = -(e[0] - e[1]) / (2.0 * eps)
        volume = abs(float(np.linalg.det(h)))
        return w + np.eye(3) * (self.tail_virial_v / volume)

    def __call__(self, positions, velocities, cell) -> np.ndarray:
        """(xx, yy, zz, xy, xz, yz) of P in bar, the spec's column order."""
        p = self.tensor(positions, velocities, cell)
        return np.array([p[0, 0], p[1, 1], p[2, 2], p[0, 1], p[0, 2], p[1, 2]])

    def tensor(self, positions, velocities, cell) -> np.ndarray:
        """P (bar, 3×3) of the state: positions [nm], velocities [nm/ps], cell [nm]."""
        v = np.asarray(velocities, dtype=float)
        kinetic = (v * self.masses[:, None]).T @ v
        volume = abs(float(np.linalg.det(np.asarray(cell, dtype=float))))
        p = (kinetic + self.virial(positions, cell)) / volume * BAR_PER_KJMOL_NM3
        if not np.all(np.isfinite(p)):
            raise ModelError("the pressure tensor is not finite")
        return p

    def with_diagnostics(self, positions, velocities, cell, groups: np.ndarray):
        """Validation only (V-1): P (bar, 3×3) and, in bar, the symmetric part of the molecular
        tensor (units.md §4: centre-of-mass kinetic part, W_mol = W − Σ_G Σ_i∈G (r_i − R_G) ⊗ F_i)
        and Σ_i F_i⊗F_i/(m_i V) in bar/ps², the half-step kinetic correction per Δt²/4. The
        forces come from the unstrained probe state; positions must be unwrapped (whole groups).
        """
        x = np.asarray(positions, dtype=float)
        v = np.asarray(velocities, dtype=float)
        h = np.asarray(cell, dtype=float)
        volume = abs(float(np.linalg.det(h)))
        w = self.virial(x, h)
        ctx = self.context
        ctx.setPeriodicBoxVectors(*(mm.Vec3(*row) for row in h))
        ctx.setPositions(x)
        f = ctx.getState(getForces=True).getForces(asNumpy=True)
        f = np.asarray(f.value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
        m = self.masses
        n_groups = int(groups.max()) + 1
        mass = np.bincount(groups, weights=m, minlength=n_groups)
        centre = np.stack([np.bincount(groups, weights=m * x[:, k], minlength=n_groups)
                           for k in range(3)], axis=1) / mass[:, None]  # fmt: skip
        momentum = np.stack([np.bincount(groups, weights=m * v[:, k], minlength=n_groups)
                             for k in range(3)], axis=1)  # fmt: skip
        d = x - centre[groups]
        w_mol = w - d.T @ f
        k_mol = momentum.T @ (momentum / mass[:, None])
        p_mol = (k_mol + w_mol) / volume * BAR_PER_KJMOL_NM3
        ff = (f / m[:, None]).T @ f / volume * BAR_PER_KJMOL_NM3
        p = ((v * m[:, None]).T @ v + w) / volume * BAR_PER_KJMOL_NM3
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(p_mol))):
            raise ModelError("the pressure tensor is not finite")
        return p, {"molecular_sym": 0.5 * (p_mol + p_mol.T), "molecular_antisym_xy": p_mol[0, 1]
                   - p_mol[1, 0], "ff_over_m": ff}  # fmt: skip


class TensorDiagnostics:
    """Validation only (V-1, not a spec output): one tab-separated row per tensor evaluation with
    the symmetric molecular tensor (bar) and Σ F⊗F/(m V) (bar/ps²), upper triangles in the
    spec's column order. Named by the environment variable of openmm_backend.DIAGNOSTICS_ENV."""

    ORDER = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

    def __init__(self, path: Path, top, append: bool) -> None:
        links = [(i, j) for i, j, *_ in top.bonds] + [(i, j) for i, j, _ in top.constraints]
        self.groups = bonded_groups(top.n_atoms, links)
        self.fh = path.open("a" if append else "w")
        if not append:
            names = [f"pmol_{a}{b}_bar" for a, b in self._labels()]
            names += [f"ff_over_m_{a}{b}_bar_per_ps2" for a, b in self._labels()]
            self.fh.write("# step\ttime_ps\t" + "\t".join(names) + "\n")

    def _labels(self):
        return [("xyz"[a], "xyz"[b]) for a, b in self.ORDER]

    def write(self, step: int, time_ps: float, extra: dict) -> None:
        cells = [str(step), f"{time_ps:.6f}"]
        for key in ("molecular_sym", "ff_over_m"):
            cells += [f"{extra[key][a, b]:.10e}" for a, b in self.ORDER]
        self.fh.write("\t".join(cells) + "\n")

    def close(self) -> None:
        self.fh.close()
