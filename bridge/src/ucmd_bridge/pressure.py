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
