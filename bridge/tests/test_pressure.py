"""The pressure column: evaluated in a context that never steps (pressure.py).

Study 2026-11-23_water-pressure-vs-density: in the stepping GPU context, MonteCarloBarostat
.computeCurrentPressure drifted upward at fixed volume (+28, +68, +80, +79, +98 bar in the
five 20 ps blocks of a 100 ps run; +1060 bar after 2 ns), while the same positions, velocities
and box evaluated in a fresh context gave a stationary value that agrees with an energy-only
calculation (scaling the molecule centres). The drift cannot appear on OpenMM's CPU platforms
(no atom reordering), so these tests pin what the probe has to deliver: the molecular pressure
of the state it is given, independent of what any other context did.
"""

from __future__ import annotations

import numpy as np
import pytest

mm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

from ucmd_bridge.openmm_backend import BAR_PER_KJMOL_NM3, BOLTZMANN_KJMOL_K  # noqa: E402
from ucmd_bridge.pressure import PressureProbe  # noqa: E402

N_SIDE, SPACING, BOND = 4, 0.75, 0.15  # 64 homonuclear dimers in a 3 nm box
MASS, T = 12.0, 300.0


def _dimers(seed: int = 7):
    """A periodic fluid of 64 flexible dimers (LJ between molecules, harmonic bond within)."""
    rng = np.random.default_rng(seed)
    box = N_SIDE * SPACING
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(*(mm.Vec3(*row) for row in np.eye(3) * box))
    bonds = mm.HarmonicBondForce()
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(0.9)
    nonbonded.setUseDispersionCorrection(True)
    bond_list, positions = [], []
    grid = np.stack(np.meshgrid(*[np.arange(N_SIDE)] * 3, indexing="ij"), -1).reshape(-1, 3)
    for site in grid:
        axis = rng.normal(size=3)
        axis *= BOND / np.linalg.norm(axis)
        centre = (site + 0.5) * SPACING
        a = system.addParticle(MASS)
        b = system.addParticle(MASS)
        for _ in (a, b):
            nonbonded.addParticle(0.0, 0.3, 0.8)
        bonds.addBond(a, b, BOND, 2.0e5)
        bond_list.append([a, b])
        positions += [centre - axis / 2, centre + axis / 2]
    nonbonded.createExceptionsFromBonds(bond_list, 0.0, 0.0)
    system.addForce(bonds)
    system.addForce(nonbonded)
    barostat = mm.MonteCarloBarostat(1.0, T, 0)
    system.addForce(barostat)
    positions = np.array(positions)
    velocities = rng.normal(size=positions.shape) * np.sqrt(BOLTZMANN_KJMOL_K * T / MASS)
    return system, barostat, positions, velocities, box


def _context(system, positions, velocities, box):
    ctx = mm.Context(
        system, mm.VerletIntegrator(0.0005), mm.Platform.getPlatformByName("Reference")
    )
    ctx.setPeriodicBoxVectors(*(mm.Vec3(*row) for row in np.eye(3) * box))
    ctx.setPositions(positions)
    ctx.setVelocities(velocities)
    return ctx


def _energy_only(system, positions, velocities, box) -> float:
    """Molecular pressure from energies alone: the kinetic part of the molecule centres plus
    -dU/dV from scaling the centres of the (whole, homonuclear) dimers, central difference."""
    ctx = _context(system, positions, velocities, box)
    pairs = positions.reshape(-1, 2, 3)
    centres = pairs.mean(1)
    volume = box**3
    eps = 1e-4

    def energy(rel: float) -> float:
        s = (1.0 + rel) ** (1.0 / 3.0)
        moved = (positions.reshape(-1, 2, 3) + ((s - 1.0) * centres)[:, None, :]).reshape(-1, 3)
        ctx.setPeriodicBoxVectors(*(mm.Vec3(*row) for row in np.eye(3) * box * s))
        ctx.setPositions(moved)
        return (
            ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )

    dudv = (energy(+eps) - energy(-eps)) / (2.0 * eps * volume)
    v_com = velocities.reshape(-1, 2, 3).mean(1)
    kinetic = (2.0 * MASS * (v_com**2).sum()) / (3.0 * volume)
    return (kinetic - dudv) * BAR_PER_KJMOL_NM3


def _state(ctx):
    st = ctx.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=False)
    return st.getPositions(asNumpy=True), st.getVelocities(asNumpy=True), st.getPeriodicBoxVectors()


def test_probe_equals_the_energy_only_molecular_pressure() -> None:
    system, barostat, x, v, box = _dimers()
    probe = PressureProbe(system, barostat, mm.Platform.getPlatformByName("Reference"), {})
    ctx = _context(system, x, v, box)
    got = probe(*_state(ctx))
    want = _energy_only(system, x, v, box)
    assert want == pytest.approx(got, rel=2e-3, abs=2.0), (got, want)  # bar


def test_probe_gives_the_same_value_every_time_and_after_other_states() -> None:
    """A second call on the same state (with no state read in between) returned +40 000 bar
    instead of the first value in OpenMM 8.5 on the GPU (2026-11-23): the probe reads the state
    itself before it asks for the pressure."""
    system, barostat, x, v, box = _dimers()
    probe = PressureProbe(system, barostat, mm.Platform.getPlatformByName("Reference"), {})
    ctx = _context(system, x, v, box)
    first = probe(*_state(ctx))
    assert probe(*_state(ctx)) == first
    ctx.getIntegrator().step(200)  # a different state in between ...
    other = probe(*_state(ctx))
    assert other != first
    ctx0 = _context(system, x, v, box)  # ... and back to the first one
    assert probe(*_state(ctx0)) == pytest.approx(first, rel=0, abs=1e-9)


def test_probe_follows_a_stepping_context_on_a_cpu_platform() -> None:
    """Where the stepping context is sound (OpenMM's CPU platforms) the probe and the stepping
    context's own estimator agree to rounding, so the probe changes nothing there."""
    system, barostat, x, v, box = _dimers()
    probe = PressureProbe(system, barostat, mm.Platform.getPlatformByName("Reference"), {})
    ctx = _context(system, x, v, box)
    for _ in range(3):
        ctx.getIntegrator().step(100)
        ctx.getState(getEnergy=True)  # the state read observe() makes before asking
        own = barostat.computeCurrentPressure(ctx)
        own = float(own.value_in_unit(unit.bar) if hasattr(own, "unit") else own)
        assert probe(*_state(ctx)) == pytest.approx(own, rel=1e-9, abs=1e-6)
