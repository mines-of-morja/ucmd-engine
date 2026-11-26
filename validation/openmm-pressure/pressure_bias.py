"""Does the pressure of a stepping OpenMM context drift? (GPU check for the bridge)

Runs the bridge's own system and integrator from a run.toml for N steps. Every EVERY steps it
evaluates the same state three ways and prints block means:

  stepping  MonteCarloBarostat.computeCurrentPressure asked of the stepping context (what the
            bridge used before commit c2cc427)
  probe     the bridge's PressureProbe (a second context that never steps, what it uses now)
  energy    an energy-only calculation: scale the molecule centres by (1 ± 1e-3)^(1/3), difference
            the energy in a third context, add the kinetic part of the molecule centres

Positions are read every 25 steps through the identity tracker, as the bridge does; the number
of label exchanges is printed. Run in the bridge image on the GPU (see run.sh). Study
2026-11-23_water-pressure-vs-density, 512 SPC/Fw water: stepping − probe grew by +28, +68, +80,
+79, +98 bar over five 20 ps blocks, probe − energy stayed within ±25 bar.

    pressure_bias.py RUN.toml STEPS [EVERY=500 [PLATFORM=HIP]]
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import unit
from ucmd_bridge.identity import Tracker
from ucmd_bridge.integrators import nose_hoover_chain
from ucmd_bridge.model import NonbondedSettings, build_system
from ucmd_bridge.openmm_backend import BAR_PER_KJMOL_NM3 as BAR
from ucmd_bridge.openmm_backend import read_plan, system_path
from ucmd_bridge.package import load as load_package
from ucmd_bridge.pressure import PressureProbe


def bar(value) -> float:
    return float(value.value_in_unit(unit.bar) if hasattr(value, "unit") else value)


def main(run_toml: str, steps: int, p_every: int = 500, platform_name: str = "HIP") -> None:
    plan = read_plan(Path(run_toml))
    top = load_package(system_path(plan))
    system, _ = build_system(top, NonbondedSettings.from_provider(plan.provider))
    system.addForce(mm.CMMotionRemover(plan.remove_com_every))
    barostat = mm.MonteCarloBarostat(1.0, plan.temperature_K, 0)
    system.addForce(barostat)
    integrator = nose_hoover_chain(
        plan.dt_ps, plan.temperature_K, float(plan.thermostat["tau_ps"]),
        int(plan.thermostat.get("chain_length", 3)), 3 * top.n_atoms - 3, False,
    )  # fmt: skip
    platform = mm.Platform.getPlatformByName(platform_name)
    props = (
        {"Precision": "double"} if platform_name in ("HIP", "CUDA", "OpenCL") else {}
    )
    ctx = mm.Context(system, integrator, platform, props)
    ctx.setPeriodicBoxVectors(*(mm.Vec3(*r) for r in top.cell_nm))
    ctx.setPositions(top.positions_nm)
    ctx.setVelocitiesToTemperature(
        float(plan.velocities["temperature_K"]), plan.seed % 2**31
    )
    v = ctx.getState(getVelocities=True).getVelocities(asNumpy=True)
    v = v.value_in_unit(unit.nanometer / unit.picosecond)
    v -= (top.masses_amu[:, None] * v).sum(0) / top.masses_amu.sum()
    ctx.setVelocities(v)

    probe = PressureProbe(system, barostat, platform, props)
    third = mm.Context(system, mm.VerletIntegrator(0.001), platform, props)
    mols = [list(a) for a in ctx.getMolecules()]
    mass = np.array(
        [
            system.getParticleMass(i).value_in_unit(unit.dalton)
            for i in range(top.n_atoms)
        ]
    )
    owner = np.zeros(top.n_atoms, int)
    for k, atoms in enumerate(mols):
        owner[atoms] = k
    tracker = Tracker(top.molecule_index, list(top.molecule_type))

    def energy_only(pos, vel, edge) -> float:
        volume = edge**3
        centre = np.array(
            [(mass[a, None] * pos[a]).sum(0) / mass[a].sum() for a in mols]
        )

        def u(rel: float) -> float:
            s = (1.0 + rel) ** (1.0 / 3.0)
            box = (
                mm.Vec3(edge * s, 0, 0),
                mm.Vec3(0, edge * s, 0),
                mm.Vec3(0, 0, edge * s),
            )
            third.setPeriodicBoxVectors(*box)
            third.setPositions(pos + ((s - 1.0) * centre)[owner])
            return (
                third.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(unit.kilojoule_per_mole)
            )

        eps = 1e-3
        dudv = (u(+eps) - u(-eps)) / (2.0 * eps * volume)
        kinetic = sum(
            mass[a].sum()
            * (((mass[a, None] * vel[a]).sum(0) / mass[a].sum()) ** 2).sum()
            for a in mols
        ) / (3.0 * volume)
        return (kinetic - dudv) * BAR

    rows, step = [], 0
    while step < steps:
        every = 10 if step * plan.dt_ps < 5.0 else 25
        n = min(every - step % every, steps - step)
        integrator.step(n)
        step += n
        want_p = step % p_every == 0
        st = ctx.getState(
            getPositions=True, getVelocities=want_p, enforcePeriodicBox=False
        )
        pos = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        tracker.update(np.asarray(pos))
        if want_p:
            vel = st.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond
            )
            box = st.getPeriodicBoxVectors()
            edge = box[0][0].value_in_unit(unit.nanometer)
            ctx.getState(getEnergy=True)
            stepping = bar(barostat.computeCurrentPressure(ctx))
            rows.append((step * plan.dt_ps, stepping, probe(pos, vel, box),
                         energy_only(np.asarray(pos), np.asarray(vel), edge)))  # fmt: skip
    r = np.array(rows)
    print(f"{plan.run_toml}: {steps} steps, {len(r)} pressure evaluations, "
          f"label exchanges {tracker.exchanges}")  # fmt: skip
    print(
        "      block (ps)   stepping    probe   energy-only   stepping-probe   probe-energy"
    )
    edges = np.linspace(0.0, r[-1, 0], 6)
    for lo, hi in itertools.pairwise(edges):
        k = (r[:, 0] > lo) & (r[:, 0] <= hi)
        print(f"  {lo:6.1f}-{hi:6.1f}  {r[k, 1].mean():9.1f} {r[k, 2].mean():9.1f} {r[k, 3].mean():11.1f} "
              f"{np.mean(r[k, 1] - r[k, 2]):14.1f} {np.mean(r[k, 2] - r[k, 3]):14.1f}")  # fmt: skip


if __name__ == "__main__":
    main(
        sys.argv[1],
        int(sys.argv[2]),
        int(sys.argv[3]) if len(sys.argv) > 3 else 500,
        *sys.argv[4:5],
    )
