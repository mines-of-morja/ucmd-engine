"""E1.12: the same NVE production with OpenMM (Reference platform, double precision).

usage: python openmm_nve.py <equilibrated package> <out.npz>
Uses tools/oracle/oracle.py of the UnitcellMD umbrella repo for the identical LJ model
(potential shift at 0.85 nm, no dispersion correction).
"""

import sys
import time
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import unit

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "oracle"))
import oracle

package, out = Path(sys.argv[1]), Path(sys.argv[2])
pkg = oracle.load_package(package)
velocities = np.load(package / "velocities.npy")
system = oracle.lj_system(pkg, 0.85, "potential-shift")
integrator = mm.VerletIntegrator(0.004)
context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
context.setPositions(pkg.positions_nm)
context.setVelocities(velocities)
energies, frames, frame_ke = [], [], []
started = time.perf_counter()
for block in range(2501):  # 25000 steps, energies every 10, positions every 100
    state = context.getState(getEnergy=True, getPositions=(block % 10 == 0), enforcePeriodicBox=False)
    u = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    k = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    energies.append((block * 10, u, k))
    if block % 10 == 0:
        frames.append(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        frame_ke.append(k)
    if block < 2500:
        integrator.step(10)
seconds = time.perf_counter() - started
np.savez(
    out,
    energies=np.array(energies),
    frames=np.array(frames),
    frame_ke=np.array(frame_ke),
    box=pkg.cell_nm,
    seconds=seconds,
    openmm=mm.__version__,
)
print(f"OpenMM {mm.__version__} Reference: 25000 steps in {seconds:.1f} s")
