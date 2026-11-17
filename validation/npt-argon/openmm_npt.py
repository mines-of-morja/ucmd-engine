"""E2.7: the same argon NPT with OpenMM's MonteCarloBarostat (the oracle, ADR 0005).

usage: python openmm_npt.py <equilibrated package> <out.npz>
Same Hamiltonian as npt.toml (tools/oracle lj_system: LJ shifted at 0.85 nm, no tail correction);
Langevin thermostat (friction 1/ps), 4 fs, volume moves every 25 steps.
"""

import sys
import time
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import unit

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "oracle"))
import oracle

T_K, P_BAR, STEPS, EVERY = 94.4, 100.0, 137500, 25
package, out = Path(sys.argv[1]), Path(sys.argv[2])
pkg = oracle.load_package(package)
system = oracle.lj_system(pkg, 0.85, "potential-shift")
system.addForce(mm.MonteCarloBarostat(P_BAR * unit.bar, T_K * unit.kelvin, EVERY))
integrator = mm.LangevinMiddleIntegrator(T_K * unit.kelvin, 1.0 / unit.picosecond, 0.004)
integrator.setRandomNumberSeed(2027)
context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CPU"))
context.setPositions(pkg.positions_nm)
context.setVelocitiesToTemperature(T_K * unit.kelvin, 2027)
volumes, started = [], time.perf_counter()
for block in range(STEPS // EVERY + 1):
    state = context.getState()
    volumes.append((block * EVERY, state.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)))
    if block < STEPS // EVERY:
        integrator.step(EVERY)
seconds = time.perf_counter() - started
np.savez(out, volumes=np.array(volumes), seconds=seconds, openmm=mm.__version__,
         n=len(pkg.masses_amu), mass_amu=float(np.sum(pkg.masses_amu)))
print(f"OpenMM {mm.__version__} CPU: {STEPS} steps in {seconds:.1f} s")
