"""The engine's integrators as OpenMM CustomIntegrators (E3.7, ADR 0010).

OpenMM's built-in VerletIntegrator, NoseHooverIntegrator and LangevinMiddleIntegrator keep the
velocities half a step behind the positions. The engine uses full-step velocities, and its
Nosé–Hoover chain controls the full-step kinetic energy. At 1 fs with C–H bonds the half-step
kinetic energy is about 1.7 % higher, so OpenMM's thermostat would hold the full-step temperature
of a 7036-atom electrolyte at 288 K instead of 293 K: a different ensemble. These integrators are
the engine's operator splittings (PLAN.md §2.5, ucmd-md), step by step:

- velocity Verlet:       B(½) A(1) F B(½)
- Nosé–Hoover chain:     O(½) B(½) A(1) F B(½) O(½), the chain propagated with the fifth-order
                         Suzuki–Yoshida weights, masses Q₁ = N_f kT τ², Q_j = kT τ² (thermostat.rs)
- Langevin BAOAB:        B(½) A(½) O(1) A(½) F B(½), O exact: v ← c v + √((1 − c²) kT/m) R

so the velocities, the kinetic energy and the conserved quantity mean the same on both backends.
"""

from __future__ import annotations

import math

import openmm as mm

BOLTZMANN_KJMOL_K = 1.380649e-23 * 6.02214076e23 / 1000.0  # CODATA 2018, exact


def suzuki_yoshida5() -> list[float]:
    w = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
    return [w, w, 1.0 - 4.0 * w, w, w]


def _start(integ: mm.CustomIntegrator) -> None:
    # MonteCarloBarostat and CMMotionRemover act here (OpenMM's convention: first in the step)
    integ.addUpdateContextState()


def _kick(integ: mm.CustomIntegrator, fraction: float) -> None:
    integ.addComputePerDof("v", f"v + {fraction}*dt*f/m")


def velocity_verlet(dt_ps: float) -> mm.CustomIntegrator:
    integ = mm.CustomIntegrator(dt_ps)
    _start(integ)
    _kick(integ, 0.5)
    integ.addComputePerDof("x", "x + dt*v")
    _kick(integ, 0.5)
    return integ


def _chain(integ: mm.CustomIntegrator, m: int, half_dt_fraction: float) -> None:
    """thermostat.rs NoseHooverChain::apply over half_dt_fraction·dt, unrolled."""
    integ.addComputeSum("twoK", "m*v*v")
    integ.addComputeGlobal("scale", "1")

    def force(j: int) -> str:
        if j == 1:
            return "(twoK - dof*kT)/q1"
        return f"(qq{j - 1}*vxi{j - 1}*vxi{j - 1} - kT)/qq{j}"

    for w in suzuki_yoshida5():
        t = w * half_dt_fraction
        integ.addComputeGlobal(f"vxi{m}", f"vxi{m} + 0.5*{t}*dt*{force(m)}")
        for j in range(m - 1, 0, -1):
            s = f"exp(-0.25*{t}*dt*vxi{j + 1})"
            integ.addComputeGlobal(f"vxi{j}", f"vxi{j}*{s}")
            integ.addComputeGlobal(f"vxi{j}", f"vxi{j} + 0.5*{t}*dt*{force(j)}")
            integ.addComputeGlobal(f"vxi{j}", f"vxi{j}*{s}")
        integ.addComputeGlobal("s1", f"exp(-{t}*dt*vxi1)")
        integ.addComputeGlobal("scale", "scale*s1")
        integ.addComputeGlobal("twoK", "twoK*s1*s1")
        for j in range(1, m + 1):
            integ.addComputeGlobal(f"xi{j}", f"xi{j} + {t}*dt*vxi{j}")
        for j in range(1, m):
            s = f"exp(-0.25*{t}*dt*vxi{j + 1})"
            integ.addComputeGlobal(f"vxi{j}", f"vxi{j}*{s}")
            integ.addComputeGlobal(f"vxi{j}", f"vxi{j} + 0.5*{t}*dt*{force(j)}")
            integ.addComputeGlobal(f"vxi{j}", f"vxi{j}*{s}")
        integ.addComputeGlobal(f"vxi{m}", f"vxi{m} + 0.5*{t}*dt*{force(m)}")
    integ.addComputePerDof("v", "v*scale")


def nose_hoover_chain(dt_ps: float, temperature_k: float, tau_ps: float, chain_length: int,
                      dof: float) -> mm.CustomIntegrator:  # fmt: skip
    m = max(1, chain_length)
    kt = BOLTZMANN_KJMOL_K * temperature_k
    integ = mm.CustomIntegrator(dt_ps)
    integ.addGlobalVariable("kT", kt)
    integ.addGlobalVariable("dof", dof)
    integ.addGlobalVariable("q1", dof * kt * tau_ps**2)
    for j in range(1, m + 1):
        integ.addGlobalVariable(f"qq{j}", (dof if j == 1 else 1.0) * kt * tau_ps**2)
        integ.addGlobalVariable(f"xi{j}", 0.0)
        integ.addGlobalVariable(f"vxi{j}", 0.0)
    for name in ("twoK", "scale", "s1"):
        integ.addGlobalVariable(name, 0.0)
    _start(integ)
    _chain(integ, m, 0.5)
    _kick(integ, 0.5)
    integ.addComputePerDof("x", "x + dt*v")
    _kick(integ, 0.5)
    _chain(integ, m, 0.5)
    return integ


def bath_energy(integ: mm.CustomIntegrator, chain_length: int) -> float:
    """Σ ½Q_j v_j² + N_f kT ξ₁ + kT Σ_{j≥2} ξ_j (kJ/mol), as thermostat.rs."""
    g = {integ.getGlobalVariableName(i): integ.getGlobalVariable(i)
         for i in range(integ.getNumGlobalVariables())}  # fmt: skip
    m = max(1, chain_length)
    kinetic = sum(0.5 * g[f"qq{j}"] * g[f"vxi{j}"] ** 2 for j in range(1, m + 1))
    potential = g["dof"] * g["kT"] * g["xi1"] + g["kT"] * sum(g[f"xi{j}"] for j in range(2, m + 1))
    return kinetic + potential


def langevin_baoab(
    dt_ps: float, temperature_k: float, friction_per_ps: float
) -> mm.CustomIntegrator:
    integ = mm.CustomIntegrator(dt_ps)
    integ.addGlobalVariable("kT", BOLTZMANN_KJMOL_K * temperature_k)
    integ.addGlobalVariable("c", math.exp(-friction_per_ps * dt_ps))
    _start(integ)
    _kick(integ, 0.5)
    integ.addComputePerDof("x", "x + 0.5*dt*v")
    integ.addComputePerDof("v", "c*v + sqrt((1 - c*c)*kT/m)*gaussian")
    integ.addComputePerDof("x", "x + 0.5*dt*v")
    _kick(integ, 0.5)
    return integ
