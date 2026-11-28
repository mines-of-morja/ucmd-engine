"""The pressure tensor of the bridge (V-1, spec 0.1.1): finite differences of the energy under
affine strains in a context that never steps (pressure.py, TensorProbe).

OpenMM 8.5 has no virial, and its MonteCarloFlexibleBarostat.computeCurrentPressure changes
only the box element for the off-diagonal components, without moving the atoms: in an
orthorhombic box that element is zero, so XY, XZ and YZ hold the kinetic part alone
(docs/design/V-1-viscosity-gpu.md §2). These tests pin what the bridge's own estimator must
deliver: the atomic virial of the forces, as the engine computes it (units.md §4), with the
engine's virial-route tail, for any image of the atoms and for triclinic cells.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

mm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

from ucmd_bridge.model import ModelError, NonbondedSettings, build_system, tail_virial_v  # noqa: E402
from ucmd_bridge.package import load  # noqa: E402
from ucmd_bridge.pressure import BAR_PER_KJMOL_NM3, TensorProbe, bonded_groups  # noqa: E402

SIGMA, EPSILON, MASS = 0.30, 0.80, 12.0  # nm, kJ/mol, amu
BOND, K_BOND = 0.15, 2.0e5  # nm, kJ/mol/nm²
CUTOFF = 0.9
N_SIDE, SPACING = 4, 0.75
REFERENCE = mm.Platform.getPlatformByName("Reference")
SETTINGS = NonbondedSettings(cutoff_nm=CUTOFF, dispersion_correction=True, coulomb="pme",
                             ewald_tolerance=1e-6)  # fmt: skip
CUBIC = np.eye(3) * N_SIDE * SPACING
TRICLINIC = np.array([[3.0, 0.0, 0.0], [0.4, 2.9, 0.0], [-0.3, 0.5, 2.8]])


def dimer_package(tmp: Path, cell: np.ndarray, seed: int = 3) -> Path:
    """64 flexible LJ dimers (no charges) in `cell`, slightly off the bond length."""
    rng = np.random.default_rng(seed)
    doc = {
        "format": "ucmd-system", "spec_version": "0.1.0", "name": "lj-dimers",
        "units": {"length": "nm", "time": "ps", "mass": "amu", "energy": "kJ/mol", "charge": "e",
                  "temperature": "K", "angle": "deg"},
        "periodicity": [True, True, True], "cell_nm": cell.tolist(),
        "molecule_types": {"D2": {
            "formal_charge_e": 0,
            "atoms": [{"name": f"A{k}", "element": "C", "mass_amu": MASS, "charge_e": 0.0,
                       "type": "A"} for k in (1, 2)],
            "bonds": [{"atoms": [0, 1], "form": "harmonic",
                       "params": {"r0_nm": BOND, "k_kjmol_nm2": K_BOND}}],
        }},
        "components": [{"molecule_type": "D2", "count": N_SIDE**3}],
        "atom_count": 2 * N_SIDE**3,
        "forcefield": {"name": "test dimers", "combination_rule": "lorentz-berthelot",
                       "scale_14_lj": 0.5, "scale_14_coulomb": 0.5,
                       "atom_types": {"A": {"sigma_nm": SIGMA, "epsilon_kjmol": EPSILON}}},
    }  # fmt: skip
    grid = np.stack(np.meshgrid(*[np.arange(N_SIDE)] * 3, indexing="ij"), -1).reshape(-1, 3)
    x = []
    for site in grid:
        centre = (site + 0.5) / N_SIDE @ cell
        axis = rng.normal(size=3)
        axis *= BOND * rng.uniform(0.95, 1.05) / np.linalg.norm(axis)
        x += [centre - axis / 2, centre + axis / 2]
    x = np.array(x) + rng.normal(scale=0.02, size=(len(x), 3))
    pkg = tmp / "dimers.ucsys"
    pkg.mkdir()
    (pkg / "system.json").write_text(json.dumps(doc))
    np.save(pkg / "positions.npy", x)
    np.save(pkg / "velocities.npy", rng.normal(scale=0.5, size=x.shape))
    return pkg


def analytic_virial(x: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """W = Σ_pairs r_ij ⊗ F_ij (minimum image, LJ truncated at the cutoff, bonded pair excluded)
    + Σ_bonds + the engine's virial-route tail, in kJ/mol."""
    n = len(x)
    inv = np.linalg.inv(cell)
    w = np.zeros((3, 3))
    for i in range(n):
        d = x[i] - x[i + 1 :]
        s = d @ inv
        d = (s - np.round(s)) @ cell
        r2 = (d * d).sum(1)
        keep = r2 < CUTOFF**2
        if i % 2 == 0:
            keep[0] = False  # the bonded partner i + 1
        s6 = (SIGMA**2 / r2[keep]) ** 3
        f = d[keep] * (24.0 * EPSILON * (2.0 * s6 * s6 - s6) / r2[keep])[:, None]
        w += d[keep].T @ f
    for i in range(0, n, 2):
        d = x[i] - x[i + 1]
        r = np.linalg.norm(d)
        w += np.outer(d, -K_BOND * (r - BOND) * d / r)
    xc3 = (SIGMA / CUTOFF) ** 3
    tail = 16.0 * math.pi / 3.0 * n * n * EPSILON * SIGMA**3 * (2.0 * xc3**3 / 3.0 - xc3)
    return w + np.eye(3) * tail / abs(np.linalg.det(cell))


def probe_for(pkg: Path) -> tuple[TensorProbe, object]:
    top = load(pkg)
    system, _ = build_system(top, SETTINGS, for_virial=True)
    return TensorProbe(system, REFERENCE, {}, tail_virial_v(top, SETTINGS)), top


@pytest.mark.parametrize("cell", [CUBIC, TRICLINIC], ids=["cubic", "triclinic"])
def test_the_tensor_is_the_atomic_virial_of_the_forces(tmp_path: Path, cell) -> None:
    probe, top = probe_for(dimer_package(tmp_path, cell))
    x, v = top.positions_nm, top.velocities_nm_ps
    got = probe.tensor(x, v, cell)
    kinetic = (v * top.masses_amu[:, None]).T @ v
    want = (kinetic + analytic_virial(x, cell)) / abs(np.linalg.det(cell)) * BAR_PER_KJMOL_NM3
    print("P (bar):", got.round(3).tolist(), "max |Δ|:", np.abs(got - want).max())
    assert np.abs(want).max() > 100.0  # a test with something in it
    np.testing.assert_allclose(got, want, rtol=0, atol=0.1)  # bar
    np.testing.assert_allclose(got, got.T, rtol=0, atol=1e-9)  # symmetric up to rounding
    np.testing.assert_allclose(
        probe(x, v, cell), [got[0, 0], got[1, 1], got[2, 2], got[0, 1], got[0, 2], got[1, 2]]
    )


def test_the_tensor_does_not_depend_on_the_image_of_a_molecule(tmp_path: Path) -> None:
    probe, top = probe_for(dimer_package(tmp_path, TRICLINIC))
    x, v = top.positions_nm.copy(), top.velocities_nm_ps
    first = probe.tensor(x, v, TRICLINIC)
    # one dimer, whole, to another image (the bridge's positions are unwrapped: molecules stay
    # whole, and bonded terms are evaluated without periodic images)
    x[6:8] += TRICLINIC[1] - 2.0 * TRICLINIC[2]
    x[20:22] -= TRICLINIC[0]
    np.testing.assert_allclose(probe.tensor(x, v, TRICLINIC), first, rtol=0, atol=1e-6)


def test_the_probe_system_has_the_forces_of_the_model(tmp_path: Path) -> None:
    """Shifting the LJ energy inside the cutoff and dropping the tail leaves every force alone."""
    top = load(dimer_package(tmp_path, CUBIC))
    forces = []
    for for_virial in (False, True):
        system, _ = build_system(top, SETTINGS, for_virial=for_virial)
        ctx = mm.Context(system, mm.VerletIntegrator(0.001), REFERENCE)
        ctx.setPeriodicBoxVectors(*(mm.Vec3(*row) for row in CUBIC))
        ctx.setPositions(top.positions_nm)
        state = ctx.getState(getForces=True)
        forces.append(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    np.testing.assert_allclose(forces[1], forces[0], rtol=0, atol=1e-9)


def test_the_truncated_energy_would_add_the_cutoff_jumps(tmp_path: Path) -> None:
    """Why the probe needs its own system: differences of the truncated energy see each pair
    that crosses the cutoff (the impulsive term), the virial of the forces does not. On the
    model system the diagonal is off by much more than the probe's tolerance."""
    top = load(dimer_package(tmp_path, CUBIC))
    plain, _ = build_system(top, NonbondedSettings(cutoff_nm=CUTOFF, coulomb="pme"))
    probe = TensorProbe(plain, REFERENCE, {}, strain=1e-2)
    x, cell = top.positions_nm, CUBIC
    got = probe.virial(x, cell)
    xc3 = (SIGMA / CUTOFF) ** 3
    tail = 16.0 * math.pi / 3.0 * len(x) ** 2 * EPSILON * SIGMA**3 * (2.0 * xc3**3 / 3.0 - xc3)
    want = analytic_virial(x, cell) - np.eye(3) * tail / abs(np.linalg.det(cell))
    jump = np.abs(np.diag(got - want)).max() / abs(np.linalg.det(cell)) * BAR_PER_KJMOL_NM3
    print(f"largest diagonal difference of the truncated energy: {jump:.2f} bar")
    assert jump > 0.5  # five times the probe's tolerance (the configuration is seeded)


def test_constraints_are_refused() -> None:
    system = mm.System()
    for _ in range(2):
        system.addParticle(1.0)
    system.addConstraint(0, 1, 0.1)
    with pytest.raises(ModelError, match="constraints"):
        TensorProbe(system, REFERENCE, {})


def test_bonded_groups_are_the_connected_atoms() -> None:
    groups = bonded_groups(6, [(0, 1), (1, 2), (4, 5)])
    assert groups.tolist() == [0, 0, 0, 1, 2, 2]
