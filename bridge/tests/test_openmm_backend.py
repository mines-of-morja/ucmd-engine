"""E3.7: `ucmd run` with `[backend] kind = "openmm"` (here on OpenMM's CPU platform).

The whole chain: `ucmd` computes the reference single point and hands over to `ucmd-openmm`,
which checks conformance, runs, writes spec outputs and restarts. Needs the `ucmd` binary
(cargo build) and the spec fixtures (../../spec, or $UCMD_SPEC_DIR).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from ucmd_bridge.openmm_backend import BOLTZMANN_KJMOL_K

ENGINE = Path(__file__).resolve().parents[2]
SPEC = Path(os.environ.get("UCMD_SPEC_DIR", ENGINE.parent / "spec"))
UCMD = next(
    (p for p in (ENGINE / "target/release/ucmd", ENGINE / "target/debug/ucmd") if p.is_file()),
    None,
)
BRIDGE_EXE = Path(sys.executable).with_name("ucmd-openmm")

pytestmark = pytest.mark.skipif(
    UCMD is None or not (SPEC / "fixtures").is_dir() or not BRIDGE_EXE.is_file(),
    reason="needs the ucmd binary, the spec fixtures and the installed bridge",
)

ARGON = """\
spec_version = "0.1.0"
system = "system.ucsys"
title = "bridge test: argon"

[run]
integrator = "velocity-verlet"
timestep_fs = 4.0
steps = {steps}
seed = 11

[velocities]
generate = "maxwell-boltzmann"
temperature_K = 94.4

[momentum]
remove_com_every = 100

[thermostat]
kind = "{thermostat}"
temperature_K = 94.4
tau_ps = 0.5

[[providers]]
kind = "force-field"
cutoff_nm = 0.85
dispersion_correction = true

[output]
directory = "out"

[output.thermo]
file = "thermo.tsv"
every = 10

[output.trajectory]
file = "trajectory.h5"
every = 50

[output.checkpoint]
file = "checkpoint.ucchk"
every = 100
"""


def setup_run(tmp: Path, fixture: str, text: str) -> Path:
    shutil.copytree(SPEC / "fixtures/valid" / fixture, tmp / "system.ucsys")
    (tmp / "run.toml").write_text(text)
    profile = tmp / "machine.toml"
    profile.write_text(
        f'[executables]\nucmd-openmm = "{BRIDGE_EXE}"\n\n[backend]\nkind = "openmm"\n'
        'platform = "Reference"\n'
    )
    return profile


def ucmd(tmp: Path, profile: Path, *args: str, wait: bool = True):
    env = {**os.environ, "UCMD_MACHINE_PROFILE": str(profile)}
    command = [str(UCMD), "run", "run.toml", *args]
    if not wait:
        return subprocess.Popen(command, cwd=tmp, env=env, stderr=subprocess.PIPE, text=True)
    return subprocess.run(command, cwd=tmp, env=env, capture_output=True, text=True)


def thermo(tmp: Path) -> tuple[list[str], np.ndarray]:
    with (tmp / "out/thermo.tsv").open() as fh:
        names = fh.readline().lstrip("# ").strip().split("\t")
        return names, np.loadtxt(fh, ndmin=2)


def test_argon_runs_through_ucmd_with_conformance_and_spec_outputs(tmp_path: Path) -> None:
    profile = setup_run(
        tmp_path, "argon-864", ARGON.format(steps=200, thermostat="nose-hoover-chain")
    )
    r = ucmd(tmp_path, profile)
    assert r.returncode == 0, r.stderr
    assert "backend openmm" in r.stderr and "conformance with ucmd" in r.stderr
    line = next(x for x in r.stderr.splitlines() if "conformance with" in x)
    energy_rel = float(line.split("energy ")[1].split(",")[0])
    # double precision on both sides; OpenMM integrates the tail correction of a custom LJ
    # numerically (2.7e-8 here), the engine analytically
    assert energy_rel < 1e-7
    names, rows = thermo(tmp_path)
    assert rows[:, 0].tolist() == list(range(0, 201, 10))
    t = rows[:, names.index("temperature_K")]
    assert 40 < t[-1] < 130  # the lattice start melts and absorbs heat; τ = 0.5 ps catches up
    p = rows[:, names.index("pressure_bar")]
    assert np.isfinite(p[0]) and np.isnan(p[1])  # pressure every 500 steps (step 0 only)
    from ucmd_spec.trajectory import validate_trajectory

    report = validate_trajectory(tmp_path / "out/trajectory.h5")
    assert report.ok, report.issues
    import h5py

    with h5py.File(tmp_path / "out/trajectory.h5") as fh:
        assert fh["particles/all/position/step"][()].tolist() == [0, 50, 100, 150, 200]
        backend = fh["parameters/ucmd/backend"].attrs
        assert backend["kind"] == "openmm" and backend["platform"] == "Reference"
        assert backend["conformance_force_relative"] < 1e-8
    assert (tmp_path / "out/backend-reference/reference.json").is_file()


def test_restart_appends_from_the_checkpoint(tmp_path: Path) -> None:
    text = ARGON.format(steps=100, thermostat="nose-hoover-chain")
    profile = setup_run(tmp_path, "argon-864", text)
    assert ucmd(tmp_path, profile).returncode == 0
    (tmp_path / "run.toml").write_text(text.replace("steps = 100", "steps = 200"))
    r = ucmd(tmp_path, profile, "--restart", "out/checkpoint.ucchk")
    assert r.returncode == 0, r.stderr
    assert "(restart)" in r.stderr
    assert "identity: 0 molecule labels corrected" in r.stderr  # the CPU does not reorder
    _names, rows = thermo(tmp_path)
    assert rows[:, 0].tolist() == list(range(0, 201, 10))  # no duplicate rows
    import h5py

    with h5py.File(tmp_path / "out/trajectory.h5") as fh:
        assert fh["particles/all/position/step"][()].tolist() == [0, 50, 100, 150, 200]
        assert "restarts" in fh["parameters/ucmd"]
        assert "identity" in fh["parameters/ucmd/backend"].attrs["identity_tracking"]
    from ucmd_bridge.openmm_backend import read_checkpoint

    # the molecule labels travel with the state (identity.py); argon: 864 one-atom molecules
    meta, _blob = read_checkpoint(tmp_path / "out/checkpoint.ucchk")
    assert meta["step"] == 200
    assert meta["identity_molecule_slots"] == list(range(864))


def test_charged_crystal_with_pme_agrees(tmp_path: Path) -> None:
    text = (
        ARGON.format(steps=20, thermostat="nose-hoover-chain")
        .replace("temperature_K = 94.4", "temperature_K = 300.0")
        .replace(
            "cutoff_nm = 0.85\ndispersion_correction = true", 'cutoff_nm = 0.35\ncoulomb = "pme"'
        )
        .replace("timestep_fs = 4.0", "timestep_fs = 1.0")
    )
    profile = setup_run(tmp_path, "nacl-crystal", text)
    r = ucmd(tmp_path, profile)
    assert r.returncode == 0, r.stderr
    line = next(x for x in r.stderr.splitlines() if "conformance with" in x)
    assert float(line.split("forces ")[1].split(" ")[0]) < 1e-6


def test_the_tail_correction_of_several_types_agrees(tmp_path: Path) -> None:
    """2026-11-21, SPC/Fw water on HIP: the conformance energy failed by 6.3e-6 in double precision
    (forces 7.6e-8). OpenMM's long-range correction counts N(N+1)/2 pairs over all atoms,
    including those without LJ; the engine's is 2π/V Σ_ab n_a n_b ∫ r² u_ab (Allen & Tildesley).
    The two agree for one type (argon) and differ by O(1/N) otherwise. The bridge must reproduce
    the engine's tail. Two SPC/Fw molecules in a periodic 2 nm box: 2 O with LJ among 6 atoms,
    where OpenMM's own count gives a tail (n_O + 1)/n_O · N/(N + 1) = 1.29 times the engine's."""
    text = (
        ARGON.format(steps=20, thermostat="nose-hoover-chain")
        .replace("temperature_K = 94.4", "temperature_K = 300.0")
        .replace(
            "cutoff_nm = 0.85\ndispersion_correction = true",
            'cutoff_nm = 0.6\ncoulomb = "pme"\ndispersion_correction = true',
        )
        .replace("timestep_fs = 4.0", "timestep_fs = 0.5")
    )
    profile = setup_run(tmp_path, "water-2", text)
    doc_path = tmp_path / "system.ucsys/system.json"
    doc = json.loads(doc_path.read_text())
    doc["periodicity"], doc["cell_nm"] = [True] * 3, [[2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.0]]
    doc_path.write_text(json.dumps(doc))
    r = ucmd(tmp_path, profile)
    assert r.returncode == 0, r.stderr
    line = next(x for x in r.stderr.splitlines() if "conformance with" in x)
    # the tail is -0.013 kJ/mol here, OpenMM's own count off by 0.004 kJ/mol (2e-4 of the
    # energy); OpenMM integrates the custom tail numerically to about 3e-7 of itself
    assert float(line.split("energy ")[1].split(",")[0]) < 1e-8


def test_csvr_and_a_wrong_model_are_refused(tmp_path: Path) -> None:
    profile = setup_run(tmp_path, "argon-864", ARGON.format(steps=20, thermostat="csvr"))
    r = ucmd(tmp_path, profile)
    assert r.returncode == 2 and "CSVR has no OpenMM equivalent" in r.stderr
    # a reference the backend cannot reproduce: the run must not start
    (tmp_path / "run.toml").write_text(ARGON.format(steps=20, thermostat="nose-hoover-chain"))
    env = {**os.environ, "UCMD_MACHINE_PROFILE": str(profile)}
    assert ucmd(tmp_path, profile).returncode == 0
    ref = tmp_path / "out/backend-reference/reference.json"
    ref.write_text(ref.read_text().replace('"energy_kjmol": -', '"energy_kjmol": -1'))
    r = subprocess.run(
        [str(BRIDGE_EXE), "run", "run.toml", "--reference", "out/backend-reference",
         "--platform", "Reference"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 2 and "conformance check failed" in r.stderr


WATER_BOX = """\
spec_version = "0.1.3"
system = "system.ucsys"
title = "bridge test: flexible water, pressure tensor"

[run]
integrator = "velocity-verlet"
timestep_fs = 0.5
steps = 20
seed = 3

[[providers]]
kind = "force-field"
cutoff_nm = 0.7
coulomb = "pme"
ewald_tolerance = 1e-6
dispersion_correction = true

[output]
directory = "{out}"

[output.thermo]
file = "thermo.tsv"
every = 2
pressure_tensor = true
pressure_molecular = {molecular}
"""
TENSOR = ["pressure_xx_bar", "pressure_yy_bar", "pressure_zz_bar", "pressure_xy_bar",
          "pressure_xz_bar", "pressure_yz_bar"]  # fmt: skip


def _water_box(tmp: Path, n_side: int = 6, spacing: float = 0.31, seed: int = 8) -> None:
    """n_side³ flexible SPC/Fw molecules (the water-2 fixture's force field) on a lattice with
    random orientations, a little off the equilibrium geometry, and 300 K velocities."""
    rng = np.random.default_rng(seed)
    pkg = tmp / "system.ucsys"
    doc = json.loads((pkg / "system.json").read_text())
    n = n_side**3
    box = n_side * spacing
    doc.update(periodicity=[True, True, True], cell_nm=(np.eye(3) * box).tolist(),
               components=[{"molecule_type": "SPCFw", "count": n}], atom_count=3 * n)  # fmt: skip
    (pkg / "system.json").write_text(json.dumps(doc))
    r_oh, half = 0.1012, np.radians(113.24) / 2
    local = np.array([[0, 0, 0], [r_oh * np.sin(half), r_oh * np.cos(half), 0],
                      [-r_oh * np.sin(half), r_oh * np.cos(half), 0]])  # fmt: skip
    grid = np.stack(np.meshgrid(*[np.arange(n_side)] * 3, indexing="ij"), -1).reshape(-1, 3)
    x = []
    for site in grid:
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        x += list((site + 0.5) * spacing + local @ q.T + rng.normal(scale=0.003, size=(3, 3)))
    np.save(pkg / "positions.npy", np.array(x))
    masses = np.tile([15.9994, 1.008, 1.008], n)
    kt = BOLTZMANN_KJMOL_K * 300.0
    np.save(pkg / "velocities.npy", rng.normal(size=(3 * n, 3)) * np.sqrt(kt / masses)[:, None])


def test_the_pressure_tensor_equals_the_engines(tmp_path: Path) -> None:
    """V-1 acceptance (design §6): flexible water with PME and the LJ tail, the same start and
    the same velocity Verlet on both backends: every tensor column within 1 bar of the CPU
    engine's at every row, pressure_bar its trace / 3, the molecular pressure in its own
    column."""
    profile = setup_run(tmp_path, "water-2", WATER_BOX.format(out="out", molecular="true"))
    _water_box(tmp_path)
    r = ucmd(tmp_path, profile)
    assert r.returncode == 0, r.stderr
    (tmp_path / "run.toml").write_text(WATER_BOX.format(out="cpu", molecular="false"))
    cpu_profile = tmp_path / "cpu.toml"
    cpu_profile.write_text("threads = 1\n")
    c = ucmd(tmp_path, cpu_profile)
    assert c.returncode == 0, c.stderr
    names, gpu = thermo(tmp_path)
    with (tmp_path / "cpu/thermo.tsv").open() as fh:
        cpu_names = fh.readline().lstrip("# ").strip().split("\t")
        cpu = np.loadtxt(fh, ndmin=2)
    assert names[-7:] == [*TENSOR, "pressure_molecular_bar"]
    got = gpu[:, [names.index(k) for k in TENSOR]]
    want = cpu[:, [cpu_names.index(k) for k in TENSOR]]
    print("bridge − engine, bar, per column:", np.abs(got - want).max(0).round(4).tolist())
    print("largest |P_ab| of the engine:", np.abs(want).max().round(1))
    assert np.abs(want[:, 3:]).max() > 100.0  # shear components with something in them
    np.testing.assert_allclose(got, want, rtol=0, atol=1.0)  # bar
    np.testing.assert_allclose(
        gpu[:, names.index("pressure_bar")], got[:, :3].mean(1), rtol=1e-9, atol=1e-6
    )
    assert np.isfinite(gpu[0, names.index("pressure_molecular_bar")])


def test_the_pressure_tensor_with_constraints_is_refused(tmp_path: Path) -> None:
    """The tensor comes from energies; a constraint virial is not among them."""
    profile = setup_run(tmp_path, "water-2", WATER_BOX.format(out="out", molecular="false"))
    _water_box(tmp_path)
    pkg = tmp_path / "system.ucsys"
    doc = json.loads((pkg / "system.json").read_text())
    doc["molecule_types"]["SPCFw"]["constraints"] = [{"atoms": [0, 1], "length_nm": 0.1012}]
    (pkg / "system.json").write_text(json.dumps(doc))
    r = ucmd(tmp_path, profile)
    assert r.returncode != 0 and "E_UNSUPPORTED" in r.stderr, r.stderr
    assert not (tmp_path / "out/thermo.tsv").exists()


def test_the_molecular_pressure_column(tmp_path: Path) -> None:
    """Spec 0.1.3: [output.thermo] pressure_molecular adds the last column. OpenMM's pressure is
    molecular, so it repeats pressure_bar, NaN rows included; a run that cannot compute a pressure
    is refused before it starts."""
    text = ARGON.format(steps=20, thermostat="nose-hoover-chain").replace(
        '[output.thermo]\nfile = "thermo.tsv"\nevery = 10\n',
        '[output.thermo]\nfile = "thermo.tsv"\nevery = 10\npressure_molecular = true\n',
    )
    assert "pressure_molecular = true" in text
    profile = setup_run(tmp_path, "argon-864", text.replace('"0.1.0"', '"0.1.3"', 1))
    r = ucmd(tmp_path, profile)
    assert r.returncode == 0, r.stderr
    names, rows = thermo(tmp_path)
    assert names[-1] == "pressure_molecular_bar"
    p = rows[:, names.index("pressure_bar")]
    assert np.isfinite(p[0])
    np.testing.assert_array_equal(rows[:, -1], p)


def test_the_molecular_pressure_without_a_pressure_is_refused(tmp_path: Path) -> None:
    """With --pressure-every 0 the column would hold only NaN: the run must not start."""
    import argparse

    from ucmd_bridge import openmm_backend

    text = ARGON.format(steps=20, thermostat="nose-hoover-chain").replace(
        '[output.thermo]\nfile = "thermo.tsv"\nevery = 10\n',
        '[output.thermo]\nfile = "thermo.tsv"\nevery = 10\npressure_molecular = true\n',
    )
    setup_run(tmp_path, "argon-864", text.replace('"0.1.0"', '"0.1.3"', 1))
    args = argparse.Namespace(run_toml=str(tmp_path / "run.toml"), pressure_every=0)
    with pytest.raises(openmm_backend.ConfigError, match="E_UNSUPPORTED"):
        openmm_backend.run(args)
    assert not (tmp_path / "out/thermo.tsv").exists()


def test_sigterm_stops_gracefully_with_a_checkpoint(tmp_path: Path) -> None:
    profile = setup_run(
        tmp_path, "argon-864", ARGON.format(steps=500000, thermostat="nose-hoover-chain")
    )
    p = ucmd(tmp_path, profile, wait=False)
    deadline = time.time() + 60
    while time.time() < deadline:
        if (tmp_path / "out/thermo.tsv").is_file() and len(
            (tmp_path / "out/thermo.tsv").read_text().splitlines()
        ) > 5:
            break
        time.sleep(0.2)
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=60)
    assert p.returncode == 143, p.stderr.read()
    assert (tmp_path / "out/checkpoint.ucchk").is_file()


def test_the_nose_hoover_chain_is_the_engines(tmp_path: Path) -> None:
    """From the same state, CPU engine and bridge give the same thermo table: the CustomIntegrator
    is the engine's operator splitting with full-step velocities and the same chain."""
    text = (
        ARGON.format(steps=100, thermostat="nose-hoover-chain")
        .replace('[velocities]\ngenerate = "maxwell-boltzmann"\ntemperature_K = 94.4\n\n', "")
        .replace("tau_ps = 0.5", "tau_ps = 0.1")
    )
    rng = np.random.default_rng(3)
    for name in ("cpu", "gpu"):
        d = tmp_path / name
        d.mkdir()
        profile = setup_run(d, "argon-864", text)
        n = len(np.load(d / "system.ucsys/positions.npy"))
        v = rng.normal(0.0, 0.14, (n, 3)) if name == "cpu" else np.load(tmp_path / "cpu/v.npy")
        v -= v.mean(0)
        np.save(d / "system.ucsys/velocities.npy", v)
        np.save(d / "v.npy", v)
        if name == "cpu":
            r = subprocess.run(
                [str(UCMD), "run", "run.toml"], cwd=d, capture_output=True, text=True
            )
        else:
            r = ucmd(d, profile)
        assert r.returncode == 0, r.stderr
    names, cpu = thermo(tmp_path / "cpu")
    _, gpu = thermo(tmp_path / "gpu")
    for column in ("temperature_K", "potential_kjmol", "conserved_energy_kjmol"):
        k = names.index(column)
        assert np.allclose(gpu[:, k], cpu[:, k], rtol=1e-6), column


def _nve_with_vectors(steps: int, every: int) -> str:
    """Argon NVE with velocities and forces in the trajectory, from the package's velocities."""
    text = (
        ARGON.format(steps=steps, thermostat="nose-hoover-chain")
        .replace('[velocities]\ngenerate = "maxwell-boltzmann"\ntemperature_K = 94.4\n\n', "")
        .replace(
            '[thermostat]\nkind = "nose-hoover-chain"\ntemperature_K = 94.4\ntau_ps = 0.5\n\n', ""
        )
        .replace("every = 50", f"every = {every}\nvelocities = true\nforces = true")
    )
    assert "[thermostat]" not in text and "[velocities]" not in text and "forces = true" in text
    return text


def test_velocities_and_forces_are_the_engines(tmp_path: Path) -> None:
    """[output.trajectory] velocities and forces: from the same start the bridge writes the same
    full-step velocities and forces as the CPU engine, in the spec's datasets."""
    import h5py
    from ucmd_spec.trajectory import validate_trajectory

    text = _nve_with_vectors(steps=40, every=10)
    rng = np.random.default_rng(3)
    for name in ("cpu", "openmm"):
        d = tmp_path / name
        d.mkdir()
        profile = setup_run(d, "argon-864", text)
        v = rng.normal(0.0, 0.14, (864, 3)) if name == "cpu" else np.load(tmp_path / "cpu/v.npy")
        v -= v.mean(0)
        np.save(d / "system.ucsys/velocities.npy", v)
        np.save(d / "v.npy", v)
        if name == "cpu":
            profile.write_text("threads = 1\n")
        r = ucmd(d, profile)
        assert r.returncode == 0, r.stderr
    report = validate_trajectory(tmp_path / "openmm/out/trajectory.h5")
    assert report.ok, report.issues
    with (
        h5py.File(tmp_path / "cpu/out/trajectory.h5") as cpu,
        h5py.File(tmp_path / "openmm/out/trajectory.h5") as omm,
    ):
        for name, unit in (("velocity", "nm ps-1"), ("force", "kJ mol-1 nm-1")):
            a, b = cpu[f"particles/all/{name}"], omm[f"particles/all/{name}"]
            assert b["value"].attrs["unit"] == a["value"].attrs["unit"] == unit
            assert b["step"][()].tolist() == a["step"][()].tolist() == [0, 10, 20, 30, 40]
            x, y = a["value"][()], b["value"][()]
            worst = np.abs(x - y).max() / np.abs(x).max()
            print(f"{name}: worst difference {worst:.1e} of the largest component")
            assert worst < 1e-6
        # the stored velocities are those of the logged kinetic energy (full step)
        m = omm["particles/all/mass"][()]
        v = omm["particles/all/velocity/value"][()]
        k = 0.5 * (m[None, :, None] * v**2).sum((1, 2))
        assert np.allclose(k, omm["observables/kinetic_energy/value"][()], rtol=1e-12)


def test_velocities_restart_and_need_their_dataset(tmp_path: Path) -> None:
    """A restart cuts and continues the velocity series; one that asks for velocities in a file
    without them is refused instead of writing a partial series."""
    import h5py

    text = ARGON.format(steps=100, thermostat="nose-hoover-chain")
    with_v = text.replace("every = 50", "every = 50\nvelocities = true")
    profile = setup_run(tmp_path, "argon-864", with_v)
    assert ucmd(tmp_path, profile).returncode == 0
    (tmp_path / "run.toml").write_text(with_v.replace("steps = 100", "steps = 200"))
    r = ucmd(tmp_path, profile, "--restart", "out/checkpoint.ucchk")
    assert r.returncode == 0, r.stderr
    with h5py.File(tmp_path / "out/trajectory.h5", "a") as fh:
        g = fh["particles/all/velocity"]
        assert g["step"][()].tolist() == [0, 50, 100, 150, 200]
        assert g["value"].shape == (5, 864, 3) and np.abs(g["value"][4]).max() > 0
        del fh["particles/all/velocity"]
    (tmp_path / "run.toml").write_text(with_v.replace("steps = 100", "steps = 300"))
    r = ucmd(tmp_path, profile, "--restart", "out/checkpoint.ucchk")
    assert r.returncode != 0 and "velocity" in r.stderr, r.stderr
    assert "Traceback" not in r.stderr, r.stderr


def test_an_xyz_trajectory_is_written(tmp_path: Path) -> None:
    """Every .xyz run failed at its first frame (XyzWriter.write was called with the thermo row)."""
    text = ARGON.format(steps=20, thermostat="nose-hoover-chain").replace(
        'file = "trajectory.h5"\nevery = 50', 'file = "t.xyz"\nevery = 10'
    )
    profile = setup_run(tmp_path, "argon-864", text)
    r = ucmd(tmp_path, profile)
    assert r.returncode == 0, r.stderr
    lines = (tmp_path / "out/t.xyz").read_text().splitlines()
    assert len(lines) == 3 * 866 and [int(x) for x in lines[::866]] == [864] * 3


@pytest.mark.parametrize("key", ["velocities", "forces"])
def test_an_xyz_trajectory_refuses_velocities_and_forces(tmp_path: Path, key: str) -> None:
    """The viewer format holds positions only; the request must not be dropped (invariant 7)."""
    text = ARGON.format(steps=20, thermostat="nose-hoover-chain").replace(
        'file = "trajectory.h5"\nevery = 50', f'file = "t.xyz"\nevery = 10\n{key} = true'
    )
    assert f"{key} = true" in text
    profile = setup_run(tmp_path, "argon-864", text)
    r = ucmd(tmp_path, profile)
    assert r.returncode != 0 and "E_UNSUPPORTED" in r.stderr and key in r.stderr, r.stderr
    assert not (tmp_path / "out/t.xyz").exists()


def test_mixed_precision_needs_a_thermostat(tmp_path: Path) -> None:
    """ADR 0012: mixed precision for thermostatted production runs only; NVE runs use double."""
    text = ARGON.format(steps=20, thermostat="nose-hoover-chain").replace(
        '[thermostat]\nkind = "nose-hoover-chain"\ntemperature_K = 94.4\ntau_ps = 0.5\n', ""
    )
    assert "[thermostat]" not in text
    profile = setup_run(tmp_path, "argon-864", text)
    profile.write_text(profile.read_text() + 'precision = "mixed"\n')
    r = ucmd(tmp_path, profile)
    assert r.returncode == 2 and "ADR 0012" in r.stderr


def test_probe_runs_a_short_job() -> None:
    import json as _json

    r = subprocess.run([str(BRIDGE_EXE), "probe", "--platform", "Reference", "--steps", "5"],
                       capture_output=True, text=True)  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert _json.loads(r.stdout)["ok"] is True


def test_the_emergency_checkpoint_holds_the_last_good_state(tmp_path: Path) -> None:
    """A provider failure writes the state of the last trajectory frame with that frame's own
    step and molecule labels, not the step at which the run failed (fixed 2026-11-20)."""
    text = ARGON.format(steps=2000, thermostat="nose-hoover-chain").replace(
        "timestep_fs = 4.0", "timestep_fs = 60.0"
    )  # far beyond stability for argon: the run fails with non-finite numbers
    profile = setup_run(tmp_path, "argon-864", text)
    r = ucmd(tmp_path, profile)
    assert r.returncode != 0
    import h5py

    from ucmd_bridge.openmm_backend import read_checkpoint

    meta, _blob = read_checkpoint(tmp_path / "out/checkpoint.emergency.ucchk")
    with h5py.File(tmp_path / "out/trajectory.h5") as fh:
        steps = fh["particles/all/position/step"][()].tolist()
    assert meta["step"] == steps[-1]
    assert meta["identity_molecule_slots"] == list(range(864))


RIGID_WATER = """\
spec_version = "0.1.0"
system = "system.ucsys"

[run]
integrator = "velocity-verlet"
timestep_fs = 1.0
steps = 2000
seed = 1

[[providers]]
kind = "force-field"
coulomb = "direct"

[output]
directory = "{out}"

[output.thermo]
file = "thermo.tsv"
every = 10

[output.trajectory]
file = "trajectory.h5"
every = 100
"""


def _rigid_water(tmp: Path) -> np.ndarray:
    """water-2 held rigid at the SPC/Fw equilibrium geometry (three distances per molecule), with
    both molecules placed exactly on it and fixed velocities, so both backends start alike."""
    pkg = tmp / "system.ucsys"
    doc = json.loads((pkg / "system.json").read_text())
    r_oh, theta = 0.1012, np.radians(113.24)
    r_hh = 2 * r_oh * np.sin(theta / 2)
    doc["molecule_types"]["SPCFw"]["constraints"] = [
        {"atoms": [0, 1], "length_nm": r_oh},
        {"atoms": [0, 2], "length_nm": r_oh},
        {"atoms": [1, 2], "length_nm": r_hh},
    ]
    (pkg / "system.json").write_text(json.dumps(doc))
    h = theta / 2
    x = []
    for o, flip in ((np.zeros(3), 1.0), (np.array([0.5, 0.1, 0.0]), -1.0)):
        x += [
            o,
            o + np.array([r_oh * np.sin(h), flip * r_oh * np.cos(h), 0.0]),
            o + np.array([-r_oh * np.sin(h), flip * r_oh * np.cos(h), 0.0]),
        ]
    np.save(pkg / "positions.npy", np.array(x))
    rng = np.random.default_rng(5)
    np.save(pkg / "velocities.npy", rng.normal(0.0, 0.5, (6, 3)))
    return np.array(
        [[0, 1, r_oh], [0, 2, r_oh], [1, 2, r_hh], [3, 4, r_oh], [3, 5, r_oh], [4, 5, r_hh]]
    )


def test_constraints_run_on_openmm_like_the_reference_backend(tmp_path: Path) -> None:
    """E2.5 step 3: OpenMM gets the constraints (RATTLE in the CustomIntegrator). From the same
    start the two backends agree at first, and OpenMM keeps the constraints and the energy."""
    import h5py

    profile = setup_run(tmp_path, "water-2", RIGID_WATER.format(out="out"))
    pairs = _rigid_water(tmp_path)
    r = ucmd(tmp_path, profile)
    assert r.returncode == 0, r.stderr
    (tmp_path / "run.toml").write_text(RIGID_WATER.format(out="cpu"))
    cpu_profile = tmp_path / "cpu.toml"
    cpu_profile.write_text("threads = 1\n")
    c = ucmd(tmp_path, cpu_profile)
    assert c.returncode == 0, c.stderr
    names, gpu = thermo(tmp_path)
    cpu = np.loadtxt(tmp_path / "cpu/thermo.tsv", ndmin=2)
    u, e = names.index("potential_kjmol"), names.index("total_kjmol")
    early = slice(0, 21)  # the first 200 fs
    diff = np.abs(gpu[early, u] - cpu[early, u]).max()
    exchanged = np.ptp(gpu[:, u])
    spread = np.ptp(gpu[:, e])
    print(
        f"rigid water: |ΔU| over 200 fs {diff:.2e}, exchanged {exchanged:.2f}, spread {spread:.2e}"
    )
    assert diff < 1e-6
    assert exchanged > 1.0 and spread < 0.01 * exchanged
    with h5py.File(tmp_path / "out/trajectory.h5") as fh:
        x = fh["particles/all/position/value"][()]
        assert "RATTLE" in fh["parameters/ucmd/backend"].attrs["constraints"]
    i, j, d = pairs[:, 0].astype(int), pairs[:, 1].astype(int), pairs[:, 2]
    r = np.linalg.norm(x[:, i] - x[:, j], axis=-1)
    # OpenMM holds the constraint lengths in single precision (it found 2026-11-20: every
    # distance short by exactly float32(d) − d, −5.4e-9 for O–H, −2.7e-8 for H–H). To that
    # length the tolerance holds; against the package length the error is below 1e-7.
    d32 = d.astype(np.float32).astype(np.float64)
    assert (np.abs(r - d32) / d).max() < 1e-9
    assert (np.abs(r - d) / d).max() < 1e-7


def test_constraints_with_langevin_run_on_openmm(tmp_path: Path) -> None:
    """E2.5 step 4: constrained BAOAB in the bridge keeps the constraints (the canonical sampling
    itself is tested in the engine)."""
    import h5py

    text = RIGID_WATER.format(out="out").replace(
        'integrator = "velocity-verlet"', 'integrator = "langevin-baoab"'
    )
    text += '\n[thermostat]\nkind = "langevin"\ntemperature_K = 300.0\nfriction_per_ps = 1.0\n'
    profile = setup_run(tmp_path, "water-2", text)
    pairs = _rigid_water(tmp_path)
    r = ucmd(tmp_path, profile)
    assert r.returncode == 0, r.stderr
    with h5py.File(tmp_path / "out/trajectory.h5") as fh:
        x = fh["particles/all/position/value"][()]
    i, j, d = pairs[:, 0].astype(int), pairs[:, 1].astype(int), pairs[:, 2]
    r_ij = np.linalg.norm(x[:, i] - x[:, j], axis=-1)
    assert (np.abs(r_ij - d) / d).max() < 1e-7  # single-precision lengths in OpenMM, see above


def test_pair_overrides_agree_with_the_engine(tmp_path: Path) -> None:
    """Spec 0.1.2 (ADR 0015, A7): kob-andersen-500, whose A-B pair is an override, runs through
    the OpenMM backend; the conformance single point agrees with the CPU engine.

    Without the tail correction: OpenMM's CustomNonbondedForce counts pairs within a type class
    as n(n+1)/2 and rescales, an O(1/N) difference from the engine's N_a N_b for several types
    (tools/oracle/test_oracle.py::test_type_pair_dispersion_correction).
    """
    text = (
        ARGON.format(steps=20, thermostat="nose-hoover-chain")
        .replace("dispersion_correction = true", 'lj_cutoff_mode = "potential-shift"')
        .replace("timestep_fs = 4.0", "timestep_fs = 2.0")
    )
    profile = setup_run(tmp_path, "kob-andersen-500", text)
    r = ucmd(tmp_path, profile)
    assert r.returncode == 0, r.stderr
    line = next(x for x in r.stderr.splitlines() if "conformance with" in x)
    assert float(line.split("energy ")[1].split(",")[0]) < 1e-9, line
    assert float(line.split("forces ")[1].split(" ")[0]) < 1e-9, line


def test_pair_overrides_with_the_tail_correction_agree_with_the_engine(tmp_path: Path) -> None:
    """Both at once (merge of the tail weights and the pair overrides, 2026-11-27).

    The classes of the tail weights are the atom types. Kob-Andersen-500: OpenMM's tail is
    -248.4928 kJ/mol against the exact 2π/V Σ n_a n_b ∫ r² u_ab of -248.4929 (relative -4.0e-7:
    OpenMM integrates a custom tail numerically, see the water test above). The tail is 6.4 % of
    the energy, so the single point agrees to 2.7e-8 here; wrong class weights would be O(1/N),
    about 1e-4. Hence: the tail within 1e-6 of itself, the single point within 1e-7."""
    import openmm as mm

    from ucmd_bridge import model, package

    fixture = SPEC / "fixtures/valid/kob-andersen-500"
    top = package.load(fixture)
    rc = 0.85
    volume = abs(float(np.linalg.det(np.asarray(top.cell_nm))))
    n = np.bincount(top.pairs.type_index)
    exact = 0.0
    for a in range(len(n)):
        for b in range(len(n)):
            s, e = top.pairs.sigma_nm[a, b], top.pairs.epsilon_kjmol[a, b]
            integral = 4 * e * s**3 * ((s / rc) ** 9 / 9 - (s / rc) ** 3 / 3)
            exact += 2 * np.pi / volume * n[a] * n[b] * integral

    def energy(tail: bool) -> float:
        settings = model.NonbondedSettings(cutoff_nm=rc, dispersion_correction=tail)
        system, _ = model.build_system(top, settings)
        platform = mm.Platform.getPlatformByName("Reference")
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(np.load(fixture / "positions.npy"))
        state = context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(mm.unit.kilojoule_per_mole)

    assert abs((energy(True) - energy(False)) / exact - 1) < 1e-6

    text = ARGON.format(steps=20, thermostat="nose-hoover-chain").replace(
        "timestep_fs = 4.0", "timestep_fs = 2.0"
    )
    assert "dispersion_correction = true" in text
    profile = setup_run(tmp_path, "kob-andersen-500", text)
    r = ucmd(tmp_path, profile)
    assert r.returncode == 0, r.stderr
    line = next(x for x in r.stderr.splitlines() if "conformance with" in x)
    assert float(line.split("energy ")[1].split(",")[0]) < 1e-7, line
    assert float(line.split("forces ")[1].split(" ")[0]) < 1e-9, line


def test_rule_equal_overrides_build_the_same_forces() -> None:
    """A1 for the bridge: a table filled by the combination rule gives the per-particle result."""
    import dataclasses

    import openmm as mm

    from ucmd_bridge import model, package

    top = package.load(SPEC / "fixtures/valid/kob-andersen-500")
    assert top.pairs is not None
    names = ["A"] * 400 + ["B"] * 100
    sa, ea, sb, eb = top.sigma_nm[0], top.epsilon_kjmol[0], top.sigma_nm[-1], top.epsilon_kjmol[-1]
    ff = {
        "combination_rule": top.combination_rule,
        "atom_types": {
            "A": {"sigma_nm": sa, "epsilon_kjmol": ea},
            "B": {"sigma_nm": sb, "epsilon_kjmol": eb},
        },
        "pair_overrides": [
            {"types": ["A", "B"], "sigma_nm": 0.5 * (sa + sb), "epsilon_kjmol": (ea * eb) ** 0.5}
        ],
    }
    settings = model.NonbondedSettings.from_provider({"kind": "force-field", "cutoff_nm": 0.85})

    def evaluate(t: package.Topology) -> tuple[float, np.ndarray]:
        system, _ = model.build_system(t, settings)
        ctx = mm.Context(
            system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference")
        )
        ctx.setPositions(t.positions_nm)
        st = ctx.getState(getEnergy=True, getForces=True)
        return st.getPotentialEnergy()._value, st.getForces(asNumpy=True)._value

    e_rule, f_rule = evaluate(dataclasses.replace(top, pairs=None))
    e_tab, f_tab = evaluate(dataclasses.replace(top, pairs=package.pair_table(ff, names)))
    e_ka, _ = evaluate(top)
    assert e_tab == pytest.approx(e_rule, rel=1e-12)
    assert np.abs(f_tab - f_rule).max() <= 1e-12 * np.abs(f_rule).max()
    assert abs(e_ka - e_rule) > 1.0  # the real override changes the energy
