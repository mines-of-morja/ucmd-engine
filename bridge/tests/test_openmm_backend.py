"""E3.7: `ucmd run` with `[backend] kind = "openmm"` (here on OpenMM's CPU platform).

The whole chain: `ucmd` computes the reference single point and hands over to `ucmd-openmm`,
which checks conformance, runs, writes spec outputs and restarts. Needs the `ucmd` binary
(cargo build) and the spec fixtures (../../spec, or $UCMD_SPEC_DIR).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

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
    _names, rows = thermo(tmp_path)
    assert rows[:, 0].tolist() == list(range(0, 201, 10))  # no duplicate rows
    import h5py

    with h5py.File(tmp_path / "out/trajectory.h5") as fh:
        assert fh["particles/all/position/step"][()].tolist() == [0, 50, 100, 150, 200]
        assert "restarts" in fh["parameters/ucmd"]


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
