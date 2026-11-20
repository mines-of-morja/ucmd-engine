"""`ucmd-openmm run`: a run.toml on OpenMM, e.g. on a GPU (PLAN.md E3.7, ADR 0010).

    ucmd-openmm run RUN_TOML --reference DIR [--platform OpenCL] [--precision double]
                             [--device 0] [--restart CHECKPOINT] [--force]

`ucmd run` starts this program when the machine profile selects `[backend] kind = "openmm"`.

- **Model identity:** it first evaluates the configuration in `--reference` (written by `ucmd run`
  with the CPU reference backend) and refuses to run unless energy and forces agree within the
  tolerances given there (ADR 0010: 1e-6 and 1e-5 relative).
- **The engine's integrators:** velocity Verlet, the Nosé–Hoover chain and Langevin BAOAB run as
  OpenMM CustomIntegrators with the engine's operator splittings and full-step velocities
  (`integrators.py`). OpenMM's own integrators keep half-step velocities, and their thermostat
  would sample a slightly different ensemble at 1 fs.
- **Other algorithms where needed:** C-rescale → MonteCarloBarostat (the same
  isothermal-isobaric ensemble, validated for argon in validation/npt-argon). CSVR: refused.
- **Pressure:** OpenMM has no virial. The thermo column holds the molecular pressure of
  MonteCarloBarostat.computeCurrentPressure every `--pressure-every` steps (a numerical volume
  derivative, about 70 ms each), and nan in between. Its mean equals the atomic virial pressure
  in equilibrium; its fluctuations are smaller.
- **Outputs** in the spec formats (thermo table, H5MD or extended XYZ), restartable from this
  backend's own checkpoint file (OpenMM's checkpoint behind a JSON header). SIGINT/SIGTERM stop
  gracefully with a checkpoint and exit status 128 + signal.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import signal
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .identity import DEFAULT_MAX_MOVE_NM, Tracker
from .integrators import bath_energy, langevin_baoab, nose_hoover_chain, velocity_verlet
from .model import ModelError, NonbondedSettings, build_system
from .package import ATOMIC_NUMBER, Topology
from .package import load as load_package

BOLTZMANN_KJMOL_K = 1.380649e-23 * 6.02214076e23 / 1000.0  # CODATA 2018, exact
BAR_PER_KJMOL_NM3 = 1e25 / 6.02214076e23
MAGIC = b"UCMDOMM1\n"
# ADR 0012 (precision tiers): the model-identity check of a mixed-precision run allows larger
# pointwise force differences (single-precision forces: 5.5e-5 measured for the electrolyte);
# mixed precision itself is admitted by a statistical comparison with double precision.
FORCE_TOLERANCE = {"double": None, "mixed": 2e-4, "single": 2e-4}
SYMBOL = {z: s for s, z in ATOMIC_NUMBER.items()}
# Molecule identity (identity.py): positions are read at least this often, OpenMM's reorder
# interval, so at most one reorder lies between two reads and molecules move ≪ 0.2 nm.
TRACK_EVERY_STEPS = 250


class ConfigError(ValueError):
    """The run.toml asks for something this backend does not implement."""


@dataclass
class Plan:
    run_toml: Path
    raw: dict
    dt_ps: float
    steps: int
    seed: int
    integrator: str
    thermostat: dict | None
    barostat: dict | None
    velocities: dict | None
    remove_com_every: int
    provider: dict
    out_dir: Path
    thermo: dict | None
    trajectory: dict | None
    checkpoint: dict | None

    @property
    def temperature_K(self) -> float | None:
        return None if self.thermostat is None else float(self.thermostat["temperature_K"])

    def fingerprint(self) -> str:
        """SHA-256 of the configuration without the run length (as the engine's)."""
        doc = json.loads(json.dumps(self.raw))
        doc.get("run", {}).pop("steps", None)
        return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def read_plan(run_toml: Path) -> Plan:
    with run_toml.open("rb") as fh:
        raw = tomllib.load(fh)
    run = raw["run"]
    providers = raw.get("providers", [])
    if len(providers) != 1:
        raise ConfigError("the OpenMM backend runs exactly one [[providers]] table")
    thermostat = raw.get("thermostat")
    integrator = run["integrator"]
    if thermostat and thermostat["kind"] == "csvr":
        raise ConfigError(
            'CSVR has no OpenMM equivalent; use kind = "nose-hoover-chain" (or the CPU backend)'
        )
    if thermostat and thermostat["kind"] == "langevin" and integrator != "langevin-baoab":
        raise ConfigError('a langevin thermostat needs integrator = "langevin-baoab"')
    if integrator not in ("velocity-verlet", "langevin-baoab"):
        raise ConfigError(f"integrator {integrator!r}")
    barostat = raw.get("barostat")
    if barostat is not None:
        if barostat.get("coupling", "isotropic") != "isotropic":
            raise ConfigError("anisotropic pressure coupling is not implemented")
        if thermostat is None:
            raise ConfigError("a barostat needs a thermostat")
    if raw.get("momentum", {}).get("remove_angular"):
        raise ConfigError("remove_angular is not implemented in the OpenMM backend")
    output = raw.get("output", {})
    if output.get("thermo", {}).get("pressure_tensor"):
        # spec 0.1.1: a backend that cannot compute the tensor refuses the run
        raise ConfigError(
            "E_UNSUPPORTED  [output.thermo] pressure_tensor: OpenMM gives no pressure tensor; "
            "run on the CPU backend"
        )
    return Plan(
        run_toml=run_toml,
        raw=raw,
        dt_ps=float(run["timestep_fs"]) / 1000.0,
        steps=int(run["steps"]),
        seed=int(run["seed"]),
        integrator=integrator,
        thermostat=thermostat,
        barostat=barostat,
        velocities=raw.get("velocities"),
        remove_com_every=int(raw.get("momentum", {}).get("remove_com_every", 0)),
        provider=providers[0],
        out_dir=run_toml.parent / output.get("directory", "out"),
        thermo=output.get("thermo"),
        trajectory=output.get("trajectory"),
        checkpoint=output.get("checkpoint"),
    )


def system_path(plan: Plan) -> Path:
    return plan.run_toml.parent / plan.raw["system"]


# -- checkpoints ----------------------------------------------------------------------------


def write_checkpoint(path: Path, meta: dict, blob: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        fh.write(MAGIC + json.dumps(meta, sort_keys=True).encode() + b"\n" + blob)
        fh.flush()
    tmp.replace(path)


def read_checkpoint(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    if not data.startswith(MAGIC):
        raise ConfigError(f"{path}: not a checkpoint of the OpenMM backend")
    header, blob = data[len(MAGIC) :].split(b"\n", 1)
    return json.loads(header), blob


# -- the run ------------------------------------------------------------------------------


class Stop:
    def __init__(self) -> None:
        self.signal: int | None = None
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle)

    def _handle(self, signum, _frame) -> None:
        if self.signal is not None:  # a second signal: stop at once
            raise SystemExit(128 + signum)
        self.signal = signum


def environment() -> dict:
    """Software and hardware stack of this run (ADR 0012: environment pinning)."""
    import glob
    import platform as pyplatform

    out = {"python": pyplatform.python_version(), "host_kernel": pyplatform.release()}
    for pattern, key in (
        ("/opt/rocm-venv/lib/python*/site-packages/rocm_sdk_core-*.dist-info", "rocm_sdk"),
    ):
        found = glob.glob(pattern)
        if found:
            out[key] = Path(found[0]).name.split("-")[1].removesuffix(".dist-info")
    for var, key in (("UCMD_OPENMM_IMAGE", "image"), ("UCMD_OPENMM_IMAGE_ID", "image_id")):
        if os.environ.get(var):
            out[key] = os.environ[var]
    return out


def conformance(context, reference: Path, unit, precision: str) -> dict:
    """Evaluate the reference configuration and compare with the CPU reference backend."""
    import openmm as mm

    ref = json.loads((reference / "reference.json").read_text())
    positions = np.load(reference / "positions.npy")
    forces_ref = np.load(reference / "forces.npy")
    if ref.get("cell_nm") is not None:
        context.setPeriodicBoxVectors(*(mm.Vec3(*row) for row in ref["cell_nm"]))
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    e_ref = float(ref["energy_kjmol"])
    de = abs(energy - e_ref) / max(abs(e_ref), 1.0)
    df = float(np.abs(forces - forces_ref).max() / np.abs(forces_ref).max())
    result = {
        "conformance_energy_relative": de,
        "conformance_force_relative": df,
        "conformance_reference_engine": ref.get("engine", "?"),
        "reference_energy_kjmol": e_ref,
        "energy_kjmol": energy,
    }
    force_tolerance = FORCE_TOLERANCE.get(precision) or ref["tolerance_force_relative"]
    result["conformance_force_tolerance"] = force_tolerance
    if de > ref["tolerance_energy_relative"] or df > force_tolerance:
        raise ModelError(
            f"conformance check failed: energy {energy:.6f} vs {e_ref:.6f} kJ/mol "
            f"(relative {de:.1e}, tolerance {ref['tolerance_energy_relative']:.0e}), "
            f"forces max relative {df:.1e} (tolerance {force_tolerance:.0e}, {precision})"
        )
    return result


def run(args: argparse.Namespace) -> int:
    import openmm as mm
    from openmm import unit

    from .outputs import H5mdWriter, ThermoWriter, XyzWriter

    started = time.perf_counter()
    plan = read_plan(Path(args.run_toml))
    top: Topology = load_package(system_path(plan))
    settings = NonbondedSettings.from_provider(plan.provider)
    system, electrostatics = build_system(top, settings)
    periodic = top.cell_nm is not None
    temperature = plan.temperature_K
    if plan.remove_com_every > 0:
        system.addForce(mm.CMMotionRemover(plan.remove_com_every))
    barostat, p0_kjmol_nm3 = None, 0.0
    if plan.barostat is not None:
        p0 = float(plan.barostat["pressure_bar"])
        barostat = mm.MonteCarloBarostat(p0, temperature, int(plan.barostat.get("every", 10)))
        p0_kjmol_nm3 = p0 / BAR_PER_KJMOL_NM3
    elif periodic:
        barostat = mm.MonteCarloBarostat(1.0, temperature or 300.0, 0)  # pressure only
    if barostat is not None:
        barostat.setRandomNumberSeed(plan.seed % 2**31)
        system.addForce(barostat)
    kind = None if plan.thermostat is None else plan.thermostat["kind"]
    langevin = kind == "langevin"
    com_removed = not (langevin and plan.remove_com_every == 0)
    dof = 3 * top.n_atoms - len(top.constraints) - (3 if com_removed else 0)
    chain = int(plan.thermostat.get("chain_length", 3)) if kind == "nose-hoover-chain" else 0
    if top.constraints:
        raise ConfigError("constraints are not implemented in the OpenMM backend's integrators")
    if kind == "nose-hoover-chain":
        tau = float(plan.thermostat["tau_ps"])
        integrator = nose_hoover_chain(plan.dt_ps, temperature, tau, chain, dof)
    elif langevin:
        friction = float(plan.thermostat["friction_per_ps"])
        integrator = langevin_baoab(plan.dt_ps, temperature, friction)
        integrator.setRandomNumberSeed(plan.seed % 2**31)
    else:
        integrator = velocity_verlet(plan.dt_ps)

    if args.precision != "double" and plan.thermostat is None:
        raise ConfigError(
            f"{args.precision} precision is for thermostatted production runs (ADR 0012); "
            "NVE and validation runs use double"
        )
    platform = mm.Platform.getPlatformByName(args.platform)
    properties = {}
    if args.platform in ("OpenCL", "CUDA", "HIP"):
        properties["Precision"] = args.precision
        if args.device is not None:
            properties["DeviceIndex"] = args.device
    context = mm.Context(system, integrator, platform, properties)
    device = ""
    for name in ("DeviceName", "OpenCLPlatformName"):
        with contextlib.suppress(Exception):
            device += context.getPlatform().getPropertyValue(context, name) + " "
    check = {}
    if args.reference is not None:
        check = conformance(context, Path(args.reference), unit, args.precision)
    elif not args.no_conformance:
        raise ConfigError("--reference is required (written by `ucmd run`); see --no-conformance")

    # start state
    restart_step = None
    restart_slots = None
    if args.restart:
        meta, blob = read_checkpoint(Path(args.restart))
        if meta["system_sha256"] != top.system_sha256 and not args.force:
            raise ConfigError("E_RESTART_MISMATCH  the package differs from the checkpoint's")
        if meta["run_config_sha256"] != plan.fingerprint() and not args.force:
            raise ConfigError("E_RESTART_MISMATCH  the run configuration differs (use --force)")
        context.loadCheckpoint(blob)
        restart_step = int(meta["step"])
        restart_slots = meta.get("identity_molecule_slots")  # absent in checkpoints before the fix
    else:
        if periodic:
            context.setPeriodicBoxVectors(*(mm.Vec3(*row) for row in top.cell_nm))
        context.setPositions(top.positions_nm)
        if plan.velocities is not None:
            context.setVelocitiesToTemperature(
                float(plan.velocities["temperature_K"]), plan.seed % 2**31
            )
            if com_removed:
                v = context.getState(getVelocities=True).getVelocities(asNumpy=True)
                v = v.value_in_unit(unit.nanometer / unit.picosecond)
                v -= (top.masses_amu[:, None] * v).sum(0) / top.masses_amu.sum()
                context.setVelocities(v)
        elif top.velocities_nm_ps is not None:
            context.setVelocities(top.velocities_nm_ps)
        context.setStepCount(0)
        context.setTime(0.0)
    step = context.getStepCount()

    # outputs
    plan.out_dir.mkdir(parents=True, exist_ok=True)
    ref_engine = check.get("conformance_reference_engine", "ucmd ?")
    git_hash = ref_engine.split("(")[-1].rstrip(")") if "(" in ref_engine else "unknown"
    provenance = {
        "spec_version": plan.raw.get("spec_version", "0.1.0"),
        "engine_version": f"ucmd-bridge {__version__} (OpenMM {mm.__version__}); {ref_engine}",
        "engine_git_hash": git_hash,
        "system_sha256": top.system_sha256,
        "run_config_toml": plan.run_toml.read_text(),
        "backend": {
            "kind": "openmm",
            "openmm_version": mm.__version__,
            "platform": args.platform,
            "precision": args.precision,
            "device": device.strip(),
            "electrostatics": json.dumps(electrostatics),
            "barostat": "MonteCarloBarostat" if plan.barostat else "none",
            "pressure": "molecular, MonteCarloBarostat.computeCurrentPressure",
            "identity_tracking": f"positions read every {TRACK_EVERY_STEPS} steps; molecules "
            f"matched by atom RMSD, limit {DEFAULT_MAX_MOVE_NM} nm (identity.py)",
            **environment(),
            **{k: v for k, v in check.items() if isinstance(v, (int, float, str))},
        },
    }
    thermo = traj = None
    if plan.thermo:
        thermo = ThermoWriter(plan.out_dir / plan.thermo["file"], restart_step)
    if plan.trajectory:
        tpath = plan.out_dir / plan.trajectory["file"]
        if tpath.suffix == ".xyz":
            symbols = [SYMBOL.get(int(z), "X") for z in top.species]
            traj = XyzWriter(tpath, symbols, restart_step)
        else:
            traj = H5mdWriter(
                tpath, top, precision=plan.trajectory.get("precision", "float64"),
                provenance=provenance, restart_step=restart_step,
            )  # fmt: skip
            if restart_step is not None:
                traj.record_restart({"step": restart_step, "forced": bool(args.force),
                                     "engine_version": provenance["engine_version"]})  # fmt: skip
    ckpt_path = plan.out_dir / plan.checkpoint["file"] if plan.checkpoint else None
    every = {
        "thermo": int(plan.thermo["every"]) if plan.thermo else 0,
        "trajectory": int(plan.trajectory["every"]) if plan.trajectory else 0,
        "checkpoint": int(plan.checkpoint["every"]) if plan.checkpoint else 0,
    }
    pressure_every = args.pressure_every
    if every["thermo"] and pressure_every % every["thermo"]:
        pressure_every = max(every["thermo"], pressure_every // every["thermo"] * every["thermo"])

    tracker = Tracker(top.molecule_index, list(top.molecule_type), slots=restart_slots)
    if restart_step is not None:  # seed the labels before the first step (a fresh start does
        seed = context.getState(getPositions=True, enforcePeriodicBox=False)  # it in observe)
        tracker.update(np.asarray(seed.getPositions(asNumpy=True).value_in_unit(unit.nanometer)))

    def meta() -> dict:
        return {
            "identity_molecule_slots": tracker.slot.tolist(),
            "step": context.getStepCount(),
            "time_ps": context.getTime().value_in_unit(unit.picosecond),
            "system_sha256": top.system_sha256,
            "run_config_sha256": plan.fingerprint(),
            "platform": args.platform,
            "precision": args.precision,
            "openmm_version": mm.__version__,
        }

    def observe(s: int, last_good: list) -> None:
        need_traj = bool(every["trajectory"]) and s % every["trajectory"] == 0
        need_pos = need_traj or s % TRACK_EVERY_STEPS == 0
        state = context.getState(getEnergy=True, getPositions=need_pos, enforcePeriodicBox=False)
        k = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        u = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        t_ps = state.getTime().value_in_unit(unit.picosecond)
        cell = None
        volume = None
        if periodic:
            box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
            cell = np.asarray(box)
            volume = float(abs(np.linalg.det(cell)))
        conserved = k + u
        if kind == "nose-hoover-chain":
            conserved += bath_energy(integrator, chain)
        if plan.barostat is not None and volume is not None:
            conserved += p0_kjmol_nm3 * volume
        pressure = None
        if barostat is not None and pressure_every and s % pressure_every == 0:
            value = barostat.computeCurrentPressure(context)  # a Quantity in newer OpenMM
            pressure = float(value.value_in_unit(unit.bar) if hasattr(value, "unit") else value)
        row = {
            "step": s, "time_ps": t_ps, "potential_kjmol": u, "kinetic_kjmol": k,
            "total_kjmol": k + u, "conserved_energy_kjmol": conserved,
            "temperature_K": 2.0 * k / (dof * BOLTZMANN_KJMOL_K), "pressure_bar": pressure,
            "volume_nm3": volume,
        }  # fmt: skip
        if not all(math.isfinite(x) for x in (k, u)):
            raise ModelError(f"non-finite energy at step {s}")
        if thermo and every["thermo"] and s % every["thermo"] == 0:
            thermo.write(row)
        if need_pos:
            raw = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            positions = tracker.update(np.asarray(raw))  # canonical order (invariant 4)
            if need_traj:
                traj.write(s, t_ps, positions, cell, row)
                # for an emergency checkpoint: the state and its own step and labels
                last_good[0] = (meta(), context.createCheckpoint())
        if ckpt_path and every["checkpoint"] and s % every["checkpoint"] == 0 and s > 0:
            write_checkpoint(ckpt_path, meta(), context.createCheckpoint())

    print(
        f"ucmd-openmm {__version__} (OpenMM {mm.__version__}, {args.platform} {args.precision}, "
        f"{device.strip()}): {top.n_atoms} atoms, steps {step}..{plan.steps}, N_dof = {dof}"
        + (" (restart)" if restart_step is not None else ""),
        file=sys.stderr, flush=True,
    )  # fmt: skip
    if check:
        print(
            f"  conformance with {check['conformance_reference_engine']}: energy "
            f"{check['conformance_energy_relative']:.1e}, forces "
            f"{check['conformance_force_relative']:.1e} (relative)",
            file=sys.stderr, flush=True,
        )  # fmt: skip
    stop = Stop()
    last_good = [(meta(), context.createCheckpoint())]
    first = step
    status = 0
    try:
        if restart_step is None:
            observe(step, last_good)
        events = [e for e in (*every.values(), pressure_every, TRACK_EVERY_STEPS) if e]
        while step < plan.steps and stop.signal is None:
            nxt = min([plan.steps, *((step // e + 1) * e for e in events)])
            integrator.step(nxt - step)
            step = nxt
            observe(step, last_good)
    except Exception as exc:  # OpenMM reports NaN coordinates etc. as exceptions
        if ckpt_path is not None:
            write_checkpoint(ckpt_path.with_suffix(".emergency.ucchk"), *last_good[0])
        print(f"error: the OpenMM backend failed after step {step}: {exc}", file=sys.stderr)
        status = 1
    if status == 0 and ckpt_path is not None:
        write_checkpoint(ckpt_path, meta(), context.createCheckpoint())
    for w in (thermo, traj):
        if w is not None:
            w.close()
    elapsed = time.perf_counter() - started
    done = step - first
    rate = done * plan.dt_ps / elapsed * 86400 / 1000 if elapsed > 0 else 0.0
    if stop.signal is not None and status == 0:
        print(f"stopped by signal after step {step}; the state is in {ckpt_path}", file=sys.stderr)
        return 128 + stop.signal
    print(
        f"identity: {tracker.exchanges} molecule labels corrected (worst match ratio "
        f"{tracker.worst_ratio:.2f})",
        file=sys.stderr,
    )
    if status == 0:
        print(f"done: {done} steps in {elapsed:.2f} s ({rate:.1f} ns/day)", file=sys.stderr)
    return status


def probe(args: argparse.Namespace) -> int:
    """A short test run on the device: does it compute, and how fast? Exit 0 when it worked.

    Used before long GPU jobs. On 2026-11-18/19 the Radeon VII's compute queue stopped
    answering while a game started on the other card; a probe with a time limit (e.g.
    `timeout 300`) tells that apart from a slow run.
    """
    import openmm as mm

    n_side, spacing = 16, 0.38  # 4096 LJ particles on a lattice, roughly liquid argon
    box = n_side * spacing
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box))
    lj = mm.NonbondedForce()
    lj.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    lj.setCutoffDistance(1.0)
    positions = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                system.addParticle(39.948)
                lj.addParticle(0.0, 0.34, 0.996)
                positions.append(mm.Vec3(i, j, k) * spacing)
    system.addForce(lj)
    integrator = mm.VerletIntegrator(0.002)
    platform = mm.Platform.getPlatformByName(args.platform)
    properties = {"Precision": args.precision} if args.platform in ("OpenCL", "CUDA", "HIP") else {}
    if args.device is not None and properties:
        properties["DeviceIndex"] = args.device
    context = mm.Context(system, integrator, platform, properties)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(100.0, 1)
    t0 = time.perf_counter()
    integrator.step(args.steps)
    energy = context.getState(getEnergy=True).getPotentialEnergy()._value
    seconds = time.perf_counter() - t0
    device = ""
    with contextlib.suppress(Exception):
        device = context.getPlatform().getPropertyValue(context, "DeviceName")
    ok = math.isfinite(energy)
    print(json.dumps({"ok": ok, "platform": args.platform, "device": device,
                      "steps": args.steps, "seconds": round(seconds, 3)}))  # fmt: skip
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ucmd-openmm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="run a run.toml on OpenMM")
    p.add_argument("run_toml")
    p.add_argument("--reference", help="reference single point written by `ucmd run`")
    p.add_argument("--no-conformance", action="store_true", help="tests only: skip the check")
    p.add_argument("--platform", default="OpenCL")
    p.add_argument("--precision", default="double", choices=["double", "mixed", "single"])
    p.add_argument("--device")
    p.add_argument("--restart")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--pressure-every",
        type=int,
        default=500,
        help="steps between pressure evaluations (each costs ~2 force evaluations)",
    )
    q = sub.add_parser("probe", help="a short test run on the device (health check)")
    q.add_argument("--platform", default="OpenCL")
    q.add_argument("--precision", default="mixed", choices=["double", "mixed", "single"])
    q.add_argument("--device")
    q.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args(argv)
    if args.command == "probe":
        return probe(args)
    try:
        return run(args)
    except (ConfigError, ModelError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
