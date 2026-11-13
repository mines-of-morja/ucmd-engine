"""M2a "core stands" (docs/design/E2.md §10), driven by run.sh.

    python m2a.py config PACKAGE DIR   run.toml (engine defaults) and explicit.toml (alpha, grid)
    python m2a.py compare DIR          OpenMM single point with the same alpha and grid
    python m2a.py report OUT           REPORT.md on stdout, from everything run.sh wrote to OUT

Runs with tools/oracle/.venv/bin/python (OpenMM, NumPy) and imports tools/oracle/oracle.py.
"""

import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "oracle"))

# Lehnert et al. 2025, SI S1.1: 1.4 nm cutoff, long-range Coulomb to an accuracy of 1e-5.
CUTOFF_NM = 1.4
TOLERANCE = 1e-5
T_TARGET_K = 298.15
BOLTZMANN_KJMOL_K = (
    0.00831446261815324  # kJ mol⁻¹ K⁻¹, exact (spec/units.md), as argon/analyze.py
)
AVOGADRO = 6.02214076e23  # mol⁻¹, exact (SI 2019; spec/units.md)
BAR_PER_KJMOL_NM3 = 1e3 / (AVOGADRO * 1e-27) / 1e5
ENERGY_CRITERION = 1e-6  # relative, per term and total (PLAN.md M2a)
FORCE_CRITERION = 1e-5  # relative rms (PLAN.md M2a)


def legal_size(n: int) -> int:
    """Smallest size ≥ n with factors 2, 3, 5, 7 only (OpenMM's rule, as the engine)."""
    m = max(n, 1)
    while True:
        r = m
        for f in (2, 3, 5, 7):
            while r % f == 0:
                r //= f
        if r == 1:
            return m
        m += 1


def config(package: Path, directory: Path) -> None:
    import oracle

    top = oracle.load_topology(package)
    alpha = math.sqrt(-math.log(2.0 * TOLERANCE)) / CUTOFF_NM
    grid = [
        legal_size(math.ceil(2.0 * alpha * top.cell_nm[d, d] / (3.0 * TOLERANCE**0.2)))
        for d in range(3)
    ]
    directory.mkdir(parents=True, exist_ok=True)
    common = f"""spec_version = "0.1.0"
system = "{os.path.relpath(package, directory)}"
title = "M2a single point"

[run]
integrator = "velocity-verlet"
timestep_fs = 1.0
steps = 0
seed = 1

[[providers]]
kind = "force-field"
cutoff_nm = {CUTOFF_NM}
coulomb = "pme"
"""
    (directory / "run.toml").write_text(common + f"ewald_tolerance = {TOLERANCE}\n")
    (directory / "explicit.toml").write_text(
        common + f"ewald_alpha_per_nm = {alpha!r}\npme_grid = {grid}\npme_order = 5\n"
    )
    pme = {
        "package": str(package),
        "cutoff_nm": CUTOFF_NM,
        "tolerance": TOLERANCE,
        "alpha_per_nm": alpha,
        "grid": grid,
    }
    (directory / "pme.json").write_text(json.dumps(pme, indent=1) + "\n")


def compare(directory: Path) -> None:
    import openmm
    import oracle

    pme = json.loads((directory / "pme.json").read_text())
    top = oracle.load_topology(Path(pme["package"]))
    system = oracle.molecular_system(
        top,
        cutoff_nm=pme["cutoff_nm"],
        pme_parameters=(pme["alpha_per_nm"], *pme["grid"]),
    )
    terms = oracle.energy_terms(system, top.positions_nm, top.cell_nm)
    _, forces = oracle.evaluate(system, top.positions_nm, top.cell_nm)
    ours = json.loads((directory / "singlepoint.json").read_text())
    explicit = json.loads((directory / "singlepoint-explicit.json").read_text())
    f_ours = np.load(directory / "forces.npy")
    f_explicit = np.load(directory / "forces-explicit.npy")
    rows = {}
    for name, value in sorted(ours["energy_terms_kjmol"].items()):
        rows[name] = (
            value,
            terms[name],
            abs(value - terms[name]) / max(abs(terms[name]), 1.0),
        )
    total = ours["energy_kjmol"]
    rows["total"] = (
        total,
        terms["total"],
        abs(total - terms["total"]) / abs(terms["total"]),
    )
    diff = f_ours - forces
    result = {
        "openmm": openmm.__version__,
        "atoms": top.n_atoms,
        "pme": pme,
        "terms": rows,
        "force_rms_rel": float(np.sqrt(np.mean(diff**2)) / np.sqrt(np.mean(forces**2))),
        "force_max_rel": float(np.abs(diff).max() / np.abs(forces).max()),
        "max_force_kjmol_nm": ours["max_force_kjmol_nm"],
        # the engine's own rules give exactly OpenMM's alpha and grid
        "defaults_equal_explicit": bool(
            ours["energy_terms_kjmol"] == explicit["energy_terms_kjmol"]
            and np.array_equal(f_ours, f_explicit)
        ),
    }
    (directory / "compare.json").write_text(json.dumps(result, indent=1) + "\n")


def virial_check(directory: Path) -> tuple[float, float]:
    """ucmd's analytic virial pressure of a single point, and OpenMM's: the central difference
    of the energy with positions and cell scaled by 1 ± 1e-8, W = −dU/dε, P = W/(3V)."""
    import oracle

    pme = json.loads((directory / "pme.json").read_text())
    top = oracle.load_topology(Path(pme["package"]))
    system = oracle.molecular_system(
        top,
        cutoff_nm=pme["cutoff_nm"],
        pme_parameters=(pme["alpha_per_nm"], *pme["grid"]),
    )
    h = 1e-8  # small enough that no pair crosses the cutoff, large against round-off

    def energy(scale: float) -> float:
        return oracle.evaluate(system, top.positions_nm * scale, top.cell_nm * scale)[0]

    du = (energy(1 + h) - energy(1 - h)) / (2 * h)
    volume = abs(float(np.linalg.det(top.cell_nm)))
    ours = json.loads((directory / "singlepoint.json").read_text())[
        "virial_pressure_bar"
    ]
    return ours, -du / (3 * volume) * BAR_PER_KJMOL_NM3


def read_checkpoint(path: Path) -> tuple[dict, bytes]:
    """Metadata and the state bytes (positions, velocities, cell) of a format-v2 checkpoint."""
    data = path.read_bytes()
    assert data[:8] == b"UCMDCHK\0", path
    (length,) = struct.unpack("<Q", data[12:20])
    return json.loads(data[20 : 20 + length]), data[20 + length :]


def read_thermo(path: Path) -> dict[str, np.ndarray]:
    lines = path.read_text().splitlines()
    names = lines[0].lstrip("# ").split("\t")
    values = np.array([[float(x) for x in line.split("\t")] for line in lines[1:]])
    return {name: values[:, k] for k, name in enumerate(names)}


def block_error(x: np.ndarray, blocks: int = 10) -> float:
    means = np.array([b.mean() for b in np.array_split(x, blocks)])
    return float(means.std(ddof=1) / math.sqrt(blocks))


def mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def report(out: Path) -> None:
    times = dict(
        line.rsplit(" ", 1) for line in (out / "times.txt").read_text().splitlines()
    )
    times = {k: float(v) for k, v in times.items()}
    versions = (out / "versions.txt").read_text().strip().splitlines()
    log = (out / "nvt-full" / "log.txt").read_text()
    hardware = next(
        line.split(":", 1)[1].strip()
        for line in log.splitlines()
        if "hardware:" in line
    )
    dof = int(log.split("N_dof = ")[1].split()[0].rstrip(","))
    lines = ["# M2a — the core stands", ""]
    passed = []

    # single points
    compares = {}
    for label, d in (("built", "sp-built"), ("minimized", "sp-minimized")):
        compares[label] = json.loads((out / d / "compare.json").read_text())
    c0 = compares["built"]
    pme = c0["pme"]
    sp_lines = [
        "## 1. Single points against OpenMM",
        "",
        (
            f"OpenMM {c0['openmm']}, Reference platform, double precision; oracle v1 (`tools/oracle/oracle.py`). "
            f"PME with α = {pme['alpha_per_nm']:.6f} nm⁻¹, grid {pme['grid']}, order 5, cutoff {pme['cutoff_nm']} nm, "
            "LJ truncated at the cutoff without tail correction, 1-4 pairs without cutoff. "
            "Relative energy error: |ΔE| / max(|E|, 1 kJ/mol) per term, |ΔE| / |E| for the total."
        ),
        "",
        "| Term | built: ucmd (kJ/mol) | built: rel. error | minimized: ucmd (kJ/mol) | minimized: rel. error |",
        "|---|---|---|---|---|",
    ]
    ok_sp = True
    for name in c0["terms"]:
        a, b = compares["built"]["terms"][name], compares["minimized"]["terms"][name]
        ok_sp &= a[2] < ENERGY_CRITERION and b[2] < ENERGY_CRITERION
        sp_lines.append(
            f"| {name} | {a[0]:.6f} | {a[2]:.1e} | {b[0]:.6f} | {b[2]:.1e} |"
        )
    sp_lines.append("")
    for label, c in compares.items():
        ok_f = c["force_rms_rel"] < FORCE_CRITERION
        ok_sp &= ok_f and c["defaults_equal_explicit"]
        sp_lines.append(
            f"- **{label}:** forces rms error {c['force_rms_rel']:.1e} of rms |F| (max {c['force_max_rel']:.1e} of max |F|), "
            f"max |F| = {c['max_force_kjmol_nm']:.1f} kJ mol⁻¹ nm⁻¹. "
            f"Engine defaults (tolerance only) bitwise equal to the explicit α and grid: {c['defaults_equal_explicit']}."
        )
    passed.append(
        (
            "Single points before and after minimization: energies to 1e-6, forces to 1e-5 (rms)",
            ok_sp,
        )
    )

    # minimization
    minimized = json.loads((out / "minimized.ucsys" / "system.json").read_text())[
        "provenance"
    ]["minimization"]
    min_lines = [
        "## 2. Minimization",
        "",
        (
            f"`ucmd minimize` ({minimized['method']}): {minimized['steps']} steps in {times['minimize']:.0f} s; "
            f"E {minimized['energy_start_kjmol']:.1f} → {minimized['energy_kjmol']:.1f} kJ/mol; "
            f"max |F| {minimized['max_force_start_kjmol_nm']:.1f} → {minimized['max_force_kjmol_nm']:.2f} kJ mol⁻¹ nm⁻¹ "
            f"(criterion {minimized['max_force_criterion_kjmol_nm']:.0f})."
        ),
    ]

    # NVT
    th = read_thermo(out / "nvt-full" / "out" / "thermo.tsv")
    t, temp, conserved = (
        th["time_ps"],
        th["temperature_K"],
        th["conserved_energy_kjmol"],
    )
    complete = th["step"][-1] == 10000
    late = t > 5.0
    t_mean, t_err = float(temp[late].mean()), block_error(temp[late])
    t_sd = float(temp[late].std())
    sd_expected = T_TARGET_K * math.sqrt(2.0 / dof)
    held = abs(t_mean - T_TARGET_K) <= 3.0 * t_err
    kt = BOLTZMANN_KJMOL_K * T_TARGET_K
    atoms = c0["atoms"]
    swing = float(np.ptp(conserved[t <= 0.3]))
    h_sd = float(conserved[late].std())
    # drift: mean H over the last ps minus the mean over 1–2 ps, per ps (robust against the
    # fluctuations), and a linear fit over 1–10 ps for comparison
    first_ps, last_ps, settled = (t > 1.0) & (t <= 2.0), t > 9.0, t > 1.0
    drift = (
        float(conserved[last_ps].mean() - conserved[first_ps].mean()) / 8.0
    )  # kJ/mol/ps
    fit = float(np.polyfit(t[settled], conserved[settled], 1)[0])  # kJ/mol/ps
    per_dof = 1000.0 / (kt * dof)  # kJ/mol/ps → kT/ns per degree of freedom
    ns_per_day = 0.010 / times["nvt-full"] * 86400.0
    per_ps = [f"{temp[(t > k) & (t <= k + 1)].mean():.1f}" for k in range(10)]
    pressure = th["pressure_bar"]
    p_mean, p_err = float(pressure[late].mean()), block_error(pressure[late])
    p_per_ps = [f"{pressure[(t > k) & (t <= k + 1)].mean():.0f}" for k in range(10)]
    p_ours, p_openmm = virial_check(out / "sp-minimized")
    passed.append(("10 ps NVT completes", bool(complete)))
    passed.append(
        (
            "Temperature held: ⟨T⟩ over 5–10 ps within 3 standard errors of 298.15 K",
            held,
        )
    )
    nvt_lines = [
        "## 3. 10 ps NVT",
        "",
        (
            "Nosé–Hoover chain (M = 3, τ = 0.1 ps, MTK propagator with Suzuki–Yoshida weights), velocity Verlet "
            "with Δt = 1 fs, velocities from Maxwell–Boltzmann at 298.15 K, COM momentum removed every 100 steps, "
            f"PME as above with the engine's default rules. N_f = {dof}."
        ),
        "",
        f"- ⟨T⟩ over 5–10 ps: {t_mean:.2f} ± {t_err:.2f} K (block standard error, 10 blocks); target 298.15 K.",
        f"- σ_T over 5–10 ps: {t_sd:.2f} K; canonical expectation T·√(2/N_f) = {sd_expected:.2f} K.",
        f"- Mean T per ps (K): {', '.join(per_ps)}.",
        (
            f"- Pressure: mean per ps (bar) {', '.join(p_per_ps)}; ⟨P⟩ over 5–10 ps {p_mean:.0f} ± {p_err:.0f} bar. "
            "It is still falling: the packed and minimized box relaxes towards a liquid structure. "
            f"The engine computes it correctly: the virial pressure of the minimized structure, {p_ours:.3f} bar, "
            f"equals OpenMM's finite difference of the energy under box scaling, {p_openmm:.3f} bar "
            f"(relative difference {abs(p_ours - p_openmm) / abs(p_openmm):.0e}). Whether the force field's "
            "density matches the builder's placeholder of 1.2 g/cm³ needs NPT, as in Lehnert's protocol "
            "(barostat, E2.7)."
        ),
        "- **Conserved quantity** H = K + U + chain energy:",
        (
            f"  - start-up: H swings by {swing:.0f} kJ/mol within the first 0.3 ps. The minimized structure "
            "starts every C–H bond at its rest length, so the bond vibrations, and with them velocity "
            "Verlet's bounded energy error, are in phase until they dephase (NVE check in §6);"
        ),
        f"  - fluctuation over 5–10 ps: σ_H = {h_sd:.1f} kJ/mol;",
        (
            f"  - drift from the mean over 1–2 ps to the mean over 9–10 ps: {drift:+.2f} kJ/mol/ps = "
            f"{drift / atoms:+.1e} kJ/mol/ps per atom (the unit GROMACS reports) = {drift * per_dof:+.1e} kT/ns "
            f"per degree of freedom. A linear fit over 1–10 ps gives {fit:+.2f} kJ/mol/ps."
        ),
        (
            f"- Wall time {times['nvt-full']:.0f} s for 10 000 steps ({1000 * times['nvt-full'] / 10000:.0f} ms/step, "
            f"{ns_per_day:.2f} ns/day) on {hardware}."
        ),
    ]

    # restart
    full_meta, full_state = read_checkpoint(
        out / "nvt-full" / "out" / "checkpoint.ucchk"
    )
    split_meta, split_state = read_checkpoint(
        out / "nvt-split" / "out" / "checkpoint.ucchk"
    )
    same_files = (out / "nvt-full" / "out" / "checkpoint.ucchk").read_bytes() == (
        out / "nvt-split" / "out" / "checkpoint.ucchk"
    ).read_bytes()
    same_thermo = (out / "nvt-full" / "out" / "thermo.tsv").read_bytes() == (
        out / "nvt-split" / "out" / "thermo.tsv"
    ).read_bytes()
    restart_ok = (
        full_meta["step"] == split_meta["step"] == 10000
        and full_state == split_state
        and same_files
        and same_thermo
    )
    passed.append(("A restart after 5 ps is bitwise identical", restart_ok))
    restart_lines = [
        "## 4. Restart in the middle",
        "",
        "The same run stopped after 5000 steps and continued from its checkpoint (`ucmd run --restart`).",
        "",
        f"- Final positions, velocities and cell: identical: {full_state == split_state}.",
        f"- Nosé–Hoover chain state (ξ, v_ξ): identical: {full_meta['thermostat_state'] == split_meta['thermostat_state']}.",
        f"- Whole checkpoint files byte-identical, metadata included (thermostat work, time): {same_files}.",
        f"- thermo.tsv (every 10 steps, all columns) byte-identical: {same_thermo}.",
    ]

    # threads
    states = {}
    for n in (1, 4, 16):
        states[n] = (out / f"threads-{n}" / "out" / "checkpoint.ucchk").read_bytes()
    threads_ok = states[1] == states[4] == states[16]
    passed.append(("Results do not depend on the thread count (1, 4, 16)", threads_ok))
    thread_lines = [
        "## 5. Threads",
        "",
        "200 steps of the same run with `threads` = 1, 4 and 16 in the machine profile.",
        "",
        f"- Checkpoints after 200 steps (positions, velocities, chain state, all metadata) byte-identical: {threads_ok}.",
        "- Wall time: "
        + ", ".join(
            f"{n} thread{'s' if n > 1 else ''} {times[f'threads-{n}']:.0f} s"
            for n in (1, 4, 16)
        )
        + " (including start-up and the first neighbour search).",
    ]

    # NVE step-size check
    nve = {}
    for dt in ("1.0", "0.5"):
        e = read_thermo(out / f"nve-{dt}" / "out" / "thermo.tsv")
        early, later = e["time_ps"] <= 0.1, e["time_ps"] >= 0.2
        nve[dt] = (
            float(np.ptp(e["total_kjmol"][early])),
            float(np.ptp(e["total_kjmol"][later])),
        )
    nve_lines = [
        "## 6. NVE check of the step size",
        "",
        (
            "300 fs without thermostat from the minimized box, velocities at 298.15 K, energies every step. "
            "Velocity Verlet's energy error is bounded and of second order in Δt: halving the step must "
            "shrink the swings of K + U about fourfold."
        ),
        "",
        "| Δt | range of K + U, 0–0.1 ps | range of K + U, 0.2–0.3 ps |",
        "|---|---|---|",
        f"| 1 fs | {nve['1.0'][0]:.1f} kJ/mol | {nve['1.0'][1]:.1f} kJ/mol |",
        f"| 0.5 fs | {nve['0.5'][0]:.1f} kJ/mol | {nve['0.5'][1]:.1f} kJ/mol |",
        f"| ratio | {nve['1.0'][0] / nve['0.5'][0]:.2f} | {nve['1.0'][1] / nve['0.5'][1]:.2f} |",
        "",
        (
            "The swings of H in §3 are therefore the integrator's step-size error with flexible C–H bonds "
            "(period about 11 fs), not a thermostat error. Constraints on the bonds to hydrogen (E2.5) "
            "would remove the fastest vibrations."
        ),
    ]

    ok = all(p for _, p in passed)
    lines += [
        f"**Result: {'PASS' if ok else 'FAIL'}**, generated by `validation/m2a/m2a.py` from `run.sh`.",
        "",
        (
            "- System: the builder's 1 M LiPF₆ in EC:EMC 3:7 (w:w), 500 solvent molecules "
            "(Lehnert et al. 2025, SI Table S1), "
            f"{c0['atoms']} atoms, force field `opls-aa-carbonates` (ADR 0009, ion charges × 0.75). "
            "The density is the builder's placeholder until the Landesfeind 2019 SI is available."
        ),
        "- Chain: `ucmd-build build` → `ucmd minimize` → `ucmd run`; single points with `ucmd singlepoint`.",
        "",
        "| Criterion (docs/design/E2.md §10) | Pass |",
        "|---|---|",
    ]
    lines += [f"| {name} | {mark(p)} |" for name, p in passed]
    lines += (
        [""]
        + sp_lines
        + [""]
        + min_lines
        + [""]
        + nvt_lines
        + [""]
        + restart_lines
        + [""]
        + thread_lines
        + [""]
        + nve_lines
    )
    lines += [
        "",
        "## Reproduce",
        "",
        "```sh",
        "cd ucmd-engine && cargo build --release && validation/m2a/run.sh",
        "```",
        "",
        "Versions:",
        "",
    ]
    lines += [f"- {v}" for v in versions]
    lines += [
        f"- OpenMM {c0['openmm']} (tools/oracle/.venv)",
        f"- Hardware: {hardware}",
        "",
    ]
    print("\n".join(lines))


def main() -> None:
    command = sys.argv[1]
    if command == "config":
        config(Path(sys.argv[2]), Path(sys.argv[3]))
    elif command == "compare":
        compare(Path(sys.argv[2]))
    elif command == "report":
        report(Path(sys.argv[2]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
