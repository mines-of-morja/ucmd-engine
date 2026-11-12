"""E1.12: compare the ucmd and OpenMM NVE argon runs and write REPORT.md.

usage: python analyze.py <out dir>   (needs numpy and h5py)
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np

K_B = 0.00831446261815324  # kJ mol-1 K-1 (spec/units.md)
BAR = 16.605390671738466  # bar per kJ mol-1 nm-3
SIGMA, EPS, RC = 0.34, 120.0 * K_B, 0.85
N_ATOMS = 864
DOF = 3 * N_ATOMS - 3
RAHMAN_D_CM2_S = 2.43e-5


def block_mean(x: np.ndarray, blocks: int = 10) -> tuple[float, float]:
    """Mean and standard error from block averages."""
    usable = len(x) // blocks * blocks
    means = x[:usable].reshape(blocks, -1).mean(axis=1)
    return float(x.mean()), float(means.std(ddof=1) / np.sqrt(blocks))


def pair_virial_trace(frame: np.ndarray, box: float) -> float:
    d = frame[:, None, :] - frame[None, :, :]
    d -= box * np.round(d / box)
    r2 = (d**2).sum(-1)
    np.fill_diagonal(r2, np.inf)
    inside = r2 < RC**2
    sr6 = np.where(inside, (SIGMA**2 / r2) ** 3, 0.0)
    f_dot_r = np.where(inside, 24 * EPS * (2 * sr6**2 - sr6), 0.0)  # F_ij . r_ij
    return 0.5 * f_dot_r.sum()


def rdf(frames: np.ndarray, box: float, dr: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(0.0, box / 2, dr)
    hist = np.zeros(len(edges) - 1)
    iu = np.triu_indices(N_ATOMS, 1)
    for frame in frames:
        d = frame[:, None, :] - frame[None, :, :]
        d -= box * np.round(d / box)
        r = np.sqrt((d**2).sum(-1))[iu]
        hist += np.histogram(r, edges)[0]
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    ideal = len(frames) * 0.5 * N_ATOMS * (N_ATOMS - 1) / box**3 * shell
    return 0.5 * (edges[1:] + edges[:-1]), hist / ideal


def diffusion(frames: np.ndarray, dt_ps: float, fit=(5.0, 40.0)) -> float:
    """D in nm²/ps from the COM-corrected MSD averaged over time origins (Einstein)."""
    x = frames - frames.mean(axis=1, keepdims=True)
    lags = np.arange(1, len(x) // 2)
    msd = np.array([((x[lag:] - x[:-lag]) ** 2).sum(-1).mean() for lag in lags])
    t = lags * dt_ps
    window = (t >= fit[0]) & (t <= fit[1])
    slope = np.polyfit(t[window], msd[window], 1)[0]
    return slope / 6.0


def drift(time_ps: np.ndarray, total: np.ndarray) -> float:
    """Energy drift in kT per ns per degree of freedom."""
    slope = np.polyfit(time_ps, total, 1)[0]  # kJ/mol per ps
    return slope * 1000.0 / (K_B * 94.4 * DOF)


def main() -> None:
    out = Path(sys.argv[1])
    thermo = np.loadtxt(out / "production" / "thermo.tsv", comments="#")
    with h5py.File(out / "production" / "trajectory.h5", "r") as f:
        frames_u = f["particles/all/position/value"][()]
        box = float(f["particles/all/box/edges/value"][0, 0, 0])
    omm = np.load(out / "openmm.npz")
    e_o, frames_o, ke_o = omm["energies"], omm["frames"], omm["frame_ke"]

    u_u, u_o = block_mean(thermo[:, 2]), block_mean(e_o[:, 1])
    t_u = block_mean(2 * thermo[:, 3] / (DOF * K_B))
    t_o = block_mean(2 * e_o[:, 2] / (DOF * K_B))
    p_u = block_mean(thermo[:, 7])
    volume = box**3
    p_o_series = np.array([(2 * k + pair_virial_trace(fr, box)) / (3 * volume) * BAR for fr, k in zip(frames_o, ke_o)])
    p_o = block_mean(p_o_series)
    r, g_u = rdf(frames_u[::2], box)
    _, g_o = rdf(frames_o[::2], box)
    peak_u, peak_o = int(np.argmax(g_u)), int(np.argmax(g_o))
    d_u, d_o = diffusion(frames_u, 0.4), diffusion(frames_o, 0.4)
    drift_u = drift(thermo[:, 1], thermo[:, 2] + thermo[:, 3])
    drift_o = drift(e_o[:, 0] * 0.004, e_o[:, 1] + e_o[:, 2])

    def agree(a, b):
        return abs(a[0] - b[0]) <= 3 * np.hypot(a[1], b[1])

    checks = [
        (
            "<U> (kJ/mol)",
            f"{u_u[0]:.2f} ± {u_u[1]:.2f}",
            f"{u_o[0]:.2f} ± {u_o[1]:.2f}",
            agree(u_u, u_o),
            "within 3 combined standard errors",
        ),
        (
            "<T> (K)",
            f"{t_u[0]:.2f} ± {t_u[1]:.2f}",
            f"{t_o[0]:.2f} ± {t_o[1]:.2f}",
            agree(t_u, t_o),
            "within 3 combined standard errors",
        ),
        (
            "<P> (bar, no tail)",
            f"{p_u[0]:.1f} ± {p_u[1]:.1f}",
            f"{p_o[0]:.1f} ± {p_o[1]:.1f}",
            agree(p_u, p_o),
            "within 3 combined standard errors",
        ),
        (
            "g(r) first peak (nm)",
            f"{r[peak_u]:.4f} (g = {g_u[peak_u]:.3f})",
            f"{r[peak_o]:.4f} (g = {g_o[peak_o]:.3f})",
            abs(peak_u - peak_o) <= 1 and abs(g_u[peak_u] / g_o[peak_o] - 1) < 0.03,
            "same bin ±1, height within 3 %",
        ),
        (
            "max |Δg(r)|",
            f"{np.max(np.abs(g_u - g_o)):.3f}",
            "",
            np.max(np.abs(g_u - g_o)) < 0.1,
            "< 0.1 (statistical noise)",
        ),
        (
            "D (10⁻⁵ cm²/s)",
            f"{d_u * 1e3:.3f}",
            f"{d_o * 1e3:.3f}",
            abs(d_u / d_o - 1) < 0.15,
            "within 15 % (one 100 ps run each)",
        ),
        (
            "D vs Rahman 1964 (2.43)",
            f"{d_u * 1e3:.3f}",
            "2.43",
            abs(d_u * 1e3 / 2.43 - 1) < 0.2,
            "within 20 % (different cutoff, finite size)",
        ),
        (
            "energy drift (kT/ns/DOF)",
            f"{drift_u:.2e}",
            f"{drift_o:.2e}",
            abs(drift_u) <= 1.5 * max(abs(drift_o), 1e-4),
            "|ucmd| ≤ 1.5 × max(|OpenMM|, 1e-4)",
        ),
    ]
    status = "PASS" if all(c[3] for c in checks) else "FAIL"
    seconds_file = out / "ucmd_seconds.txt"
    ucmd_seconds = f"{float(seconds_file.read_text().split()[-1]):.0f} s" if seconds_file.is_file() else "not recorded"
    lines = [
        "# E1.12 — argon validation (Rahman 1964 state point)",
        "",
        f"**Result: {status}** — generated by `validation/argon/analyze.py` from `run.sh`.",
        "",
        "- System: `spec/fixtures/valid/argon-864` (864 atoms, 1.374 g/cm³, σ = 0.34 nm, ε/k_B = 120 K).",
        "- Model: Lennard-Jones, potential shift at r_c = 0.85 nm (2.5σ), no tail correction, identical in both engines.",
        "- Protocol: 20 ps CSVR equilibration at 94.4 K with ucmd (dt 4 fs), exported with `ucmd export`; then 100 ps NVE",
        (
            "  production (25 000 × 4 fs) from the identical state in ucmd and in OpenMM "
            f"{omm['openmm']} (Reference platform, `VerletIntegrator`)."
        ),
        "- Statistics: block averages (10 blocks); g(r) from every second stored frame; D from the COM-corrected MSD, fit 5–40 ps.",
        "",
        "| Quantity | ucmd | OpenMM | Pass | Criterion |",
        "|---|---|---|---|---|",
        *[f"| {name} | {a} | {b} | {'✅' if ok else '❌'} | {crit} |" for name, a, b, ok, crit in checks],
        "",
        f"Wall time of the 100 ps production (one core): ucmd {ucmd_seconds}, OpenMM Reference {float(omm['seconds']):.0f} s.",
    ]
    (Path(__file__).parent / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    json.dump({c[0]: bool(c[3]) for c in checks}, sys.stderr)


if __name__ == "__main__":
    main()
