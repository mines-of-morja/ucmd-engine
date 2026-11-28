"""V-1: how often must the atomic pressure tensor be sampled for Green–Kubo? (CPU evidence)

Reads a thermo table written every `k` steps with the pressure tensor (spec 0.1.1), subsamples it
every m-th row and computes η by Green–Kubo for each interval, with the same estimator as
`ucmd-analyze viscosity` (five traceless components, FFT ACF, trapezoid, blocks). Also the power
spectrum of the shear components, to show where aliasing to zero frequency can come from.

    python sampling.py THERMO.tsv --temperature-K 300.2 --skip-ps 200 --out sampling.json

Units: bar, ps, Pa s. Numbers are printed and written to --out; nothing is judged here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KB = 1.380649e-23  # J/K, CODATA 2018 exact
BAR = 1e5  # Pa
COLS = ("xx", "yy", "zz", "xy", "xz", "yz")


def read(path: Path, skip_ps: float):
    with path.open() as fh:
        names = fh.readline().lstrip("# ").strip().split("\t")
    use = [names.index("time_ps"), names.index("volume_nm3")]
    use += [names.index(f"pressure_{c}_bar") for c in COLS]
    rows = np.loadtxt(path, usecols=use, ndmin=2)
    keep = rows[:, 0] >= skip_ps
    rows = rows[keep]
    t, vol = rows[:, 0], rows[:, 1]
    p = dict(zip(COLS, rows[:, 2:].T, strict=True))
    return t, float(vol.mean()), p


def shear(p: dict) -> np.ndarray:
    return np.stack([p["xy"], p["xz"], p["yz"], 0.5 * (p["xx"] - p["yy"]),
                     0.5 * (p["yy"] - p["zz"])])  # fmt: skip


def acf(x: np.ndarray, n_lag: int) -> np.ndarray:
    n = x.shape[1]
    f = np.fft.rfft(x - x.mean(axis=1, keepdims=True), n=2 * n, axis=1)
    c = np.fft.irfft(f * f.conj(), n=2 * n, axis=1)[:, :n_lag].real
    return c.mean(axis=0) / (n - np.arange(n_lag))


def eta_running(s: np.ndarray, dt_ps: float, vol: float, T: float, n_lag: int) -> np.ndarray:
    c = acf(s, n_lag) * BAR**2
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (c[1:] + c[:-1]))]) * dt_ps * 1e-12
    return vol * 1e-27 / (KB * T) * integral


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("thermo", type=Path)
    ap.add_argument("--temperature-K", type=float, required=True)
    ap.add_argument("--skip-ps", type=float, default=0.0)
    ap.add_argument("--max-lag-ps", type=float, default=5.0)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--intervals", default="1,2,3,4,5,6,8,10")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t, vol, p = read(a.thermo, a.skip_ps)
    s_all = shear(p)
    dt0 = float(np.median(np.diff(t)))
    result = {"thermo": a.thermo.name, "dt_ps": dt0, "samples": int(t.size),
              "volume_nm3": vol, "temperature_K": a.temperature_K, "max_lag_ps": a.max_lag_ps,
              "rms_shear_bar": float(np.sqrt((s_all[:3] ** 2).mean())), "intervals": []}  # fmt: skip
    for m in (int(x) for x in a.intervals.split(",")):
        s = s_all[:, ::m]
        dt = dt0 * m
        n_lag = round(a.max_lag_ps / dt) + 1
        whole = eta_running(s, dt, vol, a.temperature_K, n_lag)
        edges = np.linspace(0, s.shape[1], a.blocks + 1).astype(int)
        per = [eta_running(s[:, i:j], dt, vol, a.temperature_K, n_lag)[-1]
               for i, j in zip(edges[:-1], edges[1:], strict=True)]  # fmt: skip
        mean = float(np.mean(per))
        sem = float(np.std(per, ddof=1) / np.sqrt(len(per)))
        row = {"interval_fs": round(dt * 1000, 3), "eta_mPa_s": mean * 1e3,
               "sem_mPa_s": sem * 1e3, "eta_whole_mPa_s": float(whole[-1]) * 1e3,
               "eta_at_1ps_mPa_s": float(whole[min(n_lag - 1, round(1.0 / dt))]) * 1e3}  # fmt: skip
        result["intervals"].append(row)
        print(json.dumps(row), flush=True)
    # one-sided power spectrum of the three off-diagonal components (bar² ps), 0..Nyquist
    seg = 4096
    x = s_all[:3, : (s_all.shape[1] // seg) * seg].reshape(3, -1, seg)
    x = x - x.mean(axis=-1, keepdims=True)
    w = np.hanning(seg)
    spec = (np.abs(np.fft.rfft(x * w, axis=-1)) ** 2).mean(axis=(0, 1)) * dt0 / (w**2).sum()
    f_thz = np.fft.rfftfreq(seg, dt0)
    bands = []
    for lo, hi in ((0, 5), (5, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 120),
                   (120, 160), (160, 200), (200, 250)):  # fmt: skip
        sel = (f_thz >= lo) & (f_thz < hi)
        if sel.any():
            bands.append({"THz": [lo, hi], "mean_psd_bar2_ps": float(spec[sel].mean()),
                          "peak_THz": float(f_thz[sel][np.argmax(spec[sel])])})  # fmt: skip
    result["spectrum_bands"] = bands
    result["psd_zero_freq_bar2_ps"] = float(spec[:3].mean())
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
