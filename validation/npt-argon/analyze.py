"""E2.7: compare the argon density of ucmd (C-rescale) and OpenMM (MonteCarloBarostat).

usage: python analyze.py out
Discards the first 50 ps; block averages (20 blocks of 25 ps) give the standard errors.
Pass: the densities agree within 3 combined standard errors (PLAN.md E2.7).
"""

import sys
from pathlib import Path

import numpy as np

GRAM_PER_AMU = 1.66053906660e-24  # CODATA 2018 atomic mass constant in g
CM3_PER_NM3 = 1e-21
DT_PS, SKIP_PS, BLOCKS = 0.004, 50.0, 20


def block_mean(x: np.ndarray) -> tuple[float, float]:
    blocks = np.array([b.mean() for b in np.array_split(x, BLOCKS)])
    return float(blocks.mean()), float(blocks.std(ddof=1) / np.sqrt(BLOCKS))


out = Path(sys.argv[1])
omm = np.load(out / "openmm.npz")
mass_g = float(omm["mass_amu"]) * GRAM_PER_AMU
rows = []
with open(out / "ucmd" / "thermo.tsv") as fh:
    names = fh.readline().lstrip("# ").strip().split("\t")
    data = np.loadtxt(fh)
u_t, u_v = data[:, names.index("time_ps")], data[:, names.index("volume_nm3")]
o_t, o_v = omm["volumes"][:, 0] * DT_PS, omm["volumes"][:, 1]
result = {}
for name, t, v in (("ucmd C-rescale", u_t, u_v), ("OpenMM MonteCarloBarostat", o_t, o_v)):
    rho = mass_g / (v[t >= SKIP_PS] * CM3_PER_NM3)
    mean, err = block_mean(rho)
    vmean, verr = block_mean(v[t >= SKIP_PS])
    result[name] = (mean, err)
    print(f"{name:28s} rho = {mean:.5f} ± {err:.5f} g/cm³   <V> = {vmean:.4f} ± {verr:.4f} nm³"
          f"   σ_V = {v[t >= SKIP_PS].std():.4f} nm³   ({len(rho)} samples)")
(a, ea), (b, eb) = result.values()
z = abs(a - b) / np.hypot(ea, eb)
print(f"difference {a - b:+.5f} g/cm³ = {z:.2f} combined standard errors: {'PASS' if z < 3 else 'FAIL'}")
