"""ADR 0012: mixed against double precision on the GPU backend (pure EMC, 25 °C).

usage: python analyze.py out ANALYZE_CLI
Criteria:
- NPT averages over 0.2–1 ns (ρ, U, T, P) agree within 3 combined standard errors (10 blocks);
- the carbonyl O–O RDF peak agrees within 3 combined errors (error from the two halves);
- the NVT drift of H in mixed precision is at most 2× that of double, plus 1 kJ/mol/ps.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

GRAM_PER_AMU = 1.0 / 6.02214076e23
out, cli = Path(sys.argv[1]), sys.argv[2]
mass_g = 500 * 104.105 * GRAM_PER_AMU


def thermo(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open() as fh:
        names = fh.readline().lstrip("# ").strip().split("\t")
        return names, np.loadtxt(fh)


def block(x: np.ndarray, n: int = 10) -> tuple[float, float]:
    b = np.array([c.mean() for c in np.array_split(x[np.isfinite(x)], n)])
    return float(b.mean()), float(b.std(ddof=1) / np.sqrt(n))


rows, ok = [], True
data = {p: thermo(out / p / "out/thermo.tsv") for p in ("double", "mixed")}
for column, label in (("density", "ρ (g/cm³)"), ("potential_kjmol", "U (kJ/mol)"),
                      ("temperature_K", "T (K)"), ("pressure_bar", "P (bar)")):  # fmt: skip
    vals = {}
    for p, (names, d) in data.items():
        t = d[:, names.index("time_ps")]
        w = t >= 200.0
        if column == "density":
            x = mass_g / (d[w, names.index("volume_nm3")] * 1e-21)
        else:
            x = d[w, names.index(column)]
        vals[p] = block(x)
    (a, ea), (b, eb) = vals["double"], vals["mixed"]
    z = abs(a - b) / np.hypot(ea, eb)
    ok &= z < 3
    rows.append(f"| {label} | {a:.5g} ± {ea:.2g} | {b:.5g} ± {eb:.2g} | {z:.2f} |")

peaks = {}
for p in ("double", "mixed"):
    halves = []
    for start, stop in ((200, 600), (600, 1001)):
        r = subprocess.run([cli, "rdf", str(out / p / "out/trajectory.h5"), "--a", "EMC:O:1",
                            "--b", "EMC:O:1", "--skip-ps", str(start), "--r-max-nm", "1.0"],
                           capture_output=True, text=True, check=True)  # fmt: skip
        halves.append(json.loads(r.stdout)["g_max"])
    peaks[p] = (float(np.mean(halves)), float(abs(halves[0] - halves[1]) / 2))
(a, ea), (b, eb) = peaks["double"], peaks["mixed"]
z_rdf = abs(a - b) / max(np.hypot(ea, eb), 1e-9)
ok &= z_rdf < 3
rows.append(f"| carbonyl O–O g(r) peak | {a:.4f} ± {ea:.2g} | {b:.4f} ± {eb:.2g} | {z_rdf:.2f} |")

drift = {}
for p in ("double", "mixed"):
    names, d = thermo(out / f"nvt-{p}" / "out/thermo.tsv")
    t, h = d[:, names.index("time_ps")], d[:, names.index("conserved_energy_kjmol")]
    w = t >= 20.0
    drift[p] = float(np.polyfit(t[w], h[w], 1)[0])
drift_ok = abs(drift["mixed"]) <= 2 * abs(drift["double"]) + 1.0
ok &= drift_ok
print("| quantity | double | mixed | z |\n|---|---|---|---|")
print("\n".join(rows))
print(f"\nNVT drift of H: double {drift['double']:+.3f}, mixed {drift['mixed']:+.3f} kJ/mol/ps "
      f"({'ok' if drift_ok else 'too large'})")  # fmt: skip
print("PASS" if ok else "FAIL")
