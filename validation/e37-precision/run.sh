#!/usr/bin/env bash
# ADR 0012: is mixed precision on the GPU backend statistically the same as double? (REPORT.md)
# Needs the workspace: ucmd (release), the builder venv, the bridge GPU image and wrapper, the
# analysis venv. About 2 h on a Radeon VII.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../../..
UCMD=$(realpath ../../target/release/ucmd)
WRAPPER=$(realpath ../../bridge/docker/ucmd-openmm-docker.sh)
PY=$(realpath ../../bridge/.venv/bin/python)
CLI=$(realpath $ROOT/ucmd-analysis/.venv/bin/ucmd-analyze)
rm -rf out && mkdir -p out
"$ROOT/ucmd-builder/.venv/bin/ucmd-build" build emc_pure.toml -o out/built.ucsys -q
cat > out/minimize.toml <<TOML
spec_version = "0.1.0"
system = "built.ucsys"
[run]
integrator = "velocity-verlet"
timestep_fs = 1.0
steps = 0
seed = 1
[[providers]]
kind = "force-field"
cutoff_nm = 1.4
coulomb = "pme"
ewald_tolerance = 1e-5
dispersion_correction = true
TOML
(cd out && "$UCMD" minimize minimize.toml --out emc.ucsys)
for prec in double mixed; do
  printf '[executables]\nucmd-openmm = "%s"\n[backend]\nkind = "openmm"\nprecision = "%s"\n' \
    "$WRAPPER" "$prec" > "out/profile-$prec.toml"
  mkdir -p "out/$prec" && cp npt.toml "out/$prec/run.toml"
  (cd "out/$prec" && UCMD_MACHINE_PROFILE="../profile-$prec.toml" "$UCMD" run run.toml)
done
"$PY" make_package.py out/double/out/trajectory.h5 out/emc.ucsys out/equilibrated.ucsys
for prec in double mixed; do
  mkdir -p "out/nvt-$prec" && cp nvt.toml "out/nvt-$prec/run.toml"
  (cd "out/nvt-$prec" && UCMD_MACHINE_PROFILE="../profile-$prec.toml" "$UCMD" run run.toml)
done
"$PY" analyze.py out "$CLI" | tee out/result.txt
