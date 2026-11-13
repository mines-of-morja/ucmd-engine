#!/usr/bin/env bash
# M2a "core stands" (docs/design/E2.md §10): the builder's 1 M LiPF6 in EC:EMC 3:7 package,
# single points against OpenMM before and after `ucmd minimize`, a 10 ps NVT run with
# Nosé–Hoover chains and PME, a restart in the middle, runs on 1, 4 and 16 threads, and an
# NVE check of the step size.
# Writes REPORT.md. Needs the workspace layout with ucmd-builder/.venv (includes Packmol) and
# tools/oracle/.venv (OpenMM), and a release build: cargo build --release. About 30 minutes on
# 16 threads.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(realpath ../../..)
UCMD=$(realpath ../../target/release/ucmd)
BUILD=$ROOT/ucmd-builder/.venv/bin/ucmd-build
PY=$ROOT/tools/oracle/.venv/bin/python
rm -rf out && mkdir -p out
timed() { # label command…: append "label seconds" to out/times.txt
  local label=$1
  shift
  /usr/bin/time -f "$label %e" -a -o out/times.txt "$@"
}
{
  "$UCMD" --version
  "$BUILD" --version
  echo "umbrella $(git -C "$ROOT" rev-parse --short=12 HEAD)"
} > out/versions.txt

# 1. Build: Lehnert 2025 Table S1 composition, force field of ADR 0009 (zeta = 0.75).
timed build "$BUILD" build "$ROOT/ucmd-builder/examples/lp57-1M.toml" -o out/lp57-1M.ucsys -q

# 2. Single points: engine defaults, engine with OpenMM's alpha and grid given explicitly, OpenMM.
singlepoint() { # package directory
  "$PY" m2a.py config "$1" "$2"
  "$UCMD" singlepoint "$2/run.toml" --forces "$2/forces.npy" > "$2/singlepoint.json"
  "$UCMD" singlepoint "$2/explicit.toml" --forces "$2/forces-explicit.npy" > "$2/singlepoint-explicit.json"
  "$PY" m2a.py compare "$2"
}
singlepoint out/lp57-1M.ucsys out/sp-built

# 3. FIRE minimization until max |F| < 100 kJ/mol/nm, then the same single points.
timed minimize "$UCMD" minimize out/sp-built/run.toml --out out/minimized.ucsys 2> out/minimize.log
singlepoint out/minimized.ucsys out/sp-minimized

# 4. 10 ps NVT in one go.
mkdir -p out/nvt-full out/nvt-split
cp nvt.toml out/nvt-full/run.toml
timed nvt-full "$UCMD" run out/nvt-full/run.toml 2> out/nvt-full/log.txt

# 5. The same run, stopped after 5 ps and restarted from the checkpoint.
sed 's/^steps = 10000$/steps = 5000/' nvt.toml > out/nvt-split/run.toml
timed nvt-split-first "$UCMD" run out/nvt-split/run.toml 2> out/nvt-split/log-first.txt
cp nvt.toml out/nvt-split/run.toml
timed nvt-split-second "$UCMD" run out/nvt-split/run.toml \
  --restart out/nvt-split/out/checkpoint.ucchk 2> out/nvt-split/log-second.txt

# 6. 200 steps on 1, 4 and 16 threads (thread count from the machine profile).
for t in 1 4 16; do
  mkdir -p out/threads-$t
  sed -e 's/^steps = 10000$/steps = 200/' -e 's/^every = 1000$/every = 200/' nvt.toml > out/threads-$t/run.toml
  printf 'threads = %d\n' "$t" > out/threads-$t/machine.toml
  timed threads-$t env UCMD_MACHINE_PROFILE="$PWD/out/threads-$t/machine.toml" \
    "$UCMD" run out/threads-$t/run.toml 2> out/threads-$t/log.txt
done

# 7. NVE for 300 fs with 1 fs and 0.5 fs steps: how K + U depends on the step size.
for dt in 1.0 0.5; do
  mkdir -p out/nve-$dt
  steps=$([[ $dt == 1.0 ]] && echo 300 || echo 600)
  sed -e "s/^timestep_fs = 1.0$/timestep_fs = $dt/" -e "s/^steps = 300$/steps = $steps/" nve.toml > out/nve-$dt/run.toml
  timed nve-$dt "$UCMD" run out/nve-$dt/run.toml 2> out/nve-$dt/log.txt
done

# 8. Report.
"$PY" m2a.py report out > REPORT.md
echo "M2a: see validation/m2a/REPORT.md"
