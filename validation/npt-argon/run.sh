#!/usr/bin/env bash
# E2.7 acceptance: argon NPT density, ucmd C-rescale vs OpenMM MonteCarloBarostat (REPORT.md).
# Needs the workspace layout: spec fixtures, tools/oracle/.venv (OpenMM), a release build, and
# ../argon/out/equilibrated.ucsys from validation/argon/run.sh (the E1.12 equilibrated state).
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../../..
UCMD=../../target/release/ucmd
rm -rf out && mkdir -p out
cp -r ../argon/out/equilibrated.ucsys out/equilibrated.ucsys
/usr/bin/time -f "%e" -o out/ucmd_seconds.txt "$UCMD" run npt.toml
"$ROOT/tools/oracle/.venv/bin/python" openmm_npt.py out/equilibrated.ucsys out/openmm.npz
"$ROOT/tools/oracle/.venv/bin/python" analyze.py out | tee out/result.txt
