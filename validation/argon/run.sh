#!/usr/bin/env bash
# E1.12 argon validation: ucmd vs OpenMM, see REPORT.md. Needs the UnitcellMD workspace layout:
# the spec fixtures, tools/oracle/.venv (OpenMM) and spec/validator/.venv (numpy, h5py).
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../../..
UCMD=../../target/release/ucmd
rm -rf out && mkdir -p out
cp -r "$ROOT/spec/fixtures/valid/argon-864" out/argon-864.ucsys
"$UCMD" run equilibrate.toml
"$UCMD" export equilibrate.toml --checkpoint out/equilibrate/checkpoint.ucchk --out out/equilibrated.ucsys
/usr/bin/time -f "%e" -o out/ucmd_seconds.txt "$UCMD" run production.toml
"$ROOT/tools/oracle/.venv/bin/python" openmm_nve.py out/equilibrated.ucsys out/openmm.npz
"$ROOT/spec/validator/.venv/bin/python" analyze.py out
