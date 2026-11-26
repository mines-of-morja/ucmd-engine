#!/usr/bin/env bash
# GPU check of the bridge's pressure (needs the HIP image and a free GPU; one GPU job at a time).
#   validation/openmm-pressure/run.sh RUN.toml [STEPS [EVERY]]   (default 100000 steps = 100 ps at 1 fs, 500)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
RUN="$(realpath "$1")"
exec docker run --rm -i --init --device /dev/kfd --group-add "$(stat -c %g /dev/kfd)" \
  --device /dev/dri/renderD129 --group-add "$(stat -c %g /dev/dri/renderD129)" \
  -v "$WORKSPACE:$WORKSPACE" -w "$(dirname "$RUN")" --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e PYTHONPATH="$ROOT/bridge/src" "${UCMD_OPENMM_IMAGE:-ucmd-openmm:gfx906-hip}" \
  python "$HERE/pressure_bias.py" "$RUN" "${2:-100000}" "${3:-500}"
