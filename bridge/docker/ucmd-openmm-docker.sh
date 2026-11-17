#!/usr/bin/env bash
# `ucmd-openmm` in the GPU container (E3.7). Point the machine profile at this script:
#   [executables]
#   ucmd-openmm = "/abs/path/to/ucmd-engine/bridge/docker/ucmd-openmm-docker.sh"
#   [backend]
#   kind = "openmm"            # platform "OpenCL", precision "double" by default
#
# Machine-specific settings come from the environment:
#   UCMD_OPENMM_IMAGE    image built from bridge/docker/Dockerfile (default ucmd-openmm:gfx906)
#   UCMD_OPENMM_DEVICES  GPU render nodes to pass in (default /dev/dri/renderD129)
#   UCMD_OPENMM_MOUNTS   extra directories to mount (the workspace and $PWD always are)
# The bridge source is mounted, not baked in, so the image only holds OpenMM and the GPU stack.
set -euo pipefail
BRIDGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "$BRIDGE/../.." && pwd)"
IMAGE="${UCMD_OPENMM_IMAGE:-ucmd-openmm:gfx906}"
args=(--rm -i --init --device /dev/kfd --group-add "$(stat -c %g /dev/kfd)")
for dev in ${UCMD_OPENMM_DEVICES:-/dev/dri/renderD129}; do
  args+=(--device "$dev" --group-add "$(stat -c %g "$dev")")
done
for dir in "$WORKSPACE" "$PWD" ${UCMD_OPENMM_MOUNTS:-}; do
  args+=(-v "$dir:$dir")
done
exec docker run "${args[@]}" -w "$PWD" --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e PYTHONPATH="$BRIDGE/src" "$IMAGE" python -m ucmd_bridge.openmm_backend "$@"
