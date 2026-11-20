#!/usr/bin/env bash
# `ucmd-openmm` in the HIP image (2026-11-20, GPU hang test approved by the author): the OpenCL
# wrapper with the image of Dockerfile.hip. Use with platform = "HIP" in the machine profile.
export UCMD_OPENMM_IMAGE=ucmd-openmm:gfx906-hip
exec "$(dirname "${BASH_SOURCE[0]}")/ucmd-openmm-docker.sh" "$@"
