# GPU image with OpenMM's HIP platform (2026-11-20), next to the OpenCL image of `Dockerfile`.
#
# Why: OpenCL runs on the MI50 hang about every 10–30 min. The GPU finishes its kernels, but the
# host never receives the completion event (bridge threads in kfd_wait_on_events; nothing in the
# kernel log). HIP submits and signals through a different path; whether it hangs less is the
# test.
#
# Build on top of the OpenCL image, which already has micromamba and the ROCm base:
#   docker build -t ucmd-openmm:gfx906-hip -f bridge/docker/Dockerfile.hip bridge/docker
# Use:  UCMD_OPENMM_IMAGE=ucmd-openmm:gfx906-hip, and platform = "HIP" in the machine profile.
#
# OpenMM 8.5.0 with the OpenMM-HIP-7 plugin from PyPI (the official HIP distribution; the
# conda-forge OpenMM has no HIP platform). It is a different OpenMM version from the OpenCL image
# (8.3.1): the bridge's conformance check against the reference backend applies to it as to any
# backend.
ARG BASE=ucmd-openmm:gfx906
FROM ${BASE}
RUN micromamba create -y -p /opt/ucmd-hip -c conda-forge python=3.12 numpy h5py pip \
 && micromamba clean -a -y \
 && /opt/ucmd-hip/bin/pip install --no-cache-dir "openmm==8.5.0" "OpenMM-HIP-7==8.5.0"
# the HIP runtime and hipRTC of the base image's ROCm SDK (gfx906 build)
ENV LD_LIBRARY_PATH=/opt/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib \
    PATH=/opt/ucmd-hip/bin:$PATH
