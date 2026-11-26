# GPU image for the OpenMM backend (E3.7). Build:
#   docker build -t ucmd-openmm:gfx906 --build-arg BASE=<a ROCm image for your GPU> bridge/docker
# The base must provide ROCm's OpenCL runtime (libamdocl64). The author's workstation uses a
# local ROCm 7.1 image for gfx906 (Radeon VII); a public alternative is rocm/dev-ubuntu-24.04.
ARG BASE=mi50-rocm714:gfx906v5
FROM ${BASE}
# OpenCL ICD: point the loader at ROCm's OpenCL runtime (path inside the base image).
ARG AMDOCL=/opt/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/opencl/libamdocl64.so.2
RUN apt-get update && apt-get install -y --no-install-recommends curl bzip2 ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /etc/OpenCL/vendors && echo "${AMDOCL}" > /etc/OpenCL/vendors/amdocl64.icd \
 && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /usr/local bin/micromamba
# conda-forge's OpenMM has the OpenCL platform (the PyPI wheel has only CPU and Reference).
RUN micromamba create -y -p /opt/ucmd-env -c conda-forge python=3.12 openmm=8.3 ocl-icd numpy h5py \
 && micromamba clean -a -y
ENV PATH=/opt/ucmd-env/bin:$PATH OCL_ICD_VENDORS=/etc/OpenCL/vendors
