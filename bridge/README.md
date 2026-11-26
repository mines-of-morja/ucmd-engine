# ucmd-bridge

External backends for `ucmd run` (PLAN.md E3.7, ADR 0010). The first one is OpenMM, which runs
the model of a system package on a GPU through OpenCL.

## How it is used

The machine profile selects the backend; packages and `run.toml` stay unchanged:

```toml
[executables]
ucmd-openmm = "/abs/path/to/ucmd-engine/bridge/docker/ucmd-openmm-docker.sh"

[backend]
kind = "openmm"
platform = "OpenCL"      # default; "CPU" or "Reference" without a GPU
precision = "double"     # default; "mixed" is 2.3x faster but fails ADR 0010's force tolerance
```

Then `ucmd run run.toml` works as usual, from the shell, the study driver or the GUI:

1. `ucmd` evaluates the reference single point with the CPU engine. It uses the start
   positions, displaced by a seeded 0.01 nm so that forces have thermal size.
2. It hands over to `ucmd-openmm` (exec, so stop signals reach it).
3. `ucmd-openmm` refuses to run unless energy and forces agree within 1e-6 and 1e-5 relative.
4. It runs with the engine's integrators as OpenMM CustomIntegrators: velocity Verlet,
   Nosé–Hoover chain, Langevin BAOAB, all with full-step velocities. C-rescale becomes
   MonteCarloBarostat (same ensemble).
5. It writes the spec's thermo table and H5MD or XYZ trajectory, and its own checkpoint.
6. It writes positions in canonical order. On the GPU, OpenMM's output exchanged identical
   molecules (pilot of 2026-11-20); `identity.py` tracks the molecules every 250 steps and
   undoes it. An ambiguous match stops the run (see `validation/e37-identity`).

## Repairing trajectories written before the identity fix

GPU trajectories written before 2026-11-20 can contain exchanged molecule labels. Collective
quantities (κ, t₊, L_ij) are right; self-diffusion and residence times are not. Write a
corrected copy (the source stays unchanged; the copy records what was done):

```sh
python -m ucmd_bridge.repair out/trajectory.h5 out/trajectory.repaired.h5 --package SYSTEM.ucsys
```

## GPU container (AMD, ROCm)

The host needs no ROCm or OpenCL; the container brings them.

```sh
docker build -t ucmd-openmm:gfx906 --build-arg BASE=<ROCm image for your GPU> bridge/docker
```

`docker/ucmd-openmm-docker.sh` runs it:
- it passes `/dev/kfd` and the GPUs in `UCMD_OPENMM_DEVICES` (default `/dev/dri/renderD129`);
- it mounts the workspace and the current directory at the same paths, and runs as your user.

## Development

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[openmm,dev]' ../../spec/validator
.venv/bin/pytest          # needs target/release/ucmd (or debug) and ../../spec fixtures
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

The tests run the whole chain on OpenMM's Reference platform:
- argon, a charged crystal with PME;
- restarts, refusals and a graceful stop;
- the Nosé–Hoover chain against the CPU engine from the same state (thermo equal to 1e-6).
