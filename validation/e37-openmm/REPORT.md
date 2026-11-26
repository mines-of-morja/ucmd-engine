# E3.7 validation: the OpenMM backend on the GPU against the CPU reference backend

- **Date:** 2026-11-17.
- **Engine:** ucmd 0.0.0 (c9098de) with ucmd-bridge 0.1.0.dev0, branch `e37-openmm`.
- **GPU side:** OpenMM 8.3.1 (conda-forge), OpenCL platform via ROCm 7.1 in a container
  (`bridge/docker`), AMD Radeon VII (gfx906).
- **System:** 1 M LiPF₆ in EC:EMC 3:7, 7036 atoms, Lehnert's box (L = 4.27 nm); 1.4 nm cutoff,
  PME to 1e-5, 1 fs.
- **Criteria (ADR 0010):**
  - numerical level: a start-up single point within 1e-6 (energy) and 1e-5 (forces, relative);
  - statistical level: ensemble averages agree within their error bars.

## 1. Single point (the start-up conformance check)

The configuration is the start positions displaced by a seeded 0.01 nm, so that forces have
thermal size. On a minimized structure the largest force is only 98 kJ/mol/nm, and relative
force errors look inflated there.

| OpenCL precision | energy (relative) | forces (max relative) | speed, stepping only |
|---|---|---|---|
| double | 1.3e-8 | 8.7e-8 | 34 ns/day |
| mixed | 1.3e-7 | 5.5e-5 (thermal configuration) | 77 ns/day |
| CPU reference, 16 threads | – | – | ~1 ns/day |

- **Double** passes ADR 0010.
- **Mixed fails** the force tolerance by a factor of 5; using it needs an ADR amendment.
- **Single** precision produced NaN under Nosé–Hoover.
- In one process, the second and third OpenMM contexts on the GPU gave NaN coordinates. Each
  run uses one context, so this does not affect runs.

## 2. Same state, same dynamics (NVE)

- **Setup:** the state at step 60000 of a CPU run (positions and velocities, `ucmd export`),
  continued for 1 ps of velocity Verlet on both backends.

| t | ΔU GPU − CPU (kJ/mol) | ΔE_tot (kJ/mol) |
|---|---|---|
| 0.01 ps | −0.0033 | −0.0039 |
| 0.10 ps | −0.0029 | −0.0039 |
| 0.50 ps | −0.0076 | −0.0038 |
| 1.00 ps | −0.58 (chaos sets in) | −0.0055 |

The total-energy change over 1 ps is 2.42 kJ/mol on both backends.

Two bugs were found on the way:
1. **OpenMM's integrators keep half-step velocities.** VerletIntegrator and NoseHooverIntegrator
   read the start velocities as v(t − dt/2), while the engine uses v(t). The trajectory differed
   at once: +1825 kJ/mol in U after 10 fs, and the total energy was off by 417 kJ/mol.
2. **OpenMM's thermostat controls the half-step kinetic energy.** NoseHooverIntegrator held the
   full-step temperature at 288 K for a 293 K target, which is a different ensemble at 1 fs.

Both are gone: the bridge runs the engine's own operator splittings as OpenMM
CustomIntegrators, with full-step velocities.
- The Nosé–Hoover chain on argon equals the CPU engine's thermo table within 1e-6 over 100 steps
  (`bridge/tests`).

## 3. NVT from the minimized box (20 ps, different random velocities)

| window | GPU ⟨T⟩ | GPU ⟨U⟩ | CPU ⟨T⟩ | CPU ⟨U⟩ |
|---|---|---|---|---|
| 2–5 ps | 293.0 K | −39751 | 293.1 K | −39671 |
| 5–10 ps | 293.0 K | −40823 | 293.2 K | −40513 |
| 10–20 ps | 293.2 K | −41301 | 293.6 K | −41087 |

- The differences in U (80–310 kJ/mol) are within the scatter between two GPU replicas of this
  relaxing system (200–400 kJ/mol).
- The drift of H on the GPU over 10–20 ps is −1.1 kJ/mol/ps.
- Throughput with the study's output schedule: 22.6 ns/day (thermo every 20 steps, pressure every
  500 steps).

## 4. Open

- **Statistical level at equilibrium:** density, RDFs and diffusion of a pure solvent and the
  electrolyte on both backends, with error bars. The pure-solvent study
  `studies/2026-11-17_pure-solvent-densities` runs on the GPU; CPU counterparts are needed for
  the comparison.
- **Pressure:** OpenMM has no virial. The backend reports the molecular pressure of
  MonteCarloBarostat.computeCurrentPressure every 500 steps (about 70 ms each). Its mean equals
  the virial pressure in equilibrium; this still needs checking against the CPU's atomic virial
  pressure at equilibrium.
- **The RX 580** (gfx803) is not supported by ROCm 7. It needs a host OpenCL stack (Mesa
  rusticl), i.e. a system package.

## Reproduce

```sh
docker build -t ucmd-openmm:gfx906 --build-arg BASE=<ROCm image> bridge/docker
# machine profile: [executables] ucmd-openmm = ".../bridge/docker/ucmd-openmm-docker.sh"
#                  [backend] kind = "openmm"
ucmd run run.toml        # conformance is printed; outputs as with the CPU backend
```
