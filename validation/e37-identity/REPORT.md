# E3.7 — molecule identity in GPU trajectories

- **Status:**
  - fix and repair tool tested on the CPU (Reference platform) and on synthetic data;
  - repair applied to the 20 ns pilot.
  - **The GPU check of the fix is pending.** The GPU is busy with study A4.
- **Branch:** `e37-identity` (commits 40e91a8, e75d993), based on `e37-openmm` 1b6694e.

## The defect

**Run:** 1 M LiPF₆ in EC:EMC, 7036 atoms, 20 ns NVT, OpenMM 8.3.1, OpenCL, mixed precision,
the engine's Nosé–Hoover chain as a CustomIntegrator, MI50 (gfx906). Study
`2026-11-19_transport-pilot` (local data).

**What the output showed:**
- Identical molecules had their identities exchanged:
  - in 237 events, exactly every 84 000 steps, from step 170 000 on;
  - the phase stayed the same across all 9 restarts.
- Each event involved Li⁺, PF₆⁻, EC and EMC, up to 304 of the 332 EMC at once.
- The molecules themselves stayed intact. A label typically moved 2.7–4 nm, but some swaps
  were between neighbours only 0.4–0.6 nm apart.

**What stays correct:**
- Box images travel with the molecules. The sum of positions per species changes no more at
  events than between them (at most 1.3 against 1.7 nm per 2 ps).
- So κ, t₊ and the L_ij are unaffected.

**What is wrong:** self-diffusion and residence times.
- D_Li came out 3.5× too high, with 35 % block error.

**Cause:** not identified.
- OpenMM reorders identical molecules every 250 steps and maps them back through `atomIndex`
  (`platforms/common/src/ComputeContext.cpp`).
- The 84 000-step period and the fixed phase point to a trigger tied to the absolute step
  count.

## The fix (`bridge/src/ucmd_bridge/identity.py`, `openmm_backend.py`)

**Tracking:**
- Positions are read at least every 250 steps.
- Molecules that moved more than 0.2 nm (atom-wise RMSD) since the last read are matched among
  themselves.
- The permutation is kept, stored in every checkpoint, and restored on restart.

**Errors:**
- an ambiguous match (nearest / second nearest > 0.5) raises;
- so does a molecule that still moves more than 0.2 nm after relabelling.

**Also fixed:** the emergency checkpoint carried the step of the failure with the state of the
last good frame (670 against 650 in the new test). It now carries the frame's own step.

## Repair of the pilot (`python -m ucmd_bridge.repair`)

```sh
cd studies/2026-11-19_transport-pilot/runs/ec-emc_3-7w_1.0M/T20C
PYTHONPATH=<bridge>/src python -m ucmd_bridge.repair out/trajectory.h5 \
    out/trajectory.repaired.h5 --package ../../../systems/ec-emc_3-7w_1.0M/minimized.ucsys
```

**Method:** relabelling only at events, where some molecule moved more than 0.7 nm in 2 ps.
- Between events no molecule moved more than 0.6 nm.
- At an event, the assignment of least total distance is used (Hungarian algorithm).
- A greedy assignment fails at frame 967: it leaves EC 153 without its only continuation.

**Result:**

| quantity | value |
|---|---|
| events | 237, all 84 000 steps apart |
| relabellings | 95 758 |
| assignments with nearest / second nearest > 0.5 | 740 (0.8 %), resolved by least total distance |
| species position sums, original vs repaired | equal to 1.8·10⁻¹² nm |
| largest move of any molecule between frames afterwards | 0.38 (Li), 0.48 (PF₆), 0.56 (EC), 0.57 nm (EMC) |
| repeat with the final code | bit-identical positions |

**Transport** (`ucmd-analyze transport --skip-ps 1000 --blocks 5`, 293.15 K):

| | original file | repaired |
|---|---|---|
| D_Li (10⁻¹¹ m²/s) | 16.6 ± 5.8 | 4.74 ± 0.20 |
| D_PF₆ (10⁻¹¹ m²/s) | 28.3 ± 11.2 | 7.91 ± 0.62 |
| D_EC (10⁻¹¹ m²/s) | 11.7 ± 3.7 | 15.0 ± 0.4 |
| D_EMC (10⁻¹¹ m²/s) | 6.5 ± 0.9 | 9.82 ± 0.27 |
| κ (S/m) | 0.420 ± 0.086 | 0.420 ± 0.086 (identical) |
| t₊ | 0.286 ± 0.110 | 0.286 ± 0.110 (identical) |

## Tests

- `tests/test_identity.py`: 14 tests, synthetic, no GPU. They cover:
  - swaps undone exactly;
  - box images;
  - a swap of close neighbours;
  - ambiguity is an error;
  - the permutation survives a checkpoint;
  - repair-mode events;
  - the least-cost assignment, checked against brute force;
  - the frame-967 case.
- `tests/test_repair.py`: end to end on the NaCl fixture; a wrong package and an existing
  destination are refused.
- `tests/test_openmm_backend.py`: the checkpoint holds the permutation; the emergency
  checkpoint holds the last good step. The latter fails on the old code: 670 ≠ 650.
- Full suite: 23 passed; ruff clean.

## Pending: GPU check

- **Run:** the pilot system, OpenCL mixed, at least 250 000 steps, so that two events lie
  inside.
- **Pass criteria:**
  - the trajectory has no molecule moving more than 0.6 nm between 2 ps frames;
  - the tracker's exchange count is above zero;
  - throughput is within 3 % of the old rate (the read every 250 steps costs one extra energy
    evaluation).
