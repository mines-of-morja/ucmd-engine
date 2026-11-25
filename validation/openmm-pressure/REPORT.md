# Bridge pressure: stepping context against a second context (GPU)

- **Date:** 2026-11-25. **Result: the large drift was not reproduced outside the bridge.** This check
  does not by itself validate the fix of c2cc427; it narrows what the fault needs.
- **Setup:** 512 SPC/Fw, ρ = 1.012 g/cm³, NVT 298.16 K, 1 fs, Nosé–Hoover chain, PME, tail; AMD
  Radeon VII (gfx906), HIP, double precision, OpenMM 8.5 (b55e60882dff), image
  `ucmd-openmm:gfx906-hip`. Bridge sources of 2a4cc49 (contains c2cc427). 100 ps from the same start,
  system and integrator built by the bridge's functions; identity tracker read every 25 steps
  (10 in the first 5 ps).
- **Script:** `pressure_bias.py`, through the queue:

      run.sh <study>/runs/spcfw_512_rho1012/T25.01C/run.toml 100000 500   # every500.txt
      run.sh <study>/runs/spcfw_512_rho1012/T25.01C/run.toml 100000 100   # every100.txt

  with `<study>` = `studies/2026-11-23_water-pressure-vs-density` of the umbrella repository (the
  run.toml is local data there).
- **Compared on the same state:** *stepping* = `computeCurrentPressure` asked of the stepping context
  (the estimator before c2cc427); *probe* = `PressureProbe` (now); *energy-only* = molecule centres
  scaled by (1 ± 10⁻³)^(1/3), energy differenced in a third context, plus the kinetic part.

## Numbers (bar, means over 20 ps blocks)

| Block (ps) | every 500: stepping − probe | probe − energy | every 100: stepping − probe | probe − energy |
|---|---|---|---|---|
| 0–20 | 0.0 | −7.9 | 0.0 | 1.2 |
| 20–40 | 0.0 | 9.2 | 0.8 | −2.8 |
| 40–60 | 0.0 | 2.0 | 1.7 | 1.4 |
| 60–80 | 0.0 | 5.2 | 1.0 | −0.1 |
| 80–100 | 0.0 | 12.6 | 1.7 | −1.2 |
| evaluations | 200 | | 1000 | |
| label exchanges | 0 | | 4748 | |

## What this shows

1. The probe agrees with the energy-only calculation (block means within 13 bar at 200 samples,
   within 3 bar at 1000). The probe is a sound estimator.
2. With 200 calls the stepping context gives the same value and exchanges no labels. With 1000 calls
   (the bridge's cadence) labels are exchanged and the stepping value is higher by up to 1.7 bar: the
   direction of the fault, at about 1/50 of the size seen in the bridge.
3. **Not reproduced:** the bridge's own 100 ps run showed +28, +68, +80, +79, +98 bar and 93 844
   exchanges (study README, "Why the pressure drifted"). The bridge does things this script does not:
   it reads energies and writes rows every 100 steps and takes `createCheckpoint` at trajectory and
   checkpoint steps, right after the pressure call. Which of these enlarges the effect is not tested.
   The mechanism inside OpenMM is not identified.

## Evidence for the fix that does not come from this check

- Study 2026-11-23, same design before and after c2cc427: drift +383 to +489 bar/ns before, −39 to −5
  ± 20–56 bar/ns after (2 ns × 3 densities).
- Final state of the drifting run at ρ = 1.012: in-run +454 bar, fresh context −608 bar, energy-only
  −610 bar.
- 40 ps bridge run after the fix: last in-run value 953.55 bar, after checkpoint reload 953.6 bar.

## Open

- Run the bridge before c2cc427 with and without `createCheckpoint` to find what the drift needs.
