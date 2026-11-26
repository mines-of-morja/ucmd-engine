# E2.7 validation: argon density under C-rescale vs OpenMM's MonteCarloBarostat

- **Date:** 2026-11-17. **Engine:** ucmd 0.0.0 (475f31c, branch `e27-barostat`), release build,
  16 threads. **Oracle:** OpenMM 8.6.1, CPU platform (ADR 0005).
- **Criterion (PLAN.md E2.7):** argon (and water) densities match OpenMM's MonteCarloBarostat
  within the combined uncertainty.
- **Result: PASS for argon.** Water is still open (below).

## Setup

| | ucmd | OpenMM |
|---|---|---|
| system | Rahman 1964 argon, 864 atoms, the E1.12 equilibrated state (`../argon`) | same package |
| potential | LJ shifted at 0.85 nm, no tail correction | the same, `tools/oracle` `lj_system` (CustomNonbondedForce) |
| thermostat | CSVR, 94.4 K, τ = 0.5 ps | Langevin middle, 94.4 K, 1 ps⁻¹ |
| barostat | C-rescale, 100 bar, τ_p = 1 ps, β = 2·10⁻⁴ bar⁻¹, every 10 steps | MonteCarloBarostat, 100 bar, every 25 steps |
| time step, length | 4 fs, 137 500 steps (550 ps) | the same |

- Both sides use the identical Hamiltonian. A tail correction is left out because OpenMM's
  long-range correction of a CustomNonbondedForce would integrate the shifted expression; the
  barostat comparison does not need it.
- 100 bar keeps the cut-off LJ liquid well above its vapour pressure at 94.4 K.
- Analysis (`analyze.py`): the first 50 ps are discarded; standard errors from 20 blocks of 25 ps.

## Numbers

| | ⟨ρ⟩ (g/cm³) | ⟨V⟩ (nm³) | σ_V (nm³) | wall time |
|---|---|---|---|---|
| ucmd C-rescale | 1.29429 ± 0.00112 | 44.2864 ± 0.0383 | 0.4424 | 471 s |
| OpenMM MonteCarloBarostat | 1.29624 ± 0.00159 | 44.2198 ± 0.0543 | 0.4505 | 259 s |

- Difference −0.00195 g/cm³ (−0.15 %) = **1.00 combined standard errors** (pass: < 3).
- The volume fluctuations agree too (σ_V within 2 %). They measure the compressibility,
  κ_T = σ_V²/(k_BT⟨V⟩), so the barostat samples the right distribution, not only the right mean.

## Also tested (unit and CLI tests, `cargo test`)

- **Ideal gas, analytic:** ⟨V⟩ = (N+1)k_BT/P₀ = 8.698 nm³ expected, 8.705 ± 0.043 measured;
  Var V within 3 % of (N+1)(k_BT/P₀)².
  - The test first ran with the centre-of-mass momentum kept, and gave (N+2)/(N+1) too much
    volume. C-rescale's k_BT/V term assumes the COM momentum is removed, as in MD. Documented
    in `barostat.rs`.
- **Restart:** 200 steps = 100 + restart + 100 steps bitwise, with the box changing.
- **Refusals:** a barostat without a thermostat, or without a periodic cell and virial.

## The conserved quantity under NPT

- H = K + U + P₀V + bath − barostat work. The energy of each rescaling, ΔU, is estimated from
  the virial.
- **First version (475f31c):** ΔU from the virial at the start of the step only. H drifted:
  - argon: +2.2 kJ/mol/ps, 45 kJ/mol over 20 ps;
  - the 7036-atom electrolyte: +88 kJ/mol/ps.
- **Using the virial after the move alone** gave the mirror image on the identical trajectory:
  −2.2 and −97 kJ/mol/ps. Either end alone misses ½(dε/3)²·∂²U/∂e², which the stiff bonds of
  molecules make large.
- **The trapezoid rule** (mean of both virials) cancels it:
  - argon: 0.00 kJ/mol/ps, H range 0.1 kJ/mol over 20 ps;
  - electrolyte: −4.2 kJ/mol/ps over 0.5–3 ps, the NVT level after a minimization.
- The trajectory is bitwise the same in all three versions (checked: T, P, V and U identical),
  so the densities above are unaffected.
- **With the LJ tail correction,** H cannot be exactly conserved under a changing volume. The
  tail pressure is the virial integral (P_tail·V ≈ 2·E_tail), not −dE_tail/dV = E_tail/V,
  because pairs cross the cutoff. The CLI test `the_conserved_energy_holds_under_npt`
  therefore runs without it.

## Open

- **Water:** the second half of the criterion (an SPC/Fw box with PME) is not run yet.
- **Anisotropic coupling** (for crystals) is not implemented; run.toml refuses it.

## Reproduce

```sh
cargo build --release
validation/argon/run.sh          # once, for ../argon/out/equilibrated.ucsys
validation/npt-argon/run.sh      # ~12 min; result in out/result.txt
```
