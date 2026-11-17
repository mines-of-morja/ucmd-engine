//! Barostats as operators on positions, velocities and the cell (PLAN.md §2.5, E2.7).
//!
//! Stochastic cell rescaling, C-rescale (M. Bernetti, G. Bussi, J. Chem. Phys. 153, 114107
//! (2020)), isotropic. Every `every` steps the logarithm of the volume, ε = ln V, makes one
//! Euler–Maruyama step of their Eq. (5) over Δt = every·dt:
//!
//! dε = −(β_T/τ_p)(P₀ − P_int − k_B T/V) Δt + √(2 k_B T β_T Δt/(V τ_p)) R,  R ~ N(0, 1),
//!
//! with the isothermal compressibility β_T, the coupling time τ_p, the reference pressure P₀,
//! the instantaneous pressure P_int (kinetic plus virial) and the reference temperature T. The
//! positions and the cell are then scaled by μ = exp(dε/3), the velocities by 1/μ.
//!
//! β_T and τ_p set only how fast the volume relaxes, through their ratio; the sampled ensemble
//! (isothermal-isobaric) does not depend on them.
//!
//! The k_B T/V term assumes, as the paper does, that the centre-of-mass momentum is removed: the
//! kinetic pressure then counts N − 1 particles, and the term restores the N-th, so that the
//! volume follows p(V) ∝ V^N exp(−P₀V/k_BT). If the centre-of-mass momentum were kept, the
//! volume would come out too large by a factor (N + 2)/(N + 1), negligible for thousands of
//! atoms; the ideal-gas test in `dynamics.rs` measures exactly this.

use crate::random::standard_normal;
use rand_chacha::ChaCha8Rng;
use ucmd_core::units::{BAR_PER_KJMOL_NM3, BOLTZMANN_KJMOL_K};

/// Settings of [`CRescale`], in the units of `[barostat]` in run.toml.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CRescaleConfig {
    /// Reference pressure P₀ in bar.
    pub pressure_bar: f64,
    /// Coupling time τ_p in ps.
    pub tau_ps: f64,
    /// Isothermal compressibility β_T in bar⁻¹.
    pub compressibility_per_bar: f64,
    /// Steps between two applications.
    pub every: u64,
    /// Reference temperature T in K (the thermostat's).
    pub temperature_k: f64,
}

/// Isotropic stochastic cell rescaling (C-rescale).
pub struct CRescale {
    config: CRescaleConfig,
    rng: ChaCha8Rng,
}

impl CRescale {
    /// A barostat with its own random stream. Returns `None` for settings out of range
    /// (non-positive τ_p, β_T, temperature or interval, or a non-finite pressure).
    pub fn new(config: CRescaleConfig, rng: ChaCha8Rng) -> Option<Self> {
        let c = &config;
        let ok = c.tau_ps > 0.0
            && c.compressibility_per_bar > 0.0
            && c.temperature_k > 0.0
            && c.pressure_bar.is_finite()
            && c.every >= 1;
        ok.then_some(Self { config, rng })
    }

    /// The settings.
    pub fn config(&self) -> &CRescaleConfig {
        &self.config
    }

    /// The generator state, for checkpoints.
    pub fn rng(&self) -> &ChaCha8Rng {
        &self.rng
    }

    /// Replace the generator, after a restart.
    pub fn restore_rng(&mut self, rng: ChaCha8Rng) {
        self.rng = rng;
    }

    /// Steps between two applications.
    pub fn every(&self) -> u64 {
        self.config.every
    }

    /// One step of ε = ln V over `every` time steps of `dt_ps`: returns dε (dimensionless).
    ///
    /// `pressure_bar` is the instantaneous isotropic pressure (kinetic plus virial, the trace of
    /// the pressure tensor over 3) and `volume_nm3` the current volume.
    pub fn d_log_volume(&mut self, pressure_bar: f64, volume_nm3: f64, dt_ps: f64) -> f64 {
        let c = &self.config;
        let interval = c.every as f64 * dt_ps;
        // k_B T / V in bar
        let kt_over_v = BOLTZMANN_KJMOL_K * c.temperature_k / volume_nm3 * BAR_PER_KJMOL_NM3;
        let rate = c.compressibility_per_bar / c.tau_ps;
        let deterministic = -rate * (c.pressure_bar - pressure_bar - kt_over_v) * interval;
        let noise = (2.0 * kt_over_v * rate * interval).sqrt() * standard_normal(&mut self.rng);
        deterministic + noise
    }

    /// P₀ in kJ mol⁻¹ nm⁻³, for the P₀V term of the conserved quantity.
    pub fn pressure_kjmol_nm3(&self) -> f64 {
        self.config.pressure_bar / BAR_PER_KJMOL_NM3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{STREAM_BAROSTAT, stream};

    fn barostat(every: u64) -> CRescale {
        CRescale::new(
            CRescaleConfig {
                pressure_bar: 1.0,
                tau_ps: 1.0,
                compressibility_per_bar: 4.5e-5,
                every,
                temperature_k: 300.0,
            },
            stream(1, STREAM_BAROSTAT),
        )
        .unwrap()
    }

    #[test]
    fn out_of_range_settings_are_refused() {
        let good = *barostat(10).config();
        for bad in [
            CRescaleConfig {
                tau_ps: 0.0,
                ..good
            },
            CRescaleConfig {
                compressibility_per_bar: -1.0,
                ..good
            },
            CRescaleConfig {
                temperature_k: 0.0,
                ..good
            },
            CRescaleConfig { every: 0, ..good },
            CRescaleConfig {
                pressure_bar: f64::NAN,
                ..good
            },
        ] {
            assert!(CRescale::new(bad, stream(1, STREAM_BAROSTAT)).is_none());
        }
    }

    /// The mean step follows the deterministic part: an over-pressure expands the box, by
    /// β_T/τ_p · ΔP · Δt (here 4.5e-5 · 1000 bar · 0.01 ps = 4.5e-4).
    #[test]
    fn the_drift_follows_the_pressure_difference() {
        let mut b = barostat(10);
        let volume = 1.0e6; // so large that k_B T/V and the noise are negligible
        let mean: f64 = (0..2000)
            .map(|_| b.d_log_volume(1001.0, volume, 0.001))
            .sum::<f64>()
            / 2000.0;
        assert!((mean - 4.5e-4).abs() < 1e-6, "{mean}");
    }
}
