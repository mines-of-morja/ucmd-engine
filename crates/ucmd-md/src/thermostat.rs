//! Thermostats as velocity operators (operator O of PLAN.md §2.5).

use crate::kinetics::kinetic_energy;
use rand::RngExt;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma, StandardNormal};
use ucmd_core::Vec3;
use ucmd_core::units::BOLTZMANN_KJMOL_K;

/// A thermostat acting on velocities over a time interval.
///
/// Returns the kinetic energy it added to the system (kJ/mol), so the run loop can report the
/// conserved quantity K + U − Σ added.
pub trait Thermostat: Send {
    /// Short name for logs and provenance.
    fn name(&self) -> &str;

    /// Act on `velocities` for `dt_ps`; `dof` are the degrees of freedom of the kinetic energy.
    fn apply(&mut self, velocities: &mut [Vec3], masses: &[f64], dof: f64, dt_ps: f64) -> f64;

    /// The generator state, for checkpoints (`None` for deterministic thermostats).
    fn rng(&self) -> Option<&ChaCha8Rng> {
        None
    }
}

/// Canonical sampling through velocity rescaling (Bussi, Donadio, Parrinello, J. Chem. Phys.
/// 126, 014101 (2007)). Exact resampling of the kinetic energy over each interval:
///
/// K' = K + (1 − c)(K̄ (R₁² + S)/N_f − K) + 2R₁ √(c(1 − c) K K̄ / N_f),
///
/// with c = exp(−Δt/τ), K̄ = N_f k_B T / 2, R₁ ~ N(0, 1) and S ~ χ²(N_f − 1); the velocities are
/// scaled by √(K'/K) (GROMACS convention: positive factor).
pub struct Csvr {
    temperature_k: f64,
    tau_ps: f64,
    rng: ChaCha8Rng,
}

impl Csvr {
    /// Target temperature in K, coupling time in ps, and the thermostat's random stream.
    pub fn new(temperature_k: f64, tau_ps: f64, rng: ChaCha8Rng) -> Self {
        Self {
            temperature_k,
            tau_ps,
            rng,
        }
    }
}

impl Thermostat for Csvr {
    fn name(&self) -> &str {
        "csvr"
    }

    fn apply(&mut self, velocities: &mut [Vec3], masses: &[f64], dof: f64, dt_ps: f64) -> f64 {
        let k = kinetic_energy(velocities, masses);
        if k <= 0.0 || dof < 1.0 {
            return 0.0;
        }
        let target = 0.5 * dof * BOLTZMANN_KJMOL_K * self.temperature_k;
        let c = (-dt_ps / self.tau_ps).exp();
        let r1: f64 = self.rng.sample(StandardNormal);
        let rest = if dof > 1.0 {
            match Gamma::new(0.5 * (dof - 1.0), 2.0) {
                Ok(chi2) => chi2.sample(&mut self.rng),
                Err(_) => 0.0,
            }
        } else {
            0.0
        };
        let new_k = k
            + (1.0 - c) * (target * (r1 * r1 + rest) / dof - k)
            + 2.0 * r1 * (c * (1.0 - c) * k * target / dof).sqrt();
        let scale = (new_k.max(0.0) / k).sqrt();
        velocities.iter_mut().for_each(|v| *v *= scale);
        new_k.max(0.0) - k
    }

    fn rng(&self) -> Option<&ChaCha8Rng> {
        Some(&self.rng)
    }
}
