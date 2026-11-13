//! Thermostats as velocity operators (operator O of PLAN.md §2.5).

use crate::kinetics::kinetic_energy;
use crate::random::{chi_squared, standard_normal};
use rand_chacha::ChaCha8Rng;
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

    /// Energy of the thermostat's own variables, for extended-system thermostats such as
    /// Nosé–Hoover chains. The conserved quantity is then K + U + this energy instead of
    /// K + U − work.
    fn bath_energy(&self, _dof: f64) -> Option<f64> {
        None
    }

    /// Internal variables for checkpoints (empty when the RNG is the only state).
    fn state(&self) -> Vec<f64> {
        Vec::new()
    }

    /// Restore what [`Thermostat::state`] returned.
    fn restore(&mut self, state: &[f64]) -> Result<(), String> {
        if state.is_empty() {
            Ok(())
        } else {
            Err(format!("{} has no internal state to restore", self.name()))
        }
    }

    /// True for a thermostat that acts once per step between the two half drifts (Langevin
    /// BAOAB), instead of for half a step before and after velocity Verlet.
    fn acts_in_the_middle(&self) -> bool {
        false
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
        let c = libm::exp(-dt_ps / self.tau_ps);
        let r1 = standard_normal(&mut self.rng);
        // S ~ χ²(N_f − 1); with a single degree of freedom there is no S term.
        let rest = if dof > 1.0 {
            chi_squared(&mut self.rng, dof - 1.0)
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

/// Suzuki–Yoshida weights of order 4 (five stages) for the chain propagator.
fn suzuki_yoshida5() -> [f64; 5] {
    let w = 1.0 / (4.0 - libm::cbrt(4.0));
    [w, w, 1.0 - 4.0 * w, w, w]
}

/// Nosé–Hoover chains (Martyna, Tuckerman, Tobias, Klein, Mol. Phys. 87, 1117 (1996)).
///
/// The chain is propagated as the O operator of PLAN.md §2.5: for half a time step before and
/// after velocity Verlet, with fifth-order Suzuki–Yoshida weights. Thermostat masses
/// Q₁ = N_f k_B T τ² and Q_j = k_B T τ² (j ≥ 2), as LAMMPS's `fix nvt` with Tdamp = τ. The
/// conserved quantity is H = K + U + Σ ½Q_j v_j² + N_f k_B T ξ₁ + k_B T Σ_{j≥2} ξ_j.
pub struct NoseHooverChain {
    temperature_k: f64,
    tau_ps: f64,
    xi: Vec<f64>,
    v: Vec<f64>,
}

impl NoseHooverChain {
    /// Target temperature in K, coupling time τ in ps and chain length M ≥ 1.
    pub fn new(temperature_k: f64, tau_ps: f64, chain_length: usize) -> Self {
        let m = chain_length.max(1);
        Self {
            temperature_k,
            tau_ps,
            xi: vec![0.0; m],
            v: vec![0.0; m],
        }
    }

    fn masses(&self, dof: f64) -> (f64, f64) {
        let kt = BOLTZMANN_KJMOL_K * self.temperature_k;
        let tau2 = self.tau_ps * self.tau_ps;
        (dof * kt * tau2, kt * tau2)
    }
}

impl Thermostat for NoseHooverChain {
    fn name(&self) -> &str {
        "nose-hoover-chain"
    }

    fn apply(&mut self, velocities: &mut [Vec3], masses: &[f64], dof: f64, dt_ps: f64) -> f64 {
        if dof <= 0.0 {
            return 0.0; // nothing to thermostat, and Q₁ would be zero
        }
        let kt = BOLTZMANN_KJMOL_K * self.temperature_k;
        let (q1, qj) = self.masses(dof);
        let q = |j: usize| if j == 0 { q1 } else { qj };
        let m = self.v.len();
        let mut twice_k = 2.0 * kinetic_energy(velocities, masses);
        let k_start = 0.5 * twice_k;
        let (xi, v) = (&mut self.xi, &mut self.v);
        let force = |j: usize, v: &[f64], twice_k: f64| {
            if j == 0 {
                (twice_k - dof * kt) / q1
            } else {
                (q(j - 1) * v[j - 1] * v[j - 1] - kt) / q(j)
            }
        };
        let mut scale = 1.0;
        for w in suzuki_yoshida5() {
            let t = w * dt_ps;
            v[m - 1] += 0.5 * t * force(m - 1, v, twice_k);
            for j in (0..m - 1).rev() {
                let s = libm::exp(-0.25 * t * v[j + 1]);
                v[j] *= s;
                v[j] += 0.5 * t * force(j, v, twice_k);
                v[j] *= s;
            }
            let s = libm::exp(-t * v[0]);
            scale *= s;
            twice_k *= s * s;
            for (x, &vj) in xi.iter_mut().zip(v.iter()) {
                *x += t * vj;
            }
            for j in 0..m - 1 {
                let s = libm::exp(-0.25 * t * v[j + 1]);
                v[j] *= s;
                v[j] += 0.5 * t * force(j, v, twice_k);
                v[j] *= s;
            }
            v[m - 1] += 0.5 * t * force(m - 1, v, twice_k);
        }
        velocities.iter_mut().for_each(|x| *x *= scale);
        0.5 * twice_k - k_start
    }

    fn bath_energy(&self, dof: f64) -> Option<f64> {
        let kt = BOLTZMANN_KJMOL_K * self.temperature_k;
        let (q1, qj) = self.masses(dof);
        let kinetic: f64 = self
            .v
            .iter()
            .enumerate()
            .map(|(j, v)| 0.5 * if j == 0 { q1 } else { qj } * v * v)
            .sum();
        let potential = dof * kt * self.xi[0] + kt * self.xi[1..].iter().sum::<f64>();
        Some(kinetic + potential)
    }

    fn state(&self) -> Vec<f64> {
        self.xi.iter().chain(&self.v).copied().collect()
    }

    fn restore(&mut self, state: &[f64]) -> Result<(), String> {
        let m = self.v.len();
        if state.len() != 2 * m {
            return Err(format!(
                "expected {} chain variables, got {}",
                2 * m,
                state.len()
            ));
        }
        self.xi.copy_from_slice(&state[..m]);
        self.v.copy_from_slice(&state[m..]);
        Ok(())
    }
}

/// The exact Ornstein–Uhlenbeck step of Langevin dynamics, the O of BAOAB (Leimkuhler &
/// Matthews, Appl. Math. Res. Express 2013, 34): v ← c·v + √((1 − c²) k_B T / m)·R with
/// c = exp(−γΔt). It acts once per step between the half drifts. The returned work (the heat
/// exchanged) makes K + U − Σ work the monitored quantity.
pub struct Langevin {
    temperature_k: f64,
    friction_per_ps: f64,
    rng: ChaCha8Rng,
}

impl Langevin {
    /// Target temperature in K, friction γ in ps⁻¹ and the random stream.
    pub fn new(temperature_k: f64, friction_per_ps: f64, rng: ChaCha8Rng) -> Self {
        Self {
            temperature_k,
            friction_per_ps,
            rng,
        }
    }
}

impl Thermostat for Langevin {
    fn name(&self) -> &str {
        "langevin"
    }

    fn apply(&mut self, velocities: &mut [Vec3], masses: &[f64], _dof: f64, dt_ps: f64) -> f64 {
        let k_before = kinetic_energy(velocities, masses);
        let c = libm::exp(-self.friction_per_ps * dt_ps);
        let noise = (1.0 - c * c).sqrt();
        let kt = BOLTZMANN_KJMOL_K * self.temperature_k;
        for (v, &m) in velocities.iter_mut().zip(masses) {
            let sigma = noise * (kt / m).sqrt();
            let r = Vec3::new(
                standard_normal(&mut self.rng),
                standard_normal(&mut self.rng),
                standard_normal(&mut self.rng),
            );
            *v = c * *v + sigma * r;
        }
        kinetic_energy(velocities, masses) - k_before
    }

    fn rng(&self) -> Option<&ChaCha8Rng> {
        Some(&self.rng)
    }

    fn acts_in_the_middle(&self) -> bool {
        true
    }
}
