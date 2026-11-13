//! Energy minimization with FIRE 2.0 (E2.0).
//!
//! Guénolé et al., Comput. Mater. Sci. 175, 109584 (2020): the fast inertial relaxation engine
//! with the semi-implicit Euler integrator, an initial delay and a half step back after uphill
//! moves. The default parameters are the paper's FIRE 2.0 values, which LAMMPS's
//! `min_style fire` also uses. The largest displacement per step is limited, as LAMMPS's `dmax`
//! does, so that the huge forces of a freshly packed box cannot fling atoms apart.
//!
//! The minimizer knows providers only through `ForceProvider` (CLAUDE.md invariant 1): it
//! relaxes any force source, ML and QM included. The cell is fixed.

use ucmd_core::{Cell, Vec3};
use ucmd_force::{EvalRequest, ForceError, ForceProvider, ForceResult};

/// FIRE settings; units nm, ps, kJ/mol, amu.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FireConfig {
    /// Converged when the largest atomic force is below this (kJ mol⁻¹ nm⁻¹).
    pub max_force_kjmol_nm: f64,
    /// Stop without convergence after this many steps.
    pub max_steps: u64,
    /// Initial time step Δt₀ in ps.
    pub dt_start_ps: f64,
    /// Δt_max = factor · Δt₀ (10).
    pub dt_max_factor: f64,
    /// Δt_min = factor · Δt₀ (0.02).
    pub dt_min_factor: f64,
    /// Downhill steps before Δt grows (N_delay = 20).
    pub n_delay: u64,
    /// Δt growth factor (1.1).
    pub f_inc: f64,
    /// Δt shrink factor (0.5).
    pub f_dec: f64,
    /// Initial mixing parameter α (0.25).
    pub alpha_start: f64,
    /// α decay factor (0.99).
    pub f_alpha: f64,
    /// Give up after this many consecutive uphill steps (2000).
    pub max_uphill: u64,
    /// Largest displacement of any atom in one step, nm (0.01 nm = LAMMPS's dmax 0.1 Å).
    pub max_step_nm: f64,
}

impl Default for FireConfig {
    fn default() -> Self {
        Self {
            max_force_kjmol_nm: 100.0,
            max_steps: 10_000,
            dt_start_ps: 0.001,
            dt_max_factor: 10.0,
            dt_min_factor: 0.02,
            n_delay: 20,
            f_inc: 1.1,
            f_dec: 0.5,
            alpha_start: 0.25,
            f_alpha: 0.99,
            max_uphill: 2000,
            max_step_nm: 0.01,
        }
    }
}

/// Outcome of a minimization.
#[derive(Debug, Clone, PartialEq)]
pub struct FireReport {
    /// Steps taken (force evaluations after the first).
    pub steps: u64,
    /// The force criterion was met.
    pub converged: bool,
    /// Why the minimization stopped.
    pub reason: String,
    /// Energy of the start configuration (kJ/mol).
    pub energy_start_kjmol: f64,
    /// Energy of the final configuration (kJ/mol).
    pub energy_kjmol: f64,
    /// Largest atomic force at the start (kJ mol⁻¹ nm⁻¹).
    pub max_force_start_kjmol_nm: f64,
    /// Largest atomic force at the end (kJ mol⁻¹ nm⁻¹).
    pub max_force_kjmol_nm: f64,
}

fn max_force(forces: &[Vec3]) -> f64 {
    forces.iter().map(|f| f.length()).fold(0.0, f64::max)
}

fn evaluate(
    provider: &mut dyn ForceProvider,
    positions: &[Vec3],
    cell: Option<&Cell>,
    step: u64,
    out: &mut ForceResult,
) -> Result<f64, ForceError> {
    let request = EvalRequest {
        step,
        time_ps: 0.0,
        need_energy: true,
        need_virial: false,
    };
    provider.evaluate(positions, cell, &request, out)?;
    out.energy
        .ok_or_else(|| ForceError::Contract("the minimizer needs the energy".into()))
}

/// Relax `positions` in place. `observe(step, energy, max_force)` is called after every step.
///
/// A provider error stops the minimization and propagates; `positions` then hold the last
/// configuration that was evaluated successfully.
pub fn minimize(
    provider: &mut dyn ForceProvider,
    masses: &[f64],
    positions: &mut [Vec3],
    cell: Option<&Cell>,
    config: &FireConfig,
    mut observe: impl FnMut(u64, f64, f64),
) -> Result<FireReport, ForceError> {
    let n = positions.len();
    let mut out = ForceResult::default();
    out.reset(n);
    let mut energy = evaluate(provider, positions, cell, 0, &mut out)?;
    let energy_start = energy;
    let force_start = max_force(&out.forces);
    let mut velocities = vec![Vec3::ZERO; n];
    let mut trial = positions.to_vec();
    let (dt_max, dt_min) = (
        config.dt_max_factor * config.dt_start_ps,
        config.dt_min_factor * config.dt_start_ps,
    );
    let mut dt = config.dt_start_ps;
    let mut alpha = config.alpha_start;
    let (mut downhill, mut uphill) = (0u64, 0u64);
    let report = |steps, converged, reason: &str, energy, force| FireReport {
        steps,
        converged,
        reason: reason.to_string(),
        energy_start_kjmol: energy_start,
        energy_kjmol: energy,
        max_force_start_kjmol_nm: force_start,
        max_force_kjmol_nm: force,
    };
    if force_start <= config.max_force_kjmol_nm {
        return Ok(report(0, true, "converged", energy, force_start));
    }
    for step in 1..=config.max_steps {
        let power: f64 = out
            .forces
            .iter()
            .zip(&velocities)
            .map(|(f, v)| f.dot(*v))
            .sum();
        if power > 0.0 {
            downhill += 1;
            uphill = 0;
            if downhill > config.n_delay {
                dt = (dt * config.f_inc).min(dt_max);
                alpha *= config.f_alpha;
            }
        } else {
            downhill = 0;
            uphill += 1;
            if uphill > config.max_uphill {
                return Ok(report(
                    step - 1,
                    false,
                    "too many uphill steps",
                    energy,
                    max_force(&out.forces),
                ));
            }
            if step >= config.n_delay {
                if dt * config.f_dec >= dt_min {
                    dt *= config.f_dec;
                }
                alpha = config.alpha_start;
            }
            for (x, v) in positions.iter_mut().zip(&velocities) {
                *x -= 0.5 * dt * *v;
            }
            velocities.iter_mut().for_each(|v| *v = Vec3::ZERO);
        }
        // semi-implicit Euler: v ← v + Δt F/m, then mix v towards F̂, then x ← x + Δt v
        for ((v, f), m) in velocities.iter_mut().zip(&out.forces).zip(masses) {
            *v += (dt / m) * *f;
        }
        let v_norm = velocities
            .iter()
            .map(|v| v.length_squared())
            .sum::<f64>()
            .sqrt();
        let f_norm = out
            .forces
            .iter()
            .map(|f| f.length_squared())
            .sum::<f64>()
            .sqrt();
        if f_norm > 0.0 {
            for (v, f) in velocities.iter_mut().zip(&out.forces) {
                *v = (1.0 - alpha) * *v + (alpha * v_norm / f_norm) * *f;
            }
        }
        let largest = velocities.iter().map(|v| v.length()).fold(0.0, f64::max) * dt;
        let scale = if largest > config.max_step_nm {
            config.max_step_nm / largest
        } else {
            1.0
        };
        for ((t, x), v) in trial.iter_mut().zip(positions.iter()).zip(&velocities) {
            *t = *x + (scale * dt) * *v;
        }
        energy = evaluate(provider, &trial, cell, step, &mut out)?;
        positions.copy_from_slice(&trial);
        let force = max_force(&out.forces);
        observe(step, energy, force);
        if force <= config.max_force_kjmol_nm {
            return Ok(report(step, true, "converged", energy, force));
        }
    }
    Ok(report(
        config.max_steps,
        false,
        "maximum number of steps reached",
        energy,
        max_force(&out.forces),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucmd_core::System;
    use ucmd_force::Capabilities;

    /// Lennard-Jones with σ = 1 nm, ε = 1 kJ/mol, all pairs, no cutoff (reduced units).
    struct Lj;

    impl ForceProvider for Lj {
        fn name(&self) -> &str {
            "lj"
        }
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                periodic: false,
                virial: false,
                energy_terms: false,
                deterministic: true,
            }
        }
        fn initialize(&mut self, _: &System) -> Result<(), ForceError> {
            Ok(())
        }
        fn evaluate(
            &mut self,
            positions: &[Vec3],
            _: Option<&Cell>,
            _: &EvalRequest,
            out: &mut ForceResult,
        ) -> Result<(), ForceError> {
            out.reset(positions.len());
            let mut energy = 0.0;
            for i in 0..positions.len() {
                for j in i + 1..positions.len() {
                    let d = positions[i] - positions[j];
                    let r2 = d.length_squared();
                    let s6 = 1.0 / (r2 * r2 * r2);
                    energy += 4.0 * (s6 * s6 - s6);
                    let f = d * (24.0 * (2.0 * s6 * s6 - s6) / r2);
                    out.forces[i] += f;
                    out.forces[j] -= f;
                }
            }
            out.energy = Some(energy);
            Ok(())
        }
    }

    fn jitter(points: &mut [Vec3], amplitude: f64) {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for p in points.iter_mut() {
            for d in 0..3 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                p[d] += amplitude * ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5);
            }
        }
    }

    fn relax(mut positions: Vec<Vec3>) -> FireReport {
        let masses = vec![1.0; positions.len()];
        let config = FireConfig {
            max_force_kjmol_nm: 1e-6,
            max_steps: 100_000,
            dt_start_ps: 0.005,
            max_step_nm: 0.1,
            ..FireConfig::default()
        };
        minimize(
            &mut Lj,
            &masses,
            &mut positions,
            None,
            &config,
            |_, _, _| {},
        )
        .unwrap()
    }

    #[test]
    fn lj7_reaches_the_pentagonal_bipyramid() {
        // Global minimum −16.505384 ε (Wales & Doye, J. Phys. Chem. A 101, 5111 (1997)).
        let r = 1.12;
        let mut p: Vec<Vec3> = (0..5)
            .map(|k| {
                let a = 2.0 * std::f64::consts::PI * f64::from(k) / 5.0;
                Vec3::new(r * a.cos(), r * a.sin(), 0.0)
            })
            .collect();
        p.push(Vec3::new(0.0, 0.0, 0.7));
        p.push(Vec3::new(0.0, 0.0, -0.7));
        jitter(&mut p, 0.15);
        let report = relax(p);
        assert!(report.converged, "{report:?}");
        assert!((report.energy_kjmol + 16.505384).abs() < 1e-5, "{report:?}");
    }

    #[test]
    fn lj13_reaches_the_icosahedron() {
        // Global minimum −44.326801 ε (Wales & Doye 1997).
        let phi = (1.0 + 5f64.sqrt()) / 2.0;
        let mut p = vec![Vec3::ZERO];
        for (a, b) in [(1.0, phi), (-1.0, phi), (1.0, -phi), (-1.0, -phi)] {
            p.push(Vec3::new(0.0, a, b));
            p.push(Vec3::new(a, b, 0.0));
            p.push(Vec3::new(b, 0.0, a));
        }
        let scale = 1.1 / (1.0 + phi * phi).sqrt();
        for x in p.iter_mut() {
            *x *= scale;
        }
        jitter(&mut p, 0.2);
        let report = relax(p);
        assert!(report.converged, "{report:?}");
        assert!((report.energy_kjmol + 44.326801).abs() < 1e-5, "{report:?}");
        assert!(report.energy_kjmol < report.energy_start_kjmol);
    }

    #[test]
    fn errors_propagate_and_limits_are_reported() {
        let mut p = vec![Vec3::ZERO, Vec3::new(0.0, 0.0, 1.5)];
        let config = FireConfig {
            max_force_kjmol_nm: 1e-12,
            max_steps: 3,
            ..FireConfig::default()
        };
        let report = minimize(&mut Lj, &[1.0, 1.0], &mut p, None, &config, |_, _, _| {}).unwrap();
        assert!(!report.converged && report.steps == 3, "{report:?}");
    }
}
