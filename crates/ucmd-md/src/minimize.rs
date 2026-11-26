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

use crate::constraints::{ConstraintError, Constraints};
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

/// FIRE with model constraints (E2.5): forces projected onto the constraint manifold, positions
/// put back onto the constraints after every move (SHAKE), the FIRE velocity projected (RATTLE).
struct Projection<'a> {
    constraints: &'a Constraints,
    inverse_masses: Vec<f64>,
}

fn constraint_error(e: ConstraintError) -> ForceError {
    ForceError::NotConverged(format!("constraints: {e}"))
}

impl Projection<'_> {
    fn forces(&self, x: &[Vec3], f: &[Vec3], cell: Option<&Cell>) -> Result<Vec<Vec3>, ForceError> {
        if self.constraints.is_empty() {
            return Ok(f.to_vec());
        }
        self.constraints
            .project_forces(x, f, &self.inverse_masses, cell)
            .map_err(constraint_error)
    }

    /// Back onto the constraints along the bond vectors of `reference`; `v` gets Δx/dt.
    fn positions(
        &self,
        x: &mut [Vec3],
        v: &mut [Vec3],
        reference: &[Vec3],
        cell: Option<&Cell>,
        dt: f64,
    ) -> Result<f64, ForceError> {
        if self.constraints.is_empty() {
            return Ok(0.0);
        }
        self.constraints
            .shake(x, v, reference, &self.inverse_masses, cell, dt)
            .map_err(constraint_error)
    }

    fn velocities(
        &self,
        x: &[Vec3],
        v: &mut [Vec3],
        masses: &[f64],
        cell: Option<&Cell>,
    ) -> Result<(), ForceError> {
        if self.constraints.is_empty() {
            return Ok(());
        }
        self.constraints
            .rattle(x, v, masses, &self.inverse_masses, cell, 1.0)
            .map(|_| ())
            .map_err(constraint_error)
    }
}

/// Relax `positions` in place. `observe(step, energy, max_force)` is called after every step.
/// With model constraints (`constraints` not empty) the minimum is the constrained one and
/// max |F| is that of the projected forces.
///
/// A provider error stops the minimization and propagates; `positions` then hold the last
/// configuration that was evaluated successfully.
pub fn minimize(
    provider: &mut dyn ForceProvider,
    masses: &[f64],
    positions: &mut [Vec3],
    cell: Option<&Cell>,
    constraints: &Constraints,
    config: &FireConfig,
    mut observe: impl FnMut(u64, f64, f64),
) -> Result<FireReport, ForceError> {
    let n = positions.len();
    let projection = Projection {
        constraints,
        inverse_masses: masses.iter().map(|m| 1.0 / m).collect(),
    };
    {
        // the start onto the constraints; far off means the geometry does not fit them
        let reference = positions.to_vec();
        let mut scratch = vec![Vec3::ZERO; n];
        let largest = projection.positions(positions, &mut scratch, &reference, cell, 1.0)?;
        if largest > 0.01 {
            return Err(ForceError::InvalidInput(format!(
                "the start positions miss the constraints by up to {:.1} % of a constraint length",
                100.0 * largest
            )));
        }
    }
    let mut out = ForceResult::default();
    out.reset(n);
    let mut energy = evaluate(provider, positions, cell, 0, &mut out)?;
    let mut forces = projection.forces(positions, &out.forces, cell)?;
    let energy_start = energy;
    let force_start = max_force(&forces);
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
        let power: f64 = forces.iter().zip(&velocities).map(|(f, v)| f.dot(*v)).sum();
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
                    max_force(&forces),
                ));
            }
            if step >= config.n_delay {
                if dt * config.f_dec >= dt_min {
                    dt *= config.f_dec;
                }
                alpha = config.alpha_start;
            }
            let before = positions.to_vec();
            for (x, v) in positions.iter_mut().zip(&velocities) {
                *x -= 0.5 * dt * *v;
            }
            velocities.iter_mut().for_each(|v| *v = Vec3::ZERO);
            let mut scratch = vec![Vec3::ZERO; n];
            projection.positions(positions, &mut scratch, &before, cell, dt)?;
        }
        // semi-implicit Euler: v ← v + Δt F/m, then mix v towards F̂, then x ← x + Δt v
        for ((v, f), m) in velocities.iter_mut().zip(&forces).zip(masses) {
            *v += (dt / m) * *f;
        }
        let v_norm = velocities
            .iter()
            .map(|v| v.length_squared())
            .sum::<f64>()
            .sqrt();
        let f_norm = forces
            .iter()
            .map(|f| f.length_squared())
            .sum::<f64>()
            .sqrt();
        if f_norm > 0.0 {
            for (v, f) in velocities.iter_mut().zip(&forces) {
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
        projection.positions(&mut trial, &mut velocities, positions, cell, scale * dt)?;
        projection.velocities(&trial, &mut velocities, masses, cell)?;
        energy = evaluate(provider, &trial, cell, step, &mut out)?;
        positions.copy_from_slice(&trial);
        forces = projection.forces(positions, &out.forces, cell)?;
        let force = max_force(&forces);
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
        max_force(&forces),
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
            &Constraints::from_pairs(vec![]),
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
        let report = minimize(
            &mut Lj,
            &[1.0, 1.0],
            &mut p,
            None,
            &Constraints::from_pairs(vec![]),
            &config,
            |_, _, _| {},
        )
        .unwrap();
        assert!(!report.converged && report.steps == 3, "{report:?}");
    }

    /// Harmonic tether to the origin (test-only).
    struct Tether(f64);

    impl ForceProvider for Tether {
        fn name(&self) -> &str {
            "tether"
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
            x: &[Vec3],
            _: Option<&Cell>,
            _: &EvalRequest,
            out: &mut ForceResult,
        ) -> Result<(), ForceError> {
            out.energy = Some(0.5 * self.0 * x.iter().map(|r| r.length_squared()).sum::<f64>());
            for (f, r) in out.forces.iter_mut().zip(x) {
                *f = -self.0 * *r;
            }
            Ok(())
        }
    }

    /// E2.5: a rigid dimer (d = 0.1 nm) in a tether k|x|²/2 has its constrained minimum with the
    /// midpoint at the origin, E = k d²/4. There the projected forces vanish while each atom
    /// still feels |F| = k d/2 against the constraint.
    #[test]
    fn fire_finds_the_constrained_minimum() {
        let (k, d) = (100.0, 0.1);
        let constraints = Constraints::from_pairs(vec![(0, 1, d)]);
        let mut x = vec![Vec3::new(0.5, 0.2, 0.1), Vec3::new(0.5, 0.2, 0.2)];
        let config = FireConfig {
            max_force_kjmol_nm: 1e-4,
            max_steps: 100_000,
            ..FireConfig::default()
        };
        let report = minimize(
            &mut Tether(k),
            &[1.0, 2.0],
            &mut x,
            None,
            &constraints,
            &config,
            |_, _, _| {},
        )
        .unwrap();
        assert!(report.converged, "{}", report.reason);
        assert!(((x[0] - x[1]).length() - d).abs() < 1e-10 * d);
        assert!((0.5 * (x[0] + x[1])).length() < 1e-5);
        assert!((report.energy_kjmol - k * d * d / 4.0).abs() < 1e-8);
        assert!((k * x[0].length() - k * d / 2.0).abs() < 1e-3); // unprojected force remains
    }
}
