//! The run loop: velocity Verlet as a composition of operators (PLAN.md §2.5).
//!
//! One step is O(½) B(½) A(1) F B(½) O(½), where B = kick, A = drift, F = force evaluation and O
//! = the optional thermostat. Without a thermostat this is exactly the NeNePo integrator
//! (`VV_8_12.py:72-76`) written in kick-drift-kick form.
//!
//! The step writes into spare buffers and swaps only after a successful force evaluation, so
//! after a provider error the state of the last completed step is intact (emergency checkpoint).

use crate::kinetics::{degrees_of_freedom, kinetic_energy, remove_com_velocity, temperature};
use crate::thermostat::Thermostat;
use ucmd_core::{State, System, Vec3};
use ucmd_force::{EvalRequest, ForceError, ForceProvider, ForceResult, Guarded};

/// v += F/m·dt (kick, operator B).
pub fn kick(velocities: &mut [Vec3], forces: &[Vec3], inverse_masses: &[f64], dt_ps: f64) {
    for ((v, f), inv_m) in velocities.iter_mut().zip(forces).zip(inverse_masses) {
        *v += *f * (inv_m * dt_ps);
    }
}

/// x_out = x + v·dt (drift, operator A), written into a separate buffer.
pub fn drift_into(out: &mut [Vec3], positions: &[Vec3], velocities: &[Vec3], dt_ps: f64) {
    for ((o, x), v) in out.iter_mut().zip(positions).zip(velocities) {
        *o = *x + *v * dt_ps;
    }
}

/// Settings of [`Dynamics`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicsConfig {
    /// Time step in ps.
    pub dt_ps: f64,
    /// Remove the centre-of-mass velocity every n steps (0 = never during the run).
    pub remove_com_every: u64,
    /// The centre-of-mass momentum is removed (initially and/or periodically): 3 fewer degrees
    /// of freedom.
    pub com_removed: bool,
    /// Request the energy at every step (needed for conserved-energy monitoring).
    pub energy_every_step: bool,
    /// Request the virial at every step (needed for pressure output and barostats).
    pub virial_every_step: bool,
}

impl DynamicsConfig {
    /// Plain NVE dynamics with energy at every step, no momentum removal.
    pub fn nve(dt_ps: f64) -> Self {
        Self {
            dt_ps,
            remove_com_every: 0,
            com_removed: false,
            energy_every_step: true,
            virial_every_step: false,
        }
    }
}

/// Why a run stopped.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RunError {
    /// The force provider failed; the state is that of the last completed step.
    #[error("force provider failed at step {step}: {source}")]
    Provider {
        /// The step that could not be completed.
        step: u64,
        /// The provider's error.
        source: ForceError,
    },
    /// Inconsistent setup.
    #[error("invalid setup: {0}")]
    Setup(String),
}

/// A running simulation: state, current forces, provider and optional thermostat.
pub struct Dynamics<P: ForceProvider> {
    provider: Guarded<P>,
    config: DynamicsConfig,
    masses: Vec<f64>,
    inverse_masses: Vec<f64>,
    dof: f64,
    state: State,
    forces: ForceResult,
    next_forces: ForceResult,
    spare_positions: Vec<Vec3>,
    spare_velocities: Vec<Vec3>,
    thermostat: Option<Box<dyn Thermostat>>,
    thermostat_work: f64,
}

impl<P: ForceProvider> Dynamics<P> {
    /// Initialize the provider and evaluate the forces of the initial state.
    pub fn new(
        system: &System,
        provider: P,
        state: State,
        config: DynamicsConfig,
        thermostat: Option<Box<dyn Thermostat>>,
    ) -> Result<Self, RunError> {
        let n = system.n_atoms();
        if state.n_atoms() != n || state.velocities.len() != n {
            return Err(RunError::Setup(format!(
                "state has {} atoms, system {n}",
                state.n_atoms()
            )));
        }
        if !config.dt_ps.is_finite() || config.dt_ps <= 0.0 {
            return Err(RunError::Setup("time step must be positive".into()));
        }
        let mut provider = Guarded::new(provider);
        provider
            .initialize(system)
            .map_err(|source| RunError::Provider {
                step: state.step,
                source,
            })?;
        let masses = system.masses().to_vec();
        let inverse_masses = masses.iter().map(|m| 1.0 / m).collect();
        let dof = degrees_of_freedom(n, system.n_constraints(), config.com_removed, false);
        let mut dynamics = Self {
            provider,
            config,
            masses,
            inverse_masses,
            dof,
            spare_positions: state.positions.clone(),
            spare_velocities: state.velocities.clone(),
            state,
            forces: ForceResult::default(),
            next_forces: ForceResult::default(),
            thermostat,
            thermostat_work: 0.0,
        };
        let request = dynamics.request(dynamics.state.step, true);
        let state = &dynamics.state;
        dynamics
            .provider
            .evaluate(
                &state.positions,
                state.cell.as_ref(),
                &request,
                &mut dynamics.forces,
            )
            .map_err(|source| RunError::Provider {
                step: state.step,
                source,
            })?;
        Ok(dynamics)
    }

    fn request(&self, step: u64, initial: bool) -> EvalRequest {
        EvalRequest {
            step,
            time_ps: self.state.time_ps,
            need_energy: initial || self.config.energy_every_step,
            need_virial: self.config.virial_every_step && self.provider.capabilities().virial,
        }
    }

    /// Advance by one step. On error the state is unchanged.
    pub fn step(&mut self) -> Result<(), RunError> {
        let dt = self.config.dt_ps;
        let next_step = self.state.step + 1;
        self.spare_velocities
            .copy_from_slice(&self.state.velocities);
        let mut work = 0.0;
        if let Some(thermostat) = self.thermostat.as_mut() {
            work += thermostat.apply(&mut self.spare_velocities, &self.masses, self.dof, 0.5 * dt);
        }
        kick(
            &mut self.spare_velocities,
            &self.forces.forces,
            &self.inverse_masses,
            0.5 * dt,
        );
        drift_into(
            &mut self.spare_positions,
            &self.state.positions,
            &self.spare_velocities,
            dt,
        );
        let request = EvalRequest {
            time_ps: self.state.time_ps + dt,
            ..self.request(next_step, false)
        };
        self.provider
            .evaluate(
                &self.spare_positions,
                self.state.cell.as_ref(),
                &request,
                &mut self.next_forces,
            )
            .map_err(|source| RunError::Provider {
                step: next_step,
                source,
            })?;
        kick(
            &mut self.spare_velocities,
            &self.next_forces.forces,
            &self.inverse_masses,
            0.5 * dt,
        );
        if let Some(thermostat) = self.thermostat.as_mut() {
            work += thermostat.apply(&mut self.spare_velocities, &self.masses, self.dof, 0.5 * dt);
        }
        if self.config.remove_com_every > 0 && next_step % self.config.remove_com_every == 0 {
            remove_com_velocity(&mut self.spare_velocities, &self.masses);
        }
        std::mem::swap(&mut self.state.positions, &mut self.spare_positions);
        std::mem::swap(&mut self.state.velocities, &mut self.spare_velocities);
        std::mem::swap(&mut self.forces, &mut self.next_forces);
        self.thermostat_work += work;
        self.state.step = next_step;
        self.state.time_ps += dt;
        Ok(())
    }

    /// Run `steps` steps, calling `observe` after each completed step.
    pub fn run(&mut self, steps: u64, mut observe: impl FnMut(&Self)) -> Result<(), RunError> {
        for _ in 0..steps {
            self.step()?;
            observe(self);
        }
        Ok(())
    }

    /// The current state.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Mutable access, e.g. to negate velocities in a reversibility test. Forces are not
    /// recomputed, so only change velocities.
    pub fn velocities_mut(&mut self) -> &mut [Vec3] {
        &mut self.state.velocities
    }

    /// Forces and energies of the current positions.
    pub fn forces(&self) -> &ForceResult {
        &self.forces
    }

    /// Potential energy of the current positions (kJ/mol), if it was computed.
    pub fn potential_energy(&self) -> Option<f64> {
        self.forces.energy
    }

    /// Kinetic energy (kJ/mol).
    pub fn kinetic_energy(&self) -> f64 {
        kinetic_energy(&self.state.velocities, &self.masses)
    }

    /// Kinetic temperature (K) with the configured degrees of freedom.
    pub fn temperature(&self) -> f64 {
        temperature(self.kinetic_energy(), self.dof)
    }

    /// Degrees of freedom used for the temperature.
    pub fn degrees_of_freedom(&self) -> f64 {
        self.dof
    }

    /// K + U − (work done by the thermostat) in kJ/mol: constant for exact dynamics.
    pub fn conserved_energy(&self) -> Option<f64> {
        self.potential_energy()
            .map(|u| self.kinetic_energy() + u - self.thermostat_work)
    }

    /// Work done on the system by the thermostat so far (kJ/mol).
    pub fn thermostat_work(&self) -> f64 {
        self.thermostat_work
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucmd_core::{AtomTemplate, Cell, MoleculeType};
    use ucmd_force::Capabilities;

    /// Independent harmonic tethers to the origin (a test-only copy: ucmd-md must not depend on
    /// provider crates, not even in tests).
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

    pub(crate) fn particles(masses: &[f64]) -> System {
        let atoms = masses
            .iter()
            .enumerate()
            .map(|(i, &m)| AtomTemplate {
                name: format!("P{i}"),
                element: 0,
                mass_amu: m,
                charge_e: None,
                atom_type: None,
            })
            .collect();
        let mtype = MoleculeType {
            name: "particles".into(),
            formal_charge_e: 0,
            atoms,
            bonds: vec![],
            angles: vec![],
            dihedrals: vec![],
            impropers: vec![],
            constraints: vec![],
        };
        System::new("test", vec![mtype], &[(0, 1)], None, [false; 3]).unwrap()
    }

    /// max |E(t) − E₀|/E₀ for a 1D oscillator started at the minimum (q = 0, v = 1 nm/ps).
    fn max_energy_error(k: f64, mass: f64, dt: f64, steps: u64) -> f64 {
        let system = particles(&[mass]);
        let state = State::new(vec![Vec3::ZERO], Some(vec![Vec3::X]), None);
        let mut md =
            Dynamics::new(&system, Tether(k), state, DynamicsConfig::nve(dt), None).unwrap();
        let e0 = md.conserved_energy().unwrap();
        let mut worst = 0.0f64;
        md.run(steps, |m| {
            worst = worst.max((m.conserved_energy().unwrap() - e0).abs() / e0)
        })
        .unwrap();
        worst
    }

    /// E1.5 (a): the VV shadow-energy result max|ΔE|/E₀ = (ωΔt)²/4 within 5 %, and no drift.
    #[test]
    fn vv_shadow_energy_harmonic() {
        let (k, mass): (f64, f64) = (1000.0, 2.0);
        let omega = (k / mass).sqrt();
        let dt = 2.0 * std::f64::consts::PI / omega / 50.0;
        let expected = (omega * dt).powi(2) / 4.0;
        let short = max_energy_error(k, mass, dt, 500);
        assert!(
            (short / expected - 1.0).abs() < 0.05,
            "{short} vs {expected}"
        );
        let long = max_energy_error(k, mass, dt, 1_000_000);
        eprintln!(
            "shadow energy: short/expected = {:.5}, long/expected = {:.5}",
            short / expected,
            long / expected
        );
        assert!(
            long < expected * 1.05,
            "secular drift: {long} vs {expected}"
        );
    }

    /// E1.5 (b): halving Δt reduces the energy error by 4.0 ± 0.2 (second order).
    #[test]
    fn vv_second_order() {
        let (k, mass): (f64, f64) = (1000.0, 2.0);
        let dt = 2.0 * std::f64::consts::PI / (k / mass).sqrt() / 40.0;
        let ratio = max_energy_error(k, mass, dt, 800) / max_energy_error(k, mass, dt / 2.0, 1600);
        eprintln!("second order: error ratio = {ratio:.5}");
        assert!((ratio - 4.0).abs() < 0.2, "ratio {ratio}");
    }

    /// E1.5 (c): N steps, negate v, N steps → back at the start.
    #[test]
    fn vv_time_reversible() {
        let system = particles(&[1.0, 2.0, 3.0]);
        let x0 = vec![
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.0, 0.2, 0.0),
            Vec3::new(0.0, 0.0, -0.15),
        ];
        let v0 = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -0.2),
        ];
        let state = State::new(x0.clone(), Some(v0), None);
        let mut md = Dynamics::new(
            &system,
            Tether(1000.0),
            state,
            DynamicsConfig::nve(0.001),
            None,
        )
        .unwrap();
        md.run(1000, |_| {}).unwrap();
        md.velocities_mut().iter_mut().for_each(|v| *v = -*v);
        md.run(1000, |_| {}).unwrap();
        for (x, x_start) in md.state().positions.iter().zip(&x0) {
            assert!(
                (*x - *x_start).length() <= 1e-10 * x_start.length(),
                "{x:?} vs {x_start:?}"
            );
        }
    }

    /// E1.5 (d): the NeNePo integrator (VV_8_12.py:72-76), reimplemented in atomic units by
    /// validation/nenepo-vv/make_golden.py, gives the same trajectory as this engine in nm/ps/amu.
    #[test]
    fn vv_matches_nenepo_golden() {
        let golden: Vec<(u64, usize, [f64; 6])> =
            include_str!("../../../validation/nenepo-vv/golden.txt")
                .lines()
                .filter(|l| !l.starts_with('#'))
                .map(|l| {
                    let f: Vec<f64> = l.split_whitespace().map(|t| t.parse().unwrap()).collect();
                    (
                        f[0] as u64,
                        f[1] as usize,
                        [f[2], f[3], f[4], f[5], f[6], f[7]],
                    )
                })
                .collect();
        assert_eq!(golden.len(), 15);
        let system = particles(&[1.0, 2.0, 3.0]);
        let x0 = vec![
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.0, 0.2, 0.0),
            Vec3::new(0.0, 0.0, -0.15),
        ];
        let v0 = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -0.2),
        ];
        let mut md = Dynamics::new(
            &system,
            Tether(1000.0),
            State::new(x0, Some(v0), None),
            DynamicsConfig::nve(0.001),
            None,
        )
        .unwrap();
        let mut worst = 0.0f64;
        let mut compare = |md: &Dynamics<Tether>| {
            for (_, atom, row) in golden
                .iter()
                .filter(|(step, _, _)| *step == md.state().step)
            {
                let x = md.state().positions[*atom].to_array();
                let v = md.state().velocities[*atom].to_array();
                for (got, want) in x.iter().chain(&v).zip(row) {
                    worst = worst.max((got - want).abs() / want.abs().max(1e-3));
                }
            }
        };
        compare(&md);
        for _ in 0..1000 {
            md.step().unwrap();
            compare(&md);
        }
        eprintln!("NeNePo golden: largest relative deviation = {worst:.3e}");
        assert!(worst <= 1e-10, "deviation {worst}");
    }

    /// E1.6: CSVR samples the canonical ensemble. For 100 harmonic oscillators at 300 K:
    /// ⟨K⟩ = N_f k_B T/2, Var(K) = N_f (k_B T)²/2 and, by equipartition, ⟨U⟩ = N_f k_B T/2. The
    /// conserved quantity K + U − (thermostat work) stays constant up to integration error.
    #[test]
    fn csvr_samples_the_canonical_ensemble() {
        use crate::random::{STREAM_THERMOSTAT, STREAM_VELOCITIES, stream};
        use crate::thermostat::Csvr;
        use crate::velocities::maxwell_boltzmann;
        use ucmd_core::units::BOLTZMANN_KJMOL_K;

        let n = 100;
        let system = particles(&vec![39.948; n]);
        let dof = 3.0 * n as f64;
        let v0 = maxwell_boltzmann(
            system.masses(),
            300.0,
            dof,
            false,
            &mut stream(7, STREAM_VELOCITIES),
        );
        let state = State::new(vec![Vec3::ZERO; n], Some(v0), None);
        let thermostat = Box::new(Csvr::new(300.0, 0.1, stream(7, STREAM_THERMOSTAT)));
        let mut md = Dynamics::new(
            &system,
            Tether(1000.0),
            state,
            DynamicsConfig::nve(0.002),
            Some(thermostat),
        )
        .unwrap();
        md.run(20_000, |_| {}).unwrap(); // 40 ps equilibration
        let start = md.conserved_energy().unwrap();
        let (mut k_sum, mut k2_sum, mut u_sum, mut samples) = (0.0, 0.0, 0.0, 0.0);
        let mut drift = 0.0f64;
        md.run(200_000, |m| {
            if m.state().step % 20 == 0 {
                let k = m.kinetic_energy();
                k_sum += k;
                k2_sum += k * k;
                u_sum += m.potential_energy().unwrap();
                samples += 1.0;
                drift = drift.max((m.conserved_energy().unwrap() - start).abs());
            }
        })
        .unwrap();
        let kt = BOLTZMANN_KJMOL_K * 300.0;
        let (k_mean, u_mean) = (k_sum / samples, u_sum / samples);
        let k_var = k2_sum / samples - k_mean * k_mean;
        let (k_rel, var_rel, u_rel) = (
            k_mean / (0.5 * dof * kt) - 1.0,
            k_var / (0.5 * dof * kt * kt) - 1.0,
            u_mean / (0.5 * dof * kt) - 1.0,
        );
        eprintln!(
            "CSVR: <K> {k_rel:+.4}, Var(K) {var_rel:+.4}, <U> {u_rel:+.4} (relative), conserved drift {:.2e} of <K>",
            drift / k_mean
        );
        assert!(
            k_rel.abs() < 0.01 && var_rel.abs() < 0.1 && u_rel.abs() < 0.03,
            "{k_rel} {var_rel} {u_rel}"
        );
        assert!(drift / k_mean < 1e-3);
    }

    /// A provider that fails at a given step leaves the last completed state untouched.
    #[test]
    fn provider_error_keeps_the_last_good_state() {
        struct FailAt(u64, Tether);
        impl ForceProvider for FailAt {
            fn name(&self) -> &str {
                "fail"
            }
            fn capabilities(&self) -> Capabilities {
                self.1.capabilities()
            }
            fn initialize(&mut self, s: &System) -> Result<(), ForceError> {
                self.1.initialize(s)
            }
            fn evaluate(
                &mut self,
                x: &[Vec3],
                c: Option<&Cell>,
                r: &EvalRequest,
                o: &mut ForceResult,
            ) -> Result<(), ForceError> {
                if r.step == self.0 {
                    return Err(ForceError::NotConverged("test".into()));
                }
                self.1.evaluate(x, c, r, o)
            }
        }
        let system = particles(&[1.0]);
        let state = State::new(vec![Vec3::new(0.1, 0.0, 0.0)], None, None);
        let mut md = Dynamics::new(
            &system,
            FailAt(5, Tether(100.0)),
            state,
            DynamicsConfig::nve(0.001),
            None,
        )
        .unwrap();
        md.run(4, |_| {}).unwrap();
        let good = md.state().clone();
        let err = md.step().unwrap_err();
        assert!(matches!(err, RunError::Provider { step: 5, .. }));
        assert_eq!(md.state(), &good);
    }
}
