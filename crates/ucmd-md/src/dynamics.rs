//! The run loop: velocity Verlet as a composition of operators (PLAN.md §2.5).
//!
//! One step is O(½) B(½) A(1) F B(½) O(½), where B = kick, A = drift, F = force evaluation and O
//! = the optional thermostat. Without a thermostat this is exactly the NeNePo integrator
//! (`VV_8_12.py:72-76`) written in kick-drift-kick form.
//!
//! The step writes into spare buffers and swaps only after a successful force evaluation, so
//! after a provider error the state of the last completed step is intact (emergency checkpoint).
//!
//! An optional barostat (C-rescale, E2.7) acts every n-th step between the drift and the force
//! evaluation: it scales the drifted positions, the half-kicked velocities and the cell, so the
//! forces always belong to the scaled configuration. Its pressure is that of the state at the
//! start of the step; the run loop requests the virial for those states.

use crate::barostat::CRescale;
use crate::constraints::{ConstraintError, Constraints};
use crate::kinetics::{
    degrees_of_freedom, kinetic_energy, kinetic_tensor, pressure_tensor_bar, remove_com_velocity,
    temperature,
};
use crate::thermostat::Thermostat;
use ucmd_core::{State, System, Vec3};
use ucmd_force::{
    EvalRequest, ForceError, ForceProvider, ForceResult, Guarded, Observer, Snapshot,
};

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
    /// A model constraint could not be satisfied (E2.5); the state is that of the last
    /// completed step.
    #[error("constraints failed at step {step}: {source}")]
    Constraint {
        /// The step that could not be completed.
        step: u64,
        /// Which constraint.
        source: ConstraintError,
    },
    /// An observer (writer) failed at the given step.
    #[error("output failed at step {step}: {message}")]
    Observer {
        /// Step being written.
        step: u64,
        /// What went wrong.
        message: String,
    },
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
    barostat: Option<CRescale>,
    barostat_work: f64,
    constraints: Constraints,
    /// Molecules with constraints as (first atom, atom count): the barostat scales them by their
    /// centre of mass, so that the constraints stay satisfied (E2.5 step 2).
    rigid: Vec<(usize, usize)>,
    /// Constraint virial of the current state (kJ/mol), included in `forces.virial`.
    constraint_virial: ucmd_core::Mat3,
}

/// Mass-weighted mean of `values` over atoms `first..first + n` (molecules are whole).
fn molecular_mean(values: &[Vec3], masses: &[f64], first: usize, n: usize) -> Vec3 {
    let mut sum = Vec3::ZERO;
    let mut mass = 0.0;
    for k in first..first + n {
        sum += values[k] * masses[k];
        mass += masses[k];
    }
    sum / mass
}

/// Kinetic tensor ½Σ m v⊗v conjugate to the barostat's scaling: atoms of rigid molecules count
/// through their centre-of-mass velocity only (kJ/mol).
fn scaling_kinetic_tensor(v: &[Vec3], masses: &[f64], rigid: &[(usize, usize)]) -> ucmd_core::Mat3 {
    let mut k = kinetic_tensor(v, masses);
    for &(first, n) in rigid {
        let vc = molecular_mean(v, masses, first, n);
        let mass: f64 = masses[first..first + n].iter().sum();
        for i in first..first + n {
            k -= ucmd_core::Mat3::from_cols(
                v[i] * (0.5 * masses[i] * v[i].x),
                v[i] * (0.5 * masses[i] * v[i].y),
                v[i] * (0.5 * masses[i] * v[i].z),
            );
        }
        k += ucmd_core::Mat3::from_cols(
            vc * (0.5 * mass * vc.x),
            vc * (0.5 * mass * vc.y),
            vc * (0.5 * mass * vc.z),
        );
    }
    k
}

/// Virial conjugate to the barostat's scaling (kJ/mol): the provider's atomic virial minus, for
/// rigid molecules, Σ (r_i − R_I) ⊗ F_i, which turns it into their molecular virial.
fn scaling_virial(
    provider_virial: ucmd_core::Mat3,
    x: &[Vec3],
    forces: &[Vec3],
    masses: &[f64],
    rigid: &[(usize, usize)],
) -> ucmd_core::Mat3 {
    let mut w = provider_virial;
    for &(first, n) in rigid {
        let r = molecular_mean(x, masses, first, n);
        for i in first..first + n {
            let d = x[i] - r;
            let f = forces[i];
            w -= ucmd_core::Mat3::from_cols(d * f.x, d * f.y, d * f.z);
        }
    }
    w
}

/// Largest correction of the initial positions, relative to the constraint length, that the start
/// accepts. Above it the package geometry does not fit its constraints (E2.5 design §2).
const MAX_INITIAL_CORRECTION: f64 = 0.01;

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
        let constraints = Constraints::from_system(system);
        if !constraints.is_empty() && thermostat.as_ref().is_some_and(|t| t.acts_in_the_middle()) {
            return Err(RunError::Setup(
                "constraints with Langevin BAOAB are not implemented yet (E2.5, second step)"
                    .into(),
            ));
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
            barostat: None,
            barostat_work: 0.0,
            rigid: system
                .instances
                .iter()
                .filter(|i| {
                    !system.molecule_types[i.molecule_type]
                        .constraints
                        .is_empty()
                })
                .map(|i| {
                    (
                        i.first_atom,
                        system.molecule_types[i.molecule_type].atoms.len(),
                    )
                })
                .collect(),
            constraints,
            constraint_virial: ucmd_core::Mat3::ZERO,
        };
        dynamics.project_initial_state()?;
        let request = EvalRequest {
            step: dynamics.state.step,
            time_ps: dynamics.state.time_ps,
            need_energy: true,
            need_virial: dynamics.provider.capabilities().virial,
        };
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
        if let Some(w) = dynamics.forces.virial {
            let wc = dynamics.constraint_virial_of_current_state()?;
            dynamics.forces.virial = Some(w + wc);
            dynamics.constraint_virial = wc;
        }
        Ok(dynamics)
    }

    /// Put the start state on the constraints: positions (SHAKE onto themselves), then velocities
    /// (RATTLE). A satisfied state (e.g. from a checkpoint) is left bit for bit unchanged.
    fn project_initial_state(&mut self) -> Result<(), RunError> {
        if self.constraints.is_empty() {
            return Ok(());
        }
        let step = self.state.step;
        let fail = |source| RunError::Constraint { step, source };
        let reference = self.state.positions.clone();
        let mut scratch = vec![Vec3::ZERO; reference.len()];
        let largest = self
            .constraints
            .shake(
                &mut self.state.positions,
                &mut scratch,
                &reference,
                &self.inverse_masses,
                self.state.cell.as_ref(),
                1.0,
            )
            .map_err(fail)?;
        if largest > MAX_INITIAL_CORRECTION {
            return Err(RunError::Setup(format!(
                "the start positions miss the constraints by up to {:.1} % of a constraint length; \
                 the package geometry does not fit its constraints",
                100.0 * largest
            )));
        }
        self.constraints
            .rattle(
                &self.state.positions,
                &mut self.state.velocities,
                &self.masses,
                &self.inverse_masses,
                self.state.cell.as_ref(),
                0.5 * self.config.dt_ps,
            )
            .map_err(fail)?;
        Ok(())
    }

    /// Constraint virial of the current state, estimated as in a step: the velocity stage of a
    /// half kick with the current forces (the velocities themselves are not changed).
    fn constraint_virial_of_current_state(&self) -> Result<ucmd_core::Mat3, RunError> {
        if self.constraints.is_empty() {
            return Ok(ucmd_core::Mat3::ZERO);
        }
        let mut v = self.state.velocities.clone();
        let dt_half = 0.5 * self.config.dt_ps;
        kick(&mut v, &self.forces.forces, &self.inverse_masses, dt_half);
        self.constraints
            .rattle(
                &self.state.positions,
                &mut v,
                &self.masses,
                &self.inverse_masses,
                self.state.cell.as_ref(),
                dt_half,
            )
            .map_err(|source| RunError::Constraint {
                step: self.state.step,
                source,
            })
    }

    /// Add a barostat. It needs a fully periodic cell, a provider that returns the virial, and
    /// the virial of the current state (the initial evaluation requests it when the provider can).
    pub fn with_barostat(mut self, barostat: CRescale) -> Result<Self, RunError> {
        if !self
            .state
            .cell
            .as_ref()
            .is_some_and(|c| c.periodic().iter().all(|&p| p))
        {
            return Err(RunError::Setup(
                "a barostat needs a fully periodic cell".into(),
            ));
        }
        if !self.provider.capabilities().virial || self.forces.virial.is_none() {
            return Err(RunError::Setup(
                "a barostat needs the virial, which the force provider does not return".into(),
            ));
        }

        self.barostat = Some(barostat);
        Ok(self)
    }

    /// The barostat, e.g. to save its random-generator state in a checkpoint.
    pub fn barostat(&self) -> Option<&CRescale> {
        self.barostat.as_ref()
    }

    /// Mutable access to the barostat, e.g. to restore its generator after a restart.
    pub fn barostat_mut(&mut self) -> Option<&mut CRescale> {
        self.barostat.as_mut()
    }

    /// Energy the barostat has put into K + U + P₀V so far (kJ/mol).
    pub fn barostat_work(&self) -> f64 {
        self.barostat_work
    }

    /// Restore the accumulated barostat work after a restart (kJ/mol).
    pub fn restore_barostat_work(&mut self, work: f64) {
        self.barostat_work = work;
    }

    /// Whether the barostat acts in the step that starts from `step`.
    fn barostat_acts_at(&self, step: u64) -> bool {
        self.barostat
            .as_ref()
            .is_some_and(|b| step % b.every() == 0)
    }

    /// Advance by one step with the configured energy/virial requests. On error the state is
    /// unchanged.
    pub fn step(&mut self) -> Result<(), RunError> {
        self.step_with(self.config.energy_every_step, self.config.virial_every_step)
    }

    /// Advance by one step, requesting energy and/or virial of the new positions. On error the
    /// state is unchanged.
    pub fn step_with(&mut self, need_energy: bool, need_virial: bool) -> Result<(), RunError> {
        let dt = self.config.dt_ps;
        let next_step = self.state.step + 1;
        self.spare_velocities
            .copy_from_slice(&self.state.velocities);
        let middle = self
            .thermostat
            .as_ref()
            .is_some_and(|t| t.acts_in_the_middle());
        let mut work = 0.0;
        if middle {
            // BAOAB: B(½) A(½) O(1) A(½) F B(½)
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
                0.5 * dt,
            );
            if let Some(thermostat) = self.thermostat.as_mut() {
                work += thermostat.apply(&mut self.spare_velocities, &self.masses, self.dof, dt);
            }
            for (x, v) in self.spare_positions.iter_mut().zip(&self.spare_velocities) {
                *x += 0.5 * dt * *v;
            }
        } else {
            // O(½) B(½) A(1) F B(½) O(½)
            if let Some(thermostat) = self.thermostat.as_mut() {
                work +=
                    thermostat.apply(&mut self.spare_velocities, &self.masses, self.dof, 0.5 * dt);
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
            // S: back onto the constraints (E2.5)
            if !self.constraints.is_empty() {
                self.constraints
                    .shake(
                        &mut self.spare_positions,
                        &mut self.spare_velocities,
                        &self.state.positions,
                        &self.inverse_masses,
                        self.state.cell.as_ref(),
                        dt,
                    )
                    .map_err(|source| RunError::Constraint {
                        step: next_step,
                        source,
                    })?;
            }
        }
        // Barostat: scale the drifted positions, the velocities and the cell (E2.7).
        let mut new_cell = None;
        let mut barostat_work = 0.0;
        let mut d_eps = 0.0;
        let mut trace_w_before = 0.0;
        if self.barostat_acts_at(self.state.step) {
            let (cell, work, d, tw) = self.rescale(dt)?;
            new_cell = Some(cell);
            barostat_work = work;
            d_eps = d;
            trace_w_before = tw;
        }
        // The virial is needed for the barostat's next pressure, and right after a rescaling for
        // the energy bookkeeping of that move.
        let request = EvalRequest {
            step: next_step,
            time_ps: self.state.time_ps + dt,
            need_energy,
            need_virial: (need_virial || new_cell.is_some() || self.barostat_acts_at(next_step))
                && self.provider.capabilities().virial,
        };
        let cell = new_cell.as_ref().or(self.state.cell.as_ref());
        self.provider
            .evaluate(&self.spare_positions, cell, &request, &mut self.next_forces)
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
        if !middle && let Some(thermostat) = self.thermostat.as_mut() {
            work += thermostat.apply(&mut self.spare_velocities, &self.masses, self.dof, 0.5 * dt);
        }
        if self.config.remove_com_every > 0 && next_step % self.config.remove_com_every == 0 {
            remove_com_velocity(&mut self.spare_velocities, &self.masses);
        }
        // R: velocities along the constraints removed; their constraint force belongs to the end
        // of the step and enters its virial (E2.5 design §3)
        let mut next_constraint_virial = ucmd_core::Mat3::ZERO;
        if !self.constraints.is_empty() {
            let wc = self
                .constraints
                .rattle(
                    &self.spare_positions,
                    &mut self.spare_velocities,
                    &self.masses,
                    &self.inverse_masses,
                    cell,
                    0.5 * dt,
                )
                .map_err(|source| RunError::Constraint {
                    step: next_step,
                    source,
                })?;
            if let Some(w) = self.next_forces.virial.as_mut() {
                *w += wc;
            }
            next_constraint_virial = wc;
        }
        if new_cell.is_some() {
            // ΔU of the rescaling by the trapezoid rule, −½(tr W_before + tr W_after)·dε/3, with
            // the virial before (start of the step) and after the move (the rescaled
            // configuration). Either end alone misses ½(dε/3)²·∂²U/∂e², which the stiff bonds
            // of molecules make large: in a 7036-atom electrolyte H drifted by +88 or −97
            // kJ/mol/ps on the same trajectory; the two errors cancel in the mean.
            let w = self.next_forces.virial.ok_or_else(|| {
                RunError::Setup("barostat: the provider returned no virial".into())
            })?;
            let w = scaling_virial(
                w - next_constraint_virial,
                &self.spare_positions,
                &self.next_forces.forces,
                &self.masses,
                &self.rigid,
            );
            let trace_w_after = w.x_axis.x + w.y_axis.y + w.z_axis.z;
            barostat_work += -0.5 * (trace_w_before + trace_w_after) * d_eps / 3.0;
        }
        std::mem::swap(&mut self.state.positions, &mut self.spare_positions);
        std::mem::swap(&mut self.state.velocities, &mut self.spare_velocities);
        std::mem::swap(&mut self.forces, &mut self.next_forces);
        self.constraint_virial = next_constraint_virial;
        self.thermostat_work += work;
        if let Some(cell) = new_cell {
            self.state.cell = Some(cell);
            self.barostat_work += barostat_work;
        }
        self.state.step = next_step;
        self.state.time_ps += dt;
        Ok(())
    }

    /// One C-rescale move: scale the spare positions and velocities. Returns the scaled cell,
    /// the energy the move puts into K + P₀V (kJ/mol), dε, and tr W of the state at the start
    /// of the step; the change of U is added after the force evaluation (trapezoid rule).
    fn rescale(&mut self, dt: f64) -> Result<(ucmd_core::Cell, f64, f64, f64), RunError> {
        let setup = |m: &str| RunError::Setup(m.to_string());
        let cell = self
            .state
            .cell
            .as_ref()
            .ok_or_else(|| setup("barostat without a cell"))?;
        let virial = self
            .forces
            .virial
            .ok_or_else(|| setup("barostat step without the virial of the current state"))?;
        let volume = cell.volume();
        // pressure and virial conjugate to the scaling below (molecular for rigid molecules)
        let virial = scaling_virial(
            virial - self.constraint_virial,
            &self.state.positions,
            &self.forces.forces,
            &self.masses,
            &self.rigid,
        );
        let p = pressure_tensor_bar(
            scaling_kinetic_tensor(&self.state.velocities, &self.masses, &self.rigid),
            virial,
            volume,
        );
        let pressure = (p.x_axis.x + p.y_axis.y + p.z_axis.z) / 3.0;
        let barostat = self.barostat.as_mut().ok_or_else(|| setup("no barostat"))?;
        let d_eps = barostat.d_log_volume(pressure, volume, dt);
        let p0 = barostat.pressure_kjmol_nm3();
        let mu = libm::exp(d_eps / 3.0);
        let new_cell = cell
            .scaled(mu)
            .map_err(|e| setup(&format!("barostat: the scaled cell is invalid: {e}")))?;
        let k_before = kinetic_energy(&self.spare_velocities, &self.masses);
        // rigid molecules: shifted with their centre of mass, only its velocity scaled, so their
        // constraints stay satisfied; every other atom: scaled on its own (E2.7)
        let mut done = vec![false; self.spare_positions.len()];
        for &(first, n) in &self.rigid {
            let r = molecular_mean(&self.spare_positions, &self.masses, first, n);
            let vc = molecular_mean(&self.spare_velocities, &self.masses, first, n);
            for i in first..first + n {
                self.spare_positions[i] += r * (mu - 1.0);
                self.spare_velocities[i] += vc * (1.0 / mu - 1.0);
                done[i] = true;
            }
        }
        for (i, (x, v)) in self
            .spare_positions
            .iter_mut()
            .zip(self.spare_velocities.iter_mut())
            .enumerate()
        {
            if !done[i] {
                *x *= mu;
                *v /= mu;
            }
        }
        let k_after = kinetic_energy(&self.spare_velocities, &self.masses);
        let work = (k_after - k_before) + p0 * (new_cell.volume() - volume);
        let trace_w = virial.x_axis.x + virial.y_axis.y + virial.z_axis.z;
        Ok((new_cell, work, d_eps, trace_w))
    }

    /// Run `steps` steps, calling `observe` after each completed step.
    pub fn run(&mut self, steps: u64, mut observe: impl FnMut(&Self)) -> Result<(), RunError> {
        for _ in 0..steps {
            self.step()?;
            observe(self);
        }
        Ok(())
    }

    /// Run `steps` steps and let every observer record the steps on its interval; with
    /// `observe_start`, the current state is recorded first (step 0 of a fresh run).
    ///
    /// Energy and virial are requested only for steps that an observer records.
    pub fn run_observed(
        &mut self,
        steps: u64,
        observers: &mut [Box<dyn Observer>],
        observe_start: bool,
    ) -> Result<(), RunError> {
        let record = |md: &Self, observers: &mut [Box<dyn Observer>]| -> Result<(), RunError> {
            let step = md.state.step;
            let snapshot = md.snapshot();
            for observer in observers
                .iter_mut()
                .filter(|o| step % o.every().max(1) == 0)
            {
                observer
                    .observe(&snapshot)
                    .map_err(|e| RunError::Observer { step, message: e.0 })?;
            }
            Ok(())
        };
        if observe_start {
            record(self, observers)?;
        }
        for _ in 0..steps {
            let next = self.state.step + 1;
            let (mut energy, mut virial) =
                (self.config.energy_every_step, self.config.virial_every_step);
            for observer in observers.iter().filter(|o| next % o.every().max(1) == 0) {
                let (e, w) = observer.needs();
                energy |= e;
                virial |= w;
            }
            self.step_with(energy, virial)?;
            record(self, observers)?;
        }
        Ok(())
    }

    /// The current state with forces, energies and kinetic quantities, for observers.
    pub fn snapshot(&self) -> Snapshot<'_> {
        let cell = self.state.cell.as_ref();
        let pressure_bar = match (self.forces.virial, cell) {
            (Some(w), Some(c)) => Some(pressure_tensor_bar(
                kinetic_tensor(&self.state.velocities, &self.masses),
                w,
                c.volume(),
            )),
            _ => None,
        };
        Snapshot {
            step: self.state.step,
            time_ps: self.state.time_ps,
            positions: &self.state.positions,
            velocities: &self.state.velocities,
            cell,
            forces: &self.forces,
            kinetic_energy: self.kinetic_energy(),
            temperature_k: self.temperature(),
            conserved_energy: self.conserved_energy(),
            pressure_bar,
        }
    }

    /// The thermostat, e.g. to save its random-generator state in a checkpoint.
    pub fn thermostat(&self) -> Option<&dyn Thermostat> {
        self.thermostat.as_deref()
    }

    /// Restore the accumulated thermostat work after a restart (kJ/mol).
    pub fn restore_thermostat_work(&mut self, work: f64) {
        self.thermostat_work = work;
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

    /// The conserved quantity in kJ/mol, constant for exact dynamics: K + U + the bath energy
    /// for extended-system thermostats (Nosé–Hoover chains), otherwise K + U − the work done by
    /// the thermostat. With a barostat, + P₀V − the energy the barostat put in.
    pub fn conserved_energy(&self) -> Option<f64> {
        let bath = self
            .thermostat
            .as_ref()
            .and_then(|t| t.bath_energy(self.dof));
        let barostat = match (&self.barostat, &self.state.cell) {
            (Some(b), Some(c)) => b.pressure_kjmol_nm3() * c.volume() - self.barostat_work,
            _ => 0.0,
        };
        self.potential_energy().map(|u| {
            barostat
                + match bath {
                    Some(b) => self.kinetic_energy() + u + b,
                    None => self.kinetic_energy() + u - self.thermostat_work,
                }
        })
    }

    /// The thermostat's internal variables, for checkpoints.
    pub fn thermostat_state(&self) -> Vec<f64> {
        self.thermostat
            .as_ref()
            .map(|t| t.state())
            .unwrap_or_default()
    }

    /// Restore the thermostat's internal variables after a restart.
    pub fn restore_thermostat_state(&mut self, state: &[f64]) -> Result<(), RunError> {
        match self.thermostat.as_mut() {
            Some(t) => t.restore(state).map_err(RunError::Setup),
            None if state.is_empty() => Ok(()),
            None => Err(RunError::Setup(
                "the checkpoint has thermostat state, the run no thermostat".into(),
            )),
        }
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

    /// Non-interacting particles in a periodic box: zero forces, energy and virial.
    struct IdealGas;

    impl ForceProvider for IdealGas {
        fn name(&self) -> &str {
            "ideal-gas"
        }
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                periodic: true,
                virial: true,
                energy_terms: false,
                deterministic: true,
            }
        }
        fn initialize(&mut self, _: &System) -> Result<(), ForceError> {
            Ok(())
        }
        fn evaluate(
            &mut self,
            _: &[Vec3],
            cell: Option<&Cell>,
            _: &EvalRequest,
            out: &mut ForceResult,
        ) -> Result<(), ForceError> {
            assert!(cell.is_some());
            out.energy = Some(0.0);
            out.forces.iter_mut().for_each(|f| *f = Vec3::ZERO);
            out.virial = Some(ucmd_core::Mat3::ZERO);
            Ok(())
        }
    }

    fn ideal_gas_npt(n: usize, steps: u64) -> (Dynamics<IdealGas>, Vec<f64>) {
        use crate::barostat::{CRescale, CRescaleConfig};
        use crate::random::{STREAM_BAROSTAT, STREAM_THERMOSTAT, STREAM_VELOCITIES, stream};
        use crate::thermostat::Csvr;
        use crate::velocities::maxwell_boltzmann;

        let system = particles_in(&vec![39.948; n], [true; 3]);
        // MD removes the centre-of-mass momentum; C-rescale's k_BT/V term assumes so.
        let v0 = maxwell_boltzmann(
            system.masses(),
            300.0,
            3.0 * n as f64 - 3.0,
            true,
            &mut stream(3, STREAM_VELOCITIES),
        );
        let positions = (0..n)
            .map(|i| Vec3::new(0.1 * i as f64, 0.05 * i as f64, 0.0))
            .collect();
        let cell = Cell::new(
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            [true; 3],
        )
        .unwrap();
        let state = State::new(positions, Some(v0), Some(cell));
        let thermostat = Box::new(Csvr::new(300.0, 0.1, stream(3, STREAM_THERMOSTAT)));
        let barostat = CRescale::new(
            CRescaleConfig {
                pressure_bar: 100.0,
                tau_ps: 0.1,
                compressibility_per_bar: 0.01, // ≈ 1/P₀, an ideal gas's own compressibility
                every: 1,
                temperature_k: 300.0,
            },
            stream(3, STREAM_BAROSTAT),
        )
        .unwrap();
        let config = DynamicsConfig {
            com_removed: true,
            ..DynamicsConfig::nve(0.002)
        };
        let mut md = Dynamics::new(&system, IdealGas, state, config, Some(thermostat))
            .unwrap()
            .with_barostat(barostat)
            .unwrap();
        let mut volumes = Vec::new();
        md.run(steps, |m| {
            volumes.push(m.state().cell.as_ref().unwrap().volume())
        })
        .unwrap();
        (md, volumes)
    }

    /// C-rescale samples the isothermal-isobaric ensemble. For an ideal gas of N particles the
    /// volume distribution is p(V) ∝ V^N exp(−P₀V/k_BT), so ⟨V⟩ = (N+1) k_BT/P₀ and
    /// Var V = (N+1)(k_BT/P₀)² (Bernetti & Bussi 2020, the k_BT/V term of their Eq. 5).
    #[test]
    fn c_rescale_samples_the_ideal_gas_volume_distribution() {
        use ucmd_core::units::{BAR_PER_KJMOL_NM3, BOLTZMANN_KJMOL_K};
        let n = 20;
        let (_md, volumes) = ideal_gas_npt(n, 400_000);
        let v = &volumes[20_000..];
        let kt_over_p = BOLTZMANN_KJMOL_K * 300.0 / (100.0 / BAR_PER_KJMOL_NM3);
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64;
        // block averages: 40 blocks of 9500 steps (≫ the ~50-step correlation time)
        let block = v.len() / 40;
        let means: Vec<f64> = v
            .chunks_exact(block)
            .map(|c| c.iter().sum::<f64>() / c.len() as f64)
            .collect();
        let m = means.iter().sum::<f64>() / means.len() as f64;
        let err = (means.iter().map(|x| (x - m).powi(2)).sum::<f64>()
            / (means.len() * (means.len() - 1)) as f64)
            .sqrt();
        let expected_mean = (n as f64 + 1.0) * kt_over_p;
        let expected_var = (n as f64 + 1.0) * kt_over_p * kt_over_p;
        eprintln!(
            "<V> {mean:.4} ± {err:.4} (expected {expected_mean:.4}), Var V {var:.4} (expected {expected_var:.4})"
        );
        assert!((mean - expected_mean).abs() < 4.0 * err + 0.005 * expected_mean);
        assert!((var / expected_var - 1.0).abs() < 0.1);
    }

    #[test]
    fn a_barostat_needs_a_periodic_cell_and_the_virial() {
        use crate::barostat::{CRescale, CRescaleConfig};
        use crate::random::{STREAM_BAROSTAT, stream};
        let barostat = || {
            CRescale::new(
                CRescaleConfig {
                    pressure_bar: 1.0,
                    tau_ps: 1.0,
                    compressibility_per_bar: 4.5e-5,
                    every: 10,
                    temperature_k: 300.0,
                },
                stream(1, STREAM_BAROSTAT),
            )
            .unwrap()
        };
        let system = particles(&[1.0]);
        let state = State::new(vec![Vec3::ZERO], None, None);
        let md = Dynamics::new(
            &system,
            Tether(1.0),
            state,
            DynamicsConfig::nve(0.001),
            None,
        )
        .unwrap();
        assert!(matches!(
            md.with_barostat(barostat()),
            Err(RunError::Setup(_))
        ));
    }

    pub(crate) fn particles(masses: &[f64]) -> System {
        particles_in(masses, [false; 3])
    }

    pub(crate) fn particles_in(masses: &[f64], periodicity: [bool; 3]) -> System {
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
        System::new("test", vec![mtype], &[(0, 1)], None, periodicity).unwrap()
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

    /// Argon-mass particles on a ring (a test-only interacting system): each sits in a quartic
    /// trap ¼b(x⁴ + y⁴ + z⁴) and is tied to its ring neighbours by springs ½k|r_{i+1} − r_i|².
    /// The quartic traps make it anharmonic and chaotic, so a deterministic thermostat can
    /// sample it. The trap has no continuous symmetry, so neither momentum nor angular momentum
    /// is conserved and N_f = 3N. For any confining potential the canonical ensemble has
    /// ⟨K⟩ = N_f kT/2, Var(K) = N_f (kT)²/2 and ⟨−Σ r·F⟩ = N_f kT (generalized equipartition).
    struct QuarticRing {
        b: f64,
        k: f64,
    }

    impl ForceProvider for QuarticRing {
        fn name(&self) -> &str {
            "quartic-ring"
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
            let mut energy = 0.0;
            for (f, r) in out.forces.iter_mut().zip(x) {
                let r2 = *r * *r;
                energy += 0.25 * self.b * (r2 * r2).element_sum();
                *f = -self.b * r2 * *r;
            }
            let n = x.len();
            for i in 0..n {
                let j = (i + 1) % n;
                let d = x[j] - x[i];
                energy += 0.5 * self.k * d.length_squared();
                out.forces[i] += self.k * d;
                out.forces[j] -= self.k * d;
            }
            out.energy = Some(energy);
            Ok(())
        }
    }

    /// Relative deviations of ⟨K⟩, Var(K) and ⟨−Σ r·F⟩ from their canonical values for 64
    /// argon-mass particles in a [`QuarticRing`] at 300 K, sampled every 10 steps after 50 ps,
    /// and the largest change of the conserved quantity relative to ⟨K⟩. The last element is
    /// the standard error of the Var(K) deviation from 20 blocks.
    fn quartic_ring_canonical(thermostat: Box<dyn Thermostat>, dt_ps: f64, steps: u64) -> [f64; 5] {
        use crate::random::{STREAM_VELOCITIES, stream};
        use crate::velocities::maxwell_boltzmann;
        use ucmd_core::units::BOLTZMANN_KJMOL_K;

        let (n, t) = (64, 300.0);
        let system = particles(&vec![39.948; n]);
        let dof = 3.0 * n as f64;
        let x0 = (0..n)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / n as f64;
                Vec3::new(0.1 * a.cos(), 0.1 * a.sin(), 0.0)
            })
            .collect();
        let v0 = maxwell_boltzmann(
            system.masses(),
            t,
            dof,
            false,
            &mut stream(11, STREAM_VELOCITIES),
        );
        let mut md = Dynamics::new(
            &system,
            QuarticRing { b: 2.5e4, k: 500.0 },
            State::new(x0, Some(v0), None),
            DynamicsConfig::nve(dt_ps),
            Some(thermostat),
        )
        .unwrap();
        md.run((50.0 / dt_ps) as u64, |_| {}).unwrap();
        let start = md.conserved_energy().unwrap();
        let (mut k, mut w) = (Vec::new(), Vec::new());
        let mut drift = 0.0f64;
        md.run(steps, |m| {
            if m.state().step % 10 == 0 {
                k.push(m.kinetic_energy());
                w.push(
                    -m.state()
                        .positions
                        .iter()
                        .zip(&m.forces().forces)
                        .map(|(r, f)| r.dot(*f))
                        .sum::<f64>(),
                );
                drift = drift.max((m.conserved_energy().unwrap() - start).abs());
            }
        })
        .unwrap();
        let kt = BOLTZMANN_KJMOL_K * t;
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let var_dev = |v: &[f64]| {
            let m = mean(v);
            v.iter().map(|x| (x - m) * (x - m)).sum::<f64>()
                / v.len() as f64
                / (0.5 * dof * kt * kt)
                - 1.0
        };
        let blocks: Vec<f64> = k.chunks(k.len() / 20).take(20).map(var_dev).collect();
        let b_mean = mean(&blocks);
        let b_err = (blocks
            .iter()
            .map(|x| (x - b_mean) * (x - b_mean))
            .sum::<f64>()
            / (blocks.len() * (blocks.len() - 1)) as f64)
            .sqrt();
        let k_mean = mean(&k);
        [
            k_mean / (0.5 * dof * kt) - 1.0,
            var_dev(&k),
            mean(&w) / (dof * kt) - 1.0,
            drift / k_mean,
            b_err,
        ]
    }

    /// E2.6: Nosé–Hoover chains (M = 3, τ = 0.1 ps) sample the canonical ensemble of an
    /// interacting system, and the extended energy is conserved.
    #[test]
    fn nose_hoover_chain_samples_the_canonical_ensemble() {
        use crate::thermostat::NoseHooverChain;
        let [k_rel, var_rel, w_rel, drift, var_err] = quartic_ring_canonical(
            Box::new(NoseHooverChain::new(300.0, 0.1, 3)),
            0.004,
            250_000,
        );
        eprintln!(
            "NHC: <K> {k_rel:+.4}, Var(K) {var_rel:+.4} (block error {var_err:.4}), <-r.F> {w_rel:+.4} (relative), conserved drift {drift:.2e} of <K>"
        );
        assert!(
            k_rel.abs() < 0.01 && var_rel.abs() < 0.1 && w_rel.abs() < 0.01,
            "{k_rel} {var_rel} {w_rel}"
        );
        assert!(drift < 1e-3, "drift {drift}");
    }

    /// E2.6: Langevin BAOAB (γ = 2 ps⁻¹) samples the same canonical ensemble.
    #[test]
    fn langevin_baoab_samples_the_canonical_ensemble() {
        use crate::random::{STREAM_THERMOSTAT, stream};
        use crate::thermostat::Langevin;
        let [k_rel, var_rel, w_rel, drift, var_err] = quartic_ring_canonical(
            Box::new(Langevin::new(300.0, 2.0, stream(11, STREAM_THERMOSTAT))),
            0.004,
            250_000,
        );
        eprintln!(
            "Langevin BAOAB: <K> {k_rel:+.4}, Var(K) {var_rel:+.4} (block error {var_err:.4}), <-r.F> {w_rel:+.4} (relative), K + U - heat drift {drift:.2e} of <K>"
        );
        assert!(
            k_rel.abs() < 0.01 && var_rel.abs() < 0.1 && w_rel.abs() < 0.01,
            "{k_rel} {var_rel} {w_rel}"
        );
        assert!(drift < 1e-3, "drift {drift}");
    }

    /// E2.6 validation (slow; run with `cargo test --release -p ucmd-md -- --ignored`): both
    /// thermostats over 10 ns, where the statistical errors are about three times smaller.
    #[test]
    #[ignore]
    fn thermostats_sample_the_canonical_ensemble_long() {
        use crate::random::{STREAM_THERMOSTAT, stream};
        use crate::thermostat::{Langevin, NoseHooverChain};
        let runs: [(&str, Box<dyn Thermostat>); 2] = [
            ("NHC", Box::new(NoseHooverChain::new(300.0, 0.1, 3))),
            (
                "Langevin BAOAB",
                Box::new(Langevin::new(300.0, 2.0, stream(12, STREAM_THERMOSTAT))),
            ),
        ];
        for (name, thermostat) in runs {
            let [k_rel, var_rel, w_rel, drift, var_err] =
                quartic_ring_canonical(thermostat, 0.004, 2_500_000);
            eprintln!(
                "{name} (10 ns): <K> {k_rel:+.4}, Var(K) {var_rel:+.4} (block error {var_err:.4}), <-r.F> {w_rel:+.4}, drift {drift:.2e}"
            );
            assert!(
                k_rel.abs() < 0.005 && var_rel.abs() < 0.05 && w_rel.abs() < 0.005,
                "{name}: {k_rel} {var_rel} {w_rel}"
            );
            assert!(drift < 1e-3, "{name}: drift {drift}");
        }
    }

    /// E2.6: for a harmonic oscillator BAOAB samples positions exactly at any stable step,
    /// ⟨x²⟩ = kT/k, while velocities carry the step error ⟨v²⟩ = (kT/m)(1 − (ωΔt/2)²)
    /// (Leimkuhler & Matthews 2013). Checked at ωΔt = 1, where the velocity bias is 25 %.
    #[test]
    fn baoab_samples_harmonic_positions_exactly() {
        use crate::random::{STREAM_THERMOSTAT, stream};
        use crate::thermostat::Langevin;
        use ucmd_core::units::BOLTZMANN_KJMOL_K;

        let (n, t, k, mass): (usize, f64, f64, f64) = (100, 300.0, 1000.0, 39.948);
        let omega = (k / mass).sqrt();
        let dt = 1.0 / omega;
        let system = particles(&vec![mass; n]);
        let state = State::new(vec![Vec3::ZERO; n], Some(vec![Vec3::ZERO; n]), None);
        let thermostat = Box::new(Langevin::new(t, 5.0, stream(3, STREAM_THERMOSTAT)));
        let mut md = Dynamics::new(
            &system,
            Tether(k),
            state,
            DynamicsConfig::nve(dt),
            Some(thermostat),
        )
        .unwrap();
        md.run(1000, |_| {}).unwrap();
        let (mut x2, mut v2, mut samples) = (0.0, 0.0, 0.0);
        md.run(20_000, |m| {
            x2 += m
                .state()
                .positions
                .iter()
                .map(|r| r.length_squared())
                .sum::<f64>();
            v2 += m
                .state()
                .velocities
                .iter()
                .map(|v| v.length_squared())
                .sum::<f64>();
            samples += 3.0 * n as f64;
        })
        .unwrap();
        let kt = BOLTZMANN_KJMOL_K * t;
        let x_rel = x2 / samples / (kt / k) - 1.0;
        let v_expected = kt / mass * (1.0 - 0.25 * (omega * dt).powi(2));
        let v_rel = v2 / samples / v_expected - 1.0;
        eprintln!("BAOAB at ωΔt = 1: <x²> {x_rel:+.5}, <v²> vs (kT/m)(1 − (ωΔt/2)²) {v_rel:+.5}");
        assert!(x_rel.abs() < 0.01 && v_rel.abs() < 0.01, "{x_rel} {v_rel}");
    }

    /// The chain variables survive a state()/restore() round trip, and a wrong length is
    /// refused. A chain of length 1 (plain Nosé–Hoover) runs.
    #[test]
    fn nose_hoover_chain_state_round_trip() {
        use crate::thermostat::NoseHooverChain;
        let system = particles(&[39.948; 4]);
        let x0 = vec![
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.0, 0.1, 0.0),
            Vec3::new(0.0, 0.0, 0.1),
            Vec3::new(-0.1, 0.0, 0.0),
        ];
        for m in [1, 3] {
            let mut md = Dynamics::new(
                &system,
                QuarticRing { b: 2.5e4, k: 500.0 },
                State::new(x0.clone(), Some(vec![Vec3::ZERO; 4]), None),
                DynamicsConfig::nve(0.004),
                Some(Box::new(NoseHooverChain::new(300.0, 0.1, m))),
            )
            .unwrap();
            md.run(100, |_| {}).unwrap();
            let saved = md.thermostat_state();
            assert_eq!(saved.len(), 2 * m);
            assert!(saved.iter().all(|x| x.is_finite()) && saved.iter().any(|x| *x != 0.0));
            let mut copy = Dynamics::new(
                &system,
                QuarticRing { b: 2.5e4, k: 500.0 },
                md.state().clone(),
                DynamicsConfig::nve(0.004),
                Some(Box::new(NoseHooverChain::new(300.0, 0.1, m))),
            )
            .unwrap();
            copy.restore_thermostat_state(&saved).unwrap();
            assert_eq!(
                copy.conserved_energy().map(f64::to_bits),
                md.conserved_energy().map(f64::to_bits)
            );
            md.run(50, |_| {}).unwrap();
            copy.run(50, |_| {}).unwrap();
            assert_eq!(md.state(), copy.state());
            assert!(copy.restore_thermostat_state(&saved[1..]).is_err());
        }
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

    // -- constraints (E2.5) ------------------------------------------------------------------

    /// `n` copies of a rigid molecule (masses, constraints), positions from `place`.
    fn rigid_system(
        masses: &[f64],
        constraints: &[(usize, usize, f64)],
        n: usize,
        periodicity: [bool; 3],
    ) -> System {
        let atoms = masses
            .iter()
            .enumerate()
            .map(|(i, &m)| AtomTemplate {
                name: format!("A{i}"),
                element: 0,
                mass_amu: m,
                charge_e: None,
                atom_type: None,
            })
            .collect();
        let mtype = MoleculeType {
            name: "rigid".into(),
            formal_charge_e: 0,
            atoms,
            bonds: vec![],
            angles: vec![],
            dihedrals: vec![],
            impropers: vec![],
            constraints: constraints
                .iter()
                .map(|&(i, j, d)| ucmd_core::Constraint {
                    atoms: [i, j],
                    length_nm: d,
                })
                .collect(),
        };
        System::new("rigid", vec![mtype], &[(0, n)], None, periodicity).unwrap()
    }

    /// Random unit vector from a seeded stream.
    fn unit(rng: &mut rand_chacha::ChaCha8Rng) -> Vec3 {
        use crate::random::uniform;
        loop {
            let v = Vec3::new(
                2.0 * uniform(rng) - 1.0,
                2.0 * uniform(rng) - 1.0,
                2.0 * uniform(rng) - 1.0,
            );
            let l = v.length();
            if l > 0.1 && l < 1.0 {
                return v / l;
            }
        }
    }

    /// An ideal gas of rigid dimers: for a free rigid rotor the centripetal constraint force
    /// cancels the rotational part of the atomic kinetic pressure, so at every step the atomic
    /// pressure with the constraint virial equals the pressure of the molecular centres of mass,
    /// up to the O(Δt²) error of the constraint-force estimate.
    #[test]
    fn constraint_virial_turns_atomic_into_molecular_pressure() {
        use crate::random::{STREAM_VELOCITIES, stream};
        use crate::velocities::maxwell_boltzmann;
        let (n, d, edge) = (64, 0.1, 3.0);
        let system = rigid_system(&[16.0, 16.0], &[(0, 1, d)], n, [true; 3]);
        let mut rng = stream(7, STREAM_VELOCITIES);
        let mut x = Vec::new();
        for k in 0..n {
            let c = Vec3::new(
                (k % 4) as f64 * 0.75 + 0.3,
                ((k / 4) % 4) as f64 * 0.75 + 0.3,
                (k / 16) as f64 * 0.75 + 0.3,
            );
            let u = unit(&mut rng);
            x.push(c + u * (0.5 * d));
            x.push(c - u * (0.5 * d));
        }
        let dof = 6.0 * n as f64 - n as f64 - 3.0;
        let v0 = maxwell_boltzmann(
            system.masses(),
            300.0,
            dof,
            true,
            &mut stream(8, STREAM_VELOCITIES),
        );
        let cell = Cell::new(
            [[edge, 0.0, 0.0], [0.0, edge, 0.0], [0.0, 0.0, edge]],
            [true; 3],
        )
        .unwrap();
        let state = State::new(x, Some(v0), Some(cell));
        let config = DynamicsConfig {
            com_removed: true,
            virial_every_step: true,
            ..DynamicsConfig::nve(0.002)
        };
        let mut md = Dynamics::new(&system, IdealGas, state, config, None).unwrap();
        let constraints = Constraints::from_system(&system);
        let masses = system.masses().to_vec();
        let volume = edge * edge * edge;
        let k0 = md.kinetic_energy();
        let mut worst_relative = 0.0f64;
        let mut worst_violation = 0.0f64;
        let mut sum_atomic = 0.0;
        let mut sum_com = 0.0;
        md.run(5000, |m| {
            let s = m.snapshot();
            let p = s.pressure_bar.unwrap();
            let atomic = (p.x_axis.x + p.y_axis.y + p.z_axis.z) / 3.0;
            // centre-of-mass kinetic pressure: Σ_mol M V² / (3V)
            let v = &m.state().velocities;
            let mut two_k = 0.0;
            for mol in 0..n {
                let (a, b) = (2 * mol, 2 * mol + 1);
                let mtot = masses[a] + masses[b];
                let vc = (v[a] * masses[a] + v[b] * masses[b]) / mtot;
                two_k += mtot * vc.length_squared();
            }
            let com = two_k / (3.0 * volume) * ucmd_core::units::BAR_PER_KJMOL_NM3;
            worst_relative = worst_relative.max((atomic - com).abs() / com);
            worst_violation = worst_violation
                .max(constraints.max_violation(&m.state().positions, m.state().cell.as_ref()));
            sum_atomic += atomic;
            sum_com += com;
        })
        .unwrap();
        eprintln!(
            "rigid dimers: worst |P_atomic − P_com|/P_com {worst_relative:.2e}, mean ratio {:.6}, \
             worst violation {worst_violation:.1e}, ΔK/K {:.1e}",
            sum_atomic / sum_com,
            (md.kinetic_energy() - k0) / k0
        );
        assert!(worst_violation < 1e-10);
        assert!((sum_atomic / sum_com - 1.0).abs() < 1e-3);
        assert!(worst_relative < 0.02);
        // free rotors and free flight: the kinetic energy is conserved
        assert!(((md.kinetic_energy() - k0) / k0).abs() < 1e-3);
    }

    /// Rigid water-like triangles in a harmonic field, NVE: every constraint holds after every
    /// step, the relative velocities are perpendicular to the constraints, energy is conserved.
    #[test]
    fn rigid_triangles_keep_their_shape_and_conserve_energy() {
        use crate::random::{STREAM_VELOCITIES, stream};
        use crate::velocities::maxwell_boltzmann;
        let (r_oh, r_hh) = (0.1, 0.163_299_316);
        let system = rigid_system(
            &[15.9994, 1.008, 1.008],
            &[(0, 1, r_oh), (0, 2, r_oh), (1, 2, r_hh)],
            8,
            [false; 3],
        );
        let mut rng = stream(11, STREAM_VELOCITIES);
        let mut x = Vec::new();
        for k in 0..8 {
            let o = Vec3::new(k as f64 * 0.4 - 1.4, (k % 3) as f64 * 0.3, 0.1 * k as f64);
            let a = unit(&mut rng);
            let c = unit(&mut rng);
            let b = (c - a * c.dot(a)).normalize(); // perpendicular to a
            let half = (r_hh / 2.0 / r_oh).asin();
            x.push(o);
            x.push(o + (a * half.cos() + b * half.sin()) * r_oh);
            x.push(o + (a * half.cos() - b * half.sin()) * r_oh);
        }
        let v0 = maxwell_boltzmann(
            system.masses(),
            300.0,
            48.0,
            false,
            &mut stream(12, STREAM_VELOCITIES),
        );
        let state = State::new(x, Some(v0), None);
        let mut md = Dynamics::new(
            &system,
            Tether(50.0),
            state,
            DynamicsConfig::nve(0.002),
            None,
        )
        .unwrap();
        let constraints = Constraints::from_system(&system);
        let e0 = md.conserved_energy().unwrap();
        let mut worst_e = 0.0f64;
        let mut worst_violation = 0.0f64;
        let mut worst_radial = 0.0f64;
        md.run(20_000, |m| {
            let st = m.state();
            worst_violation = worst_violation.max(constraints.max_violation(&st.positions, None));
            for mol in 0..8 {
                for (i, j) in [(0, 1), (0, 2), (1, 2)] {
                    let (i, j) = (3 * mol + i, 3 * mol + j);
                    let r = st.positions[i] - st.positions[j];
                    worst_radial = worst_radial
                        .max(r.normalize().dot(st.velocities[i] - st.velocities[j]).abs());
                }
            }
            worst_e = worst_e.max((m.conserved_energy().unwrap() - e0).abs());
        })
        .unwrap();
        eprintln!(
            "rigid triangles: violation {worst_violation:.1e}, radial velocity {worst_radial:.1e} nm/ps, \
             max |ΔE| {worst_e:.2e} of E₀ {e0:.3} kJ/mol"
        );
        assert!(worst_violation < 1e-10);
        assert!(worst_radial < 1e-9);
        assert!(worst_e / e0.abs() < 1e-3);
    }

    #[test]
    fn langevin_with_constraints_is_refused_for_now() {
        use crate::barostat::{CRescale, CRescaleConfig};
        use crate::random::{STREAM_BAROSTAT, STREAM_THERMOSTAT, stream};
        use crate::thermostat::Langevin;
        let system = rigid_system(&[1.0, 1.0], &[(0, 1, 0.1)], 1, [true; 3]);
        let cell = || {
            Some(
                Cell::new(
                    [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
                    [true; 3],
                )
                .unwrap(),
            )
        };
        let x = vec![Vec3::new(1.0, 1.0, 1.0), Vec3::new(1.1, 1.0, 1.0)];
        let state = || State::new(x.clone(), None, cell());
        let langevin = Box::new(Langevin::new(300.0, 1.0, stream(1, STREAM_THERMOSTAT)));
        assert!(matches!(
            Dynamics::new(&system, IdealGas, state(), DynamicsConfig::nve(0.001), Some(langevin)),
            Err(RunError::Setup(m)) if m.contains("Langevin")
        ));
        let md =
            Dynamics::new(&system, IdealGas, state(), DynamicsConfig::nve(0.001), None).unwrap();
        let barostat = CRescale::new(
            CRescaleConfig {
                pressure_bar: 1.0,
                tau_ps: 1.0,
                compressibility_per_bar: 4.5e-5,
                every: 10,
                temperature_k: 300.0,
            },
            stream(1, STREAM_BAROSTAT),
        )
        .unwrap();
        // since E2.5 step 2 a barostat with constraints is accepted
        assert!(md.with_barostat(barostat).is_ok());
    }

    #[test]
    fn a_start_far_off_the_constraints_is_refused() {
        let system = rigid_system(&[1.0, 1.0], &[(0, 1, 0.1)], 1, [false; 3]);
        let state = State::new(vec![Vec3::ZERO, Vec3::new(0.12, 0.0, 0.0)], None, None);
        assert!(matches!(
            Dynamics::new(&system, Tether(1.0), state, DynamicsConfig::nve(0.001), None),
            Err(RunError::Setup(m)) if m.contains("do not fit") || m.contains("does not fit")
        ));
    }

    /// A restart from the state of step 100 continues bit for bit like the uninterrupted run:
    /// the start projection leaves a satisfied state unchanged (E2.5 design §4).
    #[test]
    fn a_restart_with_constraints_is_bitwise_identical() {
        use crate::random::{STREAM_VELOCITIES, stream};
        use crate::velocities::maxwell_boltzmann;
        let system = rigid_system(&[12.0, 16.0], &[(0, 1, 0.12)], 4, [false; 3]);
        let mut rng = stream(21, STREAM_VELOCITIES);
        let mut x = Vec::new();
        for k in 0..4 {
            let c = Vec3::new(0.4 * k as f64, 0.0, 0.0);
            let u = unit(&mut rng);
            x.push(c + u * 0.06);
            x.push(c - u * 0.06);
        }
        let v0 = maxwell_boltzmann(
            system.masses(),
            300.0,
            20.0,
            false,
            &mut stream(22, STREAM_VELOCITIES),
        );
        let make = |state: State| {
            Dynamics::new(
                &system,
                Tether(20.0),
                state,
                DynamicsConfig::nve(0.002),
                None,
            )
            .unwrap()
        };
        let mut whole = make(State::new(x.clone(), Some(v0.clone()), None));
        whole.run(200, |_| {}).unwrap();
        let mut first = make(State::new(x, Some(v0), None));
        first.run(100, |_| {}).unwrap();
        let mut second = make(first.state().clone());
        second.run(100, |_| {}).unwrap();
        assert_eq!(second.state().positions, whole.state().positions);
        assert_eq!(second.state().velocities, whole.state().velocities);
    }

    /// Lennard-Jones between atoms of different dimers (minimum image, shifted at the cutoff), plus
    /// optionally a stiff harmonic bond inside each dimer (atoms 2k, 2k+1). Test-only: ucmd-md
    /// must not depend on provider crates.
    struct DimerLj {
        sigma: f64,
        epsilon: f64,
        cutoff: f64,
        bond: Option<(f64, f64)>, // (k kJ/mol/nm², r0 nm)
    }

    impl ForceProvider for DimerLj {
        fn name(&self) -> &str {
            "dimer-lj"
        }
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                periodic: true,
                virial: true,
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
            cell: Option<&Cell>,
            _: &EvalRequest,
            out: &mut ForceResult,
        ) -> Result<(), ForceError> {
            let cell = cell.expect("periodic");
            assert!(2.0 * self.cutoff < cell.heights().into_iter().fold(f64::MAX, f64::min));
            let lj = |r2: f64| {
                let s6 = (self.sigma * self.sigma / r2).powi(3);
                (
                    4.0 * self.epsilon * (s6 * s6 - s6),
                    24.0 * self.epsilon * (2.0 * s6 * s6 - s6) / r2,
                )
            };
            let shift = lj(self.cutoff * self.cutoff).0;
            let mut e = 0.0;
            let mut w = ucmd_core::Mat3::ZERO;
            out.forces.iter_mut().for_each(|f| *f = Vec3::ZERO);
            let n = x.len();
            for i in 0..n {
                for j in (i + 1)..n {
                    let r = cell.minimum_image(x[i] - x[j]);
                    let r2 = r.length_squared();
                    let (f, de) = if i / 2 == j / 2 {
                        match self.bond {
                            Some((k, r0)) => {
                                let d = r2.sqrt();
                                (r * (-k * (d - r0) / d), 0.5 * k * (d - r0).powi(2))
                            }
                            None => continue,
                        }
                    } else if r2 < self.cutoff * self.cutoff {
                        let (u, f_over_r) = lj(r2);
                        (r * f_over_r, u - shift)
                    } else {
                        continue;
                    };
                    e += de;
                    out.forces[i] += f;
                    out.forces[j] -= f;
                    w += ucmd_core::Mat3::from_cols(r * f.x, r * f.y, r * f.z);
                }
            }
            out.energy = Some(e);
            out.virial = Some(w);
            Ok(())
        }
    }

    /// Rigid dimers with LJ under NPT (C-rescale + CSVR), or the same dimers held by a stiff
    /// harmonic bond. Returns (H range kJ/mol, volume range nm³, worst constraint violation).
    fn dimer_npt(rigid: bool, dt: f64, steps: u64) -> (f64, f64, f64) {
        use crate::barostat::{CRescale, CRescaleConfig};
        use crate::random::{STREAM_BAROSTAT, STREAM_THERMOSTAT, STREAM_VELOCITIES, stream};
        use crate::thermostat::Csvr;
        use crate::velocities::maxwell_boltzmann;
        let (n, d, edge, t) = (64, 0.15, 2.4, 150.0);
        let constraints: &[(usize, usize, f64)] = if rigid { &[(0, 1, d)] } else { &[] };
        let system = rigid_system(&[20.0, 20.0], constraints, n, [true; 3]);
        let mut rng = stream(31, STREAM_VELOCITIES);
        let mut x = Vec::new();
        for k in 0..n {
            let c = Vec3::new(
                (k % 4) as f64 * 0.6 + 0.3,
                ((k / 4) % 4) as f64 * 0.6 + 0.3,
                (k / 16) as f64 * 0.6 + 0.3,
            );
            let u = unit(&mut rng);
            x.push(c + u * (0.5 * d));
            x.push(c - u * (0.5 * d));
        }
        let dof = 6.0 * n as f64 - if rigid { n as f64 } else { 0.0 } - 3.0;
        let v0 = maxwell_boltzmann(
            system.masses(),
            t,
            dof,
            true,
            &mut stream(32, STREAM_VELOCITIES),
        );
        let cell = Cell::new(
            [[edge, 0.0, 0.0], [0.0, edge, 0.0], [0.0, 0.0, edge]],
            [true; 3],
        )
        .unwrap();
        let state = State::new(x, Some(v0), Some(cell));
        let provider = DimerLj {
            sigma: 0.3,
            epsilon: 1.0,
            cutoff: 0.75,
            bond: if rigid { None } else { Some((2.0e5, d)) },
        };
        let thermostat = Box::new(Csvr::new(t, 0.5, stream(33, STREAM_THERMOSTAT)));
        let barostat = CRescale::new(
            CRescaleConfig {
                pressure_bar: 100.0,
                tau_ps: 1.0,
                compressibility_per_bar: 1e-4,
                every: 10,
                temperature_k: t,
            },
            stream(34, STREAM_BAROSTAT),
        )
        .unwrap();
        let config = DynamicsConfig {
            com_removed: true,
            virial_every_step: true,
            ..DynamicsConfig::nve(dt)
        };
        let mut md = Dynamics::new(&system, provider, state, config, Some(thermostat))
            .unwrap()
            .with_barostat(barostat)
            .unwrap();
        let checks = Constraints::from_system(&system);
        md.run(2000, |_| {}).unwrap(); // settle
        let (mut h_min, mut h_max, mut v_min, mut v_max, mut worst) =
            (f64::MAX, f64::MIN, f64::MAX, f64::MIN, 0.0f64);
        md.run(steps, |m| {
            let h = m.conserved_energy().unwrap();
            let v = m.state().cell.as_ref().unwrap().volume();
            h_min = h_min.min(h);
            h_max = h_max.max(h);
            v_min = v_min.min(v);
            v_max = v_max.max(v);
            worst = worst.max(checks.max_violation(&m.state().positions, m.state().cell.as_ref()));
        })
        .unwrap();
        (h_max - h_min, v_max - v_min, worst)
    }

    /// E2.5 step 2: under NPT the constraints hold after every step, the volume moves, and the
    /// conserved quantity holds no worse than for the same dimers with a stiff harmonic bond.
    #[test]
    fn npt_with_constraints_conserves_h_like_a_stiff_bond() {
        let (h_rigid, v_rigid, violation) = dimer_npt(true, 0.001, 20_000);
        let (h_flex, v_flex, _) = dimer_npt(false, 0.001, 20_000);
        eprintln!(
            "NPT dimers, 20 ps: rigid H range {h_rigid:.4} kJ/mol (V range {v_rigid:.3} nm³, \
             violation {violation:.1e}); stiff bond H range {h_flex:.4} (V range {v_flex:.3})"
        );
        assert!(violation < 1e-10);
        assert!(v_rigid > 0.05, "the volume must move: {v_rigid}");
        assert!(
            h_rigid <= h_flex,
            "constraints {h_rigid} vs stiff bond {h_flex} kJ/mol"
        );
    }
}
