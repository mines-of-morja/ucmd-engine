//! Observers: everything that looks at the run (trajectory, thermo table, checkpoints).
//!
//! Defined here so that `ucmd-md` (calls observers) and `ucmd-io` (implements writers) share
//! the types without depending on each other (scripts/check-architecture.sh).

use crate::provider::ForceResult;
use ucmd_core::{Cell, Mat3, Vec3};

/// What an observer sees after a completed step (or at step 0).
#[derive(Debug, Clone, Copy)]
pub struct Snapshot<'a> {
    /// Completed MD steps.
    pub step: u64,
    /// Simulated time in ps.
    pub time_ps: f64,
    /// Unwrapped positions in nm.
    pub positions: &'a [Vec3],
    /// Velocities in nm/ps.
    pub velocities: &'a [Vec3],
    /// Periodic cell, if any.
    pub cell: Option<&'a Cell>,
    /// Forces, energy, energy terms, virial and observables of the current positions.
    pub forces: &'a ForceResult,
    /// Kinetic energy in kJ/mol.
    pub kinetic_energy: f64,
    /// Kinetic temperature in K.
    pub temperature_k: f64,
    /// K + U − thermostat work in kJ/mol, if the energy is known.
    pub conserved_energy: Option<f64>,
    /// Pressure tensor in bar, if a virial and a cell are available.
    pub pressure_bar: Option<Mat3>,
}

/// Failure of an observer (I/O and the like); stops the run.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{0}")]
pub struct ObserverError(pub String);

/// Something that records the run at a fixed interval.
pub trait Observer {
    /// Interval in steps (≥ 1); step 0 is always observed.
    fn every(&self) -> u64;
    /// Whether this observer needs the energy and the virial at its steps.
    fn needs(&self) -> (bool, bool) {
        (true, false)
    }
    /// Record one snapshot.
    fn observe(&mut self, snapshot: &Snapshot<'_>) -> Result<(), ObserverError>;
    /// Flush and close.
    fn finish(&mut self) -> Result<(), ObserverError> {
        Ok(())
    }
}
