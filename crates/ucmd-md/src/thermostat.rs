//! Thermostats as velocity operators (operator O of PLAN.md §2.5).

use ucmd_core::Vec3;

/// A thermostat acting on velocities over a time interval.
///
/// Returns the kinetic energy it added to the system (kJ/mol), so the run loop can report the
/// conserved quantity K + U − Σ added.
pub trait Thermostat: Send {
    /// Short name for logs and provenance.
    fn name(&self) -> &str;

    /// Act on `velocities` for `dt_ps`; `dof` are the degrees of freedom of the kinetic energy.
    fn apply(&mut self, velocities: &mut [Vec3], masses: &[f64], dof: f64, dt_ps: f64) -> f64;
}
