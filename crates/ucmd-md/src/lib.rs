//! Dynamics: integrators, thermostats, barostats, constraints, minimizer and the run loop.
//!
//! - [`dynamics`]: velocity Verlet as kick/drift/force operators with an optional thermostat
//!   (PLAN.md §2.5), double-buffered so a provider error never corrupts the state.
//! - [`kinetics`]: kinetic energy and tensor, temperature, degrees of freedom, pressure tensor,
//!   momentum removal.
//! - [`thermostat`]: the operator trait and CSVR (E1.6).
//! - [`velocities`]: Maxwell–Boltzmann initial velocities.
//! - [`random`]: per-purpose random streams derived from the master seed.
//!
//! Architecture rule (CLAUDE.md invariant 1): this crate depends only on `ucmd-core` and
//! `ucmd-force`, never on a provider crate. `scripts/check-architecture.sh` enforces this.

pub mod dynamics;
pub mod kinetics;
pub mod minimize;
pub mod random;
pub mod thermostat;
pub mod velocities;

pub use dynamics::{Dynamics, DynamicsConfig, RunError, drift_into, kick};
pub use thermostat::{Csvr, Thermostat};
