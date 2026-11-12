//! The mutable state of a run (spec/units.md §3: positions are unwrapped).

use crate::{Cell, Vec3};

/// Positions, velocities, cell, step and time of a run, in canonical atom order.
#[derive(Debug, Clone, PartialEq)]
pub struct State {
    /// Unwrapped positions in nm; molecules are whole.
    pub positions: Vec<Vec3>,
    /// Velocities in nm/ps.
    pub velocities: Vec<Vec3>,
    /// The periodic cell, or `None` for non-periodic systems.
    pub cell: Option<Cell>,
    /// Number of completed MD steps.
    pub step: u64,
    /// Simulated time in ps.
    pub time_ps: f64,
}

impl State {
    /// A state at step 0; velocities default to zero.
    pub fn new(positions: Vec<Vec3>, velocities: Option<Vec<Vec3>>, cell: Option<Cell>) -> Self {
        let velocities = velocities.unwrap_or_else(|| vec![Vec3::ZERO; positions.len()]);
        Self {
            positions,
            velocities,
            cell,
            step: 0,
            time_ps: 0.0,
        }
    }

    /// Number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.positions.len()
    }
}
