//! Core types of the UnitcellMD engine.
//!
//! - [`units`]: physical constants (spec/units.md §2), the only place they are written down.
//! - [`Cell`]: the periodic cell in reduced lower-triangular form, with minimum image and wrap.
//! - [`System`]: the immutable system (molecule types, instances, per-atom properties,
//!   optional force-field parameters).
//! - [`State`]: unwrapped positions, velocities, cell, step and time.
//!
//! Engine units: nm, ps, amu (g/mol), kJ/mol, e, K. This crate depends on no other UnitcellMD
//! crate and does no I/O.

pub mod cell;
pub mod state;
pub mod system;
pub mod units;

pub use cell::{Cell, CellError};
pub use state::State;
pub use system::{
    Angle, AngleForm, AtomTemplate, AtomType, Bond, BondForm, CombinationRule, Constraint,
    ForceField, Instance, MoleculeType, System, SystemError, Torsion, TorsionForm,
};

/// Three-vector of f64 (positions in nm, velocities in nm/ps, forces in kJ mol⁻¹ nm⁻¹, …).
pub type Vec3 = glam::DVec3;
/// 3×3 matrix of f64; glam stores columns (`Mat3::from_cols`), element (row i, column j) is
/// `m.col(j)[i]`.
pub type Mat3 = glam::DMat3;
