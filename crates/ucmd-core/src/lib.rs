//! Core types of the UnitcellMD engine.
//!
//! Planned contents (PLAN.md §2.7, task E1.1): physical constants and units
//! (nm, ps, amu, kJ/mol, e, K, derived from CODATA 2018 in exactly one place),
//! `Vec3`/`Mat3`, the periodic `Cell` (reduced triclinic form), the immutable
//! `System` (masses, charges, species, molecule index, topology) and the mutable
//! `State` (unwrapped positions, velocities, step, time).
//!
//! This crate depends on no other UnitcellMD crate.
