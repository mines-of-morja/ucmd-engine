//! In-process classical force providers.
//!
//! - [`harmonic`]: analytic tether and pair-spring providers (E1.3).
//! - Planned: Lennard-Jones (E1.7), neighbour lists (E1.8), bonded OPLS-AA terms, Coulomb,
//!   Ewald and smooth PME (E2).
//!
//! Every provider implements the `ForceProvider` trait from `ucmd-force` and passes the provider
//! test kit.

pub mod harmonic;

pub use harmonic::{HarmonicPairSpring, HarmonicTether, Spring};
