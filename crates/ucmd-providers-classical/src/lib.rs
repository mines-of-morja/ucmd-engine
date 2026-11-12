//! In-process classical force providers.
//!
//! - [`harmonic`]: analytic tether and pair-spring providers (E1.3).
//! - [`lj`]: Lennard-Jones with truncation or potential shift and tail corrections (E1.7).
//! - [`neighbor`]: cell list + half Verlet list with skin (E1.8).
//! - Planned: bonded OPLS-AA terms, exclusions, Coulomb, Ewald and smooth PME (E2).
//!
//! Every provider implements the `ForceProvider` trait from `ucmd-force` and passes the provider
//! test kit.

pub mod harmonic;
pub mod lj;
pub mod neighbor;

pub use harmonic::{HarmonicPairSpring, HarmonicTether, Spring};
pub use lj::{CutoffMode, LennardJones, LennardJonesConfig};
pub use neighbor::NeighborList;
