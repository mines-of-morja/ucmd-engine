//! In-process classical force providers.
//!
//! - [`harmonic`]: analytic tether and pair-spring providers (E1.3).
//! - [`lj`]: Lennard-Jones with truncation or potential shift and tail corrections (E1.7).
//! - [`neighbor`]: cell list + half Verlet list with skin (E1.8).
//! - [`topology`]: exclusions and 1-4 pairs from the bond graph (E2.2).
//! - [`bonded`]: bonds, angles, dihedrals and impropers (E2.1).
//! - [`forcefield`]: the `force-field` kind, bonded and nonbonded together (E2, design §6).
//! - [`nonbonded`]: Lennard-Jones, 1-4 pairs, Coulomb direct, Ewald and PME (E2.2–E2.4).
//! - [`pme`]: smooth particle-mesh Ewald, reciprocal part (E2.4).
//!
//! Every provider implements the `ForceProvider` trait from `ucmd-force` and passes the provider
//! test kit.

pub mod bonded;
pub mod forcefield;
pub mod harmonic;
pub mod lj;
pub mod neighbor;
pub mod nonbonded;
pub mod pme;
pub mod topology;

#[cfg(test)]
mod test_util;

pub use bonded::Bonded;
pub use forcefield::ForceField;
pub use harmonic::{HarmonicPairSpring, HarmonicTether, Spring};
pub use lj::{CutoffMode, LennardJones, LennardJonesConfig};
pub use neighbor::NeighborList;
pub use nonbonded::{CoulombMethod, Nonbonded, NonbondedConfig};
pub use topology::PairTopology;
