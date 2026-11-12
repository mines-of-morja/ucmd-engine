//! The force-provider contract of the UnitcellMD engine (spec/force-provider.md).
//!
//! - [`ForceProvider`], [`ForceResult`], [`Capabilities`], [`EvalRequest`], [`ForceError`]:
//!   the contract. Integrators see forces exclusively through it.
//! - [`Guarded`]: checks every result (length, finiteness, requested parts) before use.
//! - [`Composite`]: the sum of several providers.
//! - [`Observer`], [`Snapshot`]: the interface of trajectory, thermo and checkpoint writers.
//! - [`testkit`] (feature `testkit`): the conformance checks every provider must pass.
//!
//! Depends only on `ucmd-core`.

mod guarded;
pub mod observe;
mod provider;
#[cfg(any(test, feature = "testkit"))]
pub mod testkit;

pub use guarded::{Composite, ENERGY_TERMS_TOLERANCE, Guarded};
pub use observe::{Observer, ObserverError, Snapshot};
pub use provider::{
    Capabilities, EvalRequest, ForceError, ForceProvider, ForceResult, Observable, ObservableValue,
    outer,
};
