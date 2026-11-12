//! The force-provider contract of the UnitcellMD engine (spec/force-provider.md).
//!
//! - [`ForceProvider`], [`ForceResult`], [`Capabilities`], [`EvalRequest`], [`ForceError`]:
//!   the contract. Integrators see forces exclusively through it.
//! - [`Guarded`]: checks every result (length, finiteness, requested parts) before use.
//! - [`Composite`]: the sum of several providers.
//! - [`testkit`] (feature `testkit`): the conformance checks every provider must pass.
//!
//! Depends only on `ucmd-core`.

mod guarded;
mod provider;
#[cfg(any(test, feature = "testkit"))]
pub mod testkit;

pub use guarded::{Composite, ENERGY_TERMS_TOLERANCE, Guarded};
pub use provider::{
    Capabilities, EvalRequest, ForceError, ForceProvider, ForceResult, Observable, ObservableValue,
    outer,
};
