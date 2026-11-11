//! The force-provider contract of the UnitcellMD engine.
//!
//! Planned contents (PLAN.md §2.4, task E1.2): the `ForceProvider` trait,
//! `ForceResult`, `Capabilities`, `EvalRequest`, `ForceError`, the `Guarded`
//! result checker, the `Composite` provider, and the provider test kit
//! (finite-difference force and strain-virial checks, feature `testkit`).
//!
//! Depends only on `ucmd-core`. Integrators see forces exclusively through this crate.
