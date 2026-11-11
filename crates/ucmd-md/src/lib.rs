//! Dynamics: integrators, thermostats, barostats, constraints, minimizer, run loop.
//!
//! Integrators are compositions of splitting operators (drift, kick, force,
//! thermostat, constraint, barostat; PLAN.md §2.5), planned from task E1.5 on.
//!
//! Architecture rule (CLAUDE.md invariant 1): this crate depends only on
//! `ucmd-core` and `ucmd-force`, never on a provider crate.
//! `scripts/check-architecture.sh` enforces this.
