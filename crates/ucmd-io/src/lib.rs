//! Input and output for the UnitcellMD engine.
//!
//! - [`read_package`]: system packages (spec/system-package.md), with the spec's stable error
//!   codes ([`ErrorCode`]) in the spec's order, shared with the Python validator.
//! - [`load_run_config`], [`RunConfig`], [`MachineProfile`]: `run.toml` and the machine profile
//!   (spec/run-config.md).
//! - [`npy`]: strict NPY reader and writer for the state arrays.
//!
//! Planned: H5MD writer and reader (E1.9), checkpoints (E1.10), extended-XYZ debug writer.

pub mod config;
pub mod elements;
pub mod error;
pub mod npy;
pub mod package;

pub use config::{
    FileOutput, Integrator, MachineProfile, MomentumSection, OutputSection, RunConfig,
    ThermostatSection, TrajectoryOutput, VelocitiesSection, load_run_config,
};
pub use error::{ErrorCode, SpecError};
pub use package::{Package, read_package};
