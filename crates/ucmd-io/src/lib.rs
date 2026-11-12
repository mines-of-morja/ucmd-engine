//! Input and output for the UnitcellMD engine.
//!
//! - [`read_package`]: system packages (spec/system-package.md), with the spec's stable error
//!   codes ([`ErrorCode`]) in the spec's order, shared with the Python validator.
//! - [`load_run_config`], [`RunConfig`], [`MachineProfile`]: `run.toml` and the machine profile
//!   (spec/run-config.md).
//! - [`npy`]: strict NPY reader and writer for the state arrays.
//!
//! - Writers (observers): [`ThermoWriter`] (TSV), [`XyzWriter`] (extended XYZ, Å, for viewers),
//!   `H5mdWriter` (feature `h5md`, spec/trajectory-h5md.md).
//!
//! - [`Checkpoint`]: exact restart files (spec/checkpoint.md).
//!
//! Planned: H5MD reader for `ucmd rerun`.

pub mod checkpoint;
pub mod config;
pub mod elements;
pub mod error;
#[cfg(feature = "h5md")]
pub mod h5md;
pub mod npy;
pub mod package;
pub mod thermo;
pub mod xyz;

pub use checkpoint::{Checkpoint, CheckpointError, CheckpointMeta};
pub use config::{
    FileOutput, Integrator, MachineProfile, MomentumSection, OutputSection, RunConfig,
    ThermostatSection, TrajectoryOutput, VelocitiesSection, load_run_config,
};
pub use error::{ErrorCode, SpecError};
#[cfg(feature = "h5md")]
pub use h5md::{H5mdFrame, H5mdOptions, H5mdProvenance, H5mdReader, H5mdWriter};
pub use package::{Package, read_package};
pub use thermo::ThermoWriter;
pub use xyz::XyzWriter;
