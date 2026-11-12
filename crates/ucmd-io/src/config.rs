//! Reader for `run.toml` (spec/run-config.md) and the machine profile (§4).
//!
//! Order of checks, stopping at the first error: E_TOML, E_VERSION, E_PATH, E_SCHEMA,
//! E_INCOMPATIBLE; [`RunConfig::check_against_system`] adds the cross-checks with the package.

use crate::error::{ErrorCode, SpecError, non_negative, positive};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use ucmd_core::System;

/// Integrator selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Integrator {
    /// Velocity Verlet with optional thermostat operators.
    VelocityVerlet,
    /// Langevin BAOAB (E2.6).
    LangevinBaoab,
}

/// `[run]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunSection {
    /// Integrator.
    pub integrator: Integrator,
    /// Time step in fs.
    pub timestep_fs: f64,
    /// Number of steps.
    pub steps: u64,
    /// Master seed of every random stream.
    pub seed: u64,
}

/// Velocity generators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VelocityGenerator {
    /// Maxwell–Boltzmann at the given temperature.
    MaxwellBoltzmann,
}

/// `[velocities]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VelocitiesSection {
    /// Generator.
    pub generate: VelocityGenerator,
    /// Temperature in K.
    #[serde(rename = "temperature_K")]
    pub temperature_k: f64,
}

/// `[momentum]`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MomentumSection {
    /// Remove the COM velocity every n steps; 0 = only once at the start.
    #[serde(default)]
    pub remove_com_every: u64,
    /// Remove the angular momentum (non-periodic systems only).
    #[serde(default)]
    pub remove_angular: bool,
}

/// `[thermostat]`, selected by `kind`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum ThermostatSection {
    /// Bussi–Donadio–Parrinello stochastic velocity rescaling.
    Csvr {
        /// Target temperature in K.
        #[serde(rename = "temperature_K")]
        temperature_k: f64,
        /// Coupling time in ps.
        tau_ps: f64,
    },
    /// Nosé–Hoover chains (E2.6).
    NoseHooverChain {
        /// Target temperature in K.
        #[serde(rename = "temperature_K")]
        temperature_k: f64,
        /// Coupling time in ps.
        tau_ps: f64,
        /// Chain length (default 3).
        #[serde(default)]
        chain_length: Option<u32>,
    },
    /// Langevin friction (only with the BAOAB integrator, E2.6).
    Langevin {
        /// Target temperature in K.
        #[serde(rename = "temperature_K")]
        temperature_k: f64,
        /// Friction in ps⁻¹.
        friction_per_ps: f64,
    },
}

/// `[barostat]` (E2.7).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BarostatSection {
    /// Barostat kind; only "c-rescale".
    pub kind: String,
    /// Target pressure in bar.
    pub pressure_bar: f64,
    /// Coupling time in ps.
    pub tau_ps: f64,
    /// Isothermal compressibility in bar⁻¹.
    pub compressibility_per_bar: f64,
    /// Steps between applications.
    #[serde(default)]
    pub every: Option<u64>,
    /// "isotropic" or "anisotropic".
    #[serde(default)]
    pub coupling: Option<String>,
}

/// An output with a file and an interval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileOutput {
    /// File name relative to the output directory.
    pub file: String,
    /// Interval in steps.
    pub every: u64,
}

/// `[output.trajectory]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrajectoryOutput {
    /// File name relative to the output directory.
    pub file: String,
    /// Interval in steps.
    pub every: u64,
    /// Also write velocities.
    #[serde(default)]
    pub velocities: bool,
    /// Also write forces.
    #[serde(default)]
    pub forces: bool,
    /// "float64" (default) or "float32".
    #[serde(default)]
    pub precision: Option<String>,
}

/// `[output]`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputSection {
    /// Directory relative to run.toml (default "out").
    #[serde(default)]
    pub directory: Option<String>,
    /// Trajectory output.
    #[serde(default)]
    pub trajectory: Option<TrajectoryOutput>,
    /// Thermo table output.
    #[serde(default)]
    pub thermo: Option<FileOutput>,
    /// Checkpoint output.
    #[serde(default)]
    pub checkpoint: Option<FileOutput>,
}

/// A parsed and validated `run.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    /// Spec version of the file.
    pub spec_version: String,
    /// Package directory relative to `run.toml`.
    pub system: String,
    /// Free text.
    #[serde(default)]
    pub title: Option<String>,
    /// `[run]`.
    pub run: RunSection,
    /// `[velocities]`.
    #[serde(default)]
    pub velocities: Option<VelocitiesSection>,
    /// `[momentum]`.
    #[serde(default)]
    pub momentum: Option<MomentumSection>,
    /// `[thermostat]`.
    #[serde(default)]
    pub thermostat: Option<ThermostatSection>,
    /// `[barostat]`.
    #[serde(default)]
    pub barostat: Option<BarostatSection>,
    /// `[[providers]]`, each passed to its provider as a table.
    pub providers: Vec<toml::Table>,
    /// `[output]`.
    #[serde(default)]
    pub output: Option<OutputSection>,
    /// Directory containing the run.toml (not part of the file).
    #[serde(skip)]
    pub base_dir: PathBuf,
}

fn bad_path(value: &str) -> bool {
    value.starts_with(['/', '~', '\\']) || value.chars().take(3).any(|c| c == ':')
}

fn schema(message: impl Into<String>, location: impl Into<String>) -> SpecError {
    SpecError::new(ErrorCode::Schema, message, location)
}

/// Read and validate a `run.toml`.
pub fn load_run_config(path: &Path) -> Result<RunConfig, SpecError> {
    let name = path
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let text = std::fs::read_to_string(path)
        .map_err(|e| SpecError::new(ErrorCode::Toml, format!("{name} is missing ({e})"), &name))?;
    let table: toml::Table = text.parse().map_err(|e| {
        SpecError::new(
            ErrorCode::Toml,
            format!("{name} is not valid TOML ({e})"),
            &name,
        )
    })?;
    let version = table
        .get("spec_version")
        .and_then(toml::Value::as_str)
        .map(|s| serde_json::Value::String(s.into()));
    crate::package::check_spec_version(version.as_ref())?;
    let output = table.get("output").and_then(toml::Value::as_table);
    let mut paths: Vec<(String, Option<&str>)> = vec![(
        "system".into(),
        table.get("system").and_then(toml::Value::as_str),
    )];
    if let Some(out) = output {
        paths.push((
            "output.directory".into(),
            out.get("directory").and_then(toml::Value::as_str),
        ));
        for key in ["trajectory", "thermo", "checkpoint"] {
            let file = out
                .get(key)
                .and_then(toml::Value::as_table)
                .and_then(|t| t.get("file"))
                .and_then(toml::Value::as_str);
            paths.push((format!("output.{key}.file"), file));
        }
    }
    for (key, value) in paths {
        if let Some(v) = value
            && bad_path(v)
        {
            return Err(SpecError::new(
                ErrorCode::Path,
                format!("{key} = {v:?}: paths must be relative to the run.toml directory"),
                format!("/{}", key.replace('.', "/")),
            ));
        }
    }
    let mut config: RunConfig = toml::Value::Table(table)
        .try_into()
        .map_err(|e: toml::de::Error| schema(e.to_string(), &name))?;
    check_ranges(&config)?;
    let langevin_thermostat = matches!(config.thermostat, Some(ThermostatSection::Langevin { .. }));
    if langevin_thermostat != (config.run.integrator == Integrator::LangevinBaoab) {
        return Err(SpecError::new(
            ErrorCode::Incompatible,
            "the langevin thermostat and the langevin-baoab integrator require each other",
            "/thermostat",
        ));
    }
    config.base_dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
    Ok(config)
}

fn check_ranges(c: &RunConfig) -> Result<(), SpecError> {
    if c.system.is_empty() {
        return Err(schema("system must not be empty", "/system"));
    }
    if !positive(c.run.timestep_fs) {
        return Err(schema("timestep_fs must be > 0", "/run/timestep_fs"));
    }
    if c.velocities
        .as_ref()
        .is_some_and(|v| !non_negative(v.temperature_k))
    {
        return Err(schema("temperature_K must be >= 0", "/velocities"));
    }
    if let Some(t) = &c.thermostat {
        let ok = match t {
            ThermostatSection::Csvr {
                temperature_k,
                tau_ps,
            } => *temperature_k > 0.0 && *tau_ps > 0.0,
            ThermostatSection::NoseHooverChain {
                temperature_k,
                tau_ps,
                chain_length,
            } => {
                *temperature_k > 0.0
                    && *tau_ps > 0.0
                    && chain_length.is_none_or(|n| (1..=10).contains(&n))
            }
            ThermostatSection::Langevin {
                temperature_k,
                friction_per_ps,
            } => *temperature_k > 0.0 && *friction_per_ps > 0.0,
        };
        if !ok {
            return Err(schema("thermostat parameters out of range", "/thermostat"));
        }
    }
    if let Some(b) = &c.barostat {
        let ok = b.kind == "c-rescale"
            && b.tau_ps > 0.0
            && b.compressibility_per_bar > 0.0
            && b.every.is_none_or(|e| e >= 1)
            && b.coupling
                .as_deref()
                .is_none_or(|k| ["isotropic", "anisotropic"].contains(&k));
        if !ok {
            return Err(schema("barostat parameters out of range", "/barostat"));
        }
    }
    if c.providers.is_empty() {
        return Err(schema(
            "at least one [[providers]] table is required",
            "/providers",
        ));
    }
    for (k, p) in c.providers.iter().enumerate() {
        let kind_ok = p
            .get("kind")
            .and_then(toml::Value::as_str)
            .is_some_and(|s| {
                s.chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
                    && s.chars()
                        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
            });
        let name_ok = p.get("name").is_none_or(|n| {
            n.as_str().is_some_and(|s| {
                !s.is_empty()
                    && s.chars()
                        .all(|c| c.is_ascii_alphanumeric() || "_.-".contains(c))
            })
        });
        if !kind_ok || !name_ok {
            return Err(schema(
                "provider needs a kind like \"lennard-jones\" and an optional simple name",
                format!("/providers/{k}"),
            ));
        }
    }
    if let Some(out) = &c.output {
        let every_ok = out.trajectory.as_ref().is_none_or(|t| t.every >= 1)
            && out.thermo.as_ref().is_none_or(|t| t.every >= 1)
            && out.checkpoint.as_ref().is_none_or(|t| t.every >= 1);
        let precision_ok = out
            .trajectory
            .as_ref()
            .and_then(|t| t.precision.as_deref())
            .is_none_or(|p| ["float64", "float32"].contains(&p));
        if !every_ok || !precision_ok {
            return Err(schema(
                "output intervals must be >= 1 and precision float64|float32",
                "/output",
            ));
        }
    }
    Ok(())
}

impl RunConfig {
    /// The package directory.
    pub fn system_dir(&self) -> PathBuf {
        self.base_dir.join(&self.system)
    }

    /// The output directory.
    pub fn output_dir(&self) -> PathBuf {
        let dir = self
            .output
            .as_ref()
            .and_then(|o| o.directory.clone())
            .unwrap_or_else(|| "out".into());
        self.base_dir.join(dir)
    }

    /// Cross-checks with the package (E_INCOMPATIBLE).
    pub fn check_against_system(&self, system: &System) -> Result<(), SpecError> {
        if self.barostat.is_some() && !system.periodicity.iter().all(|&p| p) {
            return Err(SpecError::new(
                ErrorCode::Incompatible,
                "a barostat needs a fully periodic system",
                "/barostat",
            ));
        }
        if self.momentum.as_ref().is_some_and(|m| m.remove_angular) && system.is_periodic() {
            return Err(SpecError::new(
                ErrorCode::Incompatible,
                "remove_angular is not allowed for periodic systems",
                "/momentum/remove_angular",
            ));
        }
        Ok(())
    }

    /// The fully resolved configuration as TOML, for provenance.
    pub fn resolved_toml(&self) -> String {
        toml::to_string(self)
            .unwrap_or_else(|e| format!("# could not serialize the configuration: {e}\n"))
    }
}

/// Machine-specific settings (spec/run-config.md §4); never part of packages or run.toml.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MachineProfile {
    /// Spec version of the file.
    #[serde(default)]
    pub spec_version: Option<String>,
    /// Threads to use (default: all logical cores).
    #[serde(default)]
    pub threads: Option<usize>,
    /// Scratch directory for external programs.
    #[serde(default)]
    pub scratch_dir: Option<PathBuf>,
    /// Author name written to H5MD files (default "anonymous").
    #[serde(default)]
    pub author_name: Option<String>,
    /// Absolute paths of external programs by name.
    #[serde(default)]
    pub executables: BTreeMap<String, PathBuf>,
}

impl MachineProfile {
    /// Load from `$UCMD_MACHINE_PROFILE`, `$XDG_CONFIG_HOME/ucmd/machine.toml` or
    /// `~/.config/ucmd/machine.toml`; defaults if none exists.
    pub fn load() -> Result<Self, SpecError> {
        let candidates = [
            std::env::var_os("UCMD_MACHINE_PROFILE").map(PathBuf::from),
            std::env::var_os("XDG_CONFIG_HOME").map(|d| PathBuf::from(d).join("ucmd/machine.toml")),
            std::env::var_os("HOME").map(|d| PathBuf::from(d).join(".config/ucmd/machine.toml")),
        ];
        match candidates.into_iter().flatten().find(|p| p.is_file()) {
            Some(path) => Self::from_file(&path),
            None => Ok(Self::default()),
        }
    }

    /// Load from a given file.
    pub fn from_file(path: &Path) -> Result<Self, SpecError> {
        let location = path.display().to_string();
        let text = std::fs::read_to_string(path)
            .map_err(|e| SpecError::new(ErrorCode::Toml, e.to_string(), &location))?;
        let profile: Self = toml::from_str(&text)
            .map_err(|e| SpecError::new(ErrorCode::Schema, e.to_string(), &location))?;
        if let Some((name, exe)) = profile.executables.iter().find(|(_, p)| !p.is_absolute()) {
            return Err(SpecError::new(
                ErrorCode::Path,
                format!(
                    "executable {name} must be an absolute path, got {}",
                    exe.display()
                ),
                &location,
            ));
        }
        Ok(profile)
    }
}
