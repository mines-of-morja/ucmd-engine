//! `ucmd`: the command-line entry point and composition root of the UnitcellMD engine.
//!
//! This is the only crate that knows the concrete force providers ([`registry`]). It wires
//! package, run configuration, machine profile, providers, dynamics and writers together.

mod registry;
mod runner;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;
use ucmd_core::State;
use ucmd_core::units::{BAR_PER_KJMOL_NM3, PS_PER_FS};
use ucmd_force::{EvalRequest, ForceProvider, ForceResult, Guarded, Observer};
use ucmd_io::{
    MachineProfile, Package, RunConfig, SpecError, ThermostatSection, load_run_config, read_package,
};
use ucmd_md::kinetics::{degrees_of_freedom, remove_angular_momentum, remove_com_velocity};
use ucmd_md::random::{STREAM_THERMOSTAT, STREAM_VELOCITIES, stream};
use ucmd_md::velocities::maxwell_boltzmann;
use ucmd_md::{Csvr, Dynamics, DynamicsConfig, Thermostat};

/// Engine version and git commit, for provenance.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
const GIT_HASH: &str = env!("UCMD_GIT_HASH");

#[derive(Parser)]
#[command(name = "ucmd", version = concat!(env!("CARGO_PKG_VERSION"), " (", env!("UCMD_GIT_HASH"), ")"), about = "UnitcellMD engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate a system package (directory) or a run.toml (with its package).
    Check {
        /// Package directory or run.toml.
        path: PathBuf,
    },
    /// Run the simulation described by a run.toml.
    Run {
        /// The run configuration.
        run_toml: PathBuf,
        /// Continue from this checkpoint; outputs are cut back to its step and appended.
        #[arg(long)]
        restart: Option<PathBuf>,
        /// Restart even if the package or configuration differ from the checkpoint's.
        #[arg(long)]
        force: bool,
    },
    /// Evaluate the providers once on the initial state and print energies as JSON.
    Singlepoint {
        /// The run configuration.
        run_toml: PathBuf,
        /// Also write the forces (kJ mol⁻¹ nm⁻¹) as NPY.
        #[arg(long)]
        forces: Option<PathBuf>,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Check { path } => check(&path),
        Command::Run {
            run_toml,
            restart,
            force,
        } => run(&run_toml, restart.as_deref(), force),
        Command::Singlepoint { run_toml, forces } => singlepoint(&run_toml, forces.as_deref()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn spec(e: SpecError) -> anyhow::Error {
    anyhow!("{e}")
}

/// `ucmd check`: same output convention as `ucmd-spec validate`.
fn check(path: &Path) -> Result<()> {
    let outcome = if path.is_dir() {
        read_package(path).map(|p| p.warnings)
    } else {
        load_run_config(path).and_then(|c| {
            let package = read_package(&c.system_dir())?;
            c.check_against_system(&package.system)?;
            Ok(package.warnings)
        })
    };
    match outcome {
        Ok(warnings) => {
            println!("OK  {}", path.display());
            for (code, message) in warnings {
                println!("  {code}  {message}");
            }
            Ok(())
        }
        Err(e) => {
            println!("INVALID  {}\n  {e}", path.display());
            Err(anyhow!("{} is invalid", path.display()))
        }
    }
}

/// Configuration, package and machine profile of a run, after all cross-checks.
struct Setup {
    config: RunConfig,
    package: Package,
    machine: MachineProfile,
    system_json: String,
}

fn setup(run_toml: &Path) -> Result<Setup> {
    let config = load_run_config(run_toml).map_err(spec)?;
    let machine = MachineProfile::load().map_err(spec)?;
    let package = read_package(&config.system_dir()).map_err(spec)?;
    for (code, message) in &package.warnings {
        eprintln!("warning: {code}  {message}");
    }
    config.check_against_system(&package.system).map_err(spec)?;
    let system_json = std::fs::read_to_string(config.system_dir().join("system.json"))?;
    Ok(Setup {
        config,
        package,
        machine,
        system_json,
    })
}

fn singlepoint(run_toml: &Path, forces_out: Option<&Path>) -> Result<()> {
    let s = setup(run_toml)?;
    let state = &s.package.state;
    let mut provider = Guarded::new(registry::make_providers(&s.config, state)?);
    provider
        .initialize(&s.package.system)
        .map_err(|e| anyhow!("{e}"))?;
    let mut out = ForceResult::default();
    let request = EvalRequest {
        need_virial: provider.capabilities().virial,
        ..EvalRequest::everything()
    };
    provider
        .evaluate(&state.positions, state.cell.as_ref(), &request, &mut out)
        .map_err(|e| anyhow!("{e}"))?;
    let virial = out.virial.map(|w| {
        (0..3)
            .map(|a| (0..3).map(|b| w.col(b)[a]).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    });
    let virial_pressure = match (out.virial, state.cell.as_ref()) {
        (Some(w), Some(c)) => {
            Some((w.col(0)[0] + w.col(1)[1] + w.col(2)[2]) / (3.0 * c.volume()) * BAR_PER_KJMOL_NM3)
        }
        _ => None,
    };
    let max_force = out.forces.iter().map(|f| f.length()).fold(0.0, f64::max);
    let report = serde_json::json!({
        "system": s.package.system.name,
        "atoms": s.package.system.n_atoms(),
        "energy_kjmol": out.energy,
        "energy_terms_kjmol": out.energy_terms.iter().map(|(n, e)| (n.clone(), *e)).collect::<std::collections::BTreeMap<_, _>>(),
        "virial_kjmol": virial,
        "virial_pressure_bar": virial_pressure,
        "max_force_kjmol_nm": max_force,
        "engine": format!("ucmd {VERSION} ({GIT_HASH})"),
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    if let Some(path) = forces_out {
        ucmd_io::npy::write_vec3_array(path, &out.forces)?;
    }
    Ok(())
}

/// Initial velocities and momentum handling of spec/run-config.md (`[velocities]`, `[momentum]`).
fn initial_state(s: &Setup, dof: f64) -> State {
    let mut state = s.package.state.clone();
    let masses = s.package.system.masses();
    if let Some(v) = &s.config.velocities {
        state.velocities = maxwell_boltzmann(
            masses,
            v.temperature_k,
            dof,
            true,
            &mut stream(s.config.run.seed, STREAM_VELOCITIES),
        );
    } else {
        remove_com_velocity(&mut state.velocities, masses);
    }
    if s.config.momentum.as_ref().is_some_and(|m| m.remove_angular) {
        let positions = state.positions.clone();
        remove_angular_momentum(&positions, &mut state.velocities, masses);
    }
    state
}

fn observers(s: &Setup, restart_step: Option<u64>) -> Result<Vec<Box<dyn Observer>>> {
    let mut list: Vec<Box<dyn Observer>> = Vec::new();
    let Some(output) = &s.config.output else {
        return Ok(list);
    };
    let dir = s.config.output_dir();
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let e = |e: ucmd_force::ObserverError| anyhow!("{e}");
    if let Some(thermo) = &output.thermo {
        let path = dir.join(&thermo.file);
        list.push(Box::new(match restart_step {
            Some(step) => {
                ucmd_io::ThermoWriter::open_append(&path, thermo.every, step).map_err(e)?
            }
            None => ucmd_io::ThermoWriter::create(&path, thermo.every).map_err(e)?,
        }));
    }
    if let Some(trajectory) = &output.trajectory {
        let path = dir.join(&trajectory.file);
        let elements = s.package.system.elements();
        if path.extension().is_some_and(|x| x == "xyz") {
            list.push(Box::new(match restart_step {
                Some(step) => {
                    ucmd_io::XyzWriter::open_append(&path, trajectory.every, elements, step)
                        .map_err(e)?
                }
                None => ucmd_io::XyzWriter::create(&path, trajectory.every, elements).map_err(e)?,
            }));
        } else {
            list.push(h5md_writer(s, &path, trajectory, restart_step)?);
        }
    }
    Ok(list)
}

#[cfg(feature = "h5md")]
fn h5md_writer(
    s: &Setup,
    path: &Path,
    t: &ucmd_io::TrajectoryOutput,
    restart_step: Option<u64>,
) -> Result<Box<dyn Observer>> {
    let options = ucmd_io::H5mdOptions {
        every: t.every,
        velocities: t.velocities,
        forces: t.forces,
        float32: t.precision.as_deref() == Some("float32"),
    };
    let writer = match restart_step {
        Some(step) => ucmd_io::H5mdWriter::open_append(path, &s.package.system, options, step),
        None => {
            let provenance = ucmd_io::H5mdProvenance {
                engine_version: VERSION.into(),
                git_hash: GIT_HASH.into(),
                author: s
                    .machine
                    .author_name
                    .clone()
                    .unwrap_or_else(|| "anonymous".into()),
                system_json: s.system_json.clone(),
                run_config_toml: s.config.resolved_toml(),
                system_sha256: s.package.system.system_sha256.clone(),
            };
            ucmd_io::H5mdWriter::create(path, &s.package.system, options, &provenance)
        }
    }
    .map_err(|e| anyhow!("{e}"))?;
    Ok(Box::new(writer))
}

#[cfg(not(feature = "h5md"))]
fn h5md_writer(
    _: &Setup,
    path: &Path,
    _: &ucmd_io::TrajectoryOutput,
    _: Option<u64>,
) -> Result<Box<dyn Observer>> {
    bail!(
        "{}: this build has no H5MD support (feature \"h5md\"); use a .xyz trajectory",
        path.display()
    )
}

fn run(run_toml: &Path, restart: Option<&Path>, force: bool) -> Result<()> {
    let s = setup(run_toml)?;
    let system = &s.package.system;
    let checkpoint = restart
        .map(ucmd_io::Checkpoint::read)
        .transpose()
        .map_err(|e| anyhow!("{e}"))?;
    let fingerprint = s.config.fingerprint();
    if let Some(c) = &checkpoint {
        let same =
            c.meta.system_sha256 == system.system_sha256 && c.meta.run_config_sha256 == fingerprint;
        if !same && !force {
            bail!(
                "E_RESTART_MISMATCH  the package or run configuration differ from the checkpoint's (use --force to override)"
            );
        }
    }
    let removed_angular = s.config.momentum.as_ref().is_some_and(|m| m.remove_angular);
    let dof = degrees_of_freedom(
        system.n_atoms(),
        system.n_constraints(),
        true,
        removed_angular,
    );
    let state = match &checkpoint {
        Some(c) => c.state.clone(),
        None => initial_state(&s, dof),
    };
    let thermostat: Option<Box<dyn Thermostat>> = match &s.config.thermostat {
        None => None,
        Some(ThermostatSection::Csvr {
            temperature_k,
            tau_ps,
        }) => {
            let rng = match checkpoint
                .as_ref()
                .and_then(|c| c.meta.thermostat_rng.clone())
            {
                Some(saved) => serde_json::from_value::<rand_chacha::ChaCha8Rng>(saved)?,
                None => stream(s.config.run.seed, STREAM_THERMOSTAT),
            };
            Some(Box::new(Csvr::new(*temperature_k, *tau_ps, rng)))
        }
        Some(other) => bail!("thermostat {other:?} arrives in E2.6; available: csvr"),
    };
    if s.config.barostat.is_some() {
        bail!("barostats arrive in E2.7");
    }
    let provider = registry::make_providers(&s.config, &state)?;
    let config = DynamicsConfig {
        dt_ps: s.config.run.timestep_fs * PS_PER_FS,
        remove_com_every: s.config.momentum.as_ref().map_or(0, |m| m.remove_com_every),
        com_removed: true,
        energy_every_step: false,
        virial_every_step: false,
    };
    let mut dynamics =
        Dynamics::new(system, provider, state, config, thermostat).map_err(|e| anyhow!("{e}"))?;
    let start_step = dynamics.state().step;
    if let Some(c) = &checkpoint {
        dynamics.restore_thermostat_work(c.meta.thermostat_work_kjmol);
    }
    let mut observers = observers(&s, checkpoint.as_ref().map(|c| c.meta.step))?;
    let steps = s.config.run.steps.saturating_sub(start_step);
    let out_dir = s.config.output_dir();
    std::fs::create_dir_all(&out_dir)?;
    let plan = s
        .config
        .output
        .as_ref()
        .and_then(|o| o.checkpoint.as_ref())
        .map(|c| runner::CheckpointPlan {
            path: out_dir.join(&c.file),
            every: c.every.max(1),
        });
    let emergency = plan.as_ref().map_or_else(
        || out_dir.join("emergency.ucchk"),
        |p| p.path.with_extension("emergency.ucchk"),
    );
    let base = ucmd_io::CheckpointMeta {
        spec_version: s.config.spec_version.clone(),
        engine_version: VERSION.into(),
        engine_git_hash: GIT_HASH.into(),
        system_sha256: system.system_sha256.clone(),
        run_config_sha256: fingerprint,
        step: 0,
        time_ps: 0.0,
        thermostat_work_kjmol: 0.0,
        thermostat_rng: None,
    };
    eprintln!(
        "ucmd {VERSION} ({GIT_HASH}): {} atoms, steps {start_step}..{} of {} fs, N_dof = {}{}",
        system.n_atoms(),
        start_step + steps,
        s.config.run.timestep_fs,
        dynamics.degrees_of_freedom(),
        if checkpoint.is_some() {
            " (restart)"
        } else {
            ""
        }
    );
    let started = Instant::now();
    let outcome = runner::drive(
        &mut dynamics,
        steps,
        &mut observers,
        checkpoint.is_none(),
        plan.as_ref(),
        &emergency,
        &base,
    );
    for observer in &mut observers {
        observer.finish().map_err(|e| anyhow!("{e}"))?;
    }
    outcome?;
    let seconds = started.elapsed().as_secs_f64();
    let ns = steps as f64 * s.config.run.timestep_fs * 1e-6;
    eprintln!(
        "done: {steps} steps in {seconds:.2} s ({:.1} ns/day), final T = {:.2} K",
        if seconds > 0.0 {
            ns / seconds * 86400.0
        } else {
            0.0
        },
        dynamics.temperature()
    );
    Ok(())
}
