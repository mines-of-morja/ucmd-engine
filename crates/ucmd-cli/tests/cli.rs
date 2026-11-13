//! End-to-end tests of the `ucmd` binary on spec fixtures (E1.9, E1.11). Fixtures come from
//! `$UCMD_SPEC_DIR` or `../spec` in the workspace; tests are skipped without them unless
//! `UCMD_REQUIRE_SPEC` is set.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code

use std::path::{Path, PathBuf};
use std::process::Command;

fn ucmd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ucmd"))
}

fn spec() -> Option<PathBuf> {
    let dir = std::env::var_os("UCMD_SPEC_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../spec"));
    if dir.join("fixtures").is_dir() {
        Some(dir)
    } else {
        assert!(
            std::env::var_os("UCMD_REQUIRE_SPEC").is_none(),
            "no spec fixtures at {}",
            dir.display()
        );
        None
    }
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
    }
}

/// A run directory with a copy of `package` and the given run.toml body.
fn run_dir(name: &str, spec: &Path, package: &str, run_toml: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    copy_dir(
        &spec.join("fixtures/valid").join(package),
        &dir.join("system.ucsys"),
    );
    std::fs::write(dir.join("run.toml"), run_toml).unwrap();
    dir
}

#[test]
fn binary_runs_and_reports_version() {
    let output = ucmd().arg("--version").output().unwrap();
    assert!(output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stdout)
            .starts_with(concat!("ucmd ", env!("CARGO_PKG_VERSION")))
    );
}

#[test]
fn check_reports_codes_and_exit_status() {
    let Some(spec) = spec() else { return };
    let ok = ucmd()
        .arg("check")
        .arg(spec.join("fixtures/valid/water-2"))
        .output()
        .unwrap();
    assert!(ok.status.success());
    let bad = ucmd()
        .arg("check")
        .arg(spec.join("fixtures/invalid/molecule-split"))
        .output()
        .unwrap();
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stdout).contains("E_MOLECULE_SPLIT"));
    let run = ucmd()
        .arg("check")
        .arg(spec.join("fixtures/runs/argon-nvt.toml"))
        .output()
        .unwrap();
    assert!(run.status.success());
}

#[test]
fn harmonic_nve_run_conserves_energy() {
    let Some(spec) = spec() else { return };
    let dir = run_dir(
        "harmonic-nve",
        &spec,
        "harmonic-3d",
        r#"spec_version = "0.1.0"
system = "system.ucsys"

[run]
integrator = "velocity-verlet"
timestep_fs = 1.0
steps = 2000
seed = 1

[[providers]]
kind = "harmonic-tether"
k_kjmol_nm2 = 1000.0

[output.thermo]
file = "thermo.tsv"
every = 10
"#,
    );
    let status = ucmd()
        .arg("run")
        .arg(dir.join("run.toml"))
        .status()
        .unwrap();
    assert!(status.success());
    let table = std::fs::read_to_string(dir.join("out/thermo.tsv")).unwrap();
    let header: Vec<&str> = table
        .lines()
        .next()
        .unwrap()
        .trim_start_matches("# ")
        .split('\t')
        .collect();
    for name in [
        "step",
        "time_ps",
        "temperature_K",
        "pressure_bar",
        "volume_nm3",
        "conserved_energy_kjmol",
    ] {
        assert!(
            header.contains(&name),
            "column {name} of spec/run-config.md missing: {header:?}"
        );
    }
    assert_eq!(header[5], "conserved_energy_kjmol");
    let conserved: Vec<f64> = table
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| l.split('\t').nth(5).unwrap().parse().unwrap())
        .collect();
    assert_eq!(conserved.len(), 201);
    let (lo, hi) = conserved
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), &e| (lo.min(e), hi.max(e)));
    assert!(
        (hi - lo) / hi.abs() < 1e-3,
        "energy fluctuation {}",
        (hi - lo) / hi.abs()
    );
}

#[test]
fn argon_run_writes_a_spec_valid_trajectory() {
    let Some(spec) = spec() else { return };
    let run_toml = std::fs::read_to_string(spec.join("fixtures/runs/argon-nvt.toml"))
        .unwrap()
        .replace(
            "system = \"../valid/argon-864\"",
            "system = \"system.ucsys\"",
        )
        .replace("steps = 25000", "steps = 200");
    let dir = run_dir("argon-nvt", &spec, "argon-864", &run_toml);
    let status = ucmd()
        .arg("run")
        .arg(dir.join("run.toml"))
        .status()
        .unwrap();
    assert!(status.success());
    let trajectory = dir.join("out/trajectory.h5");
    assert!(trajectory.is_file());
    let validator = spec.join("validator/.venv/bin/ucmd-spec");
    if validator.is_file() {
        let output = Command::new(validator)
            .arg("validate-trajectory")
            .arg(&trajectory)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stdout)
        );
    } else {
        eprintln!("ucmd-spec not installed; trajectory not validated");
    }
}

/// The argon NVT fixture with checkpoints every 100 steps and frames every 50 steps.
fn restart_base(spec: &Path) -> String {
    std::fs::read_to_string(spec.join("fixtures/runs/argon-nvt.toml"))
        .unwrap()
        .replace(
            "system = \"../valid/argon-864\"",
            "system = \"system.ucsys\"",
        )
        .replace("every = 5000", "every = 100")
        .replace(
            "[output.trajectory]\nfile = \"trajectory.h5\"\nevery = 100",
            "[output.trajectory]\nfile = \"trajectory.h5\"\nevery = 50",
        )
}

/// Run `base` (with `steps = 25000`) for 200 steps in one go and as 100 steps + restart + 100
/// steps. The end states, the thermostat state and the thermo tables must agree bitwise.
/// Returns the directory of the split run.
fn assert_restart_is_bitwise(name: &str, spec: &Path, base: &str) -> PathBuf {
    let full = run_dir(
        &format!("{name}-full"),
        spec,
        "argon-864",
        &base.replace("steps = 25000", "steps = 200"),
    );
    assert!(
        ucmd()
            .arg("run")
            .arg(full.join("run.toml"))
            .status()
            .unwrap()
            .success()
    );

    let split = run_dir(
        &format!("{name}-split"),
        spec,
        "argon-864",
        &base.replace("steps = 25000", "steps = 100"),
    );
    assert!(
        ucmd()
            .arg("run")
            .arg(split.join("run.toml"))
            .status()
            .unwrap()
            .success()
    );
    std::fs::write(
        split.join("run.toml"),
        base.replace("steps = 25000", "steps = 200"),
    )
    .unwrap();
    let status = ucmd()
        .arg("run")
        .arg(split.join("run.toml"))
        .arg("--restart")
        .arg(split.join("out/checkpoint.ucchk"))
        .status()
        .unwrap();
    assert!(status.success());

    let a = ucmd_io::Checkpoint::read(&full.join("out/checkpoint.ucchk")).unwrap();
    let b = ucmd_io::Checkpoint::read(&split.join("out/checkpoint.ucchk")).unwrap();
    assert_eq!((a.meta.step, b.meta.step), (200, 200));
    let bits = |v: &[ucmd_core::Vec3]| {
        v.iter()
            .flat_map(|x| x.to_array().map(f64::to_bits))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        bits(&a.state.positions),
        bits(&b.state.positions),
        "positions differ"
    );
    assert_eq!(
        bits(&a.state.velocities),
        bits(&b.state.velocities),
        "velocities differ"
    );
    assert_eq!(
        a.meta.thermostat_work_kjmol.to_bits(),
        b.meta.thermostat_work_kjmol.to_bits()
    );
    assert_eq!(a.meta.thermostat_rng, b.meta.thermostat_rng);
    let state_bits = |s: &[f64]| s.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        state_bits(&a.meta.thermostat_state),
        state_bits(&b.meta.thermostat_state),
        "thermostat state differs"
    );
    let thermo = |d: &Path| std::fs::read_to_string(d.join("out/thermo.tsv")).unwrap();
    assert_eq!(thermo(&full), thermo(&split), "thermo tables differ");
    // the whole files, metadata included: every f64 of it must survive the restart exactly
    let bytes = |d: &Path| std::fs::read(d.join("out/checkpoint.ucchk")).unwrap();
    assert!(bytes(&full) == bytes(&split), "checkpoint files differ");
    split
}

/// E1.10: 200 steps in one go equal 100 steps + restart + 100 steps, bitwise (CSVR included:
/// its random generator state travels in the checkpoint), and the outputs are identical.
#[test]
fn restart_is_bitwise_identical() {
    let Some(spec) = spec() else { return };
    let base = restart_base(&spec);
    let split = assert_restart_is_bitwise("restart", &spec, &base);

    // a changed configuration is refused
    std::fs::write(
        split.join("run.toml"),
        base.replace("steps = 25000", "steps = 300")
            .replace("tau_ps = 0.5", "tau_ps = 0.6"),
    )
    .unwrap();
    let refused = ucmd()
        .arg("run")
        .arg(split.join("run.toml"))
        .arg("--restart")
        .arg(split.join("out/checkpoint.ucchk"))
        .output()
        .unwrap();
    assert!(!refused.status.success());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("E_RESTART_MISMATCH"));

    // provenance: the first restart is recorded; with --force the run continues and the
    // override is recorded with its difference (spec/checkpoint.md)
    #[cfg(feature = "h5md")]
    {
        let trajectory = split.join("out/trajectory.h5");
        let full = split.with_file_name("restart-full");
        assert!(
            ucmd_io::read_restarts(&full.join("out/trajectory.h5"))
                .unwrap()
                .is_empty()
        );
        let first = ucmd_io::read_restarts(&trajectory).unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!((first[0].step, first[0].forced), (100, false));
        assert!(first[0].differences.is_empty());
        assert!(first[0].run_config_toml.contains("tau_ps = 0.5"));
        let forced = ucmd()
            .arg("run")
            .arg(split.join("run.toml"))
            .arg("--restart")
            .arg(split.join("out/checkpoint.ucchk"))
            .arg("--force")
            .status()
            .unwrap();
        assert!(forced.success());
        let both = ucmd_io::read_restarts(&trajectory).unwrap();
        assert_eq!(both.len(), 2);
        assert_eq!((both[1].step, both[1].forced), (200, true));
        assert_eq!(both[1].differences.len(), 1);
        assert!(both[1].differences[0].starts_with("run configuration: "));
        assert!(both[1].run_config_toml.contains("tau_ps = 0.6"));
    }

    let validator = spec.join("validator/.venv/bin/ucmd-spec");
    if validator.is_file() {
        let output = Command::new(validator)
            .arg("validate-trajectory")
            .arg(split.join("out/trajectory.h5"))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stdout)
        );
    }
}

/// E2.6: the Nosé–Hoover chain variables travel in the checkpoint (format v2), so a restart
/// is bitwise identical; the chain moves away from its initial state.
#[test]
fn restart_with_nose_hoover_chain_is_bitwise_identical() {
    let Some(spec) = spec() else { return };
    let base = restart_base(&spec).replace(
        "kind = \"csvr\"\ntemperature_K = 94.4\ntau_ps = 0.5",
        "kind = \"nose-hoover-chain\"\ntemperature_K = 94.4\ntau_ps = 0.5\nchain_length = 3",
    );
    assert!(base.contains("nose-hoover-chain"));
    let split = assert_restart_is_bitwise("restart-nhc", &spec, &base);
    let c = ucmd_io::Checkpoint::read(&split.join("out/checkpoint.ucchk")).unwrap();
    assert_eq!(c.meta.thermostat_state.len(), 6, "ξ₁..ξ₃ and v_ξ₁..v_ξ₃");
    assert!(c.meta.thermostat_state.iter().any(|x| *x != 0.0));
    assert!(c.meta.thermostat_rng.is_none());
}

/// E2.6: Langevin BAOAB restarts bitwise: the random stream travels in the checkpoint.
#[test]
fn restart_with_langevin_baoab_is_bitwise_identical() {
    let Some(spec) = spec() else { return };
    let base = restart_base(&spec)
        .replace(
            "integrator = \"velocity-verlet\"",
            "integrator = \"langevin-baoab\"",
        )
        .replace(
            "kind = \"csvr\"\ntemperature_K = 94.4\ntau_ps = 0.5",
            "kind = \"langevin\"\ntemperature_K = 94.4\nfriction_per_ps = 1.0",
        );
    assert!(base.contains("langevin-baoab") && base.contains("friction_per_ps"));
    let split = assert_restart_is_bitwise("restart-langevin", &spec, &base);
    let c = ucmd_io::Checkpoint::read(&split.join("out/checkpoint.ucchk")).unwrap();
    assert!(c.meta.thermostat_rng.is_some());
    assert!(c.meta.thermostat_state.is_empty());
}

/// B2a: SIGTERM stops a run gracefully. The current step completes, the checkpoint is written,
/// the trajectory is closed and still valid, and the exit status is 128 + 15. The run then
/// continues from that checkpoint.
#[cfg(unix)]
#[test]
fn sigterm_stops_a_run_gracefully() {
    use std::process::Stdio;
    use std::time::{Duration, Instant};
    let Some(spec) = spec() else { return };
    let base = restart_base(&spec);
    let dir = run_dir(
        "sigterm",
        &spec,
        "argon-864",
        &base.replace("steps = 25000", "steps = 1000000"),
    );
    let child = ucmd()
        .arg("run")
        .arg(dir.join("run.toml"))
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let thermo = dir.join("out/thermo.tsv");
    let deadline = Instant::now() + Duration::from_secs(120);
    while std::fs::read_to_string(&thermo).map_or(0, |t| t.lines().count()) < 5 {
        assert!(Instant::now() < deadline, "the run made no progress");
        std::thread::sleep(Duration::from_millis(50));
    }
    let kill = Command::new("kill")
        .arg("-TERM")
        .arg(child.id().to_string())
        .status()
        .unwrap();
    assert!(kill.success());
    let output = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(143), "{stderr}");
    assert!(stderr.contains("stopped by SIGTERM"), "{stderr}");
    let stopped = ucmd_io::Checkpoint::read(&dir.join("out/checkpoint.ucchk")).unwrap();
    assert!(stopped.meta.step > 0 && stopped.meta.step < 1_000_000);

    let validator = spec.join("validator/.venv/bin/ucmd-spec");
    if validator.is_file() {
        let check = Command::new(validator)
            .arg("validate-trajectory")
            .arg(dir.join("out/trajectory.h5"))
            .output()
            .unwrap();
        assert!(
            check.status.success(),
            "{}",
            String::from_utf8_lossy(&check.stdout)
        );
    }
    std::fs::write(
        dir.join("run.toml"),
        base.replace(
            "steps = 25000",
            &format!("steps = {}", stopped.meta.step + 20),
        ),
    )
    .unwrap();
    let resumed = ucmd()
        .arg("run")
        .arg(dir.join("run.toml"))
        .arg("--restart")
        .arg(dir.join("out/checkpoint.ucchk"))
        .status()
        .unwrap();
    assert!(resumed.success());
    let end = ucmd_io::Checkpoint::read(&dir.join("out/checkpoint.ucchk")).unwrap();
    assert_eq!(end.meta.step, stopped.meta.step + 20);
}

/// `ucmd export` turns a checkpoint into a valid package that carries the checkpoint's state.
#[test]
fn export_writes_a_valid_package_with_velocities() {
    let Some(spec) = spec() else { return };
    let dir = run_dir(
        "export",
        &spec,
        "harmonic-3d",
        r#"spec_version = "0.1.0"
system = "system.ucsys"

[run]
integrator = "velocity-verlet"
timestep_fs = 1.0
steps = 50
seed = 1

[[providers]]
kind = "harmonic-tether"
k_kjmol_nm2 = 1000.0

[output.checkpoint]
file = "checkpoint.ucchk"
every = 50
"#,
    );
    assert!(
        ucmd()
            .arg("run")
            .arg(dir.join("run.toml"))
            .status()
            .unwrap()
            .success()
    );
    let out = dir.join("exported.ucsys");
    let status = ucmd()
        .arg("export")
        .arg(dir.join("run.toml"))
        .arg("--checkpoint")
        .arg(dir.join("out/checkpoint.ucchk"))
        .arg("--out")
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let package = ucmd_io::read_package(&out).unwrap();
    let checkpoint = ucmd_io::Checkpoint::read(&dir.join("out/checkpoint.ucchk")).unwrap();
    assert!(package.has_velocities);
    assert_eq!(package.state.positions, checkpoint.state.positions);
    assert_eq!(package.state.velocities, checkpoint.state.velocities);
    assert!(ucmd().arg("check").arg(&out).status().unwrap().success());
}

/// `ucmd rerun` on a trajectory reproduces the potential energies the run itself recorded.
#[test]
fn rerun_reproduces_the_recorded_energies() {
    let Some(spec) = spec() else { return };
    let run_toml = std::fs::read_to_string(spec.join("fixtures/runs/argon-nvt.toml"))
        .unwrap()
        .replace(
            "system = \"../valid/argon-864\"",
            "system = \"system.ucsys\"",
        )
        .replace("steps = 25000", "steps = 200")
        .replace(
            "[output.thermo]\nfile = \"thermo.tsv\"\nevery = 10",
            "[output.thermo]\nfile = \"thermo.tsv\"\nevery = 50",
        )
        .replace(
            "[output.trajectory]\nfile = \"trajectory.h5\"\nevery = 100",
            "[output.trajectory]\nfile = \"trajectory.h5\"\nevery = 50",
        );
    let dir = run_dir("rerun", &spec, "argon-864", &run_toml);
    assert!(
        ucmd()
            .arg("run")
            .arg(dir.join("run.toml"))
            .status()
            .unwrap()
            .success()
    );
    let out = dir.join("rerun.tsv");
    let status = ucmd()
        .arg("rerun")
        .arg(dir.join("run.toml"))
        .arg("--trajectory")
        .arg(dir.join("out/trajectory.h5"))
        .arg("--out")
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let column = |path: &Path, index: usize| -> Vec<String> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .filter(|l| !l.starts_with('#'))
            .map(|l| l.split('\t').nth(index).unwrap().to_string())
            .collect()
    };
    let recorded = column(&dir.join("out/thermo.tsv"), 2);
    let reevaluated = column(&out, 2);
    assert_eq!(recorded.len(), 5);
    assert_eq!(recorded, reevaluated);
}
