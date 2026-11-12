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

/// E1.10: 200 steps in one go equal 100 steps + restart + 100 steps, bitwise (CSVR included:
/// its random generator state travels in the checkpoint), and the outputs are identical.
#[test]
fn restart_is_bitwise_identical() {
    let Some(spec) = spec() else { return };
    let base = std::fs::read_to_string(spec.join("fixtures/runs/argon-nvt.toml"))
        .unwrap()
        .replace(
            "system = \"../valid/argon-864\"",
            "system = \"system.ucsys\"",
        )
        .replace("every = 5000", "every = 100")
        .replace(
            "[output.trajectory]\nfile = \"trajectory.h5\"\nevery = 100",
            "[output.trajectory]\nfile = \"trajectory.h5\"\nevery = 50",
        );
    let full = run_dir(
        "restart-full",
        &spec,
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
        "restart-split",
        &spec,
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
    let thermo = |d: &Path| std::fs::read_to_string(d.join("out/thermo.tsv")).unwrap();
    assert_eq!(thermo(&full), thermo(&split), "thermo tables differ");

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
