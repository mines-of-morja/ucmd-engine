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
