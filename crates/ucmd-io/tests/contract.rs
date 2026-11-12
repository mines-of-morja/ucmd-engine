//! Contract test (E1.4): this reader reports the same error code as the Python reference
//! validator for every spec fixture. Fixtures come from `$UCMD_SPEC_DIR`, by default
//! `../spec` relative to the engine repo (workspace layout). Without them the test is skipped,
//! unless `UCMD_REQUIRE_SPEC` is set.

#![allow(clippy::unwrap_used, clippy::expect_used)] // test helpers may panic

use std::path::PathBuf;
use ucmd_io::{load_run_config, read_package};

fn spec_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("UCMD_SPEC_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../spec"));
    if dir.join("fixtures").is_dir() {
        Some(dir)
    } else {
        assert!(
            std::env::var_os("UCMD_REQUIRE_SPEC").is_none(),
            "spec fixtures not found at {}",
            dir.display()
        );
        eprintln!(
            "skipping contract test: no spec fixtures at {}",
            dir.display()
        );
        None
    }
}

fn subdirs(dir: PathBuf) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.is_dir())
        .collect();
    out.sort();
    out
}

fn expected_code(json: &str) -> String {
    let value: serde_json::Value = serde_json::from_str(json).unwrap();
    value["code"].as_str().unwrap().to_string()
}

#[test]
fn every_valid_package_loads_without_warnings() {
    let Some(spec) = spec_dir() else { return };
    let valid = subdirs(spec.join("fixtures/valid"));
    assert!(valid.len() >= 4);
    for dir in valid {
        let package = read_package(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        assert!(
            package.warnings.is_empty(),
            "{}: {:?}",
            dir.display(),
            package.warnings
        );
        assert_eq!(package.system.n_atoms(), package.state.n_atoms());
        assert_eq!(package.system.system_sha256.len(), 64);
    }
}

#[test]
fn every_invalid_package_reports_the_validators_code() {
    let Some(spec) = spec_dir() else { return };
    let invalid = subdirs(spec.join("fixtures/invalid"));
    assert!(invalid.len() >= 19);
    for dir in invalid {
        let expected = expected_code(&std::fs::read_to_string(dir.join("expected.json")).unwrap());
        match read_package(&dir) {
            Ok(_) => panic!("{} loaded, expected {expected}", dir.display()),
            Err(e) => assert_eq!(e.code.as_str(), expected, "{}: {e}", dir.display()),
        }
    }
}

#[test]
fn run_configs_match_the_validator() {
    let Some(spec) = spec_dir() else { return };
    let runs = spec.join("fixtures/runs");
    for name in ["argon-nvt.toml", "harmonic-nve.toml"] {
        let config = load_run_config(&runs.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        let package = read_package(&config.system_dir()).unwrap_or_else(|e| panic!("{name}: {e}"));
        config.check_against_system(&package.system).unwrap();
        assert!(config.resolved_toml().contains("spec_version"));
    }
    let expected: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(runs.join("invalid/expected.json")).unwrap())
            .unwrap();
    for (name, code) in expected.as_object().unwrap() {
        match load_run_config(&runs.join("invalid").join(name)) {
            Ok(_) => panic!("{name} loaded, expected {code}"),
            Err(e) => assert_eq!(e.code.as_str(), code.as_str().unwrap(), "{name}: {e}"),
        }
    }
}
