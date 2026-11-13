//! E2.0 end to end: `ucmd minimize` relaxes a molecule with the `force-field` provider and writes
//! a valid package; without convergence it writes nothing and fails.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code

use std::path::PathBuf;
use std::process::Command;

fn molecule(name: &str) -> Option<PathBuf> {
    let root = std::env::var_os("UCMD_SPEC_DIR")
        .map(|s| PathBuf::from(s).join(".."))
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."));
    let dir = root.join("tools/oracle/reference/molecules").join(name);
    if dir.is_dir() {
        Some(dir)
    } else {
        assert!(
            std::env::var_os("UCMD_REQUIRE_SPEC").is_none(),
            "no reference molecule {name}"
        );
        None
    }
}

const RUN: &str = r#"
spec_version = "0.1.0"
system = "ec.ucsys"

[run]
integrator = "velocity-verlet"
timestep_fs = 1.0
steps = 0
seed = 1

[[providers]]
kind = "force-field"
"#;

#[test]
fn minimize_relaxes_a_molecule_and_refuses_unconverged_output() {
    let Some(source) = molecule("EC") else { return };
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("minimize-ec");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("ec.ucsys")).unwrap();
    for file in ["system.json", "positions.npy"] {
        std::fs::copy(source.join(file), dir.join("ec.ucsys").join(file)).unwrap();
    }
    std::fs::write(dir.join("run.toml"), RUN).unwrap();
    let ucmd = || Command::new(env!("CARGO_BIN_EXE_ucmd"));
    let out = ucmd()
        .current_dir(&dir)
        .args([
            "minimize",
            "run.toml",
            "--out",
            "relaxed.ucsys",
            "--max-force",
            "1.0",
        ])
        .output()
        .unwrap();
    let log = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{log}");
    assert!(log.contains("converged"), "{log}");
    let check = ucmd()
        .current_dir(&dir)
        .args(["check", "relaxed.ucsys"])
        .output()
        .unwrap();
    assert!(check.status.success());
    let doc: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("relaxed.ucsys/system.json")).unwrap(),
    )
    .unwrap();
    let m = &doc["provenance"]["minimization"];
    assert!(m["energy_kjmol"].as_f64().unwrap() < m["energy_start_kjmol"].as_f64().unwrap());
    assert!(m["max_force_kjmol_nm"].as_f64().unwrap() <= 1.0);
    // An impossible criterion within 5 steps: non-zero exit and no package.
    let fail = ucmd()
        .current_dir(&dir)
        .args([
            "minimize",
            "run.toml",
            "--out",
            "never.ucsys",
            "--max-force",
            "1e-9",
            "--max-steps",
            "5",
        ])
        .output()
        .unwrap();
    assert!(!fail.status.success());
    assert!(!dir.join("never.ucsys").exists());
}
