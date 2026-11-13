//! ADR 0010: the reference backend gives bit-identical results on every machine.
//!
//! A fixed argon NVT run (Maxwell–Boltzmann velocities, CSVR thermostat, Lennard-Jones with the
//! tail correction, 300 steps of 4 fs) must end in exactly the state whose SHA-256 is recorded
//! below. The value was computed on the author's workstation; CI checks it on GitHub's machines,
//! which have other CPUs and another libc. A change of the value means that the trajectory of a
//! given seed changed: allowed only on purpose (a physics or sampler change), never by accident.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code

use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::process::Command;

const GOLDEN_SHA256: &str = "aaa50c54f6fc7d2ba6986c144a664cbc26c375748832fcd2db88003f1c45068b";

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

const RUN: &str = r#"
spec_version = "0.1.0"
system = "argon.ucsys"

[run]
integrator = "velocity-verlet"
timestep_fs = 4.0
steps = 300
seed = 2026

[velocities]
generate = "maxwell-boltzmann"
temperature_K = 94.4

[thermostat]
kind = "csvr"
temperature_K = 94.4
tau_ps = 0.5

[[providers]]
kind = "lennard-jones"
cutoff_nm = 0.85
cutoff_mode = "potential-shift"
dispersion_correction = true

[output.checkpoint]
file = "checkpoint.ucchk"
every = 300
"#;

#[test]
fn argon_trajectory_is_bit_identical_on_every_machine() {
    let Some(spec) = spec() else { return };
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("golden-argon");
    let _ = std::fs::remove_dir_all(&dir);
    let package = dir.join("argon.ucsys");
    std::fs::create_dir_all(&package).unwrap();
    for entry in std::fs::read_dir(spec.join("fixtures/valid/argon-864")).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), package.join(entry.file_name())).unwrap();
    }
    std::fs::write(dir.join("run.toml"), RUN).unwrap();
    let ucmd = || Command::new(env!("CARGO_BIN_EXE_ucmd"));
    let run = ucmd()
        .current_dir(&dir)
        .args(["run", "run.toml"])
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let export = ucmd()
        .current_dir(&dir)
        .args([
            "export",
            "run.toml",
            "--checkpoint",
            "out/checkpoint.ucchk",
            "--out",
            "final.ucsys",
        ])
        .output()
        .unwrap();
    assert!(
        export.status.success(),
        "{}",
        String::from_utf8_lossy(&export.stderr)
    );
    let mut hasher = Sha256::new();
    for name in ["positions.npy", "velocities.npy"] {
        hasher.update(std::fs::read(dir.join("final.ucsys").join(name)).unwrap());
    }
    let digest: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        digest, GOLDEN_SHA256,
        "the final state differs from the recorded one (ADR 0010); see the test's doc comment"
    );
}
