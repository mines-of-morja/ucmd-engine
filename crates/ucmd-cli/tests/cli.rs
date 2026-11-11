//! Smoke test: the `ucmd` binary builds, runs and reports its version.

use std::process::Command;

#[test]
fn binary_runs_and_reports_version() {
    let output = Command::new(env!("CARGO_BIN_EXE_ucmd"))
        .output()
        .expect("failed to run the ucmd binary");
    assert!(
        output.status.success(),
        "ucmd exited with {}",
        output.status
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.starts_with(concat!("ucmd ", env!("CARGO_PKG_VERSION"))),
        "unexpected output: {stdout}"
    );
}
