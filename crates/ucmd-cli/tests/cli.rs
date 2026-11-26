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

/// E2.7: with the C-rescale barostat, 200 steps equal 100 + restart + 100 bitwise: the cell,
/// the barostat's random stream and its work travel in the checkpoint. The box does move.
#[test]
fn restart_with_c_rescale_barostat_is_bitwise_identical() {
    let Some(spec) = spec() else { return };
    let base = restart_base(&spec).replace(
        "[[providers]]",
        "[barostat]\nkind = \"c-rescale\"\npressure_bar = 1.0\ntau_ps = 0.5\n\
         compressibility_per_bar = 2e-4\nevery = 5\n\n[[providers]]",
    );
    assert!(base.contains("c-rescale"));
    let split = assert_restart_is_bitwise("restart-barostat", &spec, &base);
    let c = ucmd_io::Checkpoint::read(&split.join("out/checkpoint.ucchk")).unwrap();
    assert!(c.meta.barostat_rng.is_some());
    assert!(c.meta.barostat_work_kjmol != 0.0);
    let initial =
        std::fs::read_to_string(spec.join("fixtures/valid/argon-864/system.json")).unwrap();
    let v0: serde_json::Value = serde_json::from_str(&initial).unwrap();
    let l0 = v0["cell_nm"][0][0].as_f64().unwrap();
    let l = c.state.cell.as_ref().unwrap().rows()[0].x;
    assert!(
        (l / l0 - 1.0).abs() > 1e-6,
        "the box did not change: {l0} → {l}"
    );
    let thermo = std::fs::read_to_string(split.join("out/thermo.tsv")).unwrap();
    let volumes: std::collections::BTreeSet<_> = thermo
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| l.split('\t').nth(8).unwrap().to_string())
        .collect();
    assert!(
        volumes.len() > 10,
        "the thermo table records the changing volume"
    );
}

/// E2.7: under NPT the conserved quantity H = K + U + P₀V + bath − barostat work stays
/// constant up to integration error. The energy of each rescaling is booked by the trapezoid
/// rule; a one-sided estimate let H drift by ±2 kJ/mol/ps here (45 kJ/mol over 20 ps).
///
/// Without the LJ tail correction: its pressure is the virial integral beyond the cutoff
/// (P_tail·V ≈ 2·E_tail), not −dE_tail/dV (= E_tail/V), because pairs cross the cutoff when the
/// box scales. With it, H follows the volume by design (66 kJ/mol over 10 ps here), as in other
/// codes; the sampled ensemble uses the virial pressure either way.
#[test]
fn the_conserved_energy_holds_under_npt() {
    let Some(spec) = spec() else { return };
    let base = restart_base(&spec)
        .replace(
            "dispersion_correction = true",
            "dispersion_correction = false",
        )
        .replace("steps = 25000", "steps = 2500")
        .replace("every = 10\n", "every = 1\n")
        .replace(
            "[[providers]]",
            "[barostat]\nkind = \"c-rescale\"\npressure_bar = 100.0\ntau_ps = 1.0\n\
             compressibility_per_bar = 2e-4\nevery = 10\n\n[[providers]]",
        );
    let dir = run_dir("npt-conserved", &spec, "argon-864", &base);
    assert!(
        ucmd()
            .arg("run")
            .arg(dir.join("run.toml"))
            .status()
            .unwrap()
            .success()
    );
    let thermo = std::fs::read_to_string(dir.join("out/thermo.tsv")).unwrap();
    let mut lines = thermo.lines();
    let names: Vec<&str> = lines
        .next()
        .unwrap()
        .trim_start_matches("# ")
        .split('\t')
        .collect();
    let (h, v) = (
        names
            .iter()
            .position(|n| *n == "conserved_energy_kjmol")
            .unwrap(),
        names.iter().position(|n| *n == "volume_nm3").unwrap(),
    );
    let rows: Vec<Vec<f64>> = lines
        .map(|l| l.split('\t').map(|x| x.parse().unwrap()).collect())
        .collect();
    let range = |k: usize| {
        let xs = rows.iter().map(|r| r[k]);
        xs.clone().fold(f64::MIN, f64::max) - xs.fold(f64::MAX, f64::min)
    };
    assert!(range(v) > 0.05, "the volume moves: {}", range(v));
    assert!(range(h) < 1.0, "H range over 10 ps: {} kJ/mol", range(h));
}

/// E2.7: a barostat takes its temperature from the thermostat, so it needs one.
#[test]
fn a_barostat_without_a_thermostat_is_refused() {
    let Some(spec) = spec() else { return };
    let base = restart_base(&spec)
        .replace(
            "[thermostat]\nkind = \"csvr\"\ntemperature_K = 94.4\ntau_ps = 0.5\n",
            "",
        )
        .replace(
            "[[providers]]",
            "[barostat]\nkind = \"c-rescale\"\npressure_bar = 1.0\ntau_ps = 0.5\n\
             compressibility_per_bar = 2e-4\n\n[[providers]]",
        );
    assert!(!base.contains("[thermostat]"));
    let dir = run_dir("barostat-no-thermostat", &spec, "argon-864", &base);
    let out = ucmd()
        .arg("run")
        .arg(dir.join("run.toml"))
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("a barostat needs a thermostat"));
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

/// Spec 0.1.1: `[output.thermo] pressure_tensor` writes six columns whose trace / 3 is the
/// pressure column; a non-periodic system is refused (E_INCOMPATIBLE).
#[test]
fn the_pressure_tensor_columns_add_up_to_the_pressure() {
    let Some(spec) = spec() else { return };
    let text = std::fs::read_to_string(spec.join("fixtures/runs/argon-pressure-tensor.toml"))
        .unwrap()
        .replace(
            "system = \"../valid/argon-864\"",
            "system = \"system.ucsys\"",
        )
        .replace("steps = 25000", "steps = 200");
    let dir = run_dir("pressure_tensor", &spec, "argon-864", &text);
    let out = ucmd()
        .arg("run")
        .arg(dir.join("run.toml"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let table = std::fs::read_to_string(dir.join("out/thermo.tsv")).unwrap();
    let mut lines = table.lines();
    let names: Vec<&str> = lines
        .next()
        .unwrap()
        .trim_start_matches("# ")
        .split('\t')
        .collect();
    let col = |n: &str| names.iter().position(|c| *c == n).unwrap();
    let tensor = ucmd_io::thermo::PRESSURE_TENSOR_COLUMNS.map(col);
    assert_eq!(
        tensor,
        [9, 10, 11, 12, 13, 14].map(|k| k + names.len() - 15)
    );
    let mut rows = 0;
    for line in lines {
        let v: Vec<f64> = line.split('\t').map(|x| x.parse().unwrap()).collect();
        let trace = (v[tensor[0]] + v[tensor[1]] + v[tensor[2]]) / 3.0;
        let p = v[col("pressure_bar")];
        assert!(
            (trace - p).abs() <= 1e-8 * p.abs().max(1.0),
            "{trace} vs {p}"
        );
        assert!(v[tensor[3]].is_finite() && v[tensor[3]] != 0.0);
        rows += 1;
    }
    assert_eq!(rows, 21);

    let flat = std::fs::read_to_string(spec.join("fixtures/runs/harmonic-nve.toml"))
        .unwrap()
        .replace(
            "system = \"../valid/harmonic-3d\"",
            "system = \"system.ucsys\"",
        )
        .replace("spec_version = \"0.1.0\"", "spec_version = \"0.1.1\"")
        + "\n[output]\ndirectory = \"out\"\n\n[output.thermo]\nfile = \"t.tsv\"\nevery = 1\npressure_tensor = true\n";
    let dir = run_dir("pressure_tensor_flat", &spec, "harmonic-3d", &flat);
    let out = ucmd()
        .arg("check")
        .arg(dir.join("run.toml"))
        .output()
        .unwrap();
    assert!(!out.status.success());
    let text = String::from_utf8_lossy(&out.stdout) + String::from_utf8_lossy(&out.stderr);
    assert!(text.contains("E_INCOMPATIBLE"), "{text}");
}

/// Spec 0.1.3: `[output.thermo] pressure_molecular` adds one last column, which equals the pressure
/// column of a system without bonds and comes after the tensor; a
/// non-periodic system is refused (E_INCOMPATIBLE).
#[test]
fn the_molecular_pressure_column_is_written() {
    let Some(spec) = spec() else { return };
    let base = std::fs::read_to_string(spec.join("fixtures/runs/argon-pressure-tensor.toml"))
        .unwrap()
        .replace(
            "system = \"../valid/argon-864\"",
            "system = \"system.ucsys\"",
        )
        .replace("spec_version = \"0.1.1\"", "spec_version = \"0.1.3\"")
        .replace("steps = 25000", "steps = 200")
        .replace(
            "pressure_tensor = true",
            "pressure_tensor = true\npressure_molecular = true",
        );
    let dir = run_dir("pressure_molecular", &spec, "argon-864", &base);
    let out = ucmd()
        .arg("run")
        .arg(dir.join("run.toml"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let table = std::fs::read_to_string(dir.join("out/thermo.tsv")).unwrap();
    let mut lines = table.lines();
    let names: Vec<&str> = lines
        .next()
        .unwrap()
        .trim_start_matches("# ")
        .split('\t')
        .collect();
    assert_eq!(names.last(), Some(&"pressure_molecular_bar"));
    let atomic = names.iter().position(|c| *c == "pressure_bar").unwrap();
    let mut rows = 0;
    for line in lines {
        let v: Vec<f64> = line.split('\t').map(|x| x.parse().unwrap()).collect();
        assert_eq!(v.len(), names.len());
        let (p, m) = (v[atomic], v[names.len() - 1]);
        assert!((p - m).abs() <= 1e-8 * p.abs().max(1.0), "{p} vs {m}");
        rows += 1;
    }
    assert_eq!(rows, 21);

    let flat = std::fs::read_to_string(spec.join("fixtures/runs/harmonic-nve.toml"))
        .unwrap()
        .replace(
            "system = \"../valid/harmonic-3d\"",
            "system = \"system.ucsys\"",
        )
        .replace("spec_version = \"0.1.0\"", "spec_version = \"0.1.3\"")
        + "\n[output]\ndirectory = \"out\"\n\n[output.thermo]\nfile = \"t.tsv\"\nevery = 1\npressure_molecular = true\n";
    let dir = run_dir("pressure_molecular_flat", &spec, "harmonic-3d", &flat);
    let out = ucmd()
        .arg("check")
        .arg(dir.join("run.toml"))
        .output()
        .unwrap();
    assert!(!out.status.success());
    let text = String::from_utf8_lossy(&out.stdout) + String::from_utf8_lossy(&out.stderr);
    assert!(text.contains("E_INCOMPATIBLE"), "{text}");
}

/// E2.5: a package with model constraints runs (RATTLE) and keeps them, and minimizes on the
/// constraint manifold.
#[test]
fn rigid_water_runs_and_minimizes_with_constraints() {
    let Some(spec) = spec() else { return };
    let dir = run_dir(
        "constraints_rigid",
        &spec,
        "water-2",
        "spec_version = \"0.1.0\"\nsystem = \"system.ucsys\"\n\n[run]\nintegrator = \"velocity-verlet\"\n\
         timestep_fs = 1.0\nsteps = 2000\nseed = 1\n\n[[providers]]\nkind = \"force-field\"\n\
         coulomb = \"direct\"\n\n[output]\ndirectory = \"out\"\n\n[output.thermo]\n\
         file = \"t.tsv\"\nevery = 10\n",
    );
    // SPC/Fw held rigid at its equilibrium geometry: three distances per molecule
    let path = dir.join("system.ucsys/system.json");
    let mut doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let (r_oh, theta) = (0.1012_f64, 113.24_f64.to_radians());
    let r_hh = 2.0 * r_oh * (theta / 2.0).sin();
    doc["molecule_types"]["SPCFw"]["constraints"] = serde_json::json!([
        {"atoms": [0, 1], "length_nm": r_oh},
        {"atoms": [0, 2], "length_nm": r_oh},
        {"atoms": [1, 2], "length_nm": r_hh}
    ]);
    std::fs::write(&path, serde_json::to_string(&doc).unwrap()).unwrap();
    // the fixture is deliberately distorted (so its forces are not zero): place both molecules
    // exactly at the rigid geometry, 0.5 nm apart, in different orientations
    let h = theta / 2.0;
    let mut x = Vec::new();
    for (o, flip) in [
        (ucmd_core::Vec3::ZERO, 1.0),
        (ucmd_core::Vec3::new(0.5, 0.1, 0.0), -1.0),
    ] {
        x.push(o);
        x.push(o + ucmd_core::Vec3::new(r_oh * h.sin(), flip * r_oh * h.cos(), 0.0));
        x.push(o + ucmd_core::Vec3::new(-r_oh * h.sin(), flip * r_oh * h.cos(), 0.0));
    }
    ucmd_io::npy::write_vec3_array(&dir.join("system.ucsys/positions.npy"), &x).unwrap();
    let out = ucmd()
        .arg("run")
        .arg(dir.join("run.toml"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let table = std::fs::read_to_string(dir.join("out/t.tsv")).unwrap();
    let mut lines = table.lines();
    let names: Vec<&str> = lines
        .next()
        .unwrap()
        .trim_start_matches("# ")
        .split('\t')
        .collect();
    let col = |n: &str| names.iter().position(|c| *c == n).unwrap();
    let rows: Vec<Vec<f64>> = lines
        .map(|l| l.split('\t').map(|x| x.parse().unwrap()).collect())
        .collect();
    let range = |k: usize| {
        let v = rows.iter().map(|r| r[k]);
        v.clone().fold(f64::MIN, f64::max) - v.fold(f64::MAX, f64::min)
    };
    // the two molecules attract and exchange ~19 kJ/mol between U and K; the total holds to
    // 0.4 % of that during the close approach and drifts by ~1e-4 kJ/mol (2026-11-20)
    let (exchanged, spread) = (range(col("potential_kjmol")), range(col("total_kjmol")));
    let drift = (rows[rows.len() - 1][col("total_kjmol")] - rows[0][col("total_kjmol")]).abs();
    assert!(
        exchanged > 5.0,
        "the test needs energy exchange, got {exchanged}"
    );
    assert!(
        spread < 0.01 * exchanged,
        "spread {spread} of {exchanged} kJ/mol exchanged"
    );
    assert!(drift < 1e-3 * exchanged, "drift {drift} kJ/mol");
    // E2.5 step 5: minimization with constraints (FIRE on the constraint manifold)
    let min = ucmd()
        .args(["minimize", "--out", "min.ucsys", "--max-force", "1"])
        .arg(dir.join("run.toml"))
        .current_dir(&dir)
        .output()
        .unwrap();
    assert!(
        min.status.success(),
        "{}",
        String::from_utf8_lossy(&min.stderr)
    );
    let x = ucmd_io::npy::read_vec3_array(&dir.join("min.ucsys/positions.npy"), 6).unwrap();
    for mol in [0, 3] {
        for (i, j, d) in [(0, 1, r_oh), (0, 2, r_oh), (1, 2, r_hh)] {
            let r = (x[mol + i] - x[mol + j]).length();
            assert!((r - d).abs() < 1e-9 * d, "constraint {i}-{j}: {r} vs {d}");
        }
    }
}
