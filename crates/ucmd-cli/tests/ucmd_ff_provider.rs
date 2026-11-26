//! The `ucmd-ff` provider (the separately developed force-field engine) against the engine's own
//! `force-field` provider and the OpenMM references.
//!
//! Two independent implementations of the same terms must agree:
//! - non-periodic: as direct sums, to rounding;
//! - periodic: at the same splitting α, cutoff and converged reciprocal sums.
//!
//! ucmd-ff uses classic Ewald, so it is compared with the classic Ewald method of
//! `force-field`, not with PME.
//!
//! The electrolyte box (7036 atoms) is slow for ucmd-ff's O(N²) loops. That test is ignored by
//! default and runs with
//! `cargo test --release -p ucmd-cli --test ucmd_ff_provider -- --ignored`.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code

use std::path::{Path, PathBuf};
use std::process::Command;
use ucmd_core::{Cell, Mat3, Vec3};
use ucmd_force::{EvalRequest, ForceProvider, ForceResult};
use ucmd_io::npy::read_vec3_array;
use ucmd_io::read_package;
use ucmd_providers_classical::{CoulombMethod, ForceField, NonbondedConfig};
use ucmd_providers_ff::{UcmdFf, UcmdFfConfig};

/// The umbrella checkout (spec fixtures and oracle references), as the other tests find it.
fn umbrella() -> Option<PathBuf> {
    let root = std::env::var_os("UCMD_SPEC_DIR")
        .map(|s| PathBuf::from(s).join(".."))
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."));
    if root.join("spec/fixtures/valid").is_dir() && root.join("tools/oracle/reference").is_dir() {
        Some(root)
    } else {
        assert!(
            std::env::var_os("UCMD_REQUIRE_SPEC").is_none(),
            "spec fixtures or oracle references not found under {}",
            root.display()
        );
        eprintln!("skipping: no umbrella checkout under {}", root.display());
        None
    }
}

fn evaluate(
    p: &mut dyn ForceProvider,
    system: &ucmd_core::System,
    positions: &[Vec3],
    cell: Option<&Cell>,
) -> ForceResult {
    p.initialize(system).unwrap();
    let mut out = ForceResult::default();
    p.evaluate(positions, cell, &EvalRequest::everything(), &mut out)
        .unwrap();
    out
}

/// Sum of the named terms (0 if none is present).
fn term(out: &ForceResult, names: &[&str]) -> f64 {
    out.energy_terms
        .iter()
        .filter(|(n, _)| names.contains(&n.as_str()))
        .map(|(_, e)| e)
        .sum()
}

const COULOMB_UCMD_FF: [&str; 5] = [
    "coulomb",
    "coulomb_real",
    "coulomb_reciprocal",
    "coulomb_self",
    "coulomb_correction",
];

fn max_abs(v: &[Vec3]) -> f64 {
    v.iter().map(|f| f.abs().max_element()).fold(0.0, f64::max)
}

fn mat_max_abs(m: &Mat3) -> f64 {
    m.to_cols_array().iter().fold(0.0f64, |a, x| a.max(x.abs()))
}

/// Energies term by term, forces and virial of the two providers. Forces are compared relative
/// to the largest force, or to `force_floor` (kJ mol⁻¹ nm⁻¹) if that is larger.
fn assert_agree(label: &str, ours: &ForceResult, theirs: &ForceResult, rel: f64, force_floor: f64) {
    let pairs: [(&str, &[&str], &[&str]); 6] = [
        ("bond", &["bond"], &["bond"]),
        ("angle", &["angle"], &["angle"]),
        ("dihedral", &["dihedral"], &["dihedral"]),
        ("improper", &["improper"], &["improper"]),
        ("lj (with 1-4)", &["lj"], &["lj", "lj14"]),
        ("coulomb", &COULOMB_UCMD_FF, &["coulomb"]),
    ];
    let scale = theirs.energy.unwrap().abs().max(1.0);
    for (name, a, b) in pairs {
        let (x, y) = (term(ours, a), term(theirs, b));
        assert!(
            (x - y).abs() <= rel * scale.max(y.abs()),
            "{label} {name}: ucmd-ff {x:.12} vs force-field {y:.12}"
        );
    }
    let (e1, e2) = (ours.energy.unwrap(), theirs.energy.unwrap());
    assert!(
        (e1 - e2).abs() <= rel * e2.abs().max(1.0),
        "{label} energy: {e1:.12} vs {e2:.12}"
    );
    let f_scale = max_abs(&theirs.forces).max(force_floor);
    let df = ours
        .forces
        .iter()
        .zip(&theirs.forces)
        .map(|(a, b)| (*a - *b).abs().max_element())
        .fold(0.0, f64::max);
    assert!(
        df <= rel * f_scale,
        "{label} forces differ by {df:e} (largest force {f_scale:e})"
    );
    let (w1, w2) = (ours.virial.unwrap(), theirs.virial.unwrap());
    let dw = mat_max_abs(&(w1 - w2));
    assert!(
        dw <= rel * mat_max_abs(&w2).max(1e-3),
        "{label} virial differs by {dw:e}"
    );
    eprintln!(
        "{label}: energy {e1:.10} vs {e2:.10}, forces within {df:.1e}, virial within {dw:.1e}"
    );
}

#[test]
fn ucmd_ff_matches_openmm_on_the_reference_molecules() {
    // tools/oracle/reference/molecules: EC, EMC, Li+ + PF6- and a molecule with every bonded form,
    // OpenMM 8.6.1 Reference platform, non-periodic. Criteria as oracle_molecules.rs: each term
    // to 1e-8 relative, forces to 1e-8 of the largest force.
    let Some(root) = umbrella() else { return };
    let dir = root.join("tools/oracle/reference/molecules");
    let mut cases: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("terms.json").is_file())
        .collect();
    cases.sort();
    assert!(
        cases.len() >= 4,
        "expected 4 reference molecules in {}",
        dir.display()
    );
    for case in cases {
        let package = read_package(&case).unwrap();
        assert!(!package.system.is_periodic());
        let n = package.system.n_atoms();
        let out = evaluate(
            &mut UcmdFf::new("ucmd-ff", UcmdFfConfig::default()),
            &package.system,
            &package.state.positions,
            None,
        );
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(case.join("terms.json")).unwrap())
                .unwrap();
        let reference = &meta["terms_kjmol"];
        let r = |k: &str| reference[k].as_f64().unwrap_or(0.0);
        let checks = [
            ("bond", term(&out, &["bond"]), r("bond")),
            ("angle", term(&out, &["angle"]), r("angle")),
            ("dihedral", term(&out, &["dihedral"]), r("dihedral")),
            ("improper", term(&out, &["improper"]), r("improper")),
            ("lj + lj14", term(&out, &["lj"]), r("lj") + r("lj14")),
            ("coulomb", term(&out, &COULOMB_UCMD_FF), r("coulomb")),
            ("total", out.energy.unwrap(), r("total")),
        ];
        for (name, ours, theirs) in checks {
            let rel = (ours - theirs).abs() / theirs.abs().max(1e-6);
            assert!(
                rel < 1e-8,
                "{} {name}: ucmd-ff {ours:.12} vs OpenMM {theirs:.12} (rel {rel:.2e})",
                case.display()
            );
        }
        let mut reference_forces = vec![Vec3::ZERO; n];
        for t in [
            "bond", "angle", "dihedral", "improper", "coulomb", "lj", "lj14",
        ] {
            let path = case.join(format!("forces-{t}.npy"));
            if path.is_file() {
                for (s, f) in reference_forces
                    .iter_mut()
                    .zip(read_vec3_array(&path, n).unwrap())
                {
                    *s += f;
                }
            }
        }
        let scale = max_abs(&reference_forces);
        let worst = out
            .forces
            .iter()
            .zip(&reference_forces)
            .map(|(a, b)| (*a - *b).abs().max_element())
            .fold(0.0, f64::max);
        assert!(
            worst <= 1e-8 * scale,
            "{}: forces differ from OpenMM by {worst:e} (largest {scale:e})",
            case.display()
        );
        eprintln!(
            "{}: all terms within 1e-8, forces within {:.1e} relative",
            case.display(),
            worst / scale
        );
    }
}

#[test]
fn ucmd_ff_and_force_field_agree_on_water2() {
    let Some(root) = umbrella() else { return };
    let package = read_package(&root.join("spec/fixtures/valid/water-2")).unwrap();
    let ours = evaluate(
        &mut UcmdFf::new("ucmd-ff", UcmdFfConfig::default()),
        &package.system,
        &package.state.positions,
        None,
    );
    let theirs = evaluate(
        &mut ForceField::new("force-field", NonbondedConfig::default()),
        &package.system,
        &package.state.positions,
        None,
    );
    assert_agree("water-2", &ours, &theirs, 1e-12, 1e-3);
}

/// Both providers with classic Ewald at the same α and cutoff; reciprocal sums converged.
fn ewald_pair(cutoff_nm: f64, alpha: f64) -> (UcmdFf, ForceField) {
    let ours = UcmdFf::new(
        "ucmd-ff",
        UcmdFfConfig {
            cutoff_nm: Some(cutoff_nm),
            ewald_alpha_per_nm: Some(alpha),
            ewald_tolerance: 1e-14,
            ..UcmdFfConfig::default()
        },
    );
    let theirs = ForceField::new(
        "force-field",
        NonbondedConfig {
            cutoff_nm: Some(cutoff_nm),
            coulomb: Some(CoulombMethod::Ewald),
            ewald_alpha_per_nm: Some(alpha),
            ewald_tolerance: 1e-14,
            // cutoff + skin must stay below half the cell height (small test cells)
            neighbor_skin_nm: 0.05,
            ..NonbondedConfig::default()
        },
    );
    (ours, theirs)
}

#[test]
fn ucmd_ff_and_force_field_agree_on_the_nacl_crystal() {
    // Fixture nacl-crystal (64 ions, a = 0.564 nm): E = -32 M k_e/(a/2), M = 1.747565 (spec).
    let Some(root) = umbrella() else { return };
    let package = read_package(&root.join("spec/fixtures/valid/nacl-crystal")).unwrap();
    let cell = package.state.cell.unwrap();
    let (mut ours, mut theirs) = ewald_pair(0.5, 11.0);
    let a = evaluate(
        &mut ours,
        &package.system,
        &package.state.positions,
        Some(&cell),
    );
    let b = evaluate(
        &mut theirs,
        &package.system,
        &package.state.positions,
        Some(&cell),
    );
    // The forces of a perfect crystal vanish by symmetry; compare them with the size of one
    // nearest-neighbour force, k_e/r0^2 ~ 1.7e3 kJ mol^-1 nm^-1.
    assert_agree("nacl-crystal", &a, &b, 1e-10, 1e3);
    let edge = cell.rows()[0].x / 2.0; // conventional cell
    let k_e = ucmd_core::units::COULOMB_KJMOL_NM_E2;
    let madelung = -a.energy.unwrap() * (edge / 2.0) / (32.0 * k_e);
    assert!(
        ((madelung - 1.7475645946) / 1.7475645946).abs() < 1e-8,
        "Madelung constant from ucmd-ff {madelung}"
    );
}

#[test]
fn ucmd_ff_and_force_field_agree_on_a_periodic_molecular_box() {
    // The reference molecules (EC, EMC, LiPF6, all forms) each put into a cubic test box of
    // 2.6 nm (arbitrary test geometry): exclusions, 1-4 pairs and Ewald together.
    let Some(root) = umbrella() else { return };
    let dir = root.join("tools/oracle/reference/molecules");
    let mut cases: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("terms.json").is_file())
        .collect();
    cases.sort();
    let edge = 2.6;
    let cell = Cell::new(
        [[edge, 0.0, 0.0], [0.0, edge, 0.0], [0.0, 0.0, edge]],
        [true; 3],
    )
    .unwrap();
    for case in cases {
        let mut package = read_package(&case).unwrap();
        package.system.periodicity = [true; 3];
        let (mut ours, mut theirs) = ewald_pair(1.0, 3.0);
        let p = &package.state.positions;
        let a = evaluate(&mut ours, &package.system, p, Some(&cell));
        let b = evaluate(&mut theirs, &package.system, p, Some(&cell));
        assert_agree(&case.display().to_string(), &a, &b, 1e-10, 1e-3);
    }
}

#[test]
#[ignore = "slow: ucmd-ff is O(N²); run in release with --ignored"]
fn ucmd_ff_on_the_electrolyte_box() {
    // tools/oracle/reference/lp57-1M: 1 M LiPF6 in EC:EMC, 7036 atoms, 1.4 nm cutoff, truncated
    // LJ. Against force-field with classic Ewald at OpenMM's α (same real-space sum), and against
    // OpenMM's bonded and LJ terms; OpenMM's Coulomb is PME, so it agrees only to PME accuracy.
    let Some(root) = umbrella() else { return };
    let dir = root.join("tools/oracle/reference/lp57-1M");
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("terms.json")).unwrap()).unwrap();
    let package = read_package(&dir).unwrap();
    let cell = package.state.cell.unwrap();
    let cutoff = meta["cutoff_nm"].as_f64().unwrap();
    let alpha = meta["ewald_alpha_per_nm"].as_f64().unwrap();
    let (mut ours, mut theirs) = ewald_pair(cutoff, alpha);
    let p = &package.state.positions;
    let t0 = std::time::Instant::now();
    let a = evaluate(&mut ours, &package.system, p, Some(&cell));
    eprintln!("ucmd-ff evaluation: {:.1} s", t0.elapsed().as_secs_f64());
    let b = evaluate(&mut theirs, &package.system, p, Some(&cell));
    assert_agree("lp57-1M", &a, &b, 1e-9, 1e-3);
    let r = |k: &str| meta["terms_kjmol"][k].as_f64().unwrap();
    for (name, ours, theirs, tol) in [
        ("bond", term(&a, &["bond"]), r("bond"), 1e-8),
        ("angle", term(&a, &["angle"]), r("angle"), 1e-8),
        ("dihedral", term(&a, &["dihedral"]), r("dihedral"), 1e-8),
        ("lj + lj14", term(&a, &["lj"]), r("lj") + r("lj14"), 1e-8),
        (
            "coulomb (OpenMM: PME)",
            term(&a, &COULOMB_UCMD_FF),
            r("coulomb"),
            1e-5,
        ),
    ] {
        let rel = (ours - theirs).abs() / theirs.abs();
        eprintln!("lp57-1M {name}: ucmd-ff {ours:.6} vs OpenMM {theirs:.6} (rel {rel:.1e})");
        assert!(rel < tol, "lp57-1M {name}: rel {rel:e}");
    }
}

/// The thermo table as columns by name; `lj14` (force-field only) is added into `lj`.
fn thermo_columns(path: &Path) -> std::collections::BTreeMap<String, Vec<f64>> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let header: Vec<String> = lines
        .next()
        .unwrap()
        .trim_start_matches("# ")
        .split('\t')
        .map(str::to_string)
        .collect();
    let mut columns: std::collections::BTreeMap<String, Vec<f64>> =
        header.iter().map(|h| (h.clone(), Vec::new())).collect();
    for line in lines.filter(|l| !l.starts_with('#')) {
        for (h, x) in header.iter().zip(line.split('\t')) {
            columns.get_mut(h).unwrap().push(x.parse().unwrap());
        }
    }
    if let Some(lj14) = columns.remove("term:lj14_kjmol") {
        let lj = columns.get_mut("term:lj_kjmol").unwrap();
        for (a, b) in lj.iter_mut().zip(lj14) {
            *a += b;
        }
    }
    columns
}

#[test]
fn run_toml_switches_between_the_providers() {
    // The same NVE run of water-2 with kind = "ucmd-ff" and kind = "force-field".
    let Some(root) = umbrella() else { return };
    let mut tables = Vec::new();
    for kind in ["ucmd-ff", "force-field"] {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("switch-{kind}"));
        let _ = std::fs::remove_dir_all(&dir);
        let package = dir.join("system.ucsys");
        std::fs::create_dir_all(&package).unwrap();
        for f in ["system.json", "positions.npy"] {
            std::fs::copy(
                root.join("spec/fixtures/valid/water-2").join(f),
                package.join(f),
            )
            .unwrap();
        }
        std::fs::write(
            dir.join("run.toml"),
            format!(
                r#"spec_version = "0.1.0"
system = "system.ucsys"

[run]
integrator = "velocity-verlet"
timestep_fs = 0.25
steps = 200
seed = 1

[[providers]]
kind = "{kind}"

[output.thermo]
file = "thermo.tsv"
every = 20
"#
            ),
        )
        .unwrap();
        let status = Command::new(env!("CARGO_BIN_EXE_ucmd"))
            .arg("run")
            .arg(dir.join("run.toml"))
            .status()
            .unwrap();
        assert!(status.success(), "ucmd run with kind = {kind}");
        tables.push(thermo_columns(&dir.join("out/thermo.tsv")));
    }
    let (ours, theirs) = (&tables[0], &tables[1]);
    assert_eq!(ours["step"].len(), 11);
    // Coulomb is one term in both for a non-periodic system; every column is shared.
    assert_eq!(
        ours.keys().collect::<Vec<_>>(),
        theirs.keys().collect::<Vec<_>>()
    );
    for (name, a) in ours {
        let b = &theirs[name];
        for (x, y) in a.iter().zip(b) {
            let same = (x.is_nan() && y.is_nan()) || (x - y).abs() <= 1e-8 * y.abs().max(1.0);
            assert!(same, "{name}: {a:?} vs {b:?}");
        }
    }
}
