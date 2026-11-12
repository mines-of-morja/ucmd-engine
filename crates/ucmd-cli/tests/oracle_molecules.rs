//! E2.1/E2.2 acceptance: molecules against the OpenMM oracle, term by term.
//!
//! The references (tools/oracle/reference/molecules, made by make_molecule_references.py) are
//! EC, EMC and Li⁺ + PF₆⁻ from the builder's library, plus a synthetic molecule with every
//! bonded form and asymmetric phases. Criteria from PLAN.md: each energy term to 1e-8
//! relative; forces to 1e-8 of the largest reference force.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code

use std::path::{Path, PathBuf};
use ucmd_core::Vec3;
use ucmd_force::{EvalRequest, ForceProvider, ForceResult};
use ucmd_io::npy::read_vec3_array;
use ucmd_io::read_package;
use ucmd_providers_classical::{Bonded, Nonbonded, NonbondedConfig};

fn references() -> Option<PathBuf> {
    let root = std::env::var_os("UCMD_SPEC_DIR")
        .map(|s| PathBuf::from(s).join(".."))
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."));
    let dir = root.join("tools/oracle/reference/molecules");
    if dir.is_dir() {
        Some(dir)
    } else {
        assert!(
            std::env::var_os("UCMD_REQUIRE_SPEC").is_none(),
            "molecule references not found under {}",
            dir.display()
        );
        eprintln!("skipping: no molecule references under {}", dir.display());
        None
    }
}

fn cases(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("terms.json").is_file())
        .collect();
    out.sort();
    assert!(
        out.len() >= 4,
        "expected EC, EMC, LiPF6 and allforms in {}",
        dir.display()
    );
    out
}

fn reference_terms(case: &Path) -> serde_json::Value {
    let text = std::fs::read_to_string(case.join("terms.json")).unwrap();
    serde_json::from_str::<serde_json::Value>(&text).unwrap()["terms_kjmol"].clone()
}

fn reference_forces(case: &Path, terms: &[&str], n: usize) -> Vec<Vec3> {
    let mut sum = vec![Vec3::ZERO; n];
    for term in terms {
        let f = read_vec3_array(&case.join(format!("forces-{term}.npy")), n).unwrap();
        for (s, x) in sum.iter_mut().zip(f) {
            *s += x;
        }
    }
    sum
}

fn check(case: &Path, out: &ForceResult, terms: &[&str]) {
    let reference = reference_terms(case);
    let mut worst_energy = 0.0f64;
    for term in terms {
        let ours = out
            .energy_terms
            .iter()
            .find(|(name, _)| name == term)
            .unwrap_or_else(|| panic!("{}: no term {term}", case.display()))
            .1;
        let theirs = reference[*term].as_f64().unwrap();
        let rel = (ours - theirs).abs() / theirs.abs().max(1e-6);
        worst_energy = worst_energy.max(rel);
        assert!(
            rel < 1e-8,
            "{} {term}: engine {ours:.12} vs OpenMM {theirs:.12} (rel {rel:.2e})",
            case.display()
        );
    }
    let n = out.forces.len();
    let reference = reference_forces(case, terms, n);
    let scale = reference
        .iter()
        .map(|f| f.abs().max_element())
        .fold(1e-3, f64::max);
    let worst = out
        .forces
        .iter()
        .zip(&reference)
        .map(|(a, b)| (*a - *b).abs().max_element())
        .fold(0.0, f64::max)
        / scale;
    assert!(
        worst < 1e-8,
        "{} forces {terms:?}: max error {worst:.2e} of max|F|",
        case.display()
    );
    eprintln!(
        "{}: energy terms within {worst_energy:.1e} relative, forces within {worst:.1e} of max|F|",
        case.file_name().unwrap().to_string_lossy()
    );
}

#[test]
fn bonded_terms_match_openmm() {
    let Some(dir) = references() else { return };
    for case in cases(&dir) {
        let package = read_package(&case).unwrap();
        let mut bonded = Bonded::new("bonded");
        bonded.initialize(&package.system).unwrap();
        let mut out = ForceResult::default();
        bonded
            .evaluate(
                &package.state.positions,
                None,
                &EvalRequest::everything(),
                &mut out,
            )
            .unwrap();
        check(&case, &out, &["bond", "angle", "dihedral", "improper"]);
    }
}

#[test]
fn nonbonded_terms_match_openmm() {
    let Some(dir) = references() else { return };
    for case in cases(&dir) {
        let package = read_package(&case).unwrap();
        let mut nonbonded = Nonbonded::new("nonbonded", NonbondedConfig::default());
        nonbonded.initialize(&package.system).unwrap();
        let mut out = ForceResult::default();
        nonbonded
            .evaluate(
                &package.state.positions,
                None,
                &EvalRequest::everything(),
                &mut out,
            )
            .unwrap();
        check(&case, &out, &["lj", "lj14", "coulomb"]);
    }
}
