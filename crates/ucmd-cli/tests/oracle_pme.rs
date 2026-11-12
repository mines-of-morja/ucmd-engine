//! E2.4 and M2a acceptance: the full electrolyte box against OpenMM with PME.
//!
//! The reference (tools/oracle/reference/lp57-1M, made by make_box_references.py) is the builder's
//! 1 M LiPF₆ in EC:EMC 3:7 box (7036 atoms; Lehnert 2025, SI Table S1). It holds OpenMM energies
//! per term and forces for PME with explicitly matched α, grid and order 5, a 1.4 nm cutoff and
//! truncated LJ. Criteria from PLAN.md: energy to 1e-6, forces to 1e-5.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code

use std::path::PathBuf;
use ucmd_core::Vec3;
use ucmd_force::{EvalRequest, ForceProvider, ForceResult};
use ucmd_io::npy::read_vec3_array;
use ucmd_io::read_package;
use ucmd_providers_classical::{Bonded, CoulombMethod, Nonbonded, NonbondedConfig};

fn reference() -> Option<PathBuf> {
    let root = std::env::var_os("UCMD_SPEC_DIR")
        .map(|s| PathBuf::from(s).join(".."))
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."));
    let dir = root.join("tools/oracle/reference/lp57-1M");
    if dir.join("terms.json").is_file() {
        Some(dir)
    } else {
        assert!(
            std::env::var_os("UCMD_REQUIRE_SPEC").is_none(),
            "PME reference not found under {}",
            dir.display()
        );
        None
    }
}

#[test]
fn electrolyte_box_matches_openmm_pme() {
    let Some(dir) = reference() else { return };
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("terms.json")).unwrap()).unwrap();
    let package = read_package(&dir).unwrap();
    let cell = package.state.cell.unwrap();
    let n = package.system.n_atoms();
    let grid: Vec<usize> = meta["pme_grid"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let config = NonbondedConfig {
        cutoff_nm: Some(meta["cutoff_nm"].as_f64().unwrap()),
        coulomb: Some(CoulombMethod::Pme),
        ewald_alpha_per_nm: Some(meta["ewald_alpha_per_nm"].as_f64().unwrap()),
        pme_grid: Some([grid[0], grid[1], grid[2]]),
        pme_order: 5,
        ..NonbondedConfig::default()
    };
    let mut bonded = Bonded::new("bonded");
    let mut nonbonded = Nonbonded::new("nonbonded", config);
    bonded.initialize(&package.system).unwrap();
    nonbonded.initialize(&package.system).unwrap();
    let (mut a, mut b) = (ForceResult::default(), ForceResult::default());
    let request = EvalRequest::everything();
    bonded
        .evaluate(&package.state.positions, Some(&cell), &request, &mut a)
        .unwrap();
    nonbonded
        .evaluate(&package.state.positions, Some(&cell), &request, &mut b)
        .unwrap();

    let reference = &meta["terms_kjmol"];
    let mut worst = 0.0f64;
    for (term, value) in a.energy_terms.iter().chain(&b.energy_terms) {
        let theirs = reference[term.as_str()].as_f64().unwrap();
        let rel = (value - theirs).abs() / theirs.abs().max(1.0);
        worst = worst.max(rel);
        assert!(
            rel < 1e-6,
            "{term}: engine {value:.8} vs OpenMM {theirs:.8} (rel {rel:.2e})"
        );
    }
    let total = a.energy.unwrap() + b.energy.unwrap();
    let total_ref = reference["total"].as_f64().unwrap();
    let rel_total = (total - total_ref).abs() / total_ref.abs();
    assert!(rel_total < 1e-6, "total {total} vs {total_ref}");

    let max_error = |ours: &[Vec3], theirs: &[Vec3]| {
        let scale = theirs
            .iter()
            .map(|f| f.abs().max_element())
            .fold(0.0, f64::max);
        ours.iter()
            .zip(theirs)
            .map(|(x, y)| (*x - *y).abs().max_element())
            .fold(0.0, f64::max)
            / scale
    };
    let forces: Vec<Vec3> = a
        .forces
        .iter()
        .zip(&b.forces)
        .map(|(x, y)| *x + *y)
        .collect();
    let total_forces = read_vec3_array(&dir.join("forces-total.npy"), n).unwrap();
    let f_total = max_error(&forces, &total_forces);
    assert!(
        f_total < 1e-5,
        "total forces: max error {f_total:.2e} of max|F|"
    );
    eprintln!(
        "lp57-1M ({n} atoms, PME grid {grid:?}): energy terms within {worst:.1e}, total {rel_total:.1e}; \
         forces within {f_total:.1e} of max|F|"
    );
}
