//! E2.3 acceptance: classic Ewald gives the Madelung constant of rock salt.
//!
//! The spec fixture `nacl-crystal` is 2×2×2 conventional cells (a = 0.564 nm) of point charges
//! ±1 e. E = −N_pairs · M · k_e / r₀ with r₀ = a/2. The NaCl Madelung constant referred to the
//! nearest-neighbour distance is M = 1.747 564 594 633 18…; PLAN.md rounds it to 1.747565.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code

use std::path::PathBuf;
use ucmd_core::units::COULOMB_KJMOL_NM_E2;
use ucmd_force::{EvalRequest, ForceProvider, ForceResult};
use ucmd_io::read_package;
use ucmd_providers_classical::{CoulombMethod, Nonbonded, NonbondedConfig};

const MADELUNG_NACL: f64 = 1.747_564_594_633_18;

fn fixture() -> Option<PathBuf> {
    let dir = std::env::var_os("UCMD_SPEC_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../spec"));
    let path = dir.join("fixtures/valid/nacl-crystal");
    if path.is_dir() {
        Some(path)
    } else {
        assert!(
            std::env::var_os("UCMD_REQUIRE_SPEC").is_none(),
            "no spec fixtures at {}",
            dir.display()
        );
        None
    }
}

#[test]
fn ewald_reproduces_the_madelung_constant() {
    let Some(path) = fixture() else { return };
    let package = read_package(&path).unwrap();
    let cell = package.state.cell.unwrap();
    let edge = cell.rows()[0].length();
    let r0 = edge / 4.0; // two conventional cells per edge, r₀ = a/2
    let n_pairs = package.system.n_atoms() as f64 / 2.0;
    let config = NonbondedConfig {
        cutoff_nm: Some(0.55),
        neighbor_skin_nm: 0.0,
        coulomb: Some(CoulombMethod::Ewald),
        ewald_alpha_per_nm: Some(11.0),
        ewald_tolerance: 1e-16,
        ..NonbondedConfig::default()
    };
    let mut ewald = Nonbonded::new("ewald", config);
    ewald.initialize(&package.system).unwrap();
    let mut out = ForceResult::default();
    ewald
        .evaluate(
            &package.state.positions,
            Some(&cell),
            &EvalRequest::everything(),
            &mut out,
        )
        .unwrap();
    let energy = out.energy.unwrap();
    let madelung = -energy * r0 / (n_pairs * COULOMB_KJMOL_NM_E2);
    let rel = (madelung - MADELUNG_NACL).abs() / MADELUNG_NACL;
    eprintln!("Madelung constant from Ewald: {madelung:.12} (rel. error {rel:.1e})");
    assert!(rel < 1e-8, "{madelung} vs {MADELUNG_NACL}");
    // On the perfect lattice every force vanishes by symmetry.
    let max_force = out.forces.iter().map(|f| f.length()).fold(0.0, f64::max);
    assert!(
        max_force < 1e-6 * COULOMB_KJMOL_NM_E2 / (r0 * r0),
        "max |F| = {max_force}"
    );
}
