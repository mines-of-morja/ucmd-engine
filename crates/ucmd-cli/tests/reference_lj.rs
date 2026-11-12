//! E1.7 acceptance: the engine's Lennard-Jones on the `argon-864` fixture against the OpenMM
//! oracle references (tools/oracle/reference/argon-864, ADR 0005). Criteria from PLAN.md:
//! energy 1e-8 relative, max|ΔF|/rms(F) ≤ 1e-6, virial 1e-6 relative.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code

use std::path::PathBuf;
use ucmd_force::{EvalRequest, ForceProvider, ForceResult};
use ucmd_io::npy::read_vec3_array;
use ucmd_io::read_package;
use ucmd_providers_classical::{CutoffMode, LennardJones, LennardJonesConfig};

fn umbrella() -> Option<PathBuf> {
    let root = std::env::var_os("UCMD_SPEC_DIR")
        .map(|s| PathBuf::from(s).join(".."))
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."));
    if root.join("tools/oracle/reference/argon-864").is_dir() && root.join("spec/fixtures").is_dir()
    {
        Some(root)
    } else {
        assert!(
            std::env::var_os("UCMD_REQUIRE_SPEC").is_none(),
            "oracle references not found under {}",
            root.display()
        );
        eprintln!("skipping: no oracle references under {}", root.display());
        None
    }
}

#[test]
fn lennard_jones_matches_the_openmm_references() {
    let Some(root) = umbrella() else { return };
    let package = read_package(&root.join("spec/fixtures/valid/argon-864")).unwrap();
    let state = &package.state;
    for (mode, stem) in [
        (CutoffMode::Truncate, "lj-truncate-rc0.85"),
        (CutoffMode::PotentialShift, "lj-potential-shift-rc0.85"),
    ] {
        let reference: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(
                root.join(format!("tools/oracle/reference/argon-864/{stem}.json")),
            )
            .unwrap(),
        )
        .unwrap();
        let ref_forces = read_vec3_array(
            &root.join(format!(
                "tools/oracle/reference/argon-864/{stem}-forces.npy"
            )),
            864,
        )
        .unwrap();
        let mut lj = LennardJones::new(LennardJonesConfig {
            cutoff_nm: 0.85,
            cutoff_mode: mode,
            dispersion_correction: false,
            neighbor_skin_nm: 0.1,
            neighbor_list: true,
        });
        lj.initialize(&package.system).unwrap();
        let mut out = ForceResult::default();
        out.reset(864);
        lj.evaluate(
            &state.positions,
            state.cell.as_ref(),
            &EvalRequest::everything(),
            &mut out,
        )
        .unwrap();

        let e_ref = reference["energy_kjmol"].as_f64().unwrap();
        let e_rel = (out.energy.unwrap() - e_ref).abs() / e_ref.abs();
        let rms = (ref_forces.iter().map(|f| f.length_squared()).sum::<f64>() / 864.0).sqrt();
        let f_rel = out
            .forces
            .iter()
            .zip(&ref_forces)
            .map(|(a, b)| (*a - *b).abs().max_element())
            .fold(0.0, f64::max)
            / rms;
        let w = out.virial.unwrap();
        let w_ref = &reference["virial_kjmol"];
        let mut w_rel = 0.0f64;
        let mut w_scale = 0.0f64;
        for a in 0..3 {
            for b in 0..3 {
                let wr = w_ref[a][b].as_f64().unwrap();
                w_rel = w_rel.max((w.col(b)[a] - wr).abs());
                w_scale = w_scale.max(wr.abs());
            }
        }
        w_rel /= w_scale;
        eprintln!("{stem}: energy {e_rel:.2e}, forces {f_rel:.2e}, virial {w_rel:.2e}");
        assert!(
            e_rel <= 1e-8 && f_rel <= 1e-6 && w_rel <= 1e-6,
            "{stem}: {e_rel} {f_rel} {w_rel}"
        );
        let tail = reference["dispersion_correction_energy_kjmol"]
            .as_f64()
            .unwrap();
        lj = LennardJones::new(LennardJonesConfig {
            cutoff_nm: 0.85,
            cutoff_mode: mode,
            dispersion_correction: true,
            neighbor_skin_nm: 0.1,
            neighbor_list: true,
        });
        lj.initialize(&package.system).unwrap();
        let mut with_tail = ForceResult::default();
        with_tail.reset(864);
        lj.evaluate(
            &state.positions,
            state.cell.as_ref(),
            &EvalRequest::everything(),
            &mut with_tail,
        )
        .unwrap();
        let got_tail = with_tail.energy.unwrap() - out.energy.unwrap();
        assert!(
            (got_tail - tail).abs() <= 1e-9 * tail.abs(),
            "tail {got_tail} vs {tail}"
        );
    }
}
