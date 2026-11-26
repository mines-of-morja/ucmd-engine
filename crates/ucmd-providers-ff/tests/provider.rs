//! The `ucmd-ff` provider against the provider test kit (spec/force-provider.md §6).
//!
//! Systems:
//! - two SPC/Fw waters, with the parameters and positions of the fixture water-2 (pre-approved);
//! - two copies of a six-atom molecule with every bonded form and 1-4 pairs. Its parameters and
//!   geometry are arbitrary test values, not a physical force field.
//!
//! Each system is checked non-periodic and periodic, in a triclinic test cell.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code

use ucmd_core::{
    Angle, AngleForm, AtomTemplate, AtomType, Bond, BondForm, Cell, CombinationRule, ForceField,
    MoleculeType, PairOverride, System, Torsion, TorsionForm, Vec3,
};
use ucmd_force::testkit::{
    FdOptions, determinism_check, energy_terms_check, fd_force_check, fd_virial_check,
    momentum_check,
};
use ucmd_force::{EvalRequest, ForceError, ForceProvider, ForceResult};
use ucmd_providers_ff::{UcmdFf, UcmdFfConfig};

fn atom(name: &str, element: u8, mass_amu: f64, charge: f64, atom_type: usize) -> AtomTemplate {
    AtomTemplate {
        name: name.into(),
        element,
        mass_amu,
        charge_e: Some(charge),
        atom_type: Some(atom_type),
    }
}

fn harmonic_bond(i: usize, j: usize, r0_nm: f64, k_kjmol_nm2: f64) -> Bond {
    Bond {
        atoms: [i, j],
        form: Some(BondForm::Harmonic { r0_nm, k_kjmol_nm2 }),
    }
}

fn harmonic_angle(atoms: [usize; 3], theta0_rad: f64, k_kjmol_rad2: f64) -> Angle {
    Angle {
        atoms,
        form: Some(AngleForm::Harmonic {
            theta0_rad,
            k_kjmol_rad2,
        }),
    }
}

/// spec/fixtures/valid/water-2: SPC/Fw (Wu, Tepper, Voth 2006).
fn water2(periodicity: [bool; 3]) -> System {
    let water = MoleculeType {
        name: "SPCFw".into(),
        formal_charge_e: 0,
        atoms: vec![
            atom("OW", 8, 15.9994, -0.82, 0),
            atom("HW1", 1, 1.008, 0.41, 1),
            atom("HW2", 1, 1.008, 0.41, 1),
        ],
        bonds: vec![
            harmonic_bond(0, 1, 0.1012, 443153.3808),
            harmonic_bond(0, 2, 0.1012, 443153.3808),
        ],
        angles: vec![harmonic_angle([1, 0, 2], 113.24f64.to_radians(), 317.5656)],
        dihedrals: vec![],
        impropers: vec![],
        constraints: vec![],
    };
    let ff = ForceField {
        name: "SPC/Fw".into(),
        combination_rule: CombinationRule::LorentzBerthelot,
        scale_14_lj: 0.5,
        scale_14_coulomb: 0.5,
        atom_types: vec![
            AtomType {
                name: "OW".into(),
                sigma_nm: 0.3165492,
                epsilon_kjmol: 0.6502994552,
            },
            AtomType {
                name: "HW".into(),
                sigma_nm: 0.0,
                epsilon_kjmol: 0.0,
            },
        ],
        pair_overrides: vec![],
    };
    System::new("water-2", vec![water], &[(0, 2)], Some(ff), periodicity).unwrap()
}

fn water2_positions() -> Vec<Vec3> {
    [
        [
            0.0014879054633459199,
            0.00023628648429753563,
            -0.000789356853724116,
        ],
        [
            0.08393883634235291,
            0.05485979869227756,
            0.0018390991051459836,
        ],
        [
            -0.08620544326799455,
            0.0553171257097342,
            0.001786700948775454,
        ],
        [
            0.29147061232497856,
            0.029893536587790245,
            0.008219852201468345,
        ],
        [
            0.37580409511058643,
            0.059451909021630854,
            0.057423091208704626,
        ],
        [
            0.20354470422243195,
            0.05944960237012237,
            0.05865817528988138,
        ],
    ]
    .map(Vec3::from_array)
    .to_vec()
}

/// Every bonded form once or twice; 1-4 pairs 0-3, 0-5, 4-3, 4-5 (arbitrary test values).
fn all_forms(periodicity: [bool; 3]) -> System {
    let carbon = |q| atom("C", 6, 12.0, q, 0);
    let hydrogen = |q| atom("H", 1, 1.0, q, 1);
    let torsion = |atoms, form| Torsion {
        atoms,
        form: Some(form),
    };
    let molecule = MoleculeType {
        name: "allforms".into(),
        formal_charge_e: 0,
        atoms: vec![
            carbon(-0.3),
            carbon(0.2),
            carbon(0.1),
            carbon(-0.25),
            hydrogen(0.15),
            hydrogen(0.1),
        ],
        bonds: vec![
            harmonic_bond(0, 1, 0.15, 2.0e5),
            harmonic_bond(1, 2, 0.153, 2.0e5),
            harmonic_bond(2, 3, 0.152, 2.0e5),
            harmonic_bond(1, 4, 0.11, 2.0e5),
            harmonic_bond(2, 5, 0.109, 2.0e5),
        ],
        angles: [
            [0, 1, 2],
            [1, 2, 3],
            [0, 1, 4],
            [4, 1, 2],
            [1, 2, 5],
            [5, 2, 3],
        ]
        .map(|a| harmonic_angle(a, 1.9, 300.0))
        .to_vec(),
        dihedrals: vec![
            torsion(
                [0, 1, 2, 3],
                TorsionForm::Periodic {
                    k_kjmol: 5.0,
                    n: 3,
                    phi0_rad: 0.2,
                },
            ),
            torsion(
                [4, 1, 2, 5],
                TorsionForm::OplsFourier {
                    v_kjmol: [3.0, -1.5, 2.0, 0.5],
                },
            ),
            torsion(
                [0, 1, 2, 5],
                TorsionForm::RyckaertBellemans {
                    c_kjmol: [1.0, 2.0, -1.0, 0.5, 0.3, -0.2],
                },
            ),
        ],
        impropers: vec![
            torsion(
                [1, 0, 2, 4],
                TorsionForm::Harmonic {
                    xi0_rad: 0.3,
                    k_kjmol_rad2: 40.0,
                },
            ),
            torsion(
                [2, 1, 3, 5],
                TorsionForm::Periodic {
                    k_kjmol: 4.0,
                    n: 2,
                    phi0_rad: std::f64::consts::PI,
                },
            ),
        ],
        constraints: vec![],
    };
    let ff = ForceField {
        name: "test".into(),
        combination_rule: CombinationRule::Geometric,
        scale_14_lj: 0.5,
        scale_14_coulomb: 0.8333333333333334,
        atom_types: vec![
            AtomType {
                name: "A".into(),
                sigma_nm: 0.3,
                epsilon_kjmol: 0.5,
            },
            AtomType {
                name: "B".into(),
                sigma_nm: 0.25,
                epsilon_kjmol: 0.2,
            },
        ],
        pair_overrides: vec![],
    };
    System::new("allforms", vec![molecule], &[(0, 2)], Some(ff), periodicity).unwrap()
}

fn all_forms_positions() -> Vec<Vec3> {
    let one = [
        [0.000, 0.000, 0.000],
        [0.150, 0.020, 0.010],
        [0.210, 0.160, 0.030],
        [0.360, 0.180, 0.090],
        [0.170, -0.060, 0.120],
        [0.190, 0.230, -0.090],
    ];
    let mut p: Vec<Vec3> = one.iter().map(|r| Vec3::from_array(*r)).collect();
    for (k, r) in one.iter().enumerate() {
        let w = 0.01 * ((k * 7) % 5) as f64 - 0.02;
        p.push(Vec3::new(
            r[0] + 0.45 + w,
            r[1] + 0.35 - w,
            r[2] + 0.30 + 0.5 * w,
        ));
    }
    p
}

/// Triclinic test cell in reduced form (arbitrary test geometry), nm.
fn cell() -> Cell {
    Cell::new(
        [[1.3, 0.0, 0.0], [0.2, 1.2, 0.0], [-0.3, 0.1, 1.1]],
        [true; 3],
    )
    .unwrap()
}

fn periodic_config() -> UcmdFfConfig {
    UcmdFfConfig {
        cutoff_nm: Some(0.48),
        ewald_tolerance: 1e-10,
        ..UcmdFfConfig::default()
    }
}

fn provider(system: &System, config: UcmdFfConfig) -> UcmdFf {
    let mut p = UcmdFf::new("ucmd-ff", config);
    p.initialize(system).unwrap();
    p
}

fn conformance(system: &System, positions: &[Vec3], cell: Option<&Cell>, config: UcmdFfConfig) {
    let label = &system.name;
    let mut p = provider(system, config);
    let f = fd_force_check(&mut p, positions, cell, &FdOptions::default()).unwrap();
    assert!(
        f.max_rel_error <= 1e-5,
        "{label}: forces {f:?} (cell {})",
        cell.is_some()
    );
    let w = fd_virial_check(&mut p, positions, cell, 1e-6).unwrap();
    assert!(
        w.max_rel_error <= 1e-5 && w.asymmetry <= 1e-10,
        "{label}: virial {w:?}"
    );
    let terms = energy_terms_check(&mut p, positions, cell).unwrap();
    assert!(terms <= 1e-10, "{label}: energy terms off by {terms:e}");
    let momentum = momentum_check(&mut p, positions, cell).unwrap();
    assert!(momentum <= 1e-8, "{label}: momentum {momentum:e}");
    assert!(
        determinism_check(&mut p, positions, cell).unwrap(),
        "{label}: not deterministic"
    );
}

#[test]
fn testkit_water2_nonperiodic() {
    conformance(
        &water2([false; 3]),
        &water2_positions(),
        None,
        UcmdFfConfig::default(),
    );
}

#[test]
fn testkit_water2_periodic() {
    conformance(
        &water2([true; 3]),
        &water2_positions(),
        Some(&cell()),
        periodic_config(),
    );
}

#[test]
fn testkit_all_forms_nonperiodic() {
    conformance(
        &all_forms([false; 3]),
        &all_forms_positions(),
        None,
        UcmdFfConfig::default(),
    );
}

#[test]
fn testkit_all_forms_periodic() {
    conformance(
        &all_forms([true; 3]),
        &all_forms_positions(),
        Some(&cell()),
        periodic_config(),
    );
}

#[test]
fn terms_and_virial_are_reported_when_requested() {
    let mut p = provider(&all_forms([true; 3]), periodic_config());
    let mut out = ForceResult::default();
    p.evaluate(
        &all_forms_positions(),
        Some(&cell()),
        &EvalRequest::everything(),
        &mut out,
    )
    .unwrap();
    let names: Vec<&str> = out.energy_terms.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        [
            "bond",
            "angle",
            "dihedral",
            "improper",
            "lj",
            "coulomb_real",
            "coulomb_reciprocal",
            "coulomb_self",
            "coulomb_correction"
        ]
    );
    assert!(out.virial.is_some() && out.energy.is_some());
    let forces_only = EvalRequest {
        need_energy: false,
        need_virial: false,
        ..EvalRequest::everything()
    };
    p.evaluate(
        &all_forms_positions(),
        Some(&cell()),
        &forces_only,
        &mut out,
    )
    .unwrap();
    assert!(out.energy.is_none() && out.virial.is_none() && out.energy_terms.is_empty());
    assert_eq!(out.forces.len(), 12);
}

#[test]
fn unsupported_systems_and_settings_are_errors() {
    let unsupported = |r: Result<(), ForceError>| matches!(r, Err(ForceError::Unsupported(_)));
    let invalid = |r: Result<(), ForceError>| matches!(r, Err(ForceError::InvalidInput(_)));
    // No force field.
    let mut bare = water2([false; 3]);
    bare.forcefield = None;
    assert!(unsupported(
        UcmdFf::new("x", UcmdFfConfig::default()).initialize(&bare)
    ));
    // Partially periodic.
    let slab = water2([true, true, false]);
    assert!(unsupported(
        UcmdFf::new("x", periodic_config()).initialize(&slab)
    ));
    // Periodic without a cutoff; non-periodic with one.
    assert!(invalid(
        UcmdFf::new("x", UcmdFfConfig::default()).initialize(&water2([true; 3]))
    ));
    assert!(unsupported(
        UcmdFf::new("x", periodic_config()).initialize(&water2([false; 3]))
    ));
    // A harmonic dihedral is not a form of the spec.
    let mut odd = all_forms([false; 3]);
    odd.molecule_types[0].dihedrals[0].form = Some(TorsionForm::Harmonic {
        xi0_rad: 0.0,
        k_kjmol_rad2: 1.0,
    });
    assert!(unsupported(
        UcmdFf::new("x", UcmdFfConfig::default()).initialize(&odd)
    ));
    // Evaluation errors propagate: a cell for a non-periodic system, wrong length, LJ cutoff too long.
    let mut p = provider(&water2([false; 3]), UcmdFfConfig::default());
    let mut out = ForceResult::default();
    let req = EvalRequest::everything();
    assert!(
        p.evaluate(&water2_positions(), Some(&cell()), &req, &mut out)
            .is_err()
    );
    assert!(
        p.evaluate(&water2_positions()[..5], None, &req, &mut out)
            .is_err()
    );
    let long = UcmdFfConfig {
        cutoff_nm: Some(0.6),
        ..periodic_config()
    };
    let mut p = provider(&water2([true; 3]), long);
    assert!(matches!(
        p.evaluate(&water2_positions(), Some(&cell()), &req, &mut out),
        Err(ForceError::InvalidInput(_))
    ));
}

#[test]
fn pair_overrides_are_refused_not_ignored() {
    // ucmd-ff v0.1.0 has no per-pair table; ignoring the override would compute other physics
    let mut system = all_forms([true, true, true]);
    let ff = system.forcefield.as_mut().unwrap();
    ff.pair_overrides.push(PairOverride {
        types: [0, 1],
        sigma_nm: 0.2,
        epsilon_kjmol: 0.1,
    });
    let mut p = UcmdFf::new("ucmd-ff", periodic_config());
    let error = p
        .initialize(&system)
        .expect_err("a package with pair overrides must be refused");
    assert!(error.to_string().contains("pair overrides"), "{error}");
}
