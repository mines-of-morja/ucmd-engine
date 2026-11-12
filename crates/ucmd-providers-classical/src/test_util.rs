//! Small systems for the providers' unit tests.

use ucmd_core::{
    Angle, AngleForm, AtomTemplate, AtomType, Bond, BondForm, Cell, CombinationRule, ForceField,
    MoleculeType, System, Torsion, TorsionForm, Vec3,
};

/// A reproducible generator for test geometries (xorshift64*).
pub struct Rng(pub u64);

impl Rng {
    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn carbon(charge: f64) -> AtomTemplate {
    AtomTemplate {
        name: "C".into(),
        element: 6,
        mass_amu: 12.011,
        charge_e: Some(charge),
        atom_type: Some(0),
    }
}

fn molecule(name: &str, n: usize, bonds: &[[usize; 2]]) -> MoleculeType {
    MoleculeType {
        name: name.into(),
        formal_charge_e: 0,
        atoms: (0..n).map(|_| carbon(0.0)).collect(),
        bonds: bonds
            .iter()
            .map(|&atoms| Bond {
                atoms,
                form: Some(BondForm::Harmonic {
                    r0_nm: 0.15,
                    k_kjmol_nm2: 2.5e5,
                }),
            })
            .collect(),
        angles: Vec::new(),
        dihedrals: Vec::new(),
        impropers: Vec::new(),
        constraints: Vec::new(),
    }
}

/// A linear chain 0-1-…-(n−1) with bonds only.
pub fn chain_molecule(n: usize) -> MoleculeType {
    let bonds: Vec<[usize; 2]> = (0..n - 1).map(|i| [i, i + 1]).collect();
    molecule("chain", n, &bonds)
}

/// A ring of n atoms with bonds only.
pub fn ring_molecule(n: usize) -> MoleculeType {
    let bonds: Vec<[usize; 2]> = (0..n)
        .map(|i| [i.min((i + 1) % n), i.max((i + 1) % n)])
        .collect();
    molecule("ring", n, &bonds)
}

/// Six atoms, chain 0-1-2-3-4 with atom 5 on atom 1, carrying every bonded form of the spec:
/// several periodic dihedrals on the same atoms, Ryckaert–Bellemans, OPLS Fourier, a periodic
/// and a harmonic improper on centre 1. Charges are non-zero and sum to zero. The parameters are
/// test numbers, not a force field.
pub fn all_forms_molecule() -> MoleculeType {
    let mut m = molecule("allforms", 6, &[[0, 1], [1, 2], [2, 3], [3, 4], [1, 5]]);
    for (bond, (r0, k)) in m.bonds.iter_mut().zip([
        (0.153, 2.2e5),
        (0.141, 2.7e5),
        (0.123, 4.8e5),
        (0.109, 2.8e5),
        (0.150, 3.1e5),
    ]) {
        bond.form = Some(BondForm::Harmonic {
            r0_nm: r0,
            k_kjmol_nm2: k,
        });
    }
    let charges = [0.3, -0.25, 0.15, -0.4, 0.1, 0.1];
    for (atom, q) in m.atoms.iter_mut().zip(charges) {
        atom.charge_e = Some(q);
    }
    m.angles = [
        ([0, 1, 2], 110.0, 400.0),
        ([0, 1, 5], 108.5, 350.0),
        ([2, 1, 5], 120.0, 500.0),
        ([1, 2, 3], 116.9, 690.0),
        ([2, 3, 4], 109.5, 290.0),
    ]
    .into_iter()
    .map(|(atoms, theta0, k)| Angle {
        atoms,
        form: Some(AngleForm::Harmonic {
            theta0_rad: f64::to_radians(theta0),
            k_kjmol_rad2: k,
        }),
    })
    .collect();
    let periodic = |k: f64, n: u32, phi0: f64| {
        Some(TorsionForm::Periodic {
            k_kjmol: k,
            n,
            phi0_rad: f64::to_radians(phi0),
        })
    };
    m.dihedrals = vec![
        Torsion {
            atoms: [0, 1, 2, 3],
            form: periodic(2.5, 1, 30.0),
        },
        Torsion {
            atoms: [0, 1, 2, 3],
            form: periodic(1.2, 3, 0.0),
        },
        Torsion {
            atoms: [1, 2, 3, 4],
            form: Some(TorsionForm::RyckaertBellemans {
                c_kjmol: [9.28, 12.16, -13.12, -3.06, 26.24, -31.5],
            }),
        },
        Torsion {
            atoms: [5, 1, 2, 3],
            form: Some(TorsionForm::OplsFourier {
                v_kjmol: [3.0, -1.5, 0.8, 0.4],
            }),
        },
    ];
    m.impropers = vec![
        Torsion {
            atoms: [0, 1, 2, 5],
            form: periodic(43.9, 2, 180.0),
        },
        Torsion {
            atoms: [2, 1, 5, 0],
            form: Some(TorsionForm::Harmonic {
                xi0_rad: f64::to_radians(25.0),
                k_kjmol_rad2: 80.0,
            }),
        },
    ];
    m
}

/// A one-type force field (geometric mixing, OPLS 1-4 scales).
pub fn test_forcefield() -> ForceField {
    ForceField {
        name: "test".into(),
        combination_rule: CombinationRule::Geometric,
        scale_14_lj: 0.5,
        scale_14_coulomb: 0.5,
        atom_types: vec![AtomType {
            name: "C".into(),
            sigma_nm: 0.35,
            epsilon_kjmol: 0.276,
        }],
    }
}

/// A parameterized system of the given molecule types and components.
pub fn system_of(
    types: Vec<MoleculeType>,
    components: &[(usize, usize)],
    periodic: [bool; 3],
) -> System {
    System::new("test", types, components, Some(test_forcefield()), periodic)
        .expect("valid test system")
}

/// A cubic box for a system, or none.
pub trait WithBox {
    /// The system together with a cubic cell of edge `edge_nm` (`None`: non-periodic).
    fn with_box(self, edge_nm: Option<f64>) -> (System, Option<Cell>);
}

impl WithBox for System {
    fn with_box(self, edge_nm: Option<f64>) -> (System, Option<Cell>) {
        let cell = edge_nm.map(|l| {
            Cell::new([[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]], [true; 3]).expect("cubic cell")
        });
        (self, cell)
    }
}

/// A generic geometry: each molecule a perturbed zigzag (no special angles), molecules spread
/// on a grid 0.9 nm apart. Atoms after the chain go to random positions near their bonded atom.
pub fn positions_for(system: &System, seed: u64) -> Vec<Vec3> {
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut out = Vec::with_capacity(system.n_atoms());
    for (k, instance) in system.instances.iter().enumerate() {
        let mtype = &system.molecule_types[instance.molecule_type];
        let origin = Vec3::new(
            0.4 + 0.9 * (k % 3) as f64,
            0.4 + 0.9 * ((k / 3) % 3) as f64,
            0.4 + 0.9 * (k / 9) as f64,
        );
        let mut placed: Vec<Vec3> = Vec::with_capacity(mtype.atoms.len());
        for a in 0..mtype.atoms.len() {
            let jitter = Vec3::new(rng.uniform(), rng.uniform(), rng.uniform()) - Vec3::splat(0.5);
            let partner = mtype.bonds.iter().find_map(|b| match b.atoms {
                [x, y] if y == a && x < a => Some(x),
                [x, y] if x == a && y < a => Some(y),
                _ => None,
            });
            let p = match partner {
                Some(p) if p + 1 == a => {
                    // zigzag along x
                    let up = if a % 2 == 0 { 1.0 } else { -1.0 };
                    placed[p] + Vec3::new(0.125, 0.07 * up, 0.02) + 0.03 * jitter
                }
                Some(p) => placed[p] + Vec3::new(0.0, 0.03, 0.14) + 0.03 * jitter,
                None => origin + 0.05 * jitter,
            };
            placed.push(p);
        }
        out.extend(placed);
    }
    out
}
