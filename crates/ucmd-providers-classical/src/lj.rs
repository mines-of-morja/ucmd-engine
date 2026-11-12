//! Lennard-Jones 12-6 (PLAN.md E1.7): E = 4ε_ij[(σ_ij/r)¹² − (σ_ij/r)⁶] within a cutoff.
//!
//! Pair parameters come from the atom types via the force field's combination rule. Exclusions
//! arrive with bonded terms in E2.2; until then systems with bonds are rejected rather than
//! computed wrongly.

use crate::neighbor::NeighborList;
use serde::Deserialize;
use std::f64::consts::PI;
use ucmd_core::{Cell, Mat3, System, Vec3};
use ucmd_force::{Capabilities, EvalRequest, ForceError, ForceProvider, ForceResult, outer};

/// How the potential is cut off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CutoffMode {
    /// Plain truncation at the cutoff (LAMMPS `lj/cut` default, used by Lehnert 2025).
    Truncate,
    /// The energy is shifted by −E(r_c), so it is continuous at the cutoff; forces unchanged.
    PotentialShift,
}

fn default_skin() -> f64 {
    0.1
}

fn default_true() -> bool {
    true
}

/// `[[providers]]` keys of `kind = "lennard-jones"`.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LennardJonesConfig {
    /// Cutoff radius in nm.
    pub cutoff_nm: f64,
    /// Cutoff treatment.
    pub cutoff_mode: CutoffMode,
    /// Add the long-range tail correction to energy and virial (homogeneous fluid, g(r) = 1).
    #[serde(default)]
    pub dispersion_correction: bool,
    /// Verlet skin in nm.
    #[serde(default = "default_skin")]
    pub neighbor_skin_nm: f64,
    /// Use the neighbour list (false = all pairs; for tests).
    #[serde(default = "default_true")]
    pub neighbor_list: bool,
}

/// The Lennard-Jones provider.
pub struct LennardJones {
    config: LennardJonesConfig,
    n_types: usize,
    type_of: Vec<usize>,
    /// (σ, ε, energy shift) per ordered type pair, index a·n_types + b.
    pairs: Vec<(f64, f64, f64)>,
    tail_energy_v: f64,
    tail_virial_v: f64,
    list: NeighborList,
}

impl LennardJones {
    /// A provider with the given settings; parameters are read in `initialize`.
    pub fn new(config: LennardJonesConfig) -> Self {
        Self {
            list: NeighborList::new(config.cutoff_nm, config.neighbor_skin_nm),
            config,
            n_types: 0,
            type_of: Vec::new(),
            pairs: Vec::new(),
            tail_energy_v: 0.0,
            tail_virial_v: 0.0,
        }
    }

    /// The neighbour list (for statistics).
    pub fn neighbor_list(&self) -> &NeighborList {
        &self.list
    }
}

impl ForceProvider for LennardJones {
    fn name(&self) -> &str {
        "lennard-jones"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            periodic: true,
            virial: true,
            energy_terms: false,
            deterministic: true,
        }
    }

    fn initialize(&mut self, system: &System) -> Result<(), ForceError> {
        let rc = self.config.cutoff_nm;
        if !(rc.is_finite() && rc > 0.0 && self.config.neighbor_skin_nm >= 0.0) {
            return Err(ForceError::InvalidInput(
                "cutoff_nm must be > 0 and neighbor_skin_nm >= 0".into(),
            ));
        }
        if system.has_bonds() {
            return Err(ForceError::Unsupported(
                "Lennard-Jones without exclusions: systems with bonds need E2.2 (exclusions and 1-4 pairs)".into(),
            ));
        }
        let ff = system.forcefield.as_ref().ok_or_else(|| {
            ForceError::InvalidInput("Lennard-Jones needs a parameterized package".into())
        })?;
        let types = system
            .atom_types()
            .ok_or_else(|| ForceError::InvalidInput("every atom needs an atom type".into()))?;
        self.n_types = ff.atom_types.len();
        self.type_of = types.to_vec();
        self.pairs.clear();
        for a in &ff.atom_types {
            for b in &ff.atom_types {
                let (sigma, epsilon) = ff.combination_rule.combine(a, b);
                let shift = match self.config.cutoff_mode {
                    CutoffMode::PotentialShift => lj_energy(sigma, epsilon, rc * rc),
                    CutoffMode::Truncate => 0.0,
                };
                self.pairs.push((sigma, epsilon, shift));
            }
        }
        // tail corrections, multiplied by V: E_tail = (8π/3V) Σ_ab N_a N_b ε σ³ (x⁹/3 − x³),
        // W_tail = I·(16π/3V) Σ_ab N_a N_b ε σ³ (2x⁹/3 − x³), x = σ/r_c (virial route).
        let mut counts = vec![0.0f64; self.n_types];
        for &t in &self.type_of {
            counts[t] += 1.0;
        }
        let (mut e, mut w) = (0.0, 0.0);
        for a in 0..self.n_types {
            for b in 0..self.n_types {
                let (sigma, epsilon, _) = self.pairs[a * self.n_types + b];
                let x = sigma / rc;
                let x3 = x * x * x;
                let x9 = x3 * x3 * x3;
                let nn = counts[a] * counts[b] * epsilon * sigma * sigma * sigma;
                e += nn * (x9 / 3.0 - x3);
                w += nn * (2.0 * x9 / 3.0 - x3);
            }
        }
        self.tail_energy_v = 8.0 * PI / 3.0 * e;
        self.tail_virial_v = 16.0 * PI / 3.0 * w;
        Ok(())
    }

    fn evaluate(
        &mut self,
        positions: &[Vec3],
        cell: Option<&Cell>,
        request: &EvalRequest,
        out: &mut ForceResult,
    ) -> Result<(), ForceError> {
        let rc2 = self.config.cutoff_nm * self.config.cutoff_nm;
        if let Some(cell) = cell {
            let reach = self.config.cutoff_nm
                + if self.config.neighbor_list {
                    self.config.neighbor_skin_nm
                } else {
                    0.0
                };
            if reach > cell.max_minimum_image_distance() {
                return Err(ForceError::InvalidInput(format!(
                    "cutoff + skin ({reach} nm) exceeds half the smallest cell height ({} nm)",
                    cell.max_minimum_image_distance()
                )));
            }
        }
        if self.config.neighbor_list {
            self.list.update(positions, cell);
        }
        let (pairs, type_of, n_types, list) =
            (&self.pairs, &self.type_of, self.n_types, &self.list);
        let mut energy = 0.0;
        let mut virial = Mat3::ZERO;
        let mut pair = |i: usize, j: usize, forces: &mut [Vec3]| {
            let mut d = positions[i] - positions[j];
            if let Some(cell) = cell {
                d = cell.minimum_image(d);
            }
            let r2 = d.length_squared();
            if r2 < rc2 {
                let (sigma, epsilon, shift) = pairs[type_of[i] * n_types + type_of[j]];
                let s2 = sigma * sigma / r2;
                let s6 = s2 * s2 * s2;
                energy += 4.0 * epsilon * (s6 * s6 - s6) - shift;
                let f = d * (24.0 * epsilon * (2.0 * s6 * s6 - s6) / r2);
                forces[i] += f;
                forces[j] -= f;
                virial += outer(d, f);
            }
        };
        let n = positions.len();
        if self.config.neighbor_list {
            for i in 0..n {
                for &j in list.neighbors(i) {
                    pair(i, j as usize, &mut out.forces);
                }
            }
        } else {
            for i in 0..n {
                for j in i + 1..n {
                    pair(i, j, &mut out.forces);
                }
            }
        }
        if self.config.dispersion_correction {
            let volume = cell.map(Cell::volume).ok_or_else(|| {
                ForceError::Unsupported("the dispersion correction needs a periodic cell".into())
            })?;
            energy += self.tail_energy_v / volume;
            // isotropic: tr W_tail / (3V) is the standard tail pressure
            virial += Mat3::IDENTITY * (self.tail_virial_v / volume);
        }
        if request.need_energy {
            out.energy = Some(energy);
        }
        if request.need_virial {
            out.virial = Some(virial);
        }
        Ok(())
    }
}

fn lj_energy(sigma: f64, epsilon: f64, r2: f64) -> f64 {
    let s2 = sigma * sigma / r2;
    let s6 = s2 * s2 * s2;
    4.0 * epsilon * (s6 * s6 - s6)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucmd_core::{AtomTemplate, AtomType, Bond, CombinationRule, ForceField, MoleculeType};
    use ucmd_force::testkit::{
        FdOptions, determinism_check, evaluate_all, fd_force_check, fd_virial_check, momentum_check,
    };

    /// Deterministic xorshift for test configurations (no extra dependencies).
    struct Rng(u64);
    impl Rng {
        fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// `n_per_edge³` atoms of two LJ types on a jittered lattice in `cell`.
    fn fluid(n_per_edge: usize, cell: &Cell, seed: u64, bonded: bool) -> (System, Vec<Vec3>) {
        let atom = |name: &str, t: usize| AtomTemplate {
            name: name.into(),
            element: 18,
            mass_amu: 39.948,
            charge_e: Some(0.0),
            atom_type: Some(t),
        };
        let mtype = |name: &str, t: usize| MoleculeType {
            name: name.into(),
            formal_charge_e: 0,
            atoms: vec![atom(name, t)],
            bonds: vec![],
            angles: vec![],
            dihedrals: vec![],
            impropers: vec![],
            constraints: vec![],
        };
        let mut types = vec![mtype("A", 0), mtype("B", 1)];
        if bonded {
            types.push(MoleculeType {
                name: "AA".into(),
                atoms: vec![atom("A1", 0), atom("A2", 0)],
                bonds: vec![Bond {
                    atoms: [0, 1],
                    form: None,
                }],
                ..mtype("AA", 0)
            });
        }
        let ff = ForceField {
            name: "two LJ types".into(),
            combination_rule: CombinationRule::LorentzBerthelot,
            scale_14_lj: 0.5,
            scale_14_coulomb: 0.5,
            atom_types: vec![
                AtomType {
                    name: "A".into(),
                    sigma_nm: 0.34,
                    epsilon_kjmol: 0.998,
                },
                AtomType {
                    name: "B".into(),
                    sigma_nm: 0.30,
                    epsilon_kjmol: 0.6,
                },
            ],
        };
        let n = n_per_edge.pow(3);
        let mut components = vec![(0, n / 2), (1, n - n / 2)];
        if bonded {
            components.push((2, 1));
        }
        let system = System::new("lj-test", types, &components, Some(ff), cell.periodic()).unwrap();
        let mut rng = Rng(seed);
        let h = cell.h();
        let mut positions = Vec::new();
        for i in 0..n_per_edge {
            for j in 0..n_per_edge {
                for k in 0..n_per_edge {
                    let s = (Vec3::new(i as f64, j as f64, k as f64) + Vec3::splat(0.5))
                        / n_per_edge as f64;
                    let jitter =
                        Vec3::new(rng.uniform(), rng.uniform(), rng.uniform()) - Vec3::splat(0.5);
                    positions.push(h * s + jitter * 0.04);
                }
            }
        }
        if bonded {
            positions.push(Vec3::ZERO);
            positions.push(Vec3::new(0.1, 0.0, 0.0));
        }
        (system, positions)
    }

    fn config(mode: CutoffMode, neighbor_list: bool) -> LennardJonesConfig {
        LennardJonesConfig {
            cutoff_nm: 0.9,
            cutoff_mode: mode,
            dispersion_correction: false,
            neighbor_skin_nm: 0.1,
            neighbor_list,
        }
    }

    fn cubic(l: f64) -> Cell {
        Cell::new([[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]], [true; 3]).unwrap()
    }

    fn triclinic(l: f64) -> Cell {
        Cell::new(
            [[l, 0.0, 0.0], [0.3 * l, l, 0.0], [-0.2 * l, 0.4 * l, l]],
            [true; 3],
        )
        .unwrap()
    }

    #[test]
    fn passes_the_testkit_in_cubic_and_triclinic_cells() {
        for cell in [cubic(2.6), triclinic(2.8)] {
            let (system, x) = fluid(6, &cell, 7, false);
            for mode in [CutoffMode::Truncate, CutoffMode::PotentialShift] {
                let mut lj = LennardJones::new(config(mode, true));
                lj.initialize(&system).unwrap();
                let options = FdOptions {
                    atoms: Some(vec![0, 17, 101, 200]),
                    ..FdOptions::default()
                };
                let fd = fd_force_check(&mut lj, &x, Some(&cell), &options).unwrap();
                assert!(fd.max_rel_error < 1e-5, "{mode:?}: {fd:?}");
                assert!(momentum_check(&mut lj, &x, Some(&cell)).unwrap() < 1e-13);
                assert!(determinism_check(&mut lj, &x, Some(&cell)).unwrap());
                if mode == CutoffMode::PotentialShift {
                    // with truncation the energy jumps when a pair crosses r_c under strain
                    let w = fd_virial_check(&mut lj, &x, Some(&cell), 1e-6).unwrap();
                    assert!(w.max_rel_error < 1e-5 && w.asymmetry < 1e-12, "{w:?}");
                }
            }
        }
    }

    /// E1.8: the neighbour list gives the same forces and energy as all pairs.
    #[test]
    fn neighbor_list_equals_brute_force() {
        for (seed, cell) in [(1, cubic(2.6)), (2, triclinic(2.8)), (3, cubic(3.4))] {
            let (system, x) = fluid(6, &cell, seed, false);
            let mut results = Vec::new();
            for listed in [true, false] {
                let mut lj = LennardJones::new(config(CutoffMode::PotentialShift, listed));
                lj.initialize(&system).unwrap();
                results.push(evaluate_all(&mut lj, &x, Some(&cell)).unwrap());
            }
            let (a, b) = (&results[0], &results[1]);
            let rms = (b.forces.iter().map(|f| f.length_squared()).sum::<f64>()
                / b.forces.len() as f64)
                .sqrt();
            let worst = a
                .forces
                .iter()
                .zip(&b.forces)
                .map(|(f, g)| (*f - *g).length())
                .fold(0.0, f64::max);
            assert!(worst <= 1e-12 * rms, "seed {seed}: {worst} vs rms {rms}");
            let (ea, eb) = (a.energy.unwrap(), b.energy.unwrap());
            assert!((ea - eb).abs() <= 1e-12 * eb.abs(), "{ea} vs {eb}");
        }
    }

    #[test]
    fn neighbor_list_rebuilds_only_when_needed() {
        let cell = cubic(2.6);
        let (system, mut x) = fluid(6, &cell, 4, false);
        let mut lj = LennardJones::new(config(CutoffMode::PotentialShift, true));
        lj.initialize(&system).unwrap();
        evaluate_all(&mut lj, &x, Some(&cell)).unwrap();
        x[5].x += 0.04; // less than skin/2 = 0.05
        evaluate_all(&mut lj, &x, Some(&cell)).unwrap();
        assert_eq!(lj.neighbor_list().builds(), 1);
        x[5].x += 0.02; // now 0.06 > skin/2
        evaluate_all(&mut lj, &x, Some(&cell)).unwrap();
        assert_eq!(lj.neighbor_list().builds(), 2);
    }

    /// Single type: E_tail = 8/3 π N²/V ε σ³ (x⁹/3 − x³) (the oracle's formula, ADR 0005).
    #[test]
    fn tail_correction_matches_the_analytic_formula() {
        let cell = cubic(3.0);
        let (mut system, x) = fluid(5, &cell, 9, false);
        // make both types identical so the single-type formula applies
        if let Some(ff) = system.forcefield.as_mut() {
            ff.atom_types[1] = AtomType {
                name: "B".into(),
                sigma_nm: 0.34,
                epsilon_kjmol: 0.998,
            };
        }
        let with = |correction: bool| {
            let mut lj = LennardJones::new(LennardJonesConfig {
                dispersion_correction: correction,
                ..config(CutoffMode::Truncate, true)
            });
            lj.initialize(&system).unwrap();
            evaluate_all(&mut lj, &x, Some(&cell)).unwrap()
        };
        let (on, off) = (with(true), with(false));
        let n = 125.0;
        let xr: f64 = 0.34 / 0.9;
        let expected = 8.0 / 3.0 * PI * n * n / 27.0
            * 0.998
            * 0.34f64.powi(3)
            * (xr.powi(9) / 3.0 - xr.powi(3));
        let got = on.energy.unwrap() - off.energy.unwrap();
        assert!(
            (got - expected).abs() <= 1e-12 * expected.abs(),
            "{got} vs {expected}"
        );
        let w_tail = on.virial.unwrap() - off.virial.unwrap();
        let p_tail = 16.0 / 3.0
            * PI
            * (n / 27.0).powi(2)
            * 0.998
            * 0.34f64.powi(3)
            * (2.0 / 3.0 * xr.powi(9) - xr.powi(3));
        assert!((w_tail.col(0)[0] / 27.0 - p_tail).abs() <= 1e-12 * p_tail.abs());
    }

    #[test]
    fn refuses_bonded_systems_and_oversized_cutoffs() {
        let cell = cubic(2.6);
        let (bonded, _) = fluid(4, &cell, 5, true);
        let mut lj = LennardJones::new(config(CutoffMode::Truncate, true));
        assert!(matches!(
            lj.initialize(&bonded),
            Err(ForceError::Unsupported(_))
        ));
        let small = cubic(1.6); // half height 0.8 < 0.9 + 0.1
        let (system, x) = fluid(4, &small, 6, false);
        let mut lj = LennardJones::new(config(CutoffMode::Truncate, true));
        lj.initialize(&system).unwrap();
        assert!(matches!(
            evaluate_all(&mut lj, &x, Some(&small)),
            Err(ForceError::InvalidInput(_))
        ));
    }
}
