//! Nonbonded interactions of a parameterized package: Lennard-Jones, 1-4 pairs and Coulomb
//! (E2.2, E2.3; PME in E2.4).
//!
//! - **Pairs** follow spec/system-package.md §5.2. Bond distance 1 or 2: excluded. Distance 3:
//!   a 1-4 pair with LJ scaled by `scale_14_lj` and Coulomb by `scale_14_coulomb`, computed
//!   without cutoff or shift (GROMACS `[ pairs ]`, OpenMM exceptions). All other pairs, and all
//!   intermolecular pairs, interact fully.
//! - **Coulomb methods:**
//!   - `direct`: every pair, no cutoff; non-periodic systems.
//!   - `ewald`: classic Ewald summation, the reference for PME (E2.3).
//!   - `pme`: smooth PME (E2.4, [`crate::pme`]).
//!
//!   For Ewald and PME the reciprocal part covers every pair, so each excluded and each 1-4 pair
//!   gets −k_e·qq·erf(αr)/r, and the self energy −k_e·α/√π·Σq² is added. Periodic packages are
//!   neutral (spec E_NET_CHARGE); the provider checks it and never adds a background charge.
//! - **Energy terms:** `lj` (normal pairs and tail), `lj14`, `coulomb` (everything electrostatic,
//!   including the scaled 1-4 part). This is the split of the OpenMM oracle v1.
//!
//! All transcendental functions come from `libm` (ADR 0010).
//!
//! **Threads:** the pair list is cut into row chunks of about equal pair counts. The cut depends
//! only on the list, never on the number of threads; each chunk sums into its own buffers, and
//! the buffers are added in chunk order. Results are therefore bitwise identical on any number
//! of threads (ADR 0010, E2 design §9).

use crate::lj::CutoffMode;
use crate::neighbor::NeighborList;
use crate::pme::{Pme, grid_for};
use crate::topology::PairTopology;
use rayon::prelude::*;
use serde::Deserialize;
use std::f64::consts::PI;
use std::ops::Range;
use ucmd_core::units::COULOMB_KJMOL_NM_E2;
use ucmd_core::{Cell, Mat3, System, Vec3};
use ucmd_force::{Capabilities, EvalRequest, ForceError, ForceProvider, ForceResult, outer};

/// Energy terms reported by [`Nonbonded`], in this order.
pub const TERMS: [&str; 3] = ["lj", "lj14", "coulomb"];

/// How electrostatics are computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CoulombMethod {
    /// Every pair without cutoff (non-periodic systems).
    Direct,
    /// Classic Ewald summation (periodic; reference for PME).
    Ewald,
    /// Smooth particle-mesh Ewald (periodic; production, E2.4).
    Pme,
}

fn default_tolerance() -> f64 {
    1e-5
}
fn default_order() -> usize {
    5
}
fn default_skin() -> f64 {
    0.2
}
fn default_mode() -> CutoffMode {
    CutoffMode::Truncate
}

/// `[[providers]]` keys of the `nonbonded` and `force-field` kinds (design/E2.md §6).
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonbondedConfig {
    /// Cutoff for LJ and real-space Coulomb in nm; required for periodic systems, absent for
    /// `direct` (all pairs).
    #[serde(default)]
    pub cutoff_nm: Option<f64>,
    /// LJ cutoff treatment (default: plain truncation, as in Lehnert 2025).
    #[serde(default = "default_mode")]
    pub lj_cutoff_mode: CutoffMode,
    /// LJ tail correction for energy and pressure (homogeneous fluid).
    #[serde(default)]
    pub dispersion_correction: bool,
    /// Coulomb method; default `pme` for periodic and `direct` for non-periodic systems.
    #[serde(default)]
    pub coulomb: Option<CoulombMethod>,
    /// Ewald tolerance: sets α = √(−ln(2·tol))/r_c (OpenMM's rule), the Ewald k range and the
    /// PME grid unless given explicitly.
    #[serde(default = "default_tolerance")]
    pub ewald_tolerance: f64,
    /// Ewald splitting parameter α in nm⁻¹ (overrides the tolerance rule).
    #[serde(default)]
    pub ewald_alpha_per_nm: Option<f64>,
    /// Classic Ewald: largest |m| per lattice direction (default from the tolerance).
    #[serde(default)]
    pub ewald_kmax: Option<[u32; 3]>,
    /// PME grid points per lattice direction (default from the tolerance).
    #[serde(default)]
    pub pme_grid: Option<[usize; 3]>,
    /// PME B-spline order (4…6; default 5, as OpenMM).
    #[serde(default = "default_order")]
    pub pme_order: usize,
    /// Verlet skin in nm.
    #[serde(default = "default_skin")]
    pub neighbor_skin_nm: f64,
}

impl Default for NonbondedConfig {
    fn default() -> Self {
        Self {
            cutoff_nm: None,
            lj_cutoff_mode: CutoffMode::Truncate,
            dispersion_correction: false,
            coulomb: None,
            ewald_tolerance: default_tolerance(),
            ewald_alpha_per_nm: None,
            ewald_kmax: None,
            pme_grid: None,
            pme_order: default_order(),
            neighbor_skin_nm: default_skin(),
        }
    }
}

/// A 1-4 pair with its combined and scaled parameters.
#[derive(Debug, Clone, Copy)]
struct Pair14 {
    i: usize,
    j: usize,
    sigma: f64,
    epsilon_scaled: f64,
    qq: f64,
    qq_scaled: f64,
}

/// LJ, 1-4 pairs and Coulomb of a parameterized package.
pub struct Nonbonded {
    config: NonbondedConfig,
    name: String,
    method: CoulombMethod,
    n_types: usize,
    type_of: Vec<usize>,
    /// (σ, ε, energy shift) per ordered type pair, index a·n_types + b.
    pairs: Vec<(f64, f64, f64)>,
    charges: Vec<f64>,
    topology: PairTopology,
    p14: Vec<Pair14>,
    alpha: f64,
    self_energy: f64,
    tail_energy_v: f64,
    tail_virial_v: f64,
    list: NeighborList,
    pme: Option<Pme>,
    /// Per-chunk accumulators of the pair loop, kept between evaluations.
    chunks: Vec<PairSums>,
}

/// Most chunks the pair loop is cut into.
const MAX_CHUNKS: usize = 64;
/// Fewest pairs per chunk, so small systems are not cut into tiny pieces.
const MIN_CHUNK_PAIRS: usize = 4096;
/// Atoms per task when the chunk buffers are added up.
const REDUCE_BLOCK: usize = 1024;

/// Row ranges with about equal pair counts; `pairs_before(i)` is the number of pairs in the rows
/// before row `i`, and `pairs_before(n_rows)` the total. The ranges depend only on the counts.
fn row_chunks(n_rows: usize, pairs_before: impl Fn(usize) -> usize) -> Vec<Range<usize>> {
    let total = pairs_before(n_rows);
    let wanted = total.div_ceil(MIN_CHUNK_PAIRS).clamp(1, MAX_CHUNKS);
    let target = total.div_ceil(wanted).max(1);
    let mut ranges = Vec::with_capacity(wanted);
    let mut begin = 0;
    for i in 0..n_rows {
        if pairs_before(i + 1) >= target * (ranges.len() + 1) {
            ranges.push(begin..i + 1);
            begin = i + 1;
        }
    }
    if begin < n_rows {
        ranges.push(begin..n_rows);
    }
    ranges
}

/// What one chunk of the pair loop adds up.
#[derive(Default)]
struct PairSums {
    forces: Vec<Vec3>,
    e_lj: f64,
    e_coul: f64,
    virial: Mat3,
}

impl PairSums {
    fn reset(&mut self, n: usize) {
        self.forces.clear();
        self.forces.resize(n, Vec3::ZERO);
        self.e_lj = 0.0;
        self.e_coul = 0.0;
        self.virial = Mat3::ZERO;
    }
}

/// The normal (non-excluded, non-1-4) pair interaction: LJ and real-space or direct Coulomb.
struct PairKernel<'a> {
    positions: &'a [Vec3],
    cell: Option<&'a Cell>,
    rc2: f64,
    ewald: bool,
    alpha: f64,
    two_alpha_sqrt_pi: f64,
    /// (σ, ε, energy shift) per ordered type pair.
    pairs: &'a [(f64, f64, f64)],
    type_of: &'a [usize],
    n_types: usize,
    charges: &'a [f64],
    topology: &'a PairTopology,
}

impl PairKernel<'_> {
    #[inline]
    fn pair(&self, i: usize, j: usize, sums: &mut PairSums) {
        if self.topology.skips(i, j) {
            return;
        }
        let mut d = self.positions[i] - self.positions[j];
        if let Some(cell) = self.cell {
            d = cell.minimum_image(d);
        }
        let r2 = d.length_squared();
        if r2 >= self.rc2 {
            return;
        }
        let (sigma, epsilon, shift) = self.pairs[self.type_of[i] * self.n_types + self.type_of[j]];
        let s2 = sigma * sigma / r2;
        let s6 = s2 * s2 * s2;
        sums.e_lj += 4.0 * epsilon * (s6 * s6 - s6) - shift;
        let mut f_over_r = 24.0 * epsilon * (2.0 * s6 * s6 - s6) / r2;
        let qq = self.charges[i] * self.charges[j];
        if qq != 0.0 {
            let r = r2.sqrt();
            if self.ewald {
                let ar = self.alpha * r;
                let erfc_ar = erfc(ar);
                sums.e_coul += COULOMB_KJMOL_NM_E2 * qq * erfc_ar / r;
                f_over_r += COULOMB_KJMOL_NM_E2
                    * qq
                    * (erfc_ar / r + self.two_alpha_sqrt_pi * libm::exp(-ar * ar))
                    / r2;
            } else {
                sums.e_coul += COULOMB_KJMOL_NM_E2 * qq / r;
                f_over_r += COULOMB_KJMOL_NM_E2 * qq / (r2 * r);
            }
        }
        let f = d * f_over_r;
        sums.forces[i] += f;
        sums.forces[j] -= f;
        sums.virial += outer(d, f);
    }
}

impl Nonbonded {
    /// A provider with the given settings; parameters are read in `initialize`.
    pub fn new(name: impl Into<String>, config: NonbondedConfig) -> Self {
        Self {
            list: NeighborList::new(config.cutoff_nm.unwrap_or(0.0), config.neighbor_skin_nm),
            config,
            name: name.into(),
            method: CoulombMethod::Direct,
            n_types: 0,
            type_of: Vec::new(),
            pairs: Vec::new(),
            charges: Vec::new(),
            topology: PairTopology::default(),
            p14: Vec::new(),
            alpha: 0.0,
            self_energy: 0.0,
            tail_energy_v: 0.0,
            tail_virial_v: 0.0,
            pme: None,
            chunks: Vec::new(),
        }
    }

    /// The PME grid in use, once the first evaluation has fixed it.
    pub fn pme_grid(&self) -> Option<[usize; 3]> {
        self.pme.as_ref().map(Pme::grid)
    }

    /// The Coulomb method in use (resolved in `initialize`).
    pub fn method(&self) -> CoulombMethod {
        self.method
    }

    /// The Ewald splitting parameter α in nm⁻¹ (0 for `direct`).
    pub fn alpha_per_nm(&self) -> f64 {
        self.alpha
    }

    fn check_cell(&self, cell: Option<&Cell>) -> Result<(), ForceError> {
        match (cell, self.method) {
            (None, CoulombMethod::Ewald | CoulombMethod::Pme) => Err(ForceError::Unsupported(
                "Ewald and PME need a periodic cell; use coulomb = \"direct\"".into(),
            )),
            (Some(_), CoulombMethod::Direct) => Err(ForceError::Unsupported(
                "coulomb = \"direct\" is for non-periodic systems; use \"pme\"".into(),
            )),
            (Some(c), _) => {
                let rc = self.config.cutoff_nm.unwrap_or(0.0);
                let reach = rc + self.config.neighbor_skin_nm;
                if reach > c.max_minimum_image_distance() {
                    return Err(ForceError::InvalidInput(format!(
                        "cutoff + skin ({reach} nm) exceeds half the smallest cell height ({} nm)",
                        c.max_minimum_image_distance()
                    )));
                }
                Ok(())
            }
            (None, CoulombMethod::Direct) => Ok(()),
        }
    }
}

/// erfc for the real-space kernel (pure Rust, ADR 0010).
fn erfc(x: f64) -> f64 {
    libm::erfc(x)
}

/// −k_e·qq·erf(αr)/r and its force on i (d = r_i − r_j): the reciprocal part of an excluded pair.
fn erf_correction(qq: f64, alpha: f64, d: Vec3) -> (f64, Vec3) {
    let r2 = d.length_squared();
    let r = r2.sqrt();
    let ar = alpha * r;
    let erf = 1.0 - erfc(ar);
    let energy = -COULOMB_KJMOL_NM_E2 * qq * erf / r;
    // d/dr[erf(αr)/r] = (2α/√π e^{−α²r²} r − erf(αr)) / r²; F_i = −dE/dr · d/r
    let de_dr =
        -COULOMB_KJMOL_NM_E2 * qq * (2.0 * alpha / PI.sqrt() * libm::exp(-ar * ar) * r - erf) / r2;
    (energy, d * (-de_dr / r))
}

impl ForceProvider for Nonbonded {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            periodic: true,
            virial: true,
            energy_terms: true,
            deterministic: true,
        }
    }

    fn initialize(&mut self, system: &System) -> Result<(), ForceError> {
        let c = &self.config;
        let periodic = system.is_periodic();
        self.method = c.coulomb.unwrap_or(if periodic {
            CoulombMethod::Pme
        } else {
            CoulombMethod::Direct
        });
        match (self.method, c.cutoff_nm) {
            (CoulombMethod::Direct, Some(_)) => {
                return Err(ForceError::InvalidInput(
                    "coulomb = \"direct\" uses every pair; remove cutoff_nm".into(),
                ));
            }
            (CoulombMethod::Ewald | CoulombMethod::Pme, None) => {
                return Err(ForceError::InvalidInput(
                    "Ewald and PME need cutoff_nm".into(),
                ));
            }
            (_, Some(rc)) if !(rc.is_finite() && rc > 0.0) => {
                return Err(ForceError::InvalidInput("cutoff_nm must be > 0".into()));
            }
            _ => {}
        }
        if !(c.neighbor_skin_nm >= 0.0 && c.ewald_tolerance > 0.0 && c.ewald_tolerance < 1.0) {
            return Err(ForceError::InvalidInput(
                "neighbor_skin_nm must be ≥ 0 and 0 < ewald_tolerance < 1".into(),
            ));
        }
        let ff = system.forcefield.as_ref().ok_or_else(|| {
            ForceError::InvalidInput("nonbonded terms need a parameterized package".into())
        })?;
        let types = system
            .atom_types()
            .ok_or_else(|| ForceError::InvalidInput("every atom needs an atom type".into()))?;
        let charges = system
            .charges()
            .ok_or_else(|| ForceError::InvalidInput("every atom needs a charge".into()))?;
        self.n_types = ff.atom_types.len();
        self.type_of = types.to_vec();
        self.charges = charges.to_vec();
        let rc = c.cutoff_nm.unwrap_or(f64::INFINITY);
        self.pairs.clear();
        for a in &ff.atom_types {
            for b in &ff.atom_types {
                let (sigma, epsilon) = ff.combination_rule.combine(a, b);
                let shift = match (c.lj_cutoff_mode, c.cutoff_nm) {
                    (CutoffMode::PotentialShift, Some(rc)) => {
                        let s2 = sigma * sigma / (rc * rc);
                        let s6 = s2 * s2 * s2;
                        4.0 * epsilon * (s6 * s6 - s6)
                    }
                    _ => 0.0,
                };
                self.pairs.push((sigma, epsilon, shift));
            }
        }
        self.topology = PairTopology::new(system);
        self.p14 = self
            .topology
            .pairs14
            .iter()
            .map(|&[i, j]| {
                let (i, j) = (i as usize, j as usize);
                let (sigma, epsilon, _) =
                    self.pairs[self.type_of[i] * self.n_types + self.type_of[j]];
                let qq = self.charges[i] * self.charges[j];
                Pair14 {
                    i,
                    j,
                    sigma,
                    epsilon_scaled: ff.scale_14_lj * epsilon,
                    qq,
                    qq_scaled: ff.scale_14_coulomb * qq,
                }
            })
            .collect();
        if periodic {
            let total: f64 = self.charges.iter().sum();
            if total.abs() > 1e-6 {
                return Err(ForceError::InvalidInput(format!(
                    "a periodic system must be neutral; net charge {total:+.3e} e"
                )));
            }
        }
        self.alpha = match self.method {
            CoulombMethod::Direct => 0.0,
            _ => c
                .ewald_alpha_per_nm
                .unwrap_or_else(|| libm::sqrt(-libm::log(2.0 * c.ewald_tolerance)) / rc),
        };
        self.self_energy = -COULOMB_KJMOL_NM_E2 * self.alpha / PI.sqrt()
            * self.charges.iter().map(|q| q * q).sum::<f64>();
        // LJ tail, multiplied by V (as in the E1 Lennard-Jones provider, virial route).
        let (mut e, mut w) = (0.0, 0.0);
        if c.dispersion_correction {
            if !periodic {
                return Err(ForceError::Unsupported(
                    "the dispersion correction needs a periodic system".into(),
                ));
            }
            let mut counts = vec![0.0f64; self.n_types];
            for &t in &self.type_of {
                counts[t] += 1.0;
            }
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
        }
        self.tail_energy_v = 8.0 * PI / 3.0 * e;
        self.tail_virial_v = 16.0 * PI / 3.0 * w;
        self.list = NeighborList::new(c.cutoff_nm.unwrap_or(0.0), c.neighbor_skin_nm);
        // The PME grid depends on the cell, so the solver is set up at the first evaluation and
        // then kept, as OpenMM does.
        self.pme = None;
        Ok(())
    }

    fn evaluate(
        &mut self,
        positions: &[Vec3],
        cell: Option<&Cell>,
        request: &EvalRequest,
        out: &mut ForceResult,
    ) -> Result<(), ForceError> {
        self.check_cell(cell)?;
        out.reset(positions.len());
        let n = positions.len();
        let cutoff = self.config.cutoff_nm;
        let rc2 = cutoff.map_or(f64::INFINITY, |rc| rc * rc);
        let ewald = self.method != CoulombMethod::Direct;
        let alpha = self.alpha;
        let mut e_lj14 = 0.0;
        if cutoff.is_some() {
            self.list.update(positions, cell);
        }
        let kernel = PairKernel {
            positions,
            cell,
            rc2,
            ewald,
            alpha,
            two_alpha_sqrt_pi: 2.0 * alpha / PI.sqrt(),
            pairs: &self.pairs,
            type_of: &self.type_of,
            n_types: self.n_types,
            charges: &self.charges,
            topology: &self.topology,
        };
        let list = &self.list;
        let ranges = if cutoff.is_some() {
            row_chunks(n, |i| list.pairs_before(i))
        } else {
            row_chunks(n, |i| i * n - i * (i + 1) / 2)
        };
        if self.chunks.len() < ranges.len() {
            self.chunks.resize_with(ranges.len(), PairSums::default);
        }
        let chunks = &mut self.chunks[..ranges.len()];
        chunks
            .par_iter_mut()
            .zip(ranges.par_iter())
            .for_each(|(sums, rows)| {
                sums.reset(n);
                for i in rows.clone() {
                    if cutoff.is_some() {
                        for &j in list.neighbors(i) {
                            kernel.pair(i, j as usize, sums);
                        }
                    } else {
                        for j in i + 1..n {
                            kernel.pair(i, j, sums);
                        }
                    }
                }
            });
        // add the chunks up in chunk order: the same sums on any number of threads
        let chunks = &self.chunks[..ranges.len()];
        out.forces
            .par_chunks_mut(REDUCE_BLOCK)
            .enumerate()
            .for_each(|(block, forces)| {
                let start = block * REDUCE_BLOCK;
                for sums in chunks {
                    for (f, g) in forces.iter_mut().zip(&sums.forces[start..]) {
                        *f += *g;
                    }
                }
            });
        let (mut e_lj, mut e_coul, mut virial) = (0.0, 0.0, Mat3::ZERO);
        for sums in chunks {
            e_lj += sums.e_lj;
            e_coul += sums.e_coul;
            virial += sums.virial;
        }
        // 1-4 pairs: plain LJ and Coulomb, scaled; no cutoff, no shift. Molecules are whole.
        for p in &self.p14 {
            let d = positions[p.i] - positions[p.j];
            let r2 = d.length_squared();
            let r = r2.sqrt();
            let s2 = p.sigma * p.sigma / r2;
            let s6 = s2 * s2 * s2;
            e_lj14 += 4.0 * p.epsilon_scaled * (s6 * s6 - s6);
            e_coul += COULOMB_KJMOL_NM_E2 * p.qq_scaled / r;
            let f_over_r = 24.0 * p.epsilon_scaled * (2.0 * s6 * s6 - s6) / r2
                + COULOMB_KJMOL_NM_E2 * p.qq_scaled / (r2 * r);
            let f = d * f_over_r;
            out.forces[p.i] += f;
            out.forces[p.j] -= f;
            virial += outer(d, f);
            if ewald && p.qq != 0.0 {
                let (e, f) = erf_correction(p.qq, alpha, d);
                e_coul += e;
                out.forces[p.i] += f;
                out.forces[p.j] -= f;
                virial += outer(d, f);
            }
        }
        if ewald {
            for &[i, j] in &self.topology.excluded {
                let (i, j) = (i as usize, j as usize);
                let qq = self.charges[i] * self.charges[j];
                if qq == 0.0 {
                    continue;
                }
                let d = positions[i] - positions[j];
                let (e, f) = erf_correction(qq, alpha, d);
                e_coul += e;
                out.forces[i] += f;
                out.forces[j] -= f;
                virial += outer(d, f);
            }
            e_coul += self.self_energy;
            let cell = cell.ok_or_else(|| ForceError::Unsupported("Ewald needs a cell".into()))?;
            let (e, w) = if self.method == CoulombMethod::Pme {
                if self.pme.is_none() {
                    let grid = self
                        .config
                        .pme_grid
                        .unwrap_or_else(|| grid_for(cell, alpha, self.config.ewald_tolerance));
                    self.pme = Some(Pme::new(self.config.pme_order, grid)?);
                }
                let pme = self
                    .pme
                    .as_mut()
                    .ok_or_else(|| ForceError::Contract("PME solver missing".into()))?;
                pme.compute(positions, &self.charges, cell, alpha, &mut out.forces)
            } else {
                let kmax = self
                    .config
                    .ewald_kmax
                    .unwrap_or_else(|| ewald_kmax(cell, alpha, self.config.ewald_tolerance));
                ewald_reciprocal(positions, &self.charges, cell, alpha, kmax, &mut out.forces)
            };
            e_coul += e;
            virial += w;
        }
        if self.config.dispersion_correction {
            let volume = cell.map(Cell::volume).ok_or_else(|| {
                ForceError::Unsupported("the dispersion correction needs a periodic cell".into())
            })?;
            e_lj += self.tail_energy_v / volume;
            virial += Mat3::IDENTITY * (self.tail_virial_v / volume);
        }
        if request.need_energy {
            let energy = [e_lj, e_lj14, e_coul];
            out.energy = Some(energy.iter().sum());
            out.energy_terms = TERMS
                .iter()
                .zip(energy)
                .map(|(name, e)| ((*name).to_string(), e))
                .collect();
        }
        if request.need_virial {
            out.virial = Some(virial);
        }
        Ok(())
    }
}

/// Largest |m| per lattice direction so that the Gaussian factor exp(−k²/4α²) of the omitted
/// k vectors is below `tolerance`.
pub fn ewald_kmax(cell: &Cell, alpha: f64, tolerance: f64) -> [u32; 3] {
    let k_cut = 2.0 * alpha * libm::sqrt(-libm::log(tolerance));
    // |k| along reciprocal direction i is 2π m / height_i.
    cell.heights()
        .map(|h| libm::ceil(k_cut * h / (2.0 * PI)) as u32)
}

/// Classic Ewald reciprocal sum: energy and virial; forces are added to `forces`.
///
/// E = (2π k_e / V) Σ_{k≠0} exp(−k²/4α²)/k² |S(k)|², S(k) = Σ_j q_j exp(i k·r_j), summed over
/// half of k space (each ±k pair once, doubled). Virial
/// W = (2π k_e / V) Σ exp(−k²/4α²)/k² |S|² [I − 2(1 + k²/4α²) k⊗k / k²].
pub fn ewald_reciprocal(
    positions: &[Vec3],
    charges: &[f64],
    cell: &Cell,
    alpha: f64,
    kmax: [u32; 3],
    forces: &mut [Vec3],
) -> (f64, Mat3) {
    let volume = cell.volume();
    // reciprocal vectors b_i with a_i · b_j = δ_ij: the rows of h⁻¹ (h has columns a, b, c)
    let hinv = cell.h().inverse();
    let b = [hinv.row(0), hinv.row(1), hinv.row(2)];
    let prefactor = 4.0 * PI * COULOMB_KJMOL_NM_E2 / volume; // half space, doubled
    let four_alpha2 = 4.0 * alpha * alpha;
    let mut energy = 0.0;
    let mut virial = Mat3::ZERO;
    let (k0, k1, k2) = (kmax[0] as i64, kmax[1] as i64, kmax[2] as i64);
    let mut cos_t = vec![0.0; positions.len()];
    let mut sin_t = vec![0.0; positions.len()];
    for m0 in 0..=k0 {
        for m1 in -k1..=k1 {
            for m2 in -k2..=k2 {
                let half = m0 > 0 || (m0 == 0 && (m1 > 0 || (m1 == 0 && m2 > 0)));
                if !half {
                    continue;
                }
                let k = 2.0 * PI * (m0 as f64 * b[0] + m1 as f64 * b[1] + m2 as f64 * b[2]);
                let k2v = k.length_squared();
                let g = libm::exp(-k2v / four_alpha2) / k2v;
                let (mut c, mut s) = (0.0, 0.0);
                for (j, (&r, &q)) in positions.iter().zip(charges).enumerate() {
                    let theta = k.dot(r);
                    let (sn, cs) = (libm::sin(theta), libm::cos(theta));
                    cos_t[j] = cs;
                    sin_t[j] = sn;
                    c += q * cs;
                    s += q * sn;
                }
                let s2 = c * c + s * s;
                energy += prefactor * g * s2;
                let coeff = 2.0 * prefactor * g;
                for (j, &q) in charges.iter().enumerate() {
                    forces[j] += k * (coeff * q * (c * sin_t[j] - s * cos_t[j]));
                }
                let factor = 2.0 * (1.0 / k2v + 1.0 / four_alpha2);
                virial += (Mat3::IDENTITY - outer(k, k) * factor) * (prefactor * g * s2);
            }
        }
    }
    (energy, virial)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{WithBox, all_forms_molecule, positions_for, system_of};
    use ucmd_force::testkit::{
        FdOptions, determinism_check, energy_terms_check, fd_force_check, fd_virial_check,
        momentum_check,
    };

    /// 256 bonded ion pairs (+1/−1 e, LJ type C) on a jittered rock-salt lattice of 512 sites,
    /// 0.375 nm apart, in a 3 nm cubic box.
    fn ion_pairs(periodic: bool) -> (System, Vec<Vec3>, Cell) {
        use crate::test_util::Rng;
        use ucmd_core::{AtomTemplate, Bond, BondForm, MoleculeType};
        let atom = |name: &str, q: f64| AtomTemplate {
            name: name.into(),
            element: 6,
            mass_amu: 12.011,
            charge_e: Some(q),
            atom_type: Some(0),
        };
        let pair = MoleculeType {
            name: "ionpair".into(),
            formal_charge_e: 0,
            atoms: vec![atom("P", 1.0), atom("N", -1.0)],
            bonds: vec![Bond {
                atoms: [0, 1],
                form: Some(BondForm::Harmonic {
                    r0_nm: 0.375,
                    k_kjmol_nm2: 1000.0,
                }),
            }],
            angles: Vec::new(),
            dihedrals: Vec::new(),
            impropers: Vec::new(),
            constraints: Vec::new(),
        };
        let system = system_of(vec![pair], &[(0, 256)], [periodic; 3]);
        let (spacing, mut rng) = (0.375, Rng(17));
        let mut positions = Vec::with_capacity(512);
        for z in 0..8 {
            for y in 0..8 {
                for x in (0..8).step_by(2) {
                    // rock-salt order: the cation sits where x + y + z is even
                    let sites = if (y + z) % 2 == 0 {
                        [x, x + 1]
                    } else {
                        [x + 1, x]
                    };
                    for site in sites {
                        let jitter = Vec3::new(rng.uniform(), rng.uniform(), rng.uniform())
                            - Vec3::splat(0.5);
                        positions.push(
                            Vec3::new(site as f64, y as f64, z as f64) * spacing + 0.06 * jitter,
                        );
                    }
                }
            }
        }
        let cell = Cell::new(
            [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            [true; 3],
        )
        .unwrap();
        (system, positions, cell)
    }

    /// E2 design §9, ADR 0010: forces, energy terms and virial are bitwise identical on 1, 4
    /// and 16 threads, with PME (several pair chunks, every PME stage threaded) and with direct
    /// Coulomb (the all-pairs path).
    #[test]
    fn results_do_not_depend_on_the_thread_count() {
        let pme = NonbondedConfig {
            cutoff_nm: Some(0.9),
            coulomb: Some(CoulombMethod::Pme),
            pme_grid: Some([30, 32, 36]),
            dispersion_correction: true,
            ..NonbondedConfig::default()
        };
        let direct = NonbondedConfig {
            coulomb: Some(CoulombMethod::Direct),
            ..NonbondedConfig::default()
        };
        for (config, periodic) in [(pme, true), (direct, false)] {
            let (system, positions, cell) = ion_pairs(periodic);
            let cell = periodic.then_some(&cell);
            let run = |threads: usize| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                pool.install(|| {
                    let mut p = provider(&system, config);
                    let mut out = ForceResult::default();
                    p.evaluate(&positions, cell, &EvalRequest::everything(), &mut out)
                        .unwrap();
                    (out, p.chunks.len())
                })
            };
            let bits = |out: &ForceResult| {
                let forces: Vec<u64> = out
                    .forces
                    .iter()
                    .flat_map(|f| f.to_array().map(f64::to_bits))
                    .collect();
                let terms: Vec<(String, u64)> = out
                    .energy_terms
                    .iter()
                    .map(|(name, e)| (name.clone(), e.to_bits()))
                    .collect();
                let virial = out.virial.map(|w| w.to_cols_array().map(f64::to_bits));
                (forces, out.energy.map(f64::to_bits), terms, virial)
            };
            let (reference, chunks) = run(1);
            assert!(chunks >= 4, "only {chunks} pair chunks");
            for threads in [4, 16] {
                let (other, _) = run(threads);
                assert!(
                    bits(&reference) == bits(&other),
                    "{:?}: {threads} threads differ from 1",
                    config.coulomb
                );
            }
        }
    }

    fn provider(system: &System, config: NonbondedConfig) -> Nonbonded {
        let mut p = Nonbonded::new("nonbonded", config);
        p.initialize(system).unwrap();
        p
    }

    #[test]
    fn direct_coulomb_and_lj_pass_the_test_kit() {
        let (system, _) =
            system_of(vec![all_forms_molecule()], &[(0, 3)], [false; 3]).with_box(None);
        let positions = positions_for(&system, 5);
        let mut p = provider(&system, NonbondedConfig::default());
        assert_eq!(p.method(), CoulombMethod::Direct);
        let f = fd_force_check(&mut p, &positions, None, &FdOptions::default()).unwrap();
        assert!(f.max_rel_error < 1e-6, "{f:?}");
        let w = fd_virial_check(&mut p, &positions, None, 1e-6).unwrap();
        assert!(w.max_rel_error < 1e-6, "{w:?}");
        assert!(momentum_check(&mut p, &positions, None).unwrap() < 1e-10);
        assert!(determinism_check(&mut p, &positions, None).unwrap());
        assert!(energy_terms_check(&mut p, &positions, None).unwrap() < 1e-12);
    }

    #[test]
    fn ewald_passes_the_test_kit_in_a_triclinic_cell() {
        // Two molecules, so that no atom comes close to a periodic image of another (the r⁻¹²
        // repulsion of a close contact would make the finite differences, not the forces, inexact).
        let system = system_of(vec![all_forms_molecule()], &[(0, 2)], [true; 3]);
        let cell = Cell::new(
            [[2.4, 0.0, 0.0], [0.5, 2.3, 0.0], [-0.3, 0.4, 2.2]],
            [true; 3],
        )
        .unwrap();
        let positions = positions_for(&system, 9);
        // The LJ tail uses the virial route (as in E1), which is not the strain derivative of
        // its energy, so it is left out of the finite-difference virial check. The k range is
        // fixed so that the strained cells of the check use the same k vectors.
        let config = NonbondedConfig {
            cutoff_nm: Some(0.8),
            coulomb: Some(CoulombMethod::Ewald),
            ewald_tolerance: 1e-6,
            ewald_kmax: Some([14, 14, 14]),
            ..NonbondedConfig::default()
        };
        let mut p = provider(&system, config);
        let f = fd_force_check(&mut p, &positions, Some(&cell), &FdOptions::default()).unwrap();
        assert!(f.max_rel_error < 1e-6, "{f:?}");
        let w = fd_virial_check(&mut p, &positions, Some(&cell), 1e-6).unwrap();
        assert!(w.max_rel_error < 1e-6, "{w:?}");
        assert!(momentum_check(&mut p, &positions, Some(&cell)).unwrap() < 1e-8);
        assert!(determinism_check(&mut p, &positions, Some(&cell)).unwrap());
    }

    #[test]
    fn ewald_energy_does_not_depend_on_alpha() {
        // The split into real and reciprocal space is arbitrary when both are converged.
        let system = system_of(vec![all_forms_molecule()], &[(0, 4)], [true; 3]);
        let (_, cell) = system.clone().with_box(Some(3.0));
        let cell = cell.unwrap();
        let positions = positions_for(&system, 2);
        let mut energies = Vec::new();
        for alpha in [4.0, 5.0, 6.0] {
            let config = NonbondedConfig {
                cutoff_nm: Some(1.25),
                coulomb: Some(CoulombMethod::Ewald),
                ewald_alpha_per_nm: Some(alpha),
                ewald_tolerance: 1e-14,
                ..NonbondedConfig::default()
            };
            let mut p = provider(&system, config);
            let mut out = ForceResult::default();
            p.evaluate(
                &positions,
                Some(&cell),
                &EvalRequest::everything(),
                &mut out,
            )
            .unwrap();
            energies.push(out.energy_terms[2].1);
        }
        for e in &energies[1..] {
            assert!(
                (e - energies[0]).abs() < 1e-7 * energies[0].abs(),
                "{energies:?}"
            );
        }
    }

    fn coulomb_with(
        method: CoulombMethod,
        system: &System,
        cell: &Cell,
        positions: &[Vec3],
        extra: impl Fn(&mut NonbondedConfig),
    ) -> ForceResult {
        let mut config = NonbondedConfig {
            cutoff_nm: Some(0.9),
            coulomb: Some(method),
            ..NonbondedConfig::default()
        };
        extra(&mut config);
        let mut p = provider(system, config);
        let mut out = ForceResult::default();
        p.evaluate(positions, Some(cell), &EvalRequest::everything(), &mut out)
            .unwrap();
        out
    }

    fn rms_relative(a: &[Vec3], b: &[Vec3]) -> f64 {
        let n = a.len() as f64;
        let diff = (a
            .iter()
            .zip(b)
            .map(|(x, y)| (*x - *y).length_squared())
            .sum::<f64>()
            / n)
            .sqrt();
        let scale = (b.iter().map(|f| f.length_squared()).sum::<f64>() / n).sqrt();
        diff / scale
    }

    #[test]
    fn pme_converges_to_ewald() {
        // Same α and cutoff in both, so the real-space parts are identical and the comparison
        // measures the reciprocal part alone. The PME error falls with order and grid spacing.
        let system = system_of(vec![all_forms_molecule()], &[(0, 6)], [true; 3]);
        let cell = Cell::new(
            [[2.9, 0.0, 0.0], [0.3, 2.7, 0.0], [-0.2, 0.4, 2.8]],
            [true; 3],
        )
        .unwrap();
        let positions = positions_for(&system, 4);
        let reference = coulomb_with(CoulombMethod::Ewald, &system, &cell, &positions, |c| {
            c.ewald_alpha_per_nm = Some(3.5);
            c.ewald_kmax = Some([24, 24, 24]);
        });
        let mut previous = f64::INFINITY;
        for (order, spacing) in [(4, 0.12), (5, 0.08), (6, 0.04)] {
            let grid =
                [2.9f64, 2.7, 2.8].map(|l| crate::pme::legal_size((l / spacing).ceil() as usize));
            let pme = coulomb_with(CoulombMethod::Pme, &system, &cell, &positions, |c| {
                c.ewald_alpha_per_nm = Some(3.5);
                c.pme_order = order;
                c.pme_grid = Some(grid);
            });
            let rms = rms_relative(&pme.forces, &reference.forces);
            let de = (pme.energy.unwrap() - reference.energy.unwrap()).abs()
                / reference.energy.unwrap().abs();
            assert!(
                rms < previous,
                "order {order}: rms force error {rms:.2e} did not decrease"
            );
            previous = rms;
            if order == 6 {
                // α·h = 0.14: the expected PME accuracy is ~1e-7 (exactness is tested against
                // OpenMM with matched parameters in tests/oracle_pme.rs)
                assert!(
                    rms < 1e-6 && de < 1e-6,
                    "order 6: rms {rms:.2e}, energy {de:.2e}"
                );
            }
        }
        // Default settings (tolerance 1e-5, order 5, OpenMM's grid rule) against Ewald with the
        // same α: PLAN E2.4 asks for an rms force error ≤ 1e-5.
        let reference = coulomb_with(CoulombMethod::Ewald, &system, &cell, &positions, |c| {
            c.ewald_kmax = Some([24, 24, 24]);
        });
        let default = coulomb_with(CoulombMethod::Pme, &system, &cell, &positions, |_| {});
        let rms = rms_relative(&default.forces, &reference.forces);
        assert!(rms <= 1e-5, "default settings: rms force error {rms:.2e}");
    }

    #[test]
    fn pme_passes_the_test_kit() {
        let system = system_of(vec![all_forms_molecule()], &[(0, 2)], [true; 3]);
        let cell = Cell::new(
            [[2.4, 0.0, 0.0], [0.5, 2.3, 0.0], [-0.3, 0.4, 2.2]],
            [true; 3],
        )
        .unwrap();
        let positions = positions_for(&system, 9);
        let config = NonbondedConfig {
            cutoff_nm: Some(0.8),
            coulomb: Some(CoulombMethod::Pme),
            pme_grid: Some([24, 24, 24]),
            ..NonbondedConfig::default()
        };
        let mut p = provider(&system, config);
        let f = fd_force_check(&mut p, &positions, Some(&cell), &FdOptions::default()).unwrap();
        assert!(f.max_rel_error < 1e-6, "{f:?}");
        let w = fd_virial_check(&mut p, &positions, Some(&cell), 1e-6).unwrap();
        assert!(w.max_rel_error < 1e-6, "{w:?}");
        assert!(determinism_check(&mut p, &positions, Some(&cell)).unwrap());
        // Smooth PME does not conserve momentum exactly: the net force is of the order of the
        // interpolation error and vanishes as the grid is refined (Essmann et al. 1995). Runs
        // remove the centre-of-mass drift ([momentum] in run.toml), as with OpenMM and GROMACS.
        let coarse = momentum_check(&mut p, &positions, Some(&cell)).unwrap();
        let mut fine = provider(
            &system,
            NonbondedConfig {
                pme_grid: Some([48, 48, 48]),
                ..config
            },
        );
        let refined = momentum_check(&mut fine, &positions, Some(&cell)).unwrap();
        assert!(
            coarse < 1e-3 && refined < 0.1 * coarse,
            "net force {coarse:.2e} → {refined:.2e}"
        );
    }

    #[test]
    fn method_and_cell_must_fit() {
        let system = system_of(vec![all_forms_molecule()], &[(0, 1)], [false; 3]);
        let bad = NonbondedConfig {
            cutoff_nm: Some(1.0),
            ..NonbondedConfig::default()
        };
        assert!(Nonbonded::new("n", bad).initialize(&system).is_err());
        let periodic = system_of(vec![all_forms_molecule()], &[(0, 1)], [true; 3]);
        let no_cutoff = NonbondedConfig {
            coulomb: Some(CoulombMethod::Ewald),
            ..NonbondedConfig::default()
        };
        assert!(
            Nonbonded::new("n", no_cutoff)
                .initialize(&periodic)
                .is_err()
        );
    }
}
