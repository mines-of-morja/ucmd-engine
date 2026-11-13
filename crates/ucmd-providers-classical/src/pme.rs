//! Smooth particle-mesh Ewald, reciprocal part (E2.4; Essmann et al., J. Chem. Phys. 103, 8577
//! (1995)).
//!
//! E_rec = k_e/(2πV) Σ_{m≠0} exp(−π²m²/α²)/m² · B(m) · |Q̂(m)|². Here m = m₁a₁* + m₂a₂* + m₃a₃*
//! with the reciprocal vectors a* (rows of h⁻¹). Q is the charge grid spread with cardinal
//! B-splines of order p, and Q̂ its discrete Fourier transform. B(m) = 1/Π_d |b_d(m_d)|², with
//! |b(m)|² = |Σ_{k=0}^{p−2} M_p(k+1) e^{2πimk/K}|². Where that modulus vanishes (odd p, m = K/2),
//! the mean of its neighbours is used, as in OpenMM, so that matched comparisons are exact.
//!
//! - **Forces** are the exact derivatives of this energy (the gradient of the B-spline weights).
//! - **Virial:** W = Σ_m E(m)[I − 2(1/m² + π²/α²) m⊗m]. Fractional coordinates are invariant
//!   under strain, so this is exact for the PME energy.
//! - **FFT:** rustfft's scalar planner, so that results do not depend on CPU features
//!   (ADR 0010).
//! - **Threads:** charges are spread plane by plane, each plane adding its atoms in index order,
//!   so the grid is the same as a serial spread; FFT lines are independent; energy and virial are
//!   summed per plane, then over planes in order. Results are bitwise identical on any number of
//!   threads.

use rayon::prelude::*;
use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftDirection, FftPlannerScalar};
use std::f64::consts::PI;
use std::sync::Arc;
use ucmd_core::units::COULOMB_KJMOL_NM_E2;
use ucmd_core::{Cell, Mat3, Vec3};
use ucmd_force::{ForceError, outer};

/// Smallest FFT-friendly size ≥ n: a product of 2, 3, 5 and 7 (OpenMM's rule).
pub fn legal_size(n: usize) -> usize {
    let mut m = n.max(1);
    loop {
        let mut r = m;
        for f in [2, 3, 5, 7] {
            while r % f == 0 {
                r /= f;
            }
        }
        if r == 1 {
            return m;
        }
        m += 1;
    }
}

/// Grid from the tolerance, OpenMM's rule: n_d = ⌈2α·h_d / (3·tol^{1/5})⌉, made FFT-friendly;
/// h_d are the diagonal elements of the reduced cell (the heights).
pub fn grid_for(cell: &Cell, alpha: f64, tolerance: f64) -> [usize; 3] {
    let h = cell.h();
    let scale = 2.0 * alpha / (3.0 * libm::pow(tolerance, 0.2));
    [h.col(0)[0], h.col(1)[1], h.col(2)[2]].map(|hd| legal_size(libm::ceil(scale * hd) as usize))
}

/// B-spline weights M_p(w + j) and their derivatives for j = 0…p−1 (grid points i₀ − j).
fn bspline(order: usize, w: f64, theta: &mut [f64], dtheta: &mut [f64]) {
    // A_n[j] = M_n(w + j); A_1 = [1]; A_n[j] = ((w+j)A_{n−1}[j] + (n−w−j)A_{n−1}[j−1])/(n−1)
    let mut a = [0.0f64; 8];
    a[0] = 1.0;
    for n in 2..=order {
        if n == order {
            // M_p'(w + j) = M_{p−1}(w + j) − M_{p−1}(w + j − 1)
            for j in 0..order {
                let here = if j < n - 1 { a[j] } else { 0.0 };
                let below = if j >= 1 { a[j - 1] } else { 0.0 };
                dtheta[j] = here - below;
            }
        }
        let inv = 1.0 / (n - 1) as f64;
        for j in (0..n).rev() {
            let here = if j < n - 1 { a[j] } else { 0.0 };
            let below = if j >= 1 { a[j - 1] } else { 0.0 };
            a[j] = ((w + j as f64) * here + (n as f64 - w - j as f64) * below) * inv;
        }
    }
    theta[..order].copy_from_slice(&a[..order]);
}

/// |b(m)|² for every m of one dimension (with OpenMM's treatment of zeros).
fn bspline_moduli(order: usize, k: usize) -> Vec<f64> {
    // M_p at the integers 1 … p−1: the weights for w = 0 are M_p(j), j = 0…p−1.
    let mut theta = [0.0; 8];
    let mut dtheta = [0.0; 8];
    bspline(order, 0.0, &mut theta, &mut dtheta);
    let mut data = vec![0.0; k];
    for (j, value) in theta.iter().enumerate().take(order).skip(1) {
        data[(j - 1) % k] += value; // M_p(j) at index j − 1 (a shift changes only the phase)
    }
    let mut moduli: Vec<f64> = (0..k)
        .map(|m| {
            let (mut c, mut s) = (0.0, 0.0);
            for (j, &d) in data.iter().enumerate() {
                let arg = 2.0 * PI * (m * j) as f64 / k as f64;
                c += d * libm::cos(arg);
                s += d * libm::sin(arg);
            }
            c * c + s * s
        })
        .collect();
    for m in 0..k {
        if moduli[m] < 1e-7 {
            moduli[m] = 0.5 * (moduli[(m + k - 1) % k] + moduli[(m + 1) % k]);
        }
    }
    moduli
}

/// Columns per task in the FFT along the first axis.
const COLUMN_BLOCK: usize = 64;

/// B-spline data of one atom: the first grid point per dimension and the weights.
#[derive(Clone, Copy)]
struct Spline {
    base: [usize; 3],
    theta: [[f64; 8]; 3],
    dtheta: [[f64; 8]; 3],
}

/// The reciprocal-space solver for a fixed grid and order.
pub struct Pme {
    order: usize,
    grid: [usize; 3],
    moduli: [Vec<f64>; 3],
    forward: [Arc<dyn Fft<f64>>; 3],
    inverse: [Arc<dyn Fft<f64>>; 3],
    q: Vec<Complex64>,
    longest: usize,
    scratch_len: usize,
}

impl Pme {
    /// A solver for `grid` points per lattice direction and B-spline `order` (4…6).
    pub fn new(order: usize, grid: [usize; 3]) -> Result<Self, ForceError> {
        if !(4..=6).contains(&order) {
            return Err(ForceError::InvalidInput(format!(
                "pme_order must be 4, 5 or 6, got {order}"
            )));
        }
        if grid.iter().any(|&k| k < order) {
            return Err(ForceError::InvalidInput(format!(
                "every PME grid dimension must be at least the order {order}, got {grid:?}"
            )));
        }
        let mut planner = FftPlannerScalar::new();
        let forward = grid.map(|k| planner.plan_fft(k, FftDirection::Forward));
        let inverse = grid.map(|k| planner.plan_fft(k, FftDirection::Inverse));
        let longest = *grid.iter().max().unwrap_or(&1);
        let scratch_len = forward
            .iter()
            .chain(inverse.iter())
            .map(|f| f.get_inplace_scratch_len())
            .max()
            .unwrap_or(0);
        Ok(Self {
            order,
            grid,
            moduli: grid.map(|k| bspline_moduli(order, k)),
            forward,
            inverse,
            q: vec![Complex64::new(0.0, 0.0); grid[0] * grid[1] * grid[2]],
            longest,
            scratch_len,
        })
    }

    /// The grid dimensions.
    pub fn grid(&self) -> [usize; 3] {
        self.grid
    }

    /// 3D FFT of `self.q` in place, along each axis in turn. Every line is transformed on its
    /// own, so the result does not depend on how the lines are shared out among threads.
    fn fft3(&mut self, inverse: bool) {
        let [k0, k1, k2] = self.grid;
        let plans = if inverse {
            &self.inverse
        } else {
            &self.forward
        };
        let (longest, scratch_len) = (self.longest, self.scratch_len);
        let buffers = || {
            (
                vec![Complex64::new(0.0, 0.0); longest],
                vec![Complex64::new(0.0, 0.0); scratch_len],
            )
        };
        // axis 0: columns across the planes, handed out in blocks of columns
        let plan = &plans[0];
        let mut blocks: Vec<Vec<&mut [Complex64]>> = (0..(k1 * k2).div_ceil(COLUMN_BLOCK))
            .map(|_| Vec::with_capacity(k0))
            .collect();
        for plane in self.q.chunks_mut(k1 * k2) {
            for (block, piece) in blocks.iter_mut().zip(plane.chunks_mut(COLUMN_BLOCK)) {
                block.push(piece);
            }
        }
        blocks
            .par_iter_mut()
            .for_each_init(buffers, |(line, scratch), pieces| {
                for column in 0..pieces[0].len() {
                    for (x, piece) in line.iter_mut().zip(pieces.iter()) {
                        *x = piece[column];
                    }
                    plan.process_with_scratch(&mut line[..k0], scratch);
                    for (x, piece) in line.iter().zip(pieces.iter_mut()) {
                        piece[column] = *x;
                    }
                }
            });
        // axis 1: strided lines within each plane
        let plan = &plans[1];
        self.q
            .par_chunks_mut(k1 * k2)
            .for_each_init(buffers, |(line, scratch), plane| {
                for c in 0..k2 {
                    for (b, x) in line[..k1].iter_mut().enumerate() {
                        *x = plane[b * k2 + c];
                    }
                    plan.process_with_scratch(&mut line[..k1], scratch);
                    for (b, x) in line[..k1].iter().enumerate() {
                        plane[b * k2 + c] = *x;
                    }
                }
            });
        // axis 2: contiguous lines
        let plan = &plans[2];
        self.q
            .par_chunks_mut(k2)
            .for_each_init(buffers, |(_, scratch), line| {
                plan.process_with_scratch(line, scratch);
            });
    }

    /// Reciprocal energy and virial; forces are added to `forces`.
    pub fn compute(
        &mut self,
        positions: &[Vec3],
        charges: &[f64],
        cell: &Cell,
        alpha: f64,
        forces: &mut [Vec3],
    ) -> (f64, Mat3) {
        let p = self.order;
        let [k0, k1, k2] = self.grid;
        let kf = [k0 as f64, k1 as f64, k2 as f64];
        let hinv = cell.h().inverse();
        let astar = [hinv.row(0), hinv.row(1), hinv.row(2)];
        let volume = cell.volume();
        let grid = self.grid;
        // B-spline weights per atom and dimension
        let splines: Vec<Spline> = positions
            .par_iter()
            .map(|&r| {
                let s = hinv * r;
                let mut spline = Spline {
                    base: [0; 3],
                    theta: [[0.0; 8]; 3],
                    dtheta: [[0.0; 8]; 3],
                };
                for d in 0..3 {
                    let frac = s[d] - libm::floor(s[d]);
                    let u = frac * kf[d];
                    let i0 = libm::floor(u);
                    let w = u - i0;
                    spline.base[d] = i0 as usize % grid[d];
                    bspline(p, w, &mut spline.theta[d], &mut spline.dtheta[d]);
                }
                spline
            })
            .collect();
        let wrap = |b: usize, j: usize, k: usize| (b + k - j % k) % k;
        // spread, plane by plane; each plane adds its atoms in index order, as a serial spread
        self.q
            .par_chunks_mut(k1 * k2)
            .enumerate()
            .for_each(|(ga, plane)| {
                plane.iter_mut().for_each(|x| *x = Complex64::new(0.0, 0.0));
                for (s, &q) in splines.iter().zip(charges) {
                    let a = (s.base[0] + k0 - ga) % k0; // ga = base₀ − a (mod K₀)
                    if q == 0.0 || a >= p {
                        continue;
                    }
                    let ta = q * s.theta[0][a];
                    for b in 0..p {
                        let gb = wrap(s.base[1], b, k1);
                        let tab = ta * s.theta[1][b];
                        for (c, &tc) in s.theta[2][..p].iter().enumerate() {
                            let gc = wrap(s.base[2], c, k2);
                            plane[gb * k2 + gc].re += tab * tc;
                        }
                    }
                }
            });
        self.fft3(false);
        // energy, virial and the convolution in reciprocal space, summed per plane
        let prefactor = COULOMB_KJMOL_NM_E2 / (2.0 * PI * volume);
        let pi2_alpha2 = PI * PI / (alpha * alpha);
        let signed = |m: usize, k: usize| {
            if m <= k / 2 {
                m as f64
            } else {
                m as f64 - k as f64
            }
        };
        let moduli = &self.moduli;
        let planes: Vec<(f64, Mat3)> = self
            .q
            .par_chunks_mut(k1 * k2)
            .enumerate()
            .map(|(m0, plane)| {
                let mut energy = 0.0;
                let mut virial = Mat3::ZERO;
                for m1 in 0..k1 {
                    for m2 in 0..k2 {
                        let at = m1 * k2 + m2;
                        if m0 == 0 && m1 == 0 && m2 == 0 {
                            plane[at] = Complex64::new(0.0, 0.0);
                            continue;
                        }
                        let mvec = signed(m0, k0) * astar[0]
                            + signed(m1, k1) * astar[1]
                            + signed(m2, k2) * astar[2];
                        let m2v = mvec.length_squared();
                        let bc = libm::exp(-pi2_alpha2 * m2v)
                            / (m2v * moduli[0][m0] * moduli[1][m1] * moduli[2][m2]);
                        let e_m = prefactor * bc * plane[at].norm_sqr();
                        energy += e_m;
                        let factor = 2.0 * (1.0 / m2v + pi2_alpha2);
                        virial += (Mat3::IDENTITY - outer(mvec, mvec) * factor) * e_m;
                        // potential: ∂E/∂Q(k) = (k_e/(πV)) Re Σ_m BC(m) Q̂(m) e^{+2πimk/K}
                        plane[at] *= 2.0 * prefactor * bc;
                    }
                }
                (energy, virial)
            })
            .collect();
        let mut energy = 0.0;
        let mut virial = Mat3::ZERO;
        for (e, w) in planes {
            energy += e;
            virial += w;
        }
        self.fft3(true);
        // forces: F_i = −q_i Σ_g φ(g) ∇(θ₁θ₂θ₃), each atom on its own
        let phi_grid = &self.q;
        let index = |g: [usize; 3]| (g[0] * k1 + g[1]) * k2 + g[2];
        forces
            .par_iter_mut()
            .zip(splines.par_iter().zip(charges.par_iter()))
            .for_each(|(force, (s, &q))| {
                if q == 0.0 {
                    return;
                }
                let mut g = Vec3::ZERO; // gradient of Σ φ θθθ with respect to (u₁, u₂, u₃)
                for a in 0..p {
                    let ga = wrap(s.base[0], a, k0);
                    let (ta, da) = (s.theta[0][a], s.dtheta[0][a]);
                    for b in 0..p {
                        let gb = wrap(s.base[1], b, k1);
                        let (tb, db) = (s.theta[1][b], s.dtheta[1][b]);
                        for c in 0..p {
                            let gc = wrap(s.base[2], c, k2);
                            let (tc, dc) = (s.theta[2][c], s.dtheta[2][c]);
                            let phi = phi_grid[index([ga, gb, gc])].re;
                            g += phi * Vec3::new(da * tb * tc, ta * db * tc, ta * tb * dc);
                        }
                    }
                }
                // ∂u_d/∂r = K_d a*_d
                let grad = g.x * kf[0] * astar[0] + g.y * kf[1] * astar[1] + g.z * kf[2] * astar[2];
                *force -= q * grad;
            });
        (energy, virial)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bsplines_partition_unity_and_match_the_definition() {
        for order in 4..=6 {
            for w in [0.0, 0.1, 0.37, 0.5, 0.999] {
                let (mut t, mut d) = ([0.0; 8], [0.0; 8]);
                bspline(order, w, &mut t, &mut d);
                let sum: f64 = t[..order].iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-14,
                    "order {order}, w {w}: Σθ = {sum}"
                );
                let dsum: f64 = d[..order].iter().sum();
                assert!(dsum.abs() < 1e-14, "Σ dθ = {dsum}");
                // finite difference of the weights
                let h = 1e-6;
                let (mut tp, mut tm, mut dd) = ([0.0; 8], [0.0; 8], [0.0; 8]);
                if w + h < 1.0 && w - h >= 0.0 {
                    bspline(order, w + h, &mut tp, &mut dd);
                    bspline(order, w - h, &mut tm, &mut dd);
                    for j in 0..order {
                        let fd = (tp[j] - tm[j]) / (2.0 * h);
                        assert!((fd - d[j]).abs() < 1e-8, "order {order}, w {w}, j {j}");
                    }
                }
            }
        }
        // M_4 at the integers 1, 2, 3 is 1/6, 4/6, 1/6
        let (mut t, mut d) = ([0.0; 8], [0.0; 8]);
        bspline(4, 0.0, &mut t, &mut d);
        let expected = [0.0, 1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0];
        for j in 0..4 {
            assert!((t[j] - expected[j]).abs() < 1e-15, "{t:?}");
        }
    }

    #[test]
    fn grid_sizes_are_fft_friendly() {
        assert_eq!(legal_size(97), 98);
        assert_eq!(legal_size(121), 125);
        assert_eq!(legal_size(64), 64);
        let moduli = bspline_moduli(5, 32);
        assert!(
            moduli.iter().all(|&m| m > 1e-7),
            "the Nyquist zero of order 5 is repaired"
        );
    }
}

#[cfg(test)]
mod against_ewald {
    use super::*;
    use crate::nonbonded::{ewald_kmax, ewald_reciprocal};

    #[test]
    fn a_charge_pair_converges_to_the_ewald_sum() {
        let cell = Cell::new(
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            [true; 3],
        )
        .unwrap();
        let positions = [Vec3::new(0.31, 0.47, 0.83), Vec3::new(1.12, 0.95, 0.36)];
        let charges = [1.0, -1.0];
        let alpha = 3.0;
        let mut fe = vec![Vec3::ZERO; 2];
        let (ee, we) = ewald_reciprocal(
            &positions,
            &charges,
            &cell,
            alpha,
            ewald_kmax(&cell, alpha, 1e-14),
            &mut fe,
        );
        let mut previous = f64::INFINITY;
        for (order, k) in [(4, 32), (5, 32), (6, 32), (6, 64)] {
            let mut pme = Pme::new(order, [k, k, k]).unwrap();
            let mut fp = vec![Vec3::ZERO; 2];
            let (ep, wp) = pme.compute(&positions, &charges, &cell, alpha, &mut fp);
            let rel = (ep - ee).abs() / ee.abs();
            assert!(rel < previous, "order {order}, K {k}: {rel:.2e}");
            previous = rel;
            if k == 64 {
                assert!(rel < 1e-8, "{rel:.2e}");
                assert!((fp[0] - fe[0]).length() < 1e-6 * fe[0].length());
                let max = |m: Mat3| {
                    (0..3)
                        .map(|c| m.col(c).abs().max_element())
                        .fold(0.0, f64::max)
                };
                assert!(max(wp - we) < 1e-7 * max(we), "virial {wp} vs {we}");
            }
        }
    }
}
