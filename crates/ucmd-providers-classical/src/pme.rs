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

/// The reciprocal-space solver for a fixed grid and order.
pub struct Pme {
    order: usize,
    grid: [usize; 3],
    moduli: [Vec<f64>; 3],
    forward: [Arc<dyn Fft<f64>>; 3],
    inverse: [Arc<dyn Fft<f64>>; 3],
    q: Vec<Complex64>,
    line: Vec<Complex64>,
    scratch: Vec<Complex64>,
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
            line: vec![Complex64::new(0.0, 0.0); longest],
            scratch: vec![Complex64::new(0.0, 0.0); scratch_len],
        })
    }

    /// The grid dimensions.
    pub fn grid(&self) -> [usize; 3] {
        self.grid
    }

    /// 3D FFT of `self.q` in place, along each axis in turn.
    fn fft3(&mut self, inverse: bool) {
        let [k0, k1, k2] = self.grid;
        let strides = [k1 * k2, k2, 1];
        for axis in 0..3 {
            let (n, stride) = (self.grid[axis], strides[axis]);
            let plan = if inverse {
                &self.inverse[axis]
            } else {
                &self.forward[axis]
            };
            let total = k0 * k1 * k2;
            for start in 0..total {
                // the first element of each line along `axis`
                if (start / stride) % n != 0 {
                    continue;
                }
                for i in 0..n {
                    self.line[i] = self.q[start + i * stride];
                }
                plan.process_with_scratch(&mut self.line[..n], &mut self.scratch);
                for i in 0..n {
                    self.q[start + i * stride] = self.line[i];
                }
            }
        }
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
        // B-spline weights per atom and dimension
        let n = positions.len();
        let mut base = vec![[0usize; 3]; n];
        let mut theta = vec![[[0.0f64; 8]; 3]; n];
        let mut dtheta = vec![[[0.0f64; 8]; 3]; n];
        for (i, &r) in positions.iter().enumerate() {
            let s = hinv * r;
            for d in 0..3 {
                let frac = s[d] - libm::floor(s[d]);
                let u = frac * kf[d];
                let i0 = libm::floor(u);
                let w = u - i0;
                base[i][d] = i0 as usize % self.grid[d];
                bspline(p, w, &mut theta[i][d], &mut dtheta[i][d]);
            }
        }
        let index = |g: [usize; 3]| (g[0] * k1 + g[1]) * k2 + g[2];
        let wrap = |b: usize, j: usize, k: usize| (b + k - j % k) % k;
        // spread
        self.q
            .iter_mut()
            .for_each(|x| *x = Complex64::new(0.0, 0.0));
        for i in 0..n {
            let q = charges[i];
            if q == 0.0 {
                continue;
            }
            for a in 0..p {
                let ga = wrap(base[i][0], a, k0);
                let ta = q * theta[i][0][a];
                for b in 0..p {
                    let gb = wrap(base[i][1], b, k1);
                    let tab = ta * theta[i][1][b];
                    for (c, &tc) in theta[i][2][..p].iter().enumerate() {
                        let gc = wrap(base[i][2], c, k2);
                        self.q[index([ga, gb, gc])].re += tab * tc;
                    }
                }
            }
        }
        self.fft3(false);
        // energy, virial and the convolution in reciprocal space
        let prefactor = COULOMB_KJMOL_NM_E2 / (2.0 * PI * volume);
        let pi2_alpha2 = PI * PI / (alpha * alpha);
        let signed = |m: usize, k: usize| {
            if m <= k / 2 {
                m as f64
            } else {
                m as f64 - k as f64
            }
        };
        let mut energy = 0.0;
        let mut virial = Mat3::ZERO;
        for m0 in 0..k0 {
            for m1 in 0..k1 {
                for m2 in 0..k2 {
                    let at = index([m0, m1, m2]);
                    if m0 == 0 && m1 == 0 && m2 == 0 {
                        self.q[at] = Complex64::new(0.0, 0.0);
                        continue;
                    }
                    let mvec = signed(m0, k0) * astar[0]
                        + signed(m1, k1) * astar[1]
                        + signed(m2, k2) * astar[2];
                    let m2v = mvec.length_squared();
                    let bc = libm::exp(-pi2_alpha2 * m2v)
                        / (m2v * self.moduli[0][m0] * self.moduli[1][m1] * self.moduli[2][m2]);
                    let e_m = prefactor * bc * self.q[at].norm_sqr();
                    energy += e_m;
                    let factor = 2.0 * (1.0 / m2v + pi2_alpha2);
                    virial += (Mat3::IDENTITY - outer(mvec, mvec) * factor) * e_m;
                    // potential: ∂E/∂Q(k) = (k_e/(πV)) Re Σ_m BC(m) Q̂(m) e^{+2πimk/K}
                    self.q[at] *= 2.0 * prefactor * bc;
                }
            }
        }
        self.fft3(true);
        // forces: F_i = −q_i Σ_g φ(g) ∇(θ₁θ₂θ₃)
        for i in 0..n {
            let q = charges[i];
            if q == 0.0 {
                continue;
            }
            let mut g = Vec3::ZERO; // gradient of Σ φ θθθ with respect to (u₁, u₂, u₃)
            for a in 0..p {
                let ga = wrap(base[i][0], a, k0);
                let (ta, da) = (theta[i][0][a], dtheta[i][0][a]);
                for b in 0..p {
                    let gb = wrap(base[i][1], b, k1);
                    let (tb, db) = (theta[i][1][b], dtheta[i][1][b]);
                    for c in 0..p {
                        let gc = wrap(base[i][2], c, k2);
                        let (tc, dc) = (theta[i][2][c], dtheta[i][2][c]);
                        let phi = self.q[index([ga, gb, gc])].re;
                        g += phi * Vec3::new(da * tb * tc, ta * db * tc, ta * tb * dc);
                    }
                }
            }
            // ∂u_d/∂r = K_d a*_d
            let grad = g.x * kf[0] * astar[0] + g.y * kf[1] * astar[1] + g.z * kf[2] * astar[2];
            forces[i] -= q * grad;
        }
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
