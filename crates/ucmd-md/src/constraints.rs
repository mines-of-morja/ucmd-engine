//! Bond constraints by RATTLE (E2.5; Andersen, J. Comput. Phys. 52, 24 (1983)).
//!
//! Two operators of the step (invariant 10), never code inside kick or drift:
//! - [`Constraints::shake`] after a drift: moves the drifted positions back onto every constraint
//!   length along the bond vectors of the start of the drift, and corrects the velocities by the
//!   same displacement over the drift time;
//! - [`Constraints::rattle`] at the end of the step: removes the velocity components along the
//!   constraints.
//!
//! Both iterate over the constraints in package order (single-threaded, so bitwise reproducible,
//! ADR 0010) until every constraint holds to the relative tolerance, and check before they change
//! anything: a state that already satisfies the constraints is left bit for bit unchanged (restart).
//! No convergence within the iteration limit is an error naming the constraint (invariant 7).
//!
//! The constraint virial for pressures, W_c = Σ r_ij ⊗ g_ij (units.md §4), comes from
//! [`Constraints::force_virial`]: the exact instantaneous constraint forces of a state, solved from
//! d²|r_ij|²/dt² = 0, so it does not depend on the integrator. An estimate from the final velocity
//! stage was right for velocity Verlet but caught only half the centripetal force in BAOAB, whose
//! last drift is half a step (pressure 4/3 too high for rigid dimers, found 2026-11-20).
//!
//! Units: nm, ps, amu; virial in kJ/mol.

use ucmd_core::{Cell, Mat3, System, Vec3};

/// Relative tolerance of PLAN.md E2.5.
pub const DEFAULT_TOLERANCE: f64 = 1e-10;
/// Iterations before giving up.
pub const MAX_ITERATIONS: usize = 1000;
/// Velocity scale of the relative velocity test, nm/ps: a constraint holds when its radial
/// relative velocity is below tolerance × this. Fixed, not taken from the current velocities, so
/// that a satisfied state passes the same test again (bitwise restarts); atomic velocities at room
/// temperature are of this order.
pub const VELOCITY_SCALE_NM_PS: f64 = 1.0;

/// Why a constraint could not be satisfied.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error(
    "constraint {index} (atoms {i}-{j}, {length_nm} nm) did not converge in {iterations} iterations ({stage})"
)]
pub struct ConstraintError {
    /// Index in the global constraint list (package order).
    pub index: usize,
    /// First atom (global index).
    pub i: usize,
    /// Second atom (global index).
    pub j: usize,
    /// Constraint length in nm.
    pub length_nm: f64,
    /// Iterations done.
    pub iterations: usize,
    /// "positions" or "velocities".
    pub stage: &'static str,
}

/// The model constraints of a system, as global atom pairs with lengths (nm).
#[derive(Debug, Clone, PartialEq)]
pub struct Constraints {
    pairs: Vec<(usize, usize, f64)>,
    tolerance: f64,
}

fn separation(cell: Option<&Cell>, a: Vec3, b: Vec3) -> Vec3 {
    let d = a - b;
    cell.map_or(d, |c| c.minimum_image(d))
}

impl Constraints {
    /// All model constraints of the system, in canonical order (components → instances →
    /// template order).
    pub fn from_system(system: &System) -> Self {
        let mut pairs = Vec::with_capacity(system.n_constraints());
        for instance in &system.instances {
            let first = instance.first_atom;
            for c in &system.molecule_types[instance.molecule_type].constraints {
                pairs.push((first + c.atoms[0], first + c.atoms[1], c.length_nm));
            }
        }
        Self {
            pairs,
            tolerance: DEFAULT_TOLERANCE,
        }
    }

    /// Constraints from explicit pairs (global atoms, length in nm), e.g. for tests.
    pub fn from_pairs(pairs: Vec<(usize, usize, f64)>) -> Self {
        Self {
            pairs,
            tolerance: DEFAULT_TOLERANCE,
        }
    }

    /// Number of constraints.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// True without constraints.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Largest relative violation |r − d|/d over all constraints.
    pub fn max_violation(&self, x: &[Vec3], cell: Option<&Cell>) -> f64 {
        self.pairs
            .iter()
            .map(|&(i, j, d)| (separation(cell, x[i], x[j]).length() - d).abs() / d)
            .fold(0.0, f64::max)
    }

    /// Position stage. `x` holds the drifted positions, `reference` the positions at the start
    /// of the drift (their bond vectors give the directions of the corrections), `dt_ps` the
    /// drift time; `v` gets Δx/dt. Returns the largest relative correction |Δr|/d of the call.
    pub fn shake(
        &self,
        x: &mut [Vec3],
        v: &mut [Vec3],
        reference: &[Vec3],
        inverse_masses: &[f64],
        cell: Option<&Cell>,
        dt_ps: f64,
    ) -> Result<f64, ConstraintError> {
        let mut largest: f64 = 0.0;
        for iteration in 0..MAX_ITERATIONS {
            let mut converged = true;
            for &(i, j, d) in &self.pairs {
                let r = separation(cell, x[i], x[j]);
                // the documented criterion exactly: |r − d|/d ≤ tolerance
                if (r.length() - d).abs() <= self.tolerance * d {
                    continue;
                }
                let diff = d * d - r.length_squared();
                converged = false;
                let r_ref = separation(cell, reference[i], reference[j]);
                let (wi, wj) = (inverse_masses[i], inverse_masses[j]);
                let g = diff / (2.0 * (wi + wj) * r.dot(r_ref));
                let dxi = r_ref * (g * wi);
                let dxj = r_ref * (-g * wj);
                x[i] += dxi;
                x[j] += dxj;
                v[i] += dxi / dt_ps;
                v[j] += dxj / dt_ps;
                largest = largest.max(dxi.length().max(dxj.length()) / d);
            }
            if converged {
                return Ok(largest);
            }
            if iteration + 1 == MAX_ITERATIONS {
                break;
            }
        }
        Err(self.failure(x, cell, "positions"))
    }

    /// Velocity stage after the last operator that changes velocities. `dt_half_ps` is the time
    /// of the final half kick (dt/2). Returns the constraint virial (kJ/mol, units.md §4).
    pub fn rattle(
        &self,
        x: &[Vec3],
        v: &mut [Vec3],
        masses: &[f64],
        inverse_masses: &[f64],
        cell: Option<&Cell>,
        dt_half_ps: f64,
    ) -> Result<Mat3, ConstraintError> {
        let mut virial = Mat3::ZERO;
        for iteration in 0..MAX_ITERATIONS {
            let mut converged = true;
            for &(i, j, d) in &self.pairs {
                let r = separation(cell, x[i], x[j]);
                let rv = r.dot(v[i] - v[j]);
                if rv.abs() <= self.tolerance * d * VELOCITY_SCALE_NM_PS {
                    continue;
                }
                converged = false;
                let (wi, wj) = (inverse_masses[i], inverse_masses[j]);
                let k = rv / ((wi + wj) * r.length_squared());
                let dvi = r * (-k * wi);
                v[i] += dvi;
                v[j] += r * (k * wj);
                // constraint force on i over the half kick: g = m Δv / (dt/2); W += r ⊗ g
                let g = dvi * (masses[i] / dt_half_ps);
                virial += Mat3::from_cols(r * g.x, r * g.y, r * g.z);
            }
            if converged {
                return Ok(virial);
            }
            if iteration + 1 == MAX_ITERATIONS {
                break;
            }
        }
        Err(self.failure(x, cell, "velocities"))
    }

    /// The exact instantaneous constraint forces of a state on the constraints: g_i (kJ/mol/nm)
    /// per atom and λ per constraint (g = ±λ r_ij), such that d²|r_ij|²/dt² = 0, i.e.
    /// |v_ij|² + r_ij·(a_i − a_j) = 0 with a = (F + g)/m. Solved by Gauss–Seidel to the relative
    /// tolerance. `forces` are the provider's forces, `v` velocities on the constraints.
    pub fn constraint_forces(
        &self,
        x: &[Vec3],
        v: &[Vec3],
        forces: &[Vec3],
        inverse_masses: &[f64],
        cell: Option<&Cell>,
    ) -> Result<(Vec<Vec3>, Vec<f64>), ConstraintError> {
        let mut g = vec![Vec3::ZERO; x.len()];
        let mut lambda = vec![0.0; self.pairs.len()];
        for iteration in 0..MAX_ITERATIONS {
            let mut converged = true;
            for (k, &(i, j, _)) in self.pairs.iter().enumerate() {
                let r = separation(cell, x[i], x[j]);
                let (wi, wj) = (inverse_masses[i], inverse_masses[j]);
                let vv = (v[i] - v[j]).length_squared();
                let af = r.dot(forces[i] * wi - forces[j] * wj);
                let residual = vv + af + r.dot(g[i] * wi - g[j] * wj);
                if residual.abs() <= self.tolerance * (vv + af.abs() + 1e-30) {
                    continue;
                }
                converged = false;
                let delta = -residual / ((wi + wj) * r.length_squared());
                lambda[k] += delta;
                g[i] += r * delta;
                g[j] -= r * delta;
            }
            if converged {
                return Ok((g, lambda));
            }
            if iteration + 1 == MAX_ITERATIONS {
                break;
            }
        }
        Err(self.failure(x, cell, "constraint forces"))
    }

    /// Constraint virial W_c = Σ r_ij ⊗ g_ij (kJ/mol) of the exact instantaneous constraint
    /// forces ([`Self::constraint_forces`]); independent of the integrator.
    pub fn force_virial(
        &self,
        x: &[Vec3],
        v: &[Vec3],
        forces: &[Vec3],
        inverse_masses: &[f64],
        cell: Option<&Cell>,
    ) -> Result<Mat3, ConstraintError> {
        let (_, lambda) = self.constraint_forces(x, v, forces, inverse_masses, cell)?;
        let mut w = Mat3::ZERO;
        for (k, &(i, j, _)) in self.pairs.iter().enumerate() {
            let r = separation(cell, x[i], x[j]);
            let f = r * lambda[k];
            w += Mat3::from_cols(r * f.x, r * f.y, r * f.z);
        }
        Ok(w)
    }

    /// Forces projected onto the constraint manifold (kJ/mol/nm): F + g with the constraint
    /// forces of a state at rest, so that no constraint is accelerated. Zero projected forces mean
    /// a constrained minimum (FIRE with constraints).
    pub fn project_forces(
        &self,
        x: &[Vec3],
        forces: &[Vec3],
        inverse_masses: &[f64],
        cell: Option<&Cell>,
    ) -> Result<Vec<Vec3>, ConstraintError> {
        let rest = vec![Vec3::ZERO; x.len()];
        let (g, _) = self.constraint_forces(x, &rest, forces, inverse_masses, cell)?;
        Ok(forces.iter().zip(&g).map(|(f, g)| *f + *g).collect())
    }

    fn failure(&self, x: &[Vec3], cell: Option<&Cell>, stage: &'static str) -> ConstraintError {
        let (index, &(i, j, d)) = self
            .pairs
            .iter()
            .enumerate()
            .max_by(|a, b| {
                let va = (separation(cell, x[a.1.0], x[a.1.1]).length() - a.1.2).abs();
                let vb = (separation(cell, x[b.1.0], x[b.1.1]).length() - b.1.2).abs();
                va.total_cmp(&vb)
            })
            .unwrap_or((0, &(0, 0, 0.0)));
        ConstraintError {
            index,
            i,
            j,
            length_nm: d,
            iterations: MAX_ITERATIONS,
            stage,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn triangle() -> (Vec<Vec3>, Vec<f64>, Constraints) {
        // a water-like triangle: O at the origin, two H at 0.1 nm, H–H 0.1633 nm
        let x = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(-0.033_331, 0.094_281, 0.0),
        ];
        let d_hh = (x[1] - x[2]).length();
        let c = Constraints::from_pairs(vec![(0, 1, 0.1), (0, 2, 0.1), (1, 2, d_hh)]);
        (x, vec![15.9994, 1.008, 1.008], c)
    }

    #[test]
    fn shake_restores_every_length_and_corrects_the_velocities() {
        let (x0, m, c) = triangle();
        let w: Vec<f64> = m.iter().map(|m| 1.0 / m).collect();
        let v0 = vec![
            Vec3::new(0.3, -0.2, 0.1),
            Vec3::new(-4.0, 3.0, 1.0),
            Vec3::new(2.0, -5.0, -1.5),
        ];
        let dt = 0.002;
        let mut x: Vec<Vec3> = x0.iter().zip(&v0).map(|(x, v)| *x + *v * dt).collect();
        let mut v = v0.clone();
        let x_drifted = x.clone();
        c.shake(&mut x, &mut v, &x0, &w, None, dt).unwrap();
        assert!(c.max_violation(&x, None) < 1e-10);
        for k in 0..3 {
            // the velocity change is the position change over the drift time
            let dv = v[k] - v0[k];
            let dx = x[k] - x_drifted[k];
            assert!((dv * dt - dx).length() < 1e-15);
        }
        // momentum is unchanged: the corrections are internal forces
        let p0: Vec3 = v0.iter().zip(&m).map(|(v, m)| *v * *m).sum();
        let p1: Vec3 = v.iter().zip(&m).map(|(v, m)| *v * *m).sum();
        assert!((p1 - p0).length() < 1e-12);
    }

    #[test]
    fn rattle_removes_the_velocities_along_the_constraints() {
        let (x, m, c) = triangle();
        let w: Vec<f64> = m.iter().map(|m| 1.0 / m).collect();
        let mut v = vec![
            Vec3::new(0.3, -0.2, 0.1),
            Vec3::new(-4.0, 3.0, 1.0),
            Vec3::new(2.0, -5.0, -1.5),
        ];
        let p0: Vec3 = v.iter().zip(&m).map(|(v, m)| *v * *m).sum();
        c.rattle(&x, &mut v, &m, &w, None, 0.001).unwrap();
        for &(i, j, _) in &c.pairs {
            assert!((x[i] - x[j]).dot(v[i] - v[j]).abs() < 1e-10 * 5.0);
        }
        let p1: Vec3 = v.iter().zip(&m).map(|(v, m)| *v * *m).sum();
        assert!((p1 - p0).length() < 1e-12);
    }

    #[test]
    fn a_satisfied_state_is_left_bit_for_bit_unchanged() {
        let (x0, m, c) = triangle();
        let w: Vec<f64> = m.iter().map(|m| 1.0 / m).collect();
        let mut x = x0.clone();
        let mut v = vec![Vec3::ZERO; 3];
        c.shake(&mut x, &mut v, &x0, &w, None, 0.001).unwrap();
        c.rattle(&x, &mut v, &m, &w, None, 0.001).unwrap();
        let (x1, v1) = (x.clone(), v.clone());
        c.shake(&mut x, &mut v, &x1, &w, None, 0.001).unwrap();
        let virial = c.rattle(&x, &mut v, &m, &w, None, 0.001).unwrap();
        assert_eq!(x, x1);
        assert_eq!(v, v1);
        assert_eq!(virial, Mat3::ZERO);
    }

    #[test]
    fn impossible_constraints_are_an_error_naming_one() {
        // three collinear atoms cannot be 0.1, 0.1 and 0.5 nm apart
        let x0 = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.2, 0.0, 0.0),
        ];
        let c = Constraints::from_pairs(vec![(0, 1, 0.1), (1, 2, 0.1), (0, 2, 0.5)]);
        let mut x = x0.clone();
        let mut v = vec![Vec3::ZERO; 3];
        let e = c
            .shake(&mut x, &mut v, &x0, &[1.0, 1.0, 1.0], None, 0.001)
            .unwrap_err();
        assert_eq!(e.stage, "positions");
        assert!(e.to_string().contains("did not converge"));
    }

    #[test]
    fn minimum_image_is_used_across_the_cell() {
        let cell = Cell::new(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [true; 3],
        )
        .unwrap();
        // atom 1 sits across the boundary: its nearest image is 0.12 nm away
        let x0 = vec![Vec3::new(0.95, 0.5, 0.5), Vec3::new(0.07, 0.5, 0.5)];
        let c = Constraints::from_pairs(vec![(0, 1, 0.1)]);
        let mut x = x0.clone();
        let mut v = vec![Vec3::ZERO; 2];
        c.shake(&mut x, &mut v, &x0, &[1.0, 1.0], Some(&cell), 0.001)
            .unwrap();
        assert!(c.max_violation(&x, Some(&cell)) < 1e-10);
        assert!((x[0].x - 0.96).abs() < 1e-9 && (x[1].x - 0.06).abs() < 1e-9);
    }
}
