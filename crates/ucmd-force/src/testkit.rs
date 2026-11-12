//! Generic conformance checks for every force provider (spec/force-provider.md §6).
//!
//! Each check returns a report with the numbers, so tests can assert on them and print them.

use crate::provider::{EvalRequest, ForceError, ForceProvider, ForceResult};
use ucmd_core::{Cell, Mat3, Vec3};

/// Options of [`fd_force_check`].
#[derive(Debug, Clone)]
pub struct FdOptions {
    /// Finite-difference step sizes to try (nm); the best one is reported.
    pub steps_nm: Vec<f64>,
    /// Force magnitude (kJ mol⁻¹ nm⁻¹) below which errors are measured relative to this floor.
    pub min_force: f64,
    /// Atoms to check (all if `None`), to keep large systems affordable.
    pub atoms: Option<Vec<usize>>,
}

impl Default for FdOptions {
    fn default() -> Self {
        Self {
            steps_nm: vec![1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
            min_force: 1e-3,
            atoms: None,
        }
    }
}

/// Outcome of [`fd_force_check`].
#[derive(Debug, Clone, PartialEq)]
pub struct FdReport {
    /// Step size with the smallest error (nm).
    pub best_step_nm: f64,
    /// Largest error at the best step: |F_fd − F| / max(|F|, min_force).
    pub max_rel_error: f64,
    /// (step size, largest error) for every step tried.
    pub per_step: Vec<(f64, f64)>,
    /// Number of force components compared.
    pub components: usize,
}

fn energy(
    provider: &mut dyn ForceProvider,
    positions: &[Vec3],
    cell: Option<&Cell>,
) -> Result<f64, ForceError> {
    let mut out = ForceResult::default();
    out.reset(positions.len());
    let request = EvalRequest {
        need_virial: false,
        ..EvalRequest::everything()
    };
    provider.evaluate(positions, cell, &request, &mut out)?;
    out.energy
        .ok_or_else(|| ForceError::Contract("energy not returned".into()))
}

/// Evaluate everything once (energy, forces and, if available, virial).
pub fn evaluate_all(
    provider: &mut dyn ForceProvider,
    positions: &[Vec3],
    cell: Option<&Cell>,
) -> Result<ForceResult, ForceError> {
    let mut out = ForceResult::default();
    out.reset(positions.len());
    let request = EvalRequest {
        need_virial: provider.capabilities().virial,
        ..EvalRequest::everything()
    };
    provider.evaluate(positions, cell, &request, &mut out)?;
    Ok(out)
}

/// Compare analytic forces with central differences of the energy.
pub fn fd_force_check(
    provider: &mut dyn ForceProvider,
    positions: &[Vec3],
    cell: Option<&Cell>,
    options: &FdOptions,
) -> Result<FdReport, ForceError> {
    let analytic = evaluate_all(provider, positions, cell)?.forces;
    let atoms: Vec<usize> = options
        .atoms
        .clone()
        .unwrap_or_else(|| (0..positions.len()).collect());
    let mut moved = positions.to_vec();
    let mut per_step = Vec::new();
    for &h in &options.steps_nm {
        let mut worst = 0.0f64;
        for &i in &atoms {
            for d in 0..3 {
                let original = moved[i][d];
                moved[i][d] = original + h;
                let plus = energy(provider, &moved, cell)?;
                moved[i][d] = original - h;
                let minus = energy(provider, &moved, cell)?;
                moved[i][d] = original;
                let fd = -(plus - minus) / (2.0 * h);
                let exact = analytic[i][d];
                worst = worst.max((fd - exact).abs() / exact.abs().max(options.min_force));
            }
        }
        per_step.push((h, worst));
    }
    let (best_step_nm, max_rel_error) =
        per_step
            .iter()
            .copied()
            .fold((f64::NAN, f64::INFINITY), |best, s| {
                if s.1 < best.1 { s } else { best }
            });
    Ok(FdReport {
        best_step_nm,
        max_rel_error,
        per_step,
        components: 3 * atoms.len(),
    })
}

/// Outcome of [`fd_virial_check`].
#[derive(Debug, Clone, PartialEq)]
pub struct VirialReport {
    /// Largest |W_fd − W| over the checked components, divided by max(max|W|, 1e-3).
    pub max_rel_error: f64,
    /// |W − Wᵀ| / max(max|W|, 1e-3) of the analytic virial.
    pub asymmetry: f64,
}

/// Apply the strain component ε_ab = δ in the row-vector convention of spec/units.md §4:
/// r_b → r_b + δ·r_a.
fn strain(m: Vec3, a: usize, b: usize, delta: f64) -> Vec3 {
    let mut out = m;
    out[b] += delta * m[a];
    out
}

/// Compare the analytic virial with central differences of the energy under homogeneous strain
/// ε_ab = ±δ, applied as r_b → r_b + ε_ab·r_a to positions and lattice vectors (spec/units.md §4).
/// The diagonal and the lower components are probed; they keep the cell lower-triangular. The
/// upper ones need a rotated cell and are not probed. `asymmetry` reports |W − Wᵀ|, which is zero
/// for rotation-invariant potentials.
pub fn fd_virial_check(
    provider: &mut dyn ForceProvider,
    positions: &[Vec3],
    cell: Option<&Cell>,
    delta: f64,
) -> Result<VirialReport, ForceError> {
    let w = evaluate_all(provider, positions, cell)?
        .virial
        .ok_or_else(|| ForceError::Contract("virial not returned".into()))?;
    let scale = (0..3)
        .flat_map(|b| (0..3).map(move |a| w.col(b)[a].abs()))
        .fold(1e-3, f64::max);
    let mut worst = 0.0f64;
    for (a, b) in [(0, 0), (1, 1), (2, 2), (1, 0), (2, 0), (2, 1)] {
        let mut energies = [0.0; 2];
        for (slot, sign) in [(0, 1.0), (1, -1.0)] {
            let moved: Vec<Vec3> = positions
                .iter()
                .map(|&r| strain(r, a, b, sign * delta))
                .collect();
            let strained_cell = match cell {
                Some(c) => {
                    let rows = c.rows().map(|v| strain(v, a, b, sign * delta).to_array());
                    Some(
                        Cell::new(rows, c.periodic())
                            .map_err(|e| ForceError::InvalidInput(e.to_string()))?,
                    )
                }
                None => None,
            };
            energies[slot] = energy(provider, &moved, strained_cell.as_ref())?;
        }
        let fd = -(energies[0] - energies[1]) / (2.0 * delta);
        worst = worst.max((fd - w.col(b)[a]).abs() / scale);
    }
    let asym: Mat3 = w - w.transpose();
    let asymmetry = (0..3)
        .flat_map(|b| (0..3).map(move |a| asym.col(b)[a].abs()))
        .fold(0.0, f64::max)
        / scale;
    Ok(VirialReport {
        max_rel_error: worst,
        asymmetry,
    })
}

/// |Σ energy terms − energy| / max(|energy|, 1); 0 if the provider returns no terms.
pub fn energy_terms_check(
    provider: &mut dyn ForceProvider,
    positions: &[Vec3],
    cell: Option<&Cell>,
) -> Result<f64, ForceError> {
    let out = evaluate_all(provider, positions, cell)?;
    let energy = out
        .energy
        .ok_or_else(|| ForceError::Contract("energy not returned".into()))?;
    if out.energy_terms.is_empty() {
        return Ok(0.0);
    }
    let sum: f64 = out.energy_terms.iter().map(|(_, e)| e).sum();
    Ok((sum - energy).abs() / energy.abs().max(1.0))
}

/// |Σ F_i| / Σ |F_i|: zero for providers without external fields (Newton's third law).
pub fn momentum_check(
    provider: &mut dyn ForceProvider,
    positions: &[Vec3],
    cell: Option<&Cell>,
) -> Result<f64, ForceError> {
    let forces = evaluate_all(provider, positions, cell)?.forces;
    let total: Vec3 = forces.iter().copied().sum();
    let magnitude: f64 = forces.iter().map(|f| f.length()).sum();
    Ok(if magnitude > 0.0 {
        total.length() / magnitude
    } else {
        0.0
    })
}

/// True if two evaluations of the same input give bitwise identical results.
pub fn determinism_check(
    provider: &mut dyn ForceProvider,
    positions: &[Vec3],
    cell: Option<&Cell>,
) -> Result<bool, ForceError> {
    let first = evaluate_all(provider, positions, cell)?;
    let second = evaluate_all(provider, positions, cell)?;
    let same_forces = first
        .forces
        .iter()
        .zip(&second.forces)
        .all(|(a, b)| a.to_array().map(f64::to_bits) == b.to_array().map(f64::to_bits));
    let same_energy = first.energy.map(f64::to_bits) == second.energy.map(f64::to_bits);
    Ok(same_forces && same_energy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{Capabilities, outer};
    use ucmd_core::System;

    /// Tether to the origin, E = ½k Σ|r|², with deliberate bugs selectable for E1.2.
    #[derive(Clone, Copy, PartialEq)]
    enum Bug {
        None,
        SignFlip,
        MissingHalf,
        OffByOne,
    }

    struct Tether {
        k: f64,
        bug: Bug,
    }

    impl ForceProvider for Tether {
        fn name(&self) -> &str {
            "tether"
        }
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                periodic: false,
                virial: true,
                energy_terms: false,
                deterministic: true,
            }
        }
        fn initialize(&mut self, _: &System) -> Result<(), ForceError> {
            Ok(())
        }
        fn evaluate(
            &mut self,
            x: &[Vec3],
            _: Option<&Cell>,
            _: &EvalRequest,
            out: &mut ForceResult,
        ) -> Result<(), ForceError> {
            let n = x.len();
            let half = if self.bug == Bug::MissingHalf {
                1.0
            } else {
                0.5
            };
            out.energy = Some(half * self.k * x.iter().map(|r| r.length_squared()).sum::<f64>());
            let sign = if self.bug == Bug::SignFlip { 1.0 } else { -1.0 };
            for i in 0..n {
                let source = if self.bug == Bug::OffByOne {
                    (i + 1) % n
                } else {
                    i
                };
                out.forces[i] = sign * self.k * x[source];
            }
            out.virial = Some(x.iter().zip(&out.forces).map(|(&r, &f)| outer(r, f)).sum());
            Ok(())
        }
    }

    fn positions() -> Vec<Vec3> {
        vec![
            Vec3::new(0.1, -0.2, 0.3),
            Vec3::new(-0.4, 0.05, 0.2),
            Vec3::new(0.3, 0.3, -0.1),
        ]
    }

    #[test]
    fn correct_provider_passes_every_check() {
        let mut p = Tether {
            k: 500.0,
            bug: Bug::None,
        };
        let report = fd_force_check(&mut p, &positions(), None, &FdOptions::default()).unwrap();
        assert!(report.max_rel_error < 1e-5, "{report:?}");
        let virial = fd_virial_check(&mut p, &positions(), None, 1e-6).unwrap();
        assert!(
            virial.max_rel_error < 1e-5 && virial.asymmetry < 1e-12,
            "{virial:?}"
        );
        assert!(determinism_check(&mut p, &positions(), None).unwrap());
    }

    /// E1.2: the test kit itself must catch the classic provider bugs.
    #[test]
    fn detects_sign_flip_missing_factor_two_and_off_by_one() {
        for bug in [Bug::SignFlip, Bug::MissingHalf, Bug::OffByOne] {
            let mut p = Tether { k: 500.0, bug };
            let report = fd_force_check(&mut p, &positions(), None, &FdOptions::default()).unwrap();
            assert!(report.max_rel_error > 0.1, "bug not detected: {report:?}");
        }
    }

    #[test]
    fn momentum_check_flags_external_fields() {
        let mut p = Tether {
            k: 500.0,
            bug: Bug::None,
        };
        assert!(momentum_check(&mut p, &positions(), None).unwrap() > 0.1); // a tether is external
    }
}
