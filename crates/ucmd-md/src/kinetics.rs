//! Kinetic quantities, degrees of freedom and momentum removal (spec/units.md §4).

use ucmd_core::units::{BAR_PER_KJMOL_NM3, BOLTZMANN_KJMOL_K};
use ucmd_core::{Mat3, Vec3};
use ucmd_force::outer;

/// K = Σ ½ m v² in kJ/mol (m in amu = g/mol, v in nm/ps).
pub fn kinetic_energy(velocities: &[Vec3], masses: &[f64]) -> f64 {
    velocities
        .iter()
        .zip(masses)
        .map(|(v, m)| 0.5 * m * v.length_squared())
        .sum()
}

/// Kinetic tensor Σ ½ m v ⊗ v in kJ/mol.
pub fn kinetic_tensor(velocities: &[Vec3], masses: &[f64]) -> Mat3 {
    velocities
        .iter()
        .zip(masses)
        .map(|(&v, m)| outer(v, v) * (0.5 * m))
        .sum()
}

/// N_dof = 3N − N_constraints − 3 (COM momentum removed) − 3 (angular momentum removed).
pub fn degrees_of_freedom(
    n_atoms: usize,
    n_constraints: usize,
    com_removed: bool,
    angular_removed: bool,
) -> f64 {
    let removed = 3 * usize::from(com_removed) + 3 * usize::from(angular_removed);
    (3 * n_atoms).saturating_sub(n_constraints + removed) as f64
}

/// T = 2K/(N_dof k_B) in K; 0 when there are no degrees of freedom.
pub fn temperature(kinetic_energy: f64, dof: f64) -> f64 {
    if dof > 0.0 {
        2.0 * kinetic_energy / (dof * BOLTZMANN_KJMOL_K)
    } else {
        0.0
    }
}

/// Pressure tensor (2·K_tensor + W)/V in bar, with V in nm³.
pub fn pressure_tensor_bar(kinetic_tensor: Mat3, virial: Mat3, volume_nm3: f64) -> Mat3 {
    (kinetic_tensor * 2.0 + virial) * (BAR_PER_KJMOL_NM3 / volume_nm3)
}

/// Mass-weighted centre of mass (nm).
pub fn center_of_mass(positions: &[Vec3], masses: &[f64]) -> Vec3 {
    let total: f64 = masses.iter().sum();
    positions
        .iter()
        .zip(masses)
        .map(|(&r, &m)| r * m)
        .sum::<Vec3>()
        / total
}

/// Subtract the centre-of-mass velocity from every atom.
pub fn remove_com_velocity(velocities: &mut [Vec3], masses: &[f64]) {
    let v_com = center_of_mass(velocities, masses);
    for v in velocities.iter_mut() {
        *v -= v_com;
    }
}

/// Remove the total angular momentum about the centre of mass (non-periodic systems only).
pub fn remove_angular_momentum(positions: &[Vec3], velocities: &mut [Vec3], masses: &[f64]) {
    let com = center_of_mass(positions, masses);
    let mut angular = Vec3::ZERO;
    let mut inertia = Mat3::ZERO;
    for ((&r, v), &m) in positions.iter().zip(velocities.iter()).zip(masses) {
        let d = r - com;
        angular += m * d.cross(*v);
        inertia += (Mat3::IDENTITY * d.length_squared() - outer(d, d)) * m;
    }
    // pseudo-inverse via determinant guard: linear molecules have a singular inertia tensor
    if inertia.determinant().abs() < 1e-30 {
        return;
    }
    let omega = inertia.inverse() * angular;
    for (&r, v) in positions.iter().zip(velocities.iter_mut()) {
        *v -= omega.cross(r - com);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degrees_of_freedom_count() {
        assert_eq!(degrees_of_freedom(4, 0, true, true), 6.0); // Ag4 in the thesis: 3N-6
        assert_eq!(degrees_of_freedom(864, 0, true, false), 2589.0);
        assert_eq!(degrees_of_freedom(3, 3, false, false), 6.0);
    }

    #[test]
    fn momentum_removal_conserves_kinetic_energy_relative_to_com() {
        let masses = [1.0, 2.0, 3.0];
        let x = [
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.0, 0.3, 0.0),
            Vec3::new(-0.2, 0.0, 0.25),
        ];
        let mut v = [
            Vec3::new(1.0, 0.5, 0.0),
            Vec3::new(0.2, -0.1, 0.3),
            Vec3::new(0.0, 0.4, -0.2),
        ];
        remove_com_velocity(&mut v, &masses);
        assert!(center_of_mass(&v, &masses).length() < 1e-15);
        remove_angular_momentum(&x, &mut v, &masses);
        let com = center_of_mass(&x, &masses);
        let l: Vec3 = x
            .iter()
            .zip(&v)
            .zip(&masses)
            .map(|((&r, &vel), &m)| m * (r - com).cross(vel))
            .sum();
        assert!(l.length() < 1e-14, "{l:?}");
        assert!(center_of_mass(&v, &masses).length() < 1e-14);
    }

    #[test]
    fn temperature_of_equipartition() {
        // K = N_dof k_B T / 2  <=>  T
        let dof = 6.0;
        let k = 0.5 * dof * BOLTZMANN_KJMOL_K * 300.0;
        assert!((temperature(k, dof) - 300.0).abs() < 1e-12);
    }
}
