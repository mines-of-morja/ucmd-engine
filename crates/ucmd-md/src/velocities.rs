//! Initial velocities.

use crate::kinetics::{kinetic_energy, remove_com_velocity, temperature};
use crate::random::standard_normal;
use rand_chacha::ChaCha8Rng;
use ucmd_core::Vec3;
use ucmd_core::units::BOLTZMANN_KJMOL_K;

/// Maxwell–Boltzmann velocities at `temperature_k` (nm/ps).
///
/// Each component is drawn from N(0, k_B T/m); then the centre-of-mass velocity is removed if
/// `remove_com` and all velocities are rescaled so the kinetic temperature with `dof` degrees of
/// freedom is exactly `temperature_k` (deterministic for a given generator state).
pub fn maxwell_boltzmann(
    masses: &[f64],
    temperature_k: f64,
    dof: f64,
    remove_com: bool,
    rng: &mut ChaCha8Rng,
) -> Vec<Vec3> {
    let mut v: Vec<Vec3> = masses
        .iter()
        .map(|&m| {
            let sigma = (BOLTZMANN_KJMOL_K * temperature_k / m).sqrt();
            let mut g = || standard_normal(rng) * sigma;
            Vec3::new(g(), g(), g())
        })
        .collect();
    if remove_com {
        remove_com_velocity(&mut v, masses);
    }
    let current = temperature(kinetic_energy(&v, masses), dof);
    if current > 0.0 && temperature_k > 0.0 {
        let scale = (temperature_k / current).sqrt();
        v.iter_mut().for_each(|x| *x *= scale);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinetics::{center_of_mass, degrees_of_freedom};
    use crate::random::{STREAM_VELOCITIES, stream};

    #[test]
    fn exact_temperature_zero_momentum_and_reproducible() {
        let masses: Vec<f64> = (0..500).map(|i| 1.0 + f64::from(i % 7)).collect();
        let dof = degrees_of_freedom(500, 0, true, false);
        let v = maxwell_boltzmann(
            &masses,
            300.0,
            dof,
            true,
            &mut stream(42, STREAM_VELOCITIES),
        );
        assert!((temperature(kinetic_energy(&v, &masses), dof) - 300.0).abs() < 1e-10);
        assert!(center_of_mass(&v, &masses).length() < 1e-12);
        let again = maxwell_boltzmann(
            &masses,
            300.0,
            dof,
            true,
            &mut stream(42, STREAM_VELOCITIES),
        );
        assert_eq!(v, again);
        let other = maxwell_boltzmann(
            &masses,
            300.0,
            dof,
            true,
            &mut stream(43, STREAM_VELOCITIES),
        );
        assert_ne!(v, other);
    }
}
