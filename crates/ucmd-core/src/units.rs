//! Physical constants and unit conversions (spec/units.md §2).
//!
//! This is the only place in the engine where physical constants are written down. Engine
//! units are nm, ps, amu (= g/mol), kJ/mol, e and K; with the mass unit g/mol,
//! 1 amu·nm²·ps⁻² = 1 kJ/mol exactly, so F = m·a needs no conversion factor.

use std::f64::consts::PI;

/// Elementary charge e in C (exact, SI 2019).
pub const ELEMENTARY_CHARGE_C: f64 = 1.602176634e-19;
/// Avogadro constant N_A in mol⁻¹ (exact).
pub const AVOGADRO_PER_MOL: f64 = 6.02214076e23;
/// Boltzmann constant k_B in J/K (exact).
pub const BOLTZMANN_J_PER_K: f64 = 1.380649e-23;
/// Vacuum permittivity ε₀ in F/m (CODATA 2018).
pub const VACUUM_PERMITTIVITY_F_PER_M: f64 = 8.8541878128e-12;
/// Hartree energy E_h in J (CODATA 2018).
pub const HARTREE_J: f64 = 4.3597447222071e-18;
/// Bohr radius a₀ in m (CODATA 2018).
pub const BOHR_M: f64 = 5.29177210903e-11;

/// k_B·N_A in kJ mol⁻¹ K⁻¹.
pub const BOLTZMANN_KJMOL_K: f64 = BOLTZMANN_J_PER_K * AVOGADRO_PER_MOL / 1000.0;
/// Coulomb constant e²N_A/(4πε₀) in kJ mol⁻¹ nm e⁻².
pub const COULOMB_KJMOL_NM_E2: f64 = ELEMENTARY_CHARGE_C * ELEMENTARY_CHARGE_C * AVOGADRO_PER_MOL
    / (4.0 * PI * VACUUM_PERMITTIVITY_F_PER_M)
    * 1.0e6;
/// Pressure: 1 kJ mol⁻¹ nm⁻³ expressed in bar.
pub const BAR_PER_KJMOL_NM3: f64 = 1.0e25 / AVOGADRO_PER_MOL;
/// Energy: 1 hartree in kJ/mol.
pub const KJMOL_PER_HARTREE: f64 = HARTREE_J * AVOGADRO_PER_MOL / 1000.0;
/// Length: 1 bohr in nm.
pub const NM_PER_BOHR: f64 = BOHR_M * 1.0e9;
/// Time: 1 fs in ps.
pub const PS_PER_FS: f64 = 1.0e-3;
/// Angle: 1 degree in rad.
pub const RAD_PER_DEG: f64 = PI / 180.0;

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    /// The derived values listed in spec/units.md §2.
    #[test]
    fn derived_constants_match_the_spec_table() {
        assert!(rel(BOLTZMANN_KJMOL_K, 0.00831446261815324) < 1e-15);
        assert!(rel(COULOMB_KJMOL_NM_E2, 138.93545764438196) < 1e-15);
        assert!(rel(BAR_PER_KJMOL_NM3, 16.605390671738466) < 1e-15);
        assert!(rel(KJMOL_PER_HARTREE, 2625.4996394798254) < 1e-15);
        assert!(rel(NM_PER_BOHR, 0.0529177210903) < 1e-15);
    }

    /// 1 amu·nm²·ps⁻² = 1 kJ/mol exactly when amu means g/mol: 1e-3 kg/mol · 1e-18 m² / 1e-24 s².
    #[test]
    fn mass_length_time_energy_are_consistent() {
        let joule_per_mol = 1.0e-3 * 1.0e-18 / 1.0e-24;
        assert!(rel(joule_per_mol / 1000.0, 1.0) < 1e-15);
    }
}
