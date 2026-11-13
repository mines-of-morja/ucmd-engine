//! Bonded terms of a parameterized package: bonds, angles, dihedrals, impropers (E2.1).
//!
//! Forms and units follow spec/system-package.md §5.3: E = ½k(r − r₀)², ½k(θ − θ₀)²,
//! k[1 + cos(nφ − φ₀)], the OPLS Fourier series, Ryckaert–Bellemans in ψ = φ − π, and the
//! harmonic improper ½k(ξ − ξ₀)² with ξ − ξ₀ wrapped into (−π, π]. Positions are unwrapped
//! and molecules whole (PLAN.md D11), so no minimum image is needed. The dihedral angle uses
//! the IUPAC sign convention of GROMACS and OpenMM; the forces follow GROMACS's `do_dih_fup`.
//!
//! The virial of every term is Σ (r_a − r_ref) ⊗ F_a with a reference atom of the same term:
//! the forces of a term sum to zero, so this equals Σ r_a ⊗ F_a without the cancellation that
//! large unwrapped coordinates would cause.

use std::f64::consts::PI;
use ucmd_core::{AngleForm, BondForm, Cell, Mat3, System, TorsionForm, Vec3};
use ucmd_force::{Capabilities, EvalRequest, ForceError, ForceProvider, ForceResult, outer};

/// Energy terms reported by [`Bonded`], in this order.
pub const TERMS: [&str; 4] = ["bond", "angle", "dihedral", "improper"];

/// All bonded terms of the system, expanded to global atom indices.
#[derive(Debug, Clone, Default)]
pub struct Bonded {
    name: String,
    bonds: Vec<([usize; 2], BondForm)>,
    angles: Vec<([usize; 3], AngleForm)>,
    dihedrals: Vec<([usize; 4], TorsionForm)>,
    impropers: Vec<([usize; 4], TorsionForm)>,
}

impl Bonded {
    /// A bonded provider named `name` (prefix of nothing; terms are `bond`, `angle`, …).
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Number of (bonds, angles, dihedrals, impropers).
    pub fn counts(&self) -> [usize; 4] {
        [
            self.bonds.len(),
            self.angles.len(),
            self.dihedrals.len(),
            self.impropers.len(),
        ]
    }

    /// Energy (kJ/mol) per term, forces added to `forces`, virial added to `virial`.
    pub fn accumulate(
        &self,
        positions: &[Vec3],
        forces: &mut [Vec3],
        virial: &mut Mat3,
    ) -> Result<[f64; 4], ForceError> {
        let mut energy = [0.0; 4];
        for &([i, j], form) in &self.bonds {
            let BondForm::Harmonic { r0_nm, k_kjmol_nm2 } = form;
            let d = positions[j] - positions[i];
            let r = d.length();
            if r == 0.0 {
                return Err(ForceError::InvalidInput(format!(
                    "bonded atoms {i} and {j} coincide"
                )));
            }
            let dr = r - r0_nm;
            energy[0] += 0.5 * k_kjmol_nm2 * dr * dr;
            let fi = (k_kjmol_nm2 * dr / r) * d;
            forces[i] += fi;
            forces[j] -= fi;
            *virial += outer(-d, fi);
        }
        for &([i, j, k], form) in &self.angles {
            let AngleForm::Harmonic {
                theta0_rad,
                k_kjmol_rad2,
            } = form;
            let u = positions[i] - positions[j];
            let v = positions[k] - positions[j];
            let w = u.cross(v);
            let (wl, uu, vv) = (w.length(), u.length_squared(), v.length_squared());
            let theta = wl.atan2(u.dot(v));
            let dtheta = theta - theta0_rad;
            energy[1] += 0.5 * k_kjmol_rad2 * dtheta * dtheta;
            let de = k_kjmol_rad2 * dtheta;
            if wl <= 1e-12 * (uu * vv).sqrt() {
                // Linear angle: the force direction is undefined unless the force vanishes.
                if de.abs() > 1e-12 * k_kjmol_rad2.max(1.0) {
                    return Err(ForceError::InvalidInput(format!(
                        "angle {i}-{j}-{k} is linear but θ0 = {:.3}°",
                        theta0_rad.to_degrees()
                    )));
                }
                continue;
            }
            // ∂θ/∂u = −(w × u)/(|w| u²), ∂θ/∂v = (w × v)/(|w| v²)
            let fi = (de / (wl * uu)) * w.cross(u);
            let fk = (-de / (wl * vv)) * w.cross(v);
            forces[i] += fi;
            forces[k] += fk;
            forces[j] -= fi + fk;
            *virial += outer(u, fi) + outer(v, fk);
        }
        for (slot, list) in [(2, &self.dihedrals), (3, &self.impropers)] {
            for &(atoms, form) in list {
                energy[slot] += torsion(atoms, form, positions, forces, virial)?;
            }
        }
        Ok(energy)
    }
}

/// The dihedral angle φ ∈ (−π, π] of i-j-k-l (IUPAC sign convention), with the vectors used
/// by the force: r_ij = r_i − r_j, r_kj = r_k − r_j, r_kl = r_k − r_l, m = r_ij × r_kj,
/// n = r_kj × r_kl.
pub fn dihedral_angle(ri: Vec3, rj: Vec3, rk: Vec3, rl: Vec3) -> f64 {
    let (rij, rkj, rkl) = (ri - rj, rk - rj, rk - rl);
    let m = rij.cross(rkj);
    let n = rkj.cross(rkl);
    (rkj.length() * rij.dot(n)).atan2(m.dot(n))
}

/// Energy and dE/dφ of a torsion form at angle φ.
fn torsion_energy(form: TorsionForm, phi: f64) -> (f64, f64) {
    match form {
        TorsionForm::Periodic {
            k_kjmol,
            n,
            phi0_rad,
        } => {
            let x = f64::from(n) * phi - phi0_rad;
            (
                k_kjmol * (1.0 + libm::cos(x)),
                -k_kjmol * f64::from(n) * libm::sin(x),
            )
        }
        TorsionForm::OplsFourier {
            v_kjmol: [v1, v2, v3, v4],
        } => {
            let e = 0.5
                * (v1 * (1.0 + libm::cos(phi))
                    + v2 * (1.0 - libm::cos(2.0 * phi))
                    + v3 * (1.0 + libm::cos(3.0 * phi))
                    + v4 * (1.0 - libm::cos(4.0 * phi)));
            let de = 0.5
                * (-v1 * libm::sin(phi) + 2.0 * v2 * libm::sin(2.0 * phi)
                    - 3.0 * v3 * libm::sin(3.0 * phi)
                    + 4.0 * v4 * libm::sin(4.0 * phi));
            (e, de)
        }
        TorsionForm::RyckaertBellemans { c_kjmol: c } => {
            // ψ = φ − π, so cos ψ = −cos φ and d(cos ψ)/dφ = sin φ.
            // E = Σ Cₙ cosⁿψ, dE/d(cos ψ) = Σ (n + 1) Cₙ₊₁ cosⁿψ.
            let cpsi = -libm::cos(phi);
            let mut e = 0.0;
            let mut de_dcos = 0.0;
            let mut power = 1.0; // cosⁿψ
            for n in 0..c.len() {
                e += c[n] * power;
                if n + 1 < c.len() {
                    de_dcos += (n + 1) as f64 * c[n + 1] * power;
                }
                power *= cpsi;
            }
            (e, de_dcos * libm::sin(phi))
        }
        TorsionForm::Harmonic {
            xi0_rad,
            k_kjmol_rad2,
        } => {
            let d = phi - xi0_rad;
            let d = d - 2.0 * PI * libm::floor((d + PI) / (2.0 * PI));
            (0.5 * k_kjmol_rad2 * d * d, k_kjmol_rad2 * d)
        }
    }
}

fn torsion(
    [i, j, k, l]: [usize; 4],
    form: TorsionForm,
    positions: &[Vec3],
    forces: &mut [Vec3],
    virial: &mut Mat3,
) -> Result<f64, ForceError> {
    let (ri, rj, rk, rl) = (positions[i], positions[j], positions[k], positions[l]);
    let (rij, rkj, rkl) = (ri - rj, rk - rj, rk - rl);
    let m = rij.cross(rkj);
    let n = rkj.cross(rkl);
    let (iprm, iprn, nrkj2) = (m.length_squared(), n.length_squared(), rkj.length_squared());
    if iprm <= 1e-24 || iprn <= 1e-24 || nrkj2 <= 1e-24 {
        return Err(ForceError::InvalidInput(format!(
            "torsion {i}-{j}-{k}-{l} has collinear atoms"
        )));
    }
    let nrkj = nrkj2.sqrt();
    let phi = (nrkj * rij.dot(n)).atan2(m.dot(n));
    let (energy, ddphi) = torsion_energy(form, phi);
    // GROMACS do_dih_fup
    let f_i = (-ddphi * nrkj / iprm) * m;
    let f_l = (ddphi * nrkj / iprn) * n;
    let p = rij.dot(rkj) / nrkj2;
    let q = rkl.dot(rkj) / nrkj2;
    let svec = p * f_i - q * f_l;
    let f_j = f_i - svec;
    let f_k = f_l + svec;
    forces[i] += f_i;
    forces[j] -= f_j;
    forces[k] -= f_k;
    forces[l] += f_l;
    // relative to r_j: r_i − r_j = rij, r_k − r_j = rkj, r_l − r_j = rkj − rkl
    *virial += outer(rij, f_i) - outer(rkj, f_k) + outer(rkj - rkl, f_l);
    Ok(energy)
}

fn missing(what: &str, molecule: &str) -> ForceError {
    ForceError::Unsupported(format!(
        "{what} of molecule type {molecule:?} has no parameters; the bonded provider needs a \
         parameterized package"
    ))
}

impl ForceProvider for Bonded {
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
        self.bonds.clear();
        self.angles.clear();
        self.dihedrals.clear();
        self.impropers.clear();
        for instance in &system.instances {
            let mtype = &system.molecule_types[instance.molecule_type];
            let o = instance.first_atom;
            for b in &mtype.bonds {
                let form = b.form.ok_or_else(|| missing("a bond", &mtype.name))?;
                self.bonds.push((b.atoms.map(|a| a + o), form));
            }
            for a in &mtype.angles {
                let form = a.form.ok_or_else(|| missing("an angle", &mtype.name))?;
                self.angles.push((a.atoms.map(|x| x + o), form));
            }
            for t in &mtype.dihedrals {
                let form = t.form.ok_or_else(|| missing("a dihedral", &mtype.name))?;
                self.dihedrals.push((t.atoms.map(|x| x + o), form));
            }
            for t in &mtype.impropers {
                let form = t.form.ok_or_else(|| missing("an improper", &mtype.name))?;
                self.impropers.push((t.atoms.map(|x| x + o), form));
            }
        }
        Ok(())
    }

    fn evaluate(
        &mut self,
        positions: &[Vec3],
        _cell: Option<&Cell>,
        request: &EvalRequest,
        out: &mut ForceResult,
    ) -> Result<(), ForceError> {
        out.reset(positions.len());
        let mut virial = Mat3::ZERO;
        let energy = self.accumulate(positions, &mut out.forces, &mut virial)?;
        if request.need_energy {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{WithBox, all_forms_molecule, positions_for, system_of};
    use ucmd_force::testkit::{
        FdOptions, determinism_check, energy_terms_check, fd_force_check, fd_virial_check,
        momentum_check,
    };

    fn provider(system: &System) -> Bonded {
        let mut p = Bonded::new("bonded");
        p.initialize(system).unwrap();
        p
    }

    #[test]
    fn dihedral_sign_follows_iupac() {
        // j at the origin, k on +z, i on +x; l rotated by +60° about +z: φ = +60°.
        let phi = 60f64.to_radians();
        let got = dihedral_angle(
            Vec3::X,
            Vec3::ZERO,
            Vec3::Z,
            Vec3::new(phi.cos(), phi.sin(), 1.0),
        );
        assert!((got - phi).abs() < 1e-14, "{}", got.to_degrees());
        let trans = dihedral_angle(Vec3::X, Vec3::ZERO, Vec3::Z, Vec3::new(-1.0, 0.0, 1.0));
        assert!((trans.abs() - PI).abs() < 1e-14);
    }

    #[test]
    fn analytic_energies_of_single_terms() {
        let periodic = TorsionForm::Periodic {
            k_kjmol: 3.0,
            n: 2,
            phi0_rad: 0.4,
        };
        let phi = 1.1;
        assert!(
            (torsion_energy(periodic, phi).0 - 3.0 * (1.0 + (2.0 * phi - 0.4).cos())).abs() < 1e-14
        );
        // RB: E = Σ Cn cos^n(φ − π)
        let c = [1.0, -2.0, 0.5, 3.0, -1.5, 0.25];
        let rb = TorsionForm::RyckaertBellemans { c_kjmol: c };
        let x = (phi - PI).cos();
        let expected: f64 = c
            .iter()
            .enumerate()
            .map(|(n, cn)| cn * x.powi(n as i32))
            .sum();
        assert!((torsion_energy(rb, phi).0 - expected).abs() < 1e-13);
        // harmonic improper wraps: ξ = −170°, ξ0 = 170° is a 20° deviation
        let h = TorsionForm::Harmonic {
            xi0_rad: 170f64.to_radians(),
            k_kjmol_rad2: 2.0,
        };
        let e = torsion_energy(h, (-170f64).to_radians()).0;
        assert!((e - 0.5 * 2.0 * 20f64.to_radians().powi(2)).abs() < 1e-13);
    }

    #[test]
    fn every_form_passes_the_test_kit() {
        for periodic in [false, true] {
            let (system, cell) = system_of(vec![all_forms_molecule()], &[(0, 2)], [periodic; 3])
                .with_box(if periodic { Some(3.0) } else { None });
            let positions = positions_for(&system, 7);
            let mut p = provider(&system);
            let options = FdOptions::default();
            let f = fd_force_check(&mut p, &positions, cell.as_ref(), &options).unwrap();
            assert!(
                f.max_rel_error < 1e-6,
                "forces (periodic {periodic}): {f:?}"
            );
            let w = fd_virial_check(&mut p, &positions, cell.as_ref(), 1e-6).unwrap();
            assert!(
                w.max_rel_error < 1e-6,
                "virial (periodic {periodic}): {w:?}"
            );
            assert!(momentum_check(&mut p, &positions, cell.as_ref()).unwrap() < 1e-10);
            assert!(determinism_check(&mut p, &positions, cell.as_ref()).unwrap());
            assert!(energy_terms_check(&mut p, &positions, cell.as_ref()).unwrap() < 1e-12);
        }
    }

    #[test]
    fn counts_and_missing_parameters() {
        let (system, _) =
            system_of(vec![all_forms_molecule()], &[(0, 3)], [false; 3]).with_box(None);
        let p = provider(&system);
        let one = all_forms_molecule();
        assert_eq!(
            p.counts(),
            [
                3 * one.bonds.len(),
                3 * one.angles.len(),
                3 * one.dihedrals.len(),
                3 * one.impropers.len()
            ]
        );
        let mut bare = all_forms_molecule();
        bare.angles[0].form = None;
        let (system, _) = system_of(vec![bare], &[(0, 1)], [false; 3]).with_box(None);
        assert!(matches!(
            Bonded::new("b").initialize(&system),
            Err(ForceError::Unsupported(_))
        ));
    }

    #[test]
    fn collinear_torsion_is_an_error_not_nan() {
        let (system, _) =
            system_of(vec![all_forms_molecule()], &[(0, 1)], [false; 3]).with_box(None);
        let mut positions = positions_for(&system, 3);
        // put the first four chain atoms on a line
        for (a, x) in [(0usize, 0.0), (1, 0.15), (2, 0.30), (3, 0.45)] {
            positions[a] = Vec3::new(x, 0.0, 0.0);
        }
        let mut p = provider(&system);
        let mut out = ForceResult::default();
        let r = p.evaluate(&positions, None, &EvalRequest::everything(), &mut out);
        assert!(matches!(r, Err(ForceError::InvalidInput(_))), "{r:?}");
    }
}
