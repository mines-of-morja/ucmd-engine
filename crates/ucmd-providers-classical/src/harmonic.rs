//! Analytic harmonic providers for integrator tests and teaching (PLAN.md E1.3).

use ucmd_core::{Cell, System, Vec3};
use ucmd_force::{Capabilities, EvalRequest, ForceError, ForceProvider, ForceResult, outer};

/// Every atom tethered to its own anchor: E = ½ k Σ |r_i − r⁰_i|² (k in kJ mol⁻¹ nm⁻²).
///
/// An external potential, so not translation invariant; only for non-periodic systems.
#[derive(Debug, Clone)]
pub struct HarmonicTether {
    k_kjmol_nm2: f64,
    anchors: Vec<Vec3>,
}

impl HarmonicTether {
    /// Tethers with force constant `k_kjmol_nm2` to the given anchors (nm), one per atom.
    pub fn new(k_kjmol_nm2: f64, anchors: Vec<Vec3>) -> Self {
        Self {
            k_kjmol_nm2,
            anchors,
        }
    }
}

impl ForceProvider for HarmonicTether {
    fn name(&self) -> &str {
        "harmonic-tether"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            periodic: false,
            virial: true,
            energy_terms: false,
            deterministic: true,
        }
    }

    fn initialize(&mut self, system: &System) -> Result<(), ForceError> {
        if self.anchors.len() != system.n_atoms() {
            return Err(ForceError::InvalidInput(format!(
                "{} anchors for {} atoms",
                self.anchors.len(),
                system.n_atoms()
            )));
        }
        Ok(())
    }

    fn evaluate(
        &mut self,
        positions: &[Vec3],
        cell: Option<&Cell>,
        request: &EvalRequest,
        out: &mut ForceResult,
    ) -> Result<(), ForceError> {
        if cell.is_some() {
            return Err(ForceError::Unsupported(
                "harmonic tethers are not defined for periodic cells".into(),
            ));
        }
        let mut energy = 0.0;
        for ((force, &r), &anchor) in out.forces.iter_mut().zip(positions).zip(&self.anchors) {
            let d = r - anchor;
            energy += 0.5 * self.k_kjmol_nm2 * d.length_squared();
            *force = -self.k_kjmol_nm2 * d;
        }
        out.energy = Some(energy);
        if request.need_virial {
            out.virial = Some(
                positions
                    .iter()
                    .zip(&out.forces)
                    .map(|(&r, &f)| outer(r, f))
                    .sum(),
            );
        }
        Ok(())
    }
}

/// One spring of [`HarmonicPairSpring`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spring {
    /// Global atom indices.
    pub atoms: [usize; 2],
    /// Rest length in nm.
    pub r0_nm: f64,
    /// Force constant in kJ mol⁻¹ nm⁻².
    pub k_kjmol_nm2: f64,
}

/// Springs between atom pairs: E = Σ ½ k (|r_ij| − r₀)², minimum image under periodic boundaries.
#[derive(Debug, Clone)]
pub struct HarmonicPairSpring {
    springs: Vec<Spring>,
}

impl HarmonicPairSpring {
    /// A provider for the given springs.
    pub fn new(springs: Vec<Spring>) -> Self {
        Self { springs }
    }
}

impl ForceProvider for HarmonicPairSpring {
    fn name(&self) -> &str {
        "harmonic-pair-spring"
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
        let n = system.n_atoms();
        match self
            .springs
            .iter()
            .find(|s| s.atoms[0] >= n || s.atoms[1] >= n || s.atoms[0] == s.atoms[1])
        {
            Some(s) => Err(ForceError::InvalidInput(format!(
                "bad spring {:?} for {n} atoms",
                s.atoms
            ))),
            None => Ok(()),
        }
    }

    fn evaluate(
        &mut self,
        positions: &[Vec3],
        cell: Option<&Cell>,
        request: &EvalRequest,
        out: &mut ForceResult,
    ) -> Result<(), ForceError> {
        let mut energy = 0.0;
        let mut virial = ucmd_core::Mat3::ZERO;
        for s in &self.springs {
            let [i, j] = s.atoms;
            let mut d = positions[i] - positions[j];
            if let Some(cell) = cell {
                d = cell.minimum_image(d);
            }
            let r = d.length();
            if r == 0.0 {
                return Err(ForceError::InvalidInput(format!(
                    "atoms {i} and {j} coincide"
                )));
            }
            let stretch = r - s.r0_nm;
            energy += 0.5 * s.k_kjmol_nm2 * stretch * stretch;
            let f_ij = -s.k_kjmol_nm2 * stretch / r * d; // force on i due to j
            out.forces[i] += f_ij;
            out.forces[j] -= f_ij;
            virial += outer(d, f_ij);
        }
        out.energy = Some(energy);
        if request.need_virial {
            out.virial = Some(virial);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucmd_force::testkit::{
        FdOptions, determinism_check, fd_force_check, fd_virial_check, momentum_check,
    };

    fn positions() -> Vec<Vec3> {
        vec![
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.0, 0.2, 0.05),
            Vec3::new(-0.05, 0.0, -0.15),
        ]
    }

    #[test]
    fn tether_passes_the_testkit() {
        let mut p = HarmonicTether::new(
            1000.0,
            vec![Vec3::ZERO, Vec3::new(0.01, 0.0, 0.0), Vec3::ZERO],
        );
        let fd = fd_force_check(&mut p, &positions(), None, &FdOptions::default()).unwrap();
        assert!(fd.max_rel_error < 1e-5, "{fd:?}");
        let w = fd_virial_check(&mut p, &positions(), None, 1e-6).unwrap();
        assert!(w.max_rel_error < 1e-5, "{w:?}");
        assert!(determinism_check(&mut p, &positions(), None).unwrap());
    }

    #[test]
    fn pair_spring_passes_the_testkit_with_and_without_a_cell() {
        let springs = vec![
            Spring {
                atoms: [0, 1],
                r0_nm: 0.15,
                k_kjmol_nm2: 2000.0,
            },
            Spring {
                atoms: [1, 2],
                r0_nm: 0.30,
                k_kjmol_nm2: 500.0,
            },
        ];
        let cell = Cell::new(
            [[0.8, 0.0, 0.0], [0.2, 0.7, 0.0], [0.1, -0.2, 0.9]],
            [true; 3],
        )
        .unwrap();
        // atom 2 shifted by a lattice vector: the spring must act through the minimum image
        let mut shifted = positions();
        shifted[2] += cell.rows()[0];
        for (x, c) in [(positions(), None), (shifted, Some(&cell))] {
            let mut p = HarmonicPairSpring::new(springs.clone());
            let fd = fd_force_check(&mut p, &x, c, &FdOptions::default()).unwrap();
            assert!(fd.max_rel_error < 1e-5, "{fd:?}");
            let w = fd_virial_check(&mut p, &x, c, 1e-6).unwrap();
            assert!(w.max_rel_error < 1e-5 && w.asymmetry < 1e-12, "{w:?}");
            assert!(momentum_check(&mut p, &x, c).unwrap() < 1e-14);
        }
    }
}
