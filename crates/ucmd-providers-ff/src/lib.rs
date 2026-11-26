//! The `ucmd-ff` provider: the separately developed force-field engine
//! [ucmd-ff](https://github.com/mines-of-morja/ucmd-ff) as an in-process force provider.
//!
//! It is an alternative to the `force-field` provider of `ucmd-providers-classical`. Both read
//! the same package parameters and compute the same terms (ADR 0008: pluggable force sources),
//! and `run.toml` chooses one by `kind`. ucmd-ff is written independently and kept simple:
//! - all pairs for non-periodic systems;
//! - for periodic ones, the minimum image with a plain LJ cutoff and classic Ewald for Coulomb;
//! - no PME, no neighbour lists, no LJ tail correction.
//!
//! It is meant for cross-checks and small systems, not for production speed.
//!
//! Energy terms:
//! - always: `bond`, `angle`, `dihedral`, `improper`, `lj` (1-4 pairs included);
//! - non-periodic: `coulomb`;
//! - periodic: `coulomb_real`, `coulomb_reciprocal`, `coulomb_self`, `coulomb_correction`.
//!
//! Units: nm, kJ/mol, e; forces in kJ mol⁻¹ nm⁻¹; virial in kJ/mol.

use serde::Deserialize;
use ucmd_core::{AngleForm, BondForm, Cell, CombinationRule, Mat3, System, TorsionForm, Vec3};
use ucmd_ff::ewald::EwaldParameters;
use ucmd_ff::forcefield::{ForceField, PeriodicSettings};
use ucmd_ff::system as ff;
use ucmd_force::{Capabilities, EvalRequest, ForceError, ForceProvider, ForceResult};

/// Settings of the `ucmd-ff` provider (the keys of its `[[providers]]` table).
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UcmdFfConfig {
    /// Cutoff for LJ and real-space Coulomb in nm. Required for periodic systems, where it
    /// must be below half the smallest cell height. Not allowed for non-periodic systems, which
    /// use all pairs.
    #[serde(default)]
    pub cutoff_nm: Option<f64>,
    /// Ewald tolerance δ, as in the `force-field` provider: α = √(−ln(2δ))/r_c (OpenMM's rule)
    /// unless `ewald_alpha_per_nm` is given. The reciprocal cutoff is k_c = 2α√(−ln δ) unless
    /// `ewald_recip_cutoff_per_nm` is given.
    #[serde(default = "default_tolerance")]
    pub ewald_tolerance: f64,
    /// Ewald splitting parameter α in nm⁻¹ (overrides the tolerance rule).
    #[serde(default)]
    pub ewald_alpha_per_nm: Option<f64>,
    /// Reciprocal-space cutoff |k| in nm⁻¹ (overrides the tolerance rule).
    #[serde(default)]
    pub ewald_recip_cutoff_per_nm: Option<f64>,
}

fn default_tolerance() -> f64 {
    1e-5
}

impl Default for UcmdFfConfig {
    fn default() -> Self {
        Self {
            cutoff_nm: None,
            ewald_tolerance: default_tolerance(),
            ewald_alpha_per_nm: None,
            ewald_recip_cutoff_per_nm: None,
        }
    }
}

impl UcmdFfConfig {
    /// The periodic run settings for ucmd-ff.
    /// Errors: `InvalidInput` for a missing or non-positive cutoff, δ outside (0, 0.5), or a
    /// non-positive α or k cutoff.
    pub fn periodic_settings(&self) -> Result<PeriodicSettings, ForceError> {
        let cutoff = self
            .cutoff_nm
            .filter(|c| c.is_finite() && *c > 0.0)
            .ok_or_else(|| invalid("cutoff_nm (> 0) is required for a periodic system"))?;
        let tol = self.ewald_tolerance;
        if !(tol.is_finite() && tol > 0.0 && tol < 0.5) {
            return Err(invalid("ewald_tolerance must lie in (0, 0.5)"));
        }
        let alpha = self
            .ewald_alpha_per_nm
            .unwrap_or_else(|| (-(2.0 * tol).ln()).sqrt() / cutoff);
        let k_cut = self
            .ewald_recip_cutoff_per_nm
            .unwrap_or_else(|| 2.0 * alpha * (-tol.ln()).sqrt());
        let ewald = EwaldParameters {
            alpha_per_nm: alpha,
            real_cutoff_nm: cutoff,
            recip_cutoff_per_nm: k_cut,
        };
        ewald
            .validate()
            .map_err(|_| invalid("Ewald α and k cutoff must be finite and positive"))?;
        Ok(PeriodicSettings {
            lj_cutoff_nm: cutoff,
            ewald,
        })
    }
}

/// The ucmd-ff force field as a force provider.
pub struct UcmdFf {
    name: String,
    config: UcmdFfConfig,
    forcefield: Option<ForceField>,
    settings: Option<PeriodicSettings>,
    n_atoms: usize,
}

impl UcmdFf {
    /// A provider named `name` with the settings `config`.
    pub fn new(name: impl Into<String>, config: UcmdFfConfig) -> Self {
        Self {
            name: name.into(),
            config,
            forcefield: None,
            settings: None,
            n_atoms: 0,
        }
    }
}

fn invalid(text: &str) -> ForceError {
    ForceError::InvalidInput(format!("ucmd-ff: {text}"))
}

fn unsupported(text: &str) -> ForceError {
    ForceError::Unsupported(format!("ucmd-ff: {text}"))
}

fn from_ff(e: ucmd_ff::errors::UcmdError) -> ForceError {
    ForceError::InvalidInput(format!("ucmd-ff: {e}"))
}

fn multiplicity(n: u32) -> Result<i32, ForceError> {
    i32::try_from(n).map_err(|_| invalid("torsion multiplicity out of range"))
}

/// Converts the engine's system into ucmd-ff's data model. Consecutive instances of the same
/// molecule type become one component, which keeps the canonical atom order.
fn convert(system: &System) -> Result<ff::System, ForceError> {
    let params = system
        .forcefield
        .as_ref()
        .ok_or_else(|| unsupported("needs a parameterized package (no forcefield section)"))?;
    if !params.pair_overrides.is_empty() {
        // ucmd-ff v0.1.0 knows only the combination rule: running would silently use other pairs
        return Err(unsupported(
            "pair overrides (spec 0.1.2) are not supported; use the provider kind \"force-field\"",
        ));
    }
    let mut molecule_types = Vec::with_capacity(system.molecule_types.len());
    for m in &system.molecule_types {
        let mut atoms = Vec::with_capacity(m.atoms.len());
        for a in &m.atoms {
            let (Some(charge_e), Some(atom_type)) = (a.charge_e, a.atom_type) else {
                return Err(unsupported("every atom needs a charge and an atom type"));
            };
            atoms.push(ff::Atom {
                name: a.name.clone(),
                mass_amu: a.mass_amu,
                charge_e,
                atom_type,
            });
        }
        let mut bonds = Vec::with_capacity(m.bonds.len());
        for b in &m.bonds {
            let Some(BondForm::Harmonic { r0_nm, k_kjmol_nm2 }) = b.form else {
                return Err(unsupported("a bond has no parameters"));
            };
            bonds.push(ff::Bond {
                atoms: b.atoms,
                r0_nm,
                k_kjmol_nm2,
            });
        }
        let mut angles = Vec::with_capacity(m.angles.len());
        for a in &m.angles {
            let Some(AngleForm::Harmonic {
                theta0_rad,
                k_kjmol_rad2,
            }) = a.form
            else {
                return Err(unsupported("an angle has no parameters"));
            };
            angles.push(ff::Angle {
                atoms: a.atoms,
                theta0_rad,
                k_kjmol_rad2,
            });
        }
        let mut dihedrals = Vec::with_capacity(m.dihedrals.len());
        for d in &m.dihedrals {
            let form = match d.form {
                Some(TorsionForm::Periodic {
                    k_kjmol,
                    n,
                    phi0_rad,
                }) => ff::DihedralForm::Periodic {
                    k_kjmol,
                    n: multiplicity(n)?,
                    phi0_rad,
                },
                Some(TorsionForm::OplsFourier { v_kjmol }) => {
                    ff::DihedralForm::OplsFourier { v_kjmol }
                }
                Some(TorsionForm::RyckaertBellemans { c_kjmol }) => {
                    ff::DihedralForm::RyckaertBellemans { c_kjmol }
                }
                Some(TorsionForm::Harmonic { .. }) => {
                    return Err(unsupported("the harmonic form is for impropers only"));
                }
                None => return Err(unsupported("a dihedral has no parameters")),
            };
            dihedrals.push(ff::Dihedral {
                atoms: d.atoms,
                form,
            });
        }
        let mut impropers = Vec::with_capacity(m.impropers.len());
        for d in &m.impropers {
            let form = match d.form {
                Some(TorsionForm::Periodic {
                    k_kjmol,
                    n,
                    phi0_rad,
                }) => ff::ImproperForm::Periodic {
                    k_kjmol,
                    n: multiplicity(n)?,
                    phi0_rad,
                },
                Some(TorsionForm::Harmonic {
                    xi0_rad,
                    k_kjmol_rad2,
                }) => ff::ImproperForm::Harmonic {
                    xi0_rad,
                    k_kjmol_rad2,
                },
                Some(_) => {
                    return Err(unsupported(
                        "impropers take the periodic or the harmonic form only",
                    ));
                }
                None => return Err(unsupported("an improper has no parameters")),
            };
            impropers.push(ff::Improper {
                atoms: d.atoms,
                form,
            });
        }
        molecule_types.push(ff::MoleculeType {
            name: m.name.clone(),
            formal_charge_e: m.formal_charge_e,
            atoms,
            bonds,
            angles,
            dihedrals,
            impropers,
        });
    }
    let mut components: Vec<ff::Component> = Vec::new();
    for inst in &system.instances {
        match components.last_mut() {
            Some(c) if c.molecule_type == inst.molecule_type => c.count += 1,
            _ => components.push(ff::Component {
                molecule_type: inst.molecule_type,
                count: 1,
            }),
        }
    }
    Ok(ff::System {
        molecule_types,
        components,
        forcefield: ff::ForceFieldParameters {
            combination_rule: match params.combination_rule {
                CombinationRule::Geometric => ucmd_ff::combination_rule::CombinationRule::Geometric,
                CombinationRule::LorentzBerthelot => {
                    ucmd_ff::combination_rule::CombinationRule::LorentzBerthelot
                }
            },
            scale_14_lj: params.scale_14_lj,
            scale_14_coulomb: params.scale_14_coulomb,
            atom_types: params
                .atom_types
                .iter()
                .map(|t| ff::AtomType {
                    name: t.name.clone(),
                    sigma_nm: t.sigma_nm,
                    epsilon_kjmol: t.epsilon_kjmol,
                })
                .collect(),
        },
    })
}

/// The engine's cell as ucmd-ff's lower-triangular cell.
fn convert_cell(cell: &Cell) -> Result<ucmd_ff::cell::Cell, ForceError> {
    let [a, b, c] = cell.rows();
    if a.y != 0.0 || a.z != 0.0 || b.z != 0.0 {
        return Err(invalid(
            "the cell is not lower-triangular (spec/units.md §3)",
        ));
    }
    Ok(ucmd_ff::cell::Cell {
        ax: a.x,
        bx: b.x,
        by: b.y,
        cx: c.x,
        cy: c.y,
        cz: c.z,
    })
}

impl ForceProvider for UcmdFf {
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
        let periodic = match system.periodicity {
            [true, true, true] => true,
            [false, false, false] => false,
            _ => return Err(unsupported("partially periodic systems are not supported")),
        };
        self.settings = if periodic {
            Some(self.config.periodic_settings()?)
        } else {
            if self.config.cutoff_nm.is_some() {
                return Err(unsupported(
                    "a non-periodic system uses all pairs; remove cutoff_nm",
                ));
            }
            None
        };
        let forcefield = ForceField::new(&convert(system)?).map_err(from_ff)?;
        if forcefield.topology().atom_count() != system.n_atoms() {
            return Err(ForceError::Contract(
                "ucmd-ff: atom count differs from the system".into(),
            ));
        }
        self.n_atoms = system.n_atoms();
        self.forcefield = Some(forcefield);
        Ok(())
    }

    fn evaluate(
        &mut self,
        positions: &[Vec3],
        cell: Option<&Cell>,
        request: &EvalRequest,
        out: &mut ForceResult,
    ) -> Result<(), ForceError> {
        let forcefield = self
            .forcefield
            .as_ref()
            .ok_or_else(|| ForceError::Contract("ucmd-ff: evaluate before initialize".into()))?;
        if positions.len() != self.n_atoms {
            return Err(invalid("the number of positions differs from the system"));
        }
        let pos: Vec<[f64; 3]> = positions.iter().map(|p| p.to_array()).collect();
        let evaluation = match (&self.settings, cell) {
            (Some(settings), Some(cell)) => {
                let cell = convert_cell(cell)?;
                forcefield.evaluate(&pos, Some((&cell, settings)))
            }
            (None, None) => forcefield.evaluate(&pos, None),
            (Some(_), None) => return Err(invalid("a periodic system needs a cell")),
            (None, Some(_)) => {
                return Err(unsupported("a cell was given for a non-periodic system"));
            }
        }
        .map_err(from_ff)?;
        out.reset(positions.len());
        for (f, g) in out.forces.iter_mut().zip(&evaluation.forces) {
            *f = Vec3::from_array(*g);
        }
        if request.need_energy {
            out.energy = Some(evaluation.energy_kjmol);
            out.energy_terms = evaluation
                .energy_terms
                .iter()
                .map(|t| (t.name.to_string(), t.energy_kjmol))
                .collect();
        }
        if request.need_virial {
            let w = evaluation.virial;
            // Element (a, b) is w.col(b)[a] (ForceResult::virial).
            out.virial = Some(Mat3::from_cols(
                Vec3::new(w[0][0], w[1][0], w[2][0]),
                Vec3::new(w[0][1], w[1][1], w[2][1]),
                Vec3::new(w[0][2], w[1][2], w[2][2]),
            ));
        }
        Ok(())
    }
}
