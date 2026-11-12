//! The immutable system: molecule types, their instances, per-atom properties and optional
//! force-field parameters (spec/system-package.md). Built once by `ucmd-io`; never changed
//! during a run (CLAUDE.md invariant 5).

/// Lennard-Jones combination rule (spec/system-package.md §5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombinationRule {
    /// σ_ij = √(σ_i σ_j), ε_ij = √(ε_i ε_j) (OPLS).
    Geometric,
    /// σ_ij = (σ_i + σ_j)/2, ε_ij = √(ε_i ε_j).
    LorentzBerthelot,
}

impl CombinationRule {
    /// Pair parameters (σ_ij in nm, ε_ij in kJ/mol) from the parameters of two atom types.
    pub fn combine(self, a: &AtomType, b: &AtomType) -> (f64, f64) {
        let epsilon = (a.epsilon_kjmol * b.epsilon_kjmol).sqrt();
        let sigma = match self {
            Self::Geometric => (a.sigma_nm * b.sigma_nm).sqrt(),
            Self::LorentzBerthelot => 0.5 * (a.sigma_nm + b.sigma_nm),
        };
        (sigma, epsilon)
    }
}

/// Lennard-Jones parameters of one atom type.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomType {
    /// Type name from the package.
    pub name: String,
    /// σ in nm.
    pub sigma_nm: f64,
    /// ε in kJ/mol.
    pub epsilon_kjmol: f64,
}

/// Global force-field settings of a parameterized package.
#[derive(Debug, Clone, PartialEq)]
pub struct ForceField {
    /// Name, e.g. "OPLS-AA (source …)".
    pub name: String,
    /// Mixing rule for Lennard-Jones pairs.
    pub combination_rule: CombinationRule,
    /// Factor for 1-4 Lennard-Jones interactions (dimensionless).
    pub scale_14_lj: f64,
    /// Factor for 1-4 Coulomb interactions (dimensionless).
    pub scale_14_coulomb: f64,
    /// Atom types, referenced by index from [`AtomTemplate::atom_type`].
    pub atom_types: Vec<AtomType>,
}

/// One atom of a molecule type.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomTemplate {
    /// Atom name.
    pub name: String,
    /// Atomic number; 0 for pseudo-particles without an element.
    pub element: u8,
    /// Mass in amu (g/mol).
    pub mass_amu: f64,
    /// Partial charge in e, if the package has charges.
    pub charge_e: Option<f64>,
    /// Index into [`ForceField::atom_types`], if parameterized.
    pub atom_type: Option<usize>,
}

/// Bond energy term (spec/system-package.md §5.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BondForm {
    /// E = ½ k (r − r₀)², r₀ in nm, k in kJ mol⁻¹ nm⁻².
    Harmonic {
        /// Equilibrium length in nm.
        r0_nm: f64,
        /// Force constant in kJ mol⁻¹ nm⁻².
        k_kjmol_nm2: f64,
    },
}

/// Angle energy term.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AngleForm {
    /// E = ½ k (θ − θ₀)².
    Harmonic {
        /// Equilibrium angle in rad.
        theta0_rad: f64,
        /// Force constant in kJ mol⁻¹ rad⁻².
        k_kjmol_rad2: f64,
    },
}

/// Dihedral or improper energy term (angles in rad, energies in kJ/mol).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TorsionForm {
    /// E = k [1 + cos(nφ − φ₀)].
    Periodic {
        /// Amplitude in kJ/mol.
        k_kjmol: f64,
        /// Multiplicity.
        n: u32,
        /// Phase in rad.
        phi0_rad: f64,
    },
    /// E = ½[V₁(1+cos φ) + V₂(1−cos 2φ) + V₃(1+cos 3φ) + V₄(1−cos 4φ)], V in kJ/mol.
    OplsFourier {
        /// V₁ … V₄ in kJ/mol.
        v_kjmol: [f64; 4],
    },
    /// E = Σₙ Cₙ cosⁿψ with ψ = φ − π, C in kJ/mol.
    RyckaertBellemans {
        /// C₀ … C₅ in kJ/mol.
        c_kjmol: [f64; 6],
    },
    /// E = ½ k (ξ − ξ₀)², improper only.
    Harmonic {
        /// Equilibrium angle in rad.
        xi0_rad: f64,
        /// Force constant in kJ mol⁻¹ rad⁻².
        k_kjmol_rad2: f64,
    },
}

/// Two bonded atoms (indices within the molecule type).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bond {
    /// Atom indices within the molecule type.
    pub atoms: [usize; 2],
    /// Energy term; `None` in unparameterized packages.
    pub form: Option<BondForm>,
}

/// An angle i-j-k (indices within the molecule type).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Angle {
    /// Atom indices; j is the vertex.
    pub atoms: [usize; 3],
    /// Energy term; `None` in unparameterized packages.
    pub form: Option<AngleForm>,
}

/// A dihedral or improper i-j-k-l (indices within the molecule type).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Torsion {
    /// Atom indices.
    pub atoms: [usize; 4],
    /// Energy term; `None` in unparameterized packages.
    pub form: Option<TorsionForm>,
}

/// A fixed distance that belongs to the model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Constraint {
    /// Atom indices within the molecule type.
    pub atoms: [usize; 2],
    /// Distance in nm.
    pub length_nm: f64,
}

/// A molecule type (template); instantiated `count` times by the components.
#[derive(Debug, Clone, PartialEq)]
pub struct MoleculeType {
    /// Name from the package.
    pub name: String,
    /// Ionic charge number z (e).
    pub formal_charge_e: i32,
    /// Atoms in template order.
    pub atoms: Vec<AtomTemplate>,
    /// Bonds (connectivity, optionally with parameters).
    pub bonds: Vec<Bond>,
    /// Angles.
    pub angles: Vec<Angle>,
    /// Proper dihedrals.
    pub dihedrals: Vec<Torsion>,
    /// Impropers.
    pub impropers: Vec<Torsion>,
    /// Model constraints.
    pub constraints: Vec<Constraint>,
}

/// One molecule of the system: its type and the global index of its first atom.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Instance {
    /// Index into [`System::molecule_types`].
    pub molecule_type: usize,
    /// Global index of the first atom (canonical order).
    pub first_atom: usize,
}

/// Inconsistencies found while assembling a [`System`].
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SystemError {
    /// A component names a molecule type that does not exist.
    #[error(
        "component {component} refers to molecule type {molecule_type}, but there are {available}"
    )]
    UnknownMoleculeType {
        /// Component index.
        component: usize,
        /// Requested type index.
        molecule_type: usize,
        /// Number of molecule types.
        available: usize,
    },
    /// An interaction uses an atom index outside its molecule type, or repeats one.
    #[error("molecule type {molecule_type:?}: {what}")]
    BadIndex {
        /// Molecule type name.
        molecule_type: String,
        /// Description.
        what: String,
    },
    /// Some atoms have charges and others do not.
    #[error("charges must be given for all atoms or for none")]
    PartialCharges,
    /// An atom refers to an atom type that does not exist.
    #[error("atom {atom} of {molecule_type:?} refers to a missing atom type")]
    UnknownAtomType {
        /// Molecule type name.
        molecule_type: String,
        /// Atom index within the type.
        atom: usize,
    },
    /// The system has no atoms.
    #[error("the system has no atoms")]
    Empty,
}

/// The immutable system of a run.
#[derive(Debug, Clone, PartialEq)]
pub struct System {
    /// Package name.
    pub name: String,
    /// Molecule types.
    pub molecule_types: Vec<MoleculeType>,
    /// All molecules in canonical order.
    pub instances: Vec<Instance>,
    /// Force-field settings, if parameterized.
    pub forcefield: Option<ForceField>,
    /// Periodic flags for the a, b, c directions.
    pub periodicity: [bool; 3],
    /// SHA-256 (hex) of the package's system.json, for provenance; empty if unknown.
    pub system_sha256: String,
    masses_amu: Vec<f64>,
    charges_e: Option<Vec<f64>>,
    elements: Vec<u8>,
    atom_types: Option<Vec<usize>>,
    molecule_index: Vec<u32>,
}

impl System {
    /// Assemble a system from molecule types and components `(molecule type index, count)`,
    /// expanding all per-atom properties in canonical order.
    pub fn new(
        name: impl Into<String>,
        molecule_types: Vec<MoleculeType>,
        components: &[(usize, usize)],
        forcefield: Option<ForceField>,
        periodicity: [bool; 3],
    ) -> Result<Self, SystemError> {
        for mtype in &molecule_types {
            check_indices(mtype)?;
            if let Some(ff) = &forcefield {
                for (k, atom) in mtype.atoms.iter().enumerate() {
                    if atom.atom_type.is_some_and(|t| t >= ff.atom_types.len()) {
                        return Err(SystemError::UnknownAtomType {
                            molecule_type: mtype.name.clone(),
                            atom: k,
                        });
                    }
                }
            }
        }
        let all_atoms = molecule_types.iter().flat_map(|m| &m.atoms);
        let with_charge = all_atoms.clone().filter(|a| a.charge_e.is_some()).count();
        let total_templates = all_atoms.clone().count();
        if with_charge != 0 && with_charge != total_templates {
            return Err(SystemError::PartialCharges);
        }
        let with_type = all_atoms.filter(|a| a.atom_type.is_some()).count();

        let mut instances = Vec::new();
        let (mut masses, mut charges, mut elements, mut types, mut molecule_index) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (component, &(type_index, count)) in components.iter().enumerate() {
            let mtype = molecule_types
                .get(type_index)
                .ok_or(SystemError::UnknownMoleculeType {
                    component,
                    molecule_type: type_index,
                    available: molecule_types.len(),
                })?;
            for _ in 0..count {
                let molecule = u32::try_from(instances.len()).unwrap_or(u32::MAX);
                instances.push(Instance {
                    molecule_type: type_index,
                    first_atom: masses.len(),
                });
                for atom in &mtype.atoms {
                    masses.push(atom.mass_amu);
                    charges.push(atom.charge_e.unwrap_or(0.0));
                    elements.push(atom.element);
                    types.push(atom.atom_type.unwrap_or(usize::MAX));
                    molecule_index.push(molecule);
                }
            }
        }
        if masses.is_empty() {
            return Err(SystemError::Empty);
        }
        Ok(Self {
            name: name.into(),
            molecule_types,
            instances,
            forcefield,
            periodicity,
            system_sha256: String::new(),
            masses_amu: masses,
            charges_e: (with_charge > 0).then_some(charges),
            elements,
            atom_types: (with_type == total_templates && total_templates > 0).then_some(types),
            molecule_index,
        })
    }

    /// Number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.masses_amu.len()
    }

    /// Per-atom masses in amu (g/mol).
    pub fn masses(&self) -> &[f64] {
        &self.masses_amu
    }

    /// Per-atom partial charges in e, if the package has charges.
    pub fn charges(&self) -> Option<&[f64]> {
        self.charges_e.as_deref()
    }

    /// Per-atom atomic numbers (0 = none).
    pub fn elements(&self) -> &[u8] {
        &self.elements
    }

    /// Per-atom indices into the force field's atom types, if every atom has a type.
    pub fn atom_types(&self) -> Option<&[usize]> {
        self.atom_types.as_deref()
    }

    /// Per-atom global molecule index.
    pub fn molecule_index(&self) -> &[u32] {
        &self.molecule_index
    }

    /// True if any direction is periodic.
    pub fn is_periodic(&self) -> bool {
        self.periodicity.iter().any(|&p| p)
    }

    /// Total number of model constraints over all molecules.
    pub fn n_constraints(&self) -> usize {
        self.instances
            .iter()
            .map(|i| self.molecule_types[i.molecule_type].constraints.len())
            .sum()
    }

    /// True if any molecule type has bonds.
    pub fn has_bonds(&self) -> bool {
        self.instances
            .iter()
            .any(|i| !self.molecule_types[i.molecule_type].bonds.is_empty())
    }
}

fn check_indices(mtype: &MoleculeType) -> Result<(), SystemError> {
    let n = mtype.atoms.len();
    let bad = |what: String| SystemError::BadIndex {
        molecule_type: mtype.name.clone(),
        what,
    };
    let check = |kind: &str, k: usize, atoms: &[usize]| -> Result<(), SystemError> {
        if let Some(&max) = atoms.iter().max()
            && max >= n
        {
            return Err(bad(format!(
                "{kind}[{k}] uses atom {max}, but the type has {n} atoms"
            )));
        }
        for (i, a) in atoms.iter().enumerate() {
            if atoms[..i].contains(a) {
                return Err(bad(format!("{kind}[{k}] repeats atom {a}")));
            }
        }
        Ok(())
    };
    for (k, b) in mtype.bonds.iter().enumerate() {
        check("bonds", k, &b.atoms)?;
    }
    for (k, a) in mtype.angles.iter().enumerate() {
        check("angles", k, &a.atoms)?;
    }
    for (k, t) in mtype.dihedrals.iter().enumerate() {
        check("dihedrals", k, &t.atoms)?;
    }
    for (k, t) in mtype.impropers.iter().enumerate() {
        check("impropers", k, &t.atoms)?;
    }
    for (k, c) in mtype.constraints.iter().enumerate() {
        check("constraints", k, &c.atoms)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atom(name: &str, mass: f64) -> AtomTemplate {
        AtomTemplate {
            name: name.into(),
            element: 0,
            mass_amu: mass,
            charge_e: None,
            atom_type: None,
        }
    }

    fn diatomic() -> MoleculeType {
        MoleculeType {
            name: "AB".into(),
            formal_charge_e: 0,
            atoms: vec![atom("A", 1.0), atom("B", 2.0)],
            bonds: vec![Bond {
                atoms: [0, 1],
                form: None,
            }],
            angles: vec![],
            dihedrals: vec![],
            impropers: vec![],
            constraints: vec![],
        }
    }

    #[test]
    fn expands_components_in_canonical_order() {
        let single = MoleculeType {
            name: "C".into(),
            atoms: vec![atom("C", 3.0)],
            bonds: vec![],
            ..diatomic()
        };
        let system = System::new(
            "t",
            vec![diatomic(), single],
            &[(0, 2), (1, 1)],
            None,
            [false; 3],
        )
        .unwrap();
        assert_eq!(system.n_atoms(), 5);
        assert_eq!(system.masses(), &[1.0, 2.0, 1.0, 2.0, 3.0]);
        assert_eq!(system.molecule_index(), &[0, 0, 1, 1, 2]);
        assert_eq!(
            system.instances[2],
            Instance {
                molecule_type: 1,
                first_atom: 4
            }
        );
        assert!(system.charges().is_none() && system.atom_types().is_none());
        assert!(system.has_bonds());
    }

    #[test]
    fn rejects_inconsistent_input() {
        let mut bad = diatomic();
        bad.bonds[0].atoms = [0, 2];
        assert!(matches!(
            System::new("t", vec![bad], &[(0, 1)], None, [false; 3]),
            Err(SystemError::BadIndex { .. })
        ));
        let mut partial = diatomic();
        partial.atoms[0].charge_e = Some(1.0);
        assert_eq!(
            System::new("t", vec![partial], &[(0, 1)], None, [false; 3]),
            Err(SystemError::PartialCharges)
        );
        assert!(matches!(
            System::new("t", vec![diatomic()], &[(3, 1)], None, [false; 3]),
            Err(SystemError::UnknownMoleculeType { .. })
        ));
    }

    #[test]
    fn combination_rules() {
        let a = AtomType {
            name: "a".into(),
            sigma_nm: 0.3,
            epsilon_kjmol: 0.5,
        };
        let b = AtomType {
            name: "b".into(),
            sigma_nm: 0.4,
            epsilon_kjmol: 2.0,
        };
        let (s, e) = CombinationRule::Geometric.combine(&a, &b);
        assert!((s - 0.12f64.sqrt()).abs() < 1e-15 && (e - 1.0).abs() < 1e-15);
        let (s, _) = CombinationRule::LorentzBerthelot.combine(&a, &b);
        assert!((s - 0.35).abs() < 1e-15);
    }
}
