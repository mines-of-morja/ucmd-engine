//! The `force-field` provider: every term of the package's parameters (design/E2.md §6).
//!
//! Bonded and nonbonded parts are evaluated together and reported as one provider with the
//! terms `bond`, `angle`, `dihedral`, `improper`, `lj`, `lj14` and `coulomb`. The same parts are
//! available as the separate kinds `bonded` and `nonbonded`, for combining with other force
//! sources (ADR 0008).

use crate::bonded::Bonded;
use crate::nonbonded::{Nonbonded, NonbondedConfig};
use ucmd_core::{Cell, System, Vec3};
use ucmd_force::{Capabilities, EvalRequest, ForceError, ForceProvider, ForceResult};

/// Bonded and nonbonded terms of a parameterized package.
pub struct ForceField {
    name: String,
    bonded: Bonded,
    nonbonded: Nonbonded,
    part: ForceResult,
}

impl ForceField {
    /// A provider named `name` with the nonbonded settings `config`.
    pub fn new(name: impl Into<String>, config: NonbondedConfig) -> Self {
        let name = name.into();
        Self {
            bonded: Bonded::new("bonded"),
            nonbonded: Nonbonded::new("nonbonded", config),
            name,
            part: ForceResult::default(),
        }
    }

    /// The nonbonded part (method, α, PME grid).
    pub fn nonbonded(&self) -> &Nonbonded {
        &self.nonbonded
    }
}

impl ForceProvider for ForceField {
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
        self.bonded.initialize(system)?;
        self.nonbonded.initialize(system)
    }

    fn evaluate(
        &mut self,
        positions: &[Vec3],
        cell: Option<&Cell>,
        request: &EvalRequest,
        out: &mut ForceResult,
    ) -> Result<(), ForceError> {
        self.bonded.evaluate(positions, cell, request, out)?;
        self.nonbonded
            .evaluate(positions, cell, request, &mut self.part)?;
        for (f, g) in out.forces.iter_mut().zip(&self.part.forces) {
            *f += *g;
        }
        if let (Some(e), Some(g)) = (out.energy.as_mut(), self.part.energy) {
            *e += g;
        }
        if let (Some(w), Some(g)) = (out.virial.as_mut(), self.part.virial) {
            *w += g;
        }
        out.energy_terms.append(&mut self.part.energy_terms);
        out.observables.append(&mut self.part.observables);
        Ok(())
    }
}
