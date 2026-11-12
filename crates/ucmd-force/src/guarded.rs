//! `Guarded`: checks every provider result before the engine uses it (CLAUDE.md invariant 6)
//! and `Composite`: the sum of several providers.

use crate::provider::{Capabilities, EvalRequest, ForceError, ForceProvider, ForceResult};
use ucmd_core::{Cell, Mat3, System, Vec3};

/// Relative tolerance for energy terms summing to the energy.
pub const ENERGY_TERMS_TOLERANCE: f64 = 1e-10;

/// Wraps a provider and turns every contract violation into [`ForceError::Contract`].
pub struct Guarded<P> {
    inner: P,
    n_atoms: usize,
}

impl<P: ForceProvider> Guarded<P> {
    /// Wrap `inner`; the number of atoms is taken from the system at initialization.
    pub fn new(inner: P) -> Self {
        Self { inner, n_atoms: 0 }
    }

    /// The wrapped provider.
    pub fn inner(&self) -> &P {
        &self.inner
    }
}

fn finite3(v: &Vec3) -> bool {
    v.is_finite()
}

fn finite_mat(m: &Mat3) -> bool {
    m.is_finite()
}

impl<P: ForceProvider> ForceProvider for Guarded<P> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }

    fn initialize(&mut self, system: &System) -> Result<(), ForceError> {
        self.n_atoms = system.n_atoms();
        self.inner.initialize(system)?;
        if system.is_periodic() && !self.inner.capabilities().periodic {
            return Err(ForceError::Unsupported(format!(
                "provider {} does not support periodic systems",
                self.inner.name()
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
        if positions.len() != self.n_atoms {
            return Err(ForceError::InvalidInput(format!(
                "{} positions for {} atoms",
                positions.len(),
                self.n_atoms
            )));
        }
        if let Some(bad) = positions.iter().position(|r| !finite3(r)) {
            return Err(ForceError::InvalidInput(format!(
                "non-finite position of atom {bad}"
            )));
        }
        out.reset(self.n_atoms);
        self.inner.evaluate(positions, cell, request, out)?;
        let name = self.inner.name();
        let violation = |what: String| Err(ForceError::Contract(format!("{name}: {what}")));
        if out.forces.len() != self.n_atoms {
            return violation(format!(
                "returned {} forces for {} atoms",
                out.forces.len(),
                self.n_atoms
            ));
        }
        if let Some(bad) = out.forces.iter().position(|f| !finite3(f)) {
            return violation(format!("non-finite force on atom {bad}"));
        }
        match out.energy {
            Some(e) if !e.is_finite() => return violation("non-finite energy".into()),
            None if request.need_energy => {
                return violation("energy requested but not returned".into());
            }
            _ => {}
        }
        match &out.virial {
            Some(w) if !finite_mat(w) => return violation("non-finite virial".into()),
            None if request.need_virial && self.inner.capabilities().virial => {
                return violation("virial requested but not returned".into());
            }
            _ => {}
        }
        if let (Some(energy), false) = (out.energy, out.energy_terms.is_empty()) {
            let sum: f64 = out.energy_terms.iter().map(|(_, e)| e).sum();
            if (sum - energy).abs() > ENERGY_TERMS_TOLERANCE * energy.abs().max(1.0) {
                return violation(format!("energy terms sum to {sum}, energy is {energy}"));
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ForceError> {
        self.inner.finalize()
    }
}

/// A named child of a [`Composite`].
struct Child {
    name: String,
    provider: Box<dyn ForceProvider>,
    buffer: ForceResult,
}

/// The sum of several providers; energy terms are prefixed with `<child name>/`.
pub struct Composite {
    name: String,
    children: Vec<Child>,
}

impl Composite {
    /// An empty composite called `name`.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            children: Vec::new(),
        }
    }

    /// Add a provider under `name` (unique within the composite).
    pub fn push(&mut self, name: impl Into<String>, provider: Box<dyn ForceProvider>) {
        self.children.push(Child {
            name: name.into(),
            provider,
            buffer: ForceResult::default(),
        });
    }

    /// Number of children.
    pub fn len(&self) -> usize {
        self.children.len()
    }

    /// True without children.
    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }
}

impl ForceProvider for Composite {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> Capabilities {
        let caps: Vec<Capabilities> = self
            .children
            .iter()
            .map(|c| c.provider.capabilities())
            .collect();
        Capabilities {
            periodic: caps.iter().all(|c| c.periodic),
            virial: caps.iter().all(|c| c.virial),
            energy_terms: !caps.is_empty(),
            deterministic: caps.iter().all(|c| c.deterministic),
        }
    }

    fn initialize(&mut self, system: &System) -> Result<(), ForceError> {
        for child in &mut self.children {
            child.provider.initialize(system)?;
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
        let n = positions.len();
        out.reset(n);
        let mut energy = request.need_energy.then_some(0.0);
        let mut virial = (request.need_virial && self.capabilities().virial).then_some(Mat3::ZERO);
        for child in &mut self.children {
            child.buffer.reset(n);
            child
                .provider
                .evaluate(positions, cell, request, &mut child.buffer)?;
            for (total, f) in out.forces.iter_mut().zip(&child.buffer.forces) {
                *total += *f;
            }
            if let (Some(total), Some(e)) = (energy.as_mut(), child.buffer.energy) {
                *total += e;
                if child.buffer.energy_terms.is_empty() {
                    out.energy_terms.push((child.name.clone(), e));
                } else {
                    for (term, value) in &child.buffer.energy_terms {
                        out.energy_terms
                            .push((format!("{}/{}", child.name, term), *value));
                    }
                }
            }
            if let (Some(total), Some(w)) = (virial.as_mut(), child.buffer.virial) {
                *total += w;
            }
            for observable in child.buffer.observables.drain(..) {
                let mut observable = observable;
                observable.name = format!("{}/{}", child.name, observable.name);
                out.observables.push(observable);
            }
        }
        out.energy = energy;
        out.virial = virial;
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ForceError> {
        let mut first_error = None;
        for child in &mut self.children {
            if let Err(e) = child.provider.finalize() {
                first_error.get_or_insert(e);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucmd_core::{AtomTemplate, MoleculeType};

    /// A provider whose output can be broken in configurable ways.
    struct Fake {
        energy: Option<f64>,
        n_forces: Option<usize>,
        nan_force: bool,
        terms: Vec<(String, f64)>,
    }

    impl Fake {
        fn good(energy: f64) -> Self {
            Self {
                energy: Some(energy),
                n_forces: None,
                nan_force: false,
                terms: Vec::new(),
            }
        }
    }

    impl ForceProvider for Fake {
        fn name(&self) -> &str {
            "fake"
        }
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                periodic: false,
                virial: false,
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
            out.energy = self.energy;
            out.forces = vec![Vec3::ONE; self.n_forces.unwrap_or(x.len())];
            if self.nan_force {
                out.forces[0].x = f64::NAN;
            }
            out.energy_terms = self.terms.clone();
            Ok(())
        }
    }

    fn system(n: usize) -> System {
        let atom = AtomTemplate {
            name: "X".into(),
            element: 0,
            mass_amu: 1.0,
            charge_e: None,
            atom_type: None,
        };
        let mtype = MoleculeType {
            name: "X".into(),
            formal_charge_e: 0,
            atoms: vec![atom],
            bonds: vec![],
            angles: vec![],
            dihedrals: vec![],
            impropers: vec![],
            constraints: vec![],
        };
        System::new("t", vec![mtype], &[(0, n)], None, [false; 3]).unwrap()
    }

    fn run(provider: Fake) -> Result<ForceResult, ForceError> {
        let mut guarded = Guarded::new(provider);
        guarded.initialize(&system(2))?;
        let mut out = ForceResult::default();
        let request = EvalRequest {
            need_virial: false,
            ..EvalRequest::everything()
        };
        guarded
            .evaluate(&[Vec3::ZERO, Vec3::X], None, &request, &mut out)
            .map(|()| out)
    }

    #[test]
    fn guarded_passes_good_results_and_rejects_bad_ones() {
        assert!(run(Fake::good(1.0)).is_ok());
        let contract = |p: Fake| matches!(run(p), Err(ForceError::Contract(_)));
        assert!(contract(Fake {
            n_forces: Some(1),
            ..Fake::good(1.0)
        }));
        assert!(contract(Fake {
            nan_force: true,
            ..Fake::good(1.0)
        }));
        assert!(contract(Fake {
            energy: None,
            ..Fake::good(1.0)
        }));
        assert!(contract(Fake::good(f64::INFINITY)));
        assert!(contract(Fake {
            terms: vec![("a".into(), 0.4)],
            ..Fake::good(1.0)
        }));
    }

    #[test]
    fn guarded_rejects_bad_input_and_periodic_misuse() {
        let mut guarded = Guarded::new(Fake::good(1.0));
        guarded.initialize(&system(2)).unwrap();
        let mut out = ForceResult::default();
        let request = EvalRequest::everything();
        assert!(matches!(
            guarded.evaluate(&[Vec3::ZERO], None, &request, &mut out),
            Err(ForceError::InvalidInput(_))
        ));
        let nan = [Vec3::ZERO, Vec3::new(f64::NAN, 0.0, 0.0)];
        assert!(matches!(
            guarded.evaluate(&nan, None, &request, &mut out),
            Err(ForceError::InvalidInput(_))
        ));
        let mut periodic = system(2);
        periodic.periodicity = [true; 3];
        assert!(matches!(
            Guarded::new(Fake::good(1.0)).initialize(&periodic),
            Err(ForceError::Unsupported(_))
        ));
    }

    #[test]
    fn composite_sums_children_and_prefixes_terms() {
        let mut composite = Composite::new("ff");
        composite.push("a", Box::new(Fake::good(1.5)));
        composite.push(
            "b",
            Box::new(Fake {
                terms: vec![("x".into(), 2.0), ("y".into(), 0.5)],
                ..Fake::good(2.5)
            }),
        );
        let mut guarded = Guarded::new(composite);
        guarded.initialize(&system(2)).unwrap();
        let mut out = ForceResult::default();
        let request = EvalRequest {
            need_virial: false,
            ..EvalRequest::everything()
        };
        guarded
            .evaluate(&[Vec3::ZERO, Vec3::X], None, &request, &mut out)
            .unwrap();
        assert_eq!(out.energy, Some(4.0));
        assert_eq!(out.forces, vec![Vec3::splat(2.0); 2]);
        let names: Vec<&str> = out.energy_terms.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["a", "b/x", "b/y"]);
    }
}
