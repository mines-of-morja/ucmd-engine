//! The force-provider contract (spec/force-provider.md; PLAN.md §2.4).

use ucmd_core::{Cell, Mat3, System, Vec3};

/// What a provider can do; declared once after `initialize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capabilities {
    /// Accepts a periodic cell.
    pub periodic: bool,
    /// Can return the virial W = −∂E/∂ε (needed for pressure and barostats).
    pub virial: bool,
    /// Returns a named decomposition of the energy.
    pub energy_terms: bool,
    /// The same input always gives a bitwise identical output.
    pub deterministic: bool,
}

/// What the engine needs from one evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvalRequest {
    /// MD step of the configuration.
    pub step: u64,
    /// Simulated time in ps.
    pub time_ps: f64,
    /// The energy is needed (forces are always needed).
    pub need_energy: bool,
    /// The virial is needed.
    pub need_virial: bool,
}

impl EvalRequest {
    /// A request for energy, forces and virial at step 0, as used by single points and tests.
    pub fn everything() -> Self {
        Self {
            step: 0,
            time_ps: 0.0,
            need_energy: true,
            need_virial: true,
        }
    }
}

/// Value of an extra provider result.
#[derive(Debug, Clone, PartialEq)]
pub enum ObservableValue {
    /// A single number.
    Scalar(f64),
    /// A flat array of numbers (e.g. a dipole vector).
    Array(Vec<f64>),
}

/// An extra result of a provider, e.g. a dipole moment or another electronic state's energy.
#[derive(Debug, Clone, PartialEq)]
pub struct Observable {
    /// Name, unique per provider.
    pub name: String,
    /// Unit string in the H5MD syntax, e.g. "e nm" or "kJ mol-1".
    pub unit: String,
    /// The value.
    pub value: ObservableValue,
}

/// Result of one evaluation. Owned by the caller and reused every step; providers overwrite it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ForceResult {
    /// Potential energy in kJ/mol; `None` when it was not requested and not computed.
    pub energy: Option<f64>,
    /// Forces in kJ mol⁻¹ nm⁻¹, canonical atom order, F = −∂E/∂r.
    pub forces: Vec<Vec3>,
    /// Virial W = −∂E/∂ε in kJ/mol (spec/units.md §4); element (a, b) is `w.col(b)[a]`.
    pub virial: Option<Mat3>,
    /// Named parts of the energy (kJ/mol) that sum to `energy`.
    pub energy_terms: Vec<(String, f64)>,
    /// Extra results.
    pub observables: Vec<Observable>,
}

impl ForceResult {
    /// Clear everything and size the force array for `n_atoms` zero vectors.
    pub fn reset(&mut self, n_atoms: usize) {
        self.energy = None;
        self.forces.clear();
        self.forces.resize(n_atoms, Vec3::ZERO);
        self.virial = None;
        self.energy_terms.clear();
        self.observables.clear();
    }
}

/// Why an evaluation failed (spec/force-provider.md §4). A failed evaluation never yields
/// partial or substitute results.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ForceError {
    /// The request cannot be served, e.g. a periodic cell for a gas-phase provider.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// The input is unusable, e.g. NaN coordinates or overlapping atoms.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// An iterative solver (SCF, …) did not converge.
    #[error("not converged: {0}")]
    NotConverged(String),
    /// An external program failed.
    #[error("{program} failed (status {status:?}): {stderr_tail}")]
    External {
        /// Program name or path.
        program: String,
        /// Exit status, if any.
        status: Option<i32>,
        /// Last lines of the program's error output.
        stderr_tail: String,
    },
    /// Socket or message-level failure of an out-of-process provider.
    #[error("protocol error: {0}")]
    Protocol(String),
    /// Expected output missing, malformed or stale.
    #[error("cannot parse {file}: {detail}")]
    Parse {
        /// File or stream that was parsed.
        file: String,
        /// What went wrong.
        detail: String,
    },
    /// The provider returned a result that violates the contract (raised by `Guarded`).
    #[error("contract violation by provider: {0}")]
    Contract(String),
}

/// A source of energies and forces. The integrators know providers only through this trait.
pub trait ForceProvider: Send {
    /// Short name, used as the prefix of energy terms.
    fn name(&self) -> &str;

    /// Capabilities; valid after [`ForceProvider::initialize`].
    fn capabilities(&self) -> Capabilities;

    /// Called once with the immutable system before the first evaluation.
    fn initialize(&mut self, system: &System) -> Result<(), ForceError>;

    /// Evaluate the configuration and overwrite `out`.
    ///
    /// `positions` are unwrapped, in nm, canonical order; `cell` is `None` for non-periodic
    /// systems. Fill `out.energy` when `request.need_energy`, `out.virial` when
    /// `request.need_virial` and the virial capability is set.
    fn evaluate(
        &mut self,
        positions: &[Vec3],
        cell: Option<&Cell>,
        request: &EvalRequest,
        out: &mut ForceResult,
    ) -> Result<(), ForceError>;

    /// Called once at the end, also after errors.
    fn finalize(&mut self) -> Result<(), ForceError> {
        Ok(())
    }
}

impl<P: ForceProvider + ?Sized> ForceProvider for Box<P> {
    fn name(&self) -> &str {
        (**self).name()
    }
    fn capabilities(&self) -> Capabilities {
        (**self).capabilities()
    }
    fn initialize(&mut self, system: &System) -> Result<(), ForceError> {
        (**self).initialize(system)
    }
    fn evaluate(
        &mut self,
        positions: &[Vec3],
        cell: Option<&Cell>,
        request: &EvalRequest,
        out: &mut ForceResult,
    ) -> Result<(), ForceError> {
        (**self).evaluate(positions, cell, request, out)
    }
    fn finalize(&mut self) -> Result<(), ForceError> {
        (**self).finalize()
    }
}

/// Outer product a ⊗ b as a matrix with element (i, j) = a_i b_j.
pub fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}
