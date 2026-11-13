//! Running in segments between checkpoints, and the emergency checkpoint on provider errors
//! (spec/checkpoint.md, PLAN.md E1.10).

use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};
use ucmd_force::{ForceProvider, Observer};
use ucmd_io::{Checkpoint, CheckpointMeta};
use ucmd_md::{Dynamics, RunError};

/// Where and how often to write checkpoints.
pub struct CheckpointPlan {
    /// Checkpoint file.
    pub path: PathBuf,
    /// Interval in steps.
    pub every: u64,
}

/// A checkpoint of the current state of `dynamics`; `base` provides the identity fields.
pub fn checkpoint_of<P: ForceProvider>(
    dynamics: &Dynamics<P>,
    base: &CheckpointMeta,
) -> Result<Checkpoint> {
    let state = dynamics.state().clone();
    let thermostat_rng = dynamics
        .thermostat()
        .and_then(|t| t.rng())
        .map(serde_json::to_value)
        .transpose()?;
    Ok(Checkpoint {
        meta: CheckpointMeta {
            step: state.step,
            time_ps: state.time_ps,
            thermostat_work_kjmol: dynamics.thermostat_work(),
            thermostat_rng,
            thermostat_state: dynamics.thermostat_state(),
            ..base.clone()
        },
        state,
    })
}

/// How a run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Every requested step was done.
    Completed,
    /// `stop` asked to end early; the state after `step` is in the checkpoint at `path`.
    Stopped {
        /// The last completed step.
        step: u64,
    },
}

/// Run `steps` more steps. With `observe_start` the current state is recorded first. A
/// checkpoint is written at every multiple of `plan.every` and at the end. If the provider
/// fails, the last completed state is written to `emergency` before the error is returned.
/// `stop` is asked after every step; when it says yes, the current state is written to the
/// checkpoint file (or to `emergency` without one) and the run ends with [`Outcome::Stopped`].
#[allow(clippy::too_many_arguments)]
pub fn drive<P: ForceProvider>(
    dynamics: &mut Dynamics<P>,
    steps: u64,
    observers: &mut [Box<dyn Observer>],
    observe_start: bool,
    plan: Option<&CheckpointPlan>,
    emergency: &Path,
    base: &CheckpointMeta,
    stop: &dyn Fn() -> bool,
) -> Result<Outcome> {
    let end = dynamics.state().step + steps;
    let fail = |dynamics: &Dynamics<P>, error: RunError| -> anyhow::Error {
        if matches!(error, RunError::Provider { .. }) {
            match checkpoint_of(dynamics, base)
                .and_then(|c| c.write(emergency).map_err(|e| anyhow!("{e}")))
            {
                Ok(()) => {
                    return anyhow!(
                        "{error}; last good state (step {}) saved to {}",
                        dynamics.state().step,
                        emergency.display()
                    );
                }
                Err(e) => return anyhow!("{error}; the emergency checkpoint failed too: {e}"),
            }
        }
        anyhow!("{error}")
    };
    if observe_start {
        dynamics
            .run_observed(0, observers, true)
            .map_err(|e| fail(dynamics, e))?;
    }
    while dynamics.state().step < end {
        if stop() {
            let path = plan.map_or(emergency, |p| p.path.as_path());
            checkpoint_of(dynamics, base)?
                .write(path)
                .map_err(|e| anyhow!("{e}"))?;
            return Ok(Outcome::Stopped {
                step: dynamics.state().step,
            });
        }
        dynamics
            .run_observed(1, observers, false)
            .map_err(|e| fail(dynamics, e))?;
        if let Some(plan) = plan {
            let now = dynamics.state().step;
            if now % plan.every == 0 || now == end {
                checkpoint_of(dynamics, base)?
                    .write(&plan.path)
                    .map_err(|e| anyhow!("{e}"))?;
            }
        }
    }
    Ok(Outcome::Completed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucmd_core::{AtomTemplate, Cell, MoleculeType, State, System, Vec3};
    use ucmd_force::{Capabilities, EvalRequest, ForceError, ForceResult};
    use ucmd_md::DynamicsConfig;

    /// A tether that fails when asked for a given step.
    struct FailAt(u64);

    impl ForceProvider for FailAt {
        fn name(&self) -> &str {
            "fail-at"
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
            r: &EvalRequest,
            out: &mut ForceResult,
        ) -> Result<(), ForceError> {
            if r.step == self.0 {
                return Err(ForceError::NotConverged(
                    "SCF did not converge (test)".into(),
                ));
            }
            out.energy = Some(0.5 * 100.0 * x.iter().map(|p| p.length_squared()).sum::<f64>());
            for (f, p) in out.forces.iter_mut().zip(x) {
                *f = -100.0 * *p;
            }
            Ok(())
        }
    }

    fn base() -> CheckpointMeta {
        CheckpointMeta {
            spec_version: "0.1.0".into(),
            engine_version: "test".into(),
            engine_git_hash: "test".into(),
            system_sha256: String::new(),
            run_config_sha256: String::new(),
            step: 0,
            time_ps: 0.0,
            thermostat_work_kjmol: 0.0,
            thermostat_rng: None,
            thermostat_state: Vec::new(),
        }
    }

    /// E1.10: a provider failure leaves an emergency checkpoint of the last completed step.
    #[test]
    fn provider_failure_writes_an_emergency_checkpoint() {
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
        let system = System::new("t", vec![mtype], &[(0, 2)], None, [false; 3]).unwrap();
        let state = State::new(vec![Vec3::X * 0.1, Vec3::Y * 0.1], None, None);
        let mut dynamics =
            Dynamics::new(&system, FailAt(7), state, DynamicsConfig::nve(0.001), None).unwrap();
        let dir = std::env::temp_dir().join(format!("ucmd-emergency-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let emergency = dir.join("checkpoint.emergency.ucchk");
        let plan = CheckpointPlan {
            path: dir.join("checkpoint.ucchk"),
            every: 5,
        };
        let error = drive(
            &mut dynamics,
            20,
            &mut [],
            true,
            Some(&plan),
            &emergency,
            &base(),
            &|| false,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("SCF did not converge"),
            "{error}"
        );
        let saved = Checkpoint::read(&emergency).unwrap();
        assert_eq!(saved.meta.step, 6);
        assert_eq!(&saved.state, dynamics.state());
        assert_eq!(Checkpoint::read(&plan.path).unwrap().meta.step, 5); // the regular checkpoint
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A stop request ends the run after the current step and saves that state to the
    /// checkpoint file (B2a: graceful stop on SIGINT/SIGTERM).
    #[test]
    fn a_stop_request_saves_the_current_state() {
        let mtype = MoleculeType {
            name: "p".into(),
            formal_charge_e: 0,
            atoms: vec![AtomTemplate {
                name: "P".into(),
                element: 0,
                mass_amu: 1.0,
                charge_e: None,
                atom_type: None,
            }],
            bonds: vec![],
            angles: vec![],
            dihedrals: vec![],
            impropers: vec![],
            constraints: vec![],
        };
        let system = System::new("t", vec![mtype], &[(0, 2)], None, [false; 3]).unwrap();
        let state = State::new(vec![Vec3::X * 0.1, Vec3::Y * 0.1], None, None);
        let mut dynamics = Dynamics::new(
            &system,
            FailAt(u64::MAX),
            state,
            DynamicsConfig::nve(0.001),
            None,
        )
        .unwrap();
        let dir = std::env::temp_dir().join(format!("ucmd-stop-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let plan = CheckpointPlan {
            path: dir.join("checkpoint.ucchk"),
            every: 5,
        };
        let asked = std::cell::Cell::new(0u64);
        let outcome = drive(
            &mut dynamics,
            20,
            &mut [],
            true,
            Some(&plan),
            &dir.join("emergency.ucchk"),
            &base(),
            &|| {
                asked.set(asked.get() + 1);
                asked.get() > 7 // stop before the eighth step
            },
        )
        .unwrap();
        assert_eq!(outcome, Outcome::Stopped { step: 7 });
        let saved = Checkpoint::read(&plan.path).unwrap();
        assert_eq!(saved.meta.step, 7);
        assert_eq!(&saved.state, dynamics.state());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
