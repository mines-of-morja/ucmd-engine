//! The provider registry: the only place that knows the concrete force providers
//! (PLAN.md §2.7, CLAUDE.md invariant 1).

use anyhow::{Context, Result, bail};
use ucmd_core::{State, Vec3};
use ucmd_force::{Composite, ForceProvider};
use ucmd_io::RunConfig;
use ucmd_providers_classical::{HarmonicTether, LennardJones, LennardJonesConfig};

/// The provider kinds this build knows.
pub const KINDS: [&str; 2] = ["harmonic-tether", "lennard-jones"];

/// Build one provider from its `[[providers]]` table.
pub fn make_provider(table: &toml::Table, initial: &State) -> Result<Box<dyn ForceProvider>> {
    let kind = table
        .get("kind")
        .and_then(toml::Value::as_str)
        .unwrap_or_default();
    let mut keys = table.clone();
    keys.remove("kind");
    keys.remove("name");
    let value = toml::Value::Table(keys);
    match kind {
        "lennard-jones" => {
            let config: LennardJonesConfig = value
                .try_into()
                .context("[[providers]] kind = \"lennard-jones\"")?;
            Ok(Box::new(LennardJones::new(config)))
        }
        "harmonic-tether" => {
            #[derive(serde::Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Tether {
                k_kjmol_nm2: f64,
                #[serde(default)]
                anchor: Option<String>,
            }
            let t: Tether = value
                .try_into()
                .context("[[providers]] kind = \"harmonic-tether\"")?;
            let anchors = match t.anchor.as_deref().unwrap_or("origin") {
                "origin" => vec![Vec3::ZERO; initial.n_atoms()],
                "initial" => initial.positions.clone(),
                other => bail!(
                    "harmonic-tether: anchor must be \"origin\" or \"initial\", got {other:?}"
                ),
            };
            Ok(Box::new(HarmonicTether::new(t.k_kjmol_nm2, anchors)))
        }
        other => bail!(
            "unknown provider kind {other:?}; this build knows {}",
            KINDS.join(", ")
        ),
    }
}

/// All providers of a run: a single provider as is, several as a composite.
pub fn make_providers(config: &RunConfig, initial: &State) -> Result<Box<dyn ForceProvider>> {
    if config.providers.len() == 1 {
        return make_provider(&config.providers[0], initial);
    }
    let mut composite = Composite::new("providers");
    for table in &config.providers {
        let kind = table
            .get("kind")
            .and_then(toml::Value::as_str)
            .unwrap_or_default();
        let name = table
            .get("name")
            .and_then(toml::Value::as_str)
            .unwrap_or(kind)
            .to_string();
        composite.push(name, make_provider(table, initial)?);
    }
    Ok(Box::new(composite))
}
