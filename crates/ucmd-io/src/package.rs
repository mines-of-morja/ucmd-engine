//! Reader for system packages (spec/system-package.md).
//!
//! The checks run in the order of spec §7 and stop at the first error, so the codes agree with
//! the Python reference validator on every fixture (contract test `tests/contract.rs`).
//!
//! Known divergence: integer fields written with a fraction (`3.0`) are a schema error here but
//! valid for JSON Schema; producers write JSON integers.

use crate::elements::atomic_number;
use crate::error::{ErrorCode, SpecError, positive};
use crate::npy::read_vec3_array;
use serde::Deserialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;
use ucmd_core::units::RAD_PER_DEG;
use ucmd_core::{
    Angle, AngleForm, AtomTemplate, AtomType, Bond, BondForm, Cell, CellError, CombinationRule,
    Constraint, ForceField, MoleculeType, State, System, Torsion, TorsionForm, Vec3,
};

/// |Σ q| allowed for periodic systems, in e (spec §7, E_NET_CHARGE).
pub const NET_CHARGE_TOLERANCE_E: f64 = 1e-6;

/// A successfully read package.
#[derive(Debug, Clone)]
pub struct Package {
    /// The immutable system.
    pub system: System,
    /// The initial state (velocities zero when the package has none).
    pub state: State,
    /// True if the package contains velocities.npy.
    pub has_velocities: bool,
    /// Warnings as (code, message), e.g. `W_FORMAL_CHARGE`.
    pub warnings: Vec<(String, String)>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SystemDoc {
    #[allow(dead_code)]
    format: String,
    #[allow(dead_code)]
    spec_version: String,
    name: String,
    #[serde(default)]
    #[allow(dead_code)]
    description: Option<String>,
    #[allow(dead_code)]
    units: Value,
    periodicity: [bool; 3],
    cell_nm: Option<[[f64; 3]; 3]>,
    molecule_types: BTreeMap<String, MoleculeTypeDoc>,
    components: Vec<ComponentDoc>,
    atom_count: usize,
    #[serde(default)]
    forcefield: Option<ForceFieldDoc>,
    provenance: ProvenanceDoc,
}

#[derive(Deserialize)]
struct ProvenanceDoc {
    created_by: String,
    #[allow(dead_code)]
    created_utc: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ComponentDoc {
    molecule_type: String,
    count: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MoleculeTypeDoc {
    #[serde(default)]
    #[allow(dead_code)]
    description: Option<String>,
    formal_charge_e: i32,
    atoms: Vec<AtomDoc>,
    bonds: Vec<InteractionDoc<2>>,
    #[serde(default)]
    angles: Vec<InteractionDoc<3>>,
    #[serde(default)]
    dihedrals: Vec<InteractionDoc<4>>,
    #[serde(default)]
    impropers: Vec<InteractionDoc<4>>,
    #[serde(default)]
    constraints: Vec<ConstraintDoc>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AtomDoc {
    name: String,
    element: Option<String>,
    mass_amu: f64,
    #[serde(default)]
    charge_e: Option<f64>,
    #[serde(default, rename = "type")]
    atom_type: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InteractionDoc<const N: usize> {
    #[serde(with = "serde_arrays")]
    atoms: [usize; N],
    #[serde(default)]
    form: Option<String>,
    #[serde(default)]
    params: Option<Map<String, Value>>,
}

mod serde_arrays {
    use serde::{Deserialize, Deserializer};
    pub fn deserialize<'de, D: Deserializer<'de>, const N: usize>(
        d: D,
    ) -> Result<[usize; N], D::Error> {
        let v = Vec::<usize>::deserialize(d)?;
        let len = v.len();
        v.try_into()
            .map_err(|_| serde::de::Error::custom(format!("expected {N} atom indices, got {len}")))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConstraintDoc {
    atoms: [usize; 2],
    length_nm: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ForceFieldDoc {
    name: String,
    combination_rule: String,
    scale_14_lj: f64,
    scale_14_coulomb: f64,
    atom_types: BTreeMap<String, AtomTypeDoc>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AtomTypeDoc {
    sigma_nm: f64,
    epsilon_kjmol: f64,
    #[serde(default)]
    #[allow(dead_code)]
    description: Option<String>,
}

// Typed parameters of each functional form (spec §5.3); unknown keys are schema errors.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BondHarmonic {
    r0_nm: f64,
    k_kjmol_nm2: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AngleHarmonic {
    theta0_deg: f64,
    k_kjmol_rad2: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Periodic {
    k_kjmol: f64,
    n: u32,
    phi0_deg: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OplsFourier {
    v1_kjmol: f64,
    v2_kjmol: f64,
    v3_kjmol: f64,
    v4_kjmol: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RyckaertBellemans {
    c0_kjmol: f64,
    c1_kjmol: f64,
    c2_kjmol: f64,
    c3_kjmol: f64,
    c4_kjmol: f64,
    c5_kjmol: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImproperHarmonic {
    xi0_deg: f64,
    k_kjmol_rad2: f64,
}

const CANONICAL_UNITS: [(&str, &str); 7] = [
    ("length", "nm"),
    ("time", "ps"),
    ("mass", "amu"),
    ("energy", "kJ/mol"),
    ("charge", "e"),
    ("temperature", "K"),
    ("angle", "deg"),
];

fn schema(message: impl Into<String>, location: impl Into<String>) -> SpecError {
    SpecError::new(ErrorCode::Schema, message, location)
}

fn type_name_ok(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "_.+-".contains(c))
}

/// Check `spec_version`: semver with major.minor == 0.1 (spec §9).
pub(crate) fn check_spec_version(value: Option<&Value>) -> Result<(), SpecError> {
    let text = value.and_then(Value::as_str).unwrap_or("");
    let parts: Vec<&str> = text.split('.').collect();
    let semver = parts.len() == 3
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()));
    if !semver {
        return Err(SpecError::new(
            ErrorCode::Version,
            format!("spec_version must be semver, got {value:?}"),
            "/spec_version",
        ));
    }
    if parts[0] != "0" || parts[1] != "1" {
        return Err(SpecError::new(
            ErrorCode::Version,
            format!("spec_version {text} is not supported by this reader (supports 0.1.x)"),
            "/spec_version",
        ));
    }
    Ok(())
}

/// Read and validate a system package directory.
pub fn read_package(dir: &Path) -> Result<Package, SpecError> {
    let bytes = std::fs::read(dir.join("system.json")).map_err(|e| {
        SpecError::new(
            ErrorCode::Json,
            format!("system.json is missing ({e})"),
            "system.json",
        )
    })?;
    let raw: Value = serde_json::from_slice(&bytes).map_err(|e| {
        SpecError::new(
            ErrorCode::Json,
            format!("system.json is not valid JSON ({e})"),
            "system.json",
        )
    })?;
    if raw.get("format").and_then(Value::as_str) != Some("ucmd-system") {
        return Err(SpecError::new(
            ErrorCode::Format,
            "format must be \"ucmd-system\"",
            "/format",
        ));
    }
    check_spec_version(raw.get("spec_version"))?;
    let units_ok = raw
        .get("units")
        .and_then(Value::as_object)
        .is_some_and(|u| {
            u.len() == CANONICAL_UNITS.len()
                && CANONICAL_UNITS
                    .iter()
                    .all(|(k, v)| u.get(*k).and_then(Value::as_str) == Some(v))
        });
    if !units_ok {
        return Err(SpecError::new(
            ErrorCode::Units,
            "units must be the canonical block of spec §2",
            "/units",
        ));
    }
    let doc: SystemDoc =
        serde_json::from_value(raw).map_err(|e| schema(e.to_string(), "system.json"))?;
    check_schema_ranges(&doc)?;

    // E_REFERENCE
    for (k, c) in doc.components.iter().enumerate() {
        if !doc.molecule_types.contains_key(&c.molecule_type) {
            return Err(SpecError::new(
                ErrorCode::Reference,
                format!(
                    "component {k} names unknown molecule type {:?}",
                    c.molecule_type
                ),
                format!("/components/{k}"),
            ));
        }
    }
    if let Some(ff) = &doc.forcefield {
        for (name, mt) in &doc.molecule_types {
            for (k, atom) in mt.atoms.iter().enumerate() {
                if let Some(t) = &atom.atom_type
                    && !ff.atom_types.contains_key(t)
                {
                    return Err(SpecError::new(
                        ErrorCode::Reference,
                        format!("atom {k} of {name:?} names unknown atom type {t:?}"),
                        format!("/molecule_types/{name}/atoms/{k}"),
                    ));
                }
            }
        }
    }
    // E_INDEX
    for (name, mt) in &doc.molecule_types {
        let n = mt.atoms.len();
        let lists: [(&str, Vec<&[usize]>); 5] = [
            ("bonds", mt.bonds.iter().map(|i| &i.atoms[..]).collect()),
            ("angles", mt.angles.iter().map(|i| &i.atoms[..]).collect()),
            (
                "dihedrals",
                mt.dihedrals.iter().map(|i| &i.atoms[..]).collect(),
            ),
            (
                "impropers",
                mt.impropers.iter().map(|i| &i.atoms[..]).collect(),
            ),
            (
                "constraints",
                mt.constraints.iter().map(|i| &i.atoms[..]).collect(),
            ),
        ];
        for (kind, entries) in lists {
            for (k, atoms) in entries.iter().enumerate() {
                let location = format!("/molecule_types/{name}/{kind}/{k}");
                if let Some(&max) = atoms.iter().max()
                    && max >= n
                {
                    return Err(SpecError::new(
                        ErrorCode::Index,
                        format!(
                            "{kind}[{k}] of {name:?} uses atom {max}, but the type has {n} atoms"
                        ),
                        location,
                    ));
                }
                if (1..atoms.len()).any(|i| atoms[..i].contains(&atoms[i])) {
                    return Err(SpecError::new(
                        ErrorCode::Index,
                        format!("{kind}[{k}] of {name:?} repeats an atom"),
                        location,
                    ));
                }
            }
        }
    }
    // E_ATOM_COUNT
    let expected: usize = doc
        .components
        .iter()
        .map(|c| c.count * doc.molecule_types[&c.molecule_type].atoms.len())
        .sum();
    if doc.atom_count != expected {
        return Err(SpecError::new(
            ErrorCode::AtomCount,
            format!(
                "atom_count is {}, components give {expected}",
                doc.atom_count
            ),
            "/atom_count",
        ));
    }
    // E_CELL
    let periodic = doc.periodicity.iter().any(|&p| p);
    let cell = match (periodic, doc.cell_nm) {
        (true, None) => {
            return Err(SpecError::new(
                ErrorCode::Cell,
                "cell_nm is null although a direction is periodic",
                "/cell_nm",
            ));
        }
        (false, Some(_)) => {
            return Err(SpecError::new(
                ErrorCode::Cell,
                "cell_nm must be null when no direction is periodic",
                "/cell_nm",
            ));
        }
        (false, None) => None,
        (true, Some(rows)) => {
            Some(Cell::new(rows, doc.periodicity).map_err(|e: CellError| {
                SpecError::new(ErrorCode::Cell, e.to_string(), "/cell_nm")
            })?)
        }
    };
    // E_PARAMETERS
    let charges_present = check_parameters(&doc)?;
    // E_NET_CHARGE
    if periodic && charges_present {
        let total: f64 = doc
            .components
            .iter()
            .map(|c| {
                c.count as f64
                    * doc.molecule_types[&c.molecule_type]
                        .atoms
                        .iter()
                        .filter_map(|a| a.charge_e)
                        .sum::<f64>()
            })
            .sum();
        if total.abs() > NET_CHARGE_TOLERANCE_E {
            return Err(SpecError::new(
                ErrorCode::NetCharge,
                format!("periodic system has net charge {total:.6} e"),
                "/molecule_types",
            ));
        }
    }
    // arrays
    let positions = read_vec3_array(&dir.join("positions.npy"), doc.atom_count)?;
    let velocity_path = dir.join("velocities.npy");
    let has_velocities = velocity_path.exists();
    let velocities = if has_velocities {
        Some(read_vec3_array(&velocity_path, doc.atom_count)?)
    } else {
        None
    };
    // E_MOLECULE_SPLIT
    if let Some(cell) = &cell {
        check_whole_molecules(&doc, cell, &positions)?;
    }
    let mut warnings = Vec::new();
    if periodic {
        let formal: i64 = doc
            .components
            .iter()
            .map(|c| {
                c.count as i64 * i64::from(doc.molecule_types[&c.molecule_type].formal_charge_e)
            })
            .sum();
        if formal != 0 {
            warnings.push((
                "W_FORMAL_CHARGE".to_string(),
                format!("formal charges sum to {formal} e, not 0"),
            ));
        }
    }

    let mut system = build_system(&doc)?;
    system.system_sha256 = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let state = State::new(positions, velocities, cell);
    Ok(Package {
        system,
        state,
        has_velocities,
        warnings,
    })
}

/// Ranges and patterns that the JSON Schema enforces but serde does not.
fn check_schema_ranges(doc: &SystemDoc) -> Result<(), SpecError> {
    if doc.name.is_empty() {
        return Err(schema("name must not be empty", "/name"));
    }
    if doc.provenance.created_by.is_empty() {
        return Err(schema(
            "provenance.created_by must not be empty",
            "/provenance/created_by",
        ));
    }
    if doc.molecule_types.is_empty() {
        return Err(schema(
            "molecule_types must not be empty",
            "/molecule_types",
        ));
    }
    if doc.components.is_empty() {
        return Err(schema("components must not be empty", "/components"));
    }
    if doc.atom_count < 1 {
        return Err(schema("atom_count must be at least 1", "/atom_count"));
    }
    for (k, c) in doc.components.iter().enumerate() {
        if c.count < 1 || !type_name_ok(&c.molecule_type) {
            return Err(schema(
                "component needs a valid molecule_type and count >= 1",
                format!("/components/{k}"),
            ));
        }
    }
    for (name, mt) in &doc.molecule_types {
        let at = |rest: String| format!("/molecule_types/{name}{rest}");
        if !type_name_ok(name) {
            return Err(schema(
                format!("invalid molecule type name {name:?}"),
                at(String::new()),
            ));
        }
        if mt.atoms.is_empty() {
            return Err(schema("atoms must not be empty", at("/atoms".into())));
        }
        for (k, atom) in mt.atoms.iter().enumerate() {
            let bad = atom.name.is_empty()
                || !positive(atom.mass_amu)
                || atom.charge_e.is_some_and(|q| !q.is_finite())
                || atom.atom_type.as_deref().is_some_and(|t| !type_name_ok(t))
                || atom
                    .element
                    .as_deref()
                    .is_some_and(|e| atomic_number(e).is_none());
            if bad {
                return Err(schema(
                    format!("atom {k}: name, element, mass_amu > 0, charge_e or type invalid"),
                    at(format!("/atoms/{k}")),
                ));
            }
        }
        for (k, c) in mt.constraints.iter().enumerate() {
            if !positive(c.length_nm) {
                return Err(schema(
                    "length_nm must be > 0",
                    at(format!("/constraints/{k}")),
                ));
            }
        }
        check_forms(&mt.bonds, &["harmonic"], "bonds", &at)?;
        check_forms(&mt.angles, &["harmonic"], "angles", &at)?;
        check_forms(
            &mt.dihedrals,
            &["periodic", "opls-fourier", "ryckaert-bellemans"],
            "dihedrals",
            &at,
        )?;
        check_forms(&mt.impropers, &["periodic", "harmonic"], "impropers", &at)?;
    }
    if let Some(ff) = &doc.forcefield {
        let ok = !ff.name.is_empty()
            && ["geometric", "lorentz-berthelot"].contains(&ff.combination_rule.as_str())
            && (0.0..=1.0).contains(&ff.scale_14_lj)
            && (0.0..=1.0).contains(&ff.scale_14_coulomb)
            && !ff.atom_types.is_empty()
            && ff
                .atom_types
                .iter()
                .all(|(n, t)| type_name_ok(n) && t.sigma_nm >= 0.0 && t.epsilon_kjmol >= 0.0);
        if !ok {
            return Err(schema(
                "forcefield: name, combination_rule, scale factors in [0, 1] or atom types invalid",
                "/forcefield",
            ));
        }
    }
    Ok(())
}

fn check_forms<const N: usize>(
    list: &[InteractionDoc<N>],
    allowed: &[&str],
    kind: &str,
    at: &dyn Fn(String) -> String,
) -> Result<(), SpecError> {
    for (k, entry) in list.iter().enumerate() {
        let location = at(format!("/{kind}/{k}"));
        match (&entry.form, &entry.params) {
            (None, None) => {}
            (Some(form), Some(params)) => {
                if !allowed.contains(&form.as_str()) {
                    return Err(schema(format!("unknown {kind} form {form:?}"), location));
                }
                parse_form(kind, form, params).map_err(|m| schema(m, location.clone()))?;
            }
            _ => return Err(schema("form and params must be given together", location)),
        }
    }
    Ok(())
}

enum ParsedForm {
    Bond(BondForm),
    Angle(AngleForm),
    Torsion(TorsionForm),
}

fn parse_form(kind: &str, form: &str, params: &Map<String, Value>) -> Result<ParsedForm, String> {
    let value = Value::Object(params.clone());
    let parsed = match (kind, form) {
        ("bonds", "harmonic") => {
            let p: BondHarmonic = serde_json::from_value(value).map_err(|e| e.to_string())?;
            if !positive(p.r0_nm) || p.k_kjmol_nm2 < 0.0 {
                return Err("r0_nm must be > 0 and k_kjmol_nm2 >= 0".into());
            }
            ParsedForm::Bond(BondForm::Harmonic {
                r0_nm: p.r0_nm,
                k_kjmol_nm2: p.k_kjmol_nm2,
            })
        }
        ("angles", "harmonic") => {
            let p: AngleHarmonic = serde_json::from_value(value).map_err(|e| e.to_string())?;
            if !(0.0..=180.0).contains(&p.theta0_deg) || p.k_kjmol_rad2 < 0.0 {
                return Err("theta0_deg must be in [0, 180] and k_kjmol_rad2 >= 0".into());
            }
            ParsedForm::Angle(AngleForm::Harmonic {
                theta0_rad: p.theta0_deg * RAD_PER_DEG,
                k_kjmol_rad2: p.k_kjmol_rad2,
            })
        }
        (_, "periodic") => {
            let p: Periodic = serde_json::from_value(value).map_err(|e| e.to_string())?;
            if p.n > 6 || !(-360.0..=360.0).contains(&p.phi0_deg) {
                return Err("n must be 0..6 and phi0_deg in [-360, 360]".into());
            }
            ParsedForm::Torsion(TorsionForm::Periodic {
                k_kjmol: p.k_kjmol,
                n: p.n,
                phi0_rad: p.phi0_deg * RAD_PER_DEG,
            })
        }
        ("dihedrals", "opls-fourier") => {
            let p: OplsFourier = serde_json::from_value(value).map_err(|e| e.to_string())?;
            ParsedForm::Torsion(TorsionForm::OplsFourier {
                v_kjmol: [p.v1_kjmol, p.v2_kjmol, p.v3_kjmol, p.v4_kjmol],
            })
        }
        ("dihedrals", "ryckaert-bellemans") => {
            let p: RyckaertBellemans = serde_json::from_value(value).map_err(|e| e.to_string())?;
            ParsedForm::Torsion(TorsionForm::RyckaertBellemans {
                c_kjmol: [
                    p.c0_kjmol, p.c1_kjmol, p.c2_kjmol, p.c3_kjmol, p.c4_kjmol, p.c5_kjmol,
                ],
            })
        }
        ("impropers", "harmonic") => {
            let p: ImproperHarmonic = serde_json::from_value(value).map_err(|e| e.to_string())?;
            if !(-180.0..=180.0).contains(&p.xi0_deg) || p.k_kjmol_rad2 < 0.0 {
                return Err("xi0_deg must be in [-180, 180] and k_kjmol_rad2 >= 0".into());
            }
            ParsedForm::Torsion(TorsionForm::Harmonic {
                xi0_rad: p.xi0_deg * RAD_PER_DEG,
                k_kjmol_rad2: p.k_kjmol_rad2,
            })
        }
        _ => return Err(format!("unknown {kind} form {form:?}")),
    };
    Ok(parsed)
}

fn check_parameters(doc: &SystemDoc) -> Result<bool, SpecError> {
    let atoms: Vec<&AtomDoc> = doc.molecule_types.values().flat_map(|m| &m.atoms).collect();
    let with_charge = atoms.iter().filter(|a| a.charge_e.is_some()).count();
    if with_charge != 0 && with_charge != atoms.len() {
        return Err(SpecError::new(
            ErrorCode::Parameters,
            "charge_e must be given for all atoms or for none",
            "",
        ));
    }
    let parameterized = doc.forcefield.is_some();
    for (name, mt) in &doc.molecule_types {
        if parameterized {
            if let Some(k) = mt
                .atoms
                .iter()
                .position(|a| a.atom_type.is_none() || a.charge_e.is_none())
            {
                return Err(SpecError::new(
                    ErrorCode::Parameters,
                    format!("parameterized package: atom {k} of {name:?} needs type and charge_e"),
                    format!("/molecule_types/{name}/atoms/{k}"),
                ));
            }
        }
        let forms = [
            (
                "bonds",
                mt.bonds
                    .iter()
                    .map(|i| i.form.is_some())
                    .collect::<Vec<_>>(),
            ),
            (
                "angles",
                mt.angles.iter().map(|i| i.form.is_some()).collect(),
            ),
            (
                "dihedrals",
                mt.dihedrals.iter().map(|i| i.form.is_some()).collect(),
            ),
            (
                "impropers",
                mt.impropers.iter().map(|i| i.form.is_some()).collect(),
            ),
        ];
        for (kind, has_form) in forms {
            if let Some(k) = has_form.iter().position(|&f| f != parameterized) {
                let message = if parameterized {
                    format!("parameterized package: {kind}[{k}] of {name:?} needs form and params")
                } else {
                    format!(
                        "{kind}[{k}] of {name:?} has parameters, but the package has no forcefield"
                    )
                };
                return Err(SpecError::new(
                    ErrorCode::Parameters,
                    message,
                    format!("/molecule_types/{name}/{kind}/{k}"),
                ));
            }
        }
    }
    Ok(with_charge == atoms.len() && !atoms.is_empty())
}

fn check_whole_molecules(
    doc: &SystemDoc,
    cell: &Cell,
    positions: &[Vec3],
) -> Result<(), SpecError> {
    let limit = cell.max_minimum_image_distance();
    let mut first = 0usize;
    for component in &doc.components {
        let mt = &doc.molecule_types[&component.molecule_type];
        let n = mt.atoms.len();
        let pairs: Vec<[usize; 2]> = mt
            .bonds
            .iter()
            .map(|b| b.atoms)
            .chain(mt.constraints.iter().map(|c| c.atoms))
            .collect();
        for instance in 0..component.count {
            let offset = first + instance * n;
            for [i, j] in &pairs {
                let d = (positions[offset + i] - positions[offset + j]).length();
                if d > limit {
                    return Err(SpecError::new(
                        ErrorCode::MoleculeSplit,
                        format!(
                            "atoms {} and {} ({:?}) are {d:.4} nm apart, more than half the smallest cell height ({limit:.4} nm): the molecule is not whole",
                            offset + i,
                            offset + j,
                            component.molecule_type
                        ),
                        "positions.npy",
                    ));
                }
            }
        }
        first += n * component.count;
    }
    Ok(())
}

fn build_system(doc: &SystemDoc) -> Result<System, SpecError> {
    let type_index: BTreeMap<&str, usize> = doc
        .forcefield
        .iter()
        .flat_map(|ff| ff.atom_types.keys())
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    let forcefield = doc.forcefield.as_ref().map(|ff| ForceField {
        name: ff.name.clone(),
        combination_rule: if ff.combination_rule == "geometric" {
            CombinationRule::Geometric
        } else {
            CombinationRule::LorentzBerthelot
        },
        scale_14_lj: ff.scale_14_lj,
        scale_14_coulomb: ff.scale_14_coulomb,
        atom_types: ff
            .atom_types
            .iter()
            .map(|(name, t)| AtomType {
                name: name.clone(),
                sigma_nm: t.sigma_nm,
                epsilon_kjmol: t.epsilon_kjmol,
            })
            .collect(),
    });
    let names: Vec<&String> = doc.molecule_types.keys().collect();
    let mut molecule_types = Vec::with_capacity(names.len());
    for (name, mt) in &doc.molecule_types {
        let form = |kind: &str,
                    f: &Option<String>,
                    p: &Option<Map<String, Value>>|
         -> Result<Option<ParsedForm>, SpecError> {
            match (f, p) {
                (Some(f), Some(p)) => parse_form(kind, f, p)
                    .map(Some)
                    .map_err(|m| schema(m, format!("/molecule_types/{name}"))),
                _ => Ok(None),
            }
        };
        let mut bonds = Vec::new();
        for b in &mt.bonds {
            let parsed = match form("bonds", &b.form, &b.params)? {
                Some(ParsedForm::Bond(f)) => Some(f),
                _ => None,
            };
            bonds.push(Bond {
                atoms: b.atoms,
                form: parsed,
            });
        }
        let mut angles = Vec::new();
        for a in &mt.angles {
            let parsed = match form("angles", &a.form, &a.params)? {
                Some(ParsedForm::Angle(f)) => Some(f),
                _ => None,
            };
            angles.push(Angle {
                atoms: a.atoms,
                form: parsed,
            });
        }
        let torsions =
            |kind: &str, list: &[InteractionDoc<4>]| -> Result<Vec<Torsion>, SpecError> {
                list.iter()
                    .map(|t| {
                        let parsed = match form(kind, &t.form, &t.params)? {
                            Some(ParsedForm::Torsion(f)) => Some(f),
                            _ => None,
                        };
                        Ok(Torsion {
                            atoms: t.atoms,
                            form: parsed,
                        })
                    })
                    .collect()
            };
        molecule_types.push(MoleculeType {
            name: name.clone(),
            formal_charge_e: mt.formal_charge_e,
            atoms: mt
                .atoms
                .iter()
                .map(|a| AtomTemplate {
                    name: a.name.clone(),
                    element: a.element.as_deref().and_then(atomic_number).unwrap_or(0),
                    mass_amu: a.mass_amu,
                    charge_e: a.charge_e,
                    atom_type: a
                        .atom_type
                        .as_deref()
                        .and_then(|t| type_index.get(t).copied()),
                })
                .collect(),
            bonds,
            angles,
            dihedrals: torsions("dihedrals", &mt.dihedrals)?,
            impropers: torsions("impropers", &mt.impropers)?,
            constraints: mt
                .constraints
                .iter()
                .map(|c| Constraint {
                    atoms: c.atoms,
                    length_nm: c.length_nm,
                })
                .collect(),
        });
    }
    let components: Vec<(usize, usize)> = doc
        .components
        .iter()
        .map(|c| {
            (
                names
                    .iter()
                    .position(|n| **n == c.molecule_type)
                    .unwrap_or(usize::MAX),
                c.count,
            )
        })
        .collect();
    System::new(
        doc.name.clone(),
        molecule_types,
        &components,
        forcefield,
        doc.periodicity,
    )
    .map_err(|e| schema(e.to_string(), "system.json"))
}
