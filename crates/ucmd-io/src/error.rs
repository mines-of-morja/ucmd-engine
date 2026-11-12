//! The stable error codes of the spec, shared with the Python validator `ucmd-spec`.

use std::fmt;

/// Error codes of spec/system-package.md §7 and spec/run-config.md §3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    /// system.json missing or not JSON.
    Json,
    /// `format` is not "ucmd-system".
    Format,
    /// Unsupported or malformed spec version.
    Version,
    /// Units block is not the canonical one.
    Units,
    /// Schema violation.
    Schema,
    /// Unknown molecule type or atom type.
    Reference,
    /// Interaction index out of range or repeated.
    Index,
    /// `atom_count` does not match the components.
    AtomCount,
    /// Cell missing, superfluous or not in reduced form.
    Cell,
    /// positions.npy missing.
    ArrayMissing,
    /// Array has the wrong format, dtype, order or shape.
    ArrayFormat,
    /// NaN or infinity in an array.
    NonFinite,
    /// Parameter completeness rules violated.
    Parameters,
    /// Periodic system with net charge.
    NetCharge,
    /// Bonded atoms farther apart than half the cell height.
    MoleculeSplit,
    /// run.toml missing or not TOML.
    Toml,
    /// Absolute or home-relative path in run.toml.
    Path,
    /// Contradicting settings.
    Incompatible,
}

impl ErrorCode {
    /// The code as written in the spec, e.g. `"E_NET_CHARGE"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Json => "E_JSON",
            Self::Format => "E_FORMAT",
            Self::Version => "E_VERSION",
            Self::Units => "E_UNITS",
            Self::Schema => "E_SCHEMA",
            Self::Reference => "E_REFERENCE",
            Self::Index => "E_INDEX",
            Self::AtomCount => "E_ATOM_COUNT",
            Self::Cell => "E_CELL",
            Self::ArrayMissing => "E_ARRAY_MISSING",
            Self::ArrayFormat => "E_ARRAY_FORMAT",
            Self::NonFinite => "E_NONFINITE",
            Self::Parameters => "E_PARAMETERS",
            Self::NetCharge => "E_NET_CHARGE",
            Self::MoleculeSplit => "E_MOLECULE_SPLIT",
            Self::Toml => "E_TOML",
            Self::Path => "E_PATH",
            Self::Incompatible => "E_INCOMPATIBLE",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A spec violation: stable code, human-readable message, location (JSON pointer or file).
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{code}  {message}{}", if location.is_empty() { String::new() } else { format!("  ({location})") })]
pub struct SpecError {
    /// Stable code.
    pub code: ErrorCode,
    /// What is wrong.
    pub message: String,
    /// Where: JSON pointer, TOML key path or file name.
    pub location: String,
}

impl SpecError {
    /// Create an error.
    pub fn new(code: ErrorCode, message: impl Into<String>, location: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            location: location.into(),
        }
    }
}

/// x > 0, false for NaN (TOML allows `nan`, so every range check must reject it).
pub(crate) fn positive(x: f64) -> bool {
    x > 0.0
}

/// x >= 0, false for NaN.
pub(crate) fn non_negative(x: f64) -> bool {
    x >= 0.0
}
