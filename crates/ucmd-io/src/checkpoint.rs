//! Checkpoints (spec/checkpoint.md): enough to continue a run bitwise identically.
//!
//! Layout (format version 2, little endian; version 1 lacks `thermostat_state` and is still read):
//! `b"UCMDCHK\0"`, u32 format version, u64 JSON length, JSON metadata, u64 atom count,
//! positions (3N f64), velocities (3N f64), u8 cell flag, then if set the cell rows (9 f64) and
//! the periodic flags (3 u8). f64 values in the JSON use shortest round-trip formatting, so
//! they are exact.

use serde::{Deserialize, Serialize};
use std::path::Path;
use ucmd_core::{Cell, State, Vec3};

const MAGIC: &[u8; 8] = b"UCMDCHK\0";
const FORMAT_VERSION: u32 = 2;
/// Oldest format version this build reads.
const OLDEST_READABLE: u32 = 1;

/// Metadata of a checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointMeta {
    /// Spec version the run followed.
    pub spec_version: String,
    /// Engine version that wrote the checkpoint.
    pub engine_version: String,
    /// Engine git commit.
    pub engine_git_hash: String,
    /// SHA-256 of the package's system.json.
    pub system_sha256: String,
    /// SHA-256 of the resolved run configuration (without the run length).
    pub run_config_sha256: String,
    /// Completed steps.
    pub step: u64,
    /// Simulated time in ps.
    pub time_ps: f64,
    /// Work done by the thermostat so far (kJ/mol).
    pub thermostat_work_kjmol: f64,
    /// Serialized state of the thermostat's random generator, if any.
    pub thermostat_rng: Option<serde_json::Value>,
    /// The thermostat's internal variables (Nosé–Hoover chain: ξ₁…ξ_M, then v_ξ₁…v_ξ_M, in
    /// dimensionless and ps⁻¹ units). Empty for thermostats without such state (format v2).
    #[serde(default)]
    pub thermostat_state: Vec<f64>,
}

/// A checkpoint: metadata and state arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct Checkpoint {
    /// Metadata.
    pub meta: CheckpointMeta,
    /// Unwrapped positions (nm), velocities (nm/ps), cell, step and time.
    pub state: State,
}

/// Why a checkpoint could not be read or written.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("checkpoint {path}: {message}")]
pub struct CheckpointError {
    /// File.
    pub path: String,
    /// What went wrong.
    pub message: String,
}

fn put_f64s(out: &mut Vec<u8>, values: impl IntoIterator<Item = f64>) {
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

impl Checkpoint {
    /// Write atomically: to `<path>.tmp`, then rename.
    pub fn write(&self, path: &Path) -> Result<(), CheckpointError> {
        let fail = |message: String| CheckpointError {
            path: path.display().to_string(),
            message,
        };
        let json = serde_json::to_vec(&self.meta).map_err(|e| fail(e.to_string()))?;
        let n = self.state.positions.len();
        let mut bytes = Vec::with_capacity(32 + json.len() + 48 * n + 100);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(json.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&json);
        bytes.extend_from_slice(&(n as u64).to_le_bytes());
        put_f64s(
            &mut bytes,
            self.state.positions.iter().flat_map(|v| v.to_array()),
        );
        put_f64s(
            &mut bytes,
            self.state.velocities.iter().flat_map(|v| v.to_array()),
        );
        match &self.state.cell {
            Some(cell) => {
                bytes.push(1);
                put_f64s(&mut bytes, cell.rows().iter().flat_map(|v| v.to_array()));
                bytes.extend(cell.periodic().map(u8::from));
            }
            None => bytes.push(0),
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &bytes).map_err(|e| fail(e.to_string()))?;
        std::fs::rename(&tmp, path).map_err(|e| fail(e.to_string()))
    }

    /// Read a checkpoint.
    pub fn read(path: &Path) -> Result<Self, CheckpointError> {
        let fail = |message: &str| CheckpointError {
            path: path.display().to_string(),
            message: message.to_string(),
        };
        let bytes = std::fs::read(path).map_err(|e| fail(&e.to_string()))?;
        let mut cursor = Cursor {
            bytes: &bytes,
            at: 0,
        };
        if cursor.take(8).ok_or_else(|| fail("truncated"))? != MAGIC {
            return Err(fail("not a UnitcellMD checkpoint"));
        }
        let version = cursor.u32().ok_or_else(|| fail("truncated"))?;
        if !(OLDEST_READABLE..=FORMAT_VERSION).contains(&version) {
            return Err(fail(&format!(
                "format version {version} (supported: {OLDEST_READABLE} to {FORMAT_VERSION})"
            )));
        }
        let json_len = usize::try_from(cursor.u64().ok_or_else(|| fail("truncated"))?)
            .map_err(|_| fail("bad length"))?;
        let json = cursor.take(json_len).ok_or_else(|| fail("truncated"))?;
        let meta: CheckpointMeta =
            serde_json::from_slice(json).map_err(|e| fail(&e.to_string()))?;
        let n = usize::try_from(cursor.u64().ok_or_else(|| fail("truncated"))?)
            .map_err(|_| fail("bad atom count"))?;
        let positions = cursor.vectors(n).ok_or_else(|| fail("truncated"))?;
        let velocities = cursor.vectors(n).ok_or_else(|| fail("truncated"))?;
        let cell = match cursor.take(1).ok_or_else(|| fail("truncated"))?[0] {
            0 => None,
            _ => {
                let rows = cursor.vectors(3).ok_or_else(|| fail("truncated"))?;
                let flags = cursor.take(3).ok_or_else(|| fail("truncated"))?;
                let periodic = [flags[0] == 1, flags[1] == 1, flags[2] == 1];
                let rows = [rows[0].to_array(), rows[1].to_array(), rows[2].to_array()];
                Some(Cell::new(rows, periodic).map_err(|e| fail(&e.to_string()))?)
            }
        };
        let mut state = State::new(positions, Some(velocities), cell);
        state.step = meta.step;
        state.time_ps = meta.time_ps;
        Ok(Self { meta, state })
    }
}

/// Sequential reader over the checkpoint bytes.
struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        let slice = self.bytes.get(self.at..self.at.checked_add(len)?)?;
        self.at += len;
        Some(slice)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn vectors(&mut self, count: usize) -> Option<Vec<Vec3>> {
        let raw = self.take(count.checked_mul(24)?)?;
        Some(
            raw.chunks_exact(24)
                .map(|c| {
                    let f = |k: usize| {
                        f64::from_le_bytes([
                            c[k],
                            c[k + 1],
                            c[k + 2],
                            c[k + 3],
                            c[k + 4],
                            c[k + 5],
                            c[k + 6],
                            c[k + 7],
                        ])
                    };
                    Vec3::new(f(0), f(8), f(16))
                })
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_is_exact() {
        let cell = Cell::new(
            [[2.0, 0.0, 0.0], [0.3, 2.1, 0.0], [-0.2, 0.4, 1.9]],
            [true; 3],
        )
        .unwrap();
        let mut state = State::new(
            vec![
                Vec3::new(0.1, 1.0 / 3.0, -7.25e-9),
                Vec3::new(1e300, -0.0, 2.5),
            ],
            Some(vec![
                Vec3::new(0.7, 0.1, f64::MIN_POSITIVE),
                Vec3::new(-1.0, 1.0 / 7.0, 0.0),
            ]),
            Some(cell),
        );
        state.step = 1234;
        state.time_ps = 1234.0 * 0.004;
        let checkpoint = Checkpoint {
            meta: CheckpointMeta {
                spec_version: "0.1.0".into(),
                engine_version: "0.0.0".into(),
                engine_git_hash: "abc".into(),
                system_sha256: "s".into(),
                run_config_sha256: "r".into(),
                step: 1234,
                time_ps: state.time_ps,
                thermostat_work_kjmol: -0.1 / 3.0,
                thermostat_rng: Some(serde_json::json!({"seed": [1, 2, 3]})),
                thermostat_state: vec![-0.1 / 3.0, 1e-300, -0.0, 2.0f64.powi(60) + 1.0],
            },
            state,
        };
        let path =
            std::env::temp_dir().join(format!("ucmd-checkpoint-test-{}.ucchk", std::process::id()));
        checkpoint.write(&path).unwrap();
        let back = Checkpoint::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(back, checkpoint);
        assert_eq!(
            back.meta.thermostat_work_kjmol.to_bits(),
            checkpoint.meta.thermostat_work_kjmol.to_bits()
        );
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&back.meta.thermostat_state),
            bits(&checkpoint.meta.thermostat_state)
        );
    }

    /// Version 1 checkpoints (E1, without `thermostat_state`) are still read.
    #[test]
    fn reads_format_version_1() {
        let mut state = State::new(vec![Vec3::new(0.1, 0.2, 0.3)], None, None);
        state.step = 7;
        state.time_ps = 0.028;
        let meta = CheckpointMeta {
            spec_version: "0.1.0".into(),
            engine_version: "0.0.0".into(),
            engine_git_hash: "abc".into(),
            system_sha256: "s".into(),
            run_config_sha256: "r".into(),
            step: 7,
            time_ps: 0.028,
            thermostat_work_kjmol: 0.5,
            thermostat_rng: None,
            thermostat_state: Vec::new(),
        };
        let path =
            std::env::temp_dir().join(format!("ucmd-checkpoint-v1-{}.ucchk", std::process::id()));
        Checkpoint {
            meta: meta.clone(),
            state: state.clone(),
        }
        .write(&path)
        .unwrap();
        // rewrite as version 1: old version number, JSON without the new key
        let bytes = std::fs::read(&path).unwrap();
        let json_len = u64::from_le_bytes(bytes[12..20].try_into().unwrap()) as usize;
        let json = std::str::from_utf8(&bytes[20..20 + json_len]).unwrap();
        let old_json = json.replace(",\"thermostat_state\":[]", "");
        assert_ne!(old_json, json);
        let mut v1 = Vec::new();
        v1.extend_from_slice(MAGIC);
        v1.extend_from_slice(&1u32.to_le_bytes());
        v1.extend_from_slice(&(old_json.len() as u64).to_le_bytes());
        v1.extend_from_slice(old_json.as_bytes());
        v1.extend_from_slice(&bytes[20 + json_len..]);
        std::fs::write(&path, &v1).unwrap();
        let back = Checkpoint::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(back.meta, meta);
        assert_eq!(back.state, state);
    }

    #[test]
    fn rejects_foreign_and_truncated_files() {
        let path =
            std::env::temp_dir().join(format!("ucmd-checkpoint-junk-{}.ucchk", std::process::id()));
        std::fs::write(&path, b"UCMDCHK\0\x01\x00").unwrap();
        assert!(Checkpoint::read(&path).is_err());
        std::fs::write(&path, b"something else entirely").unwrap();
        assert!(Checkpoint::read(&path).is_err());
        std::fs::remove_file(&path).unwrap();
    }
}
