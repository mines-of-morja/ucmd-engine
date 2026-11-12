//! Strict reader for the state arrays of spec/system-package.md §6: NPY 1.0/2.0, `<f8`,
//! C order, shape (n, 3).

use crate::error::{ErrorCode, SpecError};
use std::path::Path;
use ucmd_core::Vec3;

/// Read an (n_atoms, 3) array of little-endian f64 in C order.
pub fn read_vec3_array(path: &Path, n_atoms: usize) -> Result<Vec<Vec3>, SpecError> {
    let what = path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    );
    let bytes = std::fs::read(path).map_err(|e| {
        SpecError::new(
            ErrorCode::ArrayMissing,
            format!("{what} is missing ({e})"),
            &what,
        )
    })?;
    let format =
        |msg: String| SpecError::new(ErrorCode::ArrayFormat, format!("{what}: {msg}"), &what);
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(format("not an NPY file".into()));
    }
    let (header_len, start) = match (bytes[6], bytes[7]) {
        (1, 0) => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
        (2, 0) if bytes.len() >= 12 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        (major, minor) => {
            return Err(format(format!(
                "NPY version {major}.{minor} (allowed: 1.0, 2.0)"
            )));
        }
    };
    let header = bytes
        .get(start..start + header_len)
        .and_then(|h| std::str::from_utf8(h).ok())
        .ok_or_else(|| format("truncated or non-UTF-8 header".into()))?;
    let field = |key: &str| -> Option<&str> {
        let at = header.find(&format!("'{key}'"))? + key.len() + 2;
        let rest = header[at..].trim_start().strip_prefix(':')?.trim_start();
        Some(rest)
    };
    let descr = field("descr")
        .and_then(|r| r.strip_prefix('\''))
        .and_then(|r| r.split('\'').next());
    if descr != Some("<f8") {
        return Err(format(format!(
            "dtype {} (required: <f8)",
            descr.unwrap_or("?")
        )));
    }
    if !field("fortran_order").is_some_and(|r| r.starts_with("False")) {
        return Err(format("Fortran order (required: C order)".into()));
    }
    let shape: Option<Vec<usize>> = field("shape")
        .and_then(|r| r.strip_prefix('('))
        .and_then(|r| r.split(')').next())
        .map(|inner| {
            inner
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .filter_map(|s| s.parse().ok())
                .collect()
        });
    if shape.as_deref() != Some(&[n_atoms, 3][..]) {
        return Err(format(format!(
            "shape {shape:?} (required: ({n_atoms}, 3))"
        )));
    }
    let data = bytes
        .get(start + header_len..)
        .ok_or_else(|| format("missing data".into()))?;
    if data.len() < n_atoms * 24 {
        return Err(format("file is truncated".into()));
    }
    let mut out = Vec::with_capacity(n_atoms);
    for (atom, chunk) in data.chunks_exact(24).take(n_atoms).enumerate() {
        let mut v = [0.0; 3];
        for (k, b) in chunk.chunks_exact(8).enumerate() {
            v[k] = f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
            if !v[k].is_finite() {
                return Err(SpecError::new(
                    ErrorCode::NonFinite,
                    format!("{what}: non-finite value at atom {atom}, component {k}"),
                    &what,
                ));
            }
        }
        out.push(Vec3::from_array(v));
    }
    Ok(out)
}

/// Write an (n, 3) array as NPY 1.0, `<f8`, C order.
pub fn write_vec3_array(path: &Path, values: &[Vec3]) -> std::io::Result<()> {
    let mut header = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': ({}, 3), }}",
        values.len()
    );
    // pad so that magic + version + length + header (+ newline) is a multiple of 64 bytes
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut bytes = Vec::with_capacity(10 + header.len() + 24 * values.len());
    bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
    bytes.extend_from_slice(
        &u16::try_from(header.len())
            .unwrap_or(u16::MAX)
            .to_le_bytes(),
    );
    bytes.extend_from_slice(header.as_bytes());
    for v in values {
        for x in v.to_array() {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(path, bytes)
}
