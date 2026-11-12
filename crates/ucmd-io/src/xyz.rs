//! Extended-XYZ debug trajectory for viewers (OVITO, VMD, ASE).
//!
//! Viewers expect Å, so this exporter is the one engine output in Å (a format boundary,
//! CLAUDE.md invariant 3).

use crate::elements::symbol;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use ucmd_force::{Observer, ObserverError, Snapshot};

const ANGSTROM_PER_NM: f64 = 10.0;

/// Appends one extended-XYZ frame per observed step.
pub struct XyzWriter {
    out: BufWriter<File>,
    every: u64,
    symbols: Vec<String>,
}

fn io(e: std::io::Error) -> ObserverError {
    ObserverError(format!("xyz trajectory: {e}"))
}

impl XyzWriter {
    /// Create (or truncate) `path`; `elements` are atomic numbers (0 is written as "X").
    pub fn create(path: &Path, every: u64, elements: &[u8]) -> Result<Self, ObserverError> {
        let symbols = elements
            .iter()
            .map(|&z| symbol(z).unwrap_or("X").to_string())
            .collect();
        Ok(Self {
            out: BufWriter::new(File::create(path).map_err(io)?),
            every: every.max(1),
            symbols,
        })
    }
}

impl Observer for XyzWriter {
    fn every(&self) -> u64 {
        self.every
    }

    fn needs(&self) -> (bool, bool) {
        (false, false)
    }

    fn observe(&mut self, s: &Snapshot<'_>) -> Result<(), ObserverError> {
        writeln!(self.out, "{}", s.positions.len()).map_err(io)?;
        let mut comment = String::from("Properties=species:S:1:pos:R:3");
        if let Some(cell) = s.cell {
            let rows: Vec<String> = cell
                .rows()
                .iter()
                .flat_map(|v| v.to_array())
                .map(|x| format!("{:.8}", x * ANGSTROM_PER_NM))
                .collect();
            let pbc: Vec<&str> = cell
                .periodic()
                .iter()
                .map(|&p| if p { "T" } else { "F" })
                .collect();
            comment = format!(
                "Lattice=\"{}\" {comment} pbc=\"{}\"",
                rows.join(" "),
                pbc.join(" ")
            );
        }
        writeln!(self.out, "{comment} step={} time_ps={}", s.step, s.time_ps).map_err(io)?;
        for (sym, r) in self.symbols.iter().zip(s.positions) {
            let a = *r * ANGSTROM_PER_NM;
            writeln!(self.out, "{sym} {:.8} {:.8} {:.8}", a.x, a.y, a.z).map_err(io)?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), ObserverError> {
        self.out.flush().map_err(io)
    }
}
