//! Extended-XYZ debug trajectory for viewers (OVITO, VMD, ASE).
//!
//! Viewers expect Å, so this exporter is the one engine output in Å (a format boundary,
//! CLAUDE.md invariant 3). Every frame is flushed when it is complete, so a viewer can open the
//! trajectory of a running simulation.

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
        Self::open(path, every, elements, false)
    }

    /// Reopen after a restart: keep the frames up to `last_step`, then append.
    pub fn open_append(
        path: &Path,
        every: u64,
        elements: &[u8],
        last_step: u64,
    ) -> Result<Self, ObserverError> {
        let text = std::fs::read_to_string(path).map_err(io)?;
        let lines: Vec<&str> = text.lines().collect();
        let mut kept = String::new();
        let mut at = 0;
        while at < lines.len() {
            let n: usize = lines[at].trim().parse().map_err(|_| {
                ObserverError(format!("xyz trajectory: bad frame at line {}", at + 1))
            })?;
            let step = lines
                .get(at + 1)
                .and_then(|c| c.split_whitespace().find_map(|w| w.strip_prefix("step=")))
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(u64::MAX);
            if step > last_step || at + 2 + n > lines.len() {
                break;
            }
            for line in &lines[at..at + 2 + n] {
                kept.push_str(line);
                kept.push('\n');
            }
            at += 2 + n;
        }
        std::fs::write(path, kept).map_err(io)?;
        Self::open(path, every, elements, true)
    }

    fn open(path: &Path, every: u64, elements: &[u8], append: bool) -> Result<Self, ObserverError> {
        let symbols = elements
            .iter()
            .map(|&z| symbol(z).unwrap_or("X").to_string())
            .collect();
        let file = File::options()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(path)
            .map_err(io)?;
        Ok(Self {
            out: BufWriter::new(file),
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
        self.out.flush().map_err(io)
    }

    fn finish(&mut self) -> Result<(), ObserverError> {
        self.out.flush().map_err(io)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucmd_core::Vec3;
    use ucmd_force::ForceResult;

    /// Each frame is complete on disk once `observe` returns, so viewers can read a running
    /// trajectory (ASE refuses a file whose last frame is cut).
    #[test]
    fn frames_are_complete_after_each_observation() {
        let path = std::env::temp_dir().join(format!("ucmd-xyz-test-{}.xyz", std::process::id()));
        let n = 2000; // a frame larger than the write buffer
        let positions = vec![Vec3::new(0.1, 0.2, 0.3); n];
        let velocities = vec![Vec3::ZERO; n];
        let forces = ForceResult::default();
        let mut writer = XyzWriter::create(&path, 1, &vec![6; n]).unwrap();
        for step in 0..2u64 {
            writer
                .observe(&Snapshot {
                    step,
                    time_ps: 0.001 * step as f64,
                    positions: &positions,
                    velocities: &velocities,
                    cell: None,
                    forces: &forces,
                    kinetic_energy: 0.0,
                    temperature_k: 0.0,
                    conserved_energy: None,
                    pressure_bar: None,
                })
                .unwrap();
            let text = std::fs::read_to_string(&path).unwrap();
            assert_eq!(text.lines().count(), (step as usize + 1) * (n + 2));
        }
        drop(writer);
        std::fs::remove_file(&path).unwrap();
    }
}
