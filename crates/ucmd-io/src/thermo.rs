//! Thermo table: one tab-separated line per observed step, units in the header.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use ucmd_force::{Observer, ObserverError, Snapshot};

/// Writes `step, time, energies, temperature, pressure, volume, energy terms` as TSV, with the
/// column names of spec/run-config.md (`conserved_energy_kjmol` is the conserved quantity).
pub struct ThermoWriter {
    out: BufWriter<File>,
    every: u64,
    terms: Option<Vec<String>>,
}

fn io(e: std::io::Error) -> ObserverError {
    ObserverError(format!("thermo table: {e}"))
}

fn num(x: Option<f64>) -> String {
    x.map_or_else(|| "nan".to_string(), |v| format!("{v:.10e}"))
}

impl ThermoWriter {
    /// Reopen after a restart: keep the header and the lines up to `last_step`, then append.
    pub fn open_append(path: &Path, every: u64, last_step: u64) -> Result<Self, ObserverError> {
        let text = std::fs::read_to_string(path).map_err(io)?;
        let mut kept = String::new();
        let mut terms = None;
        for line in text.lines() {
            if let Some(header) = line.strip_prefix("# ") {
                terms = Some(
                    header
                        .split('\t')
                        .filter_map(|c| {
                            c.strip_prefix("term:")
                                .and_then(|c| c.strip_suffix("_kjmol"))
                        })
                        .map(String::from)
                        .collect(),
                );
            } else if line
                .split('\t')
                .next()
                .and_then(|s| s.parse::<u64>().ok())
                .is_none_or(|step| step > last_step)
            {
                continue;
            }
            kept.push_str(line);
            kept.push('\n');
        }
        std::fs::write(path, kept).map_err(io)?;
        let file = std::fs::OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(io)?;
        Ok(Self {
            out: BufWriter::new(file),
            every: every.max(1),
            terms,
        })
    }

    /// Create (or truncate) `path`, writing every `every` steps.
    pub fn create(path: &Path, every: u64) -> Result<Self, ObserverError> {
        Ok(Self {
            out: BufWriter::new(File::create(path).map_err(io)?),
            every: every.max(1),
            terms: None,
        })
    }
}

impl Observer for ThermoWriter {
    fn every(&self) -> u64 {
        self.every
    }

    fn needs(&self) -> (bool, bool) {
        (true, true)
    }

    fn observe(&mut self, s: &Snapshot<'_>) -> Result<(), ObserverError> {
        if self.terms.is_none() {
            let names: Vec<String> = s
                .forces
                .energy_terms
                .iter()
                .map(|(n, _)| n.clone())
                .collect();
            let mut header = String::from(
                "# step\ttime_ps\tpotential_kjmol\tkinetic_kjmol\ttotal_kjmol\tconserved_energy_kjmol\ttemperature_K\tpressure_bar\tvolume_nm3",
            );
            for name in &names {
                header.push_str(&format!("\tterm:{name}_kjmol"));
            }
            writeln!(self.out, "{header}").map_err(io)?;
            self.terms = Some(names);
        }
        let potential = s.forces.energy;
        let pressure = s
            .pressure_bar
            .map(|p| (p.col(0)[0] + p.col(1)[1] + p.col(2)[2]) / 3.0);
        let mut line = format!(
            "{}\t{:.6}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{}",
            s.step,
            s.time_ps,
            num(potential),
            num(Some(s.kinetic_energy)),
            num(potential.map(|u| u + s.kinetic_energy)),
            num(s.conserved_energy),
            s.temperature_k,
            num(pressure),
            num(s.cell.map(ucmd_core::Cell::volume)),
        );
        for name in self.terms.iter().flatten() {
            let value = s
                .forces
                .energy_terms
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, v)| *v);
            line.push('\t');
            line.push_str(&num(value));
        }
        writeln!(self.out, "{line}").map_err(io)?;
        self.out.flush().map_err(io)
    }

    fn finish(&mut self) -> Result<(), ObserverError> {
        self.out.flush().map_err(io)
    }
}
