//! Thermo table: one tab-separated line per observed step, units in the header.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use ucmd_force::{Observer, ObserverError, Snapshot};

/// Writes `step, time, energies, temperature, pressure, volume, energy terms` as TSV, with the
/// column names of spec/run-config.md (`conserved_energy_kjmol` is the conserved quantity).
/// With `pressure_tensor` (spec 0.1.1) six more columns follow: the upper triangle of the
/// symmetrized pressure tensor ½(P + Pᵀ) in bar. With `pressure_molecular` (spec 0.1.3) the
/// molecular pressure in bar is the last column.
pub struct ThermoWriter {
    out: BufWriter<File>,
    every: u64,
    terms: Option<Vec<String>>,
    columns: OptionalColumns,
}

/// Columns that a run asks for explicitly (spec/run-config.md, `[output.thermo]`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OptionalColumns {
    /// Six components of the symmetrized pressure tensor, bar (since 0.1.1).
    pub pressure_tensor: bool,
    /// Molecular pressure, bar (since 0.1.3).
    pub pressure_molecular: bool,
}

/// Name of the molecular pressure column, in bar (spec/run-config.md, since 0.1.3).
pub const PRESSURE_MOLECULAR_COLUMN: &str = "pressure_molecular_bar";
/// Column names of the pressure tensor (spec/run-config.md, since 0.1.1).
pub const PRESSURE_TENSOR_COLUMNS: [&str; 6] = [
    "pressure_xx_bar",
    "pressure_yy_bar",
    "pressure_zz_bar",
    "pressure_xy_bar",
    "pressure_xz_bar",
    "pressure_yz_bar",
];
const PRESSURE_TENSOR_INDICES: [(usize, usize); 6] =
    [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];

fn io(e: std::io::Error) -> ObserverError {
    ObserverError(format!("thermo table: {e}"))
}

fn num(x: Option<f64>) -> String {
    x.map_or_else(|| "nan".to_string(), |v| format!("{v:.10e}"))
}

/// Scalar pressure of a pressure tensor (same unit).
fn trace_third(p: ucmd_core::Mat3) -> f64 {
    (p.col(0)[0] + p.col(1)[1] + p.col(2)[2]) / 3.0
}

impl ThermoWriter {
    /// Reopen after a restart: keep the header and the lines up to `last_step`, then append.
    pub fn open_append(
        path: &Path,
        every: u64,
        last_step: u64,
        columns: OptionalColumns,
    ) -> Result<Self, ObserverError> {
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
            columns,
        })
    }

    /// Create (or truncate) `path`, writing every `every` steps.
    pub fn create(
        path: &Path,
        every: u64,
        columns: OptionalColumns,
    ) -> Result<Self, ObserverError> {
        Ok(Self {
            out: BufWriter::new(File::create(path).map_err(io)?),
            every: every.max(1),
            terms: None,
            columns,
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
            if self.columns.pressure_tensor {
                for name in PRESSURE_TENSOR_COLUMNS {
                    header.push('\t');
                    header.push_str(name);
                }
            }
            if self.columns.pressure_molecular {
                header.push('\t');
                header.push_str(PRESSURE_MOLECULAR_COLUMN);
            }
            writeln!(self.out, "{header}").map_err(io)?;
            self.terms = Some(names);
        }
        let potential = s.forces.energy;
        let pressure = s.pressure_bar.map(trace_third);
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
        if self.columns.pressure_tensor {
            // requested explicitly: a missing virial is an error, not a column of NaN (invariant 7)
            let p = s.pressure_bar.ok_or_else(|| {
                ObserverError(format!(
                    "thermo table: pressure tensor requested but no virial at step {}",
                    s.step
                ))
            })?;
            for (a, b) in PRESSURE_TENSOR_INDICES {
                // p.col(b)[a] is row a, column b (as the H5MD writer)
                let value = 0.5 * (p.col(b)[a] + p.col(a)[b]);
                line.push('\t');
                line.push_str(&num(Some(value)));
            }
        }
        if self.columns.pressure_molecular {
            let p = s.molecular_pressure_bar.ok_or_else(|| {
                ObserverError(format!(
                    "thermo table: molecular pressure requested but no virial at step {}",
                    s.step
                ))
            })?;
            line.push('\t');
            line.push_str(&num(Some(trace_third(p))));
        }
        writeln!(self.out, "{line}").map_err(io)?;
        self.out.flush().map_err(io)
    }

    fn finish(&mut self) -> Result<(), ObserverError> {
        self.out.flush().map_err(io)
    }
}
