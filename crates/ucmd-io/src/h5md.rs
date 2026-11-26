//! H5MD 1.1 trajectory writer following spec/trajectory-h5md.md (ADR 0004).

use crate::checkpoint::RestartRecord;
use hdf5::types::VarLenUnicode;
use hdf5::{Dataset, File, Group};
use ndarray::{Array2, s};
use std::path::Path;
use ucmd_core::System;
use ucmd_force::{Observer, ObserverError, Snapshot};

/// What to write and how often.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H5mdOptions {
    /// Interval in steps.
    pub every: u64,
    /// Also write velocities.
    pub velocities: bool,
    /// Also write forces.
    pub forces: bool,
    /// Store positions, velocities and forces as float32 instead of float64.
    pub float32: bool,
}

/// Provenance stored in `/h5md` and `/parameters/ucmd`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct H5mdProvenance {
    /// Engine version.
    pub engine_version: String,
    /// Engine git hash ("unknown" outside git).
    pub git_hash: String,
    /// Author name from the machine profile.
    pub author: String,
    /// The package's system.json.
    pub system_json: String,
    /// The fully resolved run configuration.
    pub run_config_toml: String,
    /// SHA-256 of system.json.
    pub system_sha256: String,
    /// Backend, thread count and CPU model (ADR 0010); provenance only, never physics.
    pub hardware: String,
}

fn err(context: &str) -> impl Fn(hdf5::Error) -> ObserverError + '_ {
    move |e| ObserverError(format!("H5MD {context}: {e}"))
}

fn text(value: &str) -> Result<VarLenUnicode, ObserverError> {
    value
        .parse::<VarLenUnicode>()
        .map_err(|e| ObserverError(format!("H5MD text: {e}")))
}

fn attr_text(group: &Group, name: &str, value: &str) -> Result<(), ObserverError> {
    let value = text(value)?;
    group
        .new_attr::<VarLenUnicode>()
        .create(name)
        .and_then(|a| a.write_scalar(&value).map(|()| a))
        .map(|_| ())
        .map_err(err(name))
}

fn scalar_dataset(group: &Group, name: &str, value: &str) -> Result<(), ObserverError> {
    let value = text(value)?;
    group
        .new_dataset::<VarLenUnicode>()
        .create(name)
        .and_then(|d| d.write_scalar(&value).map(|()| d))
        .map(|_| ())
        .map_err(err(name))
}

/// Append one restart to `/parameters/ucmd/restarts` as group `NNN` (000, 001, …).
fn record_restart(file: &File, restart: &RestartRecord) -> Result<(), ObserverError> {
    let ucmd = file
        .group("parameters/ucmd")
        .map_err(err("parameters/ucmd"))?;
    let restarts = match ucmd.group("restarts") {
        Ok(group) => group,
        Err(_) => ucmd.create_group("restarts").map_err(err("restarts"))?,
    };
    let name = format!("{:03}", restarts.len());
    let g = restarts.create_group(&name).map_err(err("restart"))?;
    let step = i64::try_from(restart.step)
        .map_err(|_| ObserverError(format!("H5MD restart: step {} too large", restart.step)))?;
    g.new_attr::<i64>()
        .create("step")
        .and_then(|a| a.write_scalar(&step).map(|()| a))
        .map_err(err("step"))?;
    g.new_attr::<f64>()
        .create("time_ps")
        .and_then(|a| a.write_scalar(&restart.time_ps).map(|()| a))
        .map_err(err("time_ps"))?;
    attr_text(&g, "forced", if restart.forced { "true" } else { "false" })?;
    attr_text(&g, "differences", &restart.differences.join("\n"))?;
    attr_text(&g, "engine_version", &restart.engine_version)?;
    attr_text(&g, "engine_git_hash", &restart.engine_git_hash)?;
    attr_text(&g, "hardware", &restart.hardware)?;
    scalar_dataset(&g, "run_config_toml", &restart.run_config_toml)
}

/// The restarts recorded in a trajectory, oldest first (empty for a run never restarted).
pub fn read_restarts(path: &Path) -> Result<Vec<RestartRecord>, ObserverError> {
    let file = File::open(path).map_err(err("open"))?;
    let Ok(restarts) = file.group("parameters/ucmd/restarts") else {
        return Ok(Vec::new());
    };
    let mut names = restarts.member_names().map_err(err("restarts"))?;
    names.sort();
    let read_text = |g: &Group, name: &str| -> Result<String, ObserverError> {
        g.attr(name)
            .and_then(|a| a.read_scalar::<VarLenUnicode>())
            .map(|v| v.as_str().to_string())
            .map_err(err(name))
    };
    names
        .iter()
        .map(|name| {
            let g = restarts.group(name).map_err(err("restart"))?;
            let step: i64 = g
                .attr("step")
                .and_then(|a| a.read_scalar())
                .map_err(err("step"))?;
            let differences = read_text(&g, "differences")?;
            Ok(RestartRecord {
                step: u64::try_from(step)
                    .map_err(|_| ObserverError("H5MD: negative step".into()))?,
                time_ps: g
                    .attr("time_ps")
                    .and_then(|a| a.read_scalar())
                    .map_err(err("time_ps"))?,
                forced: read_text(&g, "forced")? == "true",
                differences: differences
                    .lines()
                    .map(str::to_string)
                    .filter(|l| !l.is_empty())
                    .collect(),
                engine_version: read_text(&g, "engine_version")?,
                engine_git_hash: read_text(&g, "engine_git_hash")?,
                hardware: read_text(&g, "hardware")?,
                run_config_toml: g
                    .dataset("run_config_toml")
                    .and_then(|d| d.read_scalar::<VarLenUnicode>())
                    .map(|v| v.as_str().to_string())
                    .map_err(err("run_config_toml"))?,
            })
        })
        .collect()
}

fn unit(dataset: &Dataset, value: &str) -> Result<(), ObserverError> {
    let value = text(value)?;
    dataset
        .new_attr::<VarLenUnicode>()
        .create("unit")
        .and_then(|a| a.write_scalar(&value).map(|()| a))
        .map(|_| ())
        .map_err(err("unit"))
}

/// One H5MD time-dependent element: `step`, `time`, `value`.
struct Series {
    step: Dataset,
    time: Dataset,
    value: Dataset,
    frame: Vec<usize>,
    float32: bool,
    count: usize,
}

impl Series {
    fn create(
        parent: &Group,
        name: &str,
        frame: &[usize],
        unit_name: &str,
        float32: bool,
    ) -> Result<Self, ObserverError> {
        let g = parent.create_group(name).map_err(err(name))?;
        let step = g
            .new_dataset::<i64>()
            .chunk(1024)
            .shape(0..)
            .create("step")
            .map_err(err(name))?;
        let time = g
            .new_dataset::<f64>()
            .chunk(1024)
            .shape(0..)
            .create("time")
            .map_err(err(name))?;
        unit(&time, "ps")?;
        let value = match (frame, float32) {
            ([], _) => g
                .new_dataset::<f64>()
                .chunk(1024)
                .shape(0..)
                .create("value"),
            ([a, b], false) => g
                .new_dataset::<f64>()
                .chunk((1, *a, *b))
                .shape((0.., *a, *b))
                .create("value"),
            ([a, b], true) => g
                .new_dataset::<f32>()
                .chunk((1, *a, *b))
                .shape((0.., *a, *b))
                .create("value"),
            _ => {
                return Err(ObserverError(format!(
                    "H5MD {name}: unsupported frame shape {frame:?}"
                )));
            }
        }
        .map_err(err(name))?;
        unit(&value, unit_name)?;
        Ok(Self {
            step,
            time,
            value,
            frame: frame.to_vec(),
            float32,
            count: 0,
        })
    }

    /// Reopen an existing element and cut it back to the frames with step <= `last_step`.
    fn open(parent: &Group, name: &str, last_step: u64) -> Result<Option<Self>, ObserverError> {
        if !parent.link_exists(name) {
            return Ok(None);
        }
        let g = parent.group(name).map_err(err(name))?;
        let step = g.dataset("step").map_err(err(name))?;
        let time = g.dataset("time").map_err(err(name))?;
        let value = g.dataset("value").map_err(err(name))?;
        let steps: Vec<i64> = step.read_raw().map_err(err(name))?;
        let count = steps
            .iter()
            .take_while(|&&k| u64::try_from(k).is_ok_and(|k| k <= last_step))
            .count();
        let shape = value.shape();
        let frame = shape[1..].to_vec();
        let float32 = value.dtype().map(|d| d.size() == 4).unwrap_or(false);
        step.resize(count).map_err(err(name))?;
        time.resize(count).map_err(err(name))?;
        match frame.as_slice() {
            [] => value.resize(count),
            [a, b] => value.resize((count, *a, *b)),
            _ => {
                return Err(ObserverError(format!(
                    "H5MD {name}: unsupported frame shape {frame:?}"
                )));
            }
        }
        .map_err(err(name))?;
        Ok(Some(Self {
            step,
            time,
            value,
            frame,
            float32,
            count,
        }))
    }

    fn append(&mut self, step: u64, time_ps: f64, data: &[f64]) -> Result<(), ObserverError> {
        let k = self.count;
        let context = "append";
        self.step.resize(k + 1).map_err(err(context))?;
        self.step
            .write_slice(&[step as i64], k..k + 1)
            .map_err(err(context))?;
        self.time.resize(k + 1).map_err(err(context))?;
        self.time
            .write_slice(&[time_ps], k..k + 1)
            .map_err(err(context))?;
        match self.frame.as_slice() {
            [] => {
                self.value.resize(k + 1).map_err(err(context))?;
                self.value
                    .write_slice(&data[..1], k..k + 1)
                    .map_err(err(context))?;
            }
            [a, b] => {
                self.value.resize((k + 1, *a, *b)).map_err(err(context))?;
                if self.float32 {
                    let frame =
                        Array2::from_shape_vec((*a, *b), data.iter().map(|&x| x as f32).collect())
                            .map_err(|e| ObserverError(e.to_string()))?;
                    self.value
                        .write_slice(&frame, s![k, .., ..])
                        .map_err(err(context))?;
                } else {
                    let frame = Array2::from_shape_vec((*a, *b), data.to_vec())
                        .map_err(|e| ObserverError(e.to_string()))?;
                    self.value
                        .write_slice(&frame, s![k, .., ..])
                        .map_err(err(context))?;
                }
            }
            _ => {}
        }
        self.count += 1;
        Ok(())
    }
}

/// Writes an H5MD trajectory as an [`Observer`].
pub struct H5mdWriter {
    file: File,
    options: H5mdOptions,
    n_atoms: usize,
    position: Series,
    velocity: Option<Series>,
    force: Option<Series>,
    edges: Series,
    observables: Group,
    scalars: Vec<(String, Series)>,
    pressure: Option<(Series, Series)>,
    terms: Vec<(String, Series)>,
    first: bool,
}

fn flatten(v: &[ucmd_core::Vec3]) -> Vec<f64> {
    v.iter().flat_map(|x| x.to_array()).collect()
}

impl H5mdWriter {
    /// Create (or truncate) `path` for `system`.
    pub fn create(
        path: &Path,
        system: &System,
        options: H5mdOptions,
        provenance: &H5mdProvenance,
    ) -> Result<Self, ObserverError> {
        let file = File::create(path).map_err(err("create"))?;
        let n = system.n_atoms();
        let h5md = file.create_group("h5md").map_err(err("h5md"))?;
        h5md.new_attr::<i32>()
            .shape(2)
            .create("version")
            .and_then(|a| a.write(&[1, 1]))
            .map_err(err("version"))?;
        attr_text(
            &h5md.create_group("author").map_err(err("author"))?,
            "name",
            &provenance.author,
        )?;
        let creator = h5md.create_group("creator").map_err(err("creator"))?;
        attr_text(&creator, "name", "ucmd")?;
        attr_text(&creator, "version", &provenance.engine_version)?;
        let modules = h5md.create_group("modules").map_err(err("modules"))?;
        modules
            .create_group("units")
            .and_then(|g| g.new_attr::<i32>().shape(2).create("version"))
            .and_then(|a| a.write(&[1, 0]))
            .map_err(err("units module"))?;

        let all = file
            .create_group("particles")
            .and_then(|p| p.create_group("all"))
            .map_err(err("particles"))?;
        let box_group = all.create_group("box").map_err(err("box"))?;
        box_group
            .new_attr::<i32>()
            .create("dimension")
            .and_then(|a| a.write_scalar(&3))
            .map_err(err("dimension"))?;
        let boundary: Vec<VarLenUnicode> = system
            .periodicity
            .iter()
            .map(|&p| text(if p { "periodic" } else { "none" }))
            .collect::<Result<_, _>>()?;
        box_group
            .new_attr::<VarLenUnicode>()
            .shape(3)
            .create("boundary")
            .and_then(|a| a.write(&boundary))
            .map_err(err("boundary"))?;
        let edges = Series::create(&box_group, "edges", &[3, 3], "nm", false)?;
        let position = Series::create(&all, "position", &[n, 3], "nm", options.float32)?;
        let velocity = if options.velocities {
            Some(Series::create(
                &all,
                "velocity",
                &[n, 3],
                "nm ps-1",
                options.float32,
            )?)
        } else {
            None
        };
        let force = if options.forces {
            Some(Series::create(
                &all,
                "force",
                &[n, 3],
                "kJ mol-1 nm-1",
                options.float32,
            )?)
        } else {
            None
        };
        let mass = all
            .new_dataset::<f64>()
            .shape(n)
            .create("mass")
            .map_err(err("mass"))?;
        mass.write(system.masses()).map_err(err("mass"))?;
        unit(&mass, "amu")?;
        if let Some(charges) = system.charges() {
            let charge = all
                .new_dataset::<f64>()
                .shape(n)
                .create("charge")
                .map_err(err("charge"))?;
            charge.write(charges).map_err(err("charge"))?;
            unit(&charge, "e")?;
        }
        let species: Vec<i32> = system.elements().iter().map(|&z| i32::from(z)).collect();
        all.new_dataset::<i32>()
            .shape(n)
            .create("species")
            .and_then(|d| d.write(&species))
            .map_err(err("species"))?;
        let molecules: Vec<i64> = system
            .molecule_index()
            .iter()
            .map(|&m| i64::from(m))
            .collect();
        all.new_dataset::<i64>()
            .shape(n)
            .create("molecule_index")
            .and_then(|d| d.write(&molecules))
            .map_err(err("molecule_index"))?;

        let observables = file
            .create_group("observables")
            .map_err(err("observables"))?;
        let mut scalars = Vec::new();
        for (name, unit_name) in [
            ("potential_energy", "kJ mol-1"),
            ("kinetic_energy", "kJ mol-1"),
            ("conserved_energy", "kJ mol-1"),
            ("temperature", "K"),
        ] {
            scalars.push((
                name.to_string(),
                Series::create(&observables, name, &[], unit_name, false)?,
            ));
        }
        if system.is_periodic() {
            scalars.push((
                "volume".into(),
                Series::create(&observables, "volume", &[], "nm3", false)?,
            ));
        }

        let ucmd = file
            .create_group("parameters")
            .and_then(|p| p.create_group("ucmd"))
            .map_err(err("parameters"))?;
        attr_text(&ucmd, "spec_version", "0.1.0")?;
        attr_text(&ucmd, "engine_version", &provenance.engine_version)?;
        attr_text(&ucmd, "engine_git_hash", &provenance.git_hash)?;
        attr_text(&ucmd, "system_sha256", &provenance.system_sha256)?;
        attr_text(&ucmd, "hardware", &provenance.hardware)?;
        attr_text(&ucmd, "positions", "unwrapped")?;
        for (name, value) in [
            ("system_json", &provenance.system_json),
            ("run_config_toml", &provenance.run_config_toml),
        ] {
            let value = text(value)?;
            ucmd.new_dataset::<VarLenUnicode>()
                .create(name)
                .and_then(|d| d.write_scalar(&value).map(|()| d))
                .map_err(err(name))?;
        }
        Ok(Self {
            file,
            options,
            n_atoms: n,
            position,
            velocity,
            force,
            edges,
            observables,
            scalars,
            pressure: None,
            terms: Vec::new(),
            first: true,
        })
    }
}

impl H5mdWriter {
    /// Reopen after a restart: every element is cut back to the frames up to the restart step
    /// and writing continues from there (spec/checkpoint.md). The restart is recorded in
    /// `/parameters/ucmd/restarts`.
    pub fn open_append(
        path: &Path,
        system: &System,
        options: H5mdOptions,
        restart: &RestartRecord,
    ) -> Result<Self, ObserverError> {
        let last_step = restart.step;
        let missing = |what: &str| {
            ObserverError(format!(
                "H5MD restart: {what} missing in {}",
                path.display()
            ))
        };
        let file = File::open_rw(path).map_err(err("open"))?;
        record_restart(&file, restart)?;
        let all = file.group("particles/all").map_err(err("particles/all"))?;
        let box_group = all.group("box").map_err(err("box"))?;
        let edges =
            Series::open(&box_group, "edges", last_step)?.ok_or_else(|| missing("box/edges"))?;
        let position =
            Series::open(&all, "position", last_step)?.ok_or_else(|| missing("position"))?;
        let velocity = if options.velocities {
            Some(Series::open(&all, "velocity", last_step)?.ok_or_else(|| missing("velocity"))?)
        } else {
            None
        };
        let force = if options.forces {
            Some(Series::open(&all, "force", last_step)?.ok_or_else(|| missing("force"))?)
        } else {
            None
        };
        let observables = file.group("observables").map_err(err("observables"))?;
        let mut scalars = Vec::new();
        for name in [
            "potential_energy",
            "kinetic_energy",
            "conserved_energy",
            "temperature",
            "volume",
        ] {
            if let Some(series) = Series::open(&observables, name, last_step)? {
                scalars.push((name.to_string(), series));
            }
        }
        let pressure = match (
            Series::open(&observables, "pressure", last_step)?,
            Series::open(&observables, "pressure_tensor", last_step)?,
        ) {
            (Some(p), Some(t)) => Some((p, t)),
            _ => None,
        };
        let mut terms = Vec::new();
        if observables.link_exists("energy_terms") {
            let group = observables
                .group("energy_terms")
                .map_err(err("energy_terms"))?;
            for member in group.member_names().map_err(err("energy_terms"))? {
                if let Some(series) = Series::open(&group, &member, last_step)? {
                    terms.push((member.replace(':', "/"), series));
                }
            }
        }
        Ok(Self {
            file,
            options,
            n_atoms: system.n_atoms(),
            position,
            velocity,
            force,
            edges,
            observables,
            scalars,
            pressure,
            terms,
            first: false,
        })
    }
}

impl Observer for H5mdWriter {
    fn every(&self) -> u64 {
        self.options.every.max(1)
    }

    fn needs(&self) -> (bool, bool) {
        (true, true)
    }

    fn observe(&mut self, s: &Snapshot<'_>) -> Result<(), ObserverError> {
        if s.positions.len() != self.n_atoms {
            return Err(ObserverError("H5MD: atom count changed".into()));
        }
        if self.first {
            self.first = false;
            if s.pressure_bar.is_some() {
                self.pressure = Some((
                    Series::create(&self.observables, "pressure", &[], "bar", false)?,
                    Series::create(&self.observables, "pressure_tensor", &[3, 3], "bar", false)?,
                ));
            }
            if !s.forces.energy_terms.is_empty() {
                let group = self
                    .observables
                    .create_group("energy_terms")
                    .map_err(err("energy_terms"))?;
                for (name, _) in &s.forces.energy_terms {
                    let safe = name.replace('/', ":");
                    self.terms.push((
                        name.clone(),
                        Series::create(&group, &safe, &[], "kJ mol-1", false)?,
                    ));
                }
            }
        }
        let (step, t) = (s.step, s.time_ps);
        self.position.append(step, t, &flatten(s.positions))?;
        if let Some(series) = self.velocity.as_mut() {
            series.append(step, t, &flatten(s.velocities))?;
        }
        if let Some(series) = self.force.as_mut() {
            series.append(step, t, &flatten(&s.forces.forces))?;
        }
        let edges: Vec<f64> = s.cell.map_or_else(
            || vec![0.0; 9],
            |c| c.rows().iter().flat_map(|v| v.to_array()).collect(),
        );
        self.edges.append(step, t, &edges)?;
        let nan = f64::NAN;
        for (name, series) in &mut self.scalars {
            let value = match name.as_str() {
                "potential_energy" => s.forces.energy.unwrap_or(nan),
                "kinetic_energy" => s.kinetic_energy,
                "conserved_energy" => s.conserved_energy.unwrap_or(nan),
                "temperature" => s.temperature_k,
                "volume" => s.cell.map_or(nan, ucmd_core::Cell::volume),
                _ => nan,
            };
            series.append(step, t, &[value])?;
        }
        if let (Some((scalar, tensor)), Some(p)) = (self.pressure.as_mut(), s.pressure_bar) {
            scalar.append(step, t, &[(p.col(0)[0] + p.col(1)[1] + p.col(2)[2]) / 3.0])?;
            let rows: Vec<f64> = (0..3)
                .flat_map(|a| (0..3).map(move |b| p.col(b)[a]))
                .collect();
            tensor.append(step, t, &rows)?;
        }
        for (name, series) in &mut self.terms {
            let value = s
                .forces
                .energy_terms
                .iter()
                .find(|(n, _)| n == name)
                .map_or(nan, |(_, v)| *v);
            series.append(step, t, &[value])?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), ObserverError> {
        self.file.flush().map_err(err("flush"))
    }
}

/// One frame of an H5MD trajectory.
#[derive(Debug, Clone, PartialEq)]
pub struct H5mdFrame {
    /// MD step.
    pub step: u64,
    /// Time in ps.
    pub time_ps: f64,
    /// Positions in nm (as stored: unwrapped for ucmd trajectories).
    pub positions: Vec<ucmd_core::Vec3>,
    /// Cell rows in nm (zeros for non-periodic systems).
    pub cell_rows: [[f64; 3]; 3],
}

/// Frame-by-frame reader of the position and box elements of an H5MD file (spec profile).
pub struct H5mdReader {
    steps: Vec<i64>,
    times: Vec<f64>,
    value: Dataset,
    edges: Dataset,
    n_atoms: usize,
}

impl H5mdReader {
    /// Open `path` for reading.
    pub fn open(path: &Path) -> Result<Self, ObserverError> {
        let file = File::open(path).map_err(err("open"))?;
        let position = file
            .group("particles/all/position")
            .map_err(err("position"))?;
        let steps: Vec<i64> = position
            .dataset("step")
            .and_then(|d| d.read_raw())
            .map_err(err("step"))?;
        let times: Vec<f64> = position
            .dataset("time")
            .and_then(|d| d.read_raw())
            .map_err(err("time"))?;
        let value = position.dataset("value").map_err(err("value"))?;
        let edges = file
            .dataset("particles/all/box/edges/value")
            .map_err(err("box/edges"))?;
        let shape = value.shape();
        if shape.len() != 3 || shape[0] != steps.len() || shape[2] != 3 {
            return Err(ObserverError(format!(
                "H5MD {}: unexpected position shape {shape:?}",
                path.display()
            )));
        }
        Ok(Self {
            steps,
            times,
            value,
            edges,
            n_atoms: shape[1],
        })
    }

    /// Number of frames.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// True without frames.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    /// Frame `k` (positions converted to f64 if stored as float32).
    pub fn frame(&self, k: usize) -> Result<H5mdFrame, ObserverError> {
        let raw: ndarray::Array2<f64> = self
            .value
            .read_slice_2d(s![k, .., ..])
            .map_err(err("read position"))?;
        let positions = raw
            .rows()
            .into_iter()
            .map(|r| ucmd_core::Vec3::new(r[0], r[1], r[2]))
            .collect();
        let box_raw: ndarray::Array2<f64> = self
            .edges
            .read_slice_2d(s![k, .., ..])
            .map_err(err("read box"))?;
        let cell_rows = std::array::from_fn(|a| std::array::from_fn(|b| box_raw[[a, b]]));
        Ok(H5mdFrame {
            step: u64::try_from(self.steps[k]).unwrap_or(0),
            time_ps: self.times[k],
            positions,
            cell_rows,
        })
    }
}
