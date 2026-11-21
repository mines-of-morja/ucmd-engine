//! Out-of-process force providers over the i-PI socket protocol.
//!
//! The engine acts as the i-PI server.  A client (for example ASE or a QM code) connects over
//! either a Unix domain socket or a TCP socket and supplies positions, then receives forces.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;
use ucmd_core::units::{KJMOL_PER_HARTREE, NM_PER_BOHR};
use ucmd_core::{Cell, Mat3, System, Vec3};
use ucmd_force::{
    Capabilities, EvalRequest, ForceError, ForceProvider, ForceResult, Observable, ObservableValue,
};

const HEADER_BYTES: usize = 12;
const DEFAULT_TIMEOUT_MS: u64 = 30_000;

fn protocol(detail: impl Into<String>) -> ForceError {
    ForceError::Protocol(detail.into())
}

fn read_exact(stream: &mut IpiStream, bytes: &mut [u8]) -> Result<(), ForceError> {
    stream.read_exact(bytes).map_err(|e| {
        if matches!(
            e.kind(),
            std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
        ) {
            protocol(format!("timed out waiting for i-PI client: {e}"))
        } else if matches!(e.kind(), std::io::ErrorKind::UnexpectedEof) {
            protocol("i-PI client disconnected")
        } else {
            protocol(format!("failed to read i-PI socket: {e}"))
        }
    })
}

fn write_all(stream: &mut IpiStream, bytes: &[u8]) -> Result<(), ForceError> {
    stream
        .write_all(bytes)
        .map_err(|e| protocol(format!("failed to write i-PI socket: {e}")))
}

fn header(text: &str) -> [u8; HEADER_BYTES] {
    let mut out = [0u8; HEADER_BYTES];
    let bytes = text.as_bytes();
    let count = bytes.len().min(HEADER_BYTES);
    out[..count].copy_from_slice(&bytes[..count]);
    out
}

fn read_header(stream: &mut IpiStream) -> Result<[u8; HEADER_BYTES], ForceError> {
    let mut bytes = [0u8; HEADER_BYTES];
    read_exact(stream, &mut bytes)?;
    Ok(bytes)
}

fn header_text(bytes: &[u8; HEADER_BYTES]) -> &str {
    std::str::from_utf8(bytes)
        .unwrap_or("")
        .trim_matches(char::from(0))
        .trim()
}

fn read_f64(stream: &mut IpiStream) -> Result<f64, ForceError> {
    let mut bytes = [0u8; 8];
    read_exact(stream, &mut bytes)?;
    Ok(f64::from_ne_bytes(bytes))
}

fn read_i32(stream: &mut IpiStream) -> Result<i32, ForceError> {
    let mut bytes = [0u8; 4];
    read_exact(stream, &mut bytes)?;
    Ok(i32::from_ne_bytes(bytes))
}

fn send_f64(stream: &mut IpiStream, value: f64) -> Result<(), ForceError> {
    write_all(stream, &value.to_ne_bytes())
}

fn send_i32(stream: &mut IpiStream, value: i32) -> Result<(), ForceError> {
    write_all(stream, &value.to_ne_bytes())
}

enum IpiListener {
    /// A Unix-domain socket listener.
    Unix(UnixListener),
    /// A TCP listener.
    Tcp(TcpListener),
}

enum IpiStream {
    /// A Unix-domain client stream.
    Unix(UnixStream),
    /// A TCP client stream.
    Tcp(TcpStream),
}

impl Read for IpiStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::Unix(stream) => stream.read(buf),
            Self::Tcp(stream) => stream.read(buf),
        }
    }
}

impl Write for IpiStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Self::Unix(stream) => stream.write(buf),
            Self::Tcp(stream) => stream.write(buf),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Self::Unix(stream) => stream.flush(),
            Self::Tcp(stream) => stream.flush(),
        }
    }
}

impl IpiStream {
    fn set_timeout(&mut self, timeout: Duration) -> Result<(), ForceError> {
        let result = match self {
            Self::Unix(stream) => stream
                .set_read_timeout(Some(timeout))
                .and_then(|_| stream.set_write_timeout(Some(timeout))),
            Self::Tcp(stream) => stream
                .set_read_timeout(Some(timeout))
                .and_then(|_| stream.set_write_timeout(Some(timeout))),
        };
        result.map_err(|e| protocol(format!("failed to set i-PI socket timeout: {e}")))
    }

    fn shutdown(&mut self) -> Result<(), ForceError> {
        let result = match self {
            Self::Unix(stream) => stream.shutdown(std::net::Shutdown::Both),
            Self::Tcp(stream) => stream.shutdown(std::net::Shutdown::Both),
        };
        result.map_err(|e| protocol(format!("failed to close i-PI socket: {e}")))
    }
}

/// A socket address accepted by [`IpiProvider`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IpiAddress {
    /// Unix-domain socket path.
    Unix(PathBuf),
    /// TCP host and port.
    Tcp(String, u16),
}

impl IpiAddress {
    /// Build a Unix-domain address.
    pub fn unix(path: impl Into<PathBuf>) -> Self {
        Self::Unix(path.into())
    }

    /// Build a TCP address.
    pub fn tcp(host: impl Into<String>, port: u16) -> Self {
        Self::Tcp(host.into(), port)
    }
}

/// Configuration for an i-PI server provider.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IpiProviderConfig {
    /// Unix socket path, mutually exclusive with host/port.
    #[serde(default)]
    pub socket: Option<PathBuf>,
    /// TCP host name or address.
    #[serde(default)]
    pub host: Option<String>,
    /// TCP port.
    #[serde(default)]
    pub port: Option<u16>,
    /// Read/write timeout in milliseconds.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

const fn default_timeout_ms() -> u64 {
    DEFAULT_TIMEOUT_MS
}

impl IpiProviderConfig {
    /// Create a Unix-socket configuration.
    pub fn unix(path: impl Into<PathBuf>) -> Self {
        Self {
            socket: Some(path.into()),
            host: None,
            port: None,
            timeout_ms: DEFAULT_TIMEOUT_MS,
        }
    }

    /// Create a TCP configuration.
    pub fn tcp(host: impl Into<String>, port: u16) -> Self {
        Self {
            socket: None,
            host: Some(host.into()),
            port: Some(port),
            timeout_ms: DEFAULT_TIMEOUT_MS,
        }
    }

    fn address(&self) -> Result<IpiAddress, ForceError> {
        if let Some(socket) = &self.socket {
            if self.host.is_some() || self.port.is_some() {
                return Err(ForceError::InvalidInput(
                    "i-PI socket cannot be combined with host or port".into(),
                ));
            }
            Ok(IpiAddress::Unix(socket.clone()))
        } else {
            let host = self.host.as_deref().unwrap_or("127.0.0.1");
            let port = self
                .port
                .ok_or_else(|| ForceError::InvalidInput("i-PI TCP provider needs port".into()))?;
            Ok(IpiAddress::Tcp(host.to_string(), port))
        }
    }
}

impl Default for IpiProviderConfig {
    fn default() -> Self {
        Self {
            socket: None,
            host: None,
            port: None,
            timeout_ms: DEFAULT_TIMEOUT_MS,
        }
    }
}

/// Alias for [`IpiProviderConfig`].
pub type IpiConfig = IpiProviderConfig;

/// i-PI server force provider.
pub struct IpiProvider {
    config: IpiProviderConfig,
    listener: Option<IpiListener>,
    stream: Option<IpiStream>,
    atom_count: Option<usize>,
    name: String,
}

impl IpiProvider {
    /// Create a provider from an i-PI configuration.
    pub fn new(config: IpiProviderConfig) -> Self {
        let name = "i-PI".to_string();
        Self {
            config,
            listener: None,
            stream: None,
            atom_count: None,
            name,
        }
    }

    /// Create a provider for a Unix-domain socket.
    pub fn unix(path: impl Into<PathBuf>) -> Self {
        Self::new(IpiProviderConfig::unix(path))
    }

    /// Create a provider for a TCP host and port.
    pub fn tcp(host: impl Into<String>, port: u16) -> Self {
        Self::new(IpiProviderConfig::tcp(host, port))
    }

    /// Set the socket read/write timeout in milliseconds.
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.config.timeout_ms = timeout_ms;
        self
    }

    fn bind(&mut self) -> Result<(), ForceError> {
        let address = self.config.address()?;
        let listener = match address {
            IpiAddress::Unix(path) => {
                remove_stale_socket(&path)?;
                IpiListener::Unix(UnixListener::bind(&path).map_err(|e| {
                    protocol(format!(
                        "failed to bind Unix socket {}: {e}",
                        path.display()
                    ))
                })?)
            }
            IpiAddress::Tcp(host, port) => {
                let addrs = format!("{host}:{port}").to_socket_addrs().map_err(|e| {
                    protocol(format!("failed to resolve i-PI address {host}:{port}: {e}"))
                })?;
                let mut last = None;
                let mut bound = None;
                for addr in addrs {
                    match TcpListener::bind(addr) {
                        Ok(listener) => {
                            bound = Some(listener);
                            break;
                        }
                        Err(e) => last = Some(e),
                    }
                }
                IpiListener::Tcp(bound.ok_or_else(|| {
                    protocol(format!(
                        "failed to bind i-PI TCP address {host}:{port}: {}",
                        last.map_or_else(|| "no addresses".into(), |e| e.to_string())
                    ))
                })?)
            }
        };
        self.listener = Some(listener);
        Ok(())
    }

    fn accept(&mut self) -> Result<(), ForceError> {
        let timeout = Duration::from_millis(self.config.timeout_ms);
        let mut stream = match self.listener.as_ref() {
            Some(listener) => listener.accept(),
            None => return Err(protocol("i-PI listener is not initialized")),
        }?;
        stream.set_timeout(timeout)?;
        let incoming = read_header(&mut stream)?;
        match header_text(&incoming).to_ascii_uppercase().as_str() {
            "STATUS" | "HELLO" | "READY" => {
                write_all(&mut stream, &header("READY"))?;
            }
            other => {
                return Err(protocol(format!(
                    "unexpected i-PI handshake message {other:?}"
                )));
            }
        }
        self.stream = Some(stream);
        Ok(())
    }
}

impl IpiListener {
    fn accept(&self) -> Result<IpiStream, ForceError> {
        match self {
            Self::Unix(listener) => listener
                .accept()
                .map(|(stream, _)| IpiStream::Unix(stream))
                .map_err(|e| protocol(format!("failed to accept i-PI Unix client: {e}"))),
            Self::Tcp(listener) => listener
                .accept()
                .map(|(stream, _)| IpiStream::Tcp(stream))
                .map_err(|e| protocol(format!("failed to accept i-PI TCP client: {e}"))),
        }
    }
}

/// Convert nanometres to i-PI bohr.
pub fn nm_to_bohr(value_nm: f64) -> f64 {
    value_nm / NM_PER_BOHR
}

/// Convert i-PI bohr to nanometres.
pub fn bohr_to_nm(value_bohr: f64) -> f64 {
    value_bohr * NM_PER_BOHR
}

/// Convert i-PI energy in hartree to engine kJ/mol.
pub fn hartree_to_kjmol(value_hartree: f64) -> f64 {
    value_hartree * KJMOL_PER_HARTREE
}

/// Convert engine force in kJ mol⁻¹ nm⁻¹ to i-PI hartree/bohr.
pub fn force_nm_to_hartree_bohr(value: f64) -> f64 {
    value * NM_PER_BOHR / KJMOL_PER_HARTREE
}

/// Convert i-PI force in hartree/bohr to engine kJ mol⁻¹ nm⁻¹.
pub fn force_hartree_bohr_to_nm(value: f64) -> f64 {
    value * KJMOL_PER_HARTREE / NM_PER_BOHR
}

fn remove_stale_socket(path: &Path) -> Result<(), ForceError> {
    if path.exists() {
        std::fs::remove_file(path).map_err(|e| {
            protocol(format!(
                "failed to remove stale Unix socket {}: {e}",
                path.display()
            ))
        })?;
    }
    Ok(())
}

/// Alias for [`IpiProvider`].
pub type IpiServer = IpiProvider;

fn mat3_row_major(values: Mat3) -> [f64; 9] {
    [
        values.col(0)[0],
        values.col(1)[0],
        values.col(2)[0],
        values.col(0)[1],
        values.col(1)[1],
        values.col(2)[1],
        values.col(0)[2],
        values.col(1)[2],
        values.col(2)[2],
    ]
}

fn mat3_from_row_major(values: [f64; 9]) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(values[0], values[3], values[6]),
        Vec3::new(values[1], values[4], values[7]),
        Vec3::new(values[2], values[5], values[8]),
    )
}

fn mat3_from_rows(cell: &Cell) -> Mat3 {
    let rows = cell.rows();
    Mat3::from_cols(
        Vec3::new(rows[0].x, rows[1].x, rows[2].x),
        Vec3::new(rows[0].y, rows[1].y, rows[2].y),
        Vec3::new(rows[0].z, rows[1].z, rows[2].z),
    )
}

fn parse_extras(bytes: &[u8]) -> Vec<Observable> {
    let mut bytes = bytes;
    if bytes.len() >= 4 {
        let length = i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if length > 0 && usize::try_from(length).is_ok_and(|n| n <= bytes.len() - 4) {
            bytes = &bytes[4..4 + usize::try_from(length).unwrap_or(0)];
        }
    }
    let text = std::str::from_utf8(bytes)
        .unwrap_or("")
        .trim_matches(char::from(0))
        .trim();
    if text.is_empty() {
        return Vec::new();
    }
    let value = match serde_json::from_str::<serde_json::Value>(text) {
        Ok(value) => value,
        Err(_) => return Vec::new(),
    };
    let object = match value.as_object() {
        Some(object) => object,
        None => return Vec::new(),
    };
    object
        .iter()
        .filter_map(|(name, value)| match value {
            serde_json::Value::Number(number) => number.as_f64().map(|value| Observable {
                name: name.clone(),
                unit: String::new(),
                value: ObservableValue::Scalar(value),
            }),
            serde_json::Value::Array(array) => {
                let values: Option<Vec<f64>> = array.iter().map(|value| value.as_f64()).collect();
                values.map(|value| Observable {
                    name: name.clone(),
                    unit: String::new(),
                    value: ObservableValue::Array(value),
                })
            }
            _ => None,
        })
        .collect()
}

impl ForceProvider for IpiProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            periodic: true,
            virial: true,
            energy_terms: false,
            deterministic: false,
        }
    }

    fn initialize(&mut self, system: &System) -> Result<(), ForceError> {
        self.atom_count = Some(system.n_atoms());
        self.bind()?;
        self.accept()
    }

    fn evaluate(
        &mut self,
        positions: &[Vec3],
        cell: Option<&Cell>,
        request: &EvalRequest,
        out: &mut ForceResult,
    ) -> Result<(), ForceError> {
        let atom_count = self.atom_count.unwrap_or(positions.len());
        if positions.len() != atom_count {
            return Err(ForceError::InvalidInput(format!(
                "i-PI provider expected {atom_count} atoms, got {}",
                positions.len()
            )));
        }
        let mut stream = self
            .stream
            .take()
            .ok_or_else(|| protocol("i-PI client is not connected"))?;
        write_all(&mut stream, &header("POSDATA"))?;

        let h = cell.map_or(Mat3::IDENTITY, mat3_from_rows);
        let inverse = h.inverse();
        for value in mat3_row_major(h) {
            send_f64(&mut stream, nm_to_bohr(value))?;
        }
        for value in mat3_row_major(inverse) {
            send_f64(&mut stream, value * NM_PER_BOHR)?;
        }
        send_i32(
            &mut stream,
            i32::try_from(atom_count)
                .map_err(|_| ForceError::InvalidInput("i-PI atom count does not fit i32".into()))?,
        )?;
        for position in positions {
            send_f64(&mut stream, nm_to_bohr(position.x))?;
            send_f64(&mut stream, nm_to_bohr(position.y))?;
            send_f64(&mut stream, nm_to_bohr(position.z))?;
        }

        loop {
            let incoming = read_header(&mut stream)?;
            match header_text(&incoming).to_ascii_uppercase().as_str() {
                "STATUS" => {
                    write_all(&mut stream, &header("READY"))?;
                }
                "GETFORCE" | "FORCES" => {
                    let energy = read_f64(&mut stream)?;
                    let returned_atoms = read_i32(&mut stream)?;
                    let returned_atoms = usize::try_from(returned_atoms)
                        .map_err(|_| protocol("i-PI returned a negative atom count"))?;
                    if returned_atoms != atom_count {
                        return Err(protocol(format!(
                            "i-PI returned {returned_atoms} forces for {atom_count} atoms"
                        )));
                    }
                    out.reset(atom_count);
                    for force in &mut out.forces {
                        let x = read_f64(&mut stream)?;
                        let y = read_f64(&mut stream)?;
                        let z = read_f64(&mut stream)?;
                        *force = Vec3::new(
                            force_hartree_bohr_to_nm(x),
                            force_hartree_bohr_to_nm(y),
                            force_hartree_bohr_to_nm(z),
                        );
                    }
                    let mut virial = [0.0f64; 9];
                    for value in &mut virial {
                        *value = read_f64(&mut stream)?;
                    }
                    let mut extras = [0u8; 1024];
                    read_exact(&mut stream, &mut extras)?;
                    out.energy = if request.need_energy {
                        Some(hartree_to_kjmol(energy))
                    } else {
                        None
                    };
                    if request.need_virial {
                        let virial = mat3_from_row_major(virial.map(hartree_to_kjmol));
                        out.virial = Some(virial);
                    }
                    out.observables = parse_extras(&extras);
                    self.stream = Some(stream);
                    return Ok(());
                }
                "POSDATA" => {
                    let mut ignored = [0u8; 9 * 8 + 9 * 8];
                    read_exact(&mut stream, &mut ignored)?;
                    let returned = read_i32(&mut stream)?;
                    let count = usize::try_from(returned)
                        .map_err(|_| protocol("negative i-PI atom count"))?;
                    let mut positions = vec![0u8; count * 3 * 8];
                    read_exact(&mut stream, &mut positions)?;
                }
                "EXIT" => {
                    return Err(protocol("i-PI client sent EXIT"));
                }
                other => {
                    return Err(protocol(format!("unexpected i-PI message {other:?}")));
                }
            }
        }
    }

    fn finalize(&mut self) -> Result<(), ForceError> {
        if let Some(mut stream) = self.stream.take() {
            let _ = write_all(&mut stream, &header("EXIT"));
            let _ = stream.shutdown();
        }
        if let IpiAddress::Unix(path) = self.config.address()? {
            let _ = std::fs::remove_file(path);
        }
        self.listener = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucmd_core::{AtomTemplate, MoleculeType};

    fn system() -> System {
        let atom = AtomTemplate {
            name: "X".into(),
            element: 0,
            mass_amu: 1.0,
            charge_e: None,
            atom_type: None,
        };
        let molecule = MoleculeType {
            name: "X".into(),
            formal_charge_e: 0,
            atoms: vec![atom],
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
            impropers: Vec::new(),
            constraints: Vec::new(),
        };
        System::new("ipi-test", vec![molecule], &[(0, 1)], None, [false; 3]).unwrap()
    }

    #[test]
    fn unit_conversions_use_core_constants() {
        let bohr = nm_to_bohr(NM_PER_BOHR);
        assert!((bohr - 1.0).abs() < 1.0e-15);
        assert!((bohr_to_nm(1.0) - NM_PER_BOHR).abs() < 1.0e-15);
        assert!((hartree_to_kjmol(1.0) - KJMOL_PER_HARTREE).abs() < 1.0e-12);
    }

    #[test]
    fn json_extras_become_observables() {
        let text = br#"{"dipole":[1.0,2.0],"spin":3.0}"#;
        let mut extras = [0u8; 1024];
        extras[..text.len()].copy_from_slice(text);
        let observables = parse_extras(&extras);
        assert_eq!(observables.len(), 2);
        assert!(observables.iter().any(|value| {
            value.name == "spin"
                && value.value == ObservableValue::Scalar(3.0)
                && value.unit.is_empty()
        }));
        assert!(observables.iter().any(|value| {
            value.name == "dipole" && value.value == ObservableValue::Array(vec![1.0, 2.0])
        }));
    }

    #[test]
    fn unix_socket_round_trip() {
        let path = std::env::temp_dir().join(format!("ucmd-ipi-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let client_path = path.clone();
        let client = std::thread::spawn(move || -> Result<(), String> {
            let mut stream = None;
            for _ in 0..100 {
                match UnixStream::connect(&client_path) {
                    Ok(value) => {
                        stream = Some(value);
                        break;
                    }
                    Err(_) => std::thread::sleep(Duration::from_millis(10)),
                }
            }
            let mut stream =
                stream.ok_or_else(|| "could not connect to i-PI Unix socket".to_string())?;
            stream
                .write_all(&header("STATUS"))
                .map_err(|e| e.to_string())?;
            let mut ready = [0u8; HEADER_BYTES];
            stream.read_exact(&mut ready).map_err(|e| e.to_string())?;
            if header_text(&ready) != "READY" {
                return Err("missing READY handshake".into());
            }

            let mut incoming = [0u8; HEADER_BYTES];
            stream
                .read_exact(&mut incoming)
                .map_err(|e| e.to_string())?;
            if header_text(&incoming) != "POSDATA" {
                return Err("missing POSDATA".into());
            }
            let mut packet = vec![0u8; 9 * 8 + 9 * 8 + 4 + 3 * 8];
            stream.read_exact(&mut packet).map_err(|e| e.to_string())?;

            stream
                .write_all(&header("GETFORCE"))
                .map_err(|e| e.to_string())?;
            stream
                .write_all(&1.0f64.to_ne_bytes())
                .map_err(|e| e.to_string())?;
            stream
                .write_all(&1i32.to_ne_bytes())
                .map_err(|e| e.to_string())?;
            stream
                .write_all(&0.25f64.to_ne_bytes())
                .map_err(|e| e.to_string())?;
            stream
                .write_all(&0.5f64.to_ne_bytes())
                .map_err(|e| e.to_string())?;
            stream
                .write_all(&(-0.25f64).to_ne_bytes())
                .map_err(|e| e.to_string())?;
            let virial = [0u8; 9 * 8];
            stream.write_all(&virial).map_err(|e| e.to_string())?;
            let mut extras = [0u8; 1024];
            let text = br#"{"test":7.0}"#;
            extras[..text.len()].copy_from_slice(text);
            stream.write_all(&extras).map_err(|e| e.to_string())?;
            Ok(())
        });

        let system = system();
        let mut provider = IpiProvider::unix(path.clone()).with_timeout_ms(2_000);
        provider.initialize(&system).unwrap();
        let mut out = ForceResult::default();
        provider
            .evaluate(
                &[Vec3::new(0.1, 0.2, 0.3)],
                None,
                &EvalRequest::everything(),
                &mut out,
            )
            .unwrap();
        client.join().unwrap().unwrap();
        assert_eq!(out.forces.len(), 1);
        assert!(out.energy.is_some());
        assert!(out.virial.is_some());
        assert_eq!(out.observables.len(), 1);
        provider.finalize().unwrap();
        let _ = std::fs::remove_file(path);
    }
}
