//! The hardware a run executes on, recorded for provenance (ADR 0010).
//!
//! Hardware is never part of the physics: it does not enter the configuration fingerprint, so
//! a checkpoint written on one machine continues on any other. The reference CPU backend gives
//! bit-identical results everywhere; this record makes every result traceable anyway.

/// The backend that evaluates the built-in providers.
pub const BACKEND: &str = "cpu-reference";

/// The CPU model from /proc/cpuinfo on Linux, else the architecture name.
pub fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split_once(':'))
                .map(|(_, model)| model.trim().to_string())
        })
        .unwrap_or_else(|| std::env::consts::ARCH.to_string())
}

/// One line for logs and provenance, e.g. "cpu-reference, 1 thread, AMD Ryzen 7 …".
pub fn describe(threads: usize) -> String {
    let unit = if threads == 1 { "thread" } else { "threads" };
    format!("{BACKEND}, {threads} {unit}, {}", cpu_model())
}

/// Start the worker threads of the built-in providers: `threads` from the machine profile, else
/// every logical core. Returns the number in use. The count changes only the speed, never the
/// results (ADR 0010); a second call in the same process keeps the first pool.
pub fn start_threads(threads: Option<usize>) -> anyhow::Result<usize> {
    let wanted = match threads {
        Some(0) => anyhow::bail!("machine profile: threads must be at least 1"),
        Some(n) => n,
        None => std::thread::available_parallelism().map_or(1, std::num::NonZero::get),
    };
    // Err only if the pool already exists, which keeps its size
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(wanted)
        .build_global();
    Ok(rayon::current_num_threads())
}
