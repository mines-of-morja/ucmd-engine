//! Random streams: one ChaCha8 stream per purpose, all derived from the run's master seed
//! (CLAUDE.md invariant 8). The full generator state is serializable for checkpoints.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Stream used to generate initial velocities.
pub const STREAM_VELOCITIES: u64 = 1;
/// Stream of the thermostat.
pub const STREAM_THERMOSTAT: u64 = 2;
/// Stream of the barostat (E2.7).
pub const STREAM_BAROSTAT: u64 = 3;

/// The generator of stream `stream` for master seed `seed`.
pub fn stream(seed: u64, stream: u64) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(stream);
    rng
}
