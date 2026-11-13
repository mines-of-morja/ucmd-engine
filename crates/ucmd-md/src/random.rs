//! Random streams: one ChaCha8 stream per purpose, all derived from the run's master seed
//! (CLAUDE.md invariant 8). The full generator state is serializable for checkpoints.
//!
//! The samplers below turn the stream into uniform, normal and gamma deviates with integer
//! arithmetic, IEEE-exact operations (+ − × ÷ √) and the pure-Rust `libm` for logarithms and
//! powers. No platform math library is involved, so a given seed gives bit-identical numbers on
//! every machine (ADR 0010).

use rand::{Rng, SeedableRng};
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

/// Uniform deviate in [0, 1) with 53 random bits: exact integer-to-float conversion.
pub fn uniform(rng: &mut ChaCha8Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

/// Standard normal deviate (Marsaglia's polar method). One deviate per call and no cached
/// second deviate, so the generator state alone determines the sequence.
pub fn standard_normal(rng: &mut ChaCha8Rng) -> f64 {
    loop {
        let u = 2.0 * uniform(rng) - 1.0;
        let v = 2.0 * uniform(rng) - 1.0;
        let s = u * u + v * v;
        if s > 0.0 && s < 1.0 {
            return u * (-2.0 * libm::log(s) / s).sqrt();
        }
    }
}

/// Gamma(shape, scale 1) deviate for `shape` > 0 (Marsaglia & Tsang, ACM Trans. Math. Softw.
/// 26, 363 (2000); for shape < 1 their boost Gamma(a) = Gamma(a + 1) · U^(1/a)).
///
/// The caller guarantees `shape` > 0 and finite.
pub fn gamma(rng: &mut ChaCha8Rng, shape: f64) -> f64 {
    debug_assert!(shape > 0.0 && shape.is_finite());
    if shape < 1.0 {
        let g = gamma(rng, shape + 1.0);
        return g * libm::pow(uniform(rng), 1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = standard_normal(rng);
        let t = 1.0 + c * x;
        if t <= 0.0 {
            continue;
        }
        let v = t * t * t;
        let u = uniform(rng);
        let x2 = x * x;
        if u < 1.0 - 0.0331 * x2 * x2 || libm::log(u) < 0.5 * x2 + d * (1.0 - v + libm::log(v)) {
            return d * v;
        }
    }
}

/// χ² deviate with `k` > 0 degrees of freedom: 2 · Gamma(k/2).
pub fn chi_squared(rng: &mut ChaCha8Rng, k: f64) -> f64 {
    2.0 * gamma(rng, 0.5 * k)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moments(xs: &[f64]) -> (f64, f64) {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
        (mean, var)
    }

    #[test]
    fn uniform_is_in_range_and_fair() {
        let mut rng = stream(1, 1);
        let xs: Vec<f64> = (0..200_000).map(|_| uniform(&mut rng)).collect();
        assert!(xs.iter().all(|&x| (0.0..1.0).contains(&x)));
        let (mean, var) = moments(&xs);
        assert!((mean - 0.5).abs() < 0.003, "{mean}");
        assert!((var - 1.0 / 12.0).abs() < 0.001, "{var}");
    }

    #[test]
    fn normal_moments() {
        let mut rng = stream(2, 1);
        let xs: Vec<f64> = (0..200_000).map(|_| standard_normal(&mut rng)).collect();
        let (mean, var) = moments(&xs);
        assert!(mean.abs() < 0.01, "{mean}");
        assert!((var - 1.0).abs() < 0.01, "{var}");
        let fourth = xs.iter().map(|x| x.powi(4)).sum::<f64>() / xs.len() as f64;
        assert!((fourth - 3.0).abs() < 0.06, "{fourth}");
    }

    #[test]
    fn gamma_moments_for_small_and_large_shapes() {
        for shape in [0.3, 1.0, 2.5, 431.5] {
            let mut rng = stream(3, 1);
            let xs: Vec<f64> = (0..100_000).map(|_| gamma(&mut rng, shape)).collect();
            let (mean, var) = moments(&xs);
            assert!(
                (mean / shape - 1.0).abs() < 0.02,
                "shape {shape}: mean {mean}"
            );
            assert!((var / shape - 1.0).abs() < 0.04, "shape {shape}: var {var}");
            assert!(xs.iter().all(|&x| x >= 0.0 && x.is_finite()));
        }
        let mut rng = stream(4, 1);
        let k = 2589.0;
        let xs: Vec<f64> = (0..20_000).map(|_| chi_squared(&mut rng, k)).collect();
        let (mean, var) = moments(&xs);
        assert!((mean / k - 1.0).abs() < 0.01 && (var / (2.0 * k) - 1.0).abs() < 0.05);
    }

    #[test]
    fn samplers_are_pinned() {
        // The exact values pin the implementation: a change here changes every trajectory
        // (ADR 0010). Update only on purpose, with a note in the changelog.
        let mut rng = stream(42, 7);
        let got = [
            uniform(&mut rng),
            standard_normal(&mut rng),
            gamma(&mut rng, 3.5),
        ];
        let bits: Vec<u64> = got.iter().map(|x| x.to_bits()).collect();
        assert_eq!(bits, PINNED, "values {got:?}");
    }

    const PINNED: [u64; 3] = [
        4593797953228562192,
        13833065431725548817,
        4613758118989381329,
    ];
}
