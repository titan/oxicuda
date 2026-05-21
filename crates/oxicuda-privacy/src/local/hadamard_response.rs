//! Hadamard response: communication-efficient ε-LDP frequency estimation.
//!
//! Reference: Acharya, Sun & Zhang (2019), "Hadamard Response: Estimating
//! Distributions Privately, Efficiently, and with Little Communication",
//! AISTATS 2019.
//!
//! # Protocol summary (Sylvester Hadamard matrix H of order K' = 2^⌈log₂ K⌉)
//! Each user holds `x ∈ [K]` and:
//! 1. picks a uniformly random column `s ∈ [K']`,
//! 2. computes the matrix sign `t = H[x, s] ∈ {−1, +1}`,
//! 3. reports `b = 1` if t was kept or `b = 0` if t was flipped, where the
//!    flip happens with probability `1 / (1 + e^ε)`.
//!
//! The aggregator accumulates `accumulated[s] += (2b − 1)` per column, then
//! performs an inverse fast Walsh-Hadamard transform with the bias correction
//! `(e^ε + 1) / (e^ε − 1)` to recover the unbiased frequency vector.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// User-facing configuration for the Hadamard response mechanism.
#[derive(Clone, Debug)]
pub struct HadamardResponseConfig {
    /// Number of categories K ≥ 1.
    pub domain_size: usize,
    /// Privacy parameter ε > 0.
    pub epsilon: f64,
}

impl HadamardResponseConfig {
    /// Construct and validate the configuration.
    ///
    /// # Errors
    /// - `EmptyInput` if `domain_size == 0`.
    /// - `NonPositiveEpsilon` if `epsilon ≤ 0`.
    pub fn new(domain_size: usize, epsilon: f64) -> PrivacyResult<Self> {
        if domain_size == 0 {
            return Err(PrivacyError::EmptyInput);
        }
        if !epsilon.is_finite() {
            return Err(PrivacyError::InvalidParameter(
                "epsilon must be finite".into(),
            ));
        }
        if epsilon <= 0.0 {
            return Err(PrivacyError::NonPositiveEpsilon(epsilon));
        }
        Ok(Self {
            domain_size,
            epsilon,
        })
    }
}

/// Round `domain_size` up to the next power of two (at least 1).
#[must_use]
pub fn next_power_of_two(domain_size: usize) -> usize {
    if domain_size <= 1 {
        1
    } else {
        domain_size.next_power_of_two()
    }
}

/// One-bit payload from a single user.
#[derive(Clone, Debug)]
pub struct HadamardPayload {
    pub column: usize,
    pub bit: u8,
}

/// Sylvester-Hadamard sign at row `i`, column `j` for the order K' (a power of 2).
///
/// `H[i, j] = (−1)^popcount(i AND j)`, returning ±1 as `i8`.
#[must_use]
pub fn hadamard_sign(row: usize, col: usize) -> i8 {
    let parity = (row & col).count_ones() & 1;
    if parity == 0 { 1 } else { -1 }
}

/// Encoder that draws random columns and reports the (possibly flipped) sign.
pub struct HadamardResponseEncoder {
    cfg: HadamardResponseConfig,
    padded_size: usize,
    keep_prob: f64,
    rng: LcgRng,
}

impl HadamardResponseEncoder {
    /// Construct a new encoder seeded by `seed`.
    ///
    /// # Errors
    /// Propagates `HadamardResponseConfig::new` errors.
    pub fn new(cfg: HadamardResponseConfig, seed: u64) -> PrivacyResult<Self> {
        let padded_size = next_power_of_two(cfg.domain_size);
        let exp_eps = cfg.epsilon.exp();
        let keep_prob = exp_eps / (1.0 + exp_eps);
        Ok(Self {
            cfg,
            padded_size,
            keep_prob,
            rng: LcgRng::new(seed),
        })
    }

    /// Size after padding to the next power of two.
    #[must_use]
    pub fn padded_size(&self) -> usize {
        self.padded_size
    }

    /// Borrow the active configuration.
    #[must_use]
    pub fn config(&self) -> &HadamardResponseConfig {
        &self.cfg
    }

    /// Encode a single user's value `x` into a `HadamardPayload`.
    ///
    /// # Errors
    /// `IndexOutOfRange` if `x ≥ domain_size`.
    pub fn encode(&mut self, x: usize) -> PrivacyResult<HadamardPayload> {
        if x >= self.cfg.domain_size {
            return Err(PrivacyError::IndexOutOfRange(x, self.cfg.domain_size));
        }
        // LCG low bits are weak; use the upper word to sample uniform columns.
        let raw = self.rng.next_u64();
        let column = ((raw >> 32) as usize) % self.padded_size;
        let true_sign = hadamard_sign(x, column);
        // bit==1 if reporting +1, bit==0 if reporting −1.
        let true_bit: u8 = if true_sign == 1 { 1 } else { 0 };
        let keep_draw = self.rng.next_f64();
        let bit = if keep_draw < self.keep_prob {
            true_bit
        } else {
            1 - true_bit
        };
        Ok(HadamardPayload { column, bit })
    }
}

/// Aggregator that sums signed payloads and inverts the Walsh-Hadamard map.
pub struct HadamardResponseAggregator {
    cfg: HadamardResponseConfig,
    padded_size: usize,
    accumulated: Vec<f64>,
    n: usize,
}

impl HadamardResponseAggregator {
    /// Construct a new aggregator with all-zero accumulators.
    ///
    /// # Errors
    /// Propagates `HadamardResponseConfig::new` errors via the supplied cfg.
    pub fn new(cfg: HadamardResponseConfig) -> PrivacyResult<Self> {
        let padded_size = next_power_of_two(cfg.domain_size);
        let accumulated = vec![0.0f64; padded_size];
        Ok(Self {
            cfg,
            padded_size,
            accumulated,
            n: 0,
        })
    }

    /// Number of payloads aggregated.
    #[must_use]
    pub fn count(&self) -> usize {
        self.n
    }

    /// Padded length used internally (= row/column count of the Hadamard matrix).
    #[must_use]
    pub fn padded_size(&self) -> usize {
        self.padded_size
    }

    /// Accumulate one payload.
    ///
    /// # Errors
    /// - `IndexOutOfRange` if `payload.column ≥ padded_size`.
    /// - `InvalidParameter` if `payload.bit ∉ {0, 1}`.
    pub fn add(&mut self, payload: &HadamardPayload) -> PrivacyResult<()> {
        if payload.column >= self.padded_size {
            return Err(PrivacyError::IndexOutOfRange(
                payload.column,
                self.padded_size,
            ));
        }
        let contribution = match payload.bit {
            1 => 1.0,
            0 => -1.0,
            _ => {
                return Err(PrivacyError::InvalidParameter(format!(
                    "bit must be 0 or 1, got {}",
                    payload.bit
                )));
            }
        };
        self.accumulated[payload.column] += contribution;
        self.n += 1;
        Ok(())
    }

    /// Produce the de-biased frequency estimate over `[0, domain_size)`.
    ///
    /// # Errors
    /// `InvalidParameter` if internal bias correction would divide by zero
    /// (only when ε = 0, which configuration validation already rejects).
    pub fn estimate(&self) -> PrivacyResult<Vec<f64>> {
        if self.n == 0 {
            return Ok(vec![0.0f64; self.cfg.domain_size]);
        }
        let exp_eps = self.cfg.epsilon.exp();
        let denom = exp_eps - 1.0;
        if denom.abs() < f64::EPSILON {
            return Err(PrivacyError::InvalidParameter(
                "bias correction undefined for ε = 0".into(),
            ));
        }
        let bias = (exp_eps + 1.0) / denom;
        let mut spectrum = self.accumulated.clone();
        fast_walsh_hadamard_transform(&mut spectrum);
        let n_f = self.n as f64;
        // E[(H·accumulated)[x]] = (e^ε−1)/(e^ε+1) · n_x because column sampling
        // contributes a 1/K' factor and the H·Hᵀ = K'·I orthogonality cancels
        // it exactly. Thus the unbiased estimate divides the FWHT output by n
        // and multiplies by the bias correction (e^ε+1)/(e^ε−1).
        let scale = bias / n_f;
        let mut estimate = Vec::with_capacity(self.cfg.domain_size);
        for &v in spectrum.iter().take(self.cfg.domain_size) {
            estimate.push(v * scale);
        }
        Ok(estimate)
    }
}

/// In-place fast Walsh-Hadamard transform on a power-of-two length slice.
///
/// Computes the unnormalised symmetric Hadamard transform y = H · x. Because
/// the Sylvester Hadamard matrix satisfies `H · H = N · I`, the inverse is
/// `x = (1/N) · H · y` — i.e. the same butterfly divided by `N`.
pub fn fast_walsh_hadamard_transform(data: &mut [f64]) {
    let n = data.len();
    if n <= 1 || !n.is_power_of_two() {
        return;
    }
    let mut h = 1usize;
    while h < n {
        let stride = h * 2;
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let a = data[j];
                let b = data[j + h];
                data[j] = a + b;
                data[j + h] = a - b;
            }
            i += stride;
        }
        h *= 2;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_domain_zero_errors() {
        assert!(HadamardResponseConfig::new(0, 1.0).is_err());
    }

    #[test]
    fn test_new_epsilon_non_positive_errors() {
        assert!(HadamardResponseConfig::new(8, 0.0).is_err());
        assert!(HadamardResponseConfig::new(8, -1.0).is_err());
    }

    #[test]
    fn test_padded_size_is_power_of_two_and_at_least_domain() {
        for k in 1..40usize {
            let padded = next_power_of_two(k);
            assert!(padded.is_power_of_two(), "padded {padded} not pow2");
            assert!(padded >= k, "padded {padded} < domain {k}");
        }
    }

    #[test]
    fn test_padded_size_seven_is_eight() {
        assert_eq!(next_power_of_two(7), 8);
    }

    #[test]
    fn test_padded_size_eight_stays_eight() {
        assert_eq!(next_power_of_two(8), 8);
    }

    #[test]
    fn test_large_epsilon_keeps_true_sign() {
        let cfg = HadamardResponseConfig::new(8, 50.0).expect("cfg");
        let mut encoder = HadamardResponseEncoder::new(cfg, 17).expect("enc");
        let x = 3usize;
        for _ in 0..512 {
            let payload = encoder.encode(x).expect("enc");
            let reported = if payload.bit == 1 { 1i8 } else { -1i8 };
            let expected = hadamard_sign(x, payload.column);
            assert_eq!(reported, expected, "ε→∞ should never flip");
        }
    }

    #[test]
    fn test_empty_aggregator_is_zero() {
        let cfg = HadamardResponseConfig::new(5, 1.0).expect("cfg");
        let agg = HadamardResponseAggregator::new(cfg).expect("agg");
        let est = agg.estimate().expect("est");
        assert_eq!(est.len(), 5);
        for v in est {
            assert!(v.abs() < 1e-12, "expected zero, got {v}");
        }
    }

    #[test]
    fn test_single_user_round_trip_no_privacy() {
        let cfg = HadamardResponseConfig::new(8, 50.0).expect("cfg");
        let mut encoder = HadamardResponseEncoder::new(cfg.clone(), 7).expect("enc");
        let mut agg = HadamardResponseAggregator::new(cfg).expect("agg");
        let x = 5usize;
        for _ in 0..4_000 {
            let payload = encoder.encode(x).expect("enc");
            agg.add(&payload).expect("add");
        }
        let est = agg.estimate().expect("est");
        // Mass concentrated at x.
        let total: f64 = est.iter().sum();
        assert!((total - 1.0).abs() < 0.1, "total mass {total}");
        let max_idx = est
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(max_idx, x, "max should be at x={x}, got {max_idx}");
        assert!(est[x] > 0.7, "expected ≈1 at x={x}, got {}", est[x]);
    }

    #[test]
    fn test_unbiased_estimator_known_distribution() {
        let cfg = HadamardResponseConfig::new(3, 4.0).expect("cfg");
        let mut encoder = HadamardResponseEncoder::new(cfg.clone(), 99).expect("enc");
        let mut agg = HadamardResponseAggregator::new(cfg).expect("agg");
        let true_freq = [0.5f64, 0.3, 0.2];
        let n = 5_000usize;
        let counts = [
            (n as f64 * true_freq[0]).round() as usize,
            (n as f64 * true_freq[1]).round() as usize,
            (n as f64 * true_freq[2]).round() as usize,
        ];
        for (label, &c) in counts.iter().enumerate() {
            for _ in 0..c {
                let payload = encoder.encode(label).expect("enc");
                agg.add(&payload).expect("add");
            }
        }
        let est = agg.estimate().expect("est");
        for (i, (&got, &want)) in est.iter().zip(true_freq.iter()).enumerate() {
            assert!(
                (got - want).abs() < 0.05,
                "label {i}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn test_estimate_sum_near_one() {
        let cfg = HadamardResponseConfig::new(4, 4.0).expect("cfg");
        let mut encoder = HadamardResponseEncoder::new(cfg.clone(), 13).expect("enc");
        let mut agg = HadamardResponseAggregator::new(cfg).expect("agg");
        let n = 5_000usize;
        for i in 0..n {
            let payload = encoder.encode(i % 4).expect("enc");
            agg.add(&payload).expect("add");
        }
        let est = agg.estimate().expect("est");
        let total: f64 = est.iter().sum();
        assert!((total - 1.0).abs() < 0.1, "total mass {total}");
    }

    #[test]
    fn test_add_invalid_column_errors() {
        let cfg = HadamardResponseConfig::new(5, 2.0).expect("cfg");
        let mut agg = HadamardResponseAggregator::new(cfg).expect("agg");
        let bad = HadamardPayload { column: 64, bit: 1 };
        assert!(agg.add(&bad).is_err());
    }

    #[test]
    fn test_add_invalid_bit_errors() {
        let cfg = HadamardResponseConfig::new(4, 2.0).expect("cfg");
        let mut agg = HadamardResponseAggregator::new(cfg).expect("agg");
        let bad = HadamardPayload { column: 0, bit: 2 };
        assert!(agg.add(&bad).is_err());
    }

    #[test]
    fn test_encode_out_of_domain_errors() {
        let cfg = HadamardResponseConfig::new(4, 1.0).expect("cfg");
        let mut encoder = HadamardResponseEncoder::new(cfg, 0).expect("enc");
        assert!(encoder.encode(4).is_err());
        assert!(encoder.encode(7).is_err());
    }

    #[test]
    fn test_hadamard_symmetry_h_times_ht() {
        // H · Hᵀ = N · I via row-dot products at N = 8.
        let n = 8usize;
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0i64;
                for s in 0..n {
                    dot += hadamard_sign(i, s) as i64 * hadamard_sign(j, s) as i64;
                }
                if i == j {
                    assert_eq!(dot, n as i64, "diagonal at i={i}");
                } else {
                    assert_eq!(dot, 0, "off-diagonal at ({i},{j})");
                }
            }
        }
    }

    #[test]
    fn test_larger_epsilon_smaller_variance() {
        fn run(epsilon: f64) -> Vec<f64> {
            let cfg = HadamardResponseConfig::new(4, epsilon).expect("cfg");
            let mut encoder = HadamardResponseEncoder::new(cfg.clone(), 31).expect("enc");
            let mut agg = HadamardResponseAggregator::new(cfg).expect("agg");
            let true_freq = [0.4f64, 0.3, 0.2, 0.1];
            let n = 1_000usize;
            let counts = [
                (n as f64 * true_freq[0]).round() as usize,
                (n as f64 * true_freq[1]).round() as usize,
                (n as f64 * true_freq[2]).round() as usize,
                (n as f64 * true_freq[3]).round() as usize,
            ];
            for (label, &c) in counts.iter().enumerate() {
                for _ in 0..c {
                    let payload = encoder.encode(label).expect("enc");
                    agg.add(&payload).expect("add");
                }
            }
            let est = agg.estimate().expect("est");
            let true_freq_v: Vec<f64> = true_freq.to_vec();
            est.iter()
                .zip(true_freq_v.iter())
                .map(|(e, t)| (e - t).powi(2))
                .collect()
        }
        let err_low = run(1.0).iter().sum::<f64>();
        let err_high = run(4.0).iter().sum::<f64>();
        assert!(
            err_high < err_low,
            "expected smaller variance at ε=4 (={err_high}) than ε=1 (={err_low})"
        );
    }

    #[test]
    fn test_encoder_deterministic_with_same_seed() {
        let cfg = HadamardResponseConfig::new(8, 2.0).expect("cfg");
        let mut a = HadamardResponseEncoder::new(cfg.clone(), 1234).expect("a");
        let mut b = HadamardResponseEncoder::new(cfg, 1234).expect("b");
        for x in 0..50usize {
            let pa = a.encode(x % 8).expect("a enc");
            let pb = b.encode(x % 8).expect("b enc");
            assert_eq!(pa.column, pb.column, "column mismatch at step {x}");
            assert_eq!(pa.bit, pb.bit, "bit mismatch at step {x}");
        }
    }
}
