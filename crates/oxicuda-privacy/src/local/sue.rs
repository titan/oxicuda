//! Symmetric Unary Encoding (SUE) — "Basic RAPPOR" — for local DP frequency
//! estimation.
//!
//! Reference: Wang, Blocki, Li & Jha (2017), "Locally Differentially Private
//! Protocols for Frequency Estimation", USENIX Security 2017 (§4, the symmetric
//! unary-encoding / Basic One-time RAPPOR protocol of Erlingsson et al. 2014).
//!
//! # Protocol
//! For input `v ∈ {0, …, k−1}`:
//! 1. Form the one-hot bit vector `B*` of length `k`: `B*[v] = 1`, others `0`.
//! 2. Flip **every** bit with the *same* symmetric probabilities:
//!    - a true bit (`B*[i] = 1`) stays `1` with probability `p`,
//!    - a false bit (`B*[i] = 0`) becomes `1` with probability `q = 1 − p`,
//!
//!    where `p = e^{ε/2} / (e^{ε/2} + 1)` and `q = 1 / (e^{ε/2} + 1)`.
//!
//! Because the per-bit perturbation is symmetric (`q = 1 − p`), the protocol is
//! `ε`-LDP: each of the (at most) two differing bits between neighbouring inputs
//! contributes `ln(p/q) = ε/2`, summing to `ε`.
//!
//! # Difference from OUE
//! Optimised Unary Encoding (`local/oue.rs`) is **asymmetric**: it keeps true
//! bits with probability `½` and flips false bits with `q = 1/(e^ε + 1)`,
//! achieving lower variance. SUE keeps the symmetric `p / (1 − p)` design of the
//! original Basic-RAPPOR and is the natural "unary-encoding randomised response"
//! variant. Both share the same unbiased-estimator template `f̂ = (mean − q)/(p − q)`
//! but with different `p`, `q`, and therefore different variance.
//!
//! # Frequency estimation
//! Unbiased estimator over `n` reports:
//!
//! `f̂_v = ( (Σᵢ B̃_v(i)) / n − q ) / (p − q)`.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Configuration for Symmetric Unary Encoding (Basic RAPPOR).
#[derive(Debug, Clone)]
pub struct SueConfig {
    /// Privacy parameter ε > 0.
    pub epsilon: f64,
    /// Domain size k ≥ 2.
    pub k: usize,
}

impl SueConfig {
    /// Construct and validate a `SueConfig`.
    ///
    /// # Errors
    /// Returns `NonPositiveEpsilon` if `epsilon ≤ 0`, or `InvalidParameter` if `k < 2`.
    pub fn new(epsilon: f64, k: usize) -> PrivacyResult<Self> {
        if epsilon <= 0.0 {
            return Err(PrivacyError::NonPositiveEpsilon(epsilon));
        }
        if k < 2 {
            return Err(PrivacyError::InvalidParameter(
                "domain size k must be ≥ 2".into(),
            ));
        }
        Ok(Self { epsilon, k })
    }

    /// Probability `p = e^{ε/2} / (e^{ε/2} + 1)` of keeping a true bit set.
    #[must_use]
    pub fn p_keep(&self) -> f64 {
        let e_half = (self.epsilon / 2.0).exp();
        e_half / (e_half + 1.0)
    }

    /// Probability `q = 1 / (e^{ε/2} + 1) = 1 − p` of setting a false bit.
    #[must_use]
    pub fn q_flip(&self) -> f64 {
        1.0 / ((self.epsilon / 2.0).exp() + 1.0)
    }

    /// Per-report variance of the SUE frequency estimator (Wang et al. 2017, §4):
    /// `Var = q·(1 − q) / (p − q)²` (independent of the true frequency under the
    /// pure-LDP analysis), divided by `n` once aggregated over `n` users.
    #[must_use]
    pub fn estimator_variance(&self) -> f64 {
        let p = self.p_keep();
        let q = self.q_flip();
        let denom = p - q;
        q * (1.0 - q) / (denom * denom)
    }
}

/// Encode an input value via Symmetric Unary Encoding.
///
/// Returns a length-`k` bit vector (each entry `0` or `1`).
///
/// # Errors
/// - `IndexOutOfRange` if `input ≥ k`.
pub fn sue_encode(input: usize, cfg: &SueConfig, rng: &mut LcgRng) -> PrivacyResult<Vec<u8>> {
    if input >= cfg.k {
        return Err(PrivacyError::IndexOutOfRange(input, cfg.k));
    }
    let p = cfg.p_keep();
    let q = cfg.q_flip();
    let mut bits = Vec::with_capacity(cfg.k);
    for i in 0..cfg.k {
        let u = rng.next_f64();
        let bit = if i == input { u < p } else { u < q };
        bits.push(bit as u8);
    }
    Ok(bits)
}

/// Estimate value frequencies from SUE reports.
///
/// Returns a length-`k` vector of unbiased frequency estimates
/// `f̂_v = (mean_v − q) / (p − q)`.
///
/// # Errors
/// - `EmptyInput` if `reports` is empty.
/// - `DimensionMismatch` if any report has length ≠ k.
pub fn sue_estimate_frequency(reports: &[Vec<u8>], cfg: &SueConfig) -> PrivacyResult<Vec<f64>> {
    if reports.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    let n = reports.len() as f64;
    let p = cfg.p_keep();
    let q = cfg.q_flip();
    let denom = p - q;

    let mut sums = vec![0u64; cfg.k];
    for report in reports {
        if report.len() != cfg.k {
            return Err(PrivacyError::DimensionMismatch {
                expected: cfg.k,
                got: report.len(),
            });
        }
        for (j, &b) in report.iter().enumerate() {
            sums[j] += u64::from(b);
        }
    }

    Ok(sums.iter().map(|&s| (s as f64 / n - q) / denom).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validation() {
        assert!(SueConfig::new(0.0, 4).is_err());
        assert!(SueConfig::new(-1.0, 4).is_err());
        assert!(SueConfig::new(1.0, 1).is_err());
        assert!(SueConfig::new(1.0, 2).is_ok());
    }

    #[test]
    fn probabilities_are_symmetric() {
        let cfg = SueConfig::new(2.0, 5).expect("ok");
        let p = cfg.p_keep();
        let q = cfg.q_flip();
        // q = 1 − p exactly (symmetric design).
        assert!((q - (1.0 - p)).abs() < 1e-12, "p={p}, q={q}");
        assert!(p > 0.5 && q < 0.5);
    }

    #[test]
    fn privacy_ratio_matches_epsilon() {
        // Per-bit ln(p/q) = ε/2; neighbouring inputs differ in exactly two bits,
        // so the full-report privacy loss is 2·(ε/2) = ε.
        let eps = 1.7;
        let cfg = SueConfig::new(eps, 8).expect("ok");
        let per_bit = (cfg.p_keep() / cfg.q_flip()).ln();
        assert!(
            (per_bit - eps / 2.0).abs() < 1e-9,
            "per-bit ln(p/q)={per_bit}, ε/2={}",
            eps / 2.0
        );
        assert!(
            (2.0 * per_bit - eps).abs() < 1e-9,
            "report loss must equal ε"
        );
    }

    #[test]
    fn encode_length_and_binary() {
        let cfg = SueConfig::new(2.0, 6).expect("ok");
        let mut rng = LcgRng::new(42);
        let bits = sue_encode(3, &cfg, &mut rng).expect("ok");
        assert_eq!(bits.len(), 6);
        for &b in &bits {
            assert!(b == 0 || b == 1);
        }
    }

    #[test]
    fn encode_out_of_range_errors() {
        let cfg = SueConfig::new(1.0, 3).expect("ok");
        let mut rng = LcgRng::new(0);
        assert!(sue_encode(3, &cfg, &mut rng).is_err());
        assert!(sue_encode(100, &cfg, &mut rng).is_err());
    }

    #[test]
    fn estimate_empty_errors() {
        let cfg = SueConfig::new(1.0, 3).expect("ok");
        assert!(sue_estimate_frequency(&[], &cfg).is_err());
    }

    #[test]
    fn estimate_dimension_mismatch_errors() {
        let cfg = SueConfig::new(1.0, 4).expect("ok");
        let reports = vec![vec![0u8, 1, 0, 0], vec![0u8, 1, 0]]; // second has wrong len
        assert!(matches!(
            sue_estimate_frequency(&reports, &cfg),
            Err(PrivacyError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn estimate_unbiased_single_value() {
        let cfg = SueConfig::new(3.0, 4).expect("ok");
        let mut rng = LcgRng::new(99);
        let n = 20_000;
        let reports: Vec<Vec<u8>> = (0..n)
            .map(|_| sue_encode(1, &cfg, &mut rng).expect("ok"))
            .collect();
        let freqs = sue_estimate_frequency(&reports, &cfg).expect("ok");
        assert!((freqs[1] - 1.0).abs() < 0.1, "f̂(1)={}", freqs[1]);
        for (j, &f) in freqs.iter().enumerate() {
            if j != 1 {
                assert!(f.abs() < 0.15, "f̂({j})={f}");
            }
        }
    }

    #[test]
    fn estimate_recovers_mixture_distribution() {
        // Two-thirds of users report value 0, one-third report value 2.
        let cfg = SueConfig::new(4.0, 3).expect("ok");
        let mut rng = LcgRng::new(2024);
        let mut reports = Vec::new();
        for i in 0..30_000usize {
            let v = if i % 3 == 0 { 2 } else { 0 };
            reports.push(sue_encode(v, &cfg, &mut rng).expect("ok"));
        }
        let freqs = sue_estimate_frequency(&reports, &cfg).expect("ok");
        assert!((freqs[0] - 2.0 / 3.0).abs() < 0.05, "f̂(0)={}", freqs[0]);
        assert!((freqs[2] - 1.0 / 3.0).abs() < 0.05, "f̂(2)={}", freqs[2]);
        assert!(freqs[1].abs() < 0.05, "f̂(1)={}", freqs[1]);
    }

    #[test]
    fn estimates_sum_to_about_one() {
        let cfg = SueConfig::new(3.0, 5).expect("ok");
        let mut rng = LcgRng::new(7);
        let mut reports = Vec::new();
        for i in 0..20_000usize {
            reports.push(sue_encode(i % 5, &cfg, &mut rng).expect("ok"));
        }
        let freqs = sue_estimate_frequency(&reports, &cfg).expect("ok");
        let total: f64 = freqs.iter().sum();
        assert!((total - 1.0).abs() < 0.05, "Σf̂={total}");
    }

    #[test]
    fn variance_decreases_with_epsilon() {
        let lo = SueConfig::new(0.5, 10).expect("ok");
        let hi = SueConfig::new(4.0, 10).expect("ok");
        assert!(
            hi.estimator_variance() < lo.estimator_variance(),
            "variance should shrink as ε grows: lo={}, hi={}",
            lo.estimator_variance(),
            hi.estimator_variance()
        );
    }

    #[test]
    fn deterministic_with_same_seed() {
        let cfg = SueConfig::new(2.0, 6).expect("ok");
        let mut a = LcgRng::new(314);
        let mut b = LcgRng::new(314);
        for _ in 0..50 {
            assert_eq!(
                sue_encode(2, &cfg, &mut a).expect("ok"),
                sue_encode(2, &cfg, &mut b).expect("ok")
            );
        }
    }
}
