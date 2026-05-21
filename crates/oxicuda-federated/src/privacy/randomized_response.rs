//! Local-DP randomized response for categorical data.
//!
//! Warner, S. L. (1965). "Randomized Response: A Survey Technique for
//! Eliminating Evasive Answer Bias." *Journal of the American Statistical
//! Association*, 60(309), 63-69.
//!
//! Generalized to k categories (k-RR): each user reports their true category
//! with probability
//!
//! ```text
//! p = e^ε / (e^ε + k − 1)
//! ```
//!
//! Otherwise the user reports a category drawn uniformly from the other
//! `k − 1` options. The resulting mechanism satisfies ε-local differential
//! privacy: for any two true values `v ≠ v'` and any reported value `r`,
//!
//! ```text
//! Pr[report=r | true=v] / Pr[report=r | true=v'] ≤ e^ε.
//! ```
//!
//! ## Unbiased frequency aggregator
//!
//! Let `q_c = count_c / n_total` be the empirical fraction of reports landing
//! in category `c`. The true frequency `f_c` satisfies
//!
//! ```text
//! E[q_c] = p · f_c + (1 − p) · (1 / k)
//!
//! ⇔  f̂_c = (q_c − (1 − p) / k) / (p − (1 − p) / k)
//! ```
//!
//! which is unbiased provided `p ≠ 1/k` (i.e. `ε > 0`).
//!
//! The exact reciprocal of the perturbation matrix can be derived in closed
//! form because the matrix has the structure
//! `P = (p − (1−p)/(k−1)) · I + ((1−p)/(k−1)) · 𝟙𝟙ᵀ`, but the
//! per-category Horvitz–Thompson scaling above is the standard inversion
//! used by Erlingsson et al. (RAPPOR) and Wang et al. (k-RR).

use crate::error::{FedError, FedResult};
use crate::handle::LcgRng;

/// Draw a uniform `[0, 1)` deviate from an [`LcgRng`].
///
/// `LcgRng::next_u32` actually returns the high *31* bits of the
/// 64-bit LCG state (`max = 2^31 − 1`), so dividing by `u32::MAX + 1`
/// only yields `[0, 0.5)`. Dividing by `2^31` fixes the range.
#[inline]
fn uniform_unit(rng: &mut LcgRng) -> f32 {
    const SCALE: f32 = (1_u64 << 31) as f32;
    rng.next_u32() as f32 / SCALE
}

/// Configuration for the k-ary randomized-response mechanism.
#[derive(Debug, Clone)]
pub struct RandomizedResponseConfig {
    /// Local-DP privacy budget `ε > 0`. Larger ε → more truth, less noise.
    pub epsilon: f32,
    /// Number of categories `k ≥ 2`.
    pub n_categories: usize,
}

/// k-ary randomized-response mechanism (Warner 1965 generalised to k > 2).
#[derive(Debug, Clone)]
pub struct RandomizedResponse {
    cfg: RandomizedResponseConfig,
    p_truth: f32,
}

impl RandomizedResponse {
    /// Construct a validated k-RR mechanism.
    ///
    /// # Errors
    /// - [`FedError::InvalidPrivacyBudget`] if `epsilon ≤ 0` or non-finite.
    /// - [`FedError::InvalidShareCount`] if `n_categories < 2`.
    pub fn new(cfg: RandomizedResponseConfig) -> FedResult<Self> {
        if !(cfg.epsilon > 0.0 && cfg.epsilon.is_finite()) {
            return Err(FedError::InvalidPrivacyBudget);
        }
        if cfg.n_categories < 2 {
            return Err(FedError::InvalidShareCount {
                min: 2,
                got: cfg.n_categories,
            });
        }
        let p_truth = Self::compute_p_truth(cfg.epsilon, cfg.n_categories);
        Ok(Self { cfg, p_truth })
    }

    /// Closed-form `p = e^ε / (e^ε + k − 1)`.
    ///
    /// Computed in `f64` then cast back to `f32` to avoid the
    /// `e^ε` overflow that happens around ε ≈ 88 in `f32`.
    fn compute_p_truth(epsilon: f32, n_categories: usize) -> f32 {
        let eps = epsilon as f64;
        let k_minus_1 = (n_categories - 1) as f64;
        // For very large ε, e^ε dominates and p → 1 numerically.
        let e_eps = eps.exp();
        if !e_eps.is_finite() {
            return 1.0;
        }
        (e_eps / (e_eps + k_minus_1)) as f32
    }

    /// Probability `p = e^ε / (e^ε + k − 1)` that a user reports their
    /// true category.
    #[must_use]
    pub fn p_truth(&self) -> f32 {
        self.p_truth
    }

    /// Return the configured ε.
    #[must_use]
    pub fn epsilon(&self) -> f32 {
        self.cfg.epsilon
    }

    /// Return the configured number of categories `k`.
    #[must_use]
    pub fn n_categories(&self) -> usize {
        self.cfg.n_categories
    }

    /// Perturb a single categorical report.
    ///
    /// With probability `p_truth`, return `true_value`.
    /// Otherwise return a category drawn uniformly from the
    /// other `k − 1` options.
    ///
    /// # Errors
    /// - [`FedError::InvalidShareCount`] if `true_value ≥ n_categories`.
    pub fn perturb(&self, true_value: usize, rng: &mut LcgRng) -> FedResult<usize> {
        let k = self.cfg.n_categories;
        if true_value >= k {
            return Err(FedError::InvalidShareCount {
                min: k,
                got: true_value,
            });
        }
        // Draw a uniform [0, 1) deviate from the LCG and decide
        // "tell the truth" vs "lie". The crate's `next_u32()` returns
        // the high 31 bits of the 64-bit LCG state (max value 2^31 − 1),
        // so we divide by 2^31 to get a proper [0, 1) sample.
        let u = uniform_unit(rng);
        if u < self.p_truth {
            return Ok(true_value);
        }
        // Otherwise pick uniformly from the other k−1 categories.
        // Sample index in [0, k−1), then skip `true_value`.
        let other = rng.next_usize(k - 1);
        if other >= true_value {
            Ok(other + 1)
        } else {
            Ok(other)
        }
    }

    /// Unbiased frequency estimator obtained by inverting the k-RR
    /// perturbation matrix.
    ///
    /// Because the user lies uniformly across the *other* `k − 1`
    /// categories (not all `k`), the per-category response distribution is
    ///
    /// ```text
    /// q_c = E[1{report = c}] = p · f_c + (1 − p)/(k − 1) · (1 − f_c)
    ///     = f_c · (p − (1 − p)/(k − 1)) + (1 − p)/(k − 1),
    /// ```
    ///
    /// so the unbiased Horvitz–Thompson estimator is
    ///
    /// ```text
    /// f̂_c = (q_c − (1 − p)/(k − 1)) / (p − (1 − p)/(k − 1)),
    /// where q_c = observed_counts[c] / n_total.
    /// ```
    ///
    /// For the binary case (`k = 2`) this collapses to Warner's original
    /// `(q − (1 − p)) / (2p − 1)`.
    ///
    /// Estimates are *not* clipped to `[0, 1]` so that downstream
    /// aggregate statistics (mean, variance, …) remain unbiased.
    ///
    /// # Errors
    /// - [`FedError::DimensionMismatch`] if `observed_counts.len() ≠ k`.
    /// - [`FedError::EmptyClientList`] if `n_total == 0`.
    /// - [`FedError::InvalidPrivacyBudget`] if `p − (1 − p)/(k − 1)` is
    ///   numerically zero (occurs only at the ε = 0 boundary).
    pub fn aggregate(&self, observed_counts: &[usize], n_total: usize) -> FedResult<Vec<f32>> {
        let k = self.cfg.n_categories;
        if observed_counts.len() != k {
            return Err(FedError::DimensionMismatch {
                expected: k,
                got: observed_counts.len(),
            });
        }
        if n_total == 0 {
            return Err(FedError::EmptyClientList);
        }
        let p = self.p_truth as f64;
        let k_minus_1 = (k - 1) as f64;
        let offset = (1.0 - p) / k_minus_1;
        let denom = p - offset;
        if denom.abs() < 1e-12 {
            return Err(FedError::InvalidPrivacyBudget);
        }
        let n_total_f = n_total as f64;
        let mut estimates = Vec::with_capacity(k);
        for &count in observed_counts {
            let q = count as f64 / n_total_f;
            let f_hat = (q - offset) / denom;
            estimates.push(f_hat as f32);
        }
        Ok(estimates)
    }

    /// Data-independent upper bound on `Var(f̂_c)` for the unbiased
    /// estimator.
    ///
    /// Under the binomial model `count_c ~ Bin(n_total, q_c)` the
    /// variance of the per-category estimator is
    ///
    /// ```text
    /// Var(f̂_c) ≈ q_c · (1 − q_c) / (n_total · (p − (1 − p)/(k − 1))²).
    /// ```
    ///
    /// Since `q_c` is unknown a priori, this method returns the worst-case
    /// `1 / (4 · n_total · (p − (1 − p)/(k − 1))²)` obtained by
    /// maximising `q(1 − q)` at `q = 1/2`.
    ///
    /// # Errors
    /// - [`FedError::EmptyClientList`] if `n_total == 0`.
    /// - [`FedError::InvalidPrivacyBudget`] if the denominator vanishes.
    pub fn variance_per_count(&self, n_total: usize) -> FedResult<f32> {
        if n_total == 0 {
            return Err(FedError::EmptyClientList);
        }
        let p = self.p_truth as f64;
        let k_minus_1 = (self.cfg.n_categories - 1) as f64;
        let denom = p - (1.0 - p) / k_minus_1;
        if denom.abs() < 1e-12 {
            return Err(FedError::InvalidPrivacyBudget);
        }
        let v = 1.0 / (4.0 * n_total as f64 * denom * denom);
        Ok(v as f32)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mech(epsilon: f32, k: usize) -> RandomizedResponse {
        RandomizedResponse::new(RandomizedResponseConfig {
            epsilon,
            n_categories: k,
        })
        .expect("test invariant: valid k-RR mechanism")
    }

    // ── Test 1: p_truth_large_epsilon ────────────────────────────────────────
    #[test]
    fn p_truth_large_epsilon() {
        let m = mech(40.0, 4);
        assert!(
            (m.p_truth() - 1.0).abs() < 1e-5,
            "ε→∞ should give p≈1, got {}",
            m.p_truth()
        );
    }

    // ── Test 2: p_truth_zero_epsilon_limit ──────────────────────────────────
    #[test]
    fn p_truth_zero_epsilon_limit() {
        // ε → 0 ⇒ p → 1/k.
        let m = mech(1e-6, 5);
        assert!(
            (m.p_truth() - 0.2).abs() < 1e-3,
            "ε→0 should give p≈1/k=0.2, got {}",
            m.p_truth()
        );
    }

    // ── Test 3: p_truth_formula ──────────────────────────────────────────────
    #[test]
    fn p_truth_formula() {
        // p = e^ε / (e^ε + k − 1)
        let eps = 2.0_f32;
        let k = 3;
        let m = mech(eps, k);
        let expected = eps.exp() / (eps.exp() + (k as f32 - 1.0));
        assert!((m.p_truth() - expected).abs() < 1e-5);
    }

    // ── Test 4: perturb_returns_valid_category ────────────────────────────────
    #[test]
    fn perturb_returns_valid_category() {
        let m = mech(1.0, 5);
        let mut rng = LcgRng::new(7);
        for _ in 0..1000 {
            let r = m.perturb(2, &mut rng).expect("perturb valid");
            assert!(r < 5, "out of range: {r}");
        }
    }

    // ── Test 5: perturb_high_epsilon_almost_truth ────────────────────────────
    #[test]
    fn perturb_high_epsilon_almost_truth() {
        let m = mech(20.0, 4);
        let mut rng = LcgRng::new(11);
        let mut truthful = 0;
        let n = 1000;
        for _ in 0..n {
            let r = m.perturb(1, &mut rng).expect("perturb");
            if r == 1 {
                truthful += 1;
            }
        }
        // p = e^20 / (e^20 + 3) ≈ 1.0 within f32 precision.
        assert!(
            truthful > 990,
            "with ε=20 should be ≥99% truthful, got {}/{}",
            truthful,
            n
        );
    }

    // ── Test 6: perturb_zero_epsilon_uniform ─────────────────────────────────
    #[test]
    fn perturb_zero_epsilon_uniform() {
        // With ε ≈ 0 the report is uniform across all k categories.
        let m = mech(1e-6, 4);
        let mut rng = LcgRng::new(31);
        let mut counts = [0_usize; 4];
        let n = 8000;
        for _ in 0..n {
            let r = m.perturb(0, &mut rng).expect("perturb");
            counts[r] += 1;
        }
        // Expect ~25% per category; allow ±5% absolute tolerance.
        for c in counts.iter() {
            let frac = *c as f32 / n as f32;
            assert!(
                (frac - 0.25).abs() < 0.05,
                "category fraction off: {frac}, counts={counts:?}"
            );
        }
    }

    // ── Test 7: aggregate_sums_to_one ─────────────────────────────────────────
    #[test]
    fn aggregate_sums_to_one() {
        let m = mech(1.5, 4);
        // Pretend we observed counts [25, 25, 25, 25] across 100 reports.
        let counts = vec![25, 25, 25, 25];
        let est = m.aggregate(&counts, 100).expect("aggregate");
        let s: f32 = est.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-4,
            "aggregate should sum to 1.0, got {s} (est={est:?})"
        );
    }

    // ── Test 8: aggregate_unbiased_recovers_truth ────────────────────────────
    #[test]
    fn aggregate_unbiased_recovers_truth() {
        // Simulate n users with a known true distribution; verify the
        // Horvitz–Thompson estimator recovers it within sampling error.
        let m = mech(2.0, 4);
        let true_dist = [0.4_f32, 0.3, 0.2, 0.1];
        let n_users = 20_000;
        let mut rng = LcgRng::new(101);
        let mut observed = [0_usize; 4];
        let cum: Vec<f32> = {
            let mut acc = 0.0;
            true_dist
                .iter()
                .map(|p| {
                    acc += *p;
                    acc
                })
                .collect()
        };
        for _ in 0..n_users {
            let u = uniform_unit(&mut rng);
            let mut t = 0;
            for (i, &c) in cum.iter().enumerate() {
                if u < c {
                    t = i;
                    break;
                }
            }
            let r = m.perturb(t, &mut rng).expect("perturb");
            observed[r] += 1;
        }
        let counts_vec: Vec<usize> = observed.to_vec();
        let est = m.aggregate(&counts_vec, n_users).expect("aggregate");
        for (e, t) in est.iter().zip(true_dist.iter()) {
            assert!(
                (e - t).abs() < 0.05,
                "estimate {e} far from truth {t}, est={est:?}, obs={observed:?}"
            );
        }
    }

    // ── Test 9: variance_per_count_sane ──────────────────────────────────────
    #[test]
    fn variance_per_count_sane() {
        let m = mech(1.0, 3);
        let v = m.variance_per_count(100).expect("variance");
        assert!(v.is_finite() && v > 0.0);
        // Larger n → smaller variance.
        let v_big = m.variance_per_count(10_000).expect("variance");
        assert!(v_big < v, "variance should decrease with n: {v_big} < {v}");
    }

    // ── Test 10: deterministic_given_seed ────────────────────────────────────
    #[test]
    fn deterministic_given_seed() {
        let m = mech(1.0, 5);
        let mut a = LcgRng::new(2026);
        let mut b = LcgRng::new(2026);
        for _ in 0..200 {
            let ra = m.perturb(3, &mut a).expect("a");
            let rb = m.perturb(3, &mut b).expect("b");
            assert_eq!(ra, rb, "non-deterministic given identical seed");
        }
    }

    // ── Test 11: err_epsilon_non_positive ────────────────────────────────────
    #[test]
    fn err_epsilon_non_positive() {
        assert!(matches!(
            RandomizedResponse::new(RandomizedResponseConfig {
                epsilon: 0.0,
                n_categories: 3,
            }),
            Err(FedError::InvalidPrivacyBudget)
        ));
        assert!(matches!(
            RandomizedResponse::new(RandomizedResponseConfig {
                epsilon: -0.5,
                n_categories: 3,
            }),
            Err(FedError::InvalidPrivacyBudget)
        ));
    }

    // ── Test 12: err_n_categories_too_small ──────────────────────────────────
    #[test]
    fn err_n_categories_too_small() {
        assert!(matches!(
            RandomizedResponse::new(RandomizedResponseConfig {
                epsilon: 1.0,
                n_categories: 1,
            }),
            Err(FedError::InvalidShareCount { .. })
        ));
        assert!(matches!(
            RandomizedResponse::new(RandomizedResponseConfig {
                epsilon: 1.0,
                n_categories: 0,
            }),
            Err(FedError::InvalidShareCount { .. })
        ));
    }

    // ── Test 13: err_true_value_out_of_range ─────────────────────────────────
    #[test]
    fn err_true_value_out_of_range() {
        let m = mech(1.0, 4);
        let mut rng = LcgRng::new(0);
        assert!(matches!(
            m.perturb(4, &mut rng),
            Err(FedError::InvalidShareCount { .. })
        ));
        assert!(matches!(
            m.perturb(100, &mut rng),
            Err(FedError::InvalidShareCount { .. })
        ));
    }

    // ── Test 14: err_observed_counts_wrong_length ────────────────────────────
    #[test]
    fn err_observed_counts_wrong_length() {
        let m = mech(1.0, 4);
        let counts = vec![10, 10, 10]; // k = 3 but mech expects 4
        assert!(matches!(
            m.aggregate(&counts, 30),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    // ── Test 15: err_n_total_zero ────────────────────────────────────────────
    #[test]
    fn err_n_total_zero() {
        let m = mech(1.0, 3);
        let counts = vec![0, 0, 0];
        assert!(matches!(
            m.aggregate(&counts, 0),
            Err(FedError::EmptyClientList)
        ));
        assert!(matches!(
            m.variance_per_count(0),
            Err(FedError::EmptyClientList)
        ));
    }

    // ── Test 16: binary_classic_warner ───────────────────────────────────────
    #[test]
    fn binary_classic_warner() {
        // Original Warner-1965 binary RR: k=2, p = e^ε / (e^ε + 1) = σ(ε).
        let eps = 1.0_f32;
        let m = mech(eps, 2);
        let expected = 1.0 / (1.0 + (-eps).exp());
        assert!((m.p_truth() - expected).abs() < 1e-5);
        // Perturbation must stay in {0, 1}.
        let mut rng = LcgRng::new(42);
        for _ in 0..500 {
            let r = m.perturb(0, &mut rng).expect("perturb");
            assert!(r < 2);
        }
    }

    // ── Test 17: aggregate_all_one_category ──────────────────────────────────
    #[test]
    fn aggregate_all_one_category() {
        // Everyone in category 0; with truthful reports (ε large), q ≈ (1,0,0,0),
        // so f̂_0 ≈ 1 and others ≈ 0.
        let m = mech(10.0, 4);
        let mut rng = LcgRng::new(2027);
        let n = 4000;
        let mut counts = [0_usize; 4];
        for _ in 0..n {
            let r = m.perturb(0, &mut rng).expect("perturb");
            counts[r] += 1;
        }
        let est = m.aggregate(counts.as_ref(), n).expect("aggregate");
        assert!(
            (est[0] - 1.0).abs() < 0.05,
            "f̂_0 should be ~1, got {} (counts={counts:?})",
            est[0]
        );
        for (i, e) in est.iter().enumerate().skip(1) {
            assert!(
                e.abs() < 0.05,
                "f̂_{} should be ~0, got {e} (counts={counts:?})",
                i
            );
        }
    }

    // ── Test 18: aggregate_output_length_matches_k ───────────────────────────
    #[test]
    fn aggregate_output_length_matches_k() {
        for k in 2..=8 {
            let m = mech(1.0, k);
            let counts = vec![10_usize; k];
            let est = m.aggregate(&counts, 10 * k).expect("aggregate");
            assert_eq!(est.len(), k, "len for k={k}");
        }
    }

    // ── Test 19: epsilon_non_finite_rejected ─────────────────────────────────
    #[test]
    fn epsilon_non_finite_rejected() {
        assert!(matches!(
            RandomizedResponse::new(RandomizedResponseConfig {
                epsilon: f32::INFINITY,
                n_categories: 3,
            }),
            Err(FedError::InvalidPrivacyBudget)
        ));
        assert!(matches!(
            RandomizedResponse::new(RandomizedResponseConfig {
                epsilon: f32::NAN,
                n_categories: 3,
            }),
            Err(FedError::InvalidPrivacyBudget)
        ));
    }

    // ── Test 20: accessor_fields ─────────────────────────────────────────────
    #[test]
    fn accessor_fields() {
        let m = mech(2.5, 7);
        assert!((m.epsilon() - 2.5).abs() < 1e-6);
        assert_eq!(m.n_categories(), 7);
    }
}
