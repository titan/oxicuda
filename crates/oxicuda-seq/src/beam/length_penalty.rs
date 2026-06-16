//! GNMT-style length penalty and coverage penalty for beam search scoring.
//!
//! Reference: Wu et al. 2016, "Google's Neural Machine Translation System:
//! Bridging the Gap between Human and Machine Translation", arXiv:1609.08144.
//!
//! Length penalty:     lp(y) = ((5 + |y|) / 6)^α
//! Coverage penalty:   cp    = Σ_i log(min(Σ_t p_{t,i}, 1.0))
//! Combined score:     score = log_prob / lp(|y|) - β * |cp|

use crate::error::{SeqError, SeqResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for GNMT-style length and coverage penalties.
#[derive(Debug, Clone)]
pub struct LengthPenaltyConfig {
    /// Length-penalty exponent α.  0 = disabled; typical values 0.6–1.0.
    pub alpha: f64,
    /// Coverage-penalty weight β.  0 = disabled.
    pub beta: f64,
    /// Minimum output length (informational; not enforced by score()).
    pub min_length: usize,
    /// Maximum output length (informational; not enforced by score()).
    pub max_length: usize,
}

// ─── LengthPenalty ────────────────────────────────────────────────────────────

/// Computes GNMT-style length and coverage penalties for beam-search hypothesis scoring.
#[derive(Debug, Clone)]
pub struct LengthPenalty {
    config: LengthPenaltyConfig,
}

impl LengthPenalty {
    /// Create a new `LengthPenalty`.  Returns `Err` if `alpha < 0` or `beta < 0`.
    pub fn new(config: LengthPenaltyConfig) -> SeqResult<Self> {
        if config.alpha < 0.0 {
            return Err(SeqError::InvalidParameter {
                name: "alpha".into(),
                value: config.alpha,
            });
        }
        if config.beta < 0.0 {
            return Err(SeqError::InvalidParameter {
                name: "beta".into(),
                value: config.beta,
            });
        }
        if config.max_length == 0 {
            return Err(SeqError::InvalidConfiguration(
                "max_length must be > 0".into(),
            ));
        }
        Ok(Self { config })
    }

    // ── Core penalty functions ────────────────────────────────────────────────

    /// GNMT length penalty: `((5 + length) / (5 + 1))^alpha`.
    ///
    /// At `length=1`: returns 1.0.
    /// Monotonically increasing for `alpha > 0`.
    #[inline]
    pub fn lp(&self, length: usize) -> f64 {
        let ratio = (5.0 + length as f64) / 6.0;
        ratio.powf(self.config.alpha)
    }

    /// Coverage penalty: `Σ_i log(min(Σ_t p_{t,i}, 1.0))`.
    ///
    /// `coverage_probs` has layout `[seq_len × n_source]`
    /// (for each target step `t`, attention over `n_source` source tokens).
    ///
    /// Returns 0.0 when all source tokens are fully covered (sum ≥ 1.0).
    /// Returns a negative value when coverage is partial.
    pub fn cp(&self, coverage_probs: &[f64], n_source: usize, seq_len: usize) -> f64 {
        if n_source == 0 || seq_len == 0 || coverage_probs.is_empty() {
            return 0.0;
        }
        // Accumulate Σ_t p_{t,i} for each source position i.
        let mut coverage = vec![0.0f64; n_source];
        for t in 0..seq_len {
            for i in 0..n_source {
                let idx = t * n_source + i;
                if idx < coverage_probs.len() {
                    coverage[i] += coverage_probs[idx];
                }
            }
        }
        // cp = Σ_i log(min(coverage_i, 1.0))
        let mut penalty = 0.0;
        for i in 0..n_source {
            penalty += coverage[i].min(1.0).ln();
        }
        penalty
    }

    /// Combined beam-search score.
    ///
    /// `score = log_prob / lp(length) - beta * |cp|`
    ///
    /// The magnitude of `cp` is used so that beta ≥ 0 always penalises
    /// under-coverage (cp is ≤ 0 when coverage < 1).
    pub fn score(
        &self,
        log_prob: f64,
        length: usize,
        coverage_probs: &[f64],
        n_source: usize,
    ) -> SeqResult<f64> {
        if !log_prob.is_finite() {
            return Err(SeqError::NumericalInstability(
                "log_prob is not finite".into(),
            ));
        }
        let lp = self.lp(length);
        let cp_val = self.cp(coverage_probs, n_source, length);
        Ok(log_prob / lp - self.config.beta * cp_val.abs())
    }

    /// Rank hypotheses by descending combined score (no coverage penalty applied
    /// in this simplified batch form — coverage is assumed uniform).
    ///
    /// Returns indices sorted best-first.
    pub fn rank(&self, log_probs: &[f64], lengths: &[usize]) -> Vec<usize> {
        if log_probs.is_empty() {
            return Vec::new();
        }
        let n = log_probs.len().min(lengths.len());
        let scores: Vec<f64> = (0..n)
            .map(|i| {
                let lp = self.lp(lengths[i]);
                log_probs[i] / lp
            })
            .collect();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_lp(alpha: f64, beta: f64) -> LengthPenalty {
        LengthPenalty::new(LengthPenaltyConfig {
            alpha,
            beta,
            min_length: 1,
            max_length: 200,
        })
        .expect("LengthPenalty::new failed")
    }

    #[test]
    fn lp_at_length_1() {
        // lp(1) = ((5+1)/(5+1))^alpha = 1.0 for any alpha
        for &alpha in &[0.0, 0.5, 1.0, 2.0] {
            let lp = make_lp(alpha, 0.0);
            let val = lp.lp(1);
            assert!(
                (val - 1.0).abs() < 1e-12,
                "lp(1) should be 1.0 for alpha={alpha}, got {val}"
            );
        }
    }

    #[test]
    fn lp_increases_with_length() {
        let lp = make_lp(0.8, 0.0);
        assert!(
            lp.lp(10) > lp.lp(5),
            "lp(10)={} should be > lp(5)={} for alpha=0.8",
            lp.lp(10),
            lp.lp(5)
        );
    }

    #[test]
    fn alpha_zero_lp_one() {
        let lp = make_lp(0.0, 0.0);
        for length in [1, 5, 10, 100] {
            let val = lp.lp(length);
            assert!(
                (val - 1.0).abs() < 1e-12,
                "alpha=0: lp({length}) should be 1.0, got {val}"
            );
        }
    }

    #[test]
    fn cp_zero_when_full_coverage() {
        let lp = make_lp(0.6, 0.1);
        let n_source = 3;
        let seq_len = 3;
        // Each target step attends uniformly: row sums to 1/3 each → column sum = 1.0
        let coverage_probs = vec![1.0 / 3.0; n_source * seq_len];
        let cp = lp.cp(&coverage_probs, n_source, seq_len);
        assert!(
            cp.abs() < 1e-10,
            "cp should be ~0 for full coverage, got {cp}"
        );
    }

    #[test]
    fn cp_negative_for_under_coverage() {
        let lp = make_lp(0.6, 0.1);
        let n_source = 4;
        let seq_len = 2;
        // Each step attends only to first source position → positions 1-3 get 0
        let mut coverage_probs = vec![0.0f64; n_source * seq_len];
        for t in 0..seq_len {
            coverage_probs[t * n_source] = 0.3; // only position 0
        }
        let cp = lp.cp(&coverage_probs, n_source, seq_len);
        assert!(cp < 0.0, "under-coverage should give negative cp, got {cp}");
    }

    #[test]
    fn score_penalizes_short() {
        // For high alpha, a longer sequence with the same total log-prob per token
        // should get a higher score (penalty reduces for longer sequences).
        let lp = make_lp(1.0, 0.0);
        let empty_cov: &[f64] = &[];
        let _short = lp.score(-10.0, 5, empty_cov, 0).expect("score short");
        let _long = lp.score(-20.0, 15, empty_cov, 0).expect("score long");
        // short log_prob per token = -2.0/tok, long = -20/15 ≈ -1.33
        // After lp division: short = -10 / lp(5), long = -20 / lp(15)
        // With alpha=1: lp(5)=(10/6)=1.667, lp(15)=(20/6)=3.333
        // short_score = -10/1.667 ≈ -6.0, long_score = -20/3.333 ≈ -6.0 → both ~equal
        // Use a cleaner example: long has much better per-token log_prob
        let better_long = lp.score(-6.0, 20, empty_cov, 0).expect("score better_long");
        let worse_short = lp.score(-10.0, 3, empty_cov, 0).expect("score worse_short");
        // better_long: -6.0 / lp(20) = -6.0 / (25/6) = -6 * 6/25 = -1.44
        // worse_short: -10.0 / lp(3) = -10.0 / (8/6) = -10 * 6/8 = -7.5
        assert!(
            better_long > worse_short,
            "better_long_score={better_long:.4} should > worse_short_score={worse_short:.4}"
        );
    }

    #[test]
    fn rank_returns_correct_order() {
        let lp = make_lp(0.6, 0.0);
        // Candidate 0: log_prob=-5, len=5  → score = -5/lp(5)
        // Candidate 1: log_prob=-2, len=3  → score = -2/lp(3)  (best)
        // Candidate 2: log_prob=-15, len=20 → score = -15/lp(20) (worst)
        let log_probs = [-5.0, -2.0, -15.0];
        let lengths = [5, 3, 20];
        let order = lp.rank(&log_probs, &lengths);
        assert_eq!(order[0], 1, "best candidate should be index 1");
        assert_eq!(order[2], 2, "worst candidate should be index 2");
    }

    #[test]
    fn max_length_exceeded_score_no_panic() {
        // score() should work even for length > max_length
        let lp = LengthPenalty::new(LengthPenaltyConfig {
            alpha: 0.6,
            beta: 0.0,
            min_length: 1,
            max_length: 10,
        })
        .expect("new");
        let result = lp.score(-5.0, 50, &[], 0);
        assert!(
            result.is_ok(),
            "score should not fail for length > max_length"
        );
    }

    #[test]
    fn beta_zero_no_coverage_penalty() {
        let lp = make_lp(0.6, 0.0); // beta=0
        // With beta=0, coverage term is 0 → score = log_prob / lp(len)
        let n_source = 3;
        let coverage_probs = vec![0.1f64; n_source * 5]; // partial coverage
        let s = lp.score(-8.0, 5, &coverage_probs, n_source).expect("score");
        let expected = -8.0 / lp.lp(5);
        assert!(
            (s - expected).abs() < 1e-12,
            "beta=0: score should be log_prob/lp, expected={expected}, got={s}"
        );
    }
}
