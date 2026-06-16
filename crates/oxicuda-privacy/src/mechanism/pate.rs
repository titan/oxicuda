//! PATE — Private Aggregation of Teacher Ensembles.
//!
//! References:
//! - Papernot, Abadi, Erlingsson, Goodfellow, Talwar (2017), "Semi-supervised
//!   Knowledge Transfer for Deep Learning from Private Training Data", ICLR.
//! - Papernot, Song, Mironov, Raghunathan, Talwar, Erlingsson (2018),
//!   "Scalable Private Learning with PATE", ICLR.
//!
//! # Setup
//! An ensemble of `T` teacher models is trained on disjoint partitions of the
//! private data.  For an unlabelled query, each teacher votes a class label in
//! `0..num_classes`.  PATE releases a privacy-preserving aggregate label by
//! adding noise to the vote histogram and reporting the (noisy) argmax.
//!
//! # Vote histogram sensitivity
//! Let `n_j = #{teachers voting class j}`.  Changing a single teacher's vote
//! moves it from one bin to another, changing the histogram by `+1` in one bin
//! and `-1` in another, so the L1 sensitivity is `2` (it is `0` if the teacher
//! does not change its vote, and exactly `2` when it switches classes).
//!
//! # Reporting modes
//! - **LNMax** (Laplace NoisyMax, Papernot 2017): add `Lap(2/ε)` to each tally
//!   and report the argmax; this yields an `(ε, 0)`-DP release of one label.
//!   Equivalently, with a precision parameter `γ` the noise scale is `1/γ`
//!   (so that `ε = 2γ` gives the per-query budget).
//! - **GNMax** (Gaussian NoisyMax, Papernot 2018): add `N(0, σ²)` to each tally
//!   and report the argmax.  Gaussian noise enables tighter Rényi-DP / data
//!   dependent accounting.
//!
//! # Confident-GNMax (Papernot 2018, §4)
//! To save privacy budget on unanimous queries, only answer when the plurality
//! vote is large enough.  Draw a noisy plurality count
//! `max_j n_j + N(0, σ₁²)`; if it meets a threshold `T`, release the GNMax
//! answer with noise `σ₂`, otherwise **abstain** (return `None`).

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Noise mechanism used by the PATE aggregator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PateMechanism {
    /// Laplace NoisyMax (LNMax): add `Lap(0, scale)` to each tally.
    ///
    /// For the `(ε, 0)`-DP guarantee with histogram L1 sensitivity `2`, set
    /// `scale = 2 / ε`.  With a precision parameter `γ`, set `scale = 1 / γ`.
    Laplace {
        /// Laplace noise scale `b > 0`.
        scale: f64,
    },
    /// Gaussian NoisyMax (GNMax): add `N(0, σ²)` to each tally.
    Gaussian {
        /// Gaussian noise standard deviation `σ > 0`.
        sigma: f64,
    },
}

/// Configuration for the PATE noisy-argmax aggregator.
#[derive(Debug, Clone)]
pub struct PateConfig {
    /// Number of classes `C ≥ 2`.
    pub num_classes: usize,
    /// Noise mechanism (Laplace or Gaussian).
    pub mechanism: PateMechanism,
}

impl PateConfig {
    /// Construct and validate a `PateConfig`.
    ///
    /// # Errors
    /// Returns `InvalidParameter` if `num_classes < 2`, or if the mechanism's
    /// `scale`/`sigma` is not strictly positive (or non-finite).
    pub fn new(num_classes: usize, mechanism: PateMechanism) -> PrivacyResult<Self> {
        if num_classes < 2 {
            return Err(PrivacyError::InvalidParameter(format!(
                "num_classes must be ≥ 2, got {num_classes}"
            )));
        }
        match mechanism {
            PateMechanism::Laplace { scale } => {
                if scale <= 0.0 || scale.is_nan() {
                    return Err(PrivacyError::InvalidParameter(format!(
                        "Laplace scale must be positive, got {scale}"
                    )));
                }
            }
            PateMechanism::Gaussian { sigma } => {
                if sigma <= 0.0 || sigma.is_nan() {
                    return Err(PrivacyError::InvalidParameter(format!(
                        "Gaussian sigma must be positive, got {sigma}"
                    )));
                }
            }
        }
        Ok(Self {
            num_classes,
            mechanism,
        })
    }
}

/// Configuration for the Confident-GNMax data-dependent aggregator.
#[derive(Debug, Clone)]
pub struct ConfidentGnmaxConfig {
    /// Number of classes `C ≥ 2`.
    pub num_classes: usize,
    /// Confidence threshold `T` on the noisy plurality count.
    pub threshold: f64,
    /// Std `σ₁ > 0` of the noise added to the plurality count for the check.
    pub sigma_threshold: f64,
    /// Std `σ₂ > 0` of the GNMax noise used to report the answer.
    pub sigma_answer: f64,
}

impl ConfidentGnmaxConfig {
    /// Construct and validate a `ConfidentGnmaxConfig`.
    ///
    /// # Errors
    /// Returns `InvalidParameter` if `num_classes < 2`, either sigma is not
    /// strictly positive, or `threshold` is not finite.
    pub fn new(
        num_classes: usize,
        threshold: f64,
        sigma_threshold: f64,
        sigma_answer: f64,
    ) -> PrivacyResult<Self> {
        if num_classes < 2 {
            return Err(PrivacyError::InvalidParameter(format!(
                "num_classes must be ≥ 2, got {num_classes}"
            )));
        }
        if !threshold.is_finite() {
            return Err(PrivacyError::InvalidParameter(format!(
                "threshold must be finite, got {threshold}"
            )));
        }
        if sigma_threshold <= 0.0 || sigma_threshold.is_nan() {
            return Err(PrivacyError::InvalidParameter(format!(
                "sigma_threshold must be positive, got {sigma_threshold}"
            )));
        }
        if sigma_answer <= 0.0 || sigma_answer.is_nan() {
            return Err(PrivacyError::InvalidParameter(format!(
                "sigma_answer must be positive, got {sigma_answer}"
            )));
        }
        Ok(Self {
            num_classes,
            threshold,
            sigma_threshold,
            sigma_answer,
        })
    }
}

/// Sample `Lap(0, scale)` via the inverse-CDF method (crate idiom).
fn laplace_sample(scale: f64, rng: &mut LcgRng) -> f64 {
    let u = rng.next_f64() - 0.5;
    -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
}

/// Tally teacher votes into a class histogram.
///
/// Returns `counts` where `counts[j]` is the number of teachers voting class
/// `j`.
///
/// # Errors
/// - `EmptyInput` if `votes` is empty.
/// - `IndexOutOfRange` if any vote is `≥ num_classes`.
pub fn tally_votes(votes: &[usize], num_classes: usize) -> PrivacyResult<Vec<usize>> {
    if votes.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    let mut counts = vec![0usize; num_classes];
    for &v in votes {
        if v >= num_classes {
            return Err(PrivacyError::IndexOutOfRange(v, num_classes));
        }
        counts[v] += 1;
    }
    Ok(counts)
}

/// Return the argmax of `noisy` tallies, breaking ties by lowest index.
fn noisy_argmax(noisy: &[f64]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f64::NEG_INFINITY;
    for (i, &v) in noisy.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

/// PATE noisy-argmax aggregation (LNMax or GNMax depending on `cfg`).
///
/// Tallies the votes, adds independent noise (Laplace or Gaussian) to each
/// class count, and returns the argmax class.  Ties (after noise) are broken
/// by the lowest index.
///
/// # Errors
/// - `EmptyInput` if `votes` is empty.
/// - `IndexOutOfRange` if any vote is `≥ cfg.num_classes`.
/// - `InvalidParameter` if `cfg`'s noise parameter is invalid.
pub fn pate_aggregate(votes: &[usize], cfg: &PateConfig, rng: &mut LcgRng) -> PrivacyResult<usize> {
    let counts = tally_votes(votes, cfg.num_classes)?;
    let mut noisy = vec![0.0f64; counts.len()];
    match cfg.mechanism {
        PateMechanism::Laplace { scale } => {
            if scale <= 0.0 || scale.is_nan() {
                return Err(PrivacyError::InvalidParameter(format!(
                    "Laplace scale must be positive, got {scale}"
                )));
            }
            for (i, &c) in counts.iter().enumerate() {
                noisy[i] = c as f64 + laplace_sample(scale, rng);
            }
        }
        PateMechanism::Gaussian { sigma } => {
            if sigma <= 0.0 || sigma.is_nan() {
                return Err(PrivacyError::InvalidParameter(format!(
                    "Gaussian sigma must be positive, got {sigma}"
                )));
            }
            for (i, &c) in counts.iter().enumerate() {
                noisy[i] = c as f64 + rng.normal_pair().0 * sigma;
            }
        }
    }
    Ok(noisy_argmax(&noisy))
}

/// Confident-GNMax: only answer when the plurality vote is large enough.
///
/// Draws a noisy plurality count `max_j n_j + N(0, σ₁²)`; if it meets the
/// threshold `T`, releases a GNMax answer (noise `σ₂`), otherwise abstains.
///
/// Returns `Some(label)` when confident, else `None`.
///
/// # Errors
/// - `EmptyInput` if `votes` is empty.
/// - `IndexOutOfRange` if any vote is `≥ cfg.num_classes`.
pub fn confident_gnmax(
    votes: &[usize],
    cfg: &ConfidentGnmaxConfig,
    rng: &mut LcgRng,
) -> PrivacyResult<Option<usize>> {
    let counts = tally_votes(votes, cfg.num_classes)?;
    let max_count = counts.iter().copied().max().unwrap_or(0) as f64;
    // Noisy plurality check.
    let noisy_plurality = max_count + rng.normal_pair().0 * cfg.sigma_threshold;
    if noisy_plurality < cfg.threshold {
        return Ok(None);
    }
    // Confident: release a GNMax answer with the answer noise.
    let mut noisy = vec![0.0f64; counts.len()];
    for (i, &c) in counts.iter().enumerate() {
        noisy[i] = c as f64 + rng.normal_pair().0 * cfg.sigma_answer;
    }
    Ok(Some(noisy_argmax(&noisy)))
}

/// Consensus fraction: `max_count / total` — the share of teachers agreeing
/// with the plurality class.  This is a *non-private* utility diagnostic.
///
/// # Errors
/// - `EmptyInput` if `votes` is empty.
/// - `IndexOutOfRange` if any vote is `≥ num_classes`.
pub fn consensus(votes: &[usize], num_classes: usize) -> PrivacyResult<f64> {
    let counts = tally_votes(votes, num_classes)?;
    let total: usize = counts.iter().sum();
    if total == 0 {
        return Err(PrivacyError::EmptyInput);
    }
    let max_count = counts.iter().copied().max().unwrap_or(0);
    Ok(max_count as f64 / total as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tally_correctness() {
        let votes = vec![0, 1, 1, 2, 2, 2];
        let counts = tally_votes(&votes, 3).expect("ok");
        assert_eq!(counts, vec![1, 2, 3]);
    }

    #[test]
    fn test_tally_rejects_empty() {
        assert!(tally_votes(&[], 3).is_err());
    }

    #[test]
    fn test_tally_rejects_out_of_range() {
        let votes = vec![0, 1, 3];
        assert!(tally_votes(&votes, 3).is_err());
    }

    #[test]
    fn test_aggregate_valid_index_laplace() {
        let votes = vec![0, 1, 1, 2, 2, 2, 0, 1];
        let cfg = PateConfig::new(3, PateMechanism::Laplace { scale: 1.0 }).expect("ok");
        let mut rng = LcgRng::new(123);
        for _ in 0..200 {
            let label = pate_aggregate(&votes, &cfg, &mut rng).expect("ok");
            assert!(label < 3);
        }
    }

    #[test]
    fn test_aggregate_valid_index_gaussian() {
        let votes = vec![0, 1, 1, 2, 2, 2, 0, 1];
        let cfg = PateConfig::new(3, PateMechanism::Gaussian { sigma: 1.0 }).expect("ok");
        let mut rng = LcgRng::new(321);
        for _ in 0..200 {
            let label = pate_aggregate(&votes, &cfg, &mut rng).expect("ok");
            assert!(label < 3);
        }
    }

    #[test]
    fn test_strong_consensus_laplace() {
        // 95 teachers vote class 2, 5 split among others.
        let mut votes = vec![2usize; 95];
        votes.extend([0, 0, 1, 1, 3]);
        let cfg = PateConfig::new(4, PateMechanism::Laplace { scale: 0.5 }).expect("ok");
        let mut rng = LcgRng::new(7);
        let mut correct = 0usize;
        for _ in 0..200 {
            if pate_aggregate(&votes, &cfg, &mut rng).expect("ok") == 2 {
                correct += 1;
            }
        }
        assert!(correct > 190, "expected >190 correct, got {correct}");
    }

    #[test]
    fn test_strong_consensus_gaussian() {
        let mut votes = vec![2usize; 95];
        votes.extend([0, 0, 1, 1, 3]);
        let cfg = PateConfig::new(4, PateMechanism::Gaussian { sigma: 0.5 }).expect("ok");
        let mut rng = LcgRng::new(11);
        let mut correct = 0usize;
        for _ in 0..200 {
            if pate_aggregate(&votes, &cfg, &mut rng).expect("ok") == 2 {
                correct += 1;
            }
        }
        assert!(correct > 190, "expected >190 correct, got {correct}");
    }

    #[test]
    fn test_consensus_fraction() {
        let mut votes = vec![2usize; 95];
        votes.extend([0, 0, 1, 1, 3]);
        let c = consensus(&votes, 4).expect("ok");
        assert!((c - 0.95).abs() < 1e-12, "expected 0.95, got {c}");
    }

    #[test]
    fn test_consensus_rejects_empty() {
        assert!(consensus(&[], 3).is_err());
    }

    #[test]
    fn test_confident_gnmax_high_consensus_answers() {
        // 90/100 vote class 1; threshold well below 90.
        let mut votes = vec![1usize; 90];
        votes.extend(vec![0usize; 10]);
        let cfg = ConfidentGnmaxConfig::new(2, 50.0, 1.0, 0.5).expect("ok");
        let mut rng = LcgRng::new(99);
        let mut answered_correct = 0usize;
        for _ in 0..200 {
            let answer = confident_gnmax(&votes, &cfg, &mut rng).expect("ok");
            if answer == Some(1) {
                answered_correct += 1;
            }
        }
        // Should answer and be correct on the overwhelming majority of trials.
        assert!(
            answered_correct > 190,
            "expected >190 correct answers, got {answered_correct}"
        );
    }

    #[test]
    fn test_confident_gnmax_threshold_above_max_abstains() {
        // 90 teachers max; threshold 200 with tiny noise => never confident.
        let mut votes = vec![1usize; 90];
        votes.extend(vec![0usize; 10]);
        let cfg = ConfidentGnmaxConfig::new(2, 200.0, 1.0, 0.5).expect("ok");
        let mut rng = LcgRng::new(5);
        for _ in 0..200 {
            let out = confident_gnmax(&votes, &cfg, &mut rng).expect("ok");
            assert!(out.is_none(), "expected abstain (None), got {out:?}");
        }
    }

    #[test]
    fn test_config_rejects_num_classes() {
        assert!(PateConfig::new(1, PateMechanism::Laplace { scale: 1.0 }).is_err());
        assert!(ConfidentGnmaxConfig::new(1, 1.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn test_config_rejects_nonpositive_noise() {
        assert!(PateConfig::new(3, PateMechanism::Laplace { scale: 0.0 }).is_err());
        assert!(PateConfig::new(3, PateMechanism::Laplace { scale: -1.0 }).is_err());
        assert!(PateConfig::new(3, PateMechanism::Gaussian { sigma: 0.0 }).is_err());
        assert!(PateConfig::new(3, PateMechanism::Gaussian { sigma: -2.0 }).is_err());
        assert!(ConfidentGnmaxConfig::new(2, 1.0, 0.0, 1.0).is_err());
        assert!(ConfidentGnmaxConfig::new(2, 1.0, 1.0, 0.0).is_err());
        assert!(ConfidentGnmaxConfig::new(2, f64::NAN, 1.0, 1.0).is_err());
    }

    #[test]
    fn test_determinism_same_seed() {
        let votes = vec![0, 1, 1, 2, 2, 2, 0, 1];
        let cfg = PateConfig::new(3, PateMechanism::Gaussian { sigma: 1.5 }).expect("ok");
        let mut rng_a = LcgRng::new(2024);
        let mut rng_b = LcgRng::new(2024);
        for _ in 0..50 {
            let a = pate_aggregate(&votes, &cfg, &mut rng_a).expect("ok");
            let b = pate_aggregate(&votes, &cfg, &mut rng_b).expect("ok");
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_aggregate_rejects_out_of_range() {
        let votes = vec![0, 1, 5];
        let cfg = PateConfig::new(3, PateMechanism::Laplace { scale: 1.0 }).expect("ok");
        let mut rng = LcgRng::new(1);
        assert!(pate_aggregate(&votes, &cfg, &mut rng).is_err());
    }
}
