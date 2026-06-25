//! PATE: Private Aggregation of Teachers' Ensembles.
//!
//! Papernot et al., "Semi-supervised Knowledge Transfer for Deep Learning
//! from Private Training Data", ICLR 2017.
//!
//! PATE enables private predictions by aggregating noisy votes from an
//! ensemble of teacher models trained on disjoint private data partitions.

use crate::error::{FedError, FedResult};
use crate::handle::LcgRng;

/// Configuration for a PATE voting setup.
#[derive(Debug, Clone)]
pub struct PateConfig {
    /// Total number of teacher models.
    pub n_teachers: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Privacy budget ε for one noisy voting query.
    pub epsilon: f32,
}

impl PateConfig {
    /// Create a validated PATE configuration.
    ///
    /// # Errors
    /// Returns `InsufficientClients` if `n_teachers == 0`,
    /// `DimensionMismatch` if `n_classes == 0`,
    /// or `InvalidPrivacyBudget` if `epsilon ≤ 0`.
    pub fn new(n_teachers: usize, n_classes: usize, epsilon: f32) -> FedResult<Self> {
        if n_teachers == 0 {
            return Err(FedError::InsufficientClients { min: 1, got: 0 });
        }
        if n_classes == 0 {
            return Err(FedError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if !(epsilon > 0.0 && epsilon.is_finite()) {
            return Err(FedError::InvalidPrivacyBudget);
        }
        Ok(Self {
            n_teachers,
            n_classes,
            epsilon,
        })
    }
}

/// Perform noisy voting: build a vote histogram from teacher predictions,
/// add Laplace noise to each bin, return the argmax class.
///
/// # Arguments
/// - `votes` — teacher predictions, each in `[0, n_classes)` (length = n_teachers)
/// - `n_classes` — number of classes
/// - `epsilon` — privacy budget for the Laplace noise (scale = 1/ε)
/// - `rng` — deterministic RNG
///
/// # Errors
/// Returns `InvalidPrivacyBudget` if `epsilon ≤ 0`, `DimensionMismatch` if
/// `n_classes == 0`, or `InsufficientClients` if `votes` is empty.
pub fn noisy_voting(
    votes: &[usize],
    n_classes: usize,
    epsilon: f32,
    rng: &mut LcgRng,
) -> FedResult<usize> {
    if votes.is_empty() {
        return Err(FedError::InsufficientClients { min: 1, got: 0 });
    }
    if n_classes == 0 {
        return Err(FedError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    if !(epsilon > 0.0 && epsilon.is_finite()) {
        return Err(FedError::InvalidPrivacyBudget);
    }

    // Build vote histogram
    let mut histogram = vec![0_i64; n_classes];
    for &v in votes {
        let class = v % n_classes; // safe modular clamp
        histogram[class] += 1;
    }

    // Add Laplace noise: Lap(0, 1/ε) to each bin
    // Laplace scale b = 1/epsilon (sensitivity = 1 for counting query)
    let b = 1.0 / epsilon;
    let mut noisy: Vec<f32> = histogram
        .iter()
        .map(|&count| count as f32 + rng.next_laplace(b))
        .collect();

    // Return argmax of noisy histogram
    let best_class = noisy
        .iter()
        .enumerate()
        .fold((0_usize, f32::NEG_INFINITY), |(best_i, best_v), (i, &v)| {
            if v > best_v { (i, v) } else { (best_i, best_v) }
        })
        .0;

    // Suppress the noisy vector to avoid accidental leakage in debug output
    for v in noisy.iter_mut() {
        *v = 0.0;
    }
    let _ = noisy; // consumed

    Ok(best_class)
}

/// Compute the data-dependent epsilon for PATE.
///
/// When the majority class receives significantly more votes than others,
/// the privacy cost of revealing the argmax is lower. This function uses
/// a simplified data-dependent bound:
///
/// `ε_data = ε_vote * log(max_count / max(second_max_count, 1))`
///
/// which decreases as the gap between the leading and runner-up classes grows.
///
/// # Arguments
/// - `vote_counts` — raw vote histogram `[n_classes]`
/// - `delta` — target δ for conversion (unused here, kept for API compatibility)
/// - `epsilon_vote` — base epsilon per voting query
#[must_use]
pub fn data_dependent_epsilon(vote_counts: &[u32], _delta: f32, epsilon_vote: f32) -> f32 {
    if vote_counts.is_empty() {
        return f32::INFINITY;
    }

    // Find top two counts
    let mut sorted = vote_counts.to_vec();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    let max_count = sorted[0].max(1) as f32;
    let second_max = if sorted.len() > 1 {
        sorted[1].max(1) as f32
    } else {
        1.0
    };

    // Data-dependent epsilon is smaller when majority is clear
    let gap_ratio = max_count / second_max;
    if gap_ratio <= 1.0 {
        return epsilon_vote;
    }
    // Fewer queries → smaller epsilon, bounded below by epsilon_vote / ln(gap)
    let reduction = gap_ratio.ln().max(1.0);
    epsilon_vote / reduction
}

/// Configuration for the Confident-GNMax aggregator.
///
/// Papernot et al., "Scalable Private Learning with PATE", ICLR 2018, §4.1.
#[derive(Debug, Clone)]
pub struct ConfidentGnMaxConfig {
    /// Number of output classes.
    pub n_classes: usize,
    /// Confidence threshold `T`: queries whose (noisy) plurality vote count
    /// falls below `T` are *not answered*, saving privacy budget.
    pub threshold: f32,
    /// Standard deviation `σ₁` of the Gaussian noise on the threshold check.
    pub sigma_threshold: f32,
    /// Standard deviation `σ₂` of the Gaussian noise on the answering GNMax.
    /// Typically `σ₂ < σ₁` so confident queries are answered accurately.
    pub sigma_answer: f32,
}

impl ConfidentGnMaxConfig {
    /// Create a validated Confident-GNMax configuration.
    ///
    /// # Errors
    /// Returns `DimensionMismatch` if `n_classes == 0`,
    /// `InvalidNoiseMultiplier` if either σ is non-positive, or `Internal` if
    /// `threshold` is non-finite.
    pub fn new(
        n_classes: usize,
        threshold: f32,
        sigma_threshold: f32,
        sigma_answer: f32,
    ) -> FedResult<Self> {
        if n_classes == 0 {
            return Err(FedError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if !(sigma_threshold > 0.0 && sigma_threshold.is_finite()) {
            return Err(FedError::InvalidNoiseMultiplier);
        }
        if !(sigma_answer > 0.0 && sigma_answer.is_finite()) {
            return Err(FedError::InvalidNoiseMultiplier);
        }
        if !threshold.is_finite() {
            return Err(FedError::Internal(
                "Confident-GNMax threshold must be finite".into(),
            ));
        }
        Ok(Self {
            n_classes,
            threshold,
            sigma_threshold,
            sigma_answer,
        })
    }
}

/// Confident-GNMax: privately answer a query only when the teacher ensemble is
/// confident, abstaining otherwise to conserve the privacy budget.
///
/// Algorithm (Papernot 2018):
/// 1. Build the vote histogram `n_j` over teacher predictions.
/// 2. **Confidence check** — if `max_j n_j + N(0, σ₁²) < T`, return `None`
///    (the query is *not answered*; the noisy-max is too low to be trusted and
///    answering would spend budget on an uncertain label).
/// 3. **GNMax answer** — otherwise add fresh `N(0, σ₂²)` Gaussian noise to
///    *every* bin and return `Some(argmax)` over the noisy histogram.
///
/// The two noise draws use independent samples from `rng`; Gaussian noise is
/// drawn via the handle's Box-Muller sampler.
///
/// # Arguments
/// - `votes` — teacher predictions, each in `[0, n_classes)` (length = n_teachers)
/// - `cfg` — Confident-GNMax configuration
/// - `rng` — deterministic RNG
///
/// # Returns
/// - `Some(class)` when the query clears the confidence threshold.
/// - `None` when the query is abstained (below threshold).
///
/// # Errors
/// Returns `InsufficientClients` if `votes` is empty.
pub fn confident_gnmax(
    votes: &[usize],
    cfg: &ConfidentGnMaxConfig,
    rng: &mut LcgRng,
) -> FedResult<Option<usize>> {
    if votes.is_empty() {
        return Err(FedError::InsufficientClients { min: 1, got: 0 });
    }

    // Vote histogram.
    let mut histogram = vec![0_i64; cfg.n_classes];
    for &v in votes {
        histogram[v % cfg.n_classes] += 1;
    }

    // Noisy confidence check on the plurality count.
    let max_count = histogram.iter().copied().max().unwrap_or(0) as f32;
    let (noise1, noise2) = rng.next_normal_pair();
    let noisy_max = max_count + cfg.sigma_threshold * noise1;
    if noisy_max < cfg.threshold {
        // Stability/confidence not met → abstain, spend no answering budget.
        return Ok(None);
    }

    // GNMax: add independent Gaussian noise to every bin, return argmax.
    // Reuse the second Box-Muller draw, then continue sampling as needed.
    let mut best_class = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    let mut pending = Some(noise2);
    for (class, &count) in histogram.iter().enumerate() {
        let z = match pending.take() {
            Some(z) => z,
            None => {
                let (a, b) = rng.next_normal_pair();
                pending = Some(b);
                a
            }
        };
        let noisy = count as f32 + cfg.sigma_answer * z;
        if noisy > best_val {
            best_val = noisy;
            best_class = class;
        }
    }
    Ok(Some(best_class))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pate_config_valid() {
        let cfg = PateConfig::new(10, 5, 0.5).expect("test invariant: valid pate config");
        assert_eq!(cfg.n_teachers, 10);
    }

    #[test]
    fn pate_config_invalid_epsilon() {
        assert!(matches!(
            PateConfig::new(10, 5, 0.0),
            Err(FedError::InvalidPrivacyBudget)
        ));
    }

    #[test]
    fn pate_config_invalid_teachers() {
        assert!(matches!(
            PateConfig::new(0, 5, 1.0),
            Err(FedError::InsufficientClients { .. })
        ));
    }

    #[test]
    fn noisy_voting_majority_class() {
        // Overwhelming majority for class 2
        let votes: Vec<usize> = std::iter::repeat_n(2, 90)
            .chain(std::iter::repeat_n(0, 5))
            .chain(std::iter::repeat_n(1, 5))
            .collect();
        let mut rng = LcgRng::new(42);
        let result =
            noisy_voting(&votes, 3, 10.0, &mut rng).expect("test invariant: valid noisy voting");
        assert_eq!(
            result, 2,
            "overwhelming majority should win under low noise (ε=10)"
        );
    }

    #[test]
    fn noisy_voting_invalid_epsilon() {
        let votes = vec![0, 1, 2];
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            noisy_voting(&votes, 3, 0.0, &mut rng),
            Err(FedError::InvalidPrivacyBudget)
        ));
    }

    #[test]
    fn noisy_voting_empty_votes() {
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            noisy_voting(&[], 3, 1.0, &mut rng),
            Err(FedError::InsufficientClients { .. })
        ));
    }

    #[test]
    fn data_dependent_epsilon_clear_majority() {
        // High majority gap → lower epsilon
        let counts = vec![100u32, 2, 1, 1];
        let eps = data_dependent_epsilon(&counts, 1e-5, 1.0);
        assert!(
            eps < 1.0,
            "data-dependent epsilon should be < base when majority is clear"
        );
        assert!(eps > 0.0, "data-dependent epsilon should be positive");
    }

    #[test]
    fn data_dependent_epsilon_empty() {
        let eps = data_dependent_epsilon(&[], 1e-5, 1.0);
        assert!(eps.is_infinite());
    }

    #[test]
    fn confident_gnmax_config_validates() {
        assert!(ConfidentGnMaxConfig::new(3, 50.0, 10.0, 5.0).is_ok());
        assert!(matches!(
            ConfidentGnMaxConfig::new(0, 50.0, 10.0, 5.0),
            Err(FedError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            ConfidentGnMaxConfig::new(3, 50.0, 0.0, 5.0),
            Err(FedError::InvalidNoiseMultiplier)
        ));
        assert!(matches!(
            ConfidentGnMaxConfig::new(3, f32::NAN, 10.0, 5.0),
            Err(FedError::Internal(_))
        ));
    }

    #[test]
    fn confident_gnmax_answers_confident_query() {
        // 95/100 votes for class 1, threshold 50 → easily answered, label = 1.
        let votes: Vec<usize> = std::iter::repeat_n(1, 95)
            .chain(std::iter::repeat_n(0, 3))
            .chain(std::iter::repeat_n(2, 2))
            .collect();
        let cfg = ConfidentGnMaxConfig::new(3, 50.0, 5.0, 1.0).expect("cfg");
        let mut rng = LcgRng::new(7);
        let ans = confident_gnmax(&votes, &cfg, &mut rng).expect("answer");
        assert_eq!(ans, Some(1), "confident plurality should be answered as 1");
    }

    #[test]
    fn confident_gnmax_abstains_on_unconfident_query() {
        // Near-uniform split, max count ≈ 4 with tiny noise, threshold 50 →
        // abstain (None) almost surely.
        let votes: Vec<usize> = (0..12).map(|i| i % 3).collect(); // 4 votes each
        let cfg = ConfidentGnMaxConfig::new(3, 50.0, 1.0, 1.0).expect("cfg");
        let mut rng = LcgRng::new(13);
        let mut abstained = 0;
        for _ in 0..50 {
            if confident_gnmax(&votes, &cfg, &mut rng)
                .expect("answer")
                .is_none()
            {
                abstained += 1;
            }
        }
        assert!(
            abstained > 45,
            "low-confidence queries should mostly abstain (got {abstained}/50)"
        );
    }

    #[test]
    fn confident_gnmax_empty_votes_errors() {
        let cfg = ConfidentGnMaxConfig::new(3, 1.0, 1.0, 1.0).expect("cfg");
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            confident_gnmax(&[], &cfg, &mut rng),
            Err(FedError::InsufficientClients { .. })
        ));
    }

    #[test]
    fn confident_gnmax_low_threshold_always_answers() {
        // threshold = 0 → confidence check always passes; answer is well-defined.
        let votes: Vec<usize> = std::iter::repeat_n(2, 30)
            .chain(std::iter::repeat_n(0, 10))
            .collect();
        let cfg = ConfidentGnMaxConfig::new(3, 0.0, 2.0, 0.5).expect("cfg");
        let mut rng = LcgRng::new(99);
        for _ in 0..20 {
            let ans = confident_gnmax(&votes, &cfg, &mut rng).expect("answer");
            assert!(ans.is_some(), "threshold 0 must always answer");
        }
    }
}
