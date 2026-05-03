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
}
