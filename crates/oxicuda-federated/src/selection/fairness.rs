//! Cohort fairness metrics for federated learning.
//!
//! Federated training over heterogeneous clients can converge to a global
//! model whose *average* quality is good while individual groups (strata) —
//! demographic cohorts, device classes, geographic regions — are served far
//! worse than others. This module tracks per-stratum accuracy/loss across
//! rounds and reports standard fairness indices so a scheduler (or an audit
//! step) can detect and react to such disparities.
//!
//! References:
//! * Li et al., "Fair Resource Allocation in Federated Learning" (q-FFL),
//!   ICLR 2020 — motivates equalising the per-client/per-cohort loss spread.
//! * Jain, Chiu & Hawe, "A Quantitative Measure of Fairness and
//!   Discrimination", DEC Technical Report 1984 — Jain's fairness index.
//!
//! All statistics are pure functions over per-stratum scalar metrics; the
//! [`CohortFairnessTracker`] accumulates them across rounds.

use crate::error::{FedError, FedResult};

/// Aggregated metrics for a single stratum (cohort) over one evaluation pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StratumMetrics {
    /// Number of correctly classified examples in this stratum.
    pub correct: u64,
    /// Total number of evaluated examples in this stratum.
    pub total: u64,
    /// Sum of per-example losses over this stratum.
    pub loss_sum: f64,
}

impl StratumMetrics {
    /// Empty metrics (no examples seen yet).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            correct: 0,
            total: 0,
            loss_sum: 0.0,
        }
    }

    /// Accuracy `correct / total`, or `0.0` if no examples were seen.
    #[must_use]
    pub fn accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64
        }
    }

    /// Mean per-example loss `loss_sum / total`, or `0.0` if empty.
    #[must_use]
    pub fn mean_loss(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.loss_sum / self.total as f64
        }
    }

    /// Fold one example's result into the running metrics.
    pub fn record(&mut self, correct: bool, loss: f64) {
        self.total += 1;
        if correct {
            self.correct += 1;
        }
        self.loss_sum += loss;
    }

    /// Merge another stratum's metrics into this one (e.g. across clients of
    /// the same cohort).
    pub fn merge(&mut self, other: &StratumMetrics) {
        self.correct += other.correct;
        self.total += other.total;
        self.loss_sum += other.loss_sum;
    }
}

/// Summary of cross-stratum fairness for one round.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FairnessSummary {
    /// Number of non-empty strata included.
    pub n_strata: usize,
    /// Unweighted mean accuracy across strata.
    pub mean_accuracy: f64,
    /// Lowest stratum accuracy (worst-served cohort).
    pub min_accuracy: f64,
    /// Highest stratum accuracy.
    pub max_accuracy: f64,
    /// Population standard deviation of per-stratum accuracy.
    pub accuracy_std: f64,
    /// Max−min accuracy gap (a direct disparity measure; smaller is fairer).
    pub accuracy_gap: f64,
    /// Unweighted mean per-stratum loss.
    pub mean_loss: f64,
    /// Worst (highest) per-stratum loss.
    pub max_loss: f64,
    /// Jain's fairness index over per-stratum accuracies, in `(0, 1]`.
    ///
    /// `J = (Σxᵢ)² / (n · Σxᵢ²)`. Equals `1.0` when all strata are equal and
    /// `1/n` when one stratum holds all the (positive) mass.
    pub jains_index: f64,
}

/// Compute the per-round fairness summary from a slice of stratum metrics.
///
/// Empty strata (`total == 0`) are ignored so that as-yet-unseen cohorts do
/// not distort the disparity measures.
///
/// # Errors
/// Returns [`FedError::EmptyClientList`] if no stratum has any examples.
pub fn fairness_summary(strata: &[StratumMetrics]) -> FedResult<FairnessSummary> {
    let active: Vec<&StratumMetrics> = strata.iter().filter(|s| s.total > 0).collect();
    if active.is_empty() {
        return Err(FedError::EmptyClientList);
    }
    let n = active.len();
    let accs: Vec<f64> = active.iter().map(|s| s.accuracy()).collect();
    let losses: Vec<f64> = active.iter().map(|s| s.mean_loss()).collect();

    let mean_accuracy = accs.iter().sum::<f64>() / n as f64;
    let mut min_accuracy = f64::INFINITY;
    let mut max_accuracy = f64::NEG_INFINITY;
    for &a in &accs {
        if a < min_accuracy {
            min_accuracy = a;
        }
        if a > max_accuracy {
            max_accuracy = a;
        }
    }
    let var = accs
        .iter()
        .map(|&a| (a - mean_accuracy) * (a - mean_accuracy))
        .sum::<f64>()
        / n as f64;
    let accuracy_std = var.sqrt();

    let mean_loss = losses.iter().sum::<f64>() / n as f64;
    let max_loss = losses.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let jains_index = jains_fairness_index(&accs);

    Ok(FairnessSummary {
        n_strata: n,
        mean_accuracy,
        min_accuracy,
        max_accuracy,
        accuracy_std,
        accuracy_gap: max_accuracy - min_accuracy,
        mean_loss,
        max_loss,
        jains_index,
    })
}

/// Jain's fairness index `J = (Σx)² / (n·Σx²)` over non-negative values.
///
/// Returns `1.0` for an empty input or an all-zero input (degenerate "perfect
/// equality") and clamps the result to `(0, 1]`.
#[must_use]
pub fn jains_fairness_index(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 1.0;
    }
    let sum: f64 = values.iter().sum();
    let sum_sq: f64 = values.iter().map(|&v| v * v).sum();
    if sum_sq <= 0.0 {
        return 1.0;
    }
    let j = (sum * sum) / (n as f64 * sum_sq);
    j.clamp(0.0, 1.0)
}

/// Tracks per-stratum metrics across federated rounds and exposes the worst
/// cohort plus a fairness summary per round.
#[derive(Debug, Clone)]
pub struct CohortFairnessTracker {
    /// Human-readable stratum names (parallel to `current`).
    names: Vec<String>,
    /// Metrics for the round currently being accumulated.
    current: Vec<StratumMetrics>,
    /// One [`FairnessSummary`] per finalised round.
    history: Vec<FairnessSummary>,
}

impl CohortFairnessTracker {
    /// Create a tracker over the named strata.
    ///
    /// # Errors
    /// Returns [`FedError::EmptyClientList`] if `stratum_names` is empty.
    pub fn new(stratum_names: &[&str]) -> FedResult<Self> {
        if stratum_names.is_empty() {
            return Err(FedError::EmptyClientList);
        }
        Ok(Self {
            names: stratum_names.iter().map(|s| (*s).to_string()).collect(),
            current: vec![StratumMetrics::empty(); stratum_names.len()],
            history: Vec::new(),
        })
    }

    /// Number of tracked strata.
    #[must_use]
    pub fn n_strata(&self) -> usize {
        self.names.len()
    }

    /// Stratum names (parallel to metric order).
    #[must_use]
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Record one evaluated example into stratum `idx`.
    ///
    /// # Errors
    /// Returns [`FedError::DimensionMismatch`] if `idx` is out of range.
    pub fn record(&mut self, idx: usize, correct: bool, loss: f64) -> FedResult<()> {
        if idx >= self.current.len() {
            return Err(FedError::DimensionMismatch {
                expected: self.current.len(),
                got: idx,
            });
        }
        self.current[idx].record(correct, loss);
        Ok(())
    }

    /// Merge a fully-aggregated stratum metric (e.g. computed on a client).
    ///
    /// # Errors
    /// Returns [`FedError::DimensionMismatch`] if `idx` is out of range.
    pub fn merge(&mut self, idx: usize, metrics: &StratumMetrics) -> FedResult<()> {
        if idx >= self.current.len() {
            return Err(FedError::DimensionMismatch {
                expected: self.current.len(),
                got: idx,
            });
        }
        self.current[idx].merge(metrics);
        Ok(())
    }

    /// Finalise the round: compute its [`FairnessSummary`], append it to the
    /// history, reset the running metrics, and return the summary.
    ///
    /// # Errors
    /// Returns [`FedError::EmptyClientList`] if no stratum saw any examples
    /// this round.
    pub fn finalize_round(&mut self) -> FedResult<FairnessSummary> {
        let summary = fairness_summary(&self.current)?;
        self.history.push(summary);
        for s in self.current.iter_mut() {
            *s = StratumMetrics::empty();
        }
        Ok(summary)
    }

    /// Index and name of the worst-served (lowest-accuracy) active stratum in
    /// the round currently being accumulated.
    ///
    /// # Errors
    /// Returns [`FedError::EmptyClientList`] if no stratum has examples.
    pub fn worst_stratum(&self) -> FedResult<(usize, &str)> {
        let mut worst: Option<(usize, f64)> = None;
        for (i, s) in self.current.iter().enumerate() {
            if s.total == 0 {
                continue;
            }
            let acc = s.accuracy();
            match worst {
                Some((_, best_acc)) if acc >= best_acc => {}
                _ => worst = Some((i, acc)),
            }
        }
        match worst {
            Some((i, _)) => Ok((i, self.names[i].as_str())),
            None => Err(FedError::EmptyClientList),
        }
    }

    /// All finalised per-round summaries.
    #[must_use]
    pub fn history(&self) -> &[FairnessSummary] {
        &self.history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn stratum_metrics_accuracy_and_loss() {
        let mut m = StratumMetrics::empty();
        m.record(true, 0.1);
        m.record(false, 0.9);
        m.record(true, 0.3);
        assert_eq!(m.total, 3);
        assert_eq!(m.correct, 2);
        assert!((m.accuracy() - 2.0 / 3.0).abs() < 1e-9);
        assert!((m.mean_loss() - (0.1 + 0.9 + 0.3) / 3.0).abs() < 1e-9);
    }

    #[test]
    fn stratum_metrics_empty_is_zero() {
        let m = StratumMetrics::empty();
        assert_eq!(m.accuracy(), 0.0);
        assert_eq!(m.mean_loss(), 0.0);
    }

    #[test]
    fn jains_index_equal_is_one() {
        let v = vec![0.8, 0.8, 0.8, 0.8];
        assert!((jains_fairness_index(&v) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn jains_index_concentrated_approaches_inverse_n() {
        // One cohort carries all accuracy mass → J = 1/n.
        let v = vec![1.0, 0.0, 0.0, 0.0];
        assert!((jains_fairness_index(&v) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn jains_index_degenerate_inputs() {
        assert_eq!(jains_fairness_index(&[]), 1.0);
        assert_eq!(jains_fairness_index(&[0.0, 0.0]), 1.0);
    }

    #[test]
    fn fairness_summary_disparity_detected() {
        let strata = vec![
            StratumMetrics {
                correct: 95,
                total: 100,
                loss_sum: 10.0,
            },
            StratumMetrics {
                correct: 55,
                total: 100,
                loss_sum: 60.0,
            },
            StratumMetrics::empty(), // ignored
        ];
        let s = fairness_summary(&strata).expect("summary");
        assert_eq!(s.n_strata, 2);
        assert!((s.max_accuracy - 0.95).abs() < 1e-9);
        assert!((s.min_accuracy - 0.55).abs() < 1e-9);
        assert!((s.accuracy_gap - 0.40).abs() < 1e-9);
        assert!(s.accuracy_std > 0.0);
        assert!(s.jains_index < 1.0, "disparate accuracies → J < 1");
        assert!((s.max_loss - 0.6).abs() < 1e-9);
    }

    #[test]
    fn fairness_summary_balanced_is_perfectly_fair() {
        let strata = vec![
            StratumMetrics {
                correct: 80,
                total: 100,
                loss_sum: 20.0,
            },
            StratumMetrics {
                correct: 160,
                total: 200,
                loss_sum: 40.0,
            },
        ];
        let s = fairness_summary(&strata).expect("summary");
        assert!((s.accuracy_gap).abs() < 1e-9, "equal accuracies → no gap");
        assert!((s.jains_index - 1.0).abs() < 1e-9);
        assert!(s.accuracy_std < 1e-9);
    }

    #[test]
    fn fairness_summary_all_empty_errors() {
        let strata = vec![StratumMetrics::empty(), StratumMetrics::empty()];
        assert!(matches!(
            fairness_summary(&strata),
            Err(FedError::EmptyClientList)
        ));
    }

    #[test]
    fn tracker_records_and_finalises_rounds() {
        let mut t = CohortFairnessTracker::new(&["urban", "rural"]).expect("tracker");
        assert_eq!(t.n_strata(), 2);
        // Urban: 9/10 correct. Rural: 4/10 correct.
        for i in 0..10 {
            t.record(0, i != 0, 0.1).expect("urban record");
            t.record(1, i < 4, 0.5).expect("rural record");
        }
        let (worst_idx, worst_name) = t.worst_stratum().expect("worst");
        assert_eq!(worst_idx, 1);
        assert_eq!(worst_name, "rural");

        let summary = t.finalize_round().expect("round");
        assert_eq!(summary.n_strata, 2);
        assert!(summary.accuracy_gap > 0.3);
        assert_eq!(t.history().len(), 1);
        // After finalisation the running metrics reset.
        assert!(matches!(t.worst_stratum(), Err(FedError::EmptyClientList)));
    }

    #[test]
    fn tracker_out_of_range_errors() {
        let mut t = CohortFairnessTracker::new(&["a"]).expect("tracker");
        assert!(matches!(
            t.record(5, true, 0.1),
            Err(FedError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            t.merge(5, &StratumMetrics::empty()),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn tracker_new_empty_errors() {
        assert!(matches!(
            CohortFairnessTracker::new(&[]),
            Err(FedError::EmptyClientList)
        ));
    }

    #[test]
    fn tracker_merge_aggregates_client_metrics() {
        let mut t = CohortFairnessTracker::new(&["x", "y"]).expect("tracker");
        // Simulate two clients reporting pre-aggregated metrics for stratum 0.
        let client_a = StratumMetrics {
            correct: 30,
            total: 50,
            loss_sum: 12.0,
        };
        let client_b = StratumMetrics {
            correct: 40,
            total: 50,
            loss_sum: 8.0,
        };
        t.merge(0, &client_a).expect("merge a");
        t.merge(0, &client_b).expect("merge b");
        t.merge(
            1,
            &StratumMetrics {
                correct: 90,
                total: 100,
                loss_sum: 10.0,
            },
        )
        .expect("merge y");
        let s = t.finalize_round().expect("round");
        // Stratum 0 accuracy = 70/100 = 0.7, stratum 1 = 0.9.
        assert!((s.min_accuracy - 0.7).abs() < 1e-9);
        assert!((s.max_accuracy - 0.9).abs() < 1e-9);
    }

    #[test]
    fn tracker_simulated_training_improves_fairness() {
        // Two cohorts; a simulated FL run gradually closes the accuracy gap as
        // the global model improves on the under-served cohort. We assert the
        // gap shrinks monotonically across rounds.
        let mut rng = LcgRng::new(2026);
        let mut t = CohortFairnessTracker::new(&["majority", "minority"]).expect("tracker");
        let mut prev_gap = f64::INFINITY;
        for round in 0..5 {
            // Majority accuracy is high and roughly flat; minority climbs.
            let maj_acc = 0.90;
            let min_acc = 0.50 + 0.08 * round as f64;
            for _ in 0..200 {
                let maj_correct = rng.next_f32() < maj_acc as f32;
                let min_correct = rng.next_f32() < min_acc as f32;
                t.record(0, maj_correct, if maj_correct { 0.1 } else { 1.0 })
                    .expect("maj");
                t.record(1, min_correct, if min_correct { 0.1 } else { 1.0 })
                    .expect("min");
            }
            let s = t.finalize_round().expect("round");
            assert!(
                s.accuracy_gap <= prev_gap + 0.05,
                "round {round}: gap {} should not grow beyond prev {prev_gap}",
                s.accuracy_gap
            );
            prev_gap = s.accuracy_gap;
        }
        // Final round Jain index should be closer to 1 than the first round.
        let hist = t.history();
        assert!(
            hist.last().expect("last").jains_index >= hist[0].jains_index - 1e-6,
            "fairness should not regress over training"
        );
    }
}
