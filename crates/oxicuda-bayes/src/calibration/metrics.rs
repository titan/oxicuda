//! Calibration error metrics and reliability diagrams.
//!
//! Implements:
//! - Expected Calibration Error (ECE) with equal-width binning (Naeini 2015, Guo 2017)
//! - Maximum Calibration Error (MCE)
//! - Adaptive Calibration Error (ACE) with equal-mass binning (Nixon 2019)
//! - Brier score for multi-class probabilistic predictions (Brier 1950)
//! - Negative log-likelihood (NLL) of categorical predictions
//! - [`ReliabilityDiagram`] data for visual calibration analysis
//!
//! All metrics consume top-1 confidence + correctness pairs derived from
//! per-class probability vectors.

use crate::error::{BayesError, BayesResult};

/// Per-bin reliability data for diagrammatic calibration analysis.
///
/// Each bin holds `(lo, hi]` confidence boundaries plus the empirical
/// average confidence and accuracy within the bin and the count of samples.
#[derive(Debug, Clone, PartialEq)]
pub struct ReliabilityBin {
    /// Inclusive lower bound of the confidence bin.
    pub lo: f32,
    /// Exclusive upper bound of the confidence bin.
    pub hi: f32,
    /// Mean confidence of predictions in this bin.
    pub avg_confidence: f32,
    /// Mean accuracy (top-1 correctness) of predictions in this bin.
    pub avg_accuracy: f32,
    /// Number of samples that fell in this bin.
    pub count: usize,
}

/// Reliability diagram: a list of confidence bins.
#[derive(Debug, Clone)]
pub struct ReliabilityDiagram {
    /// Bins in ascending confidence order.
    pub bins: Vec<ReliabilityBin>,
    /// Total number of samples used.
    pub n_samples: usize,
}

impl ReliabilityDiagram {
    /// Total weighted gap `Σ_b (n_b/N)·|conf_b − acc_b|` — i.e. ECE.
    #[must_use]
    pub fn ece(&self) -> f32 {
        if self.n_samples == 0 {
            return 0.0;
        }
        let n = self.n_samples as f32;
        self.bins
            .iter()
            .map(|b| (b.count as f32 / n) * (b.avg_confidence - b.avg_accuracy).abs())
            .sum()
    }

    /// Maximum bin gap `max_b |conf_b − acc_b|` (over non-empty bins).
    #[must_use]
    pub fn mce(&self) -> f32 {
        self.bins
            .iter()
            .filter(|b| b.count > 0)
            .map(|b| (b.avg_confidence - b.avg_accuracy).abs())
            .fold(0.0_f32, f32::max)
    }
}

/// Validate that two slices align in length and are non-empty.
fn check_pairs(confidences: &[f32], correct: &[bool]) -> BayesResult<()> {
    if confidences.is_empty() {
        return Err(BayesError::CalibrationSetEmpty);
    }
    if confidences.len() != correct.len() {
        return Err(BayesError::DimensionMismatch {
            expected: confidences.len(),
            got: correct.len(),
        });
    }
    Ok(())
}

/// Build an equal-width reliability diagram with `n_bins` confidence bins
/// over `[0, 1]`.
///
/// # Errors
/// - [`BayesError::CalibrationSetEmpty`] if `confidences` is empty.
/// - [`BayesError::DimensionMismatch`] if lengths disagree.
/// - [`BayesError::NCalibBinsTooSmall`] if `n_bins == 0`.
pub fn reliability_diagram(
    confidences: &[f32],
    correct: &[bool],
    n_bins: usize,
) -> BayesResult<ReliabilityDiagram> {
    check_pairs(confidences, correct)?;
    if n_bins == 0 {
        return Err(BayesError::NCalibBinsTooSmall);
    }
    let mut sums = vec![0.0_f32; n_bins];
    let mut accs = vec![0.0_f32; n_bins];
    let mut counts = vec![0_usize; n_bins];
    let nb = n_bins as f32;
    for (&c, &ok) in confidences.iter().zip(correct.iter()) {
        if !c.is_finite() || !(0.0..=1.0).contains(&c) {
            return Err(BayesError::NanEncountered {
                location: "reliability_diagram: confidence outside [0,1]",
            });
        }
        let raw = (c * nb) as usize;
        let idx = raw.min(n_bins - 1);
        sums[idx] += c;
        accs[idx] += if ok { 1.0 } else { 0.0 };
        counts[idx] += 1;
    }
    let bins = (0..n_bins)
        .map(|b| {
            let lo = b as f32 / nb;
            let hi = (b + 1) as f32 / nb;
            let count = counts[b];
            let (avg_confidence, avg_accuracy) = if count == 0 {
                (0.0, 0.0)
            } else {
                let inv = 1.0 / count as f32;
                (sums[b] * inv, accs[b] * inv)
            };
            ReliabilityBin {
                lo,
                hi,
                avg_confidence,
                avg_accuracy,
                count,
            }
        })
        .collect();
    Ok(ReliabilityDiagram {
        bins,
        n_samples: confidences.len(),
    })
}

/// Expected Calibration Error with equal-width binning (Naeini 2015).
///
/// `ECE = Σ_b (n_b/N) · |acc(b) − conf(b)|`
///
/// # Errors
/// Propagates errors from [`reliability_diagram`].
pub fn expected_calibration_error(
    confidences: &[f32],
    correct: &[bool],
    n_bins: usize,
) -> BayesResult<f32> {
    Ok(reliability_diagram(confidences, correct, n_bins)?.ece())
}

/// Maximum Calibration Error: max over non-empty bins.
///
/// # Errors
/// Propagates errors from [`reliability_diagram`].
pub fn maximum_calibration_error(
    confidences: &[f32],
    correct: &[bool],
    n_bins: usize,
) -> BayesResult<f32> {
    Ok(reliability_diagram(confidences, correct, n_bins)?.mce())
}

/// Adaptive Calibration Error using equal-mass quantile binning (Nixon 2019).
///
/// Predictions are sorted by confidence and split into `n_bins` equal-count
/// groups (last group absorbs remainder); each group contributes
/// `(n_b/N) · |acc_b − conf_b|`.
///
/// # Errors
/// - [`BayesError::CalibrationSetEmpty`] / [`BayesError::DimensionMismatch`].
/// - [`BayesError::NCalibBinsTooSmall`] if `n_bins == 0`.
pub fn adaptive_calibration_error(
    confidences: &[f32],
    correct: &[bool],
    n_bins: usize,
) -> BayesResult<f32> {
    check_pairs(confidences, correct)?;
    if n_bins == 0 {
        return Err(BayesError::NCalibBinsTooSmall);
    }
    let mut idx: Vec<usize> = (0..confidences.len()).collect();
    idx.sort_by(|&a, &b| {
        confidences[a]
            .partial_cmp(&confidences[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n = confidences.len();
    let n_bins = n_bins.min(n);
    let chunk = n / n_bins;
    let remainder = n % n_bins;
    let mut start = 0;
    let mut ace = 0.0_f32;
    for b in 0..n_bins {
        let extra = if b < remainder { 1 } else { 0 };
        let len = chunk + extra;
        let end = start + len;
        let mut sum_conf = 0.0_f32;
        let mut sum_acc = 0.0_f32;
        for &i in &idx[start..end] {
            sum_conf += confidences[i];
            sum_acc += if correct[i] { 1.0 } else { 0.0 };
        }
        let inv = 1.0 / len as f32;
        let conf = sum_conf * inv;
        let acc = sum_acc * inv;
        ace += (len as f32 / n as f32) * (conf - acc).abs();
        start = end;
    }
    Ok(ace)
}

/// Multi-class Brier score
/// `BS = (1/N) · Σ_n Σ_k (p_{n,k} − y_{n,k})²`.
///
/// # Errors
/// - [`BayesError::CalibrationSetEmpty`] if `probs` is empty.
/// - [`BayesError::DimensionMismatch`] if `probs.len()` is not `n·n_classes`
///   or `labels.len() != n`.
pub fn brier_score(probs: &[f32], labels: &[usize], n_classes: usize) -> BayesResult<f32> {
    if probs.is_empty() || labels.is_empty() || n_classes == 0 {
        return Err(BayesError::CalibrationSetEmpty);
    }
    if probs.len() != labels.len() * n_classes {
        return Err(BayesError::DimensionMismatch {
            expected: labels.len() * n_classes,
            got: probs.len(),
        });
    }
    let n = labels.len();
    let mut sum = 0.0_f32;
    for (i, &y) in labels.iter().enumerate() {
        if y >= n_classes {
            return Err(BayesError::DimensionMismatch {
                expected: n_classes,
                got: y + 1,
            });
        }
        let row = &probs[i * n_classes..(i + 1) * n_classes];
        for (k, &p) in row.iter().enumerate() {
            let target = if k == y { 1.0 } else { 0.0 };
            let d = p - target;
            sum += d * d;
        }
    }
    Ok(sum / n as f32)
}

/// Negative log-likelihood of categorical predictions
/// `NLL = -(1/N) · Σ_n log p_{n, y_n}`, with floor `ε = 1e-12`.
///
/// # Errors
/// - [`BayesError::CalibrationSetEmpty`] if `probs` is empty.
/// - [`BayesError::DimensionMismatch`] if `probs.len()` is not `n·n_classes`.
pub fn negative_log_likelihood(
    probs: &[f32],
    labels: &[usize],
    n_classes: usize,
) -> BayesResult<f32> {
    if probs.is_empty() || labels.is_empty() || n_classes == 0 {
        return Err(BayesError::CalibrationSetEmpty);
    }
    if probs.len() != labels.len() * n_classes {
        return Err(BayesError::DimensionMismatch {
            expected: labels.len() * n_classes,
            got: probs.len(),
        });
    }
    let n = labels.len();
    let mut acc = 0.0_f32;
    for (i, &y) in labels.iter().enumerate() {
        if y >= n_classes {
            return Err(BayesError::DimensionMismatch {
                expected: n_classes,
                got: y + 1,
            });
        }
        let p = probs[i * n_classes + y].max(1e-12);
        acc += -p.ln();
    }
    Ok(acc / n as f32)
}

/// Extract top-1 confidence and correctness from a `[N, K]` row-major
/// probability matrix.
///
/// # Errors
/// - [`BayesError::CalibrationSetEmpty`] if `probs.is_empty()`.
/// - [`BayesError::DimensionMismatch`] if shapes disagree.
pub fn top1_confidences(
    probs: &[f32],
    labels: &[usize],
    n_classes: usize,
) -> BayesResult<(Vec<f32>, Vec<bool>)> {
    if probs.is_empty() || n_classes == 0 {
        return Err(BayesError::CalibrationSetEmpty);
    }
    if probs.len() != labels.len() * n_classes {
        return Err(BayesError::DimensionMismatch {
            expected: labels.len() * n_classes,
            got: probs.len(),
        });
    }
    let n = labels.len();
    let mut confidences = Vec::with_capacity(n);
    let mut correct = Vec::with_capacity(n);
    for (i, &y) in labels.iter().enumerate() {
        if y >= n_classes {
            return Err(BayesError::DimensionMismatch {
                expected: n_classes,
                got: y + 1,
            });
        }
        let row = &probs[i * n_classes..(i + 1) * n_classes];
        let mut best = 0usize;
        let mut best_v = row[0];
        for (k, &p) in row.iter().enumerate().skip(1) {
            if p > best_v {
                best_v = p;
                best = k;
            }
        }
        confidences.push(best_v);
        correct.push(best == y);
    }
    Ok((confidences, correct))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perfectly_calibrated() -> (Vec<f32>, Vec<bool>) {
        // confidence equals empirical accuracy
        let mut c = Vec::new();
        let mut ok = Vec::new();
        // 10 samples with conf=0.9, 9 correct (acc=0.9)
        for i in 0..10 {
            c.push(0.9);
            ok.push(i < 9);
        }
        // 10 samples with conf=0.5, 5 correct
        for i in 0..10 {
            c.push(0.5);
            ok.push(i < 5);
        }
        (c, ok)
    }

    #[test]
    fn ece_zero_for_perfectly_calibrated() {
        let (c, ok) = perfectly_calibrated();
        let ece = expected_calibration_error(&c, &ok, 10)
            .expect("expected_calibration_error should succeed");
        assert!(ece < 1e-5, "ece = {ece}");
    }

    #[test]
    fn ece_positive_for_overconfident() {
        // Always says 0.99, only 50% correct
        let c = vec![0.99_f32; 100];
        let ok: Vec<bool> = (0..100).map(|i| i % 2 == 0).collect();
        let ece = expected_calibration_error(&c, &ok, 10)
            .expect("expected_calibration_error should succeed");
        assert!(ece > 0.4, "expected severe miscalibration, got ece = {ece}");
    }

    #[test]
    fn mce_geq_ece() {
        let c = vec![0.99_f32; 100];
        let ok: Vec<bool> = (0..100).map(|i| i % 4 == 0).collect();
        let ece = expected_calibration_error(&c, &ok, 10)
            .expect("expected_calibration_error should succeed");
        let mce = maximum_calibration_error(&c, &ok, 10)
            .expect("maximum_calibration_error should succeed");
        assert!(mce >= ece - 1e-6);
    }

    #[test]
    fn ace_low_for_perfectly_calibrated_bimodal() {
        // Bimodal data (10 at conf=0.5, 10 at conf=0.9) — ACE with 2 quantile bins
        // perfectly aligns with the two modes and yields zero error.
        let (c, ok) = perfectly_calibrated();
        let ace = adaptive_calibration_error(&c, &ok, 2)
            .expect("adaptive_calibration_error should succeed");
        assert!(ace < 1e-5, "ace = {ace}");
    }

    #[test]
    fn ace_handles_more_bins_than_samples() {
        let c = vec![0.5_f32, 0.6, 0.7];
        let ok = vec![true, false, true];
        let ace = adaptive_calibration_error(&c, &ok, 10)
            .expect("adaptive_calibration_error should succeed");
        assert!(ace.is_finite());
    }

    #[test]
    fn brier_score_perfect_predictions() {
        let probs = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let labels = vec![0_usize, 1, 2];
        let bs = brier_score(&probs, &labels, 3).expect("brier_score should succeed");
        assert!(bs < 1e-6, "bs = {bs}");
    }

    #[test]
    fn brier_score_uniform_three_class() {
        let probs = vec![1.0_f32 / 3.0; 9];
        let labels = vec![0_usize, 1, 2];
        let bs = brier_score(&probs, &labels, 3).expect("brier_score should succeed");
        // expected: per-sample contribution = (2/3)^2 + 2*(1/3)^2 = 4/9 + 2/9 = 6/9 = 2/3
        assert!((bs - (2.0 / 3.0)).abs() < 1e-3, "bs = {bs}");
    }

    #[test]
    fn nll_perfect_predictions_zero() {
        let probs = vec![1.0_f32, 0.0, 0.0, 1.0];
        let labels = vec![0_usize, 1];
        let nll = negative_log_likelihood(&probs, &labels, 2)
            .expect("negative_log_likelihood should succeed");
        assert!(nll.abs() < 1e-5, "nll = {nll}");
    }

    #[test]
    fn nll_uniform_predictions_log_k() {
        let probs = vec![0.5_f32, 0.5, 0.5, 0.5];
        let labels = vec![0_usize, 1];
        let nll = negative_log_likelihood(&probs, &labels, 2)
            .expect("negative_log_likelihood should succeed");
        // -ln(0.5) = ln(2) ≈ 0.6931
        assert!((nll - std::f32::consts::LN_2).abs() < 1e-4, "nll = {nll}");
    }

    #[test]
    fn nll_clamps_zero_probability() {
        let probs = vec![0.0_f32, 1.0];
        let labels = vec![0_usize];
        let nll = negative_log_likelihood(&probs, &labels, 2)
            .expect("negative_log_likelihood should succeed");
        assert!(nll.is_finite() && nll > 0.0);
    }

    #[test]
    fn top1_confidences_argmax_tie_breaks_first() {
        let probs = vec![0.5_f32, 0.5, 0.0];
        let labels = vec![0_usize];
        let (c, ok) =
            top1_confidences(&probs, &labels, 3).expect("top1_confidences should succeed");
        assert_eq!(c.len(), 1);
        assert!((c[0] - 0.5).abs() < 1e-6);
        assert!(ok[0]);
    }

    #[test]
    fn reliability_diagram_bin_count_matches() {
        let c = vec![0.05_f32, 0.15, 0.25, 0.95];
        let ok = vec![true, false, true, true];
        let rd = reliability_diagram(&c, &ok, 10).expect("reliability_diagram should succeed");
        assert_eq!(rd.bins.len(), 10);
        assert_eq!(rd.n_samples, 4);
    }

    #[test]
    fn ece_rejects_zero_bins() {
        let c = vec![0.5_f32];
        let ok = vec![true];
        assert!(expected_calibration_error(&c, &ok, 0).is_err());
    }

    #[test]
    fn ece_rejects_empty() {
        let c: Vec<f32> = vec![];
        let ok: Vec<bool> = vec![];
        assert!(expected_calibration_error(&c, &ok, 10).is_err());
    }

    #[test]
    fn ece_rejects_out_of_range_confidence() {
        let c = vec![1.5_f32];
        let ok = vec![true];
        assert!(expected_calibration_error(&c, &ok, 10).is_err());
    }

    #[test]
    fn brier_rejects_invalid_label() {
        let probs = vec![0.5_f32, 0.5];
        let labels = vec![5_usize];
        assert!(brier_score(&probs, &labels, 2).is_err());
    }
}
