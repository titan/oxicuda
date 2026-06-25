//! Calibration and calibration-fairness metrics for recommendation scores.
//!
//! A model is *calibrated* when, among all predictions with confidence `p`, the
//! empirical fraction of positives is also `p`. Ranking quality (NDCG, AUC) is
//! invariant to monotone re-scaling and therefore says nothing about
//! calibration, yet calibrated probabilities are essential for downstream
//! decisions (expected-value bidding, blended CTR×CVR, fairness audits).
//!
//! References:
//! - Naeini, Cooper, Hauskrecht, "Obtaining Well Calibrated Probabilities Using
//!   Bayesian Binning", AAAI 2015 (ECE / MCE definition).
//! - Guo, Pleiss, Sun, Weinberger, "On Calibration of Modern Neural Networks",
//!   ICML 2017 (reliability diagrams, expected calibration error).
//! - Brier, "Verification of Forecasts Expressed in Terms of Probability", 1950.
//!
//! All functions consume parallel slices of predicted probabilities `p ∈ [0, 1]`
//! and binary outcomes `y ∈ {false, true}` and return [`RecsysResult`].

use crate::error::{RecsysError, RecsysResult};

/// One equal-width reliability bin produced by [`reliability_bins`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReliabilityBin {
    /// Lower edge of the confidence interval (inclusive).
    pub lower: f32,
    /// Upper edge of the confidence interval (exclusive, except the last bin).
    pub upper: f32,
    /// Number of predictions that fell into the bin.
    pub count: usize,
    /// Mean predicted confidence of the predictions in the bin.
    pub mean_confidence: f32,
    /// Empirical positive rate (accuracy) of the predictions in the bin.
    pub mean_accuracy: f32,
}

fn check_inputs(preds: &[f32], labels: &[bool]) -> RecsysResult<()> {
    if preds.is_empty() {
        return Err(RecsysError::EmptyInput);
    }
    if preds.len() != labels.len() {
        return Err(RecsysError::DimensionMismatch {
            expected: preds.len(),
            got: labels.len(),
        });
    }
    Ok(())
}

/// Partition predictions into `n_bins` equal-width confidence bins and report
/// the per-bin confidence/accuracy used by a reliability diagram.
///
/// Predictions are clamped to `[0, 1]` before binning. The last bin is closed on
/// the right so that a prediction of exactly `1.0` is counted.
///
/// # Errors
/// - [`RecsysError::EmptyInput`] if inputs are empty.
/// - [`RecsysError::DimensionMismatch`] if lengths differ.
/// - [`RecsysError::InvalidConfig`] if `n_bins == 0`.
pub fn reliability_bins(
    preds: &[f32],
    labels: &[bool],
    n_bins: usize,
) -> RecsysResult<Vec<ReliabilityBin>> {
    check_inputs(preds, labels)?;
    if n_bins == 0 {
        return Err(RecsysError::InvalidConfig {
            msg: "n_bins must be >= 1".into(),
        });
    }
    let width = 1.0 / n_bins as f32;
    let mut sum_conf = vec![0.0_f32; n_bins];
    let mut sum_pos = vec![0.0_f32; n_bins];
    let mut count = vec![0usize; n_bins];

    for (&p, &y) in preds.iter().zip(labels.iter()) {
        let pc = p.clamp(0.0, 1.0);
        let mut b = (pc / width) as usize;
        if b >= n_bins {
            b = n_bins - 1;
        }
        sum_conf[b] += pc;
        if y {
            sum_pos[b] += 1.0;
        }
        count[b] += 1;
    }

    let mut bins = Vec::with_capacity(n_bins);
    for b in 0..n_bins {
        let c = count[b];
        let (mc, ma) = if c == 0 {
            (0.0, 0.0)
        } else {
            (sum_conf[b] / c as f32, sum_pos[b] / c as f32)
        };
        bins.push(ReliabilityBin {
            lower: b as f32 * width,
            upper: (b as f32 + 1.0) * width,
            count: c,
            mean_confidence: mc,
            mean_accuracy: ma,
        });
    }
    Ok(bins)
}

/// Expected Calibration Error: count-weighted mean absolute gap between
/// per-bin confidence and accuracy, `ECE = Σ_b (n_b/N) · |conf_b − acc_b|`.
///
/// Lower is better; `0` means perfectly calibrated at this bin resolution.
///
/// # Errors
/// Propagates [`reliability_bins`].
pub fn expected_calibration_error(
    preds: &[f32],
    labels: &[bool],
    n_bins: usize,
) -> RecsysResult<f32> {
    let bins = reliability_bins(preds, labels, n_bins)?;
    let n = preds.len() as f32;
    let ece = bins
        .iter()
        .filter(|b| b.count > 0)
        .map(|b| (b.count as f32 / n) * (b.mean_confidence - b.mean_accuracy).abs())
        .sum();
    Ok(ece)
}

/// Maximum Calibration Error: the largest per-bin `|conf_b − acc_b|` over all
/// non-empty bins (worst-case miscalibration).
///
/// # Errors
/// Propagates [`reliability_bins`].
pub fn maximum_calibration_error(
    preds: &[f32],
    labels: &[bool],
    n_bins: usize,
) -> RecsysResult<f32> {
    let bins = reliability_bins(preds, labels, n_bins)?;
    let mce = bins
        .iter()
        .filter(|b| b.count > 0)
        .map(|b| (b.mean_confidence - b.mean_accuracy).abs())
        .fold(0.0_f32, f32::max);
    Ok(mce)
}

/// Brier score: mean squared error between predicted probability and outcome,
/// `(1/N) Σ (p_i − y_i)²`. Lower is better; a proper scoring rule.
///
/// # Errors
/// - [`RecsysError::EmptyInput`] / [`RecsysError::DimensionMismatch`].
pub fn brier_score(preds: &[f32], labels: &[bool]) -> RecsysResult<f32> {
    check_inputs(preds, labels)?;
    let n = preds.len() as f32;
    let s: f32 = preds
        .iter()
        .zip(labels.iter())
        .map(|(&p, &y)| {
            let target = if y { 1.0 } else { 0.0 };
            let d = p.clamp(0.0, 1.0) - target;
            d * d
        })
        .sum();
    Ok(s / n)
}

/// Binary log loss (cross-entropy), `-(1/N) Σ [y·ln p + (1−y)·ln(1−p)]`, with
/// predictions clamped to `[eps, 1−eps]` to avoid infinities.
///
/// # Errors
/// - [`RecsysError::EmptyInput`] / [`RecsysError::DimensionMismatch`].
/// - [`RecsysError::InvalidConfig`] if `eps` is not in `(0, 0.5)`.
pub fn log_loss(preds: &[f32], labels: &[bool], eps: f32) -> RecsysResult<f32> {
    check_inputs(preds, labels)?;
    if !(eps > 0.0 && eps < 0.5) {
        return Err(RecsysError::InvalidConfig {
            msg: "eps must be in (0, 0.5)".into(),
        });
    }
    let n = preds.len() as f32;
    let s: f32 = preds
        .iter()
        .zip(labels.iter())
        .map(|(&p, &y)| {
            let pc = p.clamp(eps, 1.0 - eps);
            if y { -pc.ln() } else { -(1.0 - pc).ln() }
        })
        .sum();
    Ok(s / n)
}

/// Calibration-fairness disparity: the maximum gap in ECE across demographic
/// groups, `max_g ECE_g − min_g ECE_g`.
///
/// A model can be globally calibrated yet *systematically* over- or
/// under-confident for a protected group; this metric surfaces that gap
/// (Singh-Joachims-style group exposure but for *calibration* rather than
/// exposure). `groups[i]` is the integer group id of sample `i`. Groups with no
/// samples are ignored. Returns `0` when fewer than two groups are populated.
///
/// # Errors
/// - [`RecsysError::EmptyInput`] / [`RecsysError::DimensionMismatch`] (also if
///   `groups.len() != preds.len()`).
/// - Propagates [`expected_calibration_error`].
pub fn group_calibration_disparity(
    preds: &[f32],
    labels: &[bool],
    groups: &[usize],
    n_bins: usize,
) -> RecsysResult<f32> {
    check_inputs(preds, labels)?;
    if groups.len() != preds.len() {
        return Err(RecsysError::DimensionMismatch {
            expected: preds.len(),
            got: groups.len(),
        });
    }
    let n_groups = groups.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    if n_groups == 0 {
        return Ok(0.0);
    }
    let mut per_group_eces: Vec<f32> = Vec::new();
    for g in 0..n_groups {
        let gp: Vec<f32> = preds
            .iter()
            .zip(groups.iter())
            .filter(|&(_, &gid)| gid == g)
            .map(|(&p, _)| p)
            .collect();
        if gp.is_empty() {
            continue;
        }
        let gl: Vec<bool> = labels
            .iter()
            .zip(groups.iter())
            .filter(|&(_, &gid)| gid == g)
            .map(|(&y, _)| y)
            .collect();
        per_group_eces.push(expected_calibration_error(&gp, &gl, n_bins)?);
    }
    if per_group_eces.len() < 2 {
        return Ok(0.0);
    }
    let max = per_group_eces
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let min = per_group_eces.iter().copied().fold(f32::INFINITY, f32::min);
    Ok(max - min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn empty_and_mismatch_error() {
        assert!(expected_calibration_error(&[], &[], 10).is_err());
        assert!(expected_calibration_error(&[0.5], &[true, false], 10).is_err());
        assert!(reliability_bins(&[0.5], &[true], 0).is_err());
        assert!(log_loss(&[0.5], &[true], 0.0).is_err());
        assert!(brier_score(&[], &[]).is_err());
    }

    #[test]
    fn perfectly_calibrated_has_zero_ece() {
        // Construct bins where empirical positive rate == bin confidence.
        // Bin at 0.0: 10 negatives (acc 0, conf ≈ 0).
        // Bin at 1.0: 10 positives (acc 1, conf ≈ 1).
        let mut preds = Vec::new();
        let mut labels = Vec::new();
        for _ in 0..10 {
            preds.push(0.0);
            labels.push(false);
        }
        for _ in 0..10 {
            preds.push(1.0);
            labels.push(true);
        }
        let ece = expected_calibration_error(&preds, &labels, 10).expect("ece");
        assert!(ece.abs() < 1e-6, "expected ECE≈0, got {ece}");
    }

    #[test]
    fn fully_miscalibrated_has_unit_error() {
        // Predict 1.0 for everything but all outcomes are negative ⇒ the single
        // populated bin has confidence 1, accuracy 0 ⇒ ECE = MCE = 1.
        let preds = vec![1.0_f32; 20];
        let labels = vec![false; 20];
        let ece = expected_calibration_error(&preds, &labels, 10).expect("ece");
        let mce = maximum_calibration_error(&preds, &labels, 10).expect("mce");
        assert!((ece - 1.0).abs() < 1e-6, "ECE {ece}");
        assert!((mce - 1.0).abs() < 1e-6, "MCE {mce}");
    }

    #[test]
    fn brier_matches_hand_computation() {
        // preds .8/.2 with labels true/false ⇒ (0.2² + 0.2²)/2 = 0.04.
        let b = brier_score(&[0.8, 0.2], &[true, false]).expect("brier");
        assert!((b - 0.04).abs() < 1e-6, "got {b}");
    }

    #[test]
    fn log_loss_of_confident_correct_is_small() {
        let small = log_loss(&[0.99, 0.01], &[true, false], 1e-6).expect("ll");
        let big = log_loss(&[0.01, 0.99], &[true, false], 1e-6).expect("ll");
        assert!(small < big, "confident-correct must beat confident-wrong");
        assert!(small < 0.05, "got {small}");
    }

    #[test]
    fn bins_cover_all_samples() {
        let mut rng = LcgRng::new(2024);
        let n = 500usize;
        let preds: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let labels: Vec<bool> = (0..n).map(|_| rng.next_f32() < 0.5).collect();
        let bins = reliability_bins(&preds, &labels, 15).expect("bins");
        let total: usize = bins.iter().map(|b| b.count).sum();
        assert_eq!(total, n, "every prediction must land in exactly one bin");
        for b in &bins {
            if b.count > 0 {
                assert!((0.0..=1.0).contains(&b.mean_confidence));
                assert!((0.0..=1.0).contains(&b.mean_accuracy));
            }
        }
    }

    #[test]
    fn group_disparity_detects_imbalance() {
        // Group 0 perfectly calibrated; group 1 fully miscalibrated.
        let mut preds = Vec::new();
        let mut labels = Vec::new();
        let mut groups = Vec::new();
        // group 0: calibrated (10 at 0/neg, 10 at 1/pos)
        for _ in 0..10 {
            preds.push(0.0);
            labels.push(false);
            groups.push(0usize);
        }
        for _ in 0..10 {
            preds.push(1.0);
            labels.push(true);
            groups.push(0);
        }
        // group 1: confidently wrong
        for _ in 0..20 {
            preds.push(1.0);
            labels.push(false);
            groups.push(1);
        }
        let disp = group_calibration_disparity(&preds, &labels, &groups, 10).expect("disp");
        assert!(disp > 0.9, "expected large disparity, got {disp}");

        // A single populated group ⇒ disparity 0.
        let one = group_calibration_disparity(&[1.0, 0.0], &[true, false], &[5, 5], 10)
            .expect("one group");
        assert!(
            one.abs() < 1e-6,
            "single-group disparity must be 0, got {one}"
        );
    }
}
