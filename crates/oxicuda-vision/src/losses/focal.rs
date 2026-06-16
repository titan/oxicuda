//! Focal loss for dense object detection (Lin et al. 2017, ICCV).
//!
//! The focal loss reshapes the standard cross-entropy so that well-classified
//! examples contribute little, focusing training on the hard, misclassified
//! minority. For a predicted probability `p` of the true class:
//!
//! ```text
//! FL(p) = −α · (1 − p)^γ · log(p)
//! ```
//!
//! where `γ ≥ 0` is the focusing parameter (γ = 0 recovers weighted
//! cross-entropy) and `α ∈ (0, 1)` balances positive/negative classes.
//!
//! Two entry points are provided:
//! * [`binary_focal_loss`] — sigmoid focal loss over independent logits
//!   (the original RetinaNet formulation, one-vs-all per class).
//! * [`multiclass_focal_loss`] — softmax focal loss over a `C`-way logit vector
//!   with an integer target label.
//!
//! All computations are numerically stable: the sigmoid/softmax cross-entropy is
//! evaluated via the log-sum-exp / softplus identities rather than by taking
//! `log` of a probability that may underflow to zero.

use crate::error::{VisionError, VisionResult};

/// Reduction applied to a batch of per-example focal losses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// Arithmetic mean over all elements.
    Mean,
    /// Sum over all elements.
    Sum,
    /// No reduction — return the per-element values.
    None,
}

/// Numerically-stable `log(sigmoid(x)) = −softplus(−x)`.
#[inline]
fn log_sigmoid(x: f32) -> f32 {
    // softplus(z) = max(z, 0) + ln(1 + e^{−|z|})
    let z = -x;
    let sp = z.max(0.0) + (-z.abs()).exp().ln_1p();
    -sp
}

/// Numerically-stable sigmoid.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Binary (sigmoid) focal loss for a single logit/target pair.
///
/// `target` must be `0.0` (negative) or `1.0` (positive). `alpha` weights the
/// positive class and `1 − alpha` the negative class. `gamma ≥ 0`.
///
/// Implements `FL = −α_t · (1 − p_t)^γ · log(p_t)` where `p_t` is the
/// probability assigned to the ground-truth class.
pub fn binary_focal_loss_one(logit: f32, target: f32, alpha: f32, gamma: f32) -> VisionResult<f32> {
    if !logit.is_finite() {
        return Err(VisionError::NonFinite("focal logit"));
    }
    if !(target == 0.0 || target == 1.0) {
        return Err(VisionError::Internal(format!(
            "binary focal target must be 0 or 1, got {target}"
        )));
    }
    if !(0.0..=1.0).contains(&alpha) {
        return Err(VisionError::Internal(format!(
            "focal alpha must be in [0,1], got {alpha}"
        )));
    }
    if gamma < 0.0 {
        return Err(VisionError::Internal(format!(
            "focal gamma must be >= 0, got {gamma}"
        )));
    }

    let p = sigmoid(logit);
    // p_t = p for positives, (1 − p) for negatives.
    // log p_t computed stably: log σ(x) for positive, log σ(−x) for negative.
    let (p_t, log_pt, alpha_t) = if target == 1.0 {
        (p, log_sigmoid(logit), alpha)
    } else {
        (1.0 - p, log_sigmoid(-logit), 1.0 - alpha)
    };
    let modulating = (1.0 - p_t).max(0.0).powf(gamma);
    Ok(-alpha_t * modulating * log_pt)
}

/// Binary (sigmoid) focal loss over a batch of independent logits.
///
/// `logits` and `targets` must have equal length; each target is `0.0` or `1.0`.
pub fn binary_focal_loss(
    logits: &[f32],
    targets: &[f32],
    alpha: f32,
    gamma: f32,
    reduction: Reduction,
) -> VisionResult<Vec<f32>> {
    if logits.is_empty() {
        return Err(VisionError::EmptyInput("binary_focal_loss logits"));
    }
    if logits.len() != targets.len() {
        return Err(VisionError::ShapeMismatch {
            lhs: vec![logits.len()],
            rhs: vec![targets.len()],
        });
    }
    let mut per_elem = Vec::with_capacity(logits.len());
    for (&logit, &target) in logits.iter().zip(targets.iter()) {
        per_elem.push(binary_focal_loss_one(logit, target, alpha, gamma)?);
    }
    Ok(reduce(per_elem, reduction))
}

/// Softmax (multiclass) focal loss for one `C`-way logit vector and label.
///
/// `logits` is length `C`; `target` is the ground-truth class index `< C`.
/// `alpha` scales the loss uniformly (set to `1.0` to disable class balancing).
pub fn multiclass_focal_loss_one(
    logits: &[f32],
    target: usize,
    alpha: f32,
    gamma: f32,
) -> VisionResult<f32> {
    if logits.is_empty() {
        return Err(VisionError::EmptyInput("multiclass_focal_loss logits"));
    }
    if target >= logits.len() {
        return Err(VisionError::InvalidNumClasses(logits.len()));
    }
    if gamma < 0.0 {
        return Err(VisionError::Internal(format!(
            "focal gamma must be >= 0, got {gamma}"
        )));
    }
    for &l in logits {
        if !l.is_finite() {
            return Err(VisionError::NonFinite("focal logits"));
        }
    }
    // Stable log-softmax: log p_k = x_k − (m + log Σ_j e^{x_j − m}).
    let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum_exp = 0.0f32;
    for &l in logits {
        sum_exp += (l - m).exp();
    }
    let log_sum_exp = m + sum_exp.ln();
    let log_pt = logits[target] - log_sum_exp;
    let p_t = log_pt.exp().clamp(0.0, 1.0);
    let modulating = (1.0 - p_t).max(0.0).powf(gamma);
    Ok(-alpha * modulating * log_pt)
}

/// Softmax (multiclass) focal loss over a batch of logit rows.
///
/// `logits` is `n × num_classes` (row-major), `targets` has length `n`.
pub fn multiclass_focal_loss(
    logits: &[f32],
    targets: &[usize],
    num_classes: usize,
    alpha: f32,
    gamma: f32,
    reduction: Reduction,
) -> VisionResult<Vec<f32>> {
    if num_classes == 0 {
        return Err(VisionError::InvalidNumClasses(0));
    }
    if targets.is_empty() {
        return Err(VisionError::EmptyInput("multiclass_focal_loss targets"));
    }
    let n = targets.len();
    if logits.len() != n * num_classes {
        return Err(VisionError::ShapeMismatch {
            lhs: vec![logits.len()],
            rhs: vec![n, num_classes],
        });
    }
    let mut per_elem = Vec::with_capacity(n);
    for (row, &target) in targets.iter().enumerate() {
        let start = row * num_classes;
        let logit_row = &logits[start..start + num_classes];
        per_elem.push(multiclass_focal_loss_one(logit_row, target, alpha, gamma)?);
    }
    Ok(reduce(per_elem, reduction))
}

/// Apply the requested reduction to a vector of per-element losses.
fn reduce(per_elem: Vec<f32>, reduction: Reduction) -> Vec<f32> {
    match reduction {
        Reduction::None => per_elem,
        Reduction::Sum => vec![per_elem.iter().sum()],
        Reduction::Mean => {
            let n = per_elem.len().max(1) as f32;
            vec![per_elem.iter().sum::<f32>() / n]
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    #[test]
    fn gamma_zero_recovers_weighted_cross_entropy() {
        // With γ=0, FL = −α_t log p_t (weighted BCE).
        let logit = 0.7f32;
        let p = sigmoid(logit);
        let alpha = 0.25;
        let fl = binary_focal_loss_one(logit, 1.0, alpha, 0.0).expect("fl");
        let bce = -alpha * p.ln();
        assert!((fl - bce).abs() < 1e-4, "fl={fl} bce={bce}");
    }

    #[test]
    fn perfect_positive_prediction_near_zero_loss() {
        // Large positive logit, positive target → p≈1 → loss≈0.
        let fl = binary_focal_loss_one(20.0, 1.0, 0.5, 2.0).expect("fl");
        assert!(fl < 1e-6, "fl={fl}");
    }

    #[test]
    fn confident_wrong_prediction_has_large_loss() {
        // Large positive logit but negative target → heavily penalised.
        let fl = binary_focal_loss_one(10.0, 0.0, 0.5, 2.0).expect("fl");
        assert!(fl > 1.0, "fl={fl}");
    }

    #[test]
    fn focal_downweights_easy_examples_relative_to_ce() {
        // For an easy example, FL/CE ratio = (1−p)^γ < 1.
        let logit = 3.0f32; // p ≈ 0.9526
        let p = sigmoid(logit);
        let ce = -p.ln();
        let fl = binary_focal_loss_one(logit, 1.0, 1.0, 2.0).expect("fl");
        let ratio = fl / ce;
        let expected = (1.0 - p).powi(2);
        assert!(
            (ratio - expected).abs() < 1e-4,
            "ratio={ratio} expected={expected}"
        );
        assert!(ratio < 0.01, "easy example not downweighted: {ratio}");
    }

    #[test]
    fn binary_focal_loss_is_nonnegative() {
        let logits = [-2.0, 0.0, 1.5, 4.0];
        let targets = [0.0, 1.0, 0.0, 1.0];
        let per = binary_focal_loss(&logits, &targets, 0.25, 2.0, Reduction::None).expect("ok");
        for v in per {
            assert!(v >= -TOL, "negative loss {v}");
        }
    }

    #[test]
    fn binary_mean_matches_manual_average() {
        let logits = [-1.0, 0.5, 2.0];
        let targets = [0.0, 1.0, 1.0];
        let mean = binary_focal_loss(&logits, &targets, 0.3, 1.5, Reduction::Mean).expect("mean");
        let per = binary_focal_loss(&logits, &targets, 0.3, 1.5, Reduction::None).expect("per");
        let manual = per.iter().sum::<f32>() / per.len() as f32;
        assert!((mean[0] - manual).abs() < TOL);
    }

    #[test]
    fn binary_shape_mismatch_errors() {
        let logits = [0.0, 1.0];
        let targets = [1.0];
        assert!(binary_focal_loss(&logits, &targets, 0.5, 2.0, Reduction::Sum).is_err());
    }

    #[test]
    fn binary_invalid_target_errors() {
        assert!(binary_focal_loss_one(0.0, 0.5, 0.5, 2.0).is_err());
    }

    #[test]
    fn binary_invalid_alpha_gamma_errors() {
        assert!(binary_focal_loss_one(0.0, 1.0, 1.5, 2.0).is_err());
        assert!(binary_focal_loss_one(0.0, 1.0, 0.5, -1.0).is_err());
    }

    #[test]
    fn multiclass_correct_class_low_loss() {
        // Logit strongly favours the true class.
        let logits = [10.0, 0.0, 0.0];
        let fl = multiclass_focal_loss_one(&logits, 0, 1.0, 2.0).expect("fl");
        assert!(fl < 1e-3, "fl={fl}");
    }

    #[test]
    fn multiclass_gamma_zero_is_cross_entropy() {
        let logits = [1.0, 2.0, 0.5];
        let target = 1;
        // log-softmax of target
        let m = 2.0f32;
        let denom = ((1.0 - m).exp() + (2.0 - m).exp() + (0.5 - m).exp()).ln() + m;
        let ce = -(logits[target] - denom);
        let fl = multiclass_focal_loss_one(&logits, target, 1.0, 0.0).expect("fl");
        assert!((fl - ce).abs() < 1e-5, "fl={fl} ce={ce}");
    }

    #[test]
    fn multiclass_batched_sum_and_shape() {
        let logits = [
            1.0, 0.0, 0.0, // row 0, target 0
            0.0, 3.0, 0.0, // row 1, target 1
        ];
        let targets = [0usize, 1usize];
        let s = multiclass_focal_loss(&logits, &targets, 3, 1.0, 2.0, Reduction::Sum).expect("sum");
        assert_eq!(s.len(), 1);
        assert!(s[0] >= 0.0);
    }

    #[test]
    fn multiclass_target_out_of_range_errors() {
        let logits = [1.0, 2.0];
        assert!(multiclass_focal_loss_one(&logits, 5, 1.0, 2.0).is_err());
    }

    #[test]
    fn multiclass_shape_mismatch_errors() {
        let logits = [1.0, 2.0, 3.0]; // 3 != 2*2
        let targets = [0usize, 1usize];
        assert!(multiclass_focal_loss(&logits, &targets, 2, 1.0, 2.0, Reduction::Mean).is_err());
    }

    #[test]
    fn nonfinite_logit_errors() {
        assert!(binary_focal_loss_one(f32::INFINITY, 1.0, 0.5, 2.0).is_err());
        let logits = [1.0, f32::NAN];
        assert!(multiclass_focal_loss_one(&logits, 0, 1.0, 2.0).is_err());
    }
}
