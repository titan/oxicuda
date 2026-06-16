//! Bias Correction (BiC) for class-incremental learning.
//!
//! Implements the post-hoc linear correction from:
//! Wu et al. "Large Scale Incremental Learning." CVPR 2019.
//!
//! After training on incremental classes the classifier is biased toward new
//! classes because the new-task training set is larger relative to the exemplar
//! replay buffer. BiC fixes this with two calibration parameters on the
//! new-class logits: `g(z_new_j) = α · z_new_j + β`, estimated by minimising
//! cross-entropy on a small held-out validation set with vanilla gradient descent.

use crate::error::{ContinualError, ContinualResult};

/// Numerically stable softmax.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&e| e / sum).collect()
}

/// Bias correction parameters for new-class logits.
///
/// Applies `g(z_new_j) = α · z_new_j + β` to every logit that belongs to a
/// new (unseen-until-this-task) class; old-class logits are untouched.
#[derive(Debug, Clone)]
pub struct BicLayer {
    /// Scale factor applied to new-class logits. Initialised to `1.0`.
    pub alpha: f32,
    /// Additive bias applied to new-class logits after scaling. Initialised to `0.0`.
    pub beta: f32,
}

impl Default for BicLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl BicLayer {
    /// Create a new `BicLayer` with identity parameters (`α = 1`, `β = 0`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: 1.0_f32,
            beta: 0.0_f32,
        }
    }

    /// Returns `true` if this layer acts as an identity (α ≈ 1 and β ≈ 0).
    #[must_use]
    pub fn identity(&self) -> bool {
        (self.alpha - 1.0_f32).abs() < 1e-6 && self.beta.abs() < 1e-6
    }

    /// Apply the BiC correction to a full logit vector.
    ///
    /// Only logits at indices `[n_old, len)` are transformed; old-class logits
    /// are returned unchanged.
    ///
    /// # Errors
    ///
    /// - [`ContinualError::EmptyInput`] if `logits` is empty.
    /// - [`ContinualError::DimensionMismatch`] if `n_old > logits.len()`.
    pub fn apply(&self, logits: &[f32], n_old: usize) -> ContinualResult<Vec<f32>> {
        if logits.is_empty() {
            return Err(ContinualError::EmptyInput);
        }
        if n_old > logits.len() {
            return Err(ContinualError::DimensionMismatch {
                expected: n_old,
                got: logits.len(),
            });
        }
        let mut out = logits.to_vec();
        for v in out[n_old..].iter_mut() {
            *v = self.alpha * *v + self.beta;
        }
        Ok(out)
    }
}

/// Configuration for the BiC calibration gradient descent.
#[derive(Debug, Clone)]
pub struct BicConfig {
    /// Learning rate for the two-parameter gradient descent. Must be > 0 and finite.
    pub lr: f32,
    /// Maximum number of gradient descent iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the absolute loss change. Calibration stops early
    /// when `|ΔL| < tol` for three consecutive steps.
    pub tol: f32,
}

impl Default for BicConfig {
    fn default() -> Self {
        Self {
            lr: 0.01_f32,
            max_iter: 2000,
            tol: 1e-5_f32,
        }
    }
}

/// Calibrate BiC parameters on a held-out validation set.
///
/// # Arguments
///
/// * `logits`    — pre-softmax logits, row-major, shape `[n_samples × n_classes]`.
/// * `labels`    — integer class indices for each sample, shape `[n_samples]`.
/// * `n_samples` — number of samples (rows in `logits`).
/// * `n_classes` — total number of classes (old + new).
/// * `n_old`     — number of old classes. Must satisfy `n_old < n_classes`.
/// * `config`    — calibration hyper-parameters.
///
/// # Errors
///
/// - [`ContinualError::EmptyInput`] if `n_samples == 0`.
/// - [`ContinualError::DimensionMismatch`] if `logits.len() != n_samples * n_classes`
///   or `labels.len() != n_samples`.
/// - [`ContinualError::Internal`] if `n_old >= n_classes` or any label is out of range.
/// - [`ContinualError::InvalidLambda`] if `lr <= 0` or not finite.
/// - [`ContinualError::NanEncountered`] if any logit value is NaN.
pub fn calibrate_bic(
    logits: &[f32],
    labels: &[usize],
    n_samples: usize,
    n_classes: usize,
    n_old: usize,
    config: &BicConfig,
) -> ContinualResult<BicLayer> {
    if n_samples == 0 {
        return Err(ContinualError::EmptyInput);
    }
    if logits.len() != n_samples * n_classes {
        return Err(ContinualError::DimensionMismatch {
            expected: n_samples * n_classes,
            got: logits.len(),
        });
    }
    if labels.len() != n_samples {
        return Err(ContinualError::DimensionMismatch {
            expected: n_samples,
            got: labels.len(),
        });
    }
    if n_old >= n_classes {
        return Err(ContinualError::Internal("n_old must be < n_classes".into()));
    }
    for &lbl in labels {
        if lbl >= n_classes {
            return Err(ContinualError::Internal("label out of range".into()));
        }
    }
    if !config.lr.is_finite() || config.lr <= 0.0_f32 {
        return Err(ContinualError::InvalidLambda { lambda: config.lr });
    }
    for &l in logits {
        if l.is_nan() {
            return Err(ContinualError::NanEncountered { location: "logits" });
        }
    }

    const EPS: f32 = 1e-10;
    let n_new = n_classes - n_old;
    let inv_n = 1.0_f32 / n_samples as f32;

    let mut alpha = 1.0_f32;
    let mut beta = 0.0_f32;
    let mut prev_loss = f32::INFINITY;
    let mut consecutive_small = 0usize;
    let mut corrected = vec![0.0_f32; n_classes];

    for _iter in 0..config.max_iter {
        let mut loss = 0.0_f32;
        let mut grad_alpha = 0.0_f32;
        let mut grad_beta = 0.0_f32;

        for s in 0..n_samples {
            let row = &logits[s * n_classes..(s + 1) * n_classes];
            let y = labels[s];

            corrected[..n_old].copy_from_slice(&row[..n_old]);
            for j in 0..n_new {
                corrected[n_old + j] = alpha * row[n_old + j] + beta;
            }

            let probs = softmax(&corrected);
            loss -= (probs[y] + EPS).ln();

            // Gradient via chain rule: ∂L/∂α = Σ_{j≥n_old} (p_j - 1[y==j])·z_j
            //                          ∂L/∂β = Σ_{j≥n_old} (p_j - 1[y==j])
            for j in n_old..n_classes {
                let indicator = if y == j { 1.0_f32 } else { 0.0_f32 };
                let residual = probs[j] - indicator;
                grad_alpha += residual * row[j];
                grad_beta += residual;
            }
        }

        loss *= inv_n;
        grad_alpha *= inv_n;
        grad_beta *= inv_n;

        alpha -= config.lr * grad_alpha;
        beta -= config.lr * grad_beta;

        let delta_loss = (loss - prev_loss).abs();
        if delta_loss < config.tol {
            consecutive_small += 1;
            if consecutive_small >= 3 {
                break;
            }
        } else {
            consecutive_small = 0;
        }
        prev_loss = loss;
    }

    Ok(BicLayer { alpha, beta })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_identity_is_noop() {
        let layer = BicLayer::new();
        let logits = vec![0.5_f32, 1.0, -0.5, 2.0];
        let out = layer
            .apply(&logits, 2)
            .expect("BIC layer application should succeed with valid logits");
        for (a, b) in logits.iter().zip(out.iter()) {
            assert!(
                (a - b).abs() < 1e-7,
                "identity layer must not change logits"
            );
        }
    }

    #[test]
    fn apply_scales_new_class_logits() {
        let layer = BicLayer {
            alpha: 0.5_f32,
            beta: 1.0_f32,
        };
        let logits = vec![3.0_f32, 4.0, 2.0, 6.0];
        let out = layer
            .apply(&logits, 2)
            .expect("BIC layer application should succeed with valid logits");
        assert!((out[0] - 3.0).abs() < 1e-7);
        assert!((out[1] - 4.0).abs() < 1e-7);
        assert!((out[2] - (0.5 * 2.0 + 1.0)).abs() < 1e-7);
        assert!((out[3] - (0.5 * 6.0 + 1.0)).abs() < 1e-7);
    }

    #[test]
    fn calibrate_balanced_data_alpha_near_one() {
        let n_old = 3usize;
        let n_new = 3usize;
        let n_classes = n_old + n_new;
        let n_per_class = 20usize;
        let n_samples = n_classes * n_per_class;
        let mut logits = Vec::with_capacity(n_samples * n_classes);
        let mut labels = Vec::with_capacity(n_samples);
        for cls in 0..n_classes {
            for _ in 0..n_per_class {
                for j in 0..n_classes {
                    logits.push(if j == cls { 5.0_f32 } else { -1.0_f32 });
                }
                labels.push(cls);
            }
        }
        let cfg = BicConfig::default();
        let bic = calibrate_bic(&logits, &labels, n_samples, n_classes, n_old, &cfg)
            .expect("BIC calibration should succeed with valid data");
        assert!(
            (bic.alpha - 1.0).abs() < 0.15,
            "alpha should stay near 1 for balanced data, got {}",
            bic.alpha
        );
    }

    #[test]
    fn calibrate_biased_data_alpha_less_than_one() {
        let n_old = 2usize;
        let n_new = 2usize;
        let n_classes = n_old + n_new;
        let n_per_class = 30usize;
        let n_samples = n_classes * n_per_class;
        const BIAS_INFLATE: f32 = 8.0;
        let mut logits = Vec::with_capacity(n_samples * n_classes);
        let mut labels = Vec::with_capacity(n_samples);
        for cls in 0..n_classes {
            for _ in 0..n_per_class {
                for j in 0..n_classes {
                    let base = if j == cls { 4.0_f32 } else { -1.0_f32 };
                    logits.push(if j >= n_old {
                        base + BIAS_INFLATE
                    } else {
                        base
                    });
                }
                labels.push(cls);
            }
        }
        let cfg = BicConfig {
            lr: 0.01,
            max_iter: 3000,
            tol: 1e-6,
        };
        let bic = calibrate_bic(&logits, &labels, n_samples, n_classes, n_old, &cfg)
            .expect("BIC calibration should succeed with valid data");
        assert!(
            bic.alpha < 1.0,
            "alpha should be < 1.0 to correct inflated new-class logits, got {}",
            bic.alpha
        );
    }

    #[test]
    fn calibrate_empty_returns_empty_input_error() {
        let cfg = BicConfig::default();
        let result = calibrate_bic(&[], &[], 0, 4, 2, &cfg);
        assert!(matches!(result, Err(ContinualError::EmptyInput)));
    }

    #[test]
    fn calibrate_label_out_of_range_returns_internal_error() {
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0];
        let labels = vec![5usize];
        let cfg = BicConfig::default();
        let result = calibrate_bic(&logits, &labels, 1, 4, 2, &cfg);
        assert!(matches!(result, Err(ContinualError::Internal(_))));
    }

    #[test]
    fn calibrate_n_old_ge_n_classes_returns_internal_error() {
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0];
        let labels = vec![0usize];
        let cfg = BicConfig::default();
        let result = calibrate_bic(&logits, &labels, 1, 4, 4, &cfg);
        assert!(matches!(result, Err(ContinualError::Internal(_))));
    }

    #[test]
    fn bic_layer_identity_check() {
        let layer = BicLayer::new();
        assert!(layer.identity(), "fresh BicLayer must report identity");
        let modified = BicLayer {
            alpha: 0.8,
            beta: 0.1,
        };
        assert!(
            !modified.identity(),
            "modified layer must not report identity"
        );
    }

    #[test]
    fn calibrate_n_old_zero_applies_to_all_classes() {
        let n_classes = 4usize;
        let n_per_class = 10usize;
        let n_samples = n_classes * n_per_class;
        let mut logits = Vec::with_capacity(n_samples * n_classes);
        let mut labels = Vec::with_capacity(n_samples);
        for cls in 0..n_classes {
            for _ in 0..n_per_class {
                for j in 0..n_classes {
                    logits.push(if j == cls { 3.0_f32 } else { -0.5_f32 });
                }
                labels.push(cls);
            }
        }
        let cfg = BicConfig::default();
        let bic = calibrate_bic(&logits, &labels, n_samples, n_classes, 0, &cfg);
        assert!(bic.is_ok(), "calibration with n_old=0 should succeed");
        let layer = bic.expect("BIC calibration result should be Ok");
        assert!(layer.alpha.is_finite() && layer.beta.is_finite());
    }

    #[test]
    fn apply_n_old_equals_len_unchanged() {
        let layer = BicLayer {
            alpha: 0.1_f32,
            beta: 5.0_f32,
        };
        let logits = vec![1.0_f32, 2.0, 3.0];
        let out = layer
            .apply(&logits, logits.len())
            .expect("BIC layer application should succeed with valid logits");
        for (a, b) in logits.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }
}
