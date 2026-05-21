//! Multinomial (softmax) logistic regression for K-class classification.
//!
//! Model: `P(y=k | x) = exp(x^T β_k) / Σ_j exp(x^T β_j)`
//!
//! Estimation via **Adam optimizer** (Kingma & Ba 2015) applied to the negative
//! log-likelihood with L2 regularisation.
//!
//! Log-likelihood: `ℓ(B) = Σ_i log P(y_i = k_i | x_i)`  where B ∈ ℝ^{p × K}.
//!
//! Gradient w.r.t. column k: `∂ℓ/∂β_k = X^T (e_k − P_k) − λ β_k`  where
//! `e_k[i] = 1{y_i = k}` is the one-hot indicator and `P_k[i]` is the predicted
//! probability of class k for sample i.
//!
//! # Reference
//! Kingma & Ba (2015), *Adam: A Method for Stochastic Gradient Descent*,
//! ICLR 2015. <https://arxiv.org/abs/1412.6980>

use crate::error::{StatsError, StatsResult};

// ─────────────────────────────── Configuration ───────────────────────────────

/// Configuration for multinomial logistic regression training.
#[derive(Debug, Clone)]
pub struct MultinomialConfig {
    /// Maximum Adam iterations (default 500).
    pub max_iter: usize,
    /// Adam learning rate α (default 0.01).
    pub learning_rate: f64,
    /// L2 regularisation coefficient λ (default 1e-4).
    pub l2_reg: f64,
    /// Gradient-norm convergence tolerance (default 1e-6).
    pub tol: f64,
    /// Prepend intercept column to design matrix (default true).
    pub intercept: bool,
    /// Adam first-moment decay rate β₁ (default 0.9).
    pub beta1: f64,
    /// Adam second-moment decay rate β₂ (default 0.999).
    pub beta2: f64,
}

impl Default for MultinomialConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            learning_rate: 0.01,
            l2_reg: 1e-4,
            tol: 1e-6,
            intercept: true,
            beta1: 0.9,
            beta2: 0.999,
        }
    }
}

// ─────────────────────────────── Fitted model ─────────────────────────────────

/// Fitted multinomial logistic regression model.
#[derive(Debug, Clone)]
pub struct MultinomialFit {
    /// Coefficient matrix B flattened row-major [p × K] where p includes the
    /// intercept column when `intercept=true`.
    pub coefficients: Vec<f64>,
    /// Number of classes K.
    pub n_classes: usize,
    /// Number of feature columns p (including intercept if applicable).
    pub n_features: usize,
    /// Log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Number of Adam iterations performed.
    pub n_iter: usize,
    /// Whether the algorithm converged within `max_iter`.
    pub converged: bool,
    /// Sorted unique class labels seen during fit.
    pub class_labels: Vec<usize>,
    /// Whether the model was fit with an intercept.
    pub(crate) has_intercept: bool,
    /// Number of raw input features (without intercept).
    pub(crate) n_raw_features: usize,
}

// ─────────────────────── Numerical helper functions ───────────────────────────

/// Numerically stable softmax over a slice → in-place.
///
/// Subtracts the maximum before exponentiation to prevent overflow.
fn softmax_inplace(logits: &mut [f64]) {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for v in logits.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    } else {
        // Degenerate: uniform
        let k = logits.len() as f64;
        for v in logits.iter_mut() {
            *v = 1.0 / k;
        }
    }
}

/// Euclidean norm of a slice.
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// ─────────────────────── Design matrix construction ───────────────────────────

/// Build the full design matrix X_full [n × p] (row-major), prepending an intercept
/// column when requested.
///
/// `x_raw` is row-major `[n_samples × n_raw_features]`.
fn build_design(x_raw: &[f64], n: usize, n_raw: usize, intercept: bool) -> Vec<f64> {
    let p = if intercept { n_raw + 1 } else { n_raw };
    let mut xf = vec![0.0; n * p];
    for i in 0..n {
        let mut col = 0usize;
        if intercept {
            xf[i * p] = 1.0;
            col = 1;
        }
        for j in 0..n_raw {
            xf[i * p + col + j] = x_raw[i * n_raw + j];
        }
    }
    xf
}

// ──────────────── Forward pass: probabilities [n × K] ────────────────────────

/// Compute softmax probabilities P[n × K] (row-major) from design matrix X[n × p]
/// and coefficient matrix B[p × K] (row-major).
///
/// Performs stable row-wise softmax.
fn compute_proba(x_full: &[f64], b: &[f64], n: usize, p: usize, k: usize) -> Vec<f64> {
    let mut prob = vec![0.0; n * k];
    for i in 0..n {
        // η_k = x_i^T β_k  for each class k
        let mut logits = vec![0.0; k];
        for cls in 0..k {
            let mut eta = 0.0;
            for j in 0..p {
                eta += x_full[i * p + j] * b[j * k + cls];
            }
            logits[cls] = eta;
        }
        softmax_inplace(&mut logits);
        for cls in 0..k {
            prob[i * k + cls] = logits[cls];
        }
    }
    prob
}

// ───────────────────────── Log-likelihood computation ─────────────────────────

/// Compute the regularised log-likelihood.
///
/// `prob` is [n × K], `y_raw` is [n] with values in `0..k`.
fn compute_log_likelihood(prob: &[f64], y_raw: &[usize], n: usize, k: usize) -> f64 {
    let mut ll = 0.0;
    for i in 0..n {
        let cls = y_raw[i];
        let p_i = prob[i * k + cls].max(1e-300);
        ll += p_i.ln();
    }
    ll
}

// ─────────────────────── Gradient computation (Adam step) ─────────────────────

/// Compute gradient of ℓ w.r.t. B[p × K], **plus** L2 penalty gradient.
///
/// `∂ℓ/∂β_{jk} = Σ_i x_{ij} (1{y_i=k} − P_{ik}) − λ β_{jk}`
/// Note: for the intercept column (j=0 when `no_reg_intercept=true`) we skip
/// L2 regularisation.
fn compute_gradient(
    x_full: &[f64],
    prob: &[f64],
    y_raw: &[usize],
    b: &[f64],
    n: usize,
    p: usize,
    k: usize,
    l2_reg: f64,
    has_intercept: bool,
) -> Vec<f64> {
    // grad[j * k + cls] = ∂ℓ/∂β_{j,cls}
    let mut grad = vec![0.0; p * k];
    for i in 0..n {
        let cls_true = y_raw[i];
        for cls in 0..k {
            let delta = if cls == cls_true {
                1.0 - prob[i * k + cls]
            } else {
                -prob[i * k + cls]
            };
            for j in 0..p {
                grad[j * k + cls] += x_full[i * p + j] * delta;
            }
        }
    }
    // Subtract L2 gradient (skip intercept row when present)
    let start_j = if has_intercept { 1 } else { 0 };
    for j in start_j..p {
        for cls in 0..k {
            grad[j * k + cls] -= l2_reg * b[j * k + cls];
        }
    }
    grad
}

// ─────────────────────────────── Main fitter ──────────────────────────────────

/// Fit a multinomial logistic regression model using Adam optimizer.
///
/// # Arguments
/// * `x` — raw design matrix [n_samples × n_features] row-major (no intercept).
/// * `y` — class labels `[n_samples]`, values in `0..n_classes`.
/// * `n_samples` — number of observations n.
/// * `n_features` — number of raw input features (columns in `x`).
/// * `n_classes` — number of classes K (≥ 2).
/// * `cfg` — training configuration.
///
/// # Errors
/// Returns `Err` on empty input, K < 2, shape mismatches, or non-finite inputs.
pub fn multinomial_fit(
    x: &[f64],
    y: &[usize],
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    cfg: &MultinomialConfig,
) -> StatsResult<MultinomialFit> {
    // ── Validation ────────────────────────────────────────────────────────────
    if n_samples == 0 {
        return Err(StatsError::EmptyInput);
    }
    if n_classes < 2 {
        return Err(StatsError::InvalidParameter {
            name: "n_classes".into(),
            reason: format!("must be ≥ 2, got {n_classes}"),
        });
    }
    if x.len() != n_samples * n_features {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n_samples, n_features],
            got: vec![x.len()],
        });
    }
    if y.len() != n_samples {
        return Err(StatsError::DimensionMismatch {
            a: y.len(),
            b: n_samples,
        });
    }
    for (i, &cls) in y.iter().enumerate() {
        if cls >= n_classes {
            return Err(StatsError::IndexOutOfBounds {
                index: cls,
                len: n_classes,
            });
        }
        let _ = i;
    }
    for (i, &v) in x.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }
    if cfg.beta1 <= 0.0 || cfg.beta1 >= 1.0 {
        return Err(StatsError::InvalidParameter {
            name: "beta1".into(),
            reason: format!("must be in (0, 1), got {}", cfg.beta1),
        });
    }
    if cfg.beta2 <= 0.0 || cfg.beta2 >= 1.0 {
        return Err(StatsError::InvalidParameter {
            name: "beta2".into(),
            reason: format!("must be in (0, 1), got {}", cfg.beta2),
        });
    }

    // Unique labels (sorted)
    let mut class_labels: Vec<usize> = y.to_vec();
    class_labels.sort_unstable();
    class_labels.dedup();

    let p = if cfg.intercept {
        n_features + 1
    } else {
        n_features
    };
    let k = n_classes;
    let n = n_samples;
    let alpha = cfg.learning_rate;
    let b1 = cfg.beta1;
    let b2 = cfg.beta2;
    let eps = 1e-8_f64;

    // ── Build design matrix ───────────────────────────────────────────────────
    let x_full = build_design(x, n, n_features, cfg.intercept);

    // ── Initialise B = 0 [p × K] (flat, row-major: row = feature, col = class) ──
    let mut b = vec![0.0; p * k];

    // Adam moment vectors
    let mut m = vec![0.0; p * k]; // first moment
    let mut v = vec![0.0; p * k]; // second moment
    let mut t = 0u64; // time step

    let mut converged = false;
    let mut n_iter = 0usize;

    for _iter in 0..cfg.max_iter {
        n_iter += 1;
        t += 1;

        // Forward: compute probabilities [n × K]
        let prob = compute_proba(&x_full, &b, n, p, k);

        // Gradient (ascending ℓ)
        let grad = compute_gradient(&x_full, &prob, y, &b, n, p, k, cfg.l2_reg, cfg.intercept);

        // Gradient norm for convergence check
        let grad_norm = l2_norm(&grad);

        // Adam update (gradient ascent — maximise log-likelihood)
        let b1t = b1.powi(t as i32);
        let b2t = b2.powi(t as i32);
        for idx in 0..p * k {
            let g = grad[idx];
            m[idx] = b1 * m[idx] + (1.0 - b1) * g;
            v[idx] = b2 * v[idx] + (1.0 - b2) * g * g;
            let m_hat = m[idx] / (1.0 - b1t);
            let v_hat = v[idx] / (1.0 - b2t);
            b[idx] += alpha * m_hat / (v_hat.sqrt() + eps);
        }

        if grad_norm < cfg.tol {
            converged = true;
            break;
        }
    }

    // Final log-likelihood with converged B
    let log_likelihood = {
        let prob = compute_proba(&x_full, &b, n, p, k);
        compute_log_likelihood(&prob, y, n, k)
    };

    Ok(MultinomialFit {
        coefficients: b,
        n_classes: k,
        n_features: p,
        log_likelihood,
        n_iter,
        converged,
        class_labels,
        has_intercept: cfg.intercept,
        n_raw_features: n_features,
    })
}

// ─────────────────────────────── Prediction ───────────────────────────────────

/// Compute predicted class probabilities for new observations.
///
/// Returns a [n_new × K] matrix (row-major) where each row sums to 1.
///
/// `x_new` is row-major `[n_new × n_raw_features]` (no intercept column).
pub fn multinomial_predict_proba(
    fit: &MultinomialFit,
    x_new: &[f64],
    n_new: usize,
) -> StatsResult<Vec<f64>> {
    if n_new == 0 {
        return Ok(Vec::new());
    }
    let n_raw = fit.n_raw_features;
    if x_new.len() != n_new * n_raw {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n_new, n_raw],
            got: vec![x_new.len()],
        });
    }
    let x_full = build_design(x_new, n_new, n_raw, fit.has_intercept);
    let p = fit.n_features;
    let k = fit.n_classes;
    Ok(compute_proba(&x_full, &fit.coefficients, n_new, p, k))
}

/// Predict class labels (argmax of probabilities) for new observations.
///
/// `x_new` is row-major `[n_new × n_raw_features]`.
pub fn multinomial_predict(
    fit: &MultinomialFit,
    x_new: &[f64],
    n_new: usize,
) -> StatsResult<Vec<usize>> {
    let proba = multinomial_predict_proba(fit, x_new, n_new)?;
    let k = fit.n_classes;
    let mut labels = Vec::with_capacity(n_new);
    for i in 0..n_new {
        let row = &proba[i * k..(i + 1) * k];
        let cls = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(j, _)| j)
            .unwrap_or(0);
        labels.push(cls);
    }
    Ok(labels)
}

/// Compute classification accuracy on a labelled dataset.
///
/// Returns the fraction of correctly classified observations ∈ [0, 1].
pub fn multinomial_accuracy(
    fit: &MultinomialFit,
    x: &[f64],
    y: &[usize],
    n: usize,
) -> StatsResult<f64> {
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    if y.len() != n {
        return Err(StatsError::DimensionMismatch { a: y.len(), b: n });
    }
    let preds = multinomial_predict(fit, x, n)?;
    let correct = preds.iter().zip(y.iter()).filter(|(p, t)| *p == *t).count();
    Ok(correct as f64 / n as f64)
}

// ──────────────────────────────────── Tests ───────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Data generators ─────────────────────────────────────────────────────

    /// Build a binary linearly-separable dataset in 2D.
    /// Class 0: x1 ∈ [-3,-1], class 1: x1 ∈ [1,3]  (perfectly separated).
    fn make_binary_data() -> (Vec<f64>, Vec<usize>, usize) {
        let n = 20;
        let mut x = Vec::with_capacity(n * 2);
        let mut y = Vec::with_capacity(n);
        for i in 0..10 {
            // class 0
            let v = -(i as f64) * 0.3 - 1.0;
            x.push(v);
            x.push(0.0);
            y.push(0);
        }
        for i in 0..10 {
            // class 1
            let v = (i as f64) * 0.3 + 1.0;
            x.push(v);
            x.push(0.0);
            y.push(1);
        }
        (x, y, n)
    }

    /// Build a 3-class linearly-separable dataset (one feature per class).
    /// Class k has x[0] ∈ [k*4, k*4+1].
    fn make_3class_data() -> (Vec<f64>, Vec<usize>, usize) {
        let obs_per_class = 15;
        let n = obs_per_class * 3;
        let mut x = Vec::with_capacity(n * 2);
        let mut y = Vec::with_capacity(n);
        for cls in 0..3usize {
            for i in 0..obs_per_class {
                let v1 = cls as f64 * 10.0 + i as f64 * 0.05;
                let v2 = 0.0;
                x.push(v1);
                x.push(v2);
                y.push(cls);
            }
        }
        (x, y, n)
    }

    // ── Test 1: 2-class case works ────────────────────────────────────────────
    #[test]
    fn multinomial_fit_binary() {
        let (x, y, n) = make_binary_data();
        let cfg = MultinomialConfig {
            max_iter: 300,
            ..MultinomialConfig::default()
        };
        let fit = multinomial_fit(&x, &y, n, 2, 2, &cfg);
        assert!(fit.is_ok(), "binary fit should return Ok: {:?}", fit);
    }

    // ── Test 2: 3-class linearly separable data converges ─────────────────────
    #[test]
    fn multinomial_fit_3class() {
        let (x, y, n) = make_3class_data();
        let cfg = MultinomialConfig {
            max_iter: 800,
            learning_rate: 0.05,
            ..MultinomialConfig::default()
        };
        let fit = multinomial_fit(&x, &y, n, 2, 3, &cfg).expect("ok");
        // Should achieve reasonable accuracy on separable data
        let acc = multinomial_accuracy(&fit, &x, &y, n).expect("ok");
        assert!(
            acc > 0.8,
            "accuracy {acc:.3} too low on separable 3-class data"
        );
    }

    // ── Test 3: proba shape = [n × K] ────────────────────────────────────────
    #[test]
    fn multinomial_proba_shape() {
        let (x, y, n) = make_binary_data();
        let cfg = MultinomialConfig::default();
        let fit = multinomial_fit(&x, &y, n, 2, 2, &cfg).expect("ok");
        let proba = multinomial_predict_proba(&fit, &x, n).expect("ok");
        assert_eq!(proba.len(), n * 2, "proba should have shape [n × K]");
    }

    // ── Test 4: each row of proba sums to 1 ──────────────────────────────────
    #[test]
    fn multinomial_proba_sums_to_1() {
        let (x, y, n) = make_3class_data();
        let cfg = MultinomialConfig::default();
        let fit = multinomial_fit(&x, &y, n, 2, 3, &cfg).expect("ok");
        let proba = multinomial_predict_proba(&fit, &x, n).expect("ok");
        for i in 0..n {
            let row_sum: f64 = proba[i * 3..(i + 1) * 3].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-9,
                "row {i} sums to {row_sum}, expected 1.0"
            );
        }
    }

    // ── Test 5: predict returns n labels ─────────────────────────────────────
    #[test]
    fn multinomial_predict_shape() {
        let (x, y, n) = make_binary_data();
        let cfg = MultinomialConfig::default();
        let fit = multinomial_fit(&x, &y, n, 2, 2, &cfg).expect("ok");
        let preds = multinomial_predict(&fit, &x, n).expect("ok");
        assert_eq!(preds.len(), n, "predict should return n labels");
    }

    // ── Test 6: predicted labels in [0, K) ───────────────────────────────────
    #[test]
    fn multinomial_predict_labels_in_range() {
        let (x, y, n) = make_3class_data();
        let cfg = MultinomialConfig::default();
        let fit = multinomial_fit(&x, &y, n, 2, 3, &cfg).expect("ok");
        let preds = multinomial_predict(&fit, &x, n).expect("ok");
        for &lbl in &preds {
            assert!(lbl < 3, "label {lbl} out of range [0, 3)");
        }
    }

    // ── Test 7: 100% accuracy on well-separated data ──────────────────────────
    #[test]
    fn multinomial_accuracy_perfect_data() {
        let (x, y, n) = make_3class_data();
        // More iterations and higher LR for perfectly-separable data
        let cfg = MultinomialConfig {
            max_iter: 2000,
            learning_rate: 0.1,
            l2_reg: 0.0,
            ..MultinomialConfig::default()
        };
        let fit = multinomial_fit(&x, &y, n, 2, 3, &cfg).expect("ok");
        let acc = multinomial_accuracy(&fit, &x, &y, n).expect("ok");
        assert!(acc > 0.95, "expected near-perfect accuracy, got {acc:.3}");
    }

    // ── Test 8: n_samples=0 returns Err ──────────────────────────────────────
    #[test]
    fn multinomial_empty_error() {
        let cfg = MultinomialConfig::default();
        let result = multinomial_fit(&[], &[], 0, 2, 3, &cfg);
        assert!(result.is_err(), "empty input should return Err");
    }

    // ── Test 9: K=1 returns Err ───────────────────────────────────────────────
    #[test]
    fn multinomial_single_class_error() {
        let cfg = MultinomialConfig::default();
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0usize, 0, 0];
        let result = multinomial_fit(&x, &y, 3, 1, 1, &cfg);
        assert!(result.is_err(), "K=1 should return Err");
    }

    // ── Test 10: log_likelihood < 0 (negative log-likelihood) ────────────────
    #[test]
    fn multinomial_log_likelihood_finite() {
        let (x, y, n) = make_binary_data();
        let cfg = MultinomialConfig::default();
        let fit = multinomial_fit(&x, &y, n, 2, 2, &cfg).expect("ok");
        assert!(
            fit.log_likelihood < 0.0,
            "log-likelihood should be negative (got {})",
            fit.log_likelihood
        );
        assert!(
            fit.log_likelihood.is_finite(),
            "log-likelihood should be finite"
        );
    }

    // ── Test 11: shape mismatch on x_new returns Err ─────────────────────────
    #[test]
    fn multinomial_predict_shape_mismatch() {
        let (x, y, n) = make_binary_data();
        let cfg = MultinomialConfig::default();
        let fit = multinomial_fit(&x, &y, n, 2, 2, &cfg).expect("ok");
        // Provide wrong number of features in x_new
        let x_bad = vec![1.0]; // only 1 feature instead of 2
        let result = multinomial_predict_proba(&fit, &x_bad, 1);
        assert!(result.is_err(), "mismatched x_new should return Err");
    }

    // ── Test 12: coefficient matrix has correct shape [p × K] ────────────────
    #[test]
    fn multinomial_coeff_shape() {
        let (x, y, n) = make_3class_data();
        let cfg = MultinomialConfig {
            intercept: true,
            ..MultinomialConfig::default()
        };
        let fit = multinomial_fit(&x, &y, n, 2, 3, &cfg).expect("ok");
        // p = n_raw_features + 1 (intercept), K = 3
        let expected_len = fit.n_features * fit.n_classes;
        assert_eq!(
            fit.coefficients.len(),
            expected_len,
            "coefficient vector length should be p × K = {expected_len}"
        );
    }

    // ── Test 13: n_classes and class_labels consistent ────────────────────────
    #[test]
    fn multinomial_class_labels_consistent() {
        let (x, y, n) = make_3class_data();
        let cfg = MultinomialConfig::default();
        let fit = multinomial_fit(&x, &y, n, 2, 3, &cfg).expect("ok");
        assert_eq!(
            fit.class_labels.len(),
            3,
            "should have 3 unique class labels"
        );
        assert_eq!(fit.n_classes, 3);
        // Labels should be sorted: [0, 1, 2]
        assert_eq!(fit.class_labels, vec![0usize, 1, 2]);
    }

    // ── Test 14: no-intercept model produces correct coefficient shape ─────────
    #[test]
    fn multinomial_no_intercept_shape() {
        let (x, y, n) = make_binary_data();
        let cfg = MultinomialConfig {
            intercept: false,
            ..MultinomialConfig::default()
        };
        let fit = multinomial_fit(&x, &y, n, 2, 2, &cfg).expect("ok");
        // p = n_raw_features (no intercept), K = 2
        assert_eq!(fit.n_features, 2, "no-intercept: p = n_raw_features");
        assert_eq!(fit.coefficients.len(), 2 * 2);
    }

    // ── Test 15: accuracy helper returns 0.0..=1.0 ────────────────────────────
    #[test]
    fn multinomial_accuracy_in_range() {
        let (x, y, n) = make_binary_data();
        let cfg = MultinomialConfig::default();
        let fit = multinomial_fit(&x, &y, n, 2, 2, &cfg).expect("ok");
        let acc = multinomial_accuracy(&fit, &x, &y, n).expect("ok");
        assert!((0.0..=1.0).contains(&acc), "accuracy {acc} out of range");
    }
}
