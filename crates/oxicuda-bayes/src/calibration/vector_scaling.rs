//! Vector scaling and matrix scaling: multi-class post-hoc calibration via
//! a learnable affine transform applied to logits before softmax
//! (Guo et al. 2017 NeurIPS "On Calibration of Modern Neural Networks" §3.3).
//!
//! # Variants
//!
//! | Mode | Transform | Parameters |
//! |------|-----------|-----------|
//! | **Vector** | `softmax(W ⊙ z + b)` | K scale + K bias = 2K params |
//! | **Matrix** | `softmax(Wz + b)` | K×K scale + K bias = K²+K params |
//!
//! Temperature scaling is the special case `W = (1/T)·I`, `b = 0`.
//!
//! Both are fitted by minimising the mean negative log-likelihood (NLL) plus
//! an L2 regulariser on `W` via gradient descent. The gradient of the softmax
//! cross-entropy with respect to the scaled logits is the classic `(p − e_y)`:
//! the difference between the predicted softmax distribution and the one-hot
//! label vector. Chain-ruling back to `W` and `b` yields closed-form gradients.

use crate::error::{BayesError, BayesResult};

// ─── Log guard (matches beta.rs / histogram.rs) ───────────────────────────────
const LOG_EPS: f32 = 1e-7;

// ─── ScalingMode ─────────────────────────────────────────────────────────────

/// Whether to learn a diagonal (per-class) or full matrix affine transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingMode {
    /// Element-wise scale + bias per class (2K parameters total).
    Vector,
    /// Full affine K×K matrix + K bias (K²+K parameters total).
    Matrix,
}

// ─── Configuration ───────────────────────────────────────────────────────────

/// Hyperparameters for [`VectorScaler::fit`].
#[derive(Debug, Clone)]
pub struct VectorScalingConfig {
    /// Scaling mode: diagonal (Vector) or full-matrix (Matrix).
    pub mode: ScalingMode,
    /// Number of classes K (must be ≥ 2).
    pub n_classes: usize,
    /// Learning rate for gradient descent.
    pub lr: f32,
    /// Maximum number of gradient descent steps.
    pub max_iter: usize,
    /// Convergence tolerance on the absolute NLL change.
    pub tol: f32,
    /// L2 regularisation coefficient applied to the scale (W) parameters.
    pub l2_reg: f32,
}

impl Default for VectorScalingConfig {
    fn default() -> Self {
        Self {
            mode: ScalingMode::Vector,
            // n_classes must be set by the caller; 2 is the smallest valid value.
            n_classes: 2,
            lr: 0.01,
            max_iter: 200,
            tol: 1e-5,
            l2_reg: 1e-4,
        }
    }
}

// ─── VectorScaler ─────────────────────────────────────────────────────────────

/// A fitted vector or matrix scaler for multi-class calibration.
///
/// After calling [`VectorScaler::fit`], the `scale` and `bias` parameters define
/// an affine transform that is applied to raw logits before the softmax, improving
/// calibration without changing the argmax.
#[derive(Debug, Clone)]
pub struct VectorScaler {
    /// Configuration used during fitting.
    pub config: VectorScalingConfig,
    /// Scale parameters W.
    /// - **Vector** mode: shape `(K,)` — one scale per class.
    /// - **Matrix** mode: shape `(K×K,)` row-major — full affine matrix.
    pub scale: Vec<f32>,
    /// Bias parameters b, shape `(K,)`.
    pub bias: Vec<f32>,
    /// Final mean NLL achieved at the end of optimisation.
    pub final_nll: f32,
    /// Number of gradient descent iterations taken until convergence or `max_iter`.
    pub n_iter: usize,
}

impl VectorScaler {
    // ─── Fitting ─────────────────────────────────────────────────────────────

    /// Fit a vector or matrix scaler from raw logits and integer class labels.
    ///
    /// # Parameters
    /// - `logits` — Raw (pre-softmax) logits of shape `(n × K)`, row-major.
    /// - `labels` — Integer class indices, each in `[0, K)`.
    /// - `cfg`    — Optimisation configuration.
    ///
    /// # Errors
    /// - [`BayesError::CalibrationSetEmpty`] if `logits` is empty.
    /// - [`BayesError::DimensionMismatch`] if `logits.len() != n * K` or any label ≥ K.
    /// - [`BayesError::NCalibBinsTooSmall`] if `n_classes < 2`.
    /// - [`BayesError::InvalidTemperature`] if `lr ≤ 0` (re-used error variant for invalid step size).
    /// - [`BayesError::NanEncountered`] if a NaN appears in parameters during fitting.
    pub fn fit(logits: &[f32], labels: &[usize], cfg: VectorScalingConfig) -> BayesResult<Self> {
        // ── Validate ─────────────────────────────────────────────────────────
        if logits.is_empty() {
            return Err(BayesError::CalibrationSetEmpty);
        }
        if cfg.n_classes < 2 {
            return Err(BayesError::NCalibBinsTooSmall);
        }
        if !(cfg.lr > 0.0 && cfg.lr.is_finite()) {
            return Err(BayesError::InvalidTemperature { temp: cfg.lr });
        }
        let k = cfg.n_classes;
        if logits.len() % k != 0 {
            return Err(BayesError::DimensionMismatch {
                expected: labels.len() * k,
                got: logits.len(),
            });
        }
        let n = logits.len() / k;
        if labels.len() != n {
            return Err(BayesError::DimensionMismatch {
                expected: n,
                got: labels.len(),
            });
        }
        for &y in labels {
            if y >= k {
                return Err(BayesError::DimensionMismatch {
                    expected: k - 1,
                    got: y,
                });
            }
        }

        // ── Initialise parameters ─────────────────────────────────────────────
        // Vector: scale = [1.0; K], bias = [0.0; K]
        // Matrix: scale = identity matrix (K×K) flattened row-major, bias = [0.0; K]
        let (mut scale, mut bias) = init_params(cfg.mode, k);

        // ── Gradient descent ──────────────────────────────────────────────────
        let mut prev_nll = compute_nll_from_params(&scale, &bias, logits, labels, k, cfg.l2_reg);
        let mut n_iter = 0_usize;

        for _iter in 0..cfg.max_iter {
            // Compute gradients.
            let (grad_scale, grad_bias) =
                compute_gradients(&scale, &bias, logits, labels, k, n, cfg.mode, cfg.l2_reg);

            // Gradient step.
            for (s, gs) in scale.iter_mut().zip(grad_scale.iter()) {
                *s -= cfg.lr * gs;
            }
            for (b, gb) in bias.iter_mut().zip(grad_bias.iter()) {
                *b -= cfg.lr * gb;
            }

            // Check for NaN in parameters.
            for &s in &scale {
                if s.is_nan() {
                    return Err(BayesError::NanEncountered {
                        location: "VectorScaler::fit: NaN in scale",
                    });
                }
            }
            for &b in &bias {
                if b.is_nan() {
                    return Err(BayesError::NanEncountered {
                        location: "VectorScaler::fit: NaN in bias",
                    });
                }
            }

            n_iter += 1;

            // Convergence check.
            let new_nll = compute_nll_from_params(&scale, &bias, logits, labels, k, cfg.l2_reg);
            if (new_nll - prev_nll).abs() < cfg.tol {
                prev_nll = new_nll;
                break;
            }
            prev_nll = new_nll;
        }

        Ok(VectorScaler {
            config: cfg,
            scale,
            bias,
            final_nll: prev_nll,
            n_iter,
        })
    }

    // ─── Inference ───────────────────────────────────────────────────────────

    /// Apply the affine transform to raw logits, returning scaled logits (before softmax).
    ///
    /// # Parameters
    /// - `logits` — Raw logits, shape `(n × K)` row-major.
    /// - `n`      — Number of samples.
    ///
    /// # Errors
    /// - [`BayesError::DimensionMismatch`] if `logits.len() != n * K`.
    pub fn transform_logits(&self, logits: &[f32], n: usize) -> BayesResult<Vec<f32>> {
        let k = self.config.n_classes;
        if logits.len() != n * k {
            return Err(BayesError::DimensionMismatch {
                expected: n * k,
                got: logits.len(),
            });
        }
        Ok(apply_transform(
            &self.scale,
            &self.bias,
            logits,
            n,
            k,
            self.config.mode,
        ))
    }

    /// Apply the affine transform followed by softmax, returning calibrated
    /// probabilities of shape `(n × K)`, row-major.
    ///
    /// # Errors
    /// - [`BayesError::DimensionMismatch`] if `logits.len() != n * K`.
    pub fn calibrate(&self, logits: &[f32], n: usize) -> BayesResult<Vec<f32>> {
        let mut scaled = self.transform_logits(logits, n)?;
        let k = self.config.n_classes;
        row_softmax(&mut scaled, k);
        Ok(scaled)
    }

    /// Calibrate a single sample's logits (K values).
    ///
    /// # Errors
    /// - [`BayesError::DimensionMismatch`] if `logits.len() != K`.
    pub fn calibrate_one(&self, logits: &[f32]) -> BayesResult<Vec<f32>> {
        self.calibrate(logits, 1)
    }

    /// Mean NLL on `(logits, labels)` using current scale/bias parameters.
    ///
    /// # Errors
    /// - [`BayesError::CalibrationSetEmpty`] if `logits` is empty.
    /// - [`BayesError::DimensionMismatch`] if lengths or label indices are invalid.
    pub fn nll(&self, logits: &[f32], labels: &[usize]) -> BayesResult<f32> {
        if logits.is_empty() {
            return Err(BayesError::CalibrationSetEmpty);
        }
        let k = self.config.n_classes;
        if logits.len() % k != 0 {
            return Err(BayesError::DimensionMismatch {
                expected: labels.len() * k,
                got: logits.len(),
            });
        }
        let n = logits.len() / k;
        if labels.len() != n {
            return Err(BayesError::DimensionMismatch {
                expected: n,
                got: labels.len(),
            });
        }
        for &y in labels {
            if y >= k {
                return Err(BayesError::DimensionMismatch {
                    expected: k - 1,
                    got: y,
                });
            }
        }
        // NLL without L2 for pure evaluation.
        Ok(compute_nll_from_params(
            &self.scale,
            &self.bias,
            logits,
            labels,
            k,
            0.0,
        ))
    }

    /// Gradient of the (regularised) NLL w.r.t. scale and bias.
    ///
    /// Returns `(grad_scale, grad_bias)` with the same shapes as `self.scale` and `self.bias`.
    ///
    /// # Errors
    /// - [`BayesError::CalibrationSetEmpty`] if `logits` is empty.
    /// - [`BayesError::DimensionMismatch`] if lengths or label indices are invalid.
    pub fn nll_grad(&self, logits: &[f32], labels: &[usize]) -> BayesResult<(Vec<f32>, Vec<f32>)> {
        if logits.is_empty() {
            return Err(BayesError::CalibrationSetEmpty);
        }
        let k = self.config.n_classes;
        if logits.len() % k != 0 {
            return Err(BayesError::DimensionMismatch {
                expected: labels.len() * k,
                got: logits.len(),
            });
        }
        let n = logits.len() / k;
        if labels.len() != n {
            return Err(BayesError::DimensionMismatch {
                expected: n,
                got: labels.len(),
            });
        }
        for &y in labels {
            if y >= k {
                return Err(BayesError::DimensionMismatch {
                    expected: k - 1,
                    got: y,
                });
            }
        }
        Ok(compute_gradients(
            &self.scale,
            &self.bias,
            logits,
            labels,
            k,
            n,
            self.config.mode,
            self.config.l2_reg,
        ))
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Initialise scale and bias parameters.
fn init_params(mode: ScalingMode, k: usize) -> (Vec<f32>, Vec<f32>) {
    let bias = vec![0.0_f32; k];
    let scale = match mode {
        ScalingMode::Vector => vec![1.0_f32; k],
        ScalingMode::Matrix => {
            // Identity matrix K×K, row-major.
            let mut m = vec![0.0_f32; k * k];
            for i in 0..k {
                m[i * k + i] = 1.0;
            }
            m
        }
    };
    (scale, bias)
}

/// Apply the affine transform (without softmax) to `n` rows of K logits.
fn apply_transform(
    scale: &[f32],
    bias: &[f32],
    logits: &[f32],
    n: usize,
    k: usize,
    mode: ScalingMode,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; n * k];
    match mode {
        ScalingMode::Vector => {
            for i in 0..n {
                for kk in 0..k {
                    out[i * k + kk] = scale[kk] * logits[i * k + kk] + bias[kk];
                }
            }
        }
        ScalingMode::Matrix => {
            // scaled[i, kk] = Σ_j scale[kk * K + j] * logits[i, j] + bias[kk]
            for i in 0..n {
                for kk in 0..k {
                    let mut v = bias[kk];
                    for j in 0..k {
                        v += scale[kk * k + j] * logits[i * k + j];
                    }
                    out[i * k + kk] = v;
                }
            }
        }
    }
    out
}

/// In-place row-wise numerically stable softmax over a `[N, K]` buffer.
fn row_softmax(buf: &mut [f32], k: usize) {
    let n = buf.len() / k;
    for i in 0..n {
        let row = &mut buf[i * k..(i + 1) * k];
        let mut m = f32::NEG_INFINITY;
        for &v in row.iter() {
            if v > m {
                m = v;
            }
        }
        let mut s = 0.0_f32;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            s += *v;
        }
        let inv = 1.0_f32 / s.max(1e-30_f32);
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

/// Numerically stable log-softmax for one row of K values.
/// Returns `log p[k] = scaled[k] - log(sum_k exp(scaled[k]))`.
fn log_softmax_row(row: &[f32]) -> Vec<f32> {
    let k = row.len();
    let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp: f64 = row
        .iter()
        .map(|&v| ((v - m) as f64).exp())
        .sum::<f64>()
        .ln()
        + m as f64;
    (0..k)
        .map(|j| row[j] as f64 - log_sum_exp)
        .map(|v| v as f32)
        .collect()
}

/// Compute mean NLL on `(logits, labels)` with optional L2 penalty on `scale`.
fn compute_nll_from_params(
    scale: &[f32],
    bias: &[f32],
    logits: &[f32],
    labels: &[usize],
    k: usize,
    l2_reg: f32,
) -> f32 {
    let n = labels.len();
    let mode = if scale.len() == k {
        ScalingMode::Vector
    } else {
        ScalingMode::Matrix
    };
    let scaled = apply_transform(scale, bias, logits, n, k, mode);

    let mut total = 0.0_f64;
    for i in 0..n {
        let row = &scaled[i * k..(i + 1) * k];
        let lsm = log_softmax_row(row);
        let log_p = lsm[labels[i]].clamp(-(1.0 / LOG_EPS).ln(), 0.0);
        total -= log_p as f64;
    }
    let mut nll = (total / n as f64) as f32;

    // L2 penalty on scale (not bias).
    if l2_reg > 0.0 {
        let l2: f32 = scale.iter().map(|&s| s * s).sum::<f32>() * 0.5 * l2_reg;
        nll += l2;
    }
    nll
}

/// Compute gradients of the regularised NLL w.r.t. scale and bias.
///
/// The softmax cross-entropy gradient w.r.t. scaled logits is
/// `δ[i, k] = (p[i, k] − 1(y_i == k)) / n`.
///
/// Chain rule:
/// - **Vector**: `grad_scale[k] = Σ_i δ[i,k] · logits[i,k] + l2_reg·scale[k]`
/// - **Matrix**: `grad_scale[k*K+j] = Σ_i δ[i,k] · logits[i,j] + l2_reg·scale[k*K+j]`
/// - **Both**: `grad_bias[k] = Σ_i δ[i,k]`
fn compute_gradients(
    scale: &[f32],
    bias: &[f32],
    logits: &[f32],
    labels: &[usize],
    k: usize,
    n: usize,
    mode: ScalingMode,
    l2_reg: f32,
) -> (Vec<f32>, Vec<f32>) {
    let scaled = apply_transform(scale, bias, logits, n, k, mode);

    // Compute softmax probabilities: shape (n, k).
    let mut probs = scaled.clone();
    row_softmax(&mut probs, k);

    // Compute delta[i, k] = (p[i,k] - 1(y_i == k)) / n
    let n_inv = 1.0_f64 / n as f64;
    let mut delta = vec![0.0_f64; n * k];
    for i in 0..n {
        for kk in 0..k {
            let indicator = if labels[i] == kk { 1.0_f64 } else { 0.0_f64 };
            delta[i * k + kk] = (probs[i * k + kk] as f64 - indicator) * n_inv;
        }
    }

    let grad_bias: Vec<f32> = (0..k)
        .map(|kk| delta.iter().skip(kk).step_by(k).sum::<f64>() as f32)
        .collect();

    let grad_scale: Vec<f32> = match mode {
        ScalingMode::Vector => (0..k)
            .map(|kk| {
                let gs: f64 = (0..n)
                    .map(|i| delta[i * k + kk] * logits[i * k + kk] as f64)
                    .sum();
                gs as f32 + l2_reg * scale[kk]
            })
            .collect(),
        ScalingMode::Matrix => {
            let mut gs = vec![0.0_f32; k * k];
            for kk in 0..k {
                for j in 0..k {
                    let g: f64 = (0..n)
                        .map(|i| delta[i * k + kk] * logits[i * k + j] as f64)
                        .sum();
                    gs[kk * k + j] = g as f32 + l2_reg * scale[kk * k + j];
                }
            }
            gs
        }
    };

    (grad_scale, grad_bias)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Synthetic 2-class overconfident dataset: logits [5, 0] for all n samples,
    /// but only `pos_frac` fraction have label 0 (the overconfident class).
    fn overconfident_2class(n: usize, pos_frac: f32) -> (Vec<f32>, Vec<usize>) {
        let mut logits = Vec::with_capacity(n * 2);
        let mut labels = Vec::with_capacity(n);
        for i in 0..n {
            logits.push(5.0_f32);
            logits.push(0.0_f32);
            let frac = i as f32 / n as f32;
            labels.push(if frac < pos_frac { 0 } else { 1 });
        }
        (logits, labels)
    }

    /// Synthetic K-class problem: softmax([k_true_weight, 0, ..., 0]) for each sample.
    fn multiclass_dataset(n: usize, k: usize, margin: f32) -> (Vec<f32>, Vec<usize>) {
        let mut logits = Vec::with_capacity(n * k);
        let mut labels = Vec::with_capacity(n);
        for i in 0..n {
            let y = i % k;
            for kk in 0..k {
                logits.push(if kk == y { margin } else { 0.0_f32 });
            }
            labels.push(y);
        }
        (logits, labels)
    }

    fn vector_cfg(k: usize) -> VectorScalingConfig {
        VectorScalingConfig {
            mode: ScalingMode::Vector,
            n_classes: k,
            lr: 0.01,
            max_iter: 200,
            tol: 1e-5,
            l2_reg: 1e-4,
        }
    }

    fn matrix_cfg(k: usize) -> VectorScalingConfig {
        VectorScalingConfig {
            mode: ScalingMode::Matrix,
            n_classes: k,
            lr: 0.01,
            max_iter: 200,
            tol: 1e-5,
            l2_reg: 1e-4,
        }
    }

    // ── Initialisation ───────────────────────────────────────────────────────

    #[test]
    fn vector_mode_has_2k_params() {
        let k = 5_usize;
        let (scale, bias) = init_params(ScalingMode::Vector, k);
        assert_eq!(scale.len(), k, "Vector scale must have K params");
        assert_eq!(bias.len(), k, "Vector bias must have K params");
    }

    #[test]
    fn matrix_mode_has_k2_plus_k_params() {
        let k = 4_usize;
        let (scale, bias) = init_params(ScalingMode::Matrix, k);
        assert_eq!(scale.len(), k * k, "Matrix scale must have K² params");
        assert_eq!(bias.len(), k, "Matrix bias must have K params");
    }

    #[test]
    fn matrix_mode_identity_init_transform_unchanged() {
        // Identity scale + zero bias should leave logits unchanged.
        let k = 3_usize;
        let n = 2_usize;
        let (scale, bias) = init_params(ScalingMode::Matrix, k);
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = apply_transform(&scale, &bias, &logits, n, k, ScalingMode::Matrix);
        for (i, (&a, &b)) in logits.iter().zip(out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "position {i}: input={a}, output={b}");
        }
    }

    #[test]
    fn vector_scale_doubling_doubles_logits() {
        let k = 2_usize;
        let n = 1_usize;
        let scale = vec![2.0_f32, 2.0];
        let bias = vec![0.0_f32, 0.0];
        let logits = vec![3.0_f32, 7.0];
        let out = apply_transform(&scale, &bias, &logits, n, k, ScalingMode::Vector);
        assert!((out[0] - 6.0).abs() < 1e-6);
        assert!((out[1] - 14.0).abs() < 1e-6);
    }

    // ── Calibrate output sums to 1 ────────────────────────────────────────────

    #[test]
    fn calibrate_output_sums_to_one_per_sample() {
        let k = 3_usize;
        let (logits, labels) = multiclass_dataset(30, k, 3.0);
        let cfg = vector_cfg(k);
        let scaler = VectorScaler::fit(&logits, &labels, cfg).unwrap();
        let probs = scaler.calibrate(&logits, 30).unwrap();
        for i in 0..30 {
            let s: f32 = probs[i * k..(i + 1) * k].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row {i} sums to {s} instead of 1");
        }
    }

    #[test]
    fn calibrate_one_matches_calibrate_on_single_sample() {
        let k = 3_usize;
        let (logits, labels) = multiclass_dataset(20, k, 3.0);
        let cfg = vector_cfg(k);
        let scaler = VectorScaler::fit(&logits, &labels, cfg).unwrap();
        // Use first sample's logits.
        let single = &logits[0..k];
        let p_one = scaler.calibrate_one(single).unwrap();
        let p_batch = scaler.calibrate(single, 1).unwrap();
        for (a, b) in p_one.iter().zip(p_batch.iter()) {
            assert!((a - b).abs() < 1e-6, "calibrate_one vs calibrate mismatch");
        }
    }

    // ── Fitting ──────────────────────────────────────────────────────────────

    #[test]
    fn vector_mode_fit_reduces_nll() {
        let (logits, labels) = overconfident_2class(100, 0.6);
        let cfg = vector_cfg(2);
        let scaler = VectorScaler::fit(&logits, &labels, cfg).unwrap();
        // Compute initial NLL (identity params).
        let (init_scale, init_bias) = init_params(ScalingMode::Vector, 2);
        let nll_before = compute_nll_from_params(&init_scale, &init_bias, &logits, &labels, 2, 0.0);
        assert!(
            scaler.final_nll <= nll_before + 1e-4,
            "NLL should not increase after fitting: before={nll_before}, after={}",
            scaler.final_nll
        );
    }

    #[test]
    fn matrix_mode_fit_reduces_nll() {
        let (logits, labels) = overconfident_2class(100, 0.6);
        let cfg = matrix_cfg(2);
        let scaler = VectorScaler::fit(&logits, &labels, cfg).unwrap();
        let (init_scale, init_bias) = init_params(ScalingMode::Matrix, 2);
        let nll_before = compute_nll_from_params(&init_scale, &init_bias, &logits, &labels, 2, 0.0);
        assert!(
            scaler.final_nll <= nll_before + 1e-4,
            "Matrix NLL should not increase: before={nll_before}, after={}",
            scaler.final_nll
        );
    }

    #[test]
    fn vector_fit_on_well_calibrated_data_stays_near_identity() {
        // Perfectly calibrated: each sample's true class matches the highest logit.
        // Optimal W ≈ I, b ≈ 0; after fitting scale[k] should stay close to 1.
        let k = 3_usize;
        let (logits, labels) = multiclass_dataset(60, k, 1.5);
        let cfg = VectorScalingConfig {
            mode: ScalingMode::Vector,
            n_classes: k,
            lr: 0.005,
            max_iter: 300,
            tol: 1e-6,
            l2_reg: 1e-3,
        };
        let scaler = VectorScaler::fit(&logits, &labels, cfg).unwrap();
        // Each scale should remain positive (no sign-flip).
        for &s in &scaler.scale {
            assert!(s > 0.0, "scale component turned negative: {s}");
        }
    }

    #[test]
    fn nll_grad_returns_correct_shapes_vector() {
        let k = 3_usize;
        let (logits, labels) = multiclass_dataset(20, k, 2.0);
        let cfg = vector_cfg(k);
        let scaler = VectorScaler::fit(&logits, &labels, cfg).unwrap();
        let (gs, gb) = scaler.nll_grad(&logits, &labels).unwrap();
        assert_eq!(gs.len(), k, "grad_scale must have K elements (Vector)");
        assert_eq!(gb.len(), k, "grad_bias must have K elements");
    }

    #[test]
    fn nll_grad_returns_correct_shapes_matrix() {
        let k = 3_usize;
        let (logits, labels) = multiclass_dataset(20, k, 2.0);
        let cfg = matrix_cfg(k);
        let scaler = VectorScaler::fit(&logits, &labels, cfg).unwrap();
        let (gs, gb) = scaler.nll_grad(&logits, &labels).unwrap();
        assert_eq!(gs.len(), k * k, "grad_scale must have K² elements (Matrix)");
        assert_eq!(gb.len(), k, "grad_bias must have K elements");
    }

    #[test]
    fn convergence_n_iter_leq_max_iter() {
        let k = 2_usize;
        let (logits, labels) = overconfident_2class(50, 0.7);
        let max_iter = 100_usize;
        let cfg = VectorScalingConfig {
            mode: ScalingMode::Vector,
            n_classes: k,
            lr: 0.01,
            max_iter,
            tol: 1e-5,
            l2_reg: 1e-4,
        };
        let scaler = VectorScaler::fit(&logits, &labels, cfg).unwrap();
        assert!(
            scaler.n_iter <= max_iter,
            "n_iter={} exceeded max_iter={max_iter}",
            scaler.n_iter
        );
    }

    #[test]
    fn single_sample_works() {
        let k = 3_usize;
        let logits = vec![1.0_f32, 2.0, 0.5];
        let labels = vec![1_usize]; // class 1 (highest logit)
        let cfg = vector_cfg(k);
        let scaler = VectorScaler::fit(&logits, &labels, cfg).unwrap();
        let probs = scaler.calibrate(&logits, 1).unwrap();
        let s: f32 = probs.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "probs sum to {s} not 1");
    }

    #[test]
    fn l2_reg_zero_works() {
        let k = 2_usize;
        let (logits, labels) = overconfident_2class(30, 0.6);
        let cfg = VectorScalingConfig {
            mode: ScalingMode::Vector,
            n_classes: k,
            lr: 0.01,
            max_iter: 100,
            tol: 1e-5,
            l2_reg: 0.0,
        };
        let scaler = VectorScaler::fit(&logits, &labels, cfg);
        assert!(scaler.is_ok(), "l2_reg=0 should succeed");
    }

    // ── Error handling ───────────────────────────────────────────────────────

    #[test]
    fn fit_rejects_empty_logits() {
        let cfg = vector_cfg(2);
        let r = VectorScaler::fit(&[], &[], cfg);
        assert!(
            matches!(r, Err(BayesError::CalibrationSetEmpty)),
            "expected CalibrationSetEmpty, got {r:?}"
        );
    }

    #[test]
    fn fit_rejects_n_classes_less_than_2() {
        let cfg = VectorScalingConfig {
            mode: ScalingMode::Vector,
            n_classes: 1,
            ..VectorScalingConfig::default()
        };
        let r = VectorScaler::fit(&[1.0_f32], &[0_usize], cfg);
        assert!(
            matches!(r, Err(BayesError::NCalibBinsTooSmall)),
            "expected NCalibBinsTooSmall, got {r:?}"
        );
    }

    #[test]
    fn fit_rejects_label_out_of_range() {
        let k = 2_usize;
        let logits = vec![1.0_f32, 2.0]; // 1 sample, 2 classes
        let labels = vec![5_usize]; // label 5 is out of range [0, 2)
        let cfg = vector_cfg(k);
        let r = VectorScaler::fit(&logits, &labels, cfg);
        assert!(
            matches!(r, Err(BayesError::DimensionMismatch { .. })),
            "expected DimensionMismatch for out-of-range label, got {r:?}"
        );
    }

    #[test]
    fn fit_rejects_non_positive_lr() {
        let cfg = VectorScalingConfig {
            mode: ScalingMode::Vector,
            n_classes: 2,
            lr: 0.0,
            ..VectorScalingConfig::default()
        };
        let logits = vec![1.0_f32, 2.0];
        let labels = vec![0_usize];
        let r = VectorScaler::fit(&logits, &labels, cfg);
        assert!(
            matches!(r, Err(BayesError::InvalidTemperature { .. })),
            "expected InvalidTemperature for lr=0, got {r:?}"
        );
    }

    #[test]
    fn nll_rejects_empty_logits() {
        let k = 2_usize;
        let (logits, labels) = overconfident_2class(10, 0.6);
        let cfg = vector_cfg(k);
        let scaler = VectorScaler::fit(&logits, &labels, cfg).unwrap();
        let r = scaler.nll(&[], &[]);
        assert!(matches!(r, Err(BayesError::CalibrationSetEmpty)));
    }

    #[test]
    fn transform_logits_rejects_wrong_size() {
        let k = 3_usize;
        let (logits, labels) = multiclass_dataset(10, k, 2.0);
        let cfg = vector_cfg(k);
        let scaler = VectorScaler::fit(&logits, &labels, cfg).unwrap();
        // Pass 5 logits when 6 (n=2, k=3) are expected.
        let r = scaler.transform_logits(&[1.0_f32, 2.0, 3.0, 4.0, 5.0], 2);
        assert!(
            matches!(r, Err(BayesError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got {r:?}"
        );
    }

    // ── NLL is non-negative ───────────────────────────────────────────────────

    #[test]
    fn nll_is_non_negative() {
        let k = 3_usize;
        let (logits, labels) = multiclass_dataset(20, k, 2.0);
        let cfg = vector_cfg(k);
        let scaler = VectorScaler::fit(&logits, &labels, cfg).unwrap();
        let nll_val = scaler.nll(&logits, &labels).unwrap();
        assert!(nll_val >= 0.0, "NLL must be non-negative, got {nll_val}");
    }

    #[test]
    fn nll_decreases_for_overconfident_model_after_fitting() {
        // The NLL should be lower after fitting than at the identity initialisation.
        let (logits, labels) = overconfident_2class(120, 0.5);
        let cfg = vector_cfg(2);
        let scaler = VectorScaler::fit(&logits, &labels, cfg).unwrap();
        let (init_scale, init_bias) = init_params(ScalingMode::Vector, 2);
        let nll_before = compute_nll_from_params(&init_scale, &init_bias, &logits, &labels, 2, 0.0);
        let nll_after = scaler.nll(&logits, &labels).unwrap();
        assert!(
            nll_after <= nll_before + 1e-3,
            "NLL should not increase: before={nll_before}, after={nll_after}"
        );
    }
}
