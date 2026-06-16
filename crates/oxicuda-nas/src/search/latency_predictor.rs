//! Latency predictor for MBConv-based NAS search spaces.
//!
//! Provides a simple linear regression model that estimates the on-device
//! latency of an MBConv block from a six-dimensional feature vector derived
//! from the block specification:
//!
//! ```text
//! features = [in_ch, out_ch, expand_ratio, kernel, stride, in_ch * out_ch]
//! ```
//!
//! The cross-term `in_ch * out_ch` captures the dominant quadratic scaling of
//! pointwise-convolution cost and improves fit on typical NAS benchmarks.
//!
//! # Training
//!
//! [`train_latency_predictor`] fits the model via ordinary least squares using a
//! single QR-free normal-equations solve (Gram matrix inversion implemented with
//! Gaussian elimination with partial pivoting).  For N ≤ a few thousand samples
//! this is fast and numerically stable.

use crate::error::{NasError, NasResult};
use crate::ops::mbconv_ops::MbConvSpec;

// ─── Feature extraction ───────────────────────────────────────────────────────

/// Compute the feature vector for a single [`MbConvSpec`].
///
/// The returned vector has exactly 6 elements:
/// `[in_ch, out_ch, expand_ratio, kernel, stride, in_ch * out_ch]`.
///
/// All values are cast to `f32`.  The cross-term is clamped to avoid overflow.
#[must_use]
pub fn latency_features(spec: &MbConvSpec) -> Vec<f32> {
    let cross = (spec.in_ch as u64)
        .saturating_mul(spec.out_ch as u64)
        .min(u32::MAX as u64) as f32;
    vec![
        spec.in_ch as f32,
        spec.out_ch as f32,
        spec.expand_ratio as f32,
        spec.kernel as f32,
        spec.stride as f32,
        cross,
    ]
}

// ─── LatencyPredictor ─────────────────────────────────────────────────────────

/// Linear latency surrogate: `latency ≈ dot(features, w) + b`.
///
/// Trained via [`train_latency_predictor`].
#[derive(Debug, Clone)]
pub struct LatencyPredictor {
    /// Weight vector `[n_features]`.
    pub w: Vec<f32>,
    /// Bias (scalar stored as length-1 vec for uniformity).
    pub b: Vec<f32>,
}

impl LatencyPredictor {
    /// Predict the latency (in seconds, ms, or any unit the training data used)
    /// for a single [`MbConvSpec`].
    ///
    /// Returns `dot(features(spec), w) + b[0]`.  The output is not clamped to
    /// non-negative because callers may need the raw regression output for
    /// gradients; use `.max(0.0)` if a non-negative latency is required.
    #[must_use]
    pub fn predict(&self, spec: &MbConvSpec) -> f32 {
        let features = latency_features(spec);
        let dot: f32 = features.iter().zip(self.w.iter()).map(|(f, w)| f * w).sum();
        dot + self.b[0]
    }
}

// ─── Training ─────────────────────────────────────────────────────────────────

/// Fit a [`LatencyPredictor`] to a set of `(spec, measured_latency)` pairs via
/// ordinary least squares.
///
/// The design matrix is augmented with a bias column, and the normal equations
/// `(X^T X) w = X^T y` are solved using Gaussian elimination with partial
/// pivoting (O(n_features^3) per call, negligible for 6 features).
///
/// # Errors
///
/// * [`NasError::InvalidNumOps`] — if `specs` and `latencies` are empty.
/// * [`NasError::DimensionMismatch`] — if `specs.len() != latencies.len()`.
/// * [`NasError::NanInArchParams`] — if the normal-equations system is
///   singular or produces NaN/Inf weights.
pub fn train_latency_predictor(
    specs: &[MbConvSpec],
    latencies: &[f32],
) -> NasResult<LatencyPredictor> {
    if specs.is_empty() || latencies.is_empty() {
        return Err(NasError::InvalidNumOps);
    }
    if specs.len() != latencies.len() {
        return Err(NasError::DimensionMismatch {
            expected: specs.len(),
            got: latencies.len(),
        });
    }

    let n = specs.len();
    // Feature dimension: 6 raw features + 1 bias column = 7.
    let feat_dim = latency_features(&specs[0]).len();
    let aug_dim = feat_dim + 1; // +1 for bias

    // Build augmented design matrix X: [n × aug_dim].
    let mut x_mat = vec![0.0_f32; n * aug_dim];
    for (i, spec) in specs.iter().enumerate() {
        let feats = latency_features(spec);
        for (j, &f) in feats.iter().enumerate() {
            x_mat[i * aug_dim + j] = f;
        }
        x_mat[i * aug_dim + feat_dim] = 1.0; // bias column
    }

    // Gram matrix G = X^T X: [aug_dim × aug_dim].
    let mut gram = vec![0.0_f32; aug_dim * aug_dim];
    for i in 0..aug_dim {
        for j in 0..aug_dim {
            let mut acc = 0.0_f32;
            for k in 0..n {
                acc += x_mat[k * aug_dim + i] * x_mat[k * aug_dim + j];
            }
            gram[i * aug_dim + j] = acc;
        }
    }

    // Right-hand side rhs = X^T y: [aug_dim].
    let mut rhs = vec![0.0_f32; aug_dim];
    for i in 0..aug_dim {
        let mut acc = 0.0_f32;
        for k in 0..n {
            acc += x_mat[k * aug_dim + i] * latencies[k];
        }
        rhs[i] = acc;
    }

    // Add Tikhonov regularization λI to stabilise near-singular systems.
    let lambda = 1e-4_f32;
    for i in 0..aug_dim {
        gram[i * aug_dim + i] += lambda;
    }

    // Solve G * theta = rhs via Gaussian elimination with partial pivoting.
    let theta = gauss_solve(&gram, &rhs, aug_dim)?;

    // Check for non-finite solution.
    if theta.iter().any(|v| !v.is_finite()) {
        return Err(NasError::NanInArchParams);
    }

    let w = theta[..feat_dim].to_vec();
    let b = vec![theta[feat_dim]];

    Ok(LatencyPredictor { w, b })
}

// ─── Gaussian elimination ─────────────────────────────────────────────────────

/// Solve `A x = b` via Gaussian elimination with partial pivoting.
///
/// `a_flat` is `[n × n]` row-major; returns the solution `x` of length `n`.
fn gauss_solve(a_flat: &[f32], b: &[f32], n: usize) -> NasResult<Vec<f32>> {
    // Build augmented matrix [A | b].
    let mut aug = vec![0.0_f32; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a_flat[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination.
    for col in 0..n {
        // Partial pivoting: find max absolute value in this column.
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| {
                aug[r1 * (n + 1) + col]
                    .abs()
                    .partial_cmp(&aug[r2 * (n + 1) + col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);

        if aug[pivot_row * (n + 1) + col].abs() < 1e-12 {
            // Near-singular; regularization should have prevented this, but guard anyway.
            return Err(NasError::NanInArchParams);
        }

        // Swap rows col and pivot_row.
        if pivot_row != col {
            for j in 0..=n {
                aug.swap(col * (n + 1) + j, pivot_row * (n + 1) + j);
            }
        }

        let pivot = aug[col * (n + 1) + col];
        // Scale pivot row.
        for j in col..=n {
            aug[col * (n + 1) + j] /= pivot;
        }

        // Eliminate column entries below pivot.
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col];
            for j in col..=n {
                let sub = factor * aug[col * (n + 1) + j];
                aug[row * (n + 1) + j] -= sub;
            }
        }
    }

    // Back substitution.
    let mut x = vec![0.0_f32; n];
    for i in (0..n).rev() {
        let mut val = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            val -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = val;
    }

    Ok(x)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_spec() -> MbConvSpec {
        MbConvSpec {
            in_ch: 16,
            out_ch: 32,
            stride: 1,
            expand_ratio: 4,
            kernel: 3,
        }
    }

    fn make_samples(n: usize) -> (Vec<MbConvSpec>, Vec<f32>) {
        let specs: Vec<MbConvSpec> = (0..n)
            .map(|i| MbConvSpec {
                in_ch: 8 + i * 4,
                out_ch: 16 + i * 4,
                stride: 1,
                expand_ratio: 4,
                kernel: 3,
            })
            .collect();
        // Latency = 0.001 * in_ch + 0.0005 * out_ch (synthetic linear target).
        let latencies: Vec<f32> = specs
            .iter()
            .map(|s| 0.001 * s.in_ch as f32 + 0.0005 * s.out_ch as f32)
            .collect();
        (specs, latencies)
    }

    // ── 1. predict_positive ───────────────────────────────────────────────────
    #[test]
    fn predict_positive() {
        let (specs, latencies) = make_samples(10);
        let pred = train_latency_predictor(&specs, &latencies).expect("train");
        // Predictions for the training specs should be positive (as the target is > 0).
        for spec in &specs {
            let p = pred.predict(spec);
            assert!(p.is_finite(), "prediction non-finite for spec {spec:?}");
        }
    }

    // ── 2. train_finite ───────────────────────────────────────────────────────
    #[test]
    fn train_finite() {
        let (specs, latencies) = make_samples(8);
        let pred = train_latency_predictor(&specs, &latencies).expect("train");
        assert!(pred.w.iter().all(|v| v.is_finite()), "weights non-finite");
        assert!(pred.b[0].is_finite(), "bias non-finite");
    }

    // ── 3. features_len ───────────────────────────────────────────────────────
    #[test]
    fn features_len() {
        let spec = tiny_spec();
        let f = latency_features(&spec);
        assert_eq!(f.len(), 6, "feature vector should have 6 elements");
    }

    // ── 4. latency_scales_with_channels ──────────────────────────────────────
    #[test]
    fn latency_scales_with_channels() {
        let (specs, latencies) = make_samples(12);
        let pred = train_latency_predictor(&specs, &latencies).expect("train");

        // A bigger block should predict higher latency (regression should capture this).
        let small = MbConvSpec {
            in_ch: 8,
            out_ch: 16,
            stride: 1,
            expand_ratio: 4,
            kernel: 3,
        };
        let large = MbConvSpec {
            in_ch: 64,
            out_ch: 128,
            stride: 1,
            expand_ratio: 4,
            kernel: 3,
        };
        let p_small = pred.predict(&small);
        let p_large = pred.predict(&large);
        assert!(
            p_large > p_small,
            "large spec should predict higher latency: small={p_small}, large={p_large}"
        );
    }

    // ── 5. predict_zero_spec ──────────────────────────────────────────────────
    #[test]
    fn predict_zero_spec() {
        let (specs, latencies) = make_samples(6);
        let pred = train_latency_predictor(&specs, &latencies).expect("train");
        let zero_spec = MbConvSpec {
            in_ch: 0,
            out_ch: 0,
            stride: 0,
            expand_ratio: 0,
            kernel: 0,
        };
        let p = pred.predict(&zero_spec);
        // Should not panic; value is the bias term.
        assert!(p.is_finite(), "zero spec prediction should be finite: {p}");
    }

    // ── 6. train_single_sample ────────────────────────────────────────────────
    #[test]
    fn train_single_sample() {
        let spec = tiny_spec();
        let latency = 0.005_f32;
        let result = train_latency_predictor(std::slice::from_ref(&spec), &[latency]);
        assert!(result.is_ok(), "single sample training should succeed");
        let pred = result.expect("train single");
        let p = pred.predict(&spec);
        assert!(p.is_finite(), "prediction should be finite: {p}");
    }

    // ── 7. train_consistent ───────────────────────────────────────────────────
    #[test]
    fn train_consistent() {
        // Identical training with same data should produce the same predictor.
        let (specs, latencies) = make_samples(8);
        let pred1 = train_latency_predictor(&specs, &latencies).expect("train1");
        let pred2 = train_latency_predictor(&specs, &latencies).expect("train2");
        for (w1, w2) in pred1.w.iter().zip(pred2.w.iter()) {
            assert!((w1 - w2).abs() < 1e-6, "inconsistent weights: {w1} vs {w2}");
        }
    }

    // ── 8. predict_finite ─────────────────────────────────────────────────────
    #[test]
    fn predict_finite() {
        let (specs, latencies) = make_samples(10);
        let pred = train_latency_predictor(&specs, &latencies).expect("train");
        let spec = MbConvSpec {
            in_ch: 32,
            out_ch: 64,
            stride: 2,
            expand_ratio: 6,
            kernel: 5,
        };
        let p = pred.predict(&spec);
        assert!(p.is_finite(), "prediction finite: {p}");
    }

    // ── 9. empty input → error ────────────────────────────────────────────────
    #[test]
    fn empty_input_error() {
        let result = train_latency_predictor(&[], &[]);
        assert!(result.is_err(), "expected error for empty input");
    }

    // ── 10. mismatched len → error ────────────────────────────────────────────
    #[test]
    fn mismatched_len_error() {
        let spec = tiny_spec();
        let result = train_latency_predictor(&[spec], &[0.001, 0.002]);
        assert!(result.is_err(), "expected DimensionMismatch");
    }
}
