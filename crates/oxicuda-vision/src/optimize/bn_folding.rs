//! Batch normalization folding into preceding linear / conv layers for inference.
//!
//! After training, BN parameters can be folded into the weights and biases of
//! the preceding layer, eliminating the BN layer from the inference graph and
//! reducing memory bandwidth and computation.
//!
//! ## Formulas
//!
//! Given BatchNorm parameters `γ` (gamma), `β` (beta), `μ` (running mean),
//! `σ²` (running variance), and `ε` (epsilon):
//!
//! ```text
//! σ = sqrt(σ² + ε)
//!
//! W_new[i, j] = W[i, j] × γ[i] / σ[i]
//! b_new[i]    = (b[i] - μ[i]) × γ[i] / σ[i]  +  β[i]
//! ```

use crate::error::{VisionError, VisionResult};

// ─── BN parameter struct ──────────────────────────────────────────────────────

/// Running-statistics and learned parameters of a BatchNorm layer.
#[derive(Debug, Clone)]
pub struct BnParams {
    /// Scale parameter γ (per channel), length `c`.
    pub gamma: Vec<f32>,
    /// Shift parameter β (per channel), length `c`.
    pub beta: Vec<f32>,
    /// Running mean μ (per channel), length `c`.
    pub mean: Vec<f32>,
    /// Running variance σ² (per channel), length `c`.
    pub var: Vec<f32>,
    /// Small constant ε for numerical stability.
    pub eps: f32,
}

// ─── fold_bn_into_linear ──────────────────────────────────────────────────────

/// Fold a BatchNorm layer into the preceding fully-connected (linear) layer.
///
/// # Arguments
///
/// - `weight`       : `[out_features × in_features]` row-major weight matrix.
/// - `bias`         : `[out_features]` bias vector.
/// - `bn`           : BatchNorm parameters (length `out_features` each).
/// - `out_features` : number of output units.
/// - `in_features`  : number of input units.
///
/// # Returns
///
/// `(W_new, b_new)` where both have the same shape as the inputs.
///
/// # Errors
///
/// - [`VisionError::DimensionMismatch`] if sizes are inconsistent.
/// - [`VisionError::NonFinite`] if any variance ≤ −eps (would produce NaN).
pub fn fold_bn_into_linear(
    weight: &[f32],
    bias: &[f32],
    bn: &BnParams,
    out_features: usize,
    in_features: usize,
) -> VisionResult<(Vec<f32>, Vec<f32>)> {
    validate_bn(bn, out_features)?;
    if weight.len() != out_features * in_features {
        return Err(VisionError::DimensionMismatch {
            expected: out_features * in_features,
            got: weight.len(),
        });
    }
    if bias.len() != out_features {
        return Err(VisionError::DimensionMismatch {
            expected: out_features,
            got: bias.len(),
        });
    }

    let mut w_new = vec![0.0_f32; out_features * in_features];
    let mut b_new = vec![0.0_f32; out_features];

    for i in 0..out_features {
        let sigma = (bn.var[i] + bn.eps).sqrt();
        if !sigma.is_finite() || sigma == 0.0 {
            return Err(VisionError::NonFinite("BN sigma is zero or non-finite"));
        }
        let scale = bn.gamma[i] / sigma;
        for j in 0..in_features {
            w_new[i * in_features + j] = weight[i * in_features + j] * scale;
        }
        b_new[i] = (bias[i] - bn.mean[i]) * scale + bn.beta[i];
    }

    Ok((w_new, b_new))
}

// ─── fold_bn_into_conv ────────────────────────────────────────────────────────

/// Fold a BatchNorm layer into the preceding convolution layer.
///
/// # Arguments
///
/// - `weight` : `[out_ch × in_ch × kH × kW]` row-major weight tensor.
/// - `bias`   : `[out_ch]` bias vector.
/// - `bn`     : BatchNorm parameters, all vectors of length `out_ch`.
/// - `out_ch` : number of output channels.
///
/// The per-output-channel folding is identical to [`fold_bn_into_linear`]:
/// each "row" of the filter corresponds to one output channel.
///
/// # Errors
///
/// Same as [`fold_bn_into_linear`].  In addition:
///
/// - [`VisionError::DimensionMismatch`] if `weight.len()` is not a multiple
///   of `out_ch`.
pub fn fold_bn_into_conv(
    weight: &[f32],
    bias: &[f32],
    bn: &BnParams,
    out_ch: usize,
) -> VisionResult<(Vec<f32>, Vec<f32>)> {
    validate_bn(bn, out_ch)?;
    if bias.len() != out_ch {
        return Err(VisionError::DimensionMismatch {
            expected: out_ch,
            got: bias.len(),
        });
    }
    if weight.len() % out_ch != 0 {
        return Err(VisionError::DimensionMismatch {
            expected: 0, // multiple of out_ch
            got: weight.len() % out_ch,
        });
    }
    let slice_len = weight.len() / out_ch; // in_ch × kH × kW per output channel

    let mut w_new = vec![0.0_f32; weight.len()];
    let mut b_new = vec![0.0_f32; out_ch];

    for i in 0..out_ch {
        let sigma = (bn.var[i] + bn.eps).sqrt();
        if !sigma.is_finite() || sigma == 0.0 {
            return Err(VisionError::NonFinite("BN sigma is zero or non-finite"));
        }
        let scale = bn.gamma[i] / sigma;
        for j in 0..slice_len {
            w_new[i * slice_len + j] = weight[i * slice_len + j] * scale;
        }
        b_new[i] = (bias[i] - bn.mean[i]) * scale + bn.beta[i];
    }

    Ok((w_new, b_new))
}

// ─── verify_bn_fold ───────────────────────────────────────────────────────────

/// Verify that BN folding preserves the layer output within numerical tolerance.
///
/// Computes `max |folded(x) − original(x)|` over `n_samples` input rows.
///
/// The "original" computation is:
/// ```text
/// y = x @ W^T + b
/// bn_out = (y - mean) * gamma / sigma + beta
/// ```
///
/// The "folded" computation is:
/// ```text
/// y_folded = x @ W_new^T + b_new
/// ```
///
/// # Arguments
///
/// - `x`           : `[n_samples × in_features]` input data.
/// - `weight`      : `[out_features × in_features]`.
/// - `bias`        : `[out_features]`.
/// - `bn`          : BN parameters.
/// - `out_features`
/// - `in_features`
/// - `n_samples`
///
/// # Returns
///
/// Maximum absolute error across all outputs.
///
/// # Errors
///
/// Same as [`fold_bn_into_linear`] plus:
///
/// - [`VisionError::DimensionMismatch`] if `x.len() != n_samples * in_features`.
pub fn verify_bn_fold(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    bn: &BnParams,
    out_features: usize,
    in_features: usize,
    n_samples: usize,
) -> VisionResult<f32> {
    if x.len() != n_samples * in_features {
        return Err(VisionError::DimensionMismatch {
            expected: n_samples * in_features,
            got: x.len(),
        });
    }

    let (w_new, b_new) = fold_bn_into_linear(weight, bias, bn, out_features, in_features)?;

    let mut max_err = 0.0_f32;

    for s in 0..n_samples {
        let x_row = &x[s * in_features..(s + 1) * in_features];

        for i in 0..out_features {
            // Original path: linear → BN
            let y_orig_i: f32 = bias[i]
                + (0..in_features)
                    .map(|j| weight[i * in_features + j] * x_row[j])
                    .sum::<f32>();
            let sigma = (bn.var[i] + bn.eps).sqrt();
            let bn_out_i = (y_orig_i - bn.mean[i]) * bn.gamma[i] / sigma + bn.beta[i];

            // Folded path
            let y_fold_i: f32 = b_new[i]
                + (0..in_features)
                    .map(|j| w_new[i * in_features + j] * x_row[j])
                    .sum::<f32>();

            let err = (bn_out_i - y_fold_i).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }

    Ok(max_err)
}

// ─── Internal helper ─────────────────────────────────────────────────────────

fn validate_bn(bn: &BnParams, n_channels: usize) -> VisionResult<()> {
    if bn.var.iter().any(|&v| v < -bn.eps) {
        return Err(VisionError::NonFinite("BN variance is negative"));
    }
    for vec in [&bn.gamma, &bn.beta, &bn.mean, &bn.var] {
        if vec.len() != n_channels {
            return Err(VisionError::DimensionMismatch {
                expected: n_channels,
                got: vec.len(),
            });
        }
    }
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn default_bn(c: usize) -> BnParams {
        BnParams {
            gamma: vec![1.0_f32; c],
            beta: vec![0.0_f32; c],
            mean: vec![0.0_f32; c],
            var: vec![1.0_f32; c],
            eps: 1e-5,
        }
    }

    fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut r = LcgRng::new(seed);
        (0..n).map(|_| r.next_f32() * 2.0 - 1.0).collect()
    }

    // 1 ─ fold_bn_into_linear output shape
    #[test]
    fn fold_output_shape() {
        let out_f = 4;
        let in_f = 8;
        let w = vec![1.0_f32; out_f * in_f];
        let b = vec![0.0_f32; out_f];
        let bn = default_bn(out_f);
        let (w_new, b_new) = fold_bn_into_linear(&w, &b, &bn, out_f, in_f)
            .expect("fold_bn_into_linear should succeed");
        assert_eq!(w_new.len(), out_f * in_f);
        assert_eq!(b_new.len(), out_f);
    }

    // 2 ─ fold_bn_into_linear preserves computation
    #[test]
    fn fold_preserves_computation() {
        let out_f = 4;
        let in_f = 3;
        let n = 5;
        let w = rand_vec(out_f * in_f, 1);
        let b = rand_vec(out_f, 2);
        let bn = BnParams {
            gamma: rand_vec(out_f, 3),
            beta: rand_vec(out_f, 4),
            mean: rand_vec(out_f, 5),
            var: vec![0.5_f32, 1.0, 2.0, 0.8],
            eps: 1e-5,
        };
        let x = rand_vec(n * in_f, 6);
        let err =
            verify_bn_fold(&x, &w, &b, &bn, out_f, in_f, n).expect("verify_bn_fold should succeed");
        assert!(err < 1e-5, "max fold error = {err}");
    }

    // 3 ─ zero mean fold
    #[test]
    fn zero_mean_fold() {
        let out_f = 3;
        let in_f = 2;
        let w = vec![1.0_f32; out_f * in_f];
        let b = vec![0.5_f32; out_f];
        let bn = BnParams {
            gamma: vec![2.0_f32; out_f],
            beta: vec![0.0_f32; out_f],
            mean: vec![0.0_f32; out_f],
            var: vec![1.0_f32; out_f],
            eps: 1e-5,
        };
        let (w_new, b_new) = fold_bn_into_linear(&w, &b, &bn, out_f, in_f)
            .expect("fold_bn_into_linear should succeed");
        // With mean=0, var=1, gamma=2, beta=0: w_new = 2*w/1, b_new = 2*b/1
        let sigma = (1.0_f32 + 1e-5).sqrt(); // sqrt(var + eps)
        let expected_w = 2.0 / sigma;
        let expected_b = (0.5 - 0.0) * 2.0 / sigma; // (b - mean) * gamma / sigma
        for &v in &w_new {
            assert!(
                (v - expected_w).abs() < 1e-4,
                "w_new={v}, expected={expected_w}"
            );
        }
        for &v in &b_new {
            assert!(
                (v - expected_b).abs() < 1e-4,
                "b_new={v}, expected={expected_b}"
            );
        }
    }

    // 4 ─ unit var fold
    #[test]
    fn unit_var_fold() {
        let out_f = 2;
        let in_f = 2;
        let w = vec![1.0_f32, 0.0, 0.0, 1.0]; // identity
        let b = vec![0.0_f32; out_f];
        let bn = default_bn(out_f); // gamma=1, beta=0, mean=0, var=1
        let (w_new, b_new) = fold_bn_into_linear(&w, &b, &bn, out_f, in_f)
            .expect("fold_bn_into_linear should succeed");
        // With gamma=1, beta=0, mean=0, var=1: fold is identity
        for (orig, folded) in w.iter().zip(&w_new) {
            assert!((orig - folded).abs() < 1e-5);
        }
        for &v in &b_new {
            assert!(v.abs() < 1e-5);
        }
    }

    // 5 ─ gamma=1, beta=0 simplifies to centered-normalized
    #[test]
    fn gamma_1_beta_0_simplifies() {
        let out_f = 2;
        let in_f = 2;
        let w = rand_vec(out_f * in_f, 10);
        let b = rand_vec(out_f, 11);
        let bn = BnParams {
            gamma: vec![1.0_f32; out_f],
            beta: vec![0.0_f32; out_f],
            mean: vec![0.5_f32; out_f],
            var: vec![4.0_f32; out_f],
            eps: 1e-5,
        };
        let (w_new, b_new) = fold_bn_into_linear(&w, &b, &bn, out_f, in_f)
            .expect("fold_bn_into_linear should succeed");
        let sigma = (4.0_f32 + 1e-5).sqrt();
        // w_new[i,j] = w[i,j] / sigma
        for (orig, folded) in w.iter().zip(&w_new) {
            let expect = orig / sigma;
            assert!((folded - expect).abs() < 1e-5, "{folded} vs {expect}");
        }
        // b_new[i] = (b[i] - 0.5) / sigma
        for (i, &bv) in b_new.iter().enumerate() {
            let expect = (b[i] - 0.5) / sigma;
            assert!((bv - expect).abs() < 1e-5, "b_new[{i}]={bv} vs {expect}");
        }
    }

    // 6 ─ negative variance returns error
    #[test]
    fn var_negative_error() {
        let out_f = 2;
        let in_f = 2;
        let w = vec![1.0_f32; out_f * in_f];
        let b = vec![0.0_f32; out_f];
        let bn = BnParams {
            gamma: vec![1.0_f32; out_f],
            beta: vec![0.0_f32; out_f],
            mean: vec![0.0_f32; out_f],
            var: vec![-1.0_f32, 1.0], // negative variance!
            eps: 1e-5,
        };
        let result = fold_bn_into_linear(&w, &b, &bn, out_f, in_f);
        assert!(result.is_err());
    }

    // 7 ─ fold_bn_into_conv shape
    #[test]
    fn bn_fold_conv_shape() {
        let out_ch = 4;
        let in_ch = 3;
        let kh = 3;
        let kw = 3;
        let w = rand_vec(out_ch * in_ch * kh * kw, 20);
        let b = vec![0.0_f32; out_ch];
        let bn = default_bn(out_ch);
        let (w_new, b_new) =
            fold_bn_into_conv(&w, &b, &bn, out_ch).expect("fold_bn_into_conv should succeed");
        assert_eq!(w_new.len(), out_ch * in_ch * kh * kw);
        assert_eq!(b_new.len(), out_ch);
    }

    // 8 ─ verify_bn_fold error is small
    #[test]
    fn verify_error_small() {
        let out_f = 6;
        let in_f = 4;
        let n = 8;
        let w = rand_vec(out_f * in_f, 30);
        let b = rand_vec(out_f, 31);
        let bn = BnParams {
            gamma: rand_vec(out_f, 32),
            beta: rand_vec(out_f, 33),
            mean: rand_vec(out_f, 34),
            var: vec![0.1_f32, 0.5, 1.0, 2.0, 0.3, 1.5],
            eps: 1e-5,
        };
        let x = rand_vec(n * in_f, 35);
        let err =
            verify_bn_fold(&x, &w, &b, &bn, out_f, in_f, n).expect("verify_bn_fold should succeed");
        assert!(err < 1e-4, "max error = {err}");
    }

    // 9 ─ batch size varies (verify_bn_fold with different n_samples)
    #[test]
    fn batch_size_varies() {
        let out_f = 3;
        let in_f = 2;
        let w = rand_vec(out_f * in_f, 40);
        let b = rand_vec(out_f, 41);
        let bn = default_bn(out_f);
        for &n in &[1_usize, 4, 16] {
            let x = rand_vec(n * in_f, n as u64);
            let err = verify_bn_fold(&x, &w, &b, &bn, out_f, in_f, n)
                .expect("verify_bn_fold should succeed");
            assert!(err < 1e-4, "n={n} error={err}");
        }
    }

    // 10 ─ fold_bn_into_conv with gamma scaling
    #[test]
    fn bn_fold_conv_gamma_scales_weights() {
        let out_ch = 2;
        let w = vec![1.0_f32, 1.0]; // out_ch × in_ch × k × k (in_ch = k = 1)
        let b = vec![0.0_f32; out_ch];
        let bn = BnParams {
            gamma: vec![2.0_f32, 3.0],
            beta: vec![0.0_f32; out_ch],
            mean: vec![0.0_f32; out_ch],
            var: vec![1.0_f32; out_ch],
            eps: 1e-5,
        };
        let (w_new, _) =
            fold_bn_into_conv(&w, &b, &bn, out_ch).expect("fold_bn_into_conv should succeed");
        let sigma = (1.0_f32 + 1e-5).sqrt();
        assert!((w_new[0] - 2.0 / sigma).abs() < 1e-4);
        assert!((w_new[1] - 3.0 / sigma).abs() < 1e-4);
    }
}
