#![allow(clippy::needless_range_loop)]
//! ANN→SNN conversion with BatchNorm folding and bias absorption.
//!
//! A trained ANN block of the form `BN(Linear(x))` cannot be run directly on an
//! integrate-and-fire (IF) substrate, because the spiking layer has no native
//! BatchNorm. The standard remedy (Rueckauer et al. 2017, "Conversion of
//! Continuous-Valued Deep Networks to Efficient Event-Driven Networks";
//! Sengupta et al. 2019, "Going Deeper in Spiking Neural Networks") is to
//! *fold* the affine BatchNorm transform into the preceding linear weights and
//! biases, then *absorb* the resulting bias into a constant input current used
//! during threshold balancing, so that the IF neuron reproduces the
//! `ReLU(BN(W·x + b))` response with no separate normalisation step.
//!
//! # (a) BatchNorm folding
//!
//! At inference a BatchNorm layer applies the per-output-channel affine map
//!
//! ```text
//! BN(y_i) = γ_i · (y_i − μ_i) / √(v_i + ε) + β_i,
//! ```
//!
//! with running mean `μ`, running variance `v`, scale `γ`, shift `β` and
//! stabiliser `ε`. Composing with `y = W·x + b` and writing
//! `s_i = γ_i / √(v_i + ε)` gives a plain linear layer:
//!
//! ```text
//! W_fold[i, :] = s_i · W[i, :]                         (row-scaled per output)
//! b_fold[i]    = (b_i − μ_i) · s_i + β_i.
//! ```
//!
//! Running `Linear(W_fold, b_fold)` reproduces `BN(Linear(W, b))` exactly.
//!
//! # (b) Bias absorption
//!
//! After folding, the bias `b_fold` is constant in time. On an IF neuron a
//! constant bias acts as a steady input current injected every timestep, so it
//! can be removed from the weight layer and supplied to threshold balancing as a
//! per-output **bias current** `I_bias = b_fold`. The membrane then integrates
//! `W_fold · x_t + I_bias` each step, matching the ReLU-of-affine response while
//! keeping the synaptic weight layer bias-free.

use super::ann2snn::SnnLayer;
use crate::error::{SnnError, SnnResult};

/// Per-output-channel BatchNorm parameters captured from a trained ANN.
///
/// All four statistics plus the scale/shift are vectors of length `out_dim`
/// (one entry per output channel of the preceding linear layer).
#[derive(Debug, Clone)]
pub struct BatchNormParams {
    /// Affine scale `γ`, length `out_dim`.
    pub gamma: Vec<f32>,
    /// Affine shift `β`, length `out_dim`.
    pub beta: Vec<f32>,
    /// Running mean `μ`, length `out_dim`.
    pub running_mean: Vec<f32>,
    /// Running variance `v` (`≥ 0`), length `out_dim`.
    pub running_var: Vec<f32>,
    /// Numerical stabiliser `ε > 0` added to the variance under the square root.
    pub eps: f32,
}

impl BatchNormParams {
    /// Construct and validate a BatchNorm parameter set for `out_dim` channels.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::BadDim`] when `out_dim == 0`,
    /// [`SnnError::BadShape`] when any vector length differs from `out_dim`,
    /// [`SnnError::OutOfRange`] when `eps ≤ 0`/non-finite or any
    /// `running_var < 0`.
    pub fn new(
        gamma: Vec<f32>,
        beta: Vec<f32>,
        running_mean: Vec<f32>,
        running_var: Vec<f32>,
        eps: f32,
        out_dim: usize,
    ) -> SnnResult<Self> {
        if out_dim == 0 {
            return Err(SnnError::BadDim { got: out_dim });
        }
        for v in [&gamma, &beta, &running_mean, &running_var] {
            if v.len() != out_dim {
                return Err(SnnError::BadShape {
                    expected: out_dim,
                    got: v.len(),
                });
            }
        }
        if !eps.is_finite() || eps <= 0.0 {
            return Err(SnnError::OutOfRange {
                name: "eps".into(),
                val: eps,
            });
        }
        for (i, &var) in running_var.iter().enumerate() {
            if !var.is_finite() || var < 0.0 {
                return Err(SnnError::OutOfRange {
                    name: format!("running_var[{i}]"),
                    val: var,
                });
            }
        }
        Ok(Self {
            gamma,
            beta,
            running_mean,
            running_var,
            eps,
        })
    }

    /// Number of output channels (`out_dim`).
    #[must_use]
    #[inline]
    pub fn out_dim(&self) -> usize {
        self.gamma.len()
    }
}

/// Validate that a [`BatchNormParams`] is consistent with `layer.out_dim` and
/// that its vector lengths match.
fn validate_against_layer(layer: &SnnLayer, bn: &BatchNormParams) -> SnnResult<()> {
    if layer.in_dim == 0 {
        return Err(SnnError::BadDim { got: layer.in_dim });
    }
    if layer.out_dim == 0 {
        return Err(SnnError::BadDim { got: layer.out_dim });
    }
    if layer.w.len() != layer.in_dim * layer.out_dim {
        return Err(SnnError::BadShape {
            expected: layer.in_dim * layer.out_dim,
            got: layer.w.len(),
        });
    }
    if layer.b.len() != layer.out_dim {
        return Err(SnnError::BadShape {
            expected: layer.out_dim,
            got: layer.b.len(),
        });
    }
    for v in [&bn.gamma, &bn.beta, &bn.running_mean, &bn.running_var] {
        if v.len() != layer.out_dim {
            return Err(SnnError::BadShape {
                expected: layer.out_dim,
                got: v.len(),
            });
        }
    }
    if !bn.eps.is_finite() || bn.eps <= 0.0 {
        return Err(SnnError::OutOfRange {
            name: "eps".into(),
            val: bn.eps,
        });
    }
    Ok(())
}

/// Fold a trained BatchNorm layer into the preceding [`SnnLayer`].
///
/// Returns a new `SnnLayer` whose linear forward pass equals
/// `BN(Linear(layer.w, layer.b))`. Per output channel `i`, with
/// `s_i = γ_i / √(v_i + ε)`:
///
/// ```text
/// W_fold[i, :] = s_i · W[i, :]
/// b_fold[i]    = (b_i − μ_i) · s_i + β_i.
/// ```
///
/// The threshold `v_th` is carried through unchanged.
///
/// # Errors
///
/// Returns [`SnnError::BadDim`] for zero dimensions, [`SnnError::BadShape`] when
/// `layer.w`/`layer.b` or any BN vector has the wrong length, and
/// [`SnnError::OutOfRange`] when `eps` is invalid. The denominator
/// `√(v_i + ε)` is always `> 0` because `v_i ≥ 0` and `ε > 0`.
pub fn fold_batchnorm(layer: &SnnLayer, bn: &BatchNormParams) -> SnnResult<SnnLayer> {
    validate_against_layer(layer, bn)?;
    let in_dim = layer.in_dim;
    let out_dim = layer.out_dim;

    let mut w_fold = vec![0.0_f32; in_dim * out_dim];
    let mut b_fold = vec![0.0_f32; out_dim];

    for i in 0..out_dim {
        let denom = (bn.running_var[i] + bn.eps).sqrt();
        // denom > 0 guaranteed (var ≥ 0, eps > 0); guard against fp underflow.
        if !denom.is_finite() || denom <= 0.0 {
            return Err(SnnError::Internal {
                msg: format!("non-positive BN denominator at channel {i}"),
            });
        }
        let s_i = bn.gamma[i] / denom;
        let row_off = i * in_dim;
        for j in 0..in_dim {
            w_fold[row_off + j] = s_i * layer.w[row_off + j];
        }
        b_fold[i] = (layer.b[i] - bn.running_mean[i]) * s_i + bn.beta[i];
    }

    Ok(SnnLayer {
        w: w_fold,
        b: b_fold,
        v_th: layer.v_th,
        in_dim,
        out_dim,
    })
}

/// Absorb a layer's bias into a constant input-current term for threshold
/// balancing.
///
/// After folding, the bias `b` is constant across timesteps. On an IF neuron a
/// constant bias is equivalent to a steady current injected every step, so it is
/// stripped from the synaptic weight layer and returned separately as the
/// per-output **bias current** `I_bias[i] = b[i]`. The IF membrane should then
/// integrate `W · x_t + I_bias` each timestep, reproducing the affine response
/// while the returned [`SnnLayer`] carries an all-zero bias.
///
/// Returns `(bias_current, layer_without_bias)` where `bias_current` has length
/// `out_dim`.
///
/// # Errors
///
/// Returns [`SnnError::BadDim`] for zero dimensions and [`SnnError::BadShape`]
/// when `layer.w`/`layer.b` lengths are inconsistent.
pub fn absorb_bias_into_threshold(layer: &SnnLayer) -> SnnResult<(Vec<f32>, SnnLayer)> {
    if layer.in_dim == 0 {
        return Err(SnnError::BadDim { got: layer.in_dim });
    }
    if layer.out_dim == 0 {
        return Err(SnnError::BadDim { got: layer.out_dim });
    }
    if layer.w.len() != layer.in_dim * layer.out_dim {
        return Err(SnnError::BadShape {
            expected: layer.in_dim * layer.out_dim,
            got: layer.w.len(),
        });
    }
    if layer.b.len() != layer.out_dim {
        return Err(SnnError::BadShape {
            expected: layer.out_dim,
            got: layer.b.len(),
        });
    }
    let bias_current = layer.b.clone();
    let stripped = SnnLayer {
        w: layer.w.clone(),
        b: vec![0.0_f32; layer.out_dim],
        v_th: layer.v_th,
        in_dim: layer.in_dim,
        out_dim: layer.out_dim,
    };
    Ok((bias_current, stripped))
}

/// Convenience: fold BatchNorm and then absorb the folded bias in one call.
///
/// Equivalent to `absorb_bias_into_threshold(&fold_batchnorm(layer, bn)?)`.
/// Returns `(bias_current, bias_free_layer)`.
///
/// # Errors
///
/// Propagates errors from [`fold_batchnorm`] and [`absorb_bias_into_threshold`].
pub fn fold_and_absorb(layer: &SnnLayer, bn: &BatchNormParams) -> SnnResult<(Vec<f32>, SnnLayer)> {
    let folded = fold_batchnorm(layer, bn)?;
    absorb_bias_into_threshold(&folded)
}

/// Reference linear forward pass `y = W · x + b` over an [`SnnLayer`], used by
/// tests and as a small utility. `x` length must equal `in_dim`; the result has
/// length `out_dim`.
///
/// # Errors
///
/// Returns [`SnnError::IncompatibleLength`] when `x.len() != in_dim`.
pub fn linear_forward(layer: &SnnLayer, x: &[f32]) -> SnnResult<Vec<f32>> {
    if x.len() != layer.in_dim {
        return Err(SnnError::IncompatibleLength {
            a: layer.in_dim,
            b: x.len(),
        });
    }
    let mut y = vec![0.0_f32; layer.out_dim];
    for i in 0..layer.out_dim {
        let row_off = i * layer.in_dim;
        let mut acc = layer.b[i];
        for j in 0..layer.in_dim {
            acc += layer.w[row_off + j] * x[j];
        }
        y[i] = acc;
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_layer(in_dim: usize, out_dim: usize, w: Vec<f32>, b: Vec<f32>) -> SnnLayer {
        SnnLayer {
            w,
            b,
            v_th: 1.0,
            in_dim,
            out_dim,
        }
    }

    /// Explicit reference: BN(Linear(x)) computed directly from raw params.
    fn bn_linear_reference(layer: &SnnLayer, bn: &BatchNormParams, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0_f32; layer.out_dim];
        for i in 0..layer.out_dim {
            let row_off = i * layer.in_dim;
            let mut y = layer.b[i];
            for j in 0..layer.in_dim {
                y += layer.w[row_off + j] * x[j];
            }
            let denom = (bn.running_var[i] + bn.eps).sqrt();
            out[i] = bn.gamma[i] * (y - bn.running_mean[i]) / denom + bn.beta[i];
        }
        out
    }

    // 1. BatchNormParams::new validates lengths, eps, variance.
    #[test]
    fn bn_params_validation() {
        // wrong gamma length
        assert!(matches!(
            BatchNormParams::new(
                vec![1.0],
                vec![0.0, 0.0],
                vec![0.0, 0.0],
                vec![1.0, 1.0],
                1e-5,
                2
            ),
            Err(SnnError::BadShape { .. })
        ));
        // zero out_dim
        assert!(matches!(
            BatchNormParams::new(vec![], vec![], vec![], vec![], 1e-5, 0),
            Err(SnnError::BadDim { .. })
        ));
        // bad eps
        assert!(matches!(
            BatchNormParams::new(vec![1.0], vec![0.0], vec![0.0], vec![1.0], 0.0, 1),
            Err(SnnError::OutOfRange { .. })
        ));
        // negative variance
        assert!(matches!(
            BatchNormParams::new(vec![1.0], vec![0.0], vec![0.0], vec![-1.0], 1e-5, 1),
            Err(SnnError::OutOfRange { .. })
        ));
        // valid
        let bn = BatchNormParams::new(
            vec![1.0, 2.0],
            vec![0.0, 0.5],
            vec![0.1, 0.2],
            vec![1.0, 4.0],
            1e-5,
            2,
        )
        .expect("bn");
        assert_eq!(bn.out_dim(), 2);
    }

    // 2. Folding then linear forward equals BN∘Linear forward (random-ish input).
    #[test]
    fn fold_matches_bn_linear() {
        let in_dim = 3_usize;
        let out_dim = 2_usize;
        let layer = make_layer(
            in_dim,
            out_dim,
            vec![0.2, -0.5, 0.7, 0.1, 0.4, -0.3], // 2×3
            vec![0.05, -0.1],
        );
        let bn = BatchNormParams::new(
            vec![1.5, 0.8],
            vec![0.2, -0.4],
            vec![0.3, 0.1],
            vec![2.0, 0.5],
            1e-5,
            out_dim,
        )
        .expect("bn");
        let folded = fold_batchnorm(&layer, &bn).expect("fold");

        let inputs = [
            [0.5_f32, -0.2, 0.9],
            [1.0, 1.0, 1.0],
            [-0.3, 0.6, -0.8],
            [0.0, 0.0, 0.0],
        ];
        for x in &inputs {
            let got = linear_forward(&folded, x).expect("fwd");
            let want = bn_linear_reference(&layer, &bn, x);
            for i in 0..out_dim {
                assert!(
                    (got[i] - want[i]).abs() < 1e-4,
                    "channel {i}: got {}, want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    // 3. Identity BN (γ=1, β=0, μ=0, v=1, ε→0) leaves weights unchanged.
    #[test]
    fn identity_bn_unchanged() {
        let in_dim = 2_usize;
        let out_dim = 2_usize;
        let w = vec![0.3, -0.7, 0.9, 0.1];
        let b = vec![0.25, -0.5];
        let layer = make_layer(in_dim, out_dim, w.clone(), b.clone());
        let bn = BatchNormParams::new(
            vec![1.0, 1.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            1e-12, // ε → 0
            out_dim,
        )
        .expect("bn");
        let folded = fold_batchnorm(&layer, &bn).expect("fold");
        // s_i = 1/√(1 + 1e-12) ≈ 1 → weights essentially unchanged.
        for (a, b) in folded.w.iter().zip(w.iter()) {
            assert!((a - b).abs() < 1e-5, "weight changed: {a} vs {b}");
        }
        // b_fold = (b − 0)·1 + 0 = b.
        for (a, b) in folded.b.iter().zip(b.iter()) {
            assert!((a - b).abs() < 1e-5, "bias changed: {a} vs {b}");
        }
        assert!((folded.v_th - layer.v_th).abs() < 1e-6);
    }

    // 4. Shape validation: BN vector length must equal out_dim.
    #[test]
    fn fold_rejects_bad_bn_length() {
        let layer = make_layer(2, 2, vec![0.0; 4], vec![0.0; 2]);
        // Construct a BN with deliberately mismatched vectors by bypassing `new`.
        let bn = BatchNormParams {
            gamma: vec![1.0, 1.0, 1.0], // length 3 ≠ out_dim 2
            beta: vec![0.0, 0.0],
            running_mean: vec![0.0, 0.0],
            running_var: vec![1.0, 1.0],
            eps: 1e-5,
        };
        assert!(matches!(
            fold_batchnorm(&layer, &bn),
            Err(SnnError::BadShape { .. })
        ));
    }

    // 5. Fold rejects inconsistent layer weight length.
    #[test]
    fn fold_rejects_bad_layer_shape() {
        let layer = SnnLayer {
            w: vec![0.0; 5], // should be in_dim*out_dim = 4
            b: vec![0.0; 2],
            v_th: 1.0,
            in_dim: 2,
            out_dim: 2,
        };
        let bn = BatchNormParams::new(
            vec![1.0, 1.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            1e-5,
            2,
        )
        .expect("bn");
        assert!(matches!(
            fold_batchnorm(&layer, &bn),
            Err(SnnError::BadShape { .. })
        ));
    }

    // 6. Bias absorption: weights preserved, returned bias_current == old bias,
    //    stripped layer bias all zero.
    #[test]
    fn absorb_bias_strips_and_returns() {
        let layer = make_layer(3, 2, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], vec![0.7, -0.8]);
        let (bias_current, stripped) = absorb_bias_into_threshold(&layer).expect("absorb");
        assert_eq!(bias_current, vec![0.7, -0.8]);
        assert!(
            stripped.b.iter().all(|&x| x == 0.0),
            "stripped bias must be zero"
        );
        assert_eq!(stripped.w, layer.w, "weights must be preserved");
        assert_eq!(stripped.in_dim, layer.in_dim);
        assert_eq!(stripped.out_dim, layer.out_dim);
    }

    // 7. W·x + bias_current reproduces the original W·x + b forward.
    #[test]
    fn absorbed_current_reproduces_forward() {
        let layer = make_layer(2, 2, vec![0.5, -0.3, 0.2, 0.9], vec![0.4, -0.1]);
        let (bias_current, stripped) = absorb_bias_into_threshold(&layer).expect("absorb");
        let x = [1.2_f32, -0.6];
        // Original forward includes bias.
        let y_full = linear_forward(&layer, &x).expect("y");
        // Stripped forward (bias 0) + injected bias current must match.
        let y_stripped = linear_forward(&stripped, &x).expect("y");
        for i in 0..2 {
            let reconstructed = y_stripped[i] + bias_current[i];
            assert!(
                (reconstructed - y_full[i]).abs() < 1e-5,
                "channel {i}: {reconstructed} vs {}",
                y_full[i]
            );
        }
    }

    // 8. fold_and_absorb == fold then absorb (end-to-end).
    #[test]
    fn fold_and_absorb_matches_bn_linear() {
        let in_dim = 3_usize;
        let out_dim = 2_usize;
        let layer = make_layer(
            in_dim,
            out_dim,
            vec![0.2, -0.5, 0.7, 0.1, 0.4, -0.3],
            vec![0.05, -0.1],
        );
        let bn = BatchNormParams::new(
            vec![1.5, 0.8],
            vec![0.2, -0.4],
            vec![0.3, 0.1],
            vec![2.0, 0.5],
            1e-5,
            out_dim,
        )
        .expect("bn");
        let (bias_current, stripped) = fold_and_absorb(&layer, &bn).expect("fa");
        assert!(stripped.b.iter().all(|&x| x == 0.0));

        let x = [0.5_f32, -0.2, 0.9];
        // Reconstructed = stripped·x + bias_current must equal BN∘Linear.
        let y_stripped = linear_forward(&stripped, &x).expect("y");
        let want = bn_linear_reference(&layer, &bn, &x);
        for i in 0..out_dim {
            let reconstructed = y_stripped[i] + bias_current[i];
            assert!(
                (reconstructed - want[i]).abs() < 1e-4,
                "channel {i}: {reconstructed} vs {}",
                want[i]
            );
        }
    }

    // 9. linear_forward shape validation.
    #[test]
    fn linear_forward_shape_validation() {
        let layer = make_layer(3, 2, vec![0.0; 6], vec![0.0; 2]);
        assert!(matches!(
            linear_forward(&layer, &[0.0, 0.0]),
            Err(SnnError::IncompatibleLength { .. })
        ));
    }

    // 10. absorb rejects inconsistent shape.
    #[test]
    fn absorb_rejects_bad_shape() {
        let layer = SnnLayer {
            w: vec![0.0; 3],
            b: vec![0.0; 2],
            v_th: 1.0,
            in_dim: 2,
            out_dim: 2,
        };
        assert!(matches!(
            absorb_bias_into_threshold(&layer),
            Err(SnnError::BadShape { .. })
        ));
    }
}
