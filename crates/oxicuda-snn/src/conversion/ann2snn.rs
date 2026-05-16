//! ANN→SNN single-layer conversion via threshold balancing.
//!
//! Converts one fully-connected ReLU layer of a pre-trained ANN into an
//! integrate-and-fire spiking layer following the data-based rescaling scheme
//! of Rueckauer et al. (2017), "Conversion of Continuous-Valued Deep Networks
//! to Efficient Event-Driven Networks for Image Classification".
//!
//! Given ReLU activations `a = max(0, W·x + b)`, the q-th percentile
//! `λ = quantile(a, q)` is treated as a layer-specific normalisation factor.
//! The spiking layer's weights and biases are rescaled to
//!
//! ```text
//! W' = W · (λ_prev / λ),    b' = b / λ,    v_th = 1.
//! ```
//!
//! After conversion, the firing rate of layer `l` over a long simulation
//! window approximates `ReLU(W_l · r_{l−1} + b_l) / λ_l`, i.e. the original
//! activation expressed as a fraction of the chosen quantile.

use crate::error::{SnnError, SnnResult};

/// One spiking layer produced by ANN→SNN conversion.
#[derive(Debug, Clone)]
pub struct SnnLayer {
    /// Row-major weights of shape `(out_dim, in_dim)`.
    pub w: Vec<f32>,
    /// Bias vector of length `out_dim`.
    pub b: Vec<f32>,
    /// Spike threshold (always `1.0` after threshold balancing).
    pub v_th: f32,
    /// Number of input units (columns of `W`).
    pub in_dim: usize,
    /// Number of output units (rows of `W`).
    pub out_dim: usize,
}

/// Sorted-position quantile of a slice (linear interpolation between order
/// statistics). The probability `q` must be in `[0, 1]` and `values` must be
/// non-empty.
///
/// Errors: [`SnnError::EmptyInput`] when `values.is_empty()`,
/// [`SnnError::OutOfRange`] when `q ∉ [0, 1]` or `q` is not finite.
pub fn quantile(values: &[f32], q: f32) -> SnnResult<f32> {
    if values.is_empty() {
        return Err(SnnError::EmptyInput);
    }
    if !q.is_finite() || !(0.0..=1.0).contains(&q) {
        return Err(SnnError::OutOfRange {
            name: "q".into(),
            val: q,
        });
    }
    let mut sorted: Vec<f32> = values.to_vec();
    // Stable, NaN-aware ordering: NaN sinks to the end so it never affects
    // the percentile of well-defined activations.
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
    let n = sorted.len();
    if n == 1 {
        return Ok(sorted[0]);
    }
    let pos = q * (n as f32 - 1.0);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - (lo as f32);
    Ok(sorted[lo] * (1.0 - frac) + sorted[hi] * frac)
}

/// Convert one ReLU ANN layer into a spiking layer using percentile-based
/// threshold balancing.
///
/// * `weights` — row-major `(out_dim × in_dim)` weight matrix.
/// * `biases`  — bias vector of length `out_dim`.
/// * `activations` — sample of post-activation values used to estimate λ.
/// * `lambda_prev` — λ produced for the preceding layer (1.0 for the input).
/// * `percentile` — quantile in `[0, 1]` used to estimate the activation
///   scale; 0.99 is the standard choice.
///
/// Returns the converted [`SnnLayer`] and the layer's λ so that downstream
/// layers can be chained.
///
/// Errors: [`SnnError::BadDim`] for zero `in_dim`/`out_dim`,
/// [`SnnError::BadShape`] for size mismatches, [`SnnError::EmptyInput`] for
/// empty activations, and [`SnnError::OutOfRange`] when `percentile`,
/// `lambda_prev`, or the resulting λ is invalid.
#[allow(clippy::too_many_arguments)]
pub fn ann_to_snn_layer(
    weights: &[f32],
    biases: &[f32],
    activations: &[f32],
    lambda_prev: f32,
    percentile: f32,
    in_dim: usize,
    out_dim: usize,
) -> SnnResult<(SnnLayer, f32)> {
    if in_dim == 0 {
        return Err(SnnError::BadDim { got: in_dim });
    }
    if out_dim == 0 {
        return Err(SnnError::BadDim { got: out_dim });
    }
    if weights.len() != in_dim * out_dim {
        return Err(SnnError::BadShape {
            expected: in_dim * out_dim,
            got: weights.len(),
        });
    }
    if biases.len() != out_dim {
        return Err(SnnError::BadShape {
            expected: out_dim,
            got: biases.len(),
        });
    }
    if activations.is_empty() {
        return Err(SnnError::EmptyInput);
    }
    if !lambda_prev.is_finite() || lambda_prev <= 0.0 {
        return Err(SnnError::OutOfRange {
            name: "lambda_prev".into(),
            val: lambda_prev,
        });
    }
    let lambda = quantile(activations, percentile)?;
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(SnnError::OutOfRange {
            name: "lambda".into(),
            val: lambda,
        });
    }
    let scale_w = lambda_prev / lambda;
    let inv_lambda = 1.0 / lambda;
    let mut w = Vec::with_capacity(weights.len());
    for &wi in weights {
        w.push(wi * scale_w);
    }
    let mut b = Vec::with_capacity(biases.len());
    for &bi in biases {
        b.push(bi * inv_lambda);
    }
    Ok((
        SnnLayer {
            w,
            b,
            v_th: 1.0,
            in_dim,
            out_dim,
        },
        lambda,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_basic_correctness() {
        let v = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0];
        // q=0 → min, q=1 → max, q=0.5 → median.
        assert!((quantile(&v, 0.0).expect("q") - 0.0).abs() < 1e-6);
        assert!((quantile(&v, 1.0).expect("q") - 4.0).abs() < 1e-6);
        assert!((quantile(&v, 0.5).expect("q") - 2.0).abs() < 1e-6);
        // q=0.25 between sorted[1]=1.0 and sorted[2]=2.0 (linear interp).
        let q25 = quantile(&v, 0.25).expect("q");
        assert!((q25 - 1.0).abs() < 1e-6, "q25={q25}");
    }

    #[test]
    fn quantile_rejects_bad_inputs() {
        assert!(matches!(quantile(&[], 0.5), Err(SnnError::EmptyInput)));
        assert!(matches!(
            quantile(&[1.0_f32], 1.5),
            Err(SnnError::OutOfRange { .. })
        ));
        assert!(matches!(
            quantile(&[1.0_f32], -0.1),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn ann_to_snn_shape_correct() {
        let in_dim = 3;
        let out_dim = 2;
        let weights = vec![0.5_f32; in_dim * out_dim];
        let biases = vec![0.1_f32; out_dim];
        let activations: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let (layer, lambda) =
            ann_to_snn_layer(&weights, &biases, &activations, 1.0, 0.99, in_dim, out_dim)
                .expect("ann to snn");
        assert_eq!(layer.in_dim, in_dim);
        assert_eq!(layer.out_dim, out_dim);
        assert_eq!(layer.w.len(), in_dim * out_dim);
        assert_eq!(layer.b.len(), out_dim);
        assert!((layer.v_th - 1.0).abs() < 1e-6);
        assert!(lambda > 0.0);
    }

    #[test]
    fn ann_to_snn_scale_invariance() {
        // Scaling all activations + biases + weights by α should leave
        // (W' · x_norm + b') unchanged because λ scales linearly too.
        let in_dim = 4;
        let out_dim = 3;
        let weights: Vec<f32> = (0..in_dim * out_dim)
            .map(|i| 0.1 + i as f32 * 0.05)
            .collect();
        let biases: Vec<f32> = (0..out_dim).map(|i| 0.2 - i as f32 * 0.05).collect();
        let activations: Vec<f32> = (0..50).map(|i| i as f32 * 0.02).collect();
        let alpha = 7.5_f32;
        let weights_scaled: Vec<f32> = weights.iter().map(|w| w * alpha).collect();
        let biases_scaled: Vec<f32> = biases.iter().map(|b| b * alpha).collect();
        let activations_scaled: Vec<f32> = activations.iter().map(|a| a * alpha).collect();

        let (layer_a, _) =
            ann_to_snn_layer(&weights, &biases, &activations, 1.0, 0.95, in_dim, out_dim)
                .expect("layer a");
        let (layer_b, _) = ann_to_snn_layer(
            &weights_scaled,
            &biases_scaled,
            &activations_scaled,
            1.0,
            0.95,
            in_dim,
            out_dim,
        )
        .expect("layer b");

        for (&wa, &wb) in layer_a.w.iter().zip(layer_b.w.iter()) {
            assert!((wa - wb).abs() < 1e-4, "{wa} vs {wb}");
        }
        for (&ba, &bb) in layer_a.b.iter().zip(layer_b.b.iter()) {
            assert!((ba - bb).abs() < 1e-4, "{ba} vs {bb}");
        }
    }

    #[test]
    fn ann_to_snn_rejects_bad_shapes() {
        let activations = vec![0.5_f32; 10];
        let weights = vec![0.0_f32; 6];
        let biases = vec![0.0_f32; 2];
        assert!(matches!(
            ann_to_snn_layer(&weights, &biases, &activations, 1.0, 0.99, 0, 2),
            Err(SnnError::BadDim { .. })
        ));
        assert!(matches!(
            ann_to_snn_layer(&weights, &biases, &activations, 1.0, 0.99, 3, 0),
            Err(SnnError::BadDim { .. })
        ));
        // wrong weight buffer length
        let bad_w = vec![0.0_f32; 5];
        assert!(matches!(
            ann_to_snn_layer(&bad_w, &biases, &activations, 1.0, 0.99, 3, 2),
            Err(SnnError::BadShape { .. })
        ));
        // empty activations
        let empty_act: Vec<f32> = Vec::new();
        assert!(matches!(
            ann_to_snn_layer(&weights, &biases, &empty_act, 1.0, 0.99, 3, 2),
            Err(SnnError::EmptyInput)
        ));
        // bad lambda_prev
        assert!(matches!(
            ann_to_snn_layer(&weights, &biases, &activations, -1.0, 0.99, 3, 2),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn ann_to_snn_threshold_is_unity() {
        let in_dim = 2;
        let out_dim = 2;
        let weights = vec![1.0_f32; 4];
        let biases = vec![0.0_f32; 2];
        let activations = vec![0.4_f32, 0.5, 0.6, 0.7];
        let (layer, lambda) =
            ann_to_snn_layer(&weights, &biases, &activations, 1.0, 0.99, in_dim, out_dim)
                .expect("layer");
        assert!((layer.v_th - 1.0).abs() < 1e-6);
        // For a uniformly spaced sample, the 99th percentile is close to
        // the maximum value (0.7) — verifies that λ tracks the scale.
        assert!(lambda > 0.5 && lambda <= 0.7 + 1e-4);
    }
}
