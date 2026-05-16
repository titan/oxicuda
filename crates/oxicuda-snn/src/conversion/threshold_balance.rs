//! Layer-wise threshold balancing across an arbitrary feed-forward chain.
//!
//! Given a stack of pre-trained ReLU layers and a representative sample of
//! per-layer activations, [`crate::conversion::threshold_balance::balance_layer_chain`] runs
//! [`crate::conversion::ann2snn::ann_to_snn_layer`] sequentially, threading the previous
//! layer's λ into the next call. The procedure preserves end-to-end firing
//! rates: each spiking layer's mean rate matches the original ANN activation
//! divided by the layer's percentile estimate.

use super::ann2snn::ann_to_snn_layer;
use crate::error::{SnnError, SnnResult};

/// Apply percentile-based threshold balancing to every layer of a feed-forward
/// chain in place.
///
/// * `weights_per_layer`     — mutable list of row-major weight matrices.
/// * `biases_per_layer`      — mutable list of bias vectors.
/// * `activations_per_layer` — non-empty per-layer activation samples used to
///   estimate the percentile λ.
/// * `dims` — `(layer_count + 1)` units, where `dims[i]` is the input width of
///   layer `i` and `dims[i + 1]` its output width.
/// * `percentile`            — quantile in `[0, 1]` (typically 0.99).
///
/// Returns the per-layer λ vector.
///
/// Errors: [`SnnError::EmptyInput`] for empty layer chains,
/// [`SnnError::IncompatibleLength`] when the lengths of the four input slices
/// disagree, [`SnnError::BadDim`] for zero dimensions, [`SnnError::BadShape`]
/// for size mismatches, and [`SnnError::OutOfRange`] for invalid percentiles.
pub fn balance_layer_chain(
    weights_per_layer: &mut [Vec<f32>],
    biases_per_layer: &mut [Vec<f32>],
    activations_per_layer: &[Vec<f32>],
    dims: &[usize],
    percentile: f32,
) -> SnnResult<Vec<f32>> {
    let l = weights_per_layer.len();
    if l == 0 {
        return Err(SnnError::EmptyInput);
    }
    if biases_per_layer.len() != l {
        return Err(SnnError::IncompatibleLength {
            a: l,
            b: biases_per_layer.len(),
        });
    }
    if activations_per_layer.len() != l {
        return Err(SnnError::IncompatibleLength {
            a: l,
            b: activations_per_layer.len(),
        });
    }
    if dims.len() != l + 1 {
        return Err(SnnError::IncompatibleLength {
            a: l + 1,
            b: dims.len(),
        });
    }
    if !percentile.is_finite() || !(0.0..=1.0).contains(&percentile) {
        return Err(SnnError::OutOfRange {
            name: "percentile".into(),
            val: percentile,
        });
    }

    let mut lambdas = Vec::with_capacity(l);
    let mut lambda_prev = 1.0_f32;
    for i in 0..l {
        let in_dim = dims[i];
        let out_dim = dims[i + 1];
        let (layer, lambda) = ann_to_snn_layer(
            &weights_per_layer[i],
            &biases_per_layer[i],
            &activations_per_layer[i],
            lambda_prev,
            percentile,
            in_dim,
            out_dim,
        )?;
        weights_per_layer[i] = layer.w;
        biases_per_layer[i] = layer.b;
        lambdas.push(lambda);
        lambda_prev = lambda;
    }
    Ok(lambdas)
}

#[cfg(test)]
mod tests {
    use super::*;

    type ChainData = (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>);

    fn synthetic_chain(
        layer_dims: &[usize],
        activation_min: f32,
        activation_max: f32,
        n_samples: usize,
    ) -> ChainData {
        let l = layer_dims.len() - 1;
        let mut weights = Vec::with_capacity(l);
        let mut biases = Vec::with_capacity(l);
        let mut activations = Vec::with_capacity(l);
        for i in 0..l {
            let in_dim = layer_dims[i];
            let out_dim = layer_dims[i + 1];
            let w = (0..in_dim * out_dim)
                .map(|k| 0.1 + (k as f32) * 0.01)
                .collect::<Vec<_>>();
            let b = (0..out_dim).map(|k| 0.05 * (k as f32 + 1.0)).collect();
            let act: Vec<f32> = (0..n_samples)
                .map(|k| {
                    let t = (k as f32) / (n_samples as f32 - 1.0).max(1.0);
                    activation_min + (activation_max - activation_min) * t
                })
                .collect();
            weights.push(w);
            biases.push(b);
            activations.push(act);
        }
        (weights, biases, activations)
    }

    #[test]
    fn returns_one_lambda_per_layer() {
        let dims = vec![3_usize, 4, 5, 2];
        let (mut w, mut b, a) = synthetic_chain(&dims, 0.0, 1.0, 64);
        let lambdas = balance_layer_chain(&mut w, &mut b, &a, &dims, 0.99).expect("balance");
        assert_eq!(lambdas.len(), dims.len() - 1);
        for &lam in &lambdas {
            assert!(lam > 0.0 && lam.is_finite(), "lambda={lam}");
        }
    }

    #[test]
    fn lambda_grows_monotonically_when_activations_grow() {
        // Layer i activations span [0, i+1]; the percentile must increase.
        let dims = vec![2_usize, 2, 2, 2];
        let l = dims.len() - 1;
        let mut weights: Vec<Vec<f32>> = Vec::new();
        let mut biases: Vec<Vec<f32>> = Vec::new();
        let mut activations: Vec<Vec<f32>> = Vec::new();
        for i in 0..l {
            let in_dim = dims[i];
            let out_dim = dims[i + 1];
            weights.push(vec![0.5_f32; in_dim * out_dim]);
            biases.push(vec![0.1_f32; out_dim]);
            let max = (i + 1) as f32; // 1.0, 2.0, 3.0
            let act: Vec<f32> = (0..32).map(|k| (k as f32) * max / 31.0).collect();
            activations.push(act);
        }
        let lambdas = balance_layer_chain(&mut weights, &mut biases, &activations, &dims, 0.99)
            .expect("balance");
        for w in lambdas.windows(2) {
            assert!(w[1] > w[0], "lambdas={lambdas:?}");
        }
    }

    #[test]
    fn rejects_inconsistent_lengths() {
        let mut w = vec![vec![0.0_f32; 4]];
        let mut b = vec![vec![0.0_f32; 2]; 2];
        let a = vec![vec![0.5_f32; 8]];
        let dims = vec![2_usize, 2];
        assert!(matches!(
            balance_layer_chain(&mut w, &mut b, &a, &dims, 0.99),
            Err(SnnError::IncompatibleLength { .. })
        ));
    }

    #[test]
    fn rejects_empty_chain() {
        let mut w: Vec<Vec<f32>> = Vec::new();
        let mut b: Vec<Vec<f32>> = Vec::new();
        let a: Vec<Vec<f32>> = Vec::new();
        let dims: Vec<usize> = vec![];
        assert!(matches!(
            balance_layer_chain(&mut w, &mut b, &a, &dims, 0.99),
            Err(SnnError::EmptyInput)
        ));
    }
}
