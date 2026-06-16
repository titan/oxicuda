//! DANets: Deep Abstract Networks for Tabular Data (Chen et al. 2022, AAAI).
//!
//! The core building block is the **Abstract Layer** (ABSTLAY): a learnable
//! sparse feature-selection mask groups raw input features into a set of
//! *abstract features*, after which a learnable affine transform aggregates
//! each group.  The sparse mask is produced row-wise by **sparsemax**
//! (Martins & Astudillo 2016), the same projection onto the probability
//! simplex used elsewhere in this crate, so each abstract feature attends to
//! a small, learnable subset of the inputs (rows sum to 1 with many exact
//! zeros).
//!
//! Several abstract layers are stacked into a network.  The first layer maps
//! `input_dim → n_abstract`; subsequent layers map `n_abstract → n_abstract`
//! and are wrapped with a residual *shortcut* (the paper's Basic Block design)
//! so gradients/signal can skip a layer where the dimensions match.  A final
//! linear projection maps the last abstract representation to `output_dim`.
//!
//! This is a forward-only (inference) implementation with deterministic
//! initialisation driven by [`LcgRng`]; there is no training loop.

use crate::attention::sparsemax::sparsemax;
use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;

// ─── DanetConfig ───────────────────────────────────────────────────────────────

/// Configuration for a [`Danet`].
#[derive(Debug, Clone)]
pub struct DanetConfig {
    /// Number of raw input features.
    pub input_dim: usize,
    /// Number of abstract features produced per abstract layer (`k`).
    pub n_abstract: usize,
    /// Number of stacked abstract layers (`>= 1`).
    pub n_layers: usize,
    /// Output dimension (1 for regression, `n_classes` for classification).
    pub output_dim: usize,
    /// Number of feature groups used for the sparse grouping prior (`>= 1`).
    pub n_groups: usize,
}

// ─── AbstractLayer ───────────────────────────────────────────────────────────

/// A single Abstract Layer (ABSTLAY).
///
/// Holds, for every abstract feature, a row of feature-selection logits over
/// the inputs.  At forward time each row is passed through `sparsemax` to form
/// a sparse, simplex-normalised mask; the masked weighted sum of the inputs is
/// the *raw* abstract value, which is then passed through a learnable affine
/// transform (per-abstract scale and bias) and a gated `tanh` activation.
#[derive(Debug, Clone)]
pub struct AbstractLayer {
    /// Feature-selection mask logits laid out `n_abstract × input_dim` (row-major).
    mask_logits: Vec<f32>,
    /// Per-abstract affine scale, length `n_abstract`.
    transform_w: Vec<f32>,
    /// Per-abstract affine bias, length `n_abstract`.
    transform_b: Vec<f32>,
    /// Number of input features consumed.
    input_dim: usize,
    /// Number of abstract features produced.
    n_abstract: usize,
}

impl AbstractLayer {
    /// Construct a new abstract layer mapping `input_dim → n_abstract`.
    ///
    /// # Errors
    /// Returns [`TabularError::InvalidFeatureCount`] if `input_dim == 0` and
    /// [`TabularError::InvalidEmbedDim`] if `n_abstract == 0`.
    pub fn new(input_dim: usize, n_abstract: usize, rng: &mut LcgRng) -> TabularResult<Self> {
        if input_dim == 0 {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }
        if n_abstract == 0 {
            return Err(TabularError::InvalidEmbedDim { dim: 0 });
        }

        // Mask logits with a moderate spread so that the sparsemax projection
        // produces genuinely sparse rows (init-independent invariants are used
        // in tests, but the spread keeps the masks meaningfully selective).
        let mut mask_logits = vec![0.0_f32; n_abstract * input_dim];
        rng.fill_normal_scaled(&mut mask_logits, 1.0);

        // Affine transform: scale initialised near 1, bias at 0 (identity-ish).
        let mut transform_w = vec![0.0_f32; n_abstract];
        rng.fill_normal_scaled(&mut transform_w, 0.1);
        for w in &mut transform_w {
            *w += 1.0;
        }
        let transform_b = vec![0.0_f32; n_abstract];

        Ok(Self {
            mask_logits,
            transform_w,
            transform_b,
            input_dim,
            n_abstract,
        })
    }

    /// Number of input features this layer consumes.
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Number of abstract features this layer produces.
    #[must_use]
    pub fn n_abstract(&self) -> usize {
        self.n_abstract
    }

    /// Compute the sparse feature-selection mask, laid out `n_abstract × input_dim`.
    ///
    /// Each row is `sparsemax(logits_row)`: non-negative, sums to 1, and
    /// generally sparse (many exact zeros) for spread logits.
    ///
    /// # Errors
    /// Propagates errors from the underlying `sparsemax` projection.
    pub fn feature_mask(&self) -> TabularResult<Vec<f32>> {
        let mut mask = vec![0.0_f32; self.n_abstract * self.input_dim];
        for a in 0..self.n_abstract {
            let row = &self.mask_logits[a * self.input_dim..(a + 1) * self.input_dim];
            let masked = sparsemax(row)?;
            mask[a * self.input_dim..(a + 1) * self.input_dim].copy_from_slice(&masked);
        }
        Ok(mask)
    }

    /// Forward pass: `input_dim → n_abstract`.
    ///
    /// For each abstract feature `a`, the raw value is the sparse-masked
    /// weighted sum `Σ_i mask[a,i] · x_i`; it is then transformed by the
    /// learnable affine `w_a · raw + b_a` and a gated `tanh` activation
    /// `t · σ(t)`-style nonlinearity (here `tanh`).
    ///
    /// # Errors
    /// Returns [`TabularError::DimensionMismatch`] if `x.len() != input_dim`.
    pub fn forward(&self, x: &[f32]) -> TabularResult<Vec<f32>> {
        if x.len() != self.input_dim {
            return Err(TabularError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }

        let mut out = vec![0.0_f32; self.n_abstract];
        for (a, oa) in out.iter_mut().enumerate() {
            let row = &self.mask_logits[a * self.input_dim..(a + 1) * self.input_dim];
            let mask = sparsemax(row)?;
            let agg: f32 = mask.iter().zip(x.iter()).map(|(&m, &xi)| m * xi).sum();
            let affine = self.transform_w[a] * agg + self.transform_b[a];
            *oa = affine.tanh();
        }
        Ok(out)
    }

    /// Number of learnable parameters in this layer.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.mask_logits.len() + self.transform_w.len() + self.transform_b.len()
    }
}

// ─── Danet ─────────────────────────────────────────────────────────────────────

/// Deep Abstract Network: a stack of [`AbstractLayer`]s with shortcut paths
/// and a final linear projection to `output_dim`.
#[derive(Debug, Clone)]
pub struct Danet {
    /// Stacked abstract layers; layer 0 is `input_dim → n_abstract`, the rest
    /// are `n_abstract → n_abstract`.
    layers: Vec<AbstractLayer>,
    /// Output projection weight, laid out `output_dim × n_abstract` (row-major).
    output_w: Vec<f32>,
    /// Output projection bias, length `output_dim`.
    output_b: Vec<f32>,
    /// Resolved configuration.
    config: DanetConfig,
}

impl Danet {
    /// Construct a new `Danet` from the given configuration.
    ///
    /// # Errors
    /// Returns the appropriate [`TabularError`] variant if any configuration
    /// field is zero (`input_dim`, `n_abstract`, `n_layers`, `output_dim`,
    /// `n_groups`).
    pub fn new(cfg: DanetConfig, rng: &mut LcgRng) -> TabularResult<Self> {
        if cfg.input_dim == 0 {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }
        if cfg.n_abstract == 0 {
            return Err(TabularError::InvalidEmbedDim { dim: 0 });
        }
        if cfg.n_layers == 0 {
            return Err(TabularError::InvalidStepCount { steps: 0 });
        }
        if cfg.output_dim == 0 {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }
        if cfg.n_groups == 0 {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }

        let mut layers = Vec::with_capacity(cfg.n_layers);
        // First layer maps the raw inputs into the abstract space.
        layers.push(AbstractLayer::new(cfg.input_dim, cfg.n_abstract, rng)?);
        // Remaining layers operate within the abstract space (shortcuts apply).
        for _ in 1..cfg.n_layers {
            layers.push(AbstractLayer::new(cfg.n_abstract, cfg.n_abstract, rng)?);
        }

        let std_out = (2.0_f32 / (cfg.n_abstract + cfg.output_dim) as f32).sqrt();
        let mut output_w = vec![0.0_f32; cfg.output_dim * cfg.n_abstract];
        rng.fill_normal_scaled(&mut output_w, std_out);
        let output_b = vec![0.0_f32; cfg.output_dim];

        Ok(Self {
            layers,
            output_w,
            output_b,
            config: cfg,
        })
    }

    /// Read-only access to the resolved configuration.
    #[must_use]
    pub fn config(&self) -> &DanetConfig {
        &self.config
    }

    /// Forward pass: `input_dim → output_dim`.
    ///
    /// The first abstract layer lifts the inputs to the abstract space; each
    /// subsequent abstract layer is wrapped in a residual shortcut
    /// (`h ← h + ABSTLAY(h)`) since the dimensions match.  The final abstract
    /// representation is projected linearly to `output_dim`.
    ///
    /// # Errors
    /// Returns [`TabularError::DimensionMismatch`] if `x.len() != input_dim`,
    /// or propagates errors from the abstract layers.
    pub fn forward(&self, x: &[f32]) -> TabularResult<Vec<f32>> {
        if x.len() != self.config.input_dim {
            return Err(TabularError::DimensionMismatch {
                expected: self.config.input_dim,
                got: x.len(),
            });
        }

        // Layer 0: input_dim → n_abstract.
        let mut h = self.layers[0].forward(x)?;

        // Layers 1..: n_abstract → n_abstract with a residual shortcut.
        for layer in self.layers.iter().skip(1) {
            let transformed = layer.forward(&h)?;
            for (hi, &ti) in h.iter_mut().zip(transformed.iter()) {
                *hi += ti;
            }
        }

        // Output projection: output_dim × n_abstract.
        let n_abs = self.config.n_abstract;
        let mut logits = self.output_b.clone();
        for (o, lo) in logits.iter_mut().enumerate() {
            let base = o * n_abs;
            for (d, &hv) in h.iter().enumerate() {
                *lo += self.output_w[base + d] * hv;
            }
        }
        Ok(logits)
    }

    /// Batch forward: `x` is a flat `[batch_size * input_dim]` buffer.
    ///
    /// Returns `[batch_size * output_dim]`.
    ///
    /// # Errors
    /// Returns [`TabularError::DimensionMismatch`] if the buffer length does
    /// not equal `batch_size * input_dim`.
    pub fn forward_batch(&self, x: &[f32], batch_size: usize) -> TabularResult<Vec<f32>> {
        let in_d = self.config.input_dim;
        if x.len() != batch_size * in_d {
            return Err(TabularError::DimensionMismatch {
                expected: batch_size * in_d,
                got: x.len(),
            });
        }
        let mut out = Vec::with_capacity(batch_size * self.config.output_dim);
        for b in 0..batch_size {
            let row = &x[b * in_d..(b + 1) * in_d];
            let pred = self.forward(row)?;
            out.extend_from_slice(&pred);
        }
        Ok(out)
    }

    /// Total number of learnable parameters in the network.
    #[must_use]
    pub fn n_params(&self) -> usize {
        let layer_params: usize = self.layers.iter().map(AbstractLayer::n_params).sum();
        layer_params + self.output_w.len() + self.output_b.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::sparsemax::sparsemax;

    fn small_cfg() -> DanetConfig {
        DanetConfig {
            input_dim: 8,
            n_abstract: 4,
            n_layers: 3,
            output_dim: 3,
            n_groups: 2,
        }
    }

    #[test]
    fn feature_mask_rows_sum_to_one_and_nonneg() {
        let mut rng = LcgRng::new(42);
        let layer = AbstractLayer::new(8, 4, &mut rng).expect("new should succeed");
        let mask = layer.feature_mask().expect("feature_mask should succeed");
        assert_eq!(mask.len(), 4 * 8);
        for a in 0..4 {
            let row = &mask[a * 8..(a + 1) * 8];
            let s: f32 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "row {a} sum = {s}");
            assert!(row.iter().all(|&v| v >= -1e-7), "row {a} has negatives");
        }
    }

    #[test]
    fn sparsemax_one_hot_dominant_logit() {
        // A dominant first logit produces a near one-hot mask.
        let z = [50.0_f32, 0.0, 0.0, 0.0, 0.0];
        let out = sparsemax(&z).expect("sparsemax should succeed");
        assert!((out[0] - 1.0).abs() < 1e-5, "expected one-hot, got {out:?}");
        assert!(out[1..].iter().all(|&v| v < 1e-5));
    }

    #[test]
    fn sparsemax_uniform_logits_uniform_output() {
        // All-equal logits → uniform 1/d.
        let d = 6usize;
        let z = vec![0.7_f32; d];
        let out = sparsemax(&z).expect("sparsemax should succeed");
        let expected = 1.0_f32 / d as f32;
        for &v in &out {
            assert!((v - expected).abs() < 1e-5, "got {v}, expected {expected}");
        }
    }

    #[test]
    fn sparsemax_is_sparse_for_spread_logits() {
        // A spread vector should yield exact zeros in the projection.
        let z = [5.0_f32, 0.0, -3.0, -4.0, -5.0];
        let out = sparsemax(&z).expect("sparsemax should succeed");
        let zeros = out.iter().filter(|&&v| v == 0.0).count();
        assert!(zeros > 0, "expected some exact zeros, got {out:?}");
        let s: f32 = out.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn feature_mask_shape() {
        let mut rng = LcgRng::new(7);
        let layer = AbstractLayer::new(10, 5, &mut rng).expect("new should succeed");
        let mask = layer.feature_mask().expect("feature_mask should succeed");
        assert_eq!(mask.len(), 5 * 10);
    }

    #[test]
    fn abstract_layer_forward_length() {
        let mut rng = LcgRng::new(11);
        let layer = AbstractLayer::new(8, 4, &mut rng).expect("new should succeed");
        let x = vec![0.3_f32; 8];
        let out = layer.forward(&x).expect("forward should succeed");
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn abstract_layer_forward_finite_and_bounded() {
        let mut rng = LcgRng::new(123);
        let layer = AbstractLayer::new(6, 3, &mut rng).expect("new should succeed");
        let x = vec![2.5_f32, -1.0, 0.0, 3.3, -2.2, 1.1];
        let out = layer.forward(&x).expect("forward should succeed");
        // tanh activation → bounded in (-1, 1) and finite.
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out.iter().all(|&v| v.abs() <= 1.0 + 1e-6));
    }

    #[test]
    fn abstract_layer_wrong_length_errs() {
        let mut rng = LcgRng::new(5);
        let layer = AbstractLayer::new(8, 4, &mut rng).expect("new should succeed");
        let x = vec![0.3_f32; 7];
        assert!(matches!(
            layer.forward(&x),
            Err(TabularError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn danet_forward_length() {
        let mut rng = LcgRng::new(42);
        let model = Danet::new(small_cfg(), &mut rng).expect("value should be present");
        let x = vec![0.5_f32; 8];
        let out = model.forward(&x).expect("forward should succeed");
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn danet_forward_finite() {
        let mut rng = LcgRng::new(99);
        let model = Danet::new(small_cfg(), &mut rng).expect("value should be present");
        let x = vec![0.2_f32, -0.5, 1.0, 0.0, 0.3, -1.1, 2.0, 0.7];
        let out = model.forward(&x).expect("forward should succeed");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn danet_n_layers_one_works() {
        let cfg = DanetConfig {
            input_dim: 5,
            n_abstract: 3,
            n_layers: 1,
            output_dim: 2,
            n_groups: 1,
        };
        let mut rng = LcgRng::new(3);
        let model = Danet::new(cfg, &mut rng).expect("new should succeed");
        let x = vec![0.1_f32; 5];
        let out = model.forward(&x).expect("forward should succeed");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn danet_single_abstract_feature() {
        let cfg = DanetConfig {
            input_dim: 4,
            n_abstract: 1,
            n_layers: 2,
            output_dim: 1,
            n_groups: 1,
        };
        let mut rng = LcgRng::new(8);
        let model = Danet::new(cfg, &mut rng).expect("new should succeed");
        let x = vec![0.4_f32; 4];
        let out = model.forward(&x).expect("forward should succeed");
        assert_eq!(out.len(), 1);
        assert!(out[0].is_finite());
    }

    #[test]
    fn danet_deterministic_given_seed() {
        let mut rng_a = LcgRng::new(2024);
        let mut rng_b = LcgRng::new(2024);
        let model_a = Danet::new(small_cfg(), &mut rng_a).expect("value should be present");
        let model_b = Danet::new(small_cfg(), &mut rng_b).expect("value should be present");
        let x = vec![0.33_f32; 8];
        let out_a = model_a.forward(&x).expect("forward should succeed");
        let out_b = model_b.forward(&x).expect("forward should succeed");
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn danet_changing_x_changes_output() {
        let mut rng = LcgRng::new(17);
        let model = Danet::new(small_cfg(), &mut rng).expect("value should be present");
        let x1 = vec![0.5_f32; 8];
        let mut x2 = x1.clone();
        x2[0] = -3.0;
        let o1 = model.forward(&x1).expect("forward should succeed");
        let o2 = model.forward(&x2).expect("forward should succeed");
        assert_ne!(o1, o2);
    }

    #[test]
    fn danet_wrong_input_length_errs() {
        let mut rng = LcgRng::new(1);
        let model = Danet::new(small_cfg(), &mut rng).expect("value should be present");
        let x = vec![0.5_f32; 7];
        assert!(matches!(
            model.forward(&x),
            Err(TabularError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn danet_n_params_positive_and_formula() {
        let mut rng = LcgRng::new(42);
        let cfg = small_cfg();
        let model = Danet::new(cfg.clone(), &mut rng).expect("value should be present");
        // Layer 0: mask (n_abstract*input_dim) + 2*n_abstract.
        let l0 = cfg.n_abstract * cfg.input_dim + 2 * cfg.n_abstract;
        // Layers 1..: each mask (n_abstract*n_abstract) + 2*n_abstract.
        let lk = cfg.n_abstract * cfg.n_abstract + 2 * cfg.n_abstract;
        let out = cfg.output_dim * cfg.n_abstract + cfg.output_dim;
        let expected = l0 + (cfg.n_layers - 1) * lk + out;
        assert_eq!(model.n_params(), expected);
        assert!(model.n_params() > 0);
    }

    #[test]
    fn danet_err_input_dim_zero() {
        let cfg = DanetConfig {
            input_dim: 0,
            n_abstract: 4,
            n_layers: 2,
            output_dim: 2,
            n_groups: 1,
        };
        let mut rng = LcgRng::new(1);
        assert!(Danet::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn danet_err_n_abstract_zero() {
        let cfg = DanetConfig {
            input_dim: 4,
            n_abstract: 0,
            n_layers: 2,
            output_dim: 2,
            n_groups: 1,
        };
        let mut rng = LcgRng::new(1);
        assert!(Danet::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn danet_err_n_layers_zero() {
        let cfg = DanetConfig {
            input_dim: 4,
            n_abstract: 4,
            n_layers: 0,
            output_dim: 2,
            n_groups: 1,
        };
        let mut rng = LcgRng::new(1);
        assert!(Danet::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn danet_err_output_dim_zero() {
        let cfg = DanetConfig {
            input_dim: 4,
            n_abstract: 4,
            n_layers: 2,
            output_dim: 0,
            n_groups: 1,
        };
        let mut rng = LcgRng::new(1);
        assert!(Danet::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn danet_err_n_groups_zero() {
        let cfg = DanetConfig {
            input_dim: 4,
            n_abstract: 4,
            n_layers: 2,
            output_dim: 2,
            n_groups: 0,
        };
        let mut rng = LcgRng::new(1);
        assert!(Danet::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn abstract_layer_err_input_dim_zero() {
        let mut rng = LcgRng::new(1);
        assert!(AbstractLayer::new(0, 4, &mut rng).is_err());
    }

    #[test]
    fn abstract_layer_err_n_abstract_zero() {
        let mut rng = LcgRng::new(1);
        assert!(AbstractLayer::new(4, 0, &mut rng).is_err());
    }

    #[test]
    fn danet_batch_forward_shape() {
        let mut rng = LcgRng::new(64);
        let model = Danet::new(small_cfg(), &mut rng).expect("value should be present");
        let x = vec![0.25_f32; 3 * 8];
        let out = model
            .forward_batch(&x, 3)
            .expect("forward_batch should succeed");
        assert_eq!(out.len(), 3 * 3);
    }

    #[test]
    fn danet_batch_forward_wrong_len_errs() {
        let mut rng = LcgRng::new(64);
        let model = Danet::new(small_cfg(), &mut rng).expect("value should be present");
        let x = vec![0.25_f32; 3 * 8 + 1];
        assert!(model.forward_batch(&x, 3).is_err());
    }
}
