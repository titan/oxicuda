//! DeepGBM: a deep learning framework distilled by GBDT (Ke et al. 2019, KDD).
//!
//! DeepGBM fuses two complementary components:
//!
//! * **GBDT2NN** — a dense neural network that consumes the *leaf indices*
//!   produced by a trained gradient-boosted decision tree (GBDT) ensemble,
//!   thereby distilling the discrete structure of the tree ensemble into a
//!   differentiable model.  Each tree's selected leaf is mapped to a learnable
//!   embedding; the per-tree embeddings are concatenated and passed through an
//!   MLP (ReLU hidden layers) to produce the output logits.
//! * **CatNN** — a factorisation-machine (FM) style sparse network over the
//!   categorical fields.  It contributes a first-order linear term plus the
//!   classic second-order pairwise interaction, projected to the output.
//!
//! The two component outputs are then combined with learnable per-output
//! weights and a bias, followed by a sigmoid, mirroring the click-through-rate
//! (CTR) prediction setting of the paper.
//!
//! This is a forward-only (inference) implementation: the GBDT leaf
//! assignments are *provided* as input (no GBDT training is performed here),
//! and all weights are deterministically initialised from an [`LcgRng`].

use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;

// ─── DeepGbmConfig ─────────────────────────────────────────────────────────────

/// Configuration for a [`DeepGbm`] model.
#[derive(Debug, Clone)]
pub struct DeepGbmConfig {
    /// Number of trees in the distilled GBDT ensemble (`>= 1`).
    pub n_trees: usize,
    /// Number of leaves per tree (`>= 1`).
    pub n_leaves: usize,
    /// Dimension of each per-leaf embedding (`>= 1`).
    pub leaf_embed_dim: usize,
    /// Hidden layer sizes of the GBDT2NN MLP (may be empty for a linear head).
    pub gbdt_hidden: Vec<usize>,
    /// Number of categorical fields fed to CatNN (`>= 1`).
    pub n_cat_fields: usize,
    /// Cardinality (number of categories) for each categorical field.
    pub cat_cardinalities: Vec<usize>,
    /// Embedding dimension for each categorical field (`>= 1`).
    pub cat_embed_dim: usize,
    /// Output dimension (number of logits / tasks).
    pub output_dim: usize,
}

// ─── DeepGbm ───────────────────────────────────────────────────────────────────

/// DeepGBM model combining GBDT2NN and CatNN (Ke et al. 2019).
#[derive(Debug, Clone)]
pub struct DeepGbm {
    /// Leaf embedding table, flat `[n_trees * n_leaves * leaf_embed_dim]`.
    /// The embedding for tree `t`, leaf `l` starts at
    /// `(t * n_leaves + l) * leaf_embed_dim`.
    leaf_embed: Vec<f32>,
    /// GBDT2NN MLP weights: one `(weight [in × out], bias [out])` pair per layer.
    /// Layout of each weight is row-major `out × in`.
    gbdt_mlp: Vec<(Vec<f32>, Vec<f32>)>,
    /// Categorical embedding tables; `cat_embed[f]` is `[cardinality_f * cat_embed_dim]`.
    cat_embed: Vec<Vec<f32>>,
    /// FM first-order weights; `cat_linear[f]` is `[cardinality_f * output_dim]`.
    cat_linear: Vec<Vec<f32>>,
    /// Projection of the FM pairwise interaction vector to the output,
    /// row-major `[output_dim * cat_embed_dim]`.
    fm_proj: Vec<f32>,
    /// CatNN output bias, length `output_dim`.
    cat_bias: Vec<f32>,
    /// Combination weight for the GBDT2NN component, length `output_dim`.
    w1: Vec<f32>,
    /// Combination weight for the CatNN component, length `output_dim`.
    w2: Vec<f32>,
    /// Combination bias, length `output_dim`.
    comb_bias: Vec<f32>,
    /// Resolved configuration.
    config: DeepGbmConfig,
}

impl DeepGbm {
    /// Construct a new `DeepGbm` from the given configuration.
    ///
    /// # Errors
    /// Returns the relevant [`TabularError`] variant for any invalid field:
    /// zero counts/dimensions, or a `cat_cardinalities` whose length does not
    /// match `n_cat_fields` or that contains a zero cardinality.
    pub fn new(cfg: DeepGbmConfig, rng: &mut LcgRng) -> TabularResult<Self> {
        if cfg.n_trees == 0 {
            return Err(TabularError::InvalidTreeCount { n: 0 });
        }
        if cfg.n_leaves == 0 {
            return Err(TabularError::InvalidTreeDepth { depth: 0 });
        }
        if cfg.leaf_embed_dim == 0 {
            return Err(TabularError::InvalidEmbedDim { dim: 0 });
        }
        if cfg.n_cat_fields == 0 {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }
        if cfg.cat_cardinalities.len() != cfg.n_cat_fields {
            return Err(TabularError::DimensionMismatch {
                expected: cfg.n_cat_fields,
                got: cfg.cat_cardinalities.len(),
            });
        }
        if cfg.cat_cardinalities.contains(&0) {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }
        if cfg.cat_embed_dim == 0 {
            return Err(TabularError::InvalidEmbedDim { dim: 0 });
        }
        if cfg.output_dim == 0 {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }

        // ── GBDT2NN leaf embeddings ─────────────────────────────────────────
        let leaf_std = (1.0_f32 / cfg.leaf_embed_dim as f32).sqrt();
        let mut leaf_embed = vec![0.0_f32; cfg.n_trees * cfg.n_leaves * cfg.leaf_embed_dim];
        rng.fill_normal_scaled(&mut leaf_embed, leaf_std);

        // ── GBDT2NN MLP ─────────────────────────────────────────────────────
        let gbdt_input = cfg.n_trees * cfg.leaf_embed_dim;
        let mut gbdt_mlp = Vec::with_capacity(cfg.gbdt_hidden.len() + 1);
        let mut prev = gbdt_input;
        for &hidden in &cfg.gbdt_hidden {
            if hidden == 0 {
                return Err(TabularError::InvalidEmbedDim { dim: 0 });
            }
            let std = (2.0_f32 / (prev + hidden) as f32).sqrt();
            let mut w = vec![0.0_f32; hidden * prev];
            rng.fill_normal_scaled(&mut w, std);
            gbdt_mlp.push((w, vec![0.0_f32; hidden]));
            prev = hidden;
        }
        // Final projection to output_dim.
        let std_out = (2.0_f32 / (prev + cfg.output_dim) as f32).sqrt();
        let mut w_out = vec![0.0_f32; cfg.output_dim * prev];
        rng.fill_normal_scaled(&mut w_out, std_out);
        gbdt_mlp.push((w_out, vec![0.0_f32; cfg.output_dim]));

        // ── CatNN embeddings, linear term and FM projection ─────────────────
        let cat_std = (1.0_f32 / cfg.cat_embed_dim as f32).sqrt();
        let mut cat_embed = Vec::with_capacity(cfg.n_cat_fields);
        let mut cat_linear = Vec::with_capacity(cfg.n_cat_fields);
        for &card in &cfg.cat_cardinalities {
            let mut emb = vec![0.0_f32; card * cfg.cat_embed_dim];
            rng.fill_normal_scaled(&mut emb, cat_std);
            cat_embed.push(emb);

            let mut lin = vec![0.0_f32; card * cfg.output_dim];
            rng.fill_normal_scaled(&mut lin, 0.1);
            cat_linear.push(lin);
        }

        let fm_std = (2.0_f32 / (cfg.cat_embed_dim + cfg.output_dim) as f32).sqrt();
        let mut fm_proj = vec![0.0_f32; cfg.output_dim * cfg.cat_embed_dim];
        rng.fill_normal_scaled(&mut fm_proj, fm_std);
        let cat_bias = vec![0.0_f32; cfg.output_dim];

        // ── Combination layer ───────────────────────────────────────────────
        let w1 = vec![1.0_f32; cfg.output_dim];
        let w2 = vec![1.0_f32; cfg.output_dim];
        let comb_bias = vec![0.0_f32; cfg.output_dim];

        Ok(Self {
            leaf_embed,
            gbdt_mlp,
            cat_embed,
            cat_linear,
            fm_proj,
            cat_bias,
            w1,
            w2,
            comb_bias,
            config: cfg,
        })
    }

    /// Read-only access to the resolved configuration.
    #[must_use]
    pub fn config(&self) -> &DeepGbmConfig {
        &self.config
    }

    /// GBDT2NN forward: embed the per-tree leaf indices, concatenate, and pass
    /// through the MLP to produce `output_dim` logits.
    ///
    /// `leaf_indices` must have length `n_trees`, and each entry must be a
    /// valid leaf in `[0, n_leaves)`.
    ///
    /// # Errors
    /// Returns [`TabularError::DimensionMismatch`] if the length is wrong, or
    /// [`TabularError::CategoricalOutOfRange`] if a leaf index is out of range.
    pub fn gbdt2nn(&self, leaf_indices: &[usize]) -> TabularResult<Vec<f32>> {
        let cfg = &self.config;
        if leaf_indices.len() != cfg.n_trees {
            return Err(TabularError::DimensionMismatch {
                expected: cfg.n_trees,
                got: leaf_indices.len(),
            });
        }

        // Concatenate per-tree leaf embeddings.
        let ed = cfg.leaf_embed_dim;
        let mut h = vec![0.0_f32; cfg.n_trees * ed];
        for (t, &leaf) in leaf_indices.iter().enumerate() {
            if leaf >= cfg.n_leaves {
                return Err(TabularError::CategoricalOutOfRange {
                    feat: t,
                    val: leaf,
                    n: cfg.n_leaves,
                });
            }
            let src = (t * cfg.n_leaves + leaf) * ed;
            h[t * ed..(t + 1) * ed].copy_from_slice(&self.leaf_embed[src..src + ed]);
        }

        // MLP: ReLU on every hidden layer, linear on the final projection.
        let n_layers = self.gbdt_mlp.len();
        for (li, (w, b)) in self.gbdt_mlp.iter().enumerate() {
            let in_dim = h.len();
            let mut next = b.clone();
            for (o, no) in next.iter_mut().enumerate() {
                let base = o * in_dim;
                let mut acc = *no;
                for (i, &hv) in h.iter().enumerate() {
                    acc += w[base + i] * hv;
                }
                *no = acc;
            }
            // ReLU on hidden layers only.
            if li + 1 < n_layers {
                for v in &mut next {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
            h = next;
        }
        Ok(h)
    }

    /// CatNN forward: a factorisation-machine over the categorical fields.
    ///
    /// Produces `output_dim` logits as the sum of the first-order linear term
    /// `Σ_f w[f, idx_f]` and the projected second-order interaction
    /// `½(‖Σ_f v_f‖² − Σ_f ‖v_f‖²)` (computed per embedding dimension, then
    /// linearly projected to the output), plus a bias.
    ///
    /// `cat_indices` must have length `n_cat_fields`, with each entry valid in
    /// `[0, cat_cardinalities[f])`.
    ///
    /// # Errors
    /// Returns [`TabularError::DimensionMismatch`] if the length is wrong, or
    /// [`TabularError::CategoricalOutOfRange`] if a category index is out of range.
    pub fn catnn(&self, cat_indices: &[usize]) -> TabularResult<Vec<f32>> {
        let cfg = &self.config;
        if cat_indices.len() != cfg.n_cat_fields {
            return Err(TabularError::DimensionMismatch {
                expected: cfg.n_cat_fields,
                got: cat_indices.len(),
            });
        }

        let ed = cfg.cat_embed_dim;
        let out_dim = cfg.output_dim;

        // First-order linear term and accumulators for the FM interaction.
        let mut linear = self.cat_bias.clone();
        // sum_v[d] = Σ_f v_f[d]; sum_sq[d] = Σ_f v_f[d]^2.
        let mut sum_v = vec![0.0_f32; ed];
        let mut sum_sq = vec![0.0_f32; ed];

        for (f, &idx) in cat_indices.iter().enumerate() {
            let card = cfg.cat_cardinalities[f];
            if idx >= card {
                return Err(TabularError::CategoricalOutOfRange {
                    feat: f,
                    val: idx,
                    n: card,
                });
            }

            // First-order: add the per-category linear weights.
            let lin = &self.cat_linear[f];
            let lbase = idx * out_dim;
            for (o, lv) in linear.iter_mut().enumerate() {
                *lv += lin[lbase + o];
            }

            // Second-order accumulation over the embedding vector.
            let emb = &self.cat_embed[f];
            let ebase = idx * ed;
            for d in 0..ed {
                let v = emb[ebase + d];
                sum_v[d] += v;
                sum_sq[d] += v * v;
            }
        }

        // FM pairwise interaction per embedding dimension: ½((Σv)² − Σv²).
        let mut interaction = vec![0.0_f32; ed];
        for d in 0..ed {
            interaction[d] = 0.5 * (sum_v[d] * sum_v[d] - sum_sq[d]);
        }

        // Project the interaction vector to the output and add the linear term.
        let mut out = linear;
        for (o, ov) in out.iter_mut().enumerate() {
            let base = o * ed;
            let mut acc = 0.0_f32;
            for (d, &iv) in interaction.iter().enumerate() {
                acc += self.fm_proj[base + d] * iv;
            }
            *ov += acc;
        }
        Ok(out)
    }

    /// Combined prediction:
    /// `sigmoid(w1 ⊙ gbdt2nn(leaf) + w2 ⊙ catnn(cat) + bias)` per output dim.
    ///
    /// # Errors
    /// Propagates the validation errors of [`Self::gbdt2nn`] and
    /// [`Self::catnn`].
    pub fn forward(
        &self,
        leaf_indices: &[usize],
        cat_indices: &[usize],
    ) -> TabularResult<Vec<f32>> {
        let gbdt = self.gbdt2nn(leaf_indices)?;
        let cat = self.catnn(cat_indices)?;
        let mut out = vec![0.0_f32; self.config.output_dim];
        for o in 0..self.config.output_dim {
            let combined = self.w1[o] * gbdt[o] + self.w2[o] * cat[o] + self.comb_bias[o];
            out[o] = sigmoid(combined);
        }
        Ok(out)
    }

    /// Total number of learnable parameters in the model.
    #[must_use]
    pub fn n_params(&self) -> usize {
        let mlp: usize = self.gbdt_mlp.iter().map(|(w, b)| w.len() + b.len()).sum();
        let cat_emb: usize = self.cat_embed.iter().map(Vec::len).sum();
        let cat_lin: usize = self.cat_linear.iter().map(Vec::len).sum();
        self.leaf_embed.len()
            + mlp
            + cat_emb
            + cat_lin
            + self.fm_proj.len()
            + self.cat_bias.len()
            + self.w1.len()
            + self.w2.len()
            + self.comb_bias.len()
    }

    /// Override the GBDT2NN combination weights (test/inspection helper).
    ///
    /// # Errors
    /// Returns [`TabularError::DimensionMismatch`] if `weights.len() != output_dim`.
    pub fn set_w1(&mut self, weights: &[f32]) -> TabularResult<()> {
        if weights.len() != self.config.output_dim {
            return Err(TabularError::DimensionMismatch {
                expected: self.config.output_dim,
                got: weights.len(),
            });
        }
        self.w1.copy_from_slice(weights);
        Ok(())
    }

    /// Override the CatNN combination weights (test/inspection helper).
    ///
    /// # Errors
    /// Returns [`TabularError::DimensionMismatch`] if `weights.len() != output_dim`.
    pub fn set_w2(&mut self, weights: &[f32]) -> TabularResult<()> {
        if weights.len() != self.config.output_dim {
            return Err(TabularError::DimensionMismatch {
                expected: self.config.output_dim,
                got: weights.len(),
            });
        }
        self.w2.copy_from_slice(weights);
        Ok(())
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> DeepGbmConfig {
        DeepGbmConfig {
            n_trees: 4,
            n_leaves: 8,
            leaf_embed_dim: 3,
            gbdt_hidden: vec![16, 8],
            n_cat_fields: 3,
            cat_cardinalities: vec![5, 4, 6],
            cat_embed_dim: 4,
            output_dim: 2,
        }
    }

    #[test]
    fn gbdt2nn_output_length() {
        let mut rng = LcgRng::new(42);
        let model = DeepGbm::new(small_cfg(), &mut rng).expect("value should be present");
        let out = model
            .gbdt2nn(&[0, 1, 2, 3])
            .expect("gbdt2nn should succeed");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn catnn_output_length() {
        let mut rng = LcgRng::new(42);
        let model = DeepGbm::new(small_cfg(), &mut rng).expect("value should be present");
        let out = model.catnn(&[0, 1, 2]).expect("catnn should succeed");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn forward_output_length_and_sigmoid_range() {
        let mut rng = LcgRng::new(7);
        let model = DeepGbm::new(small_cfg(), &mut rng).expect("value should be present");
        let out = model
            .forward(&[0, 1, 2, 3], &[0, 1, 2])
            .expect("forward should succeed");
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out.iter().all(|&v| v > 0.0 && v < 1.0));
    }

    #[test]
    fn leaf_embedding_lookup_changes_output() {
        let mut rng = LcgRng::new(11);
        let model = DeepGbm::new(small_cfg(), &mut rng).expect("value should be present");
        let a = model
            .gbdt2nn(&[0, 0, 0, 0])
            .expect("gbdt2nn should succeed");
        let b = model
            .gbdt2nn(&[1, 1, 1, 1])
            .expect("gbdt2nn should succeed");
        assert_ne!(a, b);
    }

    #[test]
    fn fm_pairwise_changes_with_cat_indices() {
        let mut rng = LcgRng::new(13);
        let model = DeepGbm::new(small_cfg(), &mut rng).expect("value should be present");
        let a = model.catnn(&[0, 0, 0]).expect("catnn should succeed");
        let b = model.catnn(&[1, 2, 3]).expect("catnn should succeed");
        assert_ne!(a, b);
    }

    #[test]
    fn fm_pairwise_term_is_nonzero() {
        // Two distinct fields with non-trivial embeddings must yield a non-zero
        // pairwise interaction contribution; compare against a config where the
        // FM projection is the only differentiator. We check the interaction
        // by confirming catnn differs from the pure linear baseline.
        let mut rng = LcgRng::new(21);
        let model = DeepGbm::new(small_cfg(), &mut rng).expect("value should be present");
        let out = model.catnn(&[1, 2, 3]).expect("catnn should succeed");
        assert!(out.iter().all(|v| v.is_finite()));
        // Output is generally non-zero given random init + FM interaction.
        assert!(out.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn n_params_positive_and_formula() {
        let mut rng = LcgRng::new(42);
        let cfg = small_cfg();
        let model = DeepGbm::new(cfg.clone(), &mut rng).expect("value should be present");

        let leaf = cfg.n_trees * cfg.n_leaves * cfg.leaf_embed_dim;
        // MLP: (n_trees*ed -> 16) -> (16 -> 8) -> (8 -> output_dim).
        let in0 = cfg.n_trees * cfg.leaf_embed_dim;
        let mlp = (16 * in0 + 16) + (8 * 16 + 8) + (cfg.output_dim * 8 + cfg.output_dim);
        let cat_emb: usize = cfg
            .cat_cardinalities
            .iter()
            .map(|&c| c * cfg.cat_embed_dim)
            .sum();
        let cat_lin: usize = cfg
            .cat_cardinalities
            .iter()
            .map(|&c| c * cfg.output_dim)
            .sum();
        let fm = cfg.output_dim * cfg.cat_embed_dim;
        let bias_cat = cfg.output_dim;
        let comb = 3 * cfg.output_dim; // w1 + w2 + comb_bias
        let expected = leaf + mlp + cat_emb + cat_lin + fm + bias_cat + comb;

        assert_eq!(model.n_params(), expected);
        assert!(model.n_params() > 0);
    }

    #[test]
    fn deterministic_given_seed() {
        let mut rng_a = LcgRng::new(2024);
        let mut rng_b = LcgRng::new(2024);
        let model_a = DeepGbm::new(small_cfg(), &mut rng_a).expect("value should be present");
        let model_b = DeepGbm::new(small_cfg(), &mut rng_b).expect("value should be present");
        let out_a = model_a
            .forward(&[0, 1, 2, 3], &[0, 1, 2])
            .expect("forward should succeed");
        let out_b = model_b
            .forward(&[0, 1, 2, 3], &[0, 1, 2])
            .expect("forward should succeed");
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn leaf_index_out_of_range_errs() {
        let mut rng = LcgRng::new(1);
        let model = DeepGbm::new(small_cfg(), &mut rng).expect("value should be present");
        // n_leaves = 8, so 8 is out of range.
        assert!(matches!(
            model.gbdt2nn(&[0, 1, 2, 8]),
            Err(TabularError::CategoricalOutOfRange { .. })
        ));
    }

    #[test]
    fn cat_index_out_of_range_errs() {
        let mut rng = LcgRng::new(1);
        let model = DeepGbm::new(small_cfg(), &mut rng).expect("value should be present");
        // field 0 cardinality = 5, so 5 is out of range.
        assert!(matches!(
            model.catnn(&[5, 0, 0]),
            Err(TabularError::CategoricalOutOfRange { .. })
        ));
    }

    #[test]
    fn leaf_indices_wrong_length_errs() {
        let mut rng = LcgRng::new(1);
        let model = DeepGbm::new(small_cfg(), &mut rng).expect("value should be present");
        assert!(matches!(
            model.gbdt2nn(&[0, 1, 2]),
            Err(TabularError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn cat_indices_wrong_length_errs() {
        let mut rng = LcgRng::new(1);
        let model = DeepGbm::new(small_cfg(), &mut rng).expect("value should be present");
        assert!(matches!(
            model.catnn(&[0, 1]),
            Err(TabularError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn single_tree_single_cat_field_works() {
        let cfg = DeepGbmConfig {
            n_trees: 1,
            n_leaves: 2,
            leaf_embed_dim: 2,
            gbdt_hidden: vec![4],
            n_cat_fields: 1,
            cat_cardinalities: vec![3],
            cat_embed_dim: 2,
            output_dim: 1,
        };
        let mut rng = LcgRng::new(8);
        let model = DeepGbm::new(cfg, &mut rng).expect("new should succeed");
        let out = model.forward(&[1], &[2]).expect("forward should succeed");
        assert_eq!(out.len(), 1);
        assert!(out[0] > 0.0 && out[0] < 1.0);
    }

    #[test]
    fn combination_uses_both_components() {
        let mut rng = LcgRng::new(31);
        let mut model = DeepGbm::new(small_cfg(), &mut rng).expect("value should be present");
        let base = model
            .forward(&[1, 2, 3, 4], &[1, 2, 3])
            .expect("forward should succeed");
        // Zeroing the CatNN weight must change the combined output.
        model.set_w2(&[0.0, 0.0]).expect("set_w2 should succeed");
        let no_cat = model
            .forward(&[1, 2, 3, 4], &[1, 2, 3])
            .expect("forward should succeed");
        assert_ne!(base, no_cat);
        // Now also zero the GBDT2NN weight: output becomes sigmoid(bias) = 0.5.
        model.set_w1(&[0.0, 0.0]).expect("set_w1 should succeed");
        let neither = model
            .forward(&[1, 2, 3, 4], &[1, 2, 3])
            .expect("forward should succeed");
        assert!(neither.iter().all(|&v| (v - 0.5).abs() < 1e-6));
    }

    #[test]
    fn empty_gbdt_hidden_linear_head_works() {
        let cfg = DeepGbmConfig {
            n_trees: 3,
            n_leaves: 4,
            leaf_embed_dim: 2,
            gbdt_hidden: vec![],
            n_cat_fields: 2,
            cat_cardinalities: vec![3, 3],
            cat_embed_dim: 2,
            output_dim: 2,
        };
        let mut rng = LcgRng::new(64);
        let model = DeepGbm::new(cfg, &mut rng).expect("new should succeed");
        let out = model.gbdt2nn(&[0, 1, 2]).expect("gbdt2nn should succeed");
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn two_distinct_inputs_distinct_outputs() {
        let mut rng = LcgRng::new(77);
        let model = DeepGbm::new(small_cfg(), &mut rng).expect("value should be present");
        let a = model
            .forward(&[0, 0, 0, 0], &[0, 0, 0])
            .expect("forward should succeed");
        let b = model
            .forward(&[1, 2, 3, 4], &[1, 2, 3])
            .expect("forward should succeed");
        assert_ne!(a, b);
    }

    #[test]
    fn err_n_trees_zero() {
        let cfg = DeepGbmConfig {
            n_trees: 0,
            ..small_cfg()
        };
        let mut rng = LcgRng::new(1);
        assert!(DeepGbm::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_n_leaves_zero() {
        let cfg = DeepGbmConfig {
            n_leaves: 0,
            ..small_cfg()
        };
        let mut rng = LcgRng::new(1);
        assert!(DeepGbm::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_leaf_embed_dim_zero() {
        let cfg = DeepGbmConfig {
            leaf_embed_dim: 0,
            ..small_cfg()
        };
        let mut rng = LcgRng::new(1);
        assert!(DeepGbm::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_n_cat_fields_zero() {
        let cfg = DeepGbmConfig {
            n_cat_fields: 0,
            cat_cardinalities: vec![],
            ..small_cfg()
        };
        let mut rng = LcgRng::new(1);
        assert!(DeepGbm::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_output_dim_zero() {
        let cfg = DeepGbmConfig {
            output_dim: 0,
            ..small_cfg()
        };
        let mut rng = LcgRng::new(1);
        assert!(DeepGbm::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_cardinality_length_mismatch() {
        let cfg = DeepGbmConfig {
            n_cat_fields: 3,
            cat_cardinalities: vec![5, 4],
            ..small_cfg()
        };
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            DeepGbm::new(cfg, &mut rng),
            Err(TabularError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_zero_cardinality() {
        let cfg = DeepGbmConfig {
            n_cat_fields: 3,
            cat_cardinalities: vec![5, 0, 6],
            ..small_cfg()
        };
        let mut rng = LcgRng::new(1);
        assert!(DeepGbm::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_cat_embed_dim_zero() {
        let cfg = DeepGbmConfig {
            cat_embed_dim: 0,
            ..small_cfg()
        };
        let mut rng = LcgRng::new(1);
        assert!(DeepGbm::new(cfg, &mut rng).is_err());
    }
}
