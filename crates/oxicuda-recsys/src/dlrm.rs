//! DLRM — Deep Learning Recommendation Model (Naumov et al. 2019).
//!
//! Reference: Maxim Naumov et al., "Deep Learning Recommendation Model for
//! Personalization and Recommendation Systems", arXiv:1906.00091 (2019).
//!
//! Architecture:
//!   1. Dense (continuous) features → **bottom MLP** → a dense embedding of
//!      length `embed_dim`, matching the categorical embedding width.
//!   2. Each categorical (sparse) field has its own embedding table; a single
//!      index per field gathers one row of length `embed_dim`.
//!   3. **Interaction**: form the feature set `V = [dense_emb, cat_emb_1, …]`
//!      and compute the upper-triangular pairwise dot products. The interaction
//!      vector is `concat(dense_emb, those dot products)`.
//!   4. **Top MLP** maps the interaction vector to a single logit; a sigmoid
//!      turns it into a click-through-rate probability in `(0, 1)`.

use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

/// Dense layer: `out[o] = b[o] + Σ_i w[o*fan_in + i] · x[i]`.
fn dense(x: &[f32], w: &[f32], b: &[f32], fan_in: usize, fan_out: usize) -> Vec<f32> {
    (0..fan_out)
        .map(|o| {
            b[o] + w[o * fan_in..(o + 1) * fan_in]
                .iter()
                .zip(x.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f32>()
        })
        .collect()
}

/// In-place ReLU.
fn relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Numerically stable logistic sigmoid.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// DLRM hyper-parameters.
#[derive(Debug, Clone)]
pub struct DlrmConfig {
    /// Number of dense (continuous) input features.
    pub dense_dim: usize,
    /// Shared embedding width for dense and categorical embeddings.
    pub embed_dim: usize,
    /// Cardinality of each categorical field (one entry per field).
    pub cat_cardinalities: Vec<usize>,
    /// Hidden widths of the bottom MLP (final layer always maps to `embed_dim`).
    pub bottom_mlp: Vec<usize>,
    /// Hidden widths of the top MLP (final layer always maps to a single logit).
    pub top_mlp: Vec<usize>,
}

/// Deep Learning Recommendation Model.
pub struct Dlrm {
    /// Configuration the model was built from.
    pub cfg: DlrmConfig,
    /// One embedding table per categorical field: `cardinality_f × embed_dim`.
    pub embeddings: Vec<Vec<f32>>,
    /// Bottom MLP weight/bias pairs (`dense_dim → … → embed_dim`).
    pub bottom_layers: Vec<(Vec<f32>, Vec<f32>)>,
    /// Top MLP weight/bias pairs (`interaction_dim → … → 1`).
    pub top_layers: Vec<(Vec<f32>, Vec<f32>)>,
    /// Cached interaction-vector length (`embed_dim + (k+1)·k/2`).
    pub interaction_dim: usize,
}

impl Dlrm {
    /// Construct a DLRM with Kaiming-style normal initialisation.
    ///
    /// # Errors
    /// Returns [`RecsysError::InvalidConfig`] / [`RecsysError::InvalidEmbeddingDim`]
    /// when the configuration is degenerate (zero dense dim, zero embed dim,
    /// empty categorical fields, or a zero-cardinality field).
    pub fn new(cfg: DlrmConfig, rng: &mut LcgRng) -> RecsysResult<Self> {
        if cfg.dense_dim == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "dense_dim must be >= 1".into(),
            });
        }
        if cfg.embed_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: 0 });
        }
        if cfg.cat_cardinalities.is_empty() {
            return Err(RecsysError::InvalidConfig {
                msg: "cat_cardinalities must be non-empty".into(),
            });
        }
        for (f, &card) in cfg.cat_cardinalities.iter().enumerate() {
            if card == 0 {
                return Err(RecsysError::InvalidConfig {
                    msg: format!("cat field {f}: cardinality must be >= 1"),
                });
            }
        }

        let d = cfg.embed_dim;
        let scale = (1.0 / d as f32).sqrt();

        // Per-field embedding tables.
        let embeddings: Vec<Vec<f32>> = cfg
            .cat_cardinalities
            .iter()
            .map(|&card| (0..card * d).map(|_| rng.next_normal() * scale).collect())
            .collect();

        // Bottom MLP: dense_dim → bottom_mlp widths (ReLU) → embed_dim.
        let mut bottom_dims = vec![cfg.dense_dim];
        bottom_dims.extend_from_slice(&cfg.bottom_mlp);
        bottom_dims.push(d);
        let bottom_layers = build_mlp(&bottom_dims, rng);

        // Interaction set V = {dense_emb, cat_emb_1, …, cat_emb_k}: k+1 vectors.
        let n_vectors = cfg.cat_cardinalities.len() + 1;
        let n_pairs = n_vectors * (n_vectors - 1) / 2;
        let interaction_dim = d + n_pairs;

        // Top MLP: interaction_dim → top_mlp widths (ReLU) → 1.
        let mut top_dims = vec![interaction_dim];
        top_dims.extend_from_slice(&cfg.top_mlp);
        top_dims.push(1);
        let top_layers = build_mlp(&top_dims, rng);

        Ok(Self {
            cfg,
            embeddings,
            bottom_layers,
            top_layers,
            interaction_dim,
        })
    }

    /// Run the bottom MLP on the dense feature vector, producing the dense
    /// embedding of length `embed_dim`.
    ///
    /// # Errors
    /// Returns [`RecsysError::DimensionMismatch`] if `dense.len() != dense_dim`.
    pub fn bottom_forward(&self, dense_in: &[f32]) -> RecsysResult<Vec<f32>> {
        if dense_in.len() != self.cfg.dense_dim {
            return Err(RecsysError::DimensionMismatch {
                expected: self.cfg.dense_dim,
                got: dense_in.len(),
            });
        }
        Ok(mlp_forward(dense_in, &self.bottom_layers))
    }

    /// Gather the categorical embedding rows for the supplied indices.
    ///
    /// # Errors
    /// Returns [`RecsysError::DimensionMismatch`] if the index count differs
    /// from the number of categorical fields, or [`RecsysError::ItemOutOfBounds`]
    /// if any index exceeds its field's cardinality.
    pub fn gather_cat(&self, cat_indices: &[usize]) -> RecsysResult<Vec<Vec<f32>>> {
        if cat_indices.len() != self.cfg.cat_cardinalities.len() {
            return Err(RecsysError::DimensionMismatch {
                expected: self.cfg.cat_cardinalities.len(),
                got: cat_indices.len(),
            });
        }
        let d = self.cfg.embed_dim;
        let mut out = Vec::with_capacity(cat_indices.len());
        for (f, &idx) in cat_indices.iter().enumerate() {
            let card = self.cfg.cat_cardinalities[f];
            if idx >= card {
                return Err(RecsysError::ItemOutOfBounds { idx, n: card });
            }
            out.push(self.embeddings[f][idx * d..(idx + 1) * d].to_vec());
        }
        Ok(out)
    }

    /// DLRM feature interaction.
    ///
    /// Given the dense embedding (`embed_dim`) and the categorical embeddings,
    /// the feature set is `V = [dense_emb, cat_emb_1, …, cat_emb_k]` (`k+1`
    /// vectors of length `embed_dim`). Returns
    /// `concat(dense_emb, upper-triangular pairwise dot products of V)`.
    /// The output length is `embed_dim + (k+1)·k/2`.
    ///
    /// # Errors
    /// Returns [`RecsysError::DimensionMismatch`] if `dense_emb` or any
    /// categorical embedding has a length other than `embed_dim`.
    pub fn interact(&self, dense_emb: &[f32], cat_embs: &[Vec<f32>]) -> RecsysResult<Vec<f32>> {
        let d = self.cfg.embed_dim;
        if dense_emb.len() != d {
            return Err(RecsysError::DimensionMismatch {
                expected: d,
                got: dense_emb.len(),
            });
        }
        for emb in cat_embs {
            if emb.len() != d {
                return Err(RecsysError::DimensionMismatch {
                    expected: d,
                    got: emb.len(),
                });
            }
        }

        // Assemble the full feature set V = [dense_emb, cat_emb_1, …].
        let mut vectors: Vec<&[f32]> = Vec::with_capacity(cat_embs.len() + 1);
        vectors.push(dense_emb);
        for emb in cat_embs {
            vectors.push(emb.as_slice());
        }

        let n = vectors.len();
        let n_pairs = n * (n - 1) / 2;
        let mut out = Vec::with_capacity(d + n_pairs);
        // Concatenate the dense embedding first.
        out.extend_from_slice(dense_emb);
        // Upper-triangular pairwise dot products (i < j).
        for i in 0..n {
            for j in (i + 1)..n {
                let dot: f32 = vectors[i]
                    .iter()
                    .zip(vectors[j].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                out.push(dot);
            }
        }
        Ok(out)
    }

    /// Full forward pass: bottom MLP → embedding gather → interaction → top MLP
    /// → sigmoid. Returns a click-through-rate probability in `(0, 1)`.
    ///
    /// `dense` holds `dense_dim` continuous values; `cat_indices` holds one
    /// index per categorical field.
    ///
    /// # Errors
    /// Propagates validation errors from [`Self::bottom_forward`],
    /// [`Self::gather_cat`], and [`Self::interact`].
    pub fn forward(&self, dense_in: &[f32], cat_indices: &[usize]) -> RecsysResult<f32> {
        let dense_emb = self.bottom_forward(dense_in)?;
        let cat_embs = self.gather_cat(cat_indices)?;
        let interaction = self.interact(&dense_emb, &cat_embs)?;
        let logit_vec = mlp_forward(&interaction, &self.top_layers);
        let logit = logit_vec.first().copied().unwrap_or(0.0);
        Ok(sigmoid(logit))
    }

    /// Total number of learnable parameters (embeddings + both MLPs).
    #[must_use]
    pub fn n_params(&self) -> usize {
        let emb: usize = self.embeddings.iter().map(Vec::len).sum();
        let bottom: usize = self
            .bottom_layers
            .iter()
            .map(|(w, b)| w.len() + b.len())
            .sum();
        let top: usize = self.top_layers.iter().map(|(w, b)| w.len() + b.len()).sum();
        emb + bottom + top
    }
}

/// Build a stack of `(weight, bias)` layers from consecutive dimensions.
///
/// `dims = [in, h_1, …, out]`; each layer `i` maps `dims[i] → dims[i+1]`.
fn build_mlp(dims: &[usize], rng: &mut LcgRng) -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut layers = Vec::with_capacity(dims.len().saturating_sub(1));
    for window in dims.windows(2) {
        let (fan_in, fan_out) = (window[0], window[1]);
        let sc = (2.0 / fan_in.max(1) as f32).sqrt();
        let w: Vec<f32> = (0..fan_out * fan_in)
            .map(|_| rng.next_normal() * sc)
            .collect();
        let b = vec![0.0_f32; fan_out];
        layers.push((w, b));
    }
    layers
}

/// Forward an input through an MLP with ReLU on every layer **except** the last.
fn mlp_forward(x: &[f32], layers: &[(Vec<f32>, Vec<f32>)]) -> Vec<f32> {
    let mut current = x.to_vec();
    let mut cur_dim = x.len();
    let n = layers.len();
    for (idx, (w, b)) in layers.iter().enumerate() {
        let out_dim = b.len();
        let mut out = dense(&current, w, b, cur_dim, out_dim);
        if idx + 1 < n {
            relu(&mut out);
        }
        current = out;
        cur_dim = out_dim;
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn default_cfg() -> DlrmConfig {
        DlrmConfig {
            dense_dim: 6,
            embed_dim: 8,
            cat_cardinalities: vec![10, 20, 5],
            bottom_mlp: vec![16, 12],
            top_mlp: vec![32, 16],
        }
    }

    #[test]
    fn forward_in_open_unit_interval() {
        let mut rng = make_rng();
        let model = Dlrm::new(default_cfg(), &mut rng).expect("value should be present");
        let dense_in: Vec<f32> = (0..6).map(|_| rng.next_normal()).collect();
        let cat = vec![1usize, 3, 2];
        let p = model
            .forward(&dense_in, &cat)
            .expect("forward should succeed");
        assert!(p.is_finite(), "probability must be finite, got {p}");
        assert!(p > 0.0 && p < 1.0, "probability {p} not in (0,1)");
    }

    #[test]
    fn interact_length_k2() {
        let mut rng = make_rng();
        let cfg = DlrmConfig {
            dense_dim: 4,
            embed_dim: 8,
            cat_cardinalities: vec![5, 7],
            bottom_mlp: vec![],
            top_mlp: vec![],
        };
        let model = Dlrm::new(cfg, &mut rng).expect("new should succeed");
        let dense_emb = vec![0.5_f32; 8];
        let cat_embs = vec![vec![0.1_f32; 8], vec![0.2_f32; 8]];
        let out = model
            .interact(&dense_emb, &cat_embs)
            .expect("interact should succeed");
        // k = 2 → embed_dim + (k+1)*k/2 = 8 + 3 = 11.
        assert_eq!(out.len(), 8 + 3);
    }

    #[test]
    fn interact_length_k3() {
        let mut rng = make_rng();
        let cfg = DlrmConfig {
            dense_dim: 4,
            embed_dim: 8,
            cat_cardinalities: vec![5, 7, 3, 4],
            bottom_mlp: vec![],
            top_mlp: vec![],
        };
        let model = Dlrm::new(cfg, &mut rng).expect("new should succeed");
        let dense_emb = vec![0.5_f32; 8];
        let cat_embs = vec![vec![0.1_f32; 8]; 4];
        let out = model
            .interact(&dense_emb, &cat_embs)
            .expect("interact should succeed");
        // k = 4 → 5 vectors → C(5,2) = 10 pairs → 8 + 10 = 18.
        assert_eq!(out.len(), 8 + 10);
    }

    #[test]
    fn interact_single_cat_field() {
        let mut rng = make_rng();
        let cfg = DlrmConfig {
            dense_dim: 4,
            embed_dim: 8,
            cat_cardinalities: vec![5],
            bottom_mlp: vec![],
            top_mlp: vec![],
        };
        let model = Dlrm::new(cfg, &mut rng).expect("new should succeed");
        let dense_emb = vec![0.5_f32; 8];
        let cat_embs = vec![vec![0.1_f32; 8]];
        let out = model
            .interact(&dense_emb, &cat_embs)
            .expect("interact should succeed");
        // k = 1 → 2 vectors → 1 pair → embed_dim + 1 = 9.
        assert_eq!(out.len(), 8 + 1);
    }

    #[test]
    fn interact_pair_count_matches_upper_triangle() {
        let mut rng = make_rng();
        let model = Dlrm::new(default_cfg(), &mut rng).expect("value should be present");
        let dense_emb = vec![0.3_f32; 8];
        let cat_embs = vec![vec![0.1_f32; 8]; 3];
        let out = model
            .interact(&dense_emb, &cat_embs)
            .expect("interact should succeed");
        // 4 vectors → C(4,2) = 6 unique pairs (i<j only), not 4*4 = 16.
        let n_pairs = out.len() - 8;
        assert_eq!(n_pairs, 6);
        assert_eq!(model.interaction_dim, out.len());
    }

    #[test]
    fn interact_dot_products_are_symmetric_values() {
        // Verify the dot products correspond to the symmetric Gram entries i<j.
        let mut rng = make_rng();
        let cfg = DlrmConfig {
            dense_dim: 4,
            embed_dim: 3,
            cat_cardinalities: vec![5, 7],
            bottom_mlp: vec![],
            top_mlp: vec![],
        };
        let model = Dlrm::new(cfg, &mut rng).expect("new should succeed");
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![0.5_f32, -1.0, 2.0];
        let c = vec![-2.0_f32, 0.0, 1.0];
        let out = model
            .interact(&a, &[b.clone(), c.clone()])
            .expect("value should be present");
        // Pairs in order: (a,b), (a,c), (b,c).
        let ab: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let ac: f32 = a.iter().zip(c.iter()).map(|(x, y)| x * y).sum();
        let bc: f32 = b.iter().zip(c.iter()).map(|(x, y)| x * y).sum();
        assert!((out[3] - ab).abs() < 1e-5);
        assert!((out[4] - ac).abs() < 1e-5);
        assert!((out[5] - bc).abs() < 1e-5);
    }

    #[test]
    fn embedding_table_sizes_correct() {
        let mut rng = make_rng();
        let model = Dlrm::new(default_cfg(), &mut rng).expect("value should be present");
        assert_eq!(model.embeddings.len(), 3);
        assert_eq!(model.embeddings[0].len(), 10 * 8);
        assert_eq!(model.embeddings[1].len(), 20 * 8);
        assert_eq!(model.embeddings[2].len(), 5 * 8);
    }

    #[test]
    fn bottom_mlp_output_dim_is_embed_dim() {
        let mut rng = make_rng();
        let model = Dlrm::new(default_cfg(), &mut rng).expect("value should be present");
        let dense_in = vec![0.1_f32; 6];
        let emb = model
            .bottom_forward(&dense_in)
            .expect("bottom_forward should succeed");
        assert_eq!(emb.len(), 8);
    }

    #[test]
    fn bottom_mlp_empty_maps_dense_to_embed() {
        let mut rng = make_rng();
        let cfg = DlrmConfig {
            dense_dim: 5,
            embed_dim: 8,
            cat_cardinalities: vec![4],
            bottom_mlp: vec![],
            top_mlp: vec![],
        };
        let model = Dlrm::new(cfg, &mut rng).expect("new should succeed");
        let dense_in = vec![0.2_f32; 5];
        let emb = model
            .bottom_forward(&dense_in)
            .expect("bottom_forward should succeed");
        // Single linear layer dense_dim(5) → embed_dim(8).
        assert_eq!(emb.len(), 8);
        assert_eq!(model.bottom_layers.len(), 1);
    }

    #[test]
    fn n_params_positive_and_sane() {
        let mut rng = make_rng();
        let model = Dlrm::new(default_cfg(), &mut rng).expect("value should be present");
        let np = model.n_params();
        assert!(np > 0, "n_params must be > 0");
        // Embedding params alone are (10+20+5)*8 = 280.
        let emb_params = (10 + 20 + 5) * 8;
        assert!(np > emb_params, "total params must exceed embedding params");
    }

    #[test]
    fn deterministic_given_seed() {
        let mut rng_a = LcgRng::new(7);
        let mut rng_b = LcgRng::new(7);
        let model_a = Dlrm::new(default_cfg(), &mut rng_a).expect("value should be present");
        let model_b = Dlrm::new(default_cfg(), &mut rng_b).expect("value should be present");
        let dense_in = vec![0.3_f32; 6];
        let cat = vec![2usize, 5, 1];
        let pa = model_a
            .forward(&dense_in, &cat)
            .expect("forward should succeed");
        let pb = model_b
            .forward(&dense_in, &cat)
            .expect("forward should succeed");
        assert!((pa - pb).abs() < 1e-6, "same seed must give same output");
    }

    #[test]
    fn cat_index_out_of_range_errors() {
        let mut rng = make_rng();
        let model = Dlrm::new(default_cfg(), &mut rng).expect("value should be present");
        let dense_in = vec![0.1_f32; 6];
        // Field 0 cardinality is 10; index 10 is out of range.
        let cat = vec![10usize, 3, 2];
        let res = model.forward(&dense_in, &cat);
        assert!(matches!(res, Err(RecsysError::ItemOutOfBounds { .. })));
    }

    #[test]
    fn cat_indices_wrong_length_errors() {
        let mut rng = make_rng();
        let model = Dlrm::new(default_cfg(), &mut rng).expect("value should be present");
        let dense_in = vec![0.1_f32; 6];
        let cat = vec![1usize, 3]; // only 2 of 3 fields
        let res = model.forward(&dense_in, &cat);
        assert!(matches!(res, Err(RecsysError::DimensionMismatch { .. })));
    }

    #[test]
    fn dense_wrong_length_errors() {
        let mut rng = make_rng();
        let model = Dlrm::new(default_cfg(), &mut rng).expect("value should be present");
        let dense_in = vec![0.1_f32; 5]; // expected 6
        let cat = vec![1usize, 3, 2];
        let res = model.forward(&dense_in, &cat);
        assert!(matches!(res, Err(RecsysError::DimensionMismatch { .. })));
    }

    #[test]
    fn changing_cat_index_changes_output() {
        let mut rng = make_rng();
        let model = Dlrm::new(default_cfg(), &mut rng).expect("value should be present");
        let dense_in = vec![0.3_f32; 6];
        let p1 = model
            .forward(&dense_in, &[1usize, 3, 2])
            .expect("forward should succeed");
        let p2 = model
            .forward(&dense_in, &[4usize, 3, 2])
            .expect("forward should succeed");
        assert!(
            (p1 - p2).abs() > 1e-9,
            "changing a cat index must move output"
        );
    }

    #[test]
    fn changing_dense_changes_output() {
        let mut rng = make_rng();
        let model = Dlrm::new(default_cfg(), &mut rng).expect("value should be present");
        let cat = vec![1usize, 3, 2];
        let d1 = vec![0.1_f32; 6];
        let d2: Vec<f32> = (0..6).map(|i| i as f32 * 0.5 + 0.7).collect();
        let p1 = model.forward(&d1, &cat).expect("forward should succeed");
        let p2 = model.forward(&d2, &cat).expect("forward should succeed");
        assert!((p1 - p2).abs() > 1e-9, "changing dense must move output");
    }

    #[test]
    fn two_distinct_inputs_give_distinct_outputs() {
        let mut rng = make_rng();
        let model = Dlrm::new(default_cfg(), &mut rng).expect("value should be present");
        let p1 = model
            .forward(&[0.1_f32; 6], &[0usize, 0, 0])
            .expect("forward should succeed");
        let p2 = model
            .forward(&[0.9_f32; 6], &[9usize, 19, 4])
            .expect("forward should succeed");
        assert!((p1 - p2).abs() > 1e-9, "distinct inputs must differ");
    }

    #[test]
    fn err_dense_dim_zero() {
        let mut rng = make_rng();
        let cfg = DlrmConfig {
            dense_dim: 0,
            embed_dim: 8,
            cat_cardinalities: vec![5],
            bottom_mlp: vec![],
            top_mlp: vec![],
        };
        assert!(matches!(
            Dlrm::new(cfg, &mut rng),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn err_embed_dim_zero() {
        let mut rng = make_rng();
        let cfg = DlrmConfig {
            dense_dim: 4,
            embed_dim: 0,
            cat_cardinalities: vec![5],
            bottom_mlp: vec![],
            top_mlp: vec![],
        };
        assert!(matches!(
            Dlrm::new(cfg, &mut rng),
            Err(RecsysError::InvalidEmbeddingDim { .. })
        ));
    }

    #[test]
    fn err_empty_cat_cardinalities() {
        let mut rng = make_rng();
        let cfg = DlrmConfig {
            dense_dim: 4,
            embed_dim: 8,
            cat_cardinalities: vec![],
            bottom_mlp: vec![],
            top_mlp: vec![],
        };
        assert!(matches!(
            Dlrm::new(cfg, &mut rng),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn err_zero_cardinality_field() {
        let mut rng = make_rng();
        let cfg = DlrmConfig {
            dense_dim: 4,
            embed_dim: 8,
            cat_cardinalities: vec![5, 0, 3],
            bottom_mlp: vec![],
            top_mlp: vec![],
        };
        assert!(matches!(
            Dlrm::new(cfg, &mut rng),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn interact_wrong_dense_emb_length_errors() {
        let mut rng = make_rng();
        let model = Dlrm::new(default_cfg(), &mut rng).expect("value should be present");
        let dense_emb = vec![0.5_f32; 7]; // expected 8
        let cat_embs = vec![vec![0.1_f32; 8]; 3];
        assert!(matches!(
            model.interact(&dense_emb, &cat_embs),
            Err(RecsysError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn gather_cat_returns_correct_rows() {
        let mut rng = make_rng();
        let model = Dlrm::new(default_cfg(), &mut rng).expect("value should be present");
        let cat = vec![2usize, 7, 1];
        let rows = model.gather_cat(&cat).expect("gather_cat should succeed");
        assert_eq!(rows.len(), 3);
        for row in &rows {
            assert_eq!(row.len(), 8);
        }
        // Row 0 must equal embedding table 0's slice at index 2.
        let expected = &model.embeddings[0][2 * 8..3 * 8];
        assert_eq!(rows[0].as_slice(), expected);
    }
}
