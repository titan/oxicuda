//! FiBiNET — Feature Importance and Bilinear feature interaction NETwork.
//!
//! Reference: Tongwen Huang, Zhiqi Zhang, Junlin Zhang, "FiBiNET: Combining
//! Feature Importance and Bilinear feature Interaction for Click-Through Rate
//! Prediction", RecSys 2019.
//!
//! Architecture:
//!   1. **SENET** (Squeeze-and-Excitation): squeeze each field's embedding to a
//!      scalar (mean over `embed_dim`), excite through a two-layer MLP with a
//!      sigmoid gate `a_i ∈ [0,1]`, then reweight `field_i_out = a_i · emb_i`.
//!   2. **Bilinear interaction**: for each field pair `i < j`,
//!      `v_ij = p_i ∘ (W · p_j)` (Hadamard product with a matrix-vector
//!      product). The weight matrix is shared ([`BilinearType::FieldAll`]),
//!      per-field ([`BilinearType::FieldEach`]), or per-pair
//!      ([`BilinearType::FieldInteraction`]).
//!   3. The bilinear interactions of the **raw** field embeddings and of the
//!      **SENET-reweighted** embeddings are concatenated and fed through a DNN
//!      to a single logit; a sigmoid yields a CTR probability in `(0, 1)`.

use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

/// Numerically stable logistic sigmoid.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// In-place ReLU.
fn relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Matrix-vector product `W · v` where `W` is `embed_dim × embed_dim`
/// (row-major) and `v` has length `embed_dim`.
fn matvec(w: &[f32], v: &[f32], d: usize) -> Vec<f32> {
    (0..d)
        .map(|r| {
            w[r * d..(r + 1) * d]
                .iter()
                .zip(v.iter())
                .map(|(&wij, &vj)| wij * vj)
                .sum::<f32>()
        })
        .collect()
}

/// Bilinear weight-sharing strategy for the feature interaction layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BilinearType {
    /// One shared `W` matrix for every field pair.
    FieldAll,
    /// One `W` matrix per field `i` (used for all pairs `(i, j)`).
    FieldEach,
    /// One `W` matrix per field pair `(i, j)`.
    FieldInteraction,
}

/// FiBiNET hyper-parameters.
#[derive(Debug, Clone)]
pub struct FibinetConfig {
    /// Number of feature fields (`>= 2`).
    pub n_fields: usize,
    /// Embedding width of each field.
    pub embed_dim: usize,
    /// SENET reduction ratio (hidden width = `n_fields / reduction_ratio`,
    /// clamped to `>= 1`).
    pub reduction_ratio: usize,
    /// Bilinear weight-sharing strategy.
    pub bilinear_type: BilinearType,
    /// Hidden widths of the DNN (may be empty → a single linear layer to 1).
    pub dnn_hidden: Vec<usize>,
}

/// FiBiNET model.
pub struct Fibinet {
    /// Configuration the model was built from.
    pub cfg: FibinetConfig,
    /// SENET reduction-layer weights: `hidden × n_fields` (row-major).
    pub senet_w1: Vec<f32>,
    /// SENET reduction-layer biases: `hidden`.
    pub senet_b1: Vec<f32>,
    /// SENET excitation-layer weights: `n_fields × hidden` (row-major).
    pub senet_w2: Vec<f32>,
    /// SENET excitation-layer biases: `n_fields`.
    pub senet_b2: Vec<f32>,
    /// SENET hidden width (`>= 1`).
    pub senet_hidden: usize,
    /// Bilinear weight matrices, each `embed_dim × embed_dim`. Count depends on
    /// [`BilinearType`].
    pub bilinear_w: Vec<Vec<f32>>,
    /// DNN weight/bias pairs (final layer maps to a single logit).
    pub dnn_layers: Vec<(Vec<f32>, Vec<f32>)>,
    /// Length of one bilinear interaction vector: `C(n_fields, 2) · embed_dim`.
    pub bilinear_out_dim: usize,
}

impl Fibinet {
    /// Construct a FiBiNET with Kaiming-style normal initialisation.
    ///
    /// # Errors
    /// Returns [`RecsysError::InvalidConfig`] when `n_fields < 2` or
    /// `reduction_ratio == 0`, and [`RecsysError::InvalidEmbeddingDim`] when
    /// `embed_dim == 0`.
    pub fn new(cfg: FibinetConfig, rng: &mut LcgRng) -> RecsysResult<Self> {
        if cfg.n_fields < 2 {
            return Err(RecsysError::InvalidConfig {
                msg: format!("n_fields must be >= 2, got {}", cfg.n_fields),
            });
        }
        if cfg.embed_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: 0 });
        }
        if cfg.reduction_ratio == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "reduction_ratio must be >= 1".into(),
            });
        }

        let m = cfg.n_fields;
        let d = cfg.embed_dim;

        // SENET hidden width: n_fields / reduction_ratio, clamped to >= 1.
        let senet_hidden = (m / cfg.reduction_ratio).max(1);

        // SENET reduction layer: hidden × n_fields.
        let sc1 = (2.0 / m as f32).sqrt();
        let senet_w1: Vec<f32> = (0..senet_hidden * m)
            .map(|_| rng.next_normal() * sc1)
            .collect();
        let senet_b1 = vec![0.0_f32; senet_hidden];

        // SENET excitation layer: n_fields × hidden.
        let sc2 = (2.0 / senet_hidden as f32).sqrt();
        let senet_w2: Vec<f32> = (0..m * senet_hidden)
            .map(|_| rng.next_normal() * sc2)
            .collect();
        let senet_b2 = vec![0.0_f32; m];

        // Bilinear weight matrices.
        let n_pairs = m * (m - 1) / 2;
        let n_matrices = match cfg.bilinear_type {
            BilinearType::FieldAll => 1,
            BilinearType::FieldEach => m,
            BilinearType::FieldInteraction => n_pairs,
        };
        let bsc = (1.0 / d as f32).sqrt();
        let bilinear_w: Vec<Vec<f32>> = (0..n_matrices)
            .map(|_| (0..d * d).map(|_| rng.next_normal() * bsc).collect())
            .collect();

        // Each bilinear interaction concatenates n_pairs vectors of length d;
        // forward concatenates the raw and SENET interactions → 2× that.
        let bilinear_out_dim = n_pairs * d;
        let dnn_input_dim = 2 * bilinear_out_dim;

        // DNN: dnn_input_dim → dnn_hidden widths (ReLU) → 1 logit.
        let mut dnn_dims = vec![dnn_input_dim];
        dnn_dims.extend_from_slice(&cfg.dnn_hidden);
        dnn_dims.push(1);
        let dnn_layers = build_mlp(&dnn_dims, rng);

        Ok(Self {
            cfg,
            senet_w1,
            senet_b1,
            senet_w2,
            senet_b2,
            senet_hidden,
            bilinear_w,
            dnn_layers,
            bilinear_out_dim,
        })
    }

    /// Compute the SENET excitation gates `a_i ∈ [0, 1]` (one per field).
    ///
    /// # Errors
    /// Returns [`RecsysError::DimensionMismatch`] if `field_embs.len()` differs
    /// from `n_fields · embed_dim`.
    pub fn senet_gates(&self, field_embs: &[f32]) -> RecsysResult<Vec<f32>> {
        let m = self.cfg.n_fields;
        let d = self.cfg.embed_dim;
        if field_embs.len() != m * d {
            return Err(RecsysError::DimensionMismatch {
                expected: m * d,
                got: field_embs.len(),
            });
        }

        // Squeeze: mean over each field's embedding → z ∈ R^m.
        let inv_d = 1.0 / d as f32;
        let z: Vec<f32> = (0..m)
            .map(|f| field_embs[f * d..(f + 1) * d].iter().sum::<f32>() * inv_d)
            .collect();

        // Excitation reduction: hidden = ReLU(W1 z + b1).
        let mut hidden: Vec<f32> = (0..self.senet_hidden)
            .map(|h| {
                self.senet_b1[h]
                    + self.senet_w1[h * m..(h + 1) * m]
                        .iter()
                        .zip(z.iter())
                        .map(|(&w, &zi)| w * zi)
                        .sum::<f32>()
            })
            .collect();
        relu(&mut hidden);

        // Excitation output gate: a = sigmoid(W2 hidden + b2) ∈ [0,1]^m.
        let gates: Vec<f32> = (0..m)
            .map(|f| {
                let pre = self.senet_b2[f]
                    + self.senet_w2[f * self.senet_hidden..(f + 1) * self.senet_hidden]
                        .iter()
                        .zip(hidden.iter())
                        .map(|(&w, &hv)| w * hv)
                        .sum::<f32>();
                sigmoid(pre)
            })
            .collect();
        Ok(gates)
    }

    /// SENET-reweighted embeddings: `field_i_out = a_i · emb_i`. Same shape as
    /// the input (`n_fields × embed_dim`, row-major).
    ///
    /// # Errors
    /// Propagates the dimension check from [`Self::senet_gates`].
    pub fn senet(&self, field_embs: &[f32]) -> RecsysResult<Vec<f32>> {
        let m = self.cfg.n_fields;
        let d = self.cfg.embed_dim;
        let gates = self.senet_gates(field_embs)?;
        let mut out = vec![0.0_f32; m * d];
        for f in 0..m {
            let a = gates[f];
            for k in 0..d {
                out[f * d + k] = a * field_embs[f * d + k];
            }
        }
        Ok(out)
    }

    /// Select the bilinear matrix for the field pair `(i, j)` with `i < j`,
    /// according to the configured [`BilinearType`].
    fn bilinear_matrix(&self, i: usize, pair_idx: usize) -> &[f32] {
        match self.cfg.bilinear_type {
            BilinearType::FieldAll => &self.bilinear_w[0],
            BilinearType::FieldEach => &self.bilinear_w[i],
            BilinearType::FieldInteraction => &self.bilinear_w[pair_idx],
        }
    }

    /// Bilinear interaction over all field pairs `i < j`:
    /// `v_ij = p_i ∘ (W · p_j)`, concatenated into a vector of length
    /// `C(n_fields, 2) · embed_dim`.
    ///
    /// # Errors
    /// Returns [`RecsysError::DimensionMismatch`] if `field_embs.len()` differs
    /// from `n_fields · embed_dim`.
    pub fn bilinear_interaction(&self, field_embs: &[f32]) -> RecsysResult<Vec<f32>> {
        let m = self.cfg.n_fields;
        let d = self.cfg.embed_dim;
        if field_embs.len() != m * d {
            return Err(RecsysError::DimensionMismatch {
                expected: m * d,
                got: field_embs.len(),
            });
        }
        let mut out = Vec::with_capacity(self.bilinear_out_dim);
        let mut pair_idx = 0usize;
        for i in 0..m {
            let p_i = &field_embs[i * d..(i + 1) * d];
            for j in (i + 1)..m {
                let p_j = &field_embs[j * d..(j + 1) * d];
                let w = self.bilinear_matrix(i, pair_idx);
                let wp_j = matvec(w, p_j, d);
                // Hadamard product p_i ∘ (W p_j).
                for k in 0..d {
                    out.push(p_i[k] * wp_j[k]);
                }
                pair_idx += 1;
            }
        }
        Ok(out)
    }

    /// Full forward pass: bilinear interaction of the raw embeddings and of the
    /// SENET-reweighted embeddings, concatenated and passed through the DNN →
    /// sigmoid. Returns a CTR probability in `(0, 1)`.
    ///
    /// # Errors
    /// Propagates dimension checks from [`Self::bilinear_interaction`] and
    /// [`Self::senet`].
    pub fn forward(&self, field_embs: &[f32]) -> RecsysResult<f32> {
        let raw_inter = self.bilinear_interaction(field_embs)?;
        let senet_embs = self.senet(field_embs)?;
        let senet_inter = self.bilinear_interaction(&senet_embs)?;

        let mut combined = Vec::with_capacity(raw_inter.len() + senet_inter.len());
        combined.extend_from_slice(&raw_inter);
        combined.extend_from_slice(&senet_inter);

        let logit_vec = mlp_forward(&combined, &self.dnn_layers);
        let logit = logit_vec.first().copied().unwrap_or(0.0);
        Ok(sigmoid(logit))
    }

    /// Total number of learnable parameters (SENET + bilinear + DNN).
    #[must_use]
    pub fn n_params(&self) -> usize {
        let senet =
            self.senet_w1.len() + self.senet_b1.len() + self.senet_w2.len() + self.senet_b2.len();
        let bilinear: usize = self.bilinear_w.iter().map(Vec::len).sum();
        let dnn: usize = self.dnn_layers.iter().map(|(w, b)| w.len() + b.len()).sum();
        senet + bilinear + dnn
    }
}

/// Build a stack of `(weight, bias)` layers from consecutive dimensions.
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
        let mut out: Vec<f32> = (0..out_dim)
            .map(|o| {
                b[o] + w[o * cur_dim..(o + 1) * cur_dim]
                    .iter()
                    .zip(current.iter())
                    .map(|(&wi, &xi)| wi * xi)
                    .sum::<f32>()
            })
            .collect();
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

    fn default_cfg() -> FibinetConfig {
        FibinetConfig {
            n_fields: 4,
            embed_dim: 8,
            reduction_ratio: 2,
            bilinear_type: BilinearType::FieldInteraction,
            dnn_hidden: vec![32, 16],
        }
    }

    fn random_embs(m: usize, d: usize, rng: &mut LcgRng) -> Vec<f32> {
        (0..m * d).map(|_| rng.next_normal()).collect()
    }

    #[test]
    fn senet_output_length() {
        let mut rng = make_rng();
        let model = Fibinet::new(default_cfg(), &mut rng).expect("value should be present");
        let embs = random_embs(4, 8, &mut rng);
        let out = model.senet(&embs).expect("senet should succeed");
        assert_eq!(out.len(), 4 * 8);
    }

    #[test]
    fn senet_gates_in_unit_interval() {
        let mut rng = make_rng();
        let model = Fibinet::new(default_cfg(), &mut rng).expect("value should be present");
        let embs = random_embs(4, 8, &mut rng);
        let gates = model
            .senet_gates(&embs)
            .expect("senet_gates should succeed");
        assert_eq!(gates.len(), 4);
        for &a in &gates {
            assert!(a.is_finite(), "gate must be finite");
            assert!((0.0..=1.0).contains(&a), "gate {a} not in [0,1]");
        }
    }

    #[test]
    fn senet_gates_constant_input_in_unit_interval() {
        // A constant-input case keeps the sigmoid-bounded property explicit.
        let mut rng = make_rng();
        let model = Fibinet::new(default_cfg(), &mut rng).expect("value should be present");
        let embs = vec![0.5_f32; 4 * 8];
        let gates = model
            .senet_gates(&embs)
            .expect("senet_gates should succeed");
        for &a in &gates {
            assert!((0.0..=1.0).contains(&a), "gate {a} not in [0,1]");
        }
    }

    #[test]
    fn senet_gates_depend_only_on_field_means() {
        // SENET squeezes each field to its mean over embed_dim. Two distinct
        // embedding sets with identical per-field means must yield identical
        // gates (the excitation acts solely on the squeezed descriptor z).
        let mut rng = make_rng();
        let model = Fibinet::new(default_cfg(), &mut rng).expect("value should be present");
        let d = 8;
        // Field f has constant value (f + 1) → mean (f + 1).
        let mut embs_a = Vec::new();
        for f in 0..4 {
            embs_a.extend(std::iter::repeat_n((f as f32) + 1.0, d));
        }
        // Same per-field means via a zero-sum perturbation pattern [+v, -v, …].
        let mut embs_b = Vec::new();
        for f in 0..4 {
            let base = (f as f32) + 1.0;
            for k in 0..d {
                let perturb = if k % 2 == 0 { 0.5 } else { -0.5 };
                embs_b.push(base + perturb);
            }
        }
        let gates_a = model
            .senet_gates(&embs_a)
            .expect("senet_gates should succeed");
        let gates_b = model
            .senet_gates(&embs_b)
            .expect("senet_gates should succeed");
        for f in 0..4 {
            assert!(
                (gates_a[f] - gates_b[f]).abs() < 1e-5,
                "gates must depend only on per-field means (field {f})"
            );
        }
    }

    #[test]
    fn bilinear_interaction_length() {
        let mut rng = make_rng();
        let model = Fibinet::new(default_cfg(), &mut rng).expect("value should be present");
        let embs = random_embs(4, 8, &mut rng);
        let out = model
            .bilinear_interaction(&embs)
            .expect("bilinear_interaction should succeed");
        // C(4,2) = 6 pairs × embed_dim 8 = 48.
        assert_eq!(out.len(), 6 * 8);
        assert_eq!(out.len(), model.bilinear_out_dim);
    }

    #[test]
    fn bilinear_single_pair_length() {
        let mut rng = make_rng();
        let cfg = FibinetConfig {
            n_fields: 2,
            embed_dim: 8,
            reduction_ratio: 1,
            bilinear_type: BilinearType::FieldAll,
            dnn_hidden: vec![16],
        };
        let model = Fibinet::new(cfg, &mut rng).expect("new should succeed");
        let embs = random_embs(2, 8, &mut rng);
        let out = model
            .bilinear_interaction(&embs)
            .expect("bilinear_interaction should succeed");
        // n_fields = 2 → single pair → length == embed_dim.
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn forward_in_open_unit_interval() {
        let mut rng = make_rng();
        let model = Fibinet::new(default_cfg(), &mut rng).expect("value should be present");
        let embs = random_embs(4, 8, &mut rng);
        let p = model.forward(&embs).expect("forward should succeed");
        assert!(p.is_finite(), "probability must be finite, got {p}");
        assert!(p > 0.0 && p < 1.0, "probability {p} not in (0,1)");
    }

    #[test]
    fn deterministic_given_seed() {
        let mut rng_a = LcgRng::new(11);
        let mut rng_b = LcgRng::new(11);
        let model_a = Fibinet::new(default_cfg(), &mut rng_a).expect("value should be present");
        let model_b = Fibinet::new(default_cfg(), &mut rng_b).expect("value should be present");
        // Build identical inputs from a fresh, independent stream.
        let mut rng_in = LcgRng::new(999);
        let embs = random_embs(4, 8, &mut rng_in);
        let pa = model_a.forward(&embs).expect("forward should succeed");
        let pb = model_b.forward(&embs).expect("forward should succeed");
        assert!((pa - pb).abs() < 1e-6, "same seed must give same output");
    }

    #[test]
    fn field_embs_wrong_length_errors() {
        let mut rng = make_rng();
        let model = Fibinet::new(default_cfg(), &mut rng).expect("value should be present");
        let embs = vec![0.1_f32; 4 * 8 - 1];
        assert!(matches!(
            model.forward(&embs),
            Err(RecsysError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            model.senet(&embs),
            Err(RecsysError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            model.bilinear_interaction(&embs),
            Err(RecsysError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_n_fields_lt_2() {
        let mut rng = make_rng();
        let cfg = FibinetConfig {
            n_fields: 1,
            embed_dim: 8,
            reduction_ratio: 1,
            bilinear_type: BilinearType::FieldAll,
            dnn_hidden: vec![],
        };
        assert!(matches!(
            Fibinet::new(cfg, &mut rng),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn err_embed_dim_zero() {
        let mut rng = make_rng();
        let cfg = FibinetConfig {
            n_fields: 4,
            embed_dim: 0,
            reduction_ratio: 1,
            bilinear_type: BilinearType::FieldAll,
            dnn_hidden: vec![],
        };
        assert!(matches!(
            Fibinet::new(cfg, &mut rng),
            Err(RecsysError::InvalidEmbeddingDim { .. })
        ));
    }

    #[test]
    fn err_reduction_ratio_zero() {
        let mut rng = make_rng();
        let cfg = FibinetConfig {
            n_fields: 4,
            embed_dim: 8,
            reduction_ratio: 0,
            bilinear_type: BilinearType::FieldAll,
            dnn_hidden: vec![],
        };
        assert!(matches!(
            Fibinet::new(cfg, &mut rng),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn n_params_positive() {
        let mut rng = make_rng();
        let model = Fibinet::new(default_cfg(), &mut rng).expect("value should be present");
        assert!(model.n_params() > 0, "n_params must be > 0");
    }

    #[test]
    fn all_bilinear_types_run_with_correct_length() {
        let types = [
            BilinearType::FieldAll,
            BilinearType::FieldEach,
            BilinearType::FieldInteraction,
        ];
        for bt in types {
            let mut rng = make_rng();
            let cfg = FibinetConfig {
                n_fields: 4,
                embed_dim: 8,
                reduction_ratio: 2,
                bilinear_type: bt,
                dnn_hidden: vec![16],
            };
            let model = Fibinet::new(cfg, &mut rng).expect("new should succeed");
            let embs = random_embs(4, 8, &mut rng);
            let inter = model
                .bilinear_interaction(&embs)
                .expect("bilinear_interaction should succeed");
            assert_eq!(inter.len(), 6 * 8, "wrong bilinear length for {bt:?}");
            let p = model.forward(&embs).expect("forward should succeed");
            assert!(p > 0.0 && p < 1.0, "forward out of range for {bt:?}");
        }
    }

    #[test]
    fn changing_field_embs_changes_output() {
        let mut rng = make_rng();
        let model = Fibinet::new(default_cfg(), &mut rng).expect("value should be present");
        let e1 = random_embs(4, 8, &mut rng);
        let e2 = random_embs(4, 8, &mut rng);
        let p1 = model.forward(&e1).expect("forward should succeed");
        let p2 = model.forward(&e2).expect("forward should succeed");
        assert!((p1 - p2).abs() > 1e-9, "different inputs must differ");
    }

    #[test]
    fn reduction_ratio_larger_than_n_fields_clamps_hidden() {
        let mut rng = make_rng();
        let cfg = FibinetConfig {
            n_fields: 3,
            embed_dim: 8,
            reduction_ratio: 16, // 3/16 = 0 → clamp to 1
            bilinear_type: BilinearType::FieldAll,
            dnn_hidden: vec![8],
        };
        let model = Fibinet::new(cfg, &mut rng).expect("new should succeed");
        assert!(model.senet_hidden >= 1, "hidden width must be >= 1");
        assert_eq!(model.senet_hidden, 1);
        let embs = random_embs(3, 8, &mut rng);
        let gates = model
            .senet_gates(&embs)
            .expect("senet_gates should succeed");
        assert_eq!(gates.len(), 3);
    }

    #[test]
    fn field_interaction_has_more_params_than_field_all() {
        let mut rng_a = make_rng();
        let mut rng_b = make_rng();
        let base = |bt: BilinearType| FibinetConfig {
            n_fields: 5,
            embed_dim: 8,
            reduction_ratio: 2,
            bilinear_type: bt,
            dnn_hidden: vec![16],
        };
        let all = Fibinet::new(base(BilinearType::FieldAll), &mut rng_a)
            .expect("value should be present");
        let inter = Fibinet::new(base(BilinearType::FieldInteraction), &mut rng_b)
            .expect("value should be present");
        assert!(
            inter.n_params() > all.n_params(),
            "FieldInteraction ({}) must have more params than FieldAll ({})",
            inter.n_params(),
            all.n_params()
        );
    }

    #[test]
    fn field_each_param_count_between_all_and_interaction() {
        // FieldEach uses n_fields matrices; FieldAll 1; FieldInteraction C(n,2).
        // For n_fields = 5: 5 vs 1 vs 10 matrices.
        let cfg = |bt: BilinearType| FibinetConfig {
            n_fields: 5,
            embed_dim: 8,
            reduction_ratio: 2,
            bilinear_type: bt,
            dnn_hidden: vec![16],
        };
        let mut ra = make_rng();
        let mut rb = make_rng();
        let mut rc = make_rng();
        let all =
            Fibinet::new(cfg(BilinearType::FieldAll), &mut ra).expect("value should be present");
        let each =
            Fibinet::new(cfg(BilinearType::FieldEach), &mut rb).expect("value should be present");
        let inter = Fibinet::new(cfg(BilinearType::FieldInteraction), &mut rc)
            .expect("value should be present");
        assert!(all.n_params() < each.n_params());
        assert!(each.n_params() < inter.n_params());
    }

    #[test]
    fn bilinear_matrix_count_matches_type() {
        let mut rng = make_rng();
        let cfg_all = FibinetConfig {
            n_fields: 4,
            embed_dim: 8,
            reduction_ratio: 2,
            bilinear_type: BilinearType::FieldAll,
            dnn_hidden: vec![],
        };
        let model_all = Fibinet::new(cfg_all, &mut rng).expect("new should succeed");
        assert_eq!(model_all.bilinear_w.len(), 1);

        let mut rng2 = make_rng();
        let cfg_each = FibinetConfig {
            n_fields: 4,
            embed_dim: 8,
            reduction_ratio: 2,
            bilinear_type: BilinearType::FieldEach,
            dnn_hidden: vec![],
        };
        let model_each = Fibinet::new(cfg_each, &mut rng2).expect("new should succeed");
        assert_eq!(model_each.bilinear_w.len(), 4);

        let mut rng3 = make_rng();
        let cfg_inter = FibinetConfig {
            n_fields: 4,
            embed_dim: 8,
            reduction_ratio: 2,
            bilinear_type: BilinearType::FieldInteraction,
            dnn_hidden: vec![],
        };
        let model_inter = Fibinet::new(cfg_inter, &mut rng3).expect("new should succeed");
        assert_eq!(model_inter.bilinear_w.len(), 6);
    }

    #[test]
    fn empty_dnn_hidden_single_linear_layer() {
        let mut rng = make_rng();
        let cfg = FibinetConfig {
            n_fields: 3,
            embed_dim: 4,
            reduction_ratio: 1,
            bilinear_type: BilinearType::FieldAll,
            dnn_hidden: vec![],
        };
        let model = Fibinet::new(cfg, &mut rng).expect("new should succeed");
        assert_eq!(model.dnn_layers.len(), 1, "empty hidden → one linear layer");
        let embs = random_embs(3, 4, &mut rng);
        let p = model.forward(&embs).expect("forward should succeed");
        assert!(p > 0.0 && p < 1.0);
    }

    #[test]
    fn bilinear_matvec_hadamard_values() {
        // Verify v_ij = p_i ∘ (W p_j) on a hand-built 2-field, d=2 model.
        let mut rng = make_rng();
        let cfg = FibinetConfig {
            n_fields: 2,
            embed_dim: 2,
            reduction_ratio: 1,
            bilinear_type: BilinearType::FieldAll,
            dnn_hidden: vec![],
        };
        let mut model = Fibinet::new(cfg, &mut rng).expect("new should succeed");
        // Override W with a known matrix [[1,2],[3,4]] (row-major).
        model.bilinear_w[0] = vec![1.0, 2.0, 3.0, 4.0];
        // p_0 = [2, 3], p_1 = [5, 7].
        let embs = vec![2.0_f32, 3.0, 5.0, 7.0];
        let out = model
            .bilinear_interaction(&embs)
            .expect("bilinear_interaction should succeed");
        // W p_1 = [1*5 + 2*7, 3*5 + 4*7] = [19, 43].
        // p_0 ∘ (W p_1) = [2*19, 3*43] = [38, 129].
        assert!((out[0] - 38.0).abs() < 1e-4);
        assert!((out[1] - 129.0).abs() < 1e-4);
    }
}
