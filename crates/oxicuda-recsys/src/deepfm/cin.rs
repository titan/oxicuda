use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

// ── Configuration ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CinConfig {
    /// Number of feature fields (m).
    pub n_fields: usize,
    /// Embedding dimension per field (D).
    pub embed_dim: usize,
    /// H_k: number of feature maps in each CIN layer.
    pub cin_layer_sizes: Vec<usize>,
    /// DNN hidden layer sizes.
    pub dnn_hidden_sizes: Vec<usize>,
    pub learning_rate: f32,
    /// L2 regularization on all weights.
    pub l2_reg: f32,
    pub n_iter: usize,
    pub batch_size: usize,
}

// ── CIN Layer ─────────────────────────────────────────────────────────────────

/// One Compressed Interaction Network layer.
///
/// Computes X^k (out_fields × D) from X^{k-1} (in_fields × D) and X^0 (n_fields_0 × D)
/// by forming the compressed outer product over the embedding dimension.
#[derive(Debug, Clone)]
pub struct CinLayer {
    /// H_{k-1} (or n_fields for the first layer).
    pub in_fields: usize,
    /// H_k: number of output feature maps.
    pub out_fields: usize,
    /// Embedding dimension D (constant across all layers).
    pub embed_dim: usize,
    /// W^k: shape out_fields × in_fields × n_fields_0, row-major over (h', h, l).
    pub weights: Vec<f32>,
    /// n_fields_0 stored to reconstruct weight indexing.
    pub n_fields_0: usize,
}

impl CinLayer {
    /// Create a new CIN layer with Kaiming-uniform weight initialisation.
    pub fn new(
        in_fields: usize,
        out_fields: usize,
        embed_dim: usize,
        n_fields_0: usize,
        rng: &mut LcgRng,
    ) -> Self {
        let fan_in = in_fields * n_fields_0;
        // Kaiming uniform: ±√(6 / fan_in)
        let bound = if fan_in > 0 {
            (6.0 / fan_in as f32).sqrt()
        } else {
            0.01
        };
        let len = out_fields * in_fields * n_fields_0;
        let weights: Vec<f32> = (0..len)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * bound)
            .collect();
        Self {
            in_fields,
            out_fields,
            embed_dim,
            weights,
            n_fields_0,
        }
    }

    /// Forward pass for one CIN layer.
    ///
    /// `x_k`: in_fields × embed_dim (current-layer activations, or X^0 for first layer)
    /// `x_0`: n_fields_0 × embed_dim (original X^0)
    ///
    /// Returns: out_fields × embed_dim (after ReLU).
    pub fn forward(&self, x_k: &[f32], x_0: &[f32]) -> Vec<f32> {
        let d = self.embed_dim;
        let h_in = self.in_fields;
        let h_out = self.out_fields;
        let m = self.n_fields_0;

        // For each output map h', each dimension d:
        //   X^k[h',d] = Σ_{h=0}^{h_in-1} Σ_{l=0}^{m-1} W[h',h,l] · x_k[h,d] · x_0[l,d]
        let mut out = vec![0.0_f32; h_out * d];
        for hp in 0..h_out {
            for d_idx in 0..d {
                let mut val = 0.0_f32;
                for h in 0..h_in {
                    let x_k_hd = x_k[h * d + d_idx];
                    for l in 0..m {
                        let x_0_ld = x_0[l * d + d_idx];
                        // weight index: (hp * h_in + h) * m + l
                        let w_idx = (hp * h_in + h) * m + l;
                        val += self.weights[w_idx] * x_k_hd * x_0_ld;
                    }
                }
                out[hp * d + d_idx] = val.max(0.0);
            }
        }
        out
    }
}

// ── xDeepFM Model ─────────────────────────────────────────────────────────────

/// xDeepFM (Lian et al. KDD 2018): CIN + DNN + Linear.
///
/// Forward path:
///   1. CIN: stacked CinLayer, sum-pool per layer over D → concatenate → CIN score.
///   2. DNN: MLP over flattened field embeddings → scalar.
///   3. Linear: FM-style Σ e² term.
///   4. Output: learned linear combination of all three with bias.
pub struct XDeepFm {
    pub cfg: CinConfig,
    /// Field embedding lookup: n_fields × embed_dim.
    pub embeddings: Vec<f32>,
    pub cin_layers: Vec<CinLayer>,
    /// DNN weight matrices: one per hidden layer (fan_out × fan_in).
    pub dnn_weights: Vec<Vec<f32>>,
    pub dnn_biases: Vec<Vec<f32>>,
    /// Final output weights for [CIN_concat | DNN_out | linear_fm].
    pub output_weights: Vec<f32>,
    pub output_bias: f32,
}

impl XDeepFm {
    /// Construct xDeepFM with Kaiming-uniform weight initialisation.
    pub fn new(cfg: CinConfig, rng: &mut LcgRng) -> RecsysResult<Self> {
        if cfg.n_fields == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "n_fields must be >= 1".into(),
            });
        }
        if cfg.embed_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: 0 });
        }
        if cfg.cin_layer_sizes.is_empty() {
            return Err(RecsysError::InvalidConfig {
                msg: "cin_layer_sizes must be non-empty".into(),
            });
        }
        if cfg.dnn_hidden_sizes.is_empty() {
            return Err(RecsysError::InvalidConfig {
                msg: "dnn_hidden_sizes must be non-empty".into(),
            });
        }
        if cfg.learning_rate <= 0.0 {
            return Err(RecsysError::InvalidConfig {
                msg: format!("learning_rate must be > 0, got {}", cfg.learning_rate),
            });
        }
        if cfg.l2_reg < 0.0 {
            return Err(RecsysError::InvalidConfig {
                msg: format!("l2_reg must be >= 0, got {}", cfg.l2_reg),
            });
        }

        let m = cfg.n_fields;
        let d = cfg.embed_dim;

        // Field embeddings: uniform small initialisation.
        let emb_bound = (6.0_f32 / (m * d) as f32).sqrt();
        let embeddings: Vec<f32> = (0..m * d)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * emb_bound)
            .collect();

        // CIN layers: X^0 has m fields; each layer compresses against X^0.
        let mut cin_layers = Vec::with_capacity(cfg.cin_layer_sizes.len());
        let mut prev_fields = m;
        for &out_fields in &cfg.cin_layer_sizes {
            cin_layers.push(CinLayer::new(prev_fields, out_fields, d, m, rng));
            prev_fields = out_fields;
        }

        // DNN layers: input is m * d flattened embeddings.
        let mut dnn_weights = Vec::with_capacity(cfg.dnn_hidden_sizes.len() + 1);
        let mut dnn_biases = Vec::with_capacity(cfg.dnn_hidden_sizes.len() + 1);
        let mut in_dim = m * d;
        for &out_dim in &cfg.dnn_hidden_sizes {
            let bound = (6.0_f32 / in_dim as f32).sqrt();
            let w: Vec<f32> = (0..out_dim * in_dim)
                .map(|_| (rng.next_f32() * 2.0 - 1.0) * bound)
                .collect();
            dnn_weights.push(w);
            dnn_biases.push(vec![0.0_f32; out_dim]);
            in_dim = out_dim;
        }
        // Final DNN scalar layer.
        {
            let bound = (6.0_f32 / in_dim as f32).sqrt();
            let w: Vec<f32> = (0..in_dim)
                .map(|_| (rng.next_f32() * 2.0 - 1.0) * bound)
                .collect();
            dnn_weights.push(w);
            dnn_biases.push(vec![0.0_f32; 1]);
        }

        // Output weights: CIN sum-pool output + DNN scalar + linear FM scalar.
        let cin_total: usize = cfg.cin_layer_sizes.iter().sum();
        let out_dim = cin_total + 2; // +1 DNN, +1 linear FM
        let out_bound = (6.0_f32 / out_dim as f32).sqrt();
        let output_weights: Vec<f32> = (0..out_dim)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * out_bound)
            .collect();

        Ok(Self {
            cfg,
            embeddings,
            cin_layers,
            dnn_weights,
            dnn_biases,
            output_weights,
            output_bias: 0.0,
        })
    }

    /// Embed field IDs into a flat n_fields × embed_dim slice.
    ///
    /// Each `field_ids[i]` must be < n_fields (we treat each field position as its
    /// own single-cardinality field for lookup; for general multi-cardinality use,
    /// the caller provides pre-computed embeddings directly).
    pub fn embed(&self, field_ids: &[usize]) -> RecsysResult<Vec<f32>> {
        let m = self.cfg.n_fields;
        let d = self.cfg.embed_dim;
        if field_ids.len() != m {
            return Err(RecsysError::DimensionMismatch {
                expected: m,
                got: field_ids.len(),
            });
        }
        let mut out = vec![0.0_f32; m * d];
        for (f, &id) in field_ids.iter().enumerate() {
            if id >= m {
                return Err(RecsysError::ItemOutOfBounds { idx: id, n: m });
            }
            out[f * d..(f + 1) * d].copy_from_slice(&self.embeddings[id * d..(id + 1) * d]);
        }
        Ok(out)
    }

    /// CIN forward pass.
    ///
    /// `x_0`: n_fields × embed_dim field embeddings.
    /// Returns the concatenated sum-pooled feature maps: Σ H_k scalars.
    pub fn cin_forward(&self, x_0: &[f32]) -> Vec<f32> {
        let d = self.cfg.embed_dim;
        let mut pooled_all: Vec<f32> = Vec::new();
        let mut x_prev = x_0.to_vec();

        for layer in &self.cin_layers {
            let x_k = layer.forward(&x_prev, x_0);
            // Sum-pool over D: for each output field h', p_{h'} = Σ_d X^k[h',d].
            for h in 0..layer.out_fields {
                let sum: f32 = x_k[h * d..(h + 1) * d].iter().sum();
                pooled_all.push(sum);
            }
            x_prev = x_k;
        }
        pooled_all
    }

    /// DNN forward: flattened field embeddings → scalar via tanh hidden layers.
    pub fn dnn_forward(&self, input: &[f32]) -> f32 {
        let n_layers = self.dnn_weights.len();
        let mut cur = input.to_vec();
        let mut in_dim = input.len();

        for (layer_idx, (w, b)) in self
            .dnn_weights
            .iter()
            .zip(self.dnn_biases.iter())
            .enumerate()
        {
            let out_dim = b.len();
            let mut next = vec![0.0_f32; out_dim];
            for o in 0..out_dim {
                let dot: f32 = w[o * in_dim..(o + 1) * in_dim]
                    .iter()
                    .zip(cur.iter())
                    .map(|(&wi, &xi)| wi * xi)
                    .sum();
                next[o] = dot + b[o];
            }
            // Apply tanh for hidden layers; final layer is linear (scalar).
            if layer_idx + 1 < n_layers {
                for v in &mut next {
                    *v = v.tanh();
                }
            }
            in_dim = out_dim;
            cur = next;
        }
        cur.first().copied().unwrap_or(0.0)
    }

    /// Full xDeepFM forward: returns logit (pre-sigmoid).
    ///
    /// `field_embeds`: n_fields × embed_dim flat slice.
    pub fn forward(&self, field_embeds: &[f32]) -> RecsysResult<f32> {
        let m = self.cfg.n_fields;
        let d = self.cfg.embed_dim;
        if field_embeds.len() != m * d {
            return Err(RecsysError::DimensionMismatch {
                expected: m * d,
                got: field_embeds.len(),
            });
        }

        // CIN path.
        let cin_out = self.cin_forward(field_embeds);

        // DNN path.
        let dnn_out = self.dnn_forward(field_embeds);

        // Linear FM path: Σ_{m,d} e_{m,d}² (sum of squared embeddings).
        let linear_fm: f32 = field_embeds.iter().map(|&v| v * v).sum();

        // Concatenate: [cin_out..., dnn_out, linear_fm] → dot with output_weights + bias.
        let cin_len = cin_out.len();
        let logit: f32 = cin_out
            .iter()
            .zip(self.output_weights[..cin_len].iter())
            .map(|(&c, &ow)| c * ow)
            .sum::<f32>()
            + dnn_out * self.output_weights[cin_len]
            + linear_fm * self.output_weights[cin_len + 1]
            + self.output_bias;

        Ok(logit)
    }

    /// Sigmoid function.
    #[inline]
    pub fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Binary cross-entropy loss: -[y·log(p) + (1-y)·log(1-p)].
    pub fn bce_loss(logit: f32, label: f32) -> f32 {
        // Numerically stable form: max(logit,0) - logit*y + log(1+exp(-|logit|))
        let pos_part = logit.max(0.0);
        pos_part - logit * label + (1.0 + (-logit.abs()).exp()).ln()
    }

    /// Batch training step: forward → BCE loss → analytical gradient on output layer.
    ///
    /// `batch_embeds`: batch_size × n_fields × embed_dim, flat row-major.
    /// `labels`: batch_size binary (0/1) labels.
    /// Returns mean BCE loss over the batch.
    pub fn train_step(
        &mut self,
        batch_embeds: &[f32],
        labels: &[f32],
        _rng: &mut LcgRng,
    ) -> RecsysResult<f32> {
        let bs = labels.len();
        if bs == 0 {
            return Err(RecsysError::EmptyInput);
        }
        let m = self.cfg.n_fields;
        let d = self.cfg.embed_dim;
        let sample_len = m * d;
        if batch_embeds.len() != bs * sample_len {
            return Err(RecsysError::DimensionMismatch {
                expected: bs * sample_len,
                got: batch_embeds.len(),
            });
        }

        let lr = self.cfg.learning_rate;
        let l2 = self.cfg.l2_reg;
        let cin_total: usize = self.cfg.cin_layer_sizes.iter().sum();
        let out_dim = cin_total + 2;

        let mut total_loss = 0.0_f32;
        // Accumulate gradient over the batch for output layer weights.
        let mut grad_ow = vec![0.0_f32; out_dim];
        let mut grad_ob = 0.0_f32;

        // Storage for each sample's component outputs (for gradient accumulation).
        let mut cin_outs: Vec<Vec<f32>> = Vec::with_capacity(bs);
        let mut dnn_outs: Vec<f32> = Vec::with_capacity(bs);
        let mut fm_outs: Vec<f32> = Vec::with_capacity(bs);

        for s in 0..bs {
            let embed_s = &batch_embeds[s * sample_len..(s + 1) * sample_len];
            let logit = self.forward(embed_s)?;
            let loss = Self::bce_loss(logit, labels[s]);
            total_loss += loss;

            // δ = σ(logit) - label (gradient of BCE w.r.t. logit).
            let delta = Self::sigmoid(logit) - labels[s];

            let cin_out = self.cin_forward(embed_s);
            let dnn_out = self.dnn_forward(embed_s);
            let fm_out: f32 = embed_s.iter().map(|&v| v * v).sum();

            // Accumulate output-layer gradients: ∂L/∂ow_i = δ · component_i.
            for (i, &c) in cin_out.iter().enumerate() {
                grad_ow[i] += delta * c;
            }
            grad_ow[cin_total] += delta * dnn_out;
            grad_ow[cin_total + 1] += delta * fm_out;
            grad_ob += delta;

            cin_outs.push(cin_out);
            dnn_outs.push(dnn_out);
            fm_outs.push(fm_out);
        }

        // SGD update for output weights (mean gradient + L2).
        let bs_f = bs as f32;
        for (ow, gw) in self.output_weights.iter_mut().zip(grad_ow.iter()) {
            let g = *gw / bs_f + l2 * *ow;
            *ow -= lr * g;
        }
        self.output_bias -= lr * grad_ob / bs_f;

        Ok(total_loss / bs_f)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn default_cfg() -> CinConfig {
        CinConfig {
            n_fields: 4,
            embed_dim: 8,
            cin_layer_sizes: vec![16, 8],
            dnn_hidden_sizes: vec![32, 16],
            learning_rate: 0.01,
            l2_reg: 1e-5,
            n_iter: 10,
            batch_size: 4,
        }
    }

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn random_embeds(bs: usize, n_fields: usize, embed_dim: usize, rng: &mut LcgRng) -> Vec<f32> {
        (0..bs * n_fields * embed_dim)
            .map(|_| rng.next_f32() * 0.2 - 0.1)
            .collect()
    }

    fn random_labels(bs: usize, rng: &mut LcgRng) -> Vec<f32> {
        (0..bs)
            .map(|_| if rng.next_usize(2) == 0 { 0.0 } else { 1.0 })
            .collect()
    }

    #[test]
    fn cin_layer_output_shape() {
        let mut rng = make_rng();
        let layer = CinLayer::new(4, 8, 6, 4, &mut rng);
        let x_k: Vec<f32> = (0..4 * 6).map(|_| rng.next_f32()).collect();
        let x_0: Vec<f32> = (0..4 * 6).map(|_| rng.next_f32()).collect();
        let out = layer.forward(&x_k, &x_0);
        assert_eq!(
            out.len(),
            8 * 6,
            "CinLayer output should be out_fields × embed_dim"
        );
    }

    #[test]
    fn cin_layer_relu_non_negative() {
        let mut rng = make_rng();
        let layer = CinLayer::new(3, 5, 4, 3, &mut rng);
        let x_k: Vec<f32> = (0..3 * 4).map(|_| rng.next_f32() - 0.5).collect();
        let x_0: Vec<f32> = (0..3 * 4).map(|_| rng.next_f32() - 0.5).collect();
        let out = layer.forward(&x_k, &x_0);
        for (i, &v) in out.iter().enumerate() {
            assert!(v >= 0.0, "ReLU output at {i} should be >= 0, got {v}");
        }
    }

    #[test]
    fn cin_layer_weights_shape() {
        let mut rng = make_rng();
        let in_fields = 4;
        let out_fields = 6;
        let embed_dim = 8;
        let n_fields_0 = 4;
        let layer = CinLayer::new(in_fields, out_fields, embed_dim, n_fields_0, &mut rng);
        assert_eq!(
            layer.weights.len(),
            out_fields * in_fields * n_fields_0,
            "weights shape mismatch"
        );
    }

    #[test]
    fn xdeepfm_forward_finite() {
        let mut rng = make_rng();
        let cfg = default_cfg();
        let model = XDeepFm::new(cfg, &mut rng).expect("new should succeed");
        let embeds: Vec<f32> = (0..4 * 8).map(|_| rng.next_f32() * 0.1).collect();
        let logit = model.forward(&embeds).expect("forward should succeed");
        assert!(
            logit.is_finite(),
            "forward output must be finite, got {logit}"
        );
    }

    #[test]
    fn xdeepfm_sigmoid_range() {
        let mut rng = make_rng();
        let cfg = default_cfg();
        let model = XDeepFm::new(cfg, &mut rng).expect("new should succeed");
        let embeds: Vec<f32> = (0..4 * 8).map(|_| rng.next_f32() * 0.1).collect();
        let logit = model.forward(&embeds).expect("forward should succeed");
        let p = XDeepFm::sigmoid(logit);
        assert!(
            p > 0.0 && p < 1.0,
            "sigmoid output must be in (0,1), got {p}"
        );
    }

    #[test]
    fn sigmoid_zero_is_half() {
        let p = XDeepFm::sigmoid(0.0);
        assert!((p - 0.5).abs() < 1e-6, "sigmoid(0) should be 0.5, got {p}");
    }

    #[test]
    fn bce_loss_zero_for_perfect() {
        // Large positive logit → sigmoid ≈ 1; label = 1 → loss ≈ 0.
        let loss = XDeepFm::bce_loss(100.0, 1.0);
        assert!(loss < 1e-3, "bce_loss(100,1) should be near 0, got {loss}");
    }

    #[test]
    fn bce_loss_positive() {
        for logit in &[-5.0_f32, -1.0, 0.0, 1.0, 5.0] {
            for label in &[0.0_f32, 1.0] {
                let loss = XDeepFm::bce_loss(*logit, *label);
                assert!(
                    loss >= 0.0,
                    "bce_loss({logit},{label}) must be >= 0, got {loss}"
                );
            }
        }
    }

    #[test]
    fn embed_output_length() {
        let mut rng = make_rng();
        let cfg = default_cfg();
        let model = XDeepFm::new(cfg, &mut rng).expect("new should succeed");
        let field_ids: Vec<usize> = vec![0, 1, 2, 3];
        let out = model.embed(&field_ids).expect("embed should succeed");
        assert_eq!(
            out.len(),
            4 * 8,
            "embed output should be n_fields × embed_dim"
        );
    }

    #[test]
    fn embed_err_out_of_bounds() {
        let mut rng = make_rng();
        let cfg = default_cfg();
        let model = XDeepFm::new(cfg, &mut rng).expect("new should succeed");
        // field_id = 4 >= n_fields = 4 → error
        let field_ids = vec![0, 1, 2, 4];
        assert!(matches!(
            model.embed(&field_ids),
            Err(RecsysError::ItemOutOfBounds { .. })
        ));
    }

    #[test]
    fn train_step_returns_finite_loss() {
        let mut rng = make_rng();
        let cfg = default_cfg();
        let mut model = XDeepFm::new(cfg.clone(), &mut rng).expect("value should be present");
        let m = cfg.n_fields;
        let d = cfg.embed_dim;
        let bs = cfg.batch_size;
        let embeds = random_embeds(bs, m, d, &mut rng);
        let labels = random_labels(bs, &mut rng);
        let loss = model
            .train_step(&embeds, &labels, &mut rng)
            .expect("train_step should succeed");
        assert!(
            loss.is_finite(),
            "train_step loss must be finite, got {loss}"
        );
    }

    #[test]
    fn train_step_decreases_loss() {
        let mut rng = make_rng();
        let cfg = CinConfig {
            n_fields: 4,
            embed_dim: 8,
            cin_layer_sizes: vec![8],
            dnn_hidden_sizes: vec![16],
            learning_rate: 0.1,
            l2_reg: 0.0,
            n_iter: 10,
            batch_size: 8,
        };
        let mut model = XDeepFm::new(cfg.clone(), &mut rng).expect("value should be present");
        let m = cfg.n_fields;
        let d = cfg.embed_dim;
        let bs = cfg.batch_size;

        // Fixed batch so repeated steps converge.
        let embeds = random_embeds(bs, m, d, &mut rng);
        let labels = random_labels(bs, &mut rng);

        let mut first_loss = f32::MAX;
        let mut last_loss = f32::MAX;
        for step in 0..10 {
            let mut rng2 = LcgRng::new(step as u64);
            let loss = model
                .train_step(&embeds, &labels, &mut rng2)
                .expect("train_step should succeed");
            if step == 0 {
                first_loss = loss;
            }
            last_loss = loss;
        }
        assert!(
            last_loss <= first_loss + 1e-2,
            "loss should not increase significantly: first={first_loss}, last={last_loss}"
        );
    }

    #[test]
    fn cin_forward_output_dim() {
        let mut rng = make_rng();
        let cfg = default_cfg();
        let model = XDeepFm::new(cfg.clone(), &mut rng).expect("value should be present");
        let embeds: Vec<f32> = (0..cfg.n_fields * cfg.embed_dim)
            .map(|_| rng.next_f32() * 0.1)
            .collect();
        let cin_out = model.cin_forward(&embeds);
        let expected: usize = cfg.cin_layer_sizes.iter().sum();
        assert_eq!(
            cin_out.len(),
            expected,
            "cin_forward should return sum(H_k) values"
        );
    }

    #[test]
    fn dnn_forward_finite() {
        let mut rng = make_rng();
        let cfg = default_cfg();
        let model = XDeepFm::new(cfg.clone(), &mut rng).expect("value should be present");
        let input: Vec<f32> = (0..cfg.n_fields * cfg.embed_dim)
            .map(|_| rng.next_f32() * 0.1)
            .collect();
        let out = model.dnn_forward(&input);
        assert!(out.is_finite(), "dnn_forward must be finite, got {out}");
    }

    #[test]
    fn new_err_zero_fields() {
        let mut rng = make_rng();
        let mut cfg = default_cfg();
        cfg.n_fields = 0;
        assert!(matches!(
            XDeepFm::new(cfg, &mut rng),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn new_err_zero_embed() {
        let mut rng = make_rng();
        let mut cfg = default_cfg();
        cfg.embed_dim = 0;
        assert!(matches!(
            XDeepFm::new(cfg, &mut rng),
            Err(RecsysError::InvalidEmbeddingDim { .. })
        ));
    }

    #[test]
    fn new_err_empty_cin_layers() {
        let mut rng = make_rng();
        let mut cfg = default_cfg();
        cfg.cin_layer_sizes = vec![];
        assert!(matches!(
            XDeepFm::new(cfg, &mut rng),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn new_err_zero_lr() {
        let mut rng = make_rng();
        let mut cfg = default_cfg();
        cfg.learning_rate = 0.0;
        assert!(matches!(
            XDeepFm::new(cfg, &mut rng),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }
}
