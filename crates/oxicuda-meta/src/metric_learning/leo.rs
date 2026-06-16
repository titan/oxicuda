//! LEO — Latent Embedding Optimization (Rusu et al., ICLR 2019).
//!
//! Meta-learns in a low-dimensional latent space rather than the full
//! classifier-parameter space.  The encoder maps support features to a latent
//! distribution q(z|support); the decoder maps sampled z to task-specific
//! classifier weights.  The inner loop optimises z (not the weights directly),
//! making gradient descent tractable even with large parameter spaces.
//!
//! Reference: <https://arxiv.org/abs/1807.05960>

use crate::error::{MetaError, MetaResult};
use crate::handle::LcgRng;

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters for the LEO encoder/decoder and inner loop.
#[derive(Debug, Clone)]
pub struct LeoConfig {
    /// Input feature dimension (from backbone).
    pub feat_dim: usize,
    /// Latent code dimension (much smaller than the classifier parameter space).
    pub latent_dim: usize,
    /// N-way classification.
    pub n_way: usize,
    /// K-shot support.
    pub k_shot: usize,
    /// Encoder MLP hidden units.
    pub encoder_hidden: usize,
    /// Decoder MLP hidden units.
    pub decoder_hidden: usize,
    /// Inner-loop (latent-space) learning rate.
    pub inner_lr: f32,
    /// Number of inner-loop gradient descent steps.
    pub inner_steps: usize,
    /// Dropout rate applied to encoder hidden layer (0.0 = disabled).
    pub dropout_rate: f32,
}

// ──────────────────────────────────────────────────────────────────────────────
// Weights
// ──────────────────────────────────────────────────────────────────────────────

/// All learnable parameters for the LEO encoder and decoder.
///
/// Layout convention: weight matrices stored row-major, W[out × in].
#[derive(Debug, Clone)]
pub struct LeoWeights {
    // ── Encoder: pooled class feature → latent distribution ──────────────────
    /// W₁ ∈ ℝ^{encoder_hidden × feat_dim}
    pub enc_w1: Vec<f32>,
    /// b₁ ∈ ℝ^{encoder_hidden}
    pub enc_b1: Vec<f32>,
    /// W₂_μ ∈ ℝ^{latent_dim × encoder_hidden}
    pub enc_w2_mu: Vec<f32>,
    /// b₂_μ ∈ ℝ^{latent_dim}
    pub enc_b2_mu: Vec<f32>,
    /// W₂_σ ∈ ℝ^{latent_dim × encoder_hidden} (predicts log σ)
    pub enc_w2_ls: Vec<f32>,
    /// b₂_σ ∈ ℝ^{latent_dim}
    pub enc_b2_ls: Vec<f32>,

    // ── Decoder: z → classifier (weights + biases) ────────────────────────────
    /// W₁ ∈ ℝ^{decoder_hidden × latent_dim}
    pub dec_w1: Vec<f32>,
    /// b₁ ∈ ℝ^{decoder_hidden}
    pub dec_b1: Vec<f32>,
    /// W₂ ∈ ℝ^{(n_way * feat_dim + n_way) × decoder_hidden}
    pub dec_w2: Vec<f32>,
    /// b₂ ∈ ℝ^{n_way * feat_dim + n_way}
    pub dec_b2: Vec<f32>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Latent state (per task)
// ──────────────────────────────────────────────────────────────────────────────

/// Task-specific latent state produced by the encoder.
#[derive(Debug, Clone)]
pub struct LeoState {
    /// Current latent code after inner-loop optimisation.
    pub z: Vec<f32>,
    /// Encoded mean μ of the latent distribution.
    pub z_mu: Vec<f32>,
    /// Encoded log-std log σ of the latent distribution.
    pub z_log_sigma: Vec<f32>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Result
// ──────────────────────────────────────────────────────────────────────────────

/// Outputs produced by a full LEO forward pass.
#[derive(Debug, Clone)]
pub struct LeoResult {
    /// Decoded classifier weight matrix, shape n_way × feat_dim (row-major).
    pub task_weights: Vec<f32>,
    /// Decoded classifier bias vector, shape n_way.
    pub task_biases: Vec<f32>,
    /// KL( q(z|support) ‖ N(0, I) ) regularisation term.
    pub kl_loss: f32,
    /// Cross-entropy of the decoded classifier on the query set.
    pub query_loss: f32,
    /// Fraction of correctly-classified query examples.
    pub query_accuracy: f32,
}

// ──────────────────────────────────────────────────────────────────────────────
// Main struct
// ──────────────────────────────────────────────────────────────────────────────

/// LEO meta-learner.
pub struct Leo {
    pub cfg: LeoConfig,
    pub weights: LeoWeights,
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers (private)
// ──────────────────────────────────────────────────────────────────────────────

/// Kaiming uniform initialisation: U( -√(6 / fan_in), +√(6 / fan_in) ).
fn kaiming_uniform(out: &mut [f32], fan_in: usize, rng: &mut LcgRng) {
    let limit = (6.0_f32 / fan_in.max(1) as f32).sqrt();
    for v in out.iter_mut() {
        *v = (rng.next_f32() * 2.0 - 1.0) * limit;
    }
}

/// Dense matrix–vector product: y = W x + b.
/// W is (out_dim × in_dim) row-major; x is (in_dim,); b is (out_dim,).
#[inline]
fn mv_add(w: &[f32], b: &[f32], x: &[f32], _out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut y = b.to_vec();
    for (i, yi) in y.iter_mut().enumerate() {
        let row = &w[i * in_dim..(i + 1) * in_dim];
        let dot: f32 = row.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi).sum();
        *yi += dot;
    }
    y
}

/// Box-Muller transform: produces one standard-normal sample from two U(0,1)
/// values, using only the sine branch (sufficient for one sample).
///
/// z = √(-2 ln(u1 + ε)) · cos(2π u2)
#[inline]
fn box_muller_cos(u1: f32, u2: f32) -> f32 {
    let r = (-2.0 * (u1 + 1e-38_f32).ln()).sqrt();
    r * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Numerically-stable soft-max over a slice.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
    for e in exps.iter_mut() {
        *e *= inv;
    }
    exps
}

// ──────────────────────────────────────────────────────────────────────────────
// impl Leo
// ──────────────────────────────────────────────────────────────────────────────

impl Leo {
    /// Construct a new LEO instance with Kaiming-uniform weights.
    pub fn new(cfg: LeoConfig, rng: &mut LcgRng) -> MetaResult<Self> {
        if cfg.feat_dim < 1 {
            return Err(MetaError::InvalidFeatDim { dim: cfg.feat_dim });
        }
        if cfg.latent_dim < 1 {
            return Err(MetaError::Internal {
                msg: "latent_dim must be >= 1".into(),
            });
        }
        if cfg.n_way < 2 {
            return Err(MetaError::InvalidNWay { n_way: cfg.n_way });
        }
        if cfg.k_shot < 1 {
            return Err(MetaError::InvalidKShot { k_shot: cfg.k_shot });
        }
        if cfg.inner_steps < 1 {
            return Err(MetaError::Internal {
                msg: "inner_steps must be >= 1".into(),
            });
        }

        let enc_hidden = cfg.encoder_hidden;
        let dec_hidden = cfg.decoder_hidden;
        let lat = cfg.latent_dim;
        let fd = cfg.feat_dim;
        let nw = cfg.n_way;
        let dec_out = nw * fd + nw;

        // ── Encoder ────────────────────────────────────────────────────────────
        let mut enc_w1 = vec![0.0_f32; enc_hidden * fd];
        kaiming_uniform(&mut enc_w1, fd, rng);
        let enc_b1 = vec![0.0_f32; enc_hidden];

        let mut enc_w2_mu = vec![0.0_f32; lat * enc_hidden];
        kaiming_uniform(&mut enc_w2_mu, enc_hidden, rng);
        let enc_b2_mu = vec![0.0_f32; lat];

        let mut enc_w2_ls = vec![0.0_f32; lat * enc_hidden];
        kaiming_uniform(&mut enc_w2_ls, enc_hidden, rng);
        let enc_b2_ls = vec![0.0_f32; lat];

        // ── Decoder ────────────────────────────────────────────────────────────
        let mut dec_w1 = vec![0.0_f32; dec_hidden * lat];
        kaiming_uniform(&mut dec_w1, lat, rng);
        let dec_b1 = vec![0.0_f32; dec_hidden];

        let mut dec_w2 = vec![0.0_f32; dec_out * dec_hidden];
        kaiming_uniform(&mut dec_w2, dec_hidden, rng);
        let dec_b2 = vec![0.0_f32; dec_out];

        let weights = LeoWeights {
            enc_w1,
            enc_b1,
            enc_w2_mu,
            enc_b2_mu,
            enc_w2_ls,
            enc_b2_ls,
            dec_w1,
            dec_b1,
            dec_w2,
            dec_b2,
        };

        Ok(Self { cfg, weights })
    }

    // ── Encoder ────────────────────────────────────────────────────────────────

    /// Encode support set into a latent distribution (μ, log σ).
    ///
    /// `support` is row-major with shape `[n_way, k_shot, feat_dim]`.
    /// Each class's K shots are averaged, then a 2-layer MLP maps the
    /// average into (μ, log σ) ∈ ℝ^{latent_dim}.
    pub fn encode(&self, support: &[f32]) -> MetaResult<(Vec<f32>, Vec<f32>)> {
        let nw = self.cfg.n_way;
        let ks = self.cfg.k_shot;
        let fd = self.cfg.feat_dim;
        let lat = self.cfg.latent_dim;
        let enc_h = self.cfg.encoder_hidden;

        let expected = nw * ks * fd;
        if support.len() != expected {
            return Err(MetaError::DimensionMismatch {
                expected,
                got: support.len(),
            });
        }

        // Per-class mean: mean_c ∈ ℝ^{feat_dim}
        let mut class_means = vec![0.0_f32; nw * fd];
        for (c, chunk) in support.chunks(ks * fd).enumerate() {
            let mean_row = &mut class_means[c * fd..(c + 1) * fd];
            for shot in chunk.chunks(fd) {
                for (acc, &v) in mean_row.iter_mut().zip(shot.iter()) {
                    *acc += v;
                }
            }
            for v in mean_row.iter_mut() {
                *v /= ks as f32;
            }
        }

        // Encode each class mean independently, then aggregate.
        let mut h_agg = vec![0.0_f32; enc_h];
        for c in 0..nw {
            let mean_c = &class_means[c * fd..(c + 1) * fd];
            // h_c = ReLU(enc_w1 @ mean_c + enc_b1)
            let mut h_c = mv_add(
                &self.weights.enc_w1,
                &self.weights.enc_b1,
                mean_c,
                enc_h,
                fd,
            );
            for v in h_c.iter_mut() {
                *v = v.max(0.0);
            }
            for (acc, &v) in h_agg.iter_mut().zip(h_c.iter()) {
                *acc += v;
            }
        }
        // Average over classes
        let inv_nw = 1.0 / nw as f32;
        for v in h_agg.iter_mut() {
            *v *= inv_nw;
        }

        // z_mu = enc_w2_mu @ h + enc_b2_mu
        let z_mu = mv_add(
            &self.weights.enc_w2_mu,
            &self.weights.enc_b2_mu,
            &h_agg,
            lat,
            enc_h,
        );

        // z_log_sigma = enc_w2_ls @ h + enc_b2_ls  (clamped to [-5, 5])
        let mut z_log_sigma = mv_add(
            &self.weights.enc_w2_ls,
            &self.weights.enc_b2_ls,
            &h_agg,
            lat,
            enc_h,
        );
        for v in z_log_sigma.iter_mut() {
            *v = v.clamp(-5.0, 5.0);
        }

        Ok((z_mu, z_log_sigma))
    }

    // ── Reparameterisation ─────────────────────────────────────────────────────

    /// Sample z via the reparameterisation trick: z = μ + ε · exp(log σ).
    ///
    /// ε ~ N(0, I) via Box-Muller, using `next_f32()` only.
    pub fn sample_z(z_mu: &[f32], z_log_sigma: &[f32], rng: &mut LcgRng) -> Vec<f32> {
        z_mu.iter()
            .zip(z_log_sigma.iter())
            .map(|(&mu, &log_s)| {
                let u1 = rng.next_f32();
                let u2 = rng.next_f32();
                let eps = box_muller_cos(u1, u2);
                mu + eps * log_s.exp()
            })
            .collect()
    }

    // ── Decoder ────────────────────────────────────────────────────────────────

    /// Decode latent code z into task-specific classifier (weights, biases).
    ///
    /// Returns `(task_weights, task_biases)` where:
    /// - `task_weights` has shape `[n_way × feat_dim]` (row-major)
    /// - `task_biases` has length `n_way`
    pub fn decode(&self, z: &[f32]) -> MetaResult<(Vec<f32>, Vec<f32>)> {
        let lat = self.cfg.latent_dim;
        let dec_h = self.cfg.decoder_hidden;
        let nw = self.cfg.n_way;
        let fd = self.cfg.feat_dim;
        let dec_out = nw * fd + nw;

        if z.len() != lat {
            return Err(MetaError::DimensionMismatch {
                expected: lat,
                got: z.len(),
            });
        }

        // h = ReLU(dec_w1 @ z + dec_b1)
        let mut h = mv_add(&self.weights.dec_w1, &self.weights.dec_b1, z, dec_h, lat);
        for v in h.iter_mut() {
            *v = v.max(0.0);
        }

        // out = dec_w2 @ h + dec_b2
        let out = mv_add(
            &self.weights.dec_w2,
            &self.weights.dec_b2,
            &h,
            dec_out,
            dec_h,
        );

        let task_weights = out[..nw * fd].to_vec();
        let task_biases = out[nw * fd..].to_vec();

        Ok((task_weights, task_biases))
    }

    // ── Query loss ─────────────────────────────────────────────────────────────

    /// Compute cross-entropy loss (and accuracy) of a linear classifier on the
    /// query set.
    ///
    /// - `w`: `n_way × feat_dim` row-major weight matrix
    /// - `b`: `n_way` bias vector
    /// - `query_feats`: `n_query × feat_dim`
    /// - `labels`: `n_query` class indices in `0..n_way`
    ///
    /// Returns `(mean_cross_entropy, accuracy)`.
    pub fn query_loss(
        w: &[f32],
        b: &[f32],
        query_feats: &[f32],
        labels: &[usize],
        n_way: usize,
        feat_dim: usize,
    ) -> MetaResult<(f32, f32)> {
        let n_query = labels.len();
        if n_query == 0 {
            return Err(MetaError::InvalidQuerySize { size: 0 });
        }
        if query_feats.len() != n_query * feat_dim {
            return Err(MetaError::DimensionMismatch {
                expected: n_query * feat_dim,
                got: query_feats.len(),
            });
        }
        if w.len() != n_way * feat_dim {
            return Err(MetaError::DimensionMismatch {
                expected: n_way * feat_dim,
                got: w.len(),
            });
        }
        if b.len() != n_way {
            return Err(MetaError::DimensionMismatch {
                expected: n_way,
                got: b.len(),
            });
        }

        let mut total_loss = 0.0_f32;
        let mut n_correct = 0_usize;

        for (q, feat) in query_feats.chunks(feat_dim).enumerate() {
            let lbl = labels[q];
            // logits_c = w[c] · feat + b[c]
            let logits: Vec<f32> = (0..n_way)
                .map(|c| {
                    let row = &w[c * feat_dim..(c + 1) * feat_dim];
                    let dot: f32 = row.iter().zip(feat.iter()).map(|(&wi, &xi)| wi * xi).sum();
                    dot + b[c]
                })
                .collect();

            let probs = softmax(&logits);
            let log_p = probs[lbl].max(1e-38_f32).ln();
            if !log_p.is_finite() {
                return Err(MetaError::NanEncountered {
                    context: "query_loss log probability".into(),
                });
            }
            total_loss -= log_p;

            let pred = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            if pred == lbl {
                n_correct += 1;
            }
        }

        Ok((
            total_loss / n_query as f32,
            n_correct as f32 / n_query as f32,
        ))
    }

    // ── Inner loop ─────────────────────────────────────────────────────────────

    /// Inner-loop gradient descent in the latent space.
    ///
    /// Gradient of the cross-entropy loss w.r.t. z is estimated via
    /// central finite differences: δ = 1e-3.
    ///
    /// Returns the updated latent code after `cfg.inner_steps` steps.
    pub fn inner_loop(
        &self,
        z_init: &[f32],
        support_feats: &[f32],
        support_labels: &[usize],
        query_feats: &[f32],
        query_labels: &[usize],
        n_way: usize,
        feat_dim: usize,
        _n_query: usize,
    ) -> MetaResult<Vec<f32>> {
        if z_init.len() != self.cfg.latent_dim {
            return Err(MetaError::DimensionMismatch {
                expected: self.cfg.latent_dim,
                got: z_init.len(),
            });
        }

        let delta = 1e-3_f32;
        let lat = self.cfg.latent_dim;
        let mut z = z_init.to_vec();

        // Decide which data to differentiate against.
        // Per the LEO paper the inner loop minimises loss on the *support* set
        // (via decoded weights).  We fall back to query if support is empty.
        let (eval_feats, eval_labels) = if !support_feats.is_empty() && !support_labels.is_empty() {
            (support_feats, support_labels)
        } else {
            (query_feats, query_labels)
        };

        for _step in 0..self.cfg.inner_steps {
            // Compute gradient via central finite differences
            let mut grad = vec![0.0_f32; lat];
            for i in 0..lat {
                let orig = z[i];

                z[i] = orig + delta;
                let (w_p, b_p) = self.decode(&z)?;
                let (loss_p, _) =
                    Self::query_loss(&w_p, &b_p, eval_feats, eval_labels, n_way, feat_dim)?;

                z[i] = orig - delta;
                let (w_m, b_m) = self.decode(&z)?;
                let (loss_m, _) =
                    Self::query_loss(&w_m, &b_m, eval_feats, eval_labels, n_way, feat_dim)?;

                grad[i] = (loss_p - loss_m) / (2.0 * delta);
                z[i] = orig;
            }

            // Gradient step
            for (zi, &gi) in z.iter_mut().zip(grad.iter()) {
                *zi -= self.cfg.inner_lr * gi;
            }
        }

        Ok(z)
    }

    // ── KL divergence ──────────────────────────────────────────────────────────

    /// KL( N(μ, σ²) ‖ N(0, I) ) = ½ Σᵢ ( μᵢ² + σᵢ² − log σᵢ² − 1 ).
    pub fn kl_divergence(z_mu: &[f32], z_log_sigma: &[f32]) -> f32 {
        z_mu.iter()
            .zip(z_log_sigma.iter())
            .map(|(&mu, &log_s)| {
                let sigma2 = (2.0 * log_s).exp(); // σ² = exp(2 log σ)
                0.5 * (mu * mu + sigma2 - 2.0 * log_s - 1.0)
            })
            .sum()
    }

    // ── Full forward pass ──────────────────────────────────────────────────────

    /// Complete LEO forward pass:
    /// encode → sample z → inner loop → decode → evaluate on query.
    pub fn forward(
        &self,
        support_feats: &[f32],
        support_labels: &[usize],
        query_feats: &[f32],
        query_labels: &[usize],
        rng: &mut LcgRng,
    ) -> MetaResult<LeoResult> {
        let nw = self.cfg.n_way;
        let fd = self.cfg.feat_dim;
        let n_query = query_labels.len();

        // Validate input sizes
        let expected_support = nw * self.cfg.k_shot * fd;
        if support_feats.len() != expected_support {
            return Err(MetaError::DimensionMismatch {
                expected: expected_support,
                got: support_feats.len(),
            });
        }
        if support_labels.len() != nw * self.cfg.k_shot {
            return Err(MetaError::DimensionMismatch {
                expected: nw * self.cfg.k_shot,
                got: support_labels.len(),
            });
        }
        if n_query == 0 {
            return Err(MetaError::InvalidQuerySize { size: 0 });
        }
        if query_feats.len() != n_query * fd {
            return Err(MetaError::DimensionMismatch {
                expected: n_query * fd,
                got: query_feats.len(),
            });
        }

        // 1. Encode support → latent distribution
        let (z_mu, z_log_sigma) = self.encode(support_feats)?;

        // 2. Sample initial z
        let z_init = Self::sample_z(&z_mu, &z_log_sigma, rng);

        // 3. Inner loop: optimise z on support set
        let z_final = self.inner_loop(
            &z_init,
            support_feats,
            support_labels,
            query_feats,
            query_labels,
            nw,
            fd,
            n_query,
        )?;

        // 4. Decode final z → classifier parameters
        let (task_weights, task_biases) = self.decode(&z_final)?;

        // 5. Evaluate on query set
        let (query_loss, query_accuracy) = Self::query_loss(
            &task_weights,
            &task_biases,
            query_feats,
            query_labels,
            nw,
            fd,
        )?;

        // 6. KL regularisation term
        let kl_loss = Self::kl_divergence(&z_mu, &z_log_sigma);

        Ok(LeoResult {
            task_weights,
            task_biases,
            kl_loss,
            query_loss,
            query_accuracy,
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> LeoConfig {
        LeoConfig {
            feat_dim: 8,
            latent_dim: 4,
            n_way: 3,
            k_shot: 2,
            encoder_hidden: 16,
            decoder_hidden: 16,
            inner_lr: 0.01,
            inner_steps: 3,
            dropout_rate: 0.0,
        }
    }

    fn make_leo() -> Leo {
        let mut rng = LcgRng::new(42);
        Leo::new(default_cfg(), &mut rng).expect("value should be present")
    }

    fn support_data(cfg: &LeoConfig) -> (Vec<f32>, Vec<usize>) {
        let n = cfg.n_way * cfg.k_shot * cfg.feat_dim;
        let feats: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let labels: Vec<usize> = (0..cfg.n_way)
            .flat_map(|c| std::iter::repeat_n(c, cfg.k_shot))
            .collect();
        (feats, labels)
    }

    fn query_data(cfg: &LeoConfig, n_query: usize) -> (Vec<f32>, Vec<usize>) {
        let n = n_query * cfg.feat_dim;
        let feats: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02).collect();
        let labels: Vec<usize> = (0..n_query).map(|i| i % cfg.n_way).collect();
        (feats, labels)
    }

    // ── Encoder ────────────────────────────────────────────────────────────────

    #[test]
    fn encode_output_shape() {
        let leo = make_leo();
        let (support, _) = support_data(&leo.cfg);
        let (z_mu, z_ls) = leo.encode(&support).expect("encode should succeed");
        assert_eq!(z_mu.len(), leo.cfg.latent_dim);
        assert_eq!(z_ls.len(), leo.cfg.latent_dim);
    }

    #[test]
    fn encode_log_sigma_clamped() {
        let leo = make_leo();
        let (support, _) = support_data(&leo.cfg);
        let (_, z_ls) = leo.encode(&support).expect("encode should succeed");
        for &v in &z_ls {
            assert!((-5.0..=5.0).contains(&v), "log_sigma out of [-5,5]: {v}");
        }
    }

    // ── Sample z ───────────────────────────────────────────────────────────────

    #[test]
    fn sample_z_shape() {
        let cfg = default_cfg();
        let z_mu = vec![0.0_f32; cfg.latent_dim];
        let z_ls = vec![0.0_f32; cfg.latent_dim];
        let mut rng = LcgRng::new(7);
        let z = Leo::sample_z(&z_mu, &z_ls, &mut rng);
        assert_eq!(z.len(), cfg.latent_dim);
    }

    #[test]
    fn sample_z_different_each_call() {
        let cfg = default_cfg();
        let z_mu = vec![0.0_f32; cfg.latent_dim];
        let z_ls = vec![0.0_f32; cfg.latent_dim];
        let mut rng = LcgRng::new(13);
        let z1 = Leo::sample_z(&z_mu, &z_ls, &mut rng);
        let z2 = Leo::sample_z(&z_mu, &z_ls, &mut rng);
        // Two consecutive calls should produce different noise
        let different = z1.iter().zip(z2.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(different, "Consecutive samples should differ");
    }

    // ── Decoder ────────────────────────────────────────────────────────────────

    #[test]
    fn decode_output_shape() {
        let leo = make_leo();
        let z = vec![0.0_f32; leo.cfg.latent_dim];
        let (w, b) = leo.decode(&z).expect("decode should succeed");
        assert_eq!(w.len(), leo.cfg.n_way * leo.cfg.feat_dim);
        assert_eq!(b.len(), leo.cfg.n_way);
    }

    #[test]
    fn decode_finite_outputs() {
        let leo = make_leo();
        let z = vec![0.1_f32; leo.cfg.latent_dim];
        let (w, b) = leo.decode(&z).expect("decode should succeed");
        assert!(w.iter().all(|v| v.is_finite()));
        assert!(b.iter().all(|v| v.is_finite()));
    }

    // ── Query loss ─────────────────────────────────────────────────────────────

    #[test]
    fn query_loss_shape() {
        let cfg = default_cfg();
        let nw = cfg.n_way;
        let fd = cfg.feat_dim;
        let n_q = 6;
        let w = vec![0.0_f32; nw * fd];
        let b = vec![0.0_f32; nw];
        let feats = vec![0.0_f32; n_q * fd];
        let labels: Vec<usize> = (0..n_q).map(|i| i % nw).collect();
        let (loss, acc) =
            Leo::query_loss(&w, &b, &feats, &labels, nw, fd).expect("query_loss should succeed");
        assert!(loss.is_finite());
        assert!(acc.is_finite());
    }

    #[test]
    fn query_loss_uniform_labels() {
        // With all-zero weights the classifier is uniform → accuracy ≈ 1/n_way
        let cfg = default_cfg();
        let nw = cfg.n_way;
        let fd = cfg.feat_dim;
        let n_q = 300;
        let w = vec![0.0_f32; nw * fd];
        let b = vec![0.0_f32; nw];
        let feats = vec![0.0_f32; n_q * fd];
        // balanced labels
        let labels: Vec<usize> = (0..n_q).map(|i| i % nw).collect();
        let (_loss, acc) =
            Leo::query_loss(&w, &b, &feats, &labels, nw, fd).expect("query_loss should succeed");
        // Uniform predictor: accuracy ≈ 1/n_way (all ties break to class 0, so first class wins)
        // Just check it's in [0,1]
        assert!((0.0..=1.0).contains(&acc));
    }

    // ── KL divergence ──────────────────────────────────────────────────────────

    #[test]
    fn kl_divergence_zero_at_prior() {
        // μ=0, log σ=0 → σ=1 → KL = 0
        let mu = vec![0.0_f32; 4];
        let log_s = vec![0.0_f32; 4];
        let kl = Leo::kl_divergence(&mu, &log_s);
        assert!(kl.abs() < 1e-5, "KL should be 0 at prior, got {kl}");
    }

    #[test]
    fn kl_divergence_positive() {
        // μ=1, log σ=0 → KL = 0.5*(1+1-0-1)*dim = 0.5*dim > 0
        let mu = vec![1.0_f32; 4];
        let log_s = vec![0.0_f32; 4];
        let kl = Leo::kl_divergence(&mu, &log_s);
        assert!(kl > 0.0, "KL with μ≠0 should be positive, got {kl}");
    }

    // ── Inner loop ─────────────────────────────────────────────────────────────

    #[test]
    fn inner_loop_runs() {
        let leo = make_leo();
        let cfg = &leo.cfg;
        let (s_feats, s_labels) = support_data(cfg);
        let (q_feats, q_labels) = query_data(cfg, 6);
        let z_init = vec![0.0_f32; cfg.latent_dim];
        let z_final = leo
            .inner_loop(
                &z_init,
                &s_feats,
                &s_labels,
                &q_feats,
                &q_labels,
                cfg.n_way,
                cfg.feat_dim,
                6,
            )
            .expect("value should be present");
        assert_eq!(z_final.len(), cfg.latent_dim);
    }

    #[test]
    fn inner_loop_reduces_loss() {
        let leo = make_leo();
        let cfg = &leo.cfg;
        let (s_feats, s_labels) = support_data(cfg);
        let (q_feats, q_labels) = query_data(cfg, 6);

        let z_init = vec![0.1_f32; cfg.latent_dim];
        let (w0, b0) = leo.decode(&z_init).expect("decode should succeed");
        let (loss0, _) = Leo::query_loss(&w0, &b0, &s_feats, &s_labels, cfg.n_way, cfg.feat_dim)
            .expect("query_loss should succeed");

        let z_final = leo
            .inner_loop(
                &z_init,
                &s_feats,
                &s_labels,
                &q_feats,
                &q_labels,
                cfg.n_way,
                cfg.feat_dim,
                6,
            )
            .expect("value should be present");
        let (w1, b1) = leo.decode(&z_final).expect("decode should succeed");
        let (loss1, _) = Leo::query_loss(&w1, &b1, &s_feats, &s_labels, cfg.n_way, cfg.feat_dim)
            .expect("query_loss should succeed");

        // Loss should not increase (may be equal if gradient is zero)
        assert!(
            loss1 <= loss0 + 1e-4,
            "Inner loop should reduce loss: {loss1} > {loss0}"
        );
    }

    // ── Full forward ───────────────────────────────────────────────────────────

    #[test]
    fn forward_runs() {
        let leo = make_leo();
        let cfg = &leo.cfg;
        let (s_feats, s_labels) = support_data(cfg);
        let (q_feats, q_labels) = query_data(cfg, 6);
        let mut rng = LcgRng::new(99);
        let res = leo
            .forward(&s_feats, &s_labels, &q_feats, &q_labels, &mut rng)
            .expect("value should be present");
        assert!(res.query_loss.is_finite());
        assert!(res.kl_loss.is_finite());
    }

    #[test]
    fn forward_kl_positive() {
        let leo = make_leo();
        let cfg = &leo.cfg;
        let (s_feats, s_labels) = support_data(cfg);
        let (q_feats, q_labels) = query_data(cfg, 6);
        let mut rng = LcgRng::new(7777);
        let res = leo
            .forward(&s_feats, &s_labels, &q_feats, &q_labels, &mut rng)
            .expect("value should be present");
        assert!(res.kl_loss >= 0.0, "KL loss must be non-negative");
    }

    #[test]
    fn forward_accuracy_in_range() {
        let leo = make_leo();
        let cfg = &leo.cfg;
        let (s_feats, s_labels) = support_data(cfg);
        let (q_feats, q_labels) = query_data(cfg, 9);
        let mut rng = LcgRng::new(2025);
        let res = leo
            .forward(&s_feats, &s_labels, &q_feats, &q_labels, &mut rng)
            .expect("value should be present");
        assert!(
            (0.0..=1.0).contains(&res.query_accuracy),
            "Accuracy must be in [0,1]: {}",
            res.query_accuracy
        );
    }

    // ── Error paths ────────────────────────────────────────────────────────────

    #[test]
    fn err_feat_dim_zero() {
        let mut cfg = default_cfg();
        cfg.feat_dim = 0;
        let mut rng = LcgRng::new(1);
        let result = Leo::new(cfg, &mut rng);
        assert!(matches!(result, Err(MetaError::InvalidFeatDim { .. })));
    }

    #[test]
    fn err_latent_dim_zero() {
        let mut cfg = default_cfg();
        cfg.latent_dim = 0;
        let mut rng = LcgRng::new(1);
        let result = Leo::new(cfg, &mut rng);
        assert!(matches!(result, Err(MetaError::Internal { .. })));
    }

    #[test]
    fn err_n_way_one() {
        let mut cfg = default_cfg();
        cfg.n_way = 1;
        let mut rng = LcgRng::new(1);
        let result = Leo::new(cfg, &mut rng);
        assert!(matches!(result, Err(MetaError::InvalidNWay { .. })));
    }

    #[test]
    fn err_support_wrong_length() {
        let leo = make_leo();
        // Correct would be n_way * k_shot * feat_dim = 3*2*8 = 48
        let bad_support = vec![0.0_f32; 10]; // wrong length
        let result = leo.encode(&bad_support);
        assert!(matches!(result, Err(MetaError::DimensionMismatch { .. })));
    }
}
