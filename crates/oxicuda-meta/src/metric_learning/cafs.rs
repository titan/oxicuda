//! Cross-Attention Few-Shot (CAFS) classifier (Ye et al. 2021).
//!
//! CAFS enhances prototypical classification by applying cross-attention
//! transformer layers between query features and class-mean prototypes,
//! allowing prototypes to be dynamically refined based on the query distribution.
//!
//! Reference: Ye et al., "Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions",
//! CVPR 2020 / CAFS variant ECCV 2021.

use crate::error::{MetaError, MetaResult};
use crate::handle::LcgRng;

/// Configuration for a [`CafsFewShot`] model.
#[derive(Debug, Clone)]
pub struct CafsConfig {
    /// Feature dimensionality (must be > 0 and divisible by `n_heads`).
    pub d_feat: usize,
    /// Number of attention heads (>= 1).
    pub n_heads: usize,
    /// Number of cross-attention transformer layers (>= 1).
    pub n_layers: usize,
    /// Feedforward hidden dimension inside each transformer block (> 0).
    pub d_ff: usize,
}

/// Cross-Attention Few-Shot network.
///
/// Applies `n_layers` of multi-head cross-attention (query tokens attend to
/// support prototype key/value pairs) plus a position-wise feedforward block,
/// each wrapped with a residual connection and layer normalization.
#[derive(Debug, Clone)]
pub struct CafsFewShot {
    /// Query projection weights, one matrix per layer: `[d_feat × d_feat]` row-major.
    pub q_w: Vec<Vec<f32>>,
    /// Key projection weights, one matrix per layer.
    pub k_w: Vec<Vec<f32>>,
    /// Value projection weights, one matrix per layer.
    pub v_w: Vec<Vec<f32>>,
    /// Output projection weights, one matrix per layer.
    pub out_w: Vec<Vec<f32>>,
    /// Layer-norm scale γ per layer (length `d_feat` each).
    pub ln_scale: Vec<Vec<f32>>,
    /// Layer-norm bias β per layer (length `d_feat` each).
    pub ln_bias: Vec<Vec<f32>>,
    /// FF layer-1 weights per layer: `[d_ff × d_feat]` row-major.
    pub ff_w1: Vec<Vec<f32>>,
    /// FF layer-1 bias per layer: length `d_ff`.
    pub ff_b1: Vec<Vec<f32>>,
    /// FF layer-2 weights per layer: `[d_feat × d_ff]` row-major.
    pub ff_w2: Vec<Vec<f32>>,
    /// FF layer-2 bias per layer: length `d_feat`.
    pub ff_b2: Vec<Vec<f32>>,
    /// Model configuration.
    pub config: CafsConfig,
}

impl CafsFewShot {
    /// Construct a new `CafsFewShot` with Kaiming-initialized projection weights,
    /// ones for ln_scale, and zeros for ln_bias.
    ///
    /// # Errors
    ///
    /// Returns [`MetaError::InvalidFeatDim`] if `d_feat == 0` or `d_ff == 0`.
    /// Returns [`MetaError::InvalidEpisodeConfig`] if `n_heads == 0`, `n_layers == 0`,
    /// or `d_feat % n_heads != 0`.
    pub fn new(config: CafsConfig, rng: &mut LcgRng) -> MetaResult<Self> {
        if config.d_feat == 0 {
            return Err(MetaError::InvalidFeatDim { dim: 0 });
        }
        if config.d_ff == 0 {
            return Err(MetaError::InvalidFeatDim { dim: 0 });
        }
        if config.n_heads == 0 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "n_heads must be >= 1".into(),
            });
        }
        if config.n_layers == 0 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "n_layers must be >= 1".into(),
            });
        }
        if !config.d_feat.is_multiple_of(config.n_heads) {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: format!(
                    "d_feat ({}) must be divisible by n_heads ({})",
                    config.d_feat, config.n_heads
                ),
            });
        }

        let d = config.d_feat;
        let d_ff = config.d_ff;
        let nl = config.n_layers;

        // Kaiming scale for d×d matrices: sqrt(2/d)
        let scale_dd = (2.0_f32 / d as f32).sqrt();
        let scale_ff1 = (2.0_f32 / d as f32).sqrt();
        let scale_ff2 = (2.0_f32 / d_ff as f32).sqrt();

        let mut make_mat = |rows: usize, cols: usize, scale: f32| -> Vec<f32> {
            (0..rows * cols)
                .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
                .collect()
        };

        let mut q_w = Vec::with_capacity(nl);
        let mut k_w = Vec::with_capacity(nl);
        let mut v_w = Vec::with_capacity(nl);
        let mut out_w = Vec::with_capacity(nl);
        let mut ln_scale = Vec::with_capacity(nl);
        let mut ln_bias = Vec::with_capacity(nl);
        let mut ff_w1 = Vec::with_capacity(nl);
        let mut ff_b1 = Vec::with_capacity(nl);
        let mut ff_w2 = Vec::with_capacity(nl);
        let mut ff_b2 = Vec::with_capacity(nl);

        for _ in 0..nl {
            q_w.push(make_mat(d, d, scale_dd));
            k_w.push(make_mat(d, d, scale_dd));
            v_w.push(make_mat(d, d, scale_dd));
            out_w.push(make_mat(d, d, scale_dd));
            ln_scale.push(vec![1.0_f32; d]);
            ln_bias.push(vec![0.0_f32; d]);
            ff_w1.push(make_mat(d_ff, d, scale_ff1));
            ff_b1.push(vec![0.0_f32; d_ff]);
            ff_w2.push(make_mat(d, d_ff, scale_ff2));
            ff_b2.push(vec![0.0_f32; d]);
        }

        Ok(Self {
            q_w,
            k_w,
            v_w,
            out_w,
            ln_scale,
            ln_bias,
            ff_w1,
            ff_b1,
            ff_w2,
            ff_b2,
            config,
        })
    }

    /// Apply layer normalization: `(x - mean) / sqrt(var + eps) * scale + bias`.
    ///
    /// All inputs must have the same length as `scale` and `bias`.
    ///
    /// # Errors
    ///
    /// Returns [`MetaError::DimensionMismatch`] on size mismatch.
    /// Returns [`MetaError::NanEncountered`] if variance is non-finite.
    pub fn layer_norm(x: &[f32], scale: &[f32], bias: &[f32]) -> MetaResult<Vec<f32>> {
        let n = x.len();
        if scale.len() != n || bias.len() != n {
            return Err(MetaError::DimensionMismatch {
                expected: n,
                got: scale.len().min(bias.len()),
            });
        }
        let mean = x.iter().sum::<f32>() / n as f32;
        let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let std_inv = 1.0 / (var + 1e-5_f32).sqrt();
        if !std_inv.is_finite() {
            return Err(MetaError::NanEncountered {
                context: "layer_norm: std_inv is non-finite".into(),
            });
        }
        Ok(x.iter()
            .zip(scale.iter().zip(bias.iter()))
            .map(|(&xi, (&s, &b))| (xi - mean) * std_inv * s + b)
            .collect())
    }

    /// Apply one multi-head cross-attention + FFN layer with residuals and layer norms.
    ///
    /// `queries`: flat `[n_q × d_feat]`, queries attending to key-value pairs.
    /// `kv`: flat `[n_kv × d_feat]`, the key/value source.
    ///
    /// Returns updated query representations: flat `[n_q × d_feat]`.
    ///
    /// # Errors
    ///
    /// Returns [`MetaError`] on dimension mismatch or non-finite computation.
    pub fn attention_layer(
        &self,
        layer: usize,
        queries: &[f32],
        n_q: usize,
        kv: &[f32],
        n_kv: usize,
    ) -> MetaResult<Vec<f32>> {
        let d = self.config.d_feat;
        let h = self.config.n_heads;
        let d_h = d / h;

        if queries.len() != n_q * d {
            return Err(MetaError::DimensionMismatch {
                expected: n_q * d,
                got: queries.len(),
            });
        }
        if kv.len() != n_kv * d {
            return Err(MetaError::DimensionMismatch {
                expected: n_kv * d,
                got: kv.len(),
            });
        }

        let qw = &self.q_w[layer];
        let kw = &self.k_w[layer];
        let vw = &self.v_w[layer];
        let ow = &self.out_w[layer];
        let ls = &self.ln_scale[layer];
        let lb = &self.ln_bias[layer];
        let fw1 = &self.ff_w1[layer];
        let fb1 = &self.ff_b1[layer];
        let fw2 = &self.ff_w2[layer];
        let fb2 = &self.ff_b2[layer];
        let d_ff = self.config.d_ff;

        // Linear projection helper: mat [rows × cols], vec [cols] -> vec [rows]
        let matmul_vec = |mat: &[f32], rows: usize, cols: usize, vec: &[f32]| -> Vec<f32> {
            (0..rows)
                .map(|r| (0..cols).map(|c| mat[r * cols + c] * vec[c]).sum::<f32>())
                .collect()
        };

        // Project all queries, keys, values: [n × d]
        let mut q_proj = vec![0.0_f32; n_q * d];
        let mut k_proj = vec![0.0_f32; n_kv * d];
        let mut v_proj = vec![0.0_f32; n_kv * d];

        for i in 0..n_q {
            let qi = &queries[i * d..(i + 1) * d];
            let pq = matmul_vec(qw, d, d, qi);
            q_proj[i * d..(i + 1) * d].copy_from_slice(&pq);
        }
        for j in 0..n_kv {
            let kj = &kv[j * d..(j + 1) * d];
            let pk = matmul_vec(kw, d, d, kj);
            let pv = matmul_vec(vw, d, d, kj);
            k_proj[j * d..(j + 1) * d].copy_from_slice(&pk);
            v_proj[j * d..(j + 1) * d].copy_from_slice(&pv);
        }

        let scale = (d_h as f32).sqrt();

        // Multi-head attention output: [n_q × d]
        let mut attn_out = vec![0.0_f32; n_q * d];

        for head in 0..h {
            let head_start = head * d_h;
            let head_end = head_start + d_h;

            for i in 0..n_q {
                // Attention scores for query i: [n_kv]
                let qi_head = &q_proj[i * d + head_start..i * d + head_end];
                let mut scores = vec![0.0_f32; n_kv];
                for j in 0..n_kv {
                    let kj_head = &k_proj[j * d + head_start..j * d + head_end];
                    scores[j] = qi_head
                        .iter()
                        .zip(kj_head.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f32>()
                        / scale;
                }

                // Softmax over scores
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
                let sum_e: f32 = exps.iter().sum();
                if sum_e == 0.0 || !sum_e.is_finite() {
                    return Err(MetaError::NanEncountered {
                        context: "attention_layer: softmax sum is zero".into(),
                    });
                }
                let attn_weights: Vec<f32> = exps.iter().map(|&e| e / sum_e).collect();

                // Weighted sum of values
                let base = i * d + head_start;
                for j in 0..n_kv {
                    let vj_head = &v_proj[j * d + head_start..j * d + head_end];
                    for k_idx in 0..d_h {
                        attn_out[base + k_idx] += attn_weights[j] * vj_head[k_idx];
                    }
                }
            }
        }

        // Output projection + residual + layer norm
        let mut out1 = vec![0.0_f32; n_q * d];
        for i in 0..n_q {
            let ao = &attn_out[i * d..(i + 1) * d];
            let proj = matmul_vec(ow, d, d, ao);
            // Residual: original query + projected attention output
            let qi = &queries[i * d..(i + 1) * d];
            let res: Vec<f32> = qi.iter().zip(proj.iter()).map(|(&a, &b)| a + b).collect();
            let ln = Self::layer_norm(&res, ls, lb)?;
            out1[i * d..(i + 1) * d].copy_from_slice(&ln);
        }

        // Position-wise FFN + residual + layer norm
        let mut out2 = vec![0.0_f32; n_q * d];
        for i in 0..n_q {
            let xi = &out1[i * d..(i + 1) * d];
            // FF layer 1: [d_ff × d], with ReLU
            let mut h1 = matmul_vec(fw1, d_ff, d, xi);
            for j in 0..d_ff {
                h1[j] += fb1[j];
                if h1[j] < 0.0 {
                    h1[j] = 0.0;
                }
            }
            // FF layer 2: [d × d_ff]
            let mut h2 = matmul_vec(fw2, d, d_ff, &h1);
            for j in 0..d {
                h2[j] += fb2[j];
            }
            // Residual + LN
            let res: Vec<f32> = xi.iter().zip(h2.iter()).map(|(&a, &b)| a + b).collect();
            let ln = Self::layer_norm(&res, ls, lb)?;
            out2[i * d..(i + 1) * d].copy_from_slice(&ln);
        }

        Ok(out2)
    }

    /// Compute adapted prototypes by cross-attending query features to class-mean
    /// support prototypes through all `n_layers` transformer layers.
    ///
    /// `support` shape: `[n_way * n_shot * d_feat]`.
    /// `query` shape: `[n_q * d_feat]`.
    ///
    /// Returns adapted prototypes: flat `[n_way * d_feat]`.
    ///
    /// # Errors
    ///
    /// Returns [`MetaError`] on shape mismatch or non-finite values.
    pub fn adapted_prototypes(
        &self,
        support: &[f32],
        query: &[f32],
        n_way: usize,
        n_shot: usize,
    ) -> MetaResult<Vec<f32>> {
        if n_way < 2 {
            return Err(MetaError::InvalidNWay { n_way });
        }
        if n_shot < 1 {
            return Err(MetaError::InvalidKShot { k_shot: n_shot });
        }
        let d = self.config.d_feat;
        let expected_s = n_way * n_shot * d;
        if support.len() != expected_s {
            return Err(MetaError::DimensionMismatch {
                expected: expected_s,
                got: support.len(),
            });
        }
        if query.is_empty() || !query.len().is_multiple_of(d) {
            return Err(MetaError::DimensionMismatch {
                expected: d,
                got: query.len() % d,
            });
        }
        let n_q = query.len() / d;

        // Compute class-mean prototypes: [n_way × d]
        let mut protos = vec![0.0_f32; n_way * d];
        for cls in 0..n_way {
            for shot in 0..n_shot {
                let start = (cls * n_shot + shot) * d;
                for j in 0..d {
                    protos[cls * d + j] += support[start + j];
                }
            }
            let inv = 1.0 / n_shot as f32;
            for j in 0..d {
                protos[cls * d + j] *= inv;
            }
        }

        // Iteratively apply cross-attention layers:
        // queries = prototypes (n_way tokens), kv = query features (n_q tokens)
        let mut current = protos;
        for layer in 0..self.config.n_layers {
            current = self.attention_layer(layer, &current, n_way, query, n_q)?;
        }

        Ok(current)
    }

    /// Predict the class of a single query example by argmin L2 distance
    /// to the adapted prototypes.
    ///
    /// `query` shape: `[d_feat]`.
    /// `prototypes` shape: `[n_way * d_feat]`.
    ///
    /// # Errors
    ///
    /// Returns [`MetaError::DimensionMismatch`] on size mismatch.
    /// Returns [`MetaError::InvalidNWay`] if `n_way < 2`.
    pub fn predict(&self, query: &[f32], prototypes: &[f32], n_way: usize) -> MetaResult<usize> {
        if n_way < 2 {
            return Err(MetaError::InvalidNWay { n_way });
        }
        let d = self.config.d_feat;
        if query.len() != d {
            return Err(MetaError::DimensionMismatch {
                expected: d,
                got: query.len(),
            });
        }
        if prototypes.len() != n_way * d {
            return Err(MetaError::DimensionMismatch {
                expected: n_way * d,
                got: prototypes.len(),
            });
        }

        let mut best_cls = 0usize;
        let mut best_dist = f32::INFINITY;
        for cls in 0..n_way {
            let proto = &prototypes[cls * d..(cls + 1) * d];
            let dist: f32 = query
                .iter()
                .zip(proto.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();
            if dist < best_dist {
                best_dist = dist;
                best_cls = cls;
            }
        }

        Ok(best_cls)
    }

    /// Compute episodic negative log-softmax loss using L2 distances to adapted prototypes.
    ///
    /// `support` shape: `[n_way * n_shot * d_feat]`.
    /// `queries` shape: `[n_q * d_feat]`.
    /// `query_labels` shape: `[n_q]`, values in `[0, n_way)`.
    ///
    /// # Errors
    ///
    /// Returns [`MetaError`] on any shape mismatch or numerical failure.
    pub fn episode_loss(
        &self,
        support: &[f32],
        queries: &[f32],
        query_labels: &[usize],
        n_way: usize,
        n_shot: usize,
        n_query: usize,
    ) -> MetaResult<f32> {
        if n_way < 2 {
            return Err(MetaError::InvalidNWay { n_way });
        }
        if n_shot < 1 {
            return Err(MetaError::InvalidKShot { k_shot: n_shot });
        }
        if n_query == 0 {
            return Err(MetaError::InvalidQuerySize { size: 0 });
        }
        let d = self.config.d_feat;
        if queries.len() != n_query * d {
            return Err(MetaError::DimensionMismatch {
                expected: n_query * d,
                got: queries.len(),
            });
        }
        if query_labels.len() != n_query {
            return Err(MetaError::DimensionMismatch {
                expected: n_query,
                got: query_labels.len(),
            });
        }

        let protos = self.adapted_prototypes(support, queries, n_way, n_shot)?;
        let mut total_loss = 0.0_f32;

        for qi in 0..n_query {
            let q = &queries[qi * d..(qi + 1) * d];
            let neg_dists: Vec<f32> = (0..n_way)
                .map(|cls| {
                    let proto = &protos[cls * d..(cls + 1) * d];
                    -q.iter()
                        .zip(proto.iter())
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum::<f32>()
                })
                .collect();

            let max_nd = neg_dists.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = neg_dists.iter().map(|&v| (v - max_nd).exp()).collect();
            let sum_e: f32 = exps.iter().sum();
            if !sum_e.is_finite() || sum_e == 0.0 {
                return Err(MetaError::NanEncountered {
                    context: "episode_loss: softmax sum is non-finite".into(),
                });
            }

            let true_cls = query_labels[qi];
            if true_cls >= n_way {
                return Err(MetaError::InvalidEpisodeConfig {
                    msg: format!("label {true_cls} >= n_way {n_way}"),
                });
            }
            let log_prob = (exps[true_cls] / sum_e).ln();
            if !log_prob.is_finite() {
                return Err(MetaError::NanEncountered {
                    context: "episode_loss: log_prob non-finite".into(),
                });
            }
            total_loss -= log_prob;
        }

        let loss = total_loss / n_query as f32;
        if !loss.is_finite() {
            return Err(MetaError::NanEncountered {
                context: "episode_loss: final loss non-finite".into(),
            });
        }
        Ok(loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cafs(d_feat: usize, n_heads: usize, n_layers: usize, d_ff: usize) -> CafsFewShot {
        let cfg = CafsConfig {
            d_feat,
            n_heads,
            n_layers,
            d_ff,
        };
        CafsFewShot::new(cfg, &mut LcgRng::new(42)).expect("value should be present")
    }

    #[test]
    fn adapted_prototypes_shape() {
        let net = make_cafs(8, 2, 1, 16);
        let support = vec![0.1_f32; 3 * 2 * 8];
        let query = vec![0.2_f32; 4 * 8];
        let protos = net
            .adapted_prototypes(&support, &query, 3, 2)
            .expect("adapted_prototypes should succeed");
        assert_eq!(
            protos.len(),
            3 * 8,
            "adapted_prototypes must be n_way * d_feat"
        );
    }

    #[test]
    fn predict_in_range() {
        let net = make_cafs(8, 2, 1, 16);
        let support = vec![0.1_f32; 3 * 2 * 8];
        let query = vec![0.2_f32; 4 * 8];
        let protos = net
            .adapted_prototypes(&support, &query, 3, 2)
            .expect("adapted_prototypes should succeed");
        let q_single = vec![0.3_f32; 8];
        let pred = net
            .predict(&q_single, &protos, 3)
            .expect("predict should succeed");
        assert!(pred < 3, "predicted class must be in [0, n_way)");
    }

    #[test]
    fn episode_loss_finite() {
        let net = make_cafs(8, 2, 1, 16);
        let support = vec![0.1_f32; 3 * 2 * 8];
        let queries = vec![0.2_f32; 4 * 8];
        let labels = vec![0usize, 1, 2, 0];
        let loss = net
            .episode_loss(&support, &queries, &labels, 3, 2, 4)
            .expect("value should be present");
        assert!(loss.is_finite(), "episode_loss must be finite, got {loss}");
    }

    #[test]
    fn episode_loss_nonneg() {
        let net = make_cafs(8, 2, 1, 16);
        let support = vec![0.1_f32; 3 * 2 * 8];
        let queries = vec![0.2_f32; 4 * 8];
        let labels = vec![0usize, 1, 2, 0];
        let loss = net
            .episode_loss(&support, &queries, &labels, 3, 2, 4)
            .expect("value should be present");
        assert!(loss >= 0.0, "episode_loss must be non-negative, got {loss}");
    }

    #[test]
    fn single_head_works() {
        let net = make_cafs(4, 1, 1, 8);
        let support = vec![0.5_f32; 2 * 4];
        let query = vec![0.3_f32; 2 * 4];
        let protos = net
            .adapted_prototypes(&support, &query, 2, 1)
            .expect("adapted_prototypes should succeed");
        assert_eq!(protos.len(), 2 * 4);
    }

    #[test]
    fn single_layer_works() {
        let net = make_cafs(4, 2, 1, 8);
        let support = vec![0.1_f32; 2 * 2 * 4];
        let query = vec![0.2_f32; 3 * 4];
        let protos = net
            .adapted_prototypes(&support, &query, 2, 2)
            .expect("adapted_prototypes should succeed");
        assert_eq!(protos.len(), 2 * 4);
    }

    #[test]
    fn d_feat_not_divisible_by_heads_error() {
        let cfg = CafsConfig {
            d_feat: 5,
            n_heads: 2,
            n_layers: 1,
            d_ff: 8,
        };
        let result = CafsFewShot::new(cfg, &mut LcgRng::new(1));
        assert!(
            matches!(result, Err(MetaError::InvalidEpisodeConfig { .. })),
            "d_feat not divisible by n_heads must return InvalidEpisodeConfig"
        );
    }

    #[test]
    fn n_heads_0_error() {
        let cfg = CafsConfig {
            d_feat: 4,
            n_heads: 0,
            n_layers: 1,
            d_ff: 8,
        };
        let result = CafsFewShot::new(cfg, &mut LcgRng::new(1));
        assert!(
            matches!(result, Err(MetaError::InvalidEpisodeConfig { .. })),
            "n_heads=0 must return InvalidEpisodeConfig"
        );
    }

    #[test]
    fn attention_output_shape() {
        let net = make_cafs(8, 2, 1, 16);
        let queries = vec![0.1_f32; 3 * 8];
        let kv = vec![0.2_f32; 4 * 8];
        let out = net
            .attention_layer(0, &queries, 3, &kv, 4)
            .expect("attention_layer should succeed");
        assert_eq!(out.len(), 3 * 8, "attention output must be n_q * d_feat");
    }
}
