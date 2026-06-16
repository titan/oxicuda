//! CAN — Cross Attention Network for Few-shot Classification (Hou et al., NeurIPS 2019).
//!
//! Instead of simple concatenation+MLP (RelationNet), CAN computes multi-head cross-attention
//! from each query to all support examples, producing attention-reweighted class prototypes.
//! Classification is via cosine similarity between the query and attended prototypes.
//!
//! Reference: <https://arxiv.org/abs/1910.07677>

use crate::episode::types::FewShotEpisode;
use crate::error::{MetaError, MetaResult};
use crate::handle::LcgRng;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters for Cross Attention Network (CAN).
#[derive(Debug, Clone)]
pub struct CanConfig {
    /// Feature dimension d.  Must be divisible by `attention_heads`.
    pub feat_dim: usize,
    /// Number of classes for N-way classification.
    pub n_way: usize,
    /// Number of support examples per class.
    pub k_shot: usize,
    /// Number of attention heads h.  d must be divisible by h.
    pub attention_heads: usize,
    /// Attention temperature scale.  Use `1.0 / sqrt(feat_dim / attention_heads)` if unsure.
    pub scale: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Weights
// ─────────────────────────────────────────────────────────────────────────────

/// Projection matrices for multi-head cross-attention.
#[derive(Debug, Clone)]
pub struct CanWeights {
    /// Query projection W_Q: d × d, row-major (W_Q[i*d+j] = entry at row i, col j).
    pub w_q: Vec<f32>,
    /// Key projection W_K: d × d, row-major.
    pub w_k: Vec<f32>,
    /// Value projection W_V: d × d, row-major.
    pub w_v: Vec<f32>,
    /// Output projection W_O: d × d, row-major.
    pub w_o: Vec<f32>,
    /// Feature dimension d.
    pub feat_dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Output type
// ─────────────────────────────────────────────────────────────────────────────

/// Result of the cross-attention prototype computation for a single query.
#[derive(Debug, Clone)]
pub struct CanAttentionOutput {
    /// Attention-weighted prototype per class: n_way × feat_dim, row-major.
    pub attended_prototypes: Vec<f32>,
    /// Per-class attention weight distribution over that class's support: n_way × k_shot.
    pub attention_weights: Vec<f32>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Main struct
// ─────────────────────────────────────────────────────────────────────────────

/// Cross Attention Network (CAN) few-shot classifier.
pub struct Can {
    /// Configuration.
    pub config: CanConfig,
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically-stable softmax over a slice.
fn softmax_stable(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let inv = if sum > 1e-38 { 1.0 / sum } else { 1.0 };
    for e in exps.iter_mut() {
        *e *= inv;
    }
    exps
}

/// Apply a d×d weight matrix (row-major) to a d-dimensional input vector.
/// out[i] = Σ_j w[i*d+j] * x[j]
fn mat_vec(w: &[f32], x: &[f32], d: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; d];
    for i in 0..d {
        for j in 0..d {
            out[i] += w[i * d + j] * x[j];
        }
    }
    out
}

/// Apply rows [row_start..row_end] of a d×d weight matrix to x (d,).
/// Produces a (row_end-row_start,) output.
fn mat_vec_rows(w: &[f32], x: &[f32], d: usize, row_start: usize, row_end: usize) -> Vec<f32> {
    let out_len = row_end - row_start;
    let mut out = vec![0.0_f32; out_len];
    for (local_i, global_i) in (row_start..row_end).enumerate() {
        for j in 0..d {
            out[local_i] += w[global_i * d + j] * x[j];
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// impl Can
// ─────────────────────────────────────────────────────────────────────────────

impl Can {
    /// Construct and validate a CAN instance.
    pub fn new(config: CanConfig) -> MetaResult<Self> {
        if config.n_way < 2 {
            return Err(MetaError::InvalidNWay {
                n_way: config.n_way,
            });
        }
        if config.k_shot < 1 {
            return Err(MetaError::InvalidKShot {
                k_shot: config.k_shot,
            });
        }
        if config.feat_dim < 1 {
            return Err(MetaError::InvalidFeatDim {
                dim: config.feat_dim,
            });
        }
        if config.attention_heads < 1 {
            return Err(MetaError::Internal {
                msg: "attention_heads must be >= 1".into(),
            });
        }
        if !config.feat_dim.is_multiple_of(config.attention_heads) {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: format!(
                    "feat_dim ({}) must be divisible by attention_heads ({})",
                    config.feat_dim, config.attention_heads
                ),
            });
        }
        Ok(Self { config })
    }

    /// Initialise all four projection matrices with Xavier uniform.
    ///
    /// Xavier limit = sqrt(6 / (d + d)) = sqrt(3 / d).
    /// Uses LCG range trick: (next_f32() - 0.25) * 2 * limit → approximately [-limit, +limit).
    pub fn init_weights(feat_dim: usize, n_heads: usize, rng: &mut LcgRng) -> CanWeights {
        let d = feat_dim;
        let limit = (3.0_f32 / d.max(1) as f32).sqrt();
        let n_elem = d * d;

        let mut w_q = vec![0.0_f32; n_elem];
        for v in w_q.iter_mut() {
            *v = (rng.next_f32() - 0.25) * 2.0 * limit;
        }
        let mut w_k = vec![0.0_f32; n_elem];
        for v in w_k.iter_mut() {
            *v = (rng.next_f32() - 0.25) * 2.0 * limit;
        }
        let mut w_v = vec![0.0_f32; n_elem];
        for v in w_v.iter_mut() {
            *v = (rng.next_f32() - 0.25) * 2.0 * limit;
        }
        let mut w_o = vec![0.0_f32; n_elem];
        for v in w_o.iter_mut() {
            *v = (rng.next_f32() - 0.25) * 2.0 * limit;
        }

        CanWeights {
            w_q,
            w_k,
            w_v,
            w_o,
            feat_dim,
            n_heads,
        }
    }

    /// Compute class prototypes by averaging support examples per class.
    ///
    /// `support`: (n_way * k_shot) × feat_dim, arranged so that examples for class c
    /// occupy rows c*k_shot .. (c+1)*k_shot (ordered by class then shot).
    ///
    /// Returns prototypes: n_way × feat_dim, row-major.
    pub fn compute_prototypes(
        support: &[f32],
        n_way: usize,
        k_shot: usize,
        feat_dim: usize,
    ) -> MetaResult<Vec<f32>> {
        let n_support = n_way * k_shot;
        if support.len() != n_support * feat_dim {
            return Err(MetaError::DimensionMismatch {
                expected: n_support * feat_dim,
                got: support.len(),
            });
        }
        let mut prototypes = vec![0.0_f32; n_way * feat_dim];
        for c in 0..n_way {
            let mut count = 0_usize;
            for k in 0..k_shot {
                let row_idx = c * k_shot + k;
                let feat = &support[row_idx * feat_dim..(row_idx + 1) * feat_dim];
                for (j, &fj) in feat.iter().enumerate() {
                    prototypes[c * feat_dim + j] += fj;
                }
                count += 1;
            }
            if count == 0 {
                return Err(MetaError::InsufficientExamples {
                    cls: c,
                    need: 1,
                    got: 0,
                });
            }
            let inv = 1.0 / count as f32;
            for j in 0..feat_dim {
                prototypes[c * feat_dim + j] *= inv;
            }
        }
        Ok(prototypes)
    }

    /// Multi-head cross-attention: query attends to all support examples.
    ///
    /// - `query`: feat_dim-dimensional query vector.
    /// - `support_all`: n_support × feat_dim, row-major.
    ///
    /// Returns `(attended_output, attn_weights)`:
    /// - `attended_output`: feat_dim — the projected attended context.
    /// - `attn_weights`: n_support — softmax attention from the last head (representative).
    pub fn cross_attend(
        weights: &CanWeights,
        query: &[f32],
        support_all: &[f32],
        n_support: usize,
        scale: f32,
    ) -> MetaResult<(Vec<f32>, Vec<f32>)> {
        let d = weights.feat_dim;
        let h = weights.n_heads;
        let d_head = d / h;

        if query.len() != d {
            return Err(MetaError::DimensionMismatch {
                expected: d,
                got: query.len(),
            });
        }
        if support_all.len() != n_support * d {
            return Err(MetaError::DimensionMismatch {
                expected: n_support * d,
                got: support_all.len(),
            });
        }
        if n_support == 0 {
            return Err(MetaError::EmptySupport);
        }

        let mut concat_heads = vec![0.0_f32; d]; // h heads × d_head each = d total
        let mut last_attn = vec![0.0_f32; n_support];

        for head in 0..h {
            let row_start = head * d_head;
            let row_end = row_start + d_head;

            // q_h = W_Q[head rows] @ query  →  (d_head,)
            let q_h = mat_vec_rows(&weights.w_q, query, d, row_start, row_end);

            // For each support vector, compute k_h = W_K[head rows] @ s_i  →  (d_head,)
            // Attention score: a[i] = dot(q_h, k_h[i]) * scale
            let mut scores = Vec::with_capacity(n_support);
            let mut k_heads: Vec<Vec<f32>> = Vec::with_capacity(n_support);
            for i in 0..n_support {
                let s_i = &support_all[i * d..(i + 1) * d];
                let k_h = mat_vec_rows(&weights.w_k, s_i, d, row_start, row_end);
                let dot: f32 = q_h.iter().zip(k_h.iter()).map(|(&a, &b)| a * b).sum();
                scores.push(dot * scale);
                k_heads.push(k_h);
            }

            // Softmax over scores → α (n_support,)
            let alpha = softmax_stable(&scores);

            // v_h[i] = W_V[head rows] @ s_i  →  (d_head,)
            // attended_h = Σ_i α[i] * v_h[i]  →  (d_head,)
            let mut attended_h = vec![0.0_f32; d_head];
            for i in 0..n_support {
                let s_i = &support_all[i * d..(i + 1) * d];
                let v_h = mat_vec_rows(&weights.w_v, s_i, d, row_start, row_end);
                for j in 0..d_head {
                    attended_h[j] += alpha[i] * v_h[j];
                }
            }

            // Store head output into concat_heads
            concat_heads[row_start..row_end].copy_from_slice(&attended_h);

            // Record attention weights from last head as representative
            if head == h - 1 {
                last_attn.copy_from_slice(&alpha);
            }
        }

        // Output projection: output = W_O @ concat_heads  →  (d,)
        let output = mat_vec(&weights.w_o, &concat_heads, d);
        Ok((output, last_attn))
    }

    /// Compute attention-reweighted prototypes for a single query feature vector.
    ///
    /// Returns `CanAttentionOutput` with:
    /// - `attended_prototypes`: n_way × feat_dim
    /// - `attention_weights`: n_way × k_shot (per-class attention over that class's support)
    pub fn attend_prototypes(
        weights: &CanWeights,
        query_feat: &[f32],
        support_x: &[f32],
        support_y: &[u32],
        n_way: usize,
        k_shot: usize,
        feat_dim: usize,
        scale: f32,
    ) -> MetaResult<CanAttentionOutput> {
        let n_support = n_way * k_shot;
        let d = feat_dim;

        if query_feat.len() != d {
            return Err(MetaError::DimensionMismatch {
                expected: d,
                got: query_feat.len(),
            });
        }
        if support_x.len() != n_support * d {
            return Err(MetaError::DimensionMismatch {
                expected: n_support * d,
                got: support_x.len(),
            });
        }
        if support_y.len() != n_support {
            return Err(MetaError::DimensionMismatch {
                expected: n_support,
                got: support_y.len(),
            });
        }

        // Cross-attend query against ALL support examples
        let (_attended_ctx, full_attn) =
            Self::cross_attend(weights, query_feat, support_x, n_support, scale)?;

        // Group attention by class and compute normalized per-class weights
        // attended_prototypes[c] = Σ_{i in class c} attn_norm[c,i] * support_x[i]
        let mut class_attn_sum = vec![0.0_f32; n_way];
        let mut class_attn_by_idx: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_way];

        for i in 0..n_support {
            let c = support_y[i] as usize;
            if c < n_way {
                class_attn_sum[c] += full_attn[i];
                class_attn_by_idx[c].push((i, full_attn[i]));
            }
        }

        let mut attended_prototypes = vec![0.0_f32; n_way * d];
        let mut attention_weights = vec![0.0_f32; n_way * k_shot];

        for c in 0..n_way {
            let sum = class_attn_sum[c];
            let inv = if sum > 1e-38 { 1.0 / sum } else { 1.0 };
            // Sort by global support index so we can assign k_shot positions stably
            let mut indexed = class_attn_by_idx[c].clone();
            indexed.sort_by_key(|&(idx, _)| idx);

            for (slot, (global_idx, raw_attn)) in indexed.iter().enumerate() {
                let norm_attn = raw_attn * inv;
                // Accumulate into prototype
                let feat = &support_x[global_idx * d..(global_idx + 1) * d];
                for j in 0..d {
                    attended_prototypes[c * d + j] += norm_attn * feat[j];
                }
                // Store normalized attention weight for this slot
                if slot < k_shot {
                    attention_weights[c * k_shot + slot] = norm_attn;
                }
            }
        }

        Ok(CanAttentionOutput {
            attended_prototypes,
            attention_weights,
        })
    }

    /// Classify a query by cosine similarity against attended prototypes.
    ///
    /// Returns logits (n_way,): `logit[c] = cosine_similarity(query, attended_proto[c])`.
    pub fn classify(
        query_feat: &[f32],
        attended_prototypes: &[f32],
        n_way: usize,
        feat_dim: usize,
    ) -> MetaResult<Vec<f32>> {
        if query_feat.len() != feat_dim {
            return Err(MetaError::DimensionMismatch {
                expected: feat_dim,
                got: query_feat.len(),
            });
        }
        if attended_prototypes.len() != n_way * feat_dim {
            return Err(MetaError::DimensionMismatch {
                expected: n_way * feat_dim,
                got: attended_prototypes.len(),
            });
        }

        let q_norm_sq: f32 = query_feat.iter().map(|&v| v * v).sum();
        let q_norm = q_norm_sq.sqrt();

        let mut logits = vec![0.0_f32; n_way];
        for c in 0..n_way {
            let proto = &attended_prototypes[c * feat_dim..(c + 1) * feat_dim];
            let dot: f32 = query_feat
                .iter()
                .zip(proto.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            let p_norm_sq: f32 = proto.iter().map(|&v| v * v).sum();
            let p_norm = p_norm_sq.sqrt();
            logits[c] = dot / (q_norm * p_norm + 1e-8);
        }
        Ok(logits)
    }

    /// Full episode prediction: for each query, attend prototypes → cosine classify → argmax.
    ///
    /// Returns predicted class indices, one per query example.
    pub fn predict_episode(
        &self,
        weights: &CanWeights,
        episode: &FewShotEpisode,
    ) -> MetaResult<Vec<u32>> {
        let cfg = &episode.config;
        let n_way = cfg.n_way;
        let k_shot = cfg.k_shot;
        let feat_dim = cfg.feat_dim;
        let n_query = n_way * cfg.n_query;
        let scale = self.config.scale;

        if weights.feat_dim != feat_dim {
            return Err(MetaError::DimensionMismatch {
                expected: weights.feat_dim,
                got: feat_dim,
            });
        }

        let mut preds = Vec::with_capacity(n_query);

        for q in 0..n_query {
            let q_feat = &episode.query_x[q * feat_dim..(q + 1) * feat_dim];

            let attn_out = Self::attend_prototypes(
                weights,
                q_feat,
                &episode.support_x,
                &episode.support_y,
                n_way,
                k_shot,
                feat_dim,
                scale,
            )?;

            let logits = Self::classify(q_feat, &attn_out.attended_prototypes, n_way, feat_dim)?;

            let best = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            preds.push(best as u32);
        }
        Ok(preds)
    }

    /// Compute accuracy over a full episode.
    ///
    /// Returns accuracy ∈ [0, 1].
    pub fn episode_accuracy(
        &self,
        weights: &CanWeights,
        episode: &FewShotEpisode,
    ) -> MetaResult<f32> {
        let preds = self.predict_episode(weights, episode)?;
        let n = preds.len();
        if n == 0 {
            return Err(MetaError::InvalidQuerySize { size: 0 });
        }
        let n_correct = preds
            .iter()
            .zip(episode.query_y.iter())
            .filter(|&(&p, &y)| p == y)
            .count();
        Ok(n_correct as f32 / n as f32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episode::types::EpisodeConfig;
    use crate::handle::LcgRng;

    fn default_config() -> CanConfig {
        CanConfig {
            feat_dim: 8,
            n_way: 3,
            k_shot: 2,
            attention_heads: 2,
            scale: 0.5,
        }
    }

    fn make_weights(cfg: &CanConfig) -> CanWeights {
        let mut rng = LcgRng::new(42);
        Can::init_weights(cfg.feat_dim, cfg.attention_heads, &mut rng)
    }

    fn make_episode(
        n_way: usize,
        k_shot: usize,
        n_query: usize,
        feat_dim: usize,
    ) -> FewShotEpisode {
        let mut rng = LcgRng::new(99);
        let n_support = n_way * k_shot;
        let n_q = n_way * n_query;
        // Generate slightly separated features per class
        let support_x: Vec<f32> = (0..n_support)
            .flat_map(|s| {
                let c = s / k_shot;
                (0..feat_dim).map(move |j| if j == c % feat_dim { 1.0_f32 } else { 0.0_f32 })
            })
            .collect();
        let support_y: Vec<u32> = (0..n_way)
            .flat_map(|c| std::iter::repeat_n(c as u32, k_shot))
            .collect();
        let query_x: Vec<f32> = (0..n_q * feat_dim).map(|_| rng.next_f32()).collect();
        let query_y: Vec<u32> = (0..n_way)
            .flat_map(|c| std::iter::repeat_n(c as u32, n_query))
            .collect();
        FewShotEpisode {
            config: EpisodeConfig {
                n_way,
                k_shot,
                n_query,
                feat_dim,
            },
            support_x,
            support_y,
            query_x,
            query_y,
        }
    }

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn new_valid_config_succeeds() {
        assert!(Can::new(default_config()).is_ok());
    }

    #[test]
    fn new_feat_dim_not_divisible_by_heads_fails() {
        let cfg = CanConfig {
            feat_dim: 7,
            n_way: 2,
            k_shot: 1,
            attention_heads: 3,
            scale: 0.5,
        };
        assert!(matches!(
            Can::new(cfg),
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    #[test]
    fn new_n_way_one_fails() {
        let mut cfg = default_config();
        cfg.n_way = 1;
        assert!(matches!(Can::new(cfg), Err(MetaError::InvalidNWay { .. })));
    }

    #[test]
    fn new_zero_heads_fails() {
        let mut cfg = default_config();
        cfg.attention_heads = 0;
        assert!(matches!(Can::new(cfg), Err(MetaError::Internal { .. })));
    }

    // ── Prototypes ────────────────────────────────────────────────────────────

    #[test]
    fn compute_prototypes_shape() {
        let n_way = 3;
        let k_shot = 2;
        let feat_dim = 8;
        let support = vec![0.0_f32; n_way * k_shot * feat_dim];
        let protos = Can::compute_prototypes(&support, n_way, k_shot, feat_dim)
            .expect("compute_prototypes should succeed");
        assert_eq!(protos.len(), n_way * feat_dim);
    }

    #[test]
    fn compute_prototypes_k_shot_one_equals_support() {
        let feat_dim = 4;
        let n_way = 2;
        let k_shot = 1;
        let support = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let protos = Can::compute_prototypes(&support, n_way, k_shot, feat_dim)
            .expect("compute_prototypes should succeed");
        assert_eq!(protos.len(), n_way * feat_dim);
        for (a, b) in protos.iter().zip(support.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "k=1 prototype should equal support: {a} vs {b}"
            );
        }
    }

    // ── Cross-attention ───────────────────────────────────────────────────────

    #[test]
    fn cross_attend_output_shape() {
        let cfg = default_config();
        let weights = make_weights(&cfg);
        let mut rng = LcgRng::new(7);
        let query: Vec<f32> = (0..cfg.feat_dim).map(|_| rng.next_f32()).collect();
        let n_support = cfg.n_way * cfg.k_shot;
        let support: Vec<f32> = (0..n_support * cfg.feat_dim)
            .map(|_| rng.next_f32())
            .collect();
        let (out, _attn) = Can::cross_attend(&weights, &query, &support, n_support, cfg.scale)
            .expect("cross_attend should succeed");
        assert_eq!(
            out.len(),
            cfg.feat_dim,
            "attended output must have feat_dim elements"
        );
    }

    #[test]
    fn cross_attend_attention_sums_to_one() {
        let cfg = default_config();
        let weights = make_weights(&cfg);
        let mut rng = LcgRng::new(11);
        let query: Vec<f32> = (0..cfg.feat_dim).map(|_| rng.next_f32()).collect();
        let n_support = cfg.n_way * cfg.k_shot;
        let support: Vec<f32> = (0..n_support * cfg.feat_dim)
            .map(|_| rng.next_f32())
            .collect();
        let (_out, attn) = Can::cross_attend(&weights, &query, &support, n_support, cfg.scale)
            .expect("cross_attend should succeed");
        let sum: f32 = attn.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "attention weights must sum to 1.0, got {sum}"
        );
    }

    // ── attend_prototypes ─────────────────────────────────────────────────────

    #[test]
    fn attend_prototypes_output_shape() {
        let cfg = default_config();
        let weights = make_weights(&cfg);
        let episode = make_episode(cfg.n_way, cfg.k_shot, 2, cfg.feat_dim);
        let mut rng = LcgRng::new(5);
        let q_feat: Vec<f32> = (0..cfg.feat_dim).map(|_| rng.next_f32()).collect();
        let out = Can::attend_prototypes(
            &weights,
            &q_feat,
            &episode.support_x,
            &episode.support_y,
            cfg.n_way,
            cfg.k_shot,
            cfg.feat_dim,
            cfg.scale,
        )
        .expect("value should be present");
        assert_eq!(out.attended_prototypes.len(), cfg.n_way * cfg.feat_dim);
    }

    #[test]
    fn attend_prototypes_attn_weights_shape() {
        let cfg = default_config();
        let weights = make_weights(&cfg);
        let episode = make_episode(cfg.n_way, cfg.k_shot, 2, cfg.feat_dim);
        let mut rng = LcgRng::new(6);
        let q_feat: Vec<f32> = (0..cfg.feat_dim).map(|_| rng.next_f32()).collect();
        let out = Can::attend_prototypes(
            &weights,
            &q_feat,
            &episode.support_x,
            &episode.support_y,
            cfg.n_way,
            cfg.k_shot,
            cfg.feat_dim,
            cfg.scale,
        )
        .expect("value should be present");
        assert_eq!(out.attention_weights.len(), cfg.n_way * cfg.k_shot);
    }

    // ── Classify ──────────────────────────────────────────────────────────────

    #[test]
    fn classify_returns_n_way_logits() {
        let n_way = 3;
        let feat_dim = 4;
        let query = vec![1.0_f32, 0.0, 0.0, 0.0];
        let protos = vec![
            1.0_f32, 0.0, 0.0, 0.0, // class 0 = identical to query
            0.0, 1.0, 0.0, 0.0, // class 1
            0.0, 0.0, 1.0, 0.0, // class 2
        ];
        let logits =
            Can::classify(&query, &protos, n_way, feat_dim).expect("classify should succeed");
        assert_eq!(logits.len(), n_way);
    }

    #[test]
    fn classify_identical_query_proto_highest_similarity() {
        let n_way = 3;
        let feat_dim = 4;
        let query = vec![1.0_f32, 0.0, 0.0, 0.0];
        let protos = vec![
            1.0_f32, 0.0, 0.0, 0.0, // class 0 = identical to query
            0.0, 1.0, 0.0, 0.0, // class 1
            0.0, 0.0, 1.0, 0.0, // class 2
        ];
        let logits =
            Can::classify(&query, &protos, n_way, feat_dim).expect("classify should succeed");
        let best = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("value should be present");
        assert_eq!(
            best, 0,
            "identical query should have highest cosine sim to class 0"
        );
    }

    // ── predict_episode / episode_accuracy ────────────────────────────────────

    #[test]
    fn predict_episode_returns_n_query_preds() {
        let cfg = default_config();
        let can = Can::new(cfg.clone()).expect("value should be present");
        let weights = make_weights(&cfg);
        let n_query = 3;
        let episode = make_episode(cfg.n_way, cfg.k_shot, n_query, cfg.feat_dim);
        let preds = can
            .predict_episode(&weights, &episode)
            .expect("predict_episode should succeed");
        assert_eq!(preds.len(), cfg.n_way * n_query);
    }

    #[test]
    fn episode_accuracy_in_range() {
        let cfg = default_config();
        let can = Can::new(cfg.clone()).expect("value should be present");
        let weights = make_weights(&cfg);
        let episode = make_episode(cfg.n_way, cfg.k_shot, 2, cfg.feat_dim);
        let acc = can
            .episode_accuracy(&weights, &episode)
            .expect("episode_accuracy should succeed");
        assert!((0.0..=1.0).contains(&acc), "accuracy must be in [0,1]");
    }

    // ── Single-head specialization ────────────────────────────────────────────

    #[test]
    fn single_head_works() {
        let cfg = CanConfig {
            feat_dim: 8,
            n_way: 3,
            k_shot: 2,
            attention_heads: 1,
            scale: 0.35,
        };
        let can = Can::new(cfg.clone()).expect("value should be present");
        let weights = make_weights(&cfg);
        let episode = make_episode(cfg.n_way, cfg.k_shot, 2, cfg.feat_dim);
        let acc = can
            .episode_accuracy(&weights, &episode)
            .expect("episode_accuracy should succeed");
        assert!((0.0..=1.0).contains(&acc));
    }

    #[test]
    fn multi_head_four_works() {
        let cfg = CanConfig {
            feat_dim: 8,
            n_way: 3,
            k_shot: 2,
            attention_heads: 4,
            scale: 0.5,
        };
        let can = Can::new(cfg.clone()).expect("value should be present");
        let weights = make_weights(&cfg);
        let episode = make_episode(cfg.n_way, cfg.k_shot, 2, cfg.feat_dim);
        let acc = can
            .episode_accuracy(&weights, &episode)
            .expect("episode_accuracy should succeed");
        assert!((0.0..=1.0).contains(&acc));
    }

    #[test]
    fn k_shot_one_works() {
        let cfg = CanConfig {
            feat_dim: 8,
            n_way: 3,
            k_shot: 1,
            attention_heads: 2,
            scale: 0.5,
        };
        let can = Can::new(cfg.clone()).expect("value should be present");
        let weights = make_weights(&cfg);
        let episode = make_episode(cfg.n_way, cfg.k_shot, 2, cfg.feat_dim);
        let acc = can
            .episode_accuracy(&weights, &episode)
            .expect("episode_accuracy should succeed");
        assert!((0.0..=1.0).contains(&acc));
    }

    #[test]
    fn n_way_two_works() {
        let cfg = CanConfig {
            feat_dim: 8,
            n_way: 2,
            k_shot: 2,
            attention_heads: 2,
            scale: 0.5,
        };
        let can = Can::new(cfg.clone()).expect("value should be present");
        let weights = make_weights(&cfg);
        let episode = make_episode(cfg.n_way, cfg.k_shot, 2, cfg.feat_dim);
        let acc = can
            .episode_accuracy(&weights, &episode)
            .expect("episode_accuracy should succeed");
        assert!((0.0..=1.0).contains(&acc));
    }

    #[test]
    fn scale_affects_attention_sharpness() {
        // With a higher temperature scale, attention should be more peaked (higher max weight)
        let cfg_sharp = CanConfig {
            feat_dim: 8,
            n_way: 2,
            k_shot: 3,
            attention_heads: 2,
            scale: 10.0,
        };
        let cfg_flat = CanConfig {
            scale: 0.01,
            ..cfg_sharp.clone()
        };
        let weights = make_weights(&cfg_sharp);
        let mut rng = LcgRng::new(23);
        let n_support = cfg_sharp.n_way * cfg_sharp.k_shot;
        let query: Vec<f32> = (0..cfg_sharp.feat_dim).map(|_| rng.next_f32()).collect();
        let support: Vec<f32> = (0..n_support * cfg_sharp.feat_dim)
            .map(|_| rng.next_f32())
            .collect();
        let (_out_sharp, attn_sharp) =
            Can::cross_attend(&weights, &query, &support, n_support, cfg_sharp.scale)
                .expect("cross_attend should succeed");
        let (_out_flat, attn_flat) =
            Can::cross_attend(&weights, &query, &support, n_support, cfg_flat.scale)
                .expect("cross_attend should succeed");
        let max_sharp = attn_sharp.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max_flat = attn_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // Higher scale → sharper distribution → higher max weight
        assert!(
            max_sharp > max_flat,
            "higher scale should give sharper attention: max_sharp={max_sharp}, max_flat={max_flat}"
        );
    }
}
