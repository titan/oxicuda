//! FEAT — Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions
//! (Ye, Hu, Zhan & Sha, CVPR 2020).
//!
//! Vanilla ProtoNet treats every class prototype independently.  FEAT instead
//! pushes the per-class support prototypes through a **set-to-set** function —
//! a multi-head self-attention Transformer block — so that each adapted
//! prototype is task-contextualised by the other classes in the episode.
//! Classification then proceeds ProtoNet-style: squared-Euclidean distance from
//! a query to the *adapted* prototypes, scaled by a temperature and turned into
//! a softmax distribution.
//!
//! The set-to-set block is a standard scaled-dot-product multi-head attention
//! with `Q = K = V = prototypes`, a residual connection and an output
//! projection `W_o`:
//!
//! ```text
//! head_h          = softmax( (P W_q^h)(P W_k^h)^T / sqrt(d_head) ) (P W_v^h)
//! attn            = concat_h(head_h)                      (n_way × feat_dim)
//! adapted         = P + attn W_o                          (residual + output proj)
//! ```
//!
//! Reference: <https://arxiv.org/abs/1812.03664>

use crate::error::{MetaError, MetaResult};
use crate::handle::LcgRng;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters for FEAT.
#[derive(Debug, Clone)]
pub struct FeatConfig {
    /// Feature / embedding dimension `d`.  Must be divisible by `n_heads`.
    pub feat_dim: usize,
    /// Number of classes for N-way classification.
    pub n_way: usize,
    /// Number of support examples per class.
    pub k_shot: usize,
    /// Number of self-attention heads `h`.  `feat_dim` must be divisible by `h`.
    pub n_heads: usize,
    /// Softmax temperature `τ > 0` used in the distance-based classifier.
    pub temperature: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Main struct
// ─────────────────────────────────────────────────────────────────────────────

/// FEAT few-shot classifier with a learned set-to-set Transformer block.
pub struct Feat {
    /// Query projection `W_q`: `feat_dim × feat_dim`, row-major.
    w_q: Vec<f32>,
    /// Key projection `W_k`: `feat_dim × feat_dim`, row-major.
    w_k: Vec<f32>,
    /// Value projection `W_v`: `feat_dim × feat_dim`, row-major.
    w_v: Vec<f32>,
    /// Output projection `W_o`: `feat_dim × feat_dim`, row-major.
    w_o: Vec<f32>,
    /// Configuration.
    cfg: FeatConfig,
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically-stable softmax over a slice (in place over a fresh allocation).
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

/// Project a stack of `n` row-major `d`-vectors `x` (n × d) by a `d × d`
/// row-major weight matrix `w`: `out[r, i] = Σ_j x[r, j] · w[i, j]`.
///
/// (Mirrors the convention `out = x · Wᵀ` with `W` stored row-major, i.e. the
/// row `i` of `W` is the projection vector that produces output coordinate `i`.)
fn project_rows(x: &[f32], w: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n * d];
    for r in 0..n {
        let x_row = &x[r * d..(r + 1) * d];
        for i in 0..d {
            let w_row = &w[i * d..(i + 1) * d];
            let mut acc = 0.0_f32;
            for j in 0..d {
                acc += x_row[j] * w_row[j];
            }
            out[r * d + i] = acc;
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// impl Feat
// ─────────────────────────────────────────────────────────────────────────────

impl Feat {
    /// Construct and validate a FEAT instance, initialising the four projection
    /// matrices with Xavier-style uniform sampling.
    ///
    /// Xavier limit = `sqrt(6 / (d + d)) = sqrt(3 / d)`.  The LCG range trick
    /// `(u − 0.5) · 2 · limit` with `u ∈ [0, 1)` yields values in
    /// `[−limit, +limit)`.
    pub fn new(cfg: FeatConfig, rng: &mut LcgRng) -> MetaResult<Self> {
        if cfg.feat_dim < 1 {
            return Err(MetaError::InvalidFeatDim { dim: cfg.feat_dim });
        }
        if cfg.n_way < 1 {
            return Err(MetaError::InvalidNWay { n_way: cfg.n_way });
        }
        if cfg.k_shot < 1 {
            return Err(MetaError::InvalidKShot { k_shot: cfg.k_shot });
        }
        if cfg.n_heads < 1 {
            return Err(MetaError::Internal {
                msg: "n_heads must be >= 1".into(),
            });
        }
        if !cfg.feat_dim.is_multiple_of(cfg.n_heads) {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: format!(
                    "feat_dim ({}) must be divisible by n_heads ({})",
                    cfg.feat_dim, cfg.n_heads
                ),
            });
        }
        if cfg.temperature <= 0.0 || cfg.temperature.is_nan() {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: format!("temperature must be > 0, got {}", cfg.temperature),
            });
        }

        let d = cfg.feat_dim;
        let limit = (3.0_f32 / d as f32).sqrt();
        let n_elem = d * d;
        let sample = |rng: &mut LcgRng| -> Vec<f32> {
            (0..n_elem)
                .map(|_| {
                    let u = rng.next_u32() as f32 / (u32::MAX as f32 + 1.0);
                    (u - 0.5) * 2.0 * limit
                })
                .collect()
        };

        let w_q = sample(rng);
        let w_k = sample(rng);
        let w_v = sample(rng);
        let w_o = sample(rng);

        Ok(Self {
            w_q,
            w_k,
            w_v,
            w_o,
            cfg,
        })
    }

    /// Compute class prototypes by averaging the `k_shot` support embeddings of
    /// each class.
    ///
    /// `support`: `(n_way · k_shot) × feat_dim`, row-major, **class-major**
    /// (rows `c·k_shot .. (c+1)·k_shot` belong to class `c`).
    ///
    /// Returns prototypes: `n_way × feat_dim`, row-major.
    pub fn compute_prototypes(&self, support: &[f32]) -> MetaResult<Vec<f32>> {
        let n_way = self.cfg.n_way;
        let k_shot = self.cfg.k_shot;
        let d = self.cfg.feat_dim;
        let expected = n_way * k_shot * d;
        if support.len() != expected {
            return Err(MetaError::DimensionMismatch {
                expected,
                got: support.len(),
            });
        }

        let mut prototypes = vec![0.0_f32; n_way * d];
        let inv = 1.0 / k_shot as f32;
        for c in 0..n_way {
            for k in 0..k_shot {
                let row = c * k_shot + k;
                let feat = &support[row * d..(row + 1) * d];
                let proto = &mut prototypes[c * d..(c + 1) * d];
                for (p, &f) in proto.iter_mut().zip(feat.iter()) {
                    *p += f;
                }
            }
            let proto = &mut prototypes[c * d..(c + 1) * d];
            for p in proto.iter_mut() {
                *p *= inv;
            }
        }
        Ok(prototypes)
    }

    /// Set-to-set adaptation: multi-head self-attention over the `n_way`
    /// prototypes (`Q = K = V = prototypes`) with scale `1/√head_dim`, a
    /// residual connection and an output projection `W_o`.
    ///
    /// `prototypes`: `n_way × feat_dim`, row-major.
    /// Returns adapted prototypes: `n_way × feat_dim`, row-major.
    pub fn adapt_prototypes(&self, prototypes: &[f32]) -> MetaResult<Vec<f32>> {
        let n_way = self.cfg.n_way;
        let d = self.cfg.feat_dim;
        let h = self.cfg.n_heads;
        let d_head = d / h;
        let expected = n_way * d;
        if prototypes.len() != expected {
            return Err(MetaError::DimensionMismatch {
                expected,
                got: prototypes.len(),
            });
        }

        // Linear projections: Q, K, V are each (n_way × d), row-major.
        let q = project_rows(prototypes, &self.w_q, n_way, d);
        let k = project_rows(prototypes, &self.w_k, n_way, d);
        let v = project_rows(prototypes, &self.w_v, n_way, d);

        let scale = 1.0_f32 / (d_head as f32).sqrt();

        // Concatenated multi-head attention context: (n_way × d), row-major.
        let mut context = vec![0.0_f32; n_way * d];

        for head in 0..h {
            let off = head * d_head;
            for i in 0..n_way {
                let q_i = &q[i * d + off..i * d + off + d_head];

                // Scores against every prototype j for this head.
                let mut scores = vec![0.0_f32; n_way];
                for (j, score) in scores.iter_mut().enumerate() {
                    let k_j = &k[j * d + off..j * d + off + d_head];
                    let mut dot = 0.0_f32;
                    for t in 0..d_head {
                        dot += q_i[t] * k_j[t];
                    }
                    *score = dot * scale;
                }

                let attn = softmax_stable(&scores);

                // Weighted sum of values for this head → context slice [off, off+d_head).
                for (j, &a) in attn.iter().enumerate() {
                    let v_j = &v[j * d + off..j * d + off + d_head];
                    let ctx = &mut context[i * d + off..i * d + off + d_head];
                    for (c_t, &v_t) in ctx.iter_mut().zip(v_j.iter()) {
                        *c_t += a * v_t;
                    }
                }
            }
        }

        // Output projection of the attention context, then residual add.
        let projected = project_rows(&context, &self.w_o, n_way, d);
        let mut adapted = vec![0.0_f32; n_way * d];
        for (a, (&p, &res)) in adapted
            .iter_mut()
            .zip(projected.iter().zip(prototypes.iter()))
        {
            *a = res + p;
        }
        Ok(adapted)
    }

    /// Classify a single query against the adapted prototypes.
    ///
    /// `logit_c = −‖q − proto_c‖² / temperature`, then softmax over classes.
    ///
    /// `query`: `feat_dim`.  `adapted_protos`: `n_way × feat_dim`, row-major.
    /// Returns a probability distribution of length `n_way`.
    pub fn classify(&self, query: &[f32], adapted_protos: &[f32]) -> MetaResult<Vec<f32>> {
        let n_way = self.cfg.n_way;
        let d = self.cfg.feat_dim;
        if query.len() != d {
            return Err(MetaError::DimensionMismatch {
                expected: d,
                got: query.len(),
            });
        }
        let expected = n_way * d;
        if adapted_protos.len() != expected {
            return Err(MetaError::DimensionMismatch {
                expected,
                got: adapted_protos.len(),
            });
        }

        let inv_temp = 1.0 / self.cfg.temperature;
        let mut logits = vec![0.0_f32; n_way];
        for c in 0..n_way {
            let proto = &adapted_protos[c * d..(c + 1) * d];
            let mut dist_sq = 0.0_f32;
            for (&qi, &pi) in query.iter().zip(proto.iter()) {
                let diff = qi - pi;
                dist_sq += diff * diff;
            }
            logits[c] = -dist_sq * inv_temp;
        }
        Ok(softmax_stable(&logits))
    }

    /// Full episode prediction: compute prototypes, adapt them with the
    /// set-to-set Transformer, then argmax-classify each query.
    ///
    /// `support`: `(n_way · k_shot) × feat_dim`, class-major row-major.
    /// `query`: `n_query × feat_dim`, row-major.
    /// Returns predicted class indices, one per query.
    pub fn predict_episode(
        &self,
        support: &[f32],
        query: &[f32],
        n_query: usize,
    ) -> MetaResult<Vec<usize>> {
        let d = self.cfg.feat_dim;
        let expected_q = n_query * d;
        if query.len() != expected_q {
            return Err(MetaError::DimensionMismatch {
                expected: expected_q,
                got: query.len(),
            });
        }

        let protos = self.compute_prototypes(support)?;
        let adapted = self.adapt_prototypes(&protos)?;

        let mut preds = Vec::with_capacity(n_query);
        for q in 0..n_query {
            let q_feat = &query[q * d..(q + 1) * d];
            let probs = self.classify(q_feat, &adapted)?;
            let best = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            preds.push(best);
        }
        Ok(preds)
    }

    /// Episode accuracy ∈ `[0, 1]` over `n_query` labelled query examples.
    pub fn episode_accuracy(
        &self,
        support: &[f32],
        query: &[f32],
        query_labels: &[usize],
        n_query: usize,
    ) -> MetaResult<f32> {
        if query_labels.len() != n_query {
            return Err(MetaError::DimensionMismatch {
                expected: n_query,
                got: query_labels.len(),
            });
        }
        if n_query == 0 {
            return Err(MetaError::InvalidQuerySize { size: 0 });
        }
        let preds = self.predict_episode(support, query, n_query)?;
        let n_correct = preds
            .iter()
            .zip(query_labels.iter())
            .filter(|&(&p, &y)| p == y)
            .count();
        Ok(n_correct as f32 / n_query as f32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> FeatConfig {
        FeatConfig {
            feat_dim: 8,
            n_way: 3,
            k_shot: 2,
            n_heads: 2,
            temperature: 1.0,
        }
    }

    fn make_feat(cfg: FeatConfig) -> Feat {
        let mut rng = LcgRng::new(42);
        Feat::new(cfg, &mut rng).unwrap()
    }

    // ── Construction validation ─────────────────────────────────────────────

    #[test]
    fn new_valid_config_succeeds() {
        let mut rng = LcgRng::new(1);
        assert!(Feat::new(default_config(), &mut rng).is_ok());
    }

    #[test]
    fn new_feat_dim_not_divisible_by_heads_errs() {
        let mut cfg = default_config();
        cfg.feat_dim = 9;
        cfg.n_heads = 2;
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            Feat::new(cfg, &mut rng),
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    #[test]
    fn new_temperature_non_positive_errs() {
        let mut cfg = default_config();
        cfg.temperature = 0.0;
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            Feat::new(cfg, &mut rng),
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    #[test]
    fn new_feat_dim_zero_errs() {
        let mut cfg = default_config();
        cfg.feat_dim = 0;
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            Feat::new(cfg, &mut rng),
            Err(MetaError::InvalidFeatDim { .. })
        ));
    }

    #[test]
    fn new_n_way_zero_errs() {
        let mut cfg = default_config();
        cfg.n_way = 0;
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            Feat::new(cfg, &mut rng),
            Err(MetaError::InvalidNWay { .. })
        ));
    }

    #[test]
    fn new_k_shot_zero_errs() {
        let mut cfg = default_config();
        cfg.k_shot = 0;
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            Feat::new(cfg, &mut rng),
            Err(MetaError::InvalidKShot { .. })
        ));
    }

    // ── compute_prototypes ──────────────────────────────────────────────────

    #[test]
    fn compute_prototypes_shape() {
        let cfg = default_config();
        let feat = make_feat(cfg.clone());
        let support = vec![0.5_f32; cfg.n_way * cfg.k_shot * cfg.feat_dim];
        let protos = feat.compute_prototypes(&support).unwrap();
        assert_eq!(protos.len(), cfg.n_way * cfg.feat_dim);
    }

    #[test]
    fn compute_prototypes_is_class_mean() {
        // n_way=2, k_shot=2, feat_dim=2. Class 0 rows: [1,1],[3,3] -> mean [2,2].
        // Class 1 rows: [0,4],[2,6] -> mean [1,5].
        let cfg = FeatConfig {
            feat_dim: 2,
            n_way: 2,
            k_shot: 2,
            n_heads: 1,
            temperature: 1.0,
        };
        let feat = make_feat(cfg);
        let support = vec![1.0, 1.0, 3.0, 3.0, 0.0, 4.0, 2.0, 6.0];
        let protos = feat.compute_prototypes(&support).unwrap();
        assert!((protos[0] - 2.0).abs() < 1e-6);
        assert!((protos[1] - 2.0).abs() < 1e-6);
        assert!((protos[2] - 1.0).abs() < 1e-6);
        assert!((protos[3] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn compute_prototypes_wrong_length_errs() {
        let feat = make_feat(default_config());
        let support = vec![0.0_f32; 10];
        assert!(matches!(
            feat.compute_prototypes(&support),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    // ── adapt_prototypes ────────────────────────────────────────────────────

    #[test]
    fn adapt_prototypes_preserves_shape() {
        let cfg = default_config();
        let feat = make_feat(cfg.clone());
        let protos = vec![0.3_f32; cfg.n_way * cfg.feat_dim];
        let adapted = feat.adapt_prototypes(&protos).unwrap();
        assert_eq!(adapted.len(), cfg.n_way * cfg.feat_dim);
    }

    #[test]
    fn adapt_prototypes_wrong_length_errs() {
        let feat = make_feat(default_config());
        let protos = vec![0.0_f32; 5];
        assert!(matches!(
            feat.adapt_prototypes(&protos),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn adapt_with_n_way_one_works() {
        // With a single prototype, self-attention is trivial (attends to itself).
        let cfg = FeatConfig {
            feat_dim: 4,
            n_way: 1,
            k_shot: 2,
            n_heads: 2,
            temperature: 1.0,
        };
        let feat = make_feat(cfg.clone());
        let protos = vec![0.2_f32, 0.4, 0.6, 0.8];
        let adapted = feat.adapt_prototypes(&protos).unwrap();
        assert_eq!(adapted.len(), cfg.feat_dim);
        assert!(adapted.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn single_head_works() {
        let cfg = FeatConfig {
            feat_dim: 6,
            n_way: 3,
            k_shot: 2,
            n_heads: 1,
            temperature: 1.0,
        };
        let feat = make_feat(cfg.clone());
        let support = vec![0.25_f32; cfg.n_way * cfg.k_shot * cfg.feat_dim];
        let protos = feat.compute_prototypes(&support).unwrap();
        let adapted = feat.adapt_prototypes(&protos).unwrap();
        assert_eq!(adapted.len(), cfg.n_way * cfg.feat_dim);
    }

    // ── classify ────────────────────────────────────────────────────────────

    #[test]
    fn classify_softmax_sums_to_one() {
        let cfg = default_config();
        let feat = make_feat(cfg.clone());
        let mut rng = LcgRng::new(7);
        let query: Vec<f32> = (0..cfg.feat_dim).map(|_| rng.next_f32()).collect();
        let protos: Vec<f32> = (0..cfg.n_way * cfg.feat_dim)
            .map(|_| rng.next_f32())
            .collect();
        let probs = feat.classify(&query, &protos).unwrap();
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax must sum to 1, got {sum}");
    }

    #[test]
    fn classify_length_is_n_way() {
        let cfg = default_config();
        let feat = make_feat(cfg.clone());
        let query = vec![0.1_f32; cfg.feat_dim];
        let protos = vec![0.2_f32; cfg.n_way * cfg.feat_dim];
        let probs = feat.classify(&query, &protos).unwrap();
        assert_eq!(probs.len(), cfg.n_way);
    }

    #[test]
    fn classify_query_wrong_length_errs() {
        let cfg = default_config();
        let feat = make_feat(cfg.clone());
        let query = vec![0.1_f32; cfg.feat_dim + 1];
        let protos = vec![0.2_f32; cfg.n_way * cfg.feat_dim];
        assert!(matches!(
            feat.classify(&query, &protos),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn classify_nearest_proto_gets_highest_prob() {
        let cfg = FeatConfig {
            feat_dim: 3,
            n_way: 3,
            k_shot: 1,
            n_heads: 1,
            temperature: 1.0,
        };
        let feat = make_feat(cfg);
        let query = vec![1.0_f32, 0.0, 0.0];
        let protos = vec![
            1.0_f32, 0.0, 0.0, // class 0 identical to query (dist 0)
            0.0, 1.0, 0.0, // class 1
            0.0, 0.0, 1.0, // class 2
        ];
        let probs = feat.classify(&query, &protos).unwrap();
        let best = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(best, 0);
    }

    // ── predict_episode / accuracy ──────────────────────────────────────────

    #[test]
    fn predict_episode_labels_in_range() {
        let cfg = default_config();
        let feat = make_feat(cfg.clone());
        let mut rng = LcgRng::new(13);
        let support: Vec<f32> = (0..cfg.n_way * cfg.k_shot * cfg.feat_dim)
            .map(|_| rng.next_f32())
            .collect();
        let n_query = 4;
        let query: Vec<f32> = (0..n_query * cfg.feat_dim)
            .map(|_| rng.next_f32())
            .collect();
        let preds = feat.predict_episode(&support, &query, n_query).unwrap();
        assert_eq!(preds.len(), n_query);
        assert!(preds.iter().all(|&p| p < cfg.n_way));
    }

    #[test]
    fn predict_episode_query_wrong_length_errs() {
        let cfg = default_config();
        let feat = make_feat(cfg.clone());
        let support = vec![0.1_f32; cfg.n_way * cfg.k_shot * cfg.feat_dim];
        let query = vec![0.0_f32; 3 * cfg.feat_dim + 1];
        assert!(matches!(
            feat.predict_episode(&support, &query, 3),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn episode_accuracy_in_range() {
        let cfg = default_config();
        let feat = make_feat(cfg.clone());
        let mut rng = LcgRng::new(21);
        let support: Vec<f32> = (0..cfg.n_way * cfg.k_shot * cfg.feat_dim)
            .map(|_| rng.next_f32())
            .collect();
        let n_query = 6;
        let query: Vec<f32> = (0..n_query * cfg.feat_dim)
            .map(|_| rng.next_f32())
            .collect();
        let labels: Vec<usize> = (0..n_query).map(|i| i % cfg.n_way).collect();
        let acc = feat
            .episode_accuracy(&support, &query, &labels, n_query)
            .unwrap();
        assert!((0.0..=1.0).contains(&acc));
    }

    #[test]
    fn episode_accuracy_labels_wrong_length_errs() {
        let cfg = default_config();
        let feat = make_feat(cfg.clone());
        let support = vec![0.1_f32; cfg.n_way * cfg.k_shot * cfg.feat_dim];
        let n_query = 4;
        let query = vec![0.0_f32; n_query * cfg.feat_dim];
        let labels = vec![0_usize; n_query + 1];
        assert!(matches!(
            feat.episode_accuracy(&support, &query, &labels, n_query),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn perfectly_separated_episode_accuracy_one() {
        // Build well-separated one-hot prototypes; queries equal the *adapted*
        // prototype of their own class, so classification must be perfect.
        let cfg = FeatConfig {
            feat_dim: 4,
            n_way: 4,
            k_shot: 1,
            n_heads: 2,
            temperature: 1.0,
        };
        let feat = make_feat(cfg.clone());
        // One-hot support: class c -> e_c.
        let mut support = vec![0.0_f32; cfg.n_way * cfg.k_shot * cfg.feat_dim];
        for c in 0..cfg.n_way {
            support[c * cfg.feat_dim + c] = 1.0;
        }
        let protos = feat.compute_prototypes(&support).unwrap();
        let adapted = feat.adapt_prototypes(&protos).unwrap();
        // Queries == adapted prototypes.
        let query = adapted.clone();
        let labels: Vec<usize> = (0..cfg.n_way).collect();
        let acc = feat
            .episode_accuracy(&support, &query, &labels, cfg.n_way)
            .unwrap();
        assert!((acc - 1.0).abs() < 1e-6, "expected accuracy 1.0, got {acc}");
    }

    #[test]
    fn deterministic_given_seed() {
        let cfg = default_config();
        let mut rng_a = LcgRng::new(123);
        let mut rng_b = LcgRng::new(123);
        let feat_a = Feat::new(cfg.clone(), &mut rng_a).unwrap();
        let feat_b = Feat::new(cfg.clone(), &mut rng_b).unwrap();
        let protos = vec![0.37_f32; cfg.n_way * cfg.feat_dim];
        let a = feat_a.adapt_prototypes(&protos).unwrap();
        let b = feat_b.adapt_prototypes(&protos).unwrap();
        assert_eq!(a, b, "same seed must give identical adaptation");
    }

    #[test]
    fn changing_support_changes_prediction() {
        let cfg = FeatConfig {
            feat_dim: 4,
            n_way: 2,
            k_shot: 1,
            n_heads: 1,
            temperature: 1.0,
        };
        let feat = make_feat(cfg.clone());
        let query = vec![1.0_f32, 0.0, 0.0, 0.0];
        // Support A: class 0 near query, class 1 far.
        let support_a = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        // Support B: swap so class 1 is near the query.
        let support_b = vec![0.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let pred_a = feat.predict_episode(&support_a, &query, 1).unwrap();
        let pred_b = feat.predict_episode(&support_b, &query, 1).unwrap();
        assert_ne!(
            pred_a, pred_b,
            "swapping which class is near the query must change the prediction"
        );
    }
}
