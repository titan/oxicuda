//! MIND — Multi-Interest Network with Dynamic Routing for Recommendation.
//!
//! Reference: Li et al., "Multi-Interest Network with Dynamic Routing for
//! Recommendation at Tmall", CIKM 2019.
//!
//! Architecture:
//!   Given a user's interaction history `[i_1, …, i_H]`, MIND produces `K`
//!   interest vectors via *capsule-routing* (dynamic routing as in CapsNet):
//!
//!   Initialization:
//!     `b[h][k] = 0`    for h ∈ {0..H}, k ∈ {0..K}
//!
//!   For r = 1..=n_routing_iters:
//!     `c[h]     = softmax(b[h])`                         (over K interests)
//!     `cap_k    = normalize(Σ_h c[h][k] * item_emb[history[h]])`
//!     `b[h][k] += cap_k · item_emb[history[h]]`
//!
//!   Final interest vectors = the K normalized capsules.
//!
//!   Scoring: `max_k (cap_k · item_emb[target])`.
//!   Top-K retrieval: score all items, return the top-k by max-interest score.

use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Numerically-stable softmax over a mutable slice (in-place).
fn softmax_inplace(v: &mut [f32]) {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inv = 1.0 / (sum + 1e-10);
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// L2-normalize a mutable slice to unit norm. No-op for near-zero vectors.
fn normalize_inplace(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters for [`MindNetwork`].
#[derive(Debug, Clone)]
pub struct MindConfig {
    /// Total number of items in the catalogue (must be >= 1).
    pub n_items: usize,
    /// Embedding dimensionality (must be >= 1).
    pub embed_dim: usize,
    /// Number of interest capsules K (must be >= 1).
    pub n_interests: usize,
    /// Number of dynamic-routing iterations (typically 3; must be >= 1).
    pub n_routing_iters: usize,
    /// Maximum history length considered per user (must be >= 1).
    pub history_len: usize,
}

/// Multi-Interest Network with Dynamic Routing.
pub struct MindNetwork {
    /// Item embedding table: `[n_items × embed_dim]` (row-major).
    item_emb: Vec<f32>,
    /// Model configuration.
    cfg: MindConfig,
}

impl MindNetwork {
    /// Construct a new `MindNetwork` with Xavier-normal initialisation.
    ///
    /// # Errors
    /// - [`RecsysError::InvalidNumItems`] when `cfg.n_items == 0`.
    /// - [`RecsysError::InvalidEmbeddingDim`] when `cfg.embed_dim == 0`.
    /// - [`RecsysError::InvalidConfig`] when `n_interests`, `n_routing_iters`,
    ///   or `history_len` is zero.
    pub fn new(cfg: MindConfig, rng: &mut LcgRng) -> RecsysResult<Self> {
        if cfg.n_items == 0 {
            return Err(RecsysError::InvalidNumItems { n: 0 });
        }
        if cfg.embed_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: 0 });
        }
        if cfg.n_interests == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "n_interests must be >= 1".into(),
            });
        }
        if cfg.n_routing_iters == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "n_routing_iters must be >= 1".into(),
            });
        }
        if cfg.history_len == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "history_len must be >= 1".into(),
            });
        }

        let d = cfg.embed_dim;
        let scale = (1.0 / d as f32).sqrt();
        let item_emb: Vec<f32> = (0..cfg.n_items * d)
            .map(|_| rng.next_normal() * scale)
            .collect();

        Ok(Self { item_emb, cfg })
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Core: dynamic-routing capsule encoding
    // ──────────────────────────────────────────────────────────────────────────

    /// Encode a user's interaction history into `K` interest vectors via
    /// dynamic routing.
    ///
    /// Returns a flat `Vec<f32>` of length `n_interests × embed_dim` where the
    /// k-th interest capsule occupies `[k * embed_dim .. (k+1) * embed_dim]`.
    ///
    /// The history is truncated to `cfg.history_len` most-recent items if it
    /// exceeds that length.
    ///
    /// # Errors
    /// - [`RecsysError::EmptyInput`] when `history` is empty.
    /// - [`RecsysError::UnknownItem`] when any item id is out-of-bounds.
    pub fn encode_user(&self, history: &[usize]) -> RecsysResult<Vec<f32>> {
        if history.is_empty() {
            return Err(RecsysError::EmptyInput);
        }
        for &id in history {
            if id >= self.cfg.n_items {
                return Err(RecsysError::UnknownItem { id });
            }
        }

        // Truncate to history_len most-recent entries.
        let h_len = history.len().min(self.cfg.history_len);
        let history = &history[history.len() - h_len..];

        let d = self.cfg.embed_dim;
        let k_cap = self.cfg.n_interests;

        // Routing logits b[h][k], initialised to 0.
        // Stored as row-major [h_len × k_cap].
        let mut b = vec![0.0_f32; h_len * k_cap];

        // Capsule vectors [k_cap × d], row-major.
        let mut caps = vec![0.0_f32; k_cap * d];

        for _iter in 0..self.cfg.n_routing_iters {
            // Step A: c[h] = softmax(b[h]) over the K dimension.
            // c is [h_len × k_cap].
            let mut c = vec![0.0_f32; h_len * k_cap];
            for h in 0..h_len {
                c[h * k_cap..(h + 1) * k_cap].copy_from_slice(&b[h * k_cap..(h + 1) * k_cap]);
                softmax_inplace(&mut c[h * k_cap..(h + 1) * k_cap]);
            }

            // Step B: for each capsule k, aggregate:
            //   raw_k = Σ_h c[h][k] * item_emb[history[h]]
            // then normalize raw_k → caps[k].
            caps.iter_mut().for_each(|x| *x = 0.0);
            for h in 0..h_len {
                let item_id = history[h];
                let e_h = &self.item_emb[item_id * d..(item_id + 1) * d];
                for k in 0..k_cap {
                    let coef = c[h * k_cap + k];
                    for dim in 0..d {
                        caps[k * d + dim] += coef * e_h[dim];
                    }
                }
            }
            for k in 0..k_cap {
                normalize_inplace(&mut caps[k * d..(k + 1) * d]);
            }

            // Step C: update routing logits.
            //   b[h][k] += cap_k · item_emb[history[h]]
            for h in 0..h_len {
                let item_id = history[h];
                let e_h = &self.item_emb[item_id * d..(item_id + 1) * d];
                for k in 0..k_cap {
                    b[h * k_cap + k] += dot(&caps[k * d..(k + 1) * d], e_h);
                }
            }
        }

        Ok(caps)
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Scoring
    // ──────────────────────────────────────────────────────────────────────────

    /// Score a target item against all K interest vectors; return the maximum.
    ///
    /// # Errors
    /// Propagates errors from `encode_user` and checks `target_item`.
    pub fn score(&self, history: &[usize], target_item: usize) -> RecsysResult<f32> {
        if target_item >= self.cfg.n_items {
            return Err(RecsysError::UnknownItem { id: target_item });
        }
        let interests = self.encode_user(history)?;
        let d = self.cfg.embed_dim;
        let k_cap = self.cfg.n_interests;
        let target_emb = &self.item_emb[target_item * d..(target_item + 1) * d];

        let max_score = (0..k_cap)
            .map(|k| dot(&interests[k * d..(k + 1) * d], target_emb))
            .fold(f32::NEG_INFINITY, f32::max);

        Ok(max_score)
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Top-K retrieval
    // ──────────────────────────────────────────────────────────────────────────

    /// Given a user's history, score all items by `max-interest` score and
    /// return the top-`k` as `(item_id, score)` pairs, sorted descending.
    ///
    /// # Errors
    /// - [`RecsysError::InvalidK`] when `k == 0` or `k > n_items`.
    /// - Propagates errors from `encode_user`.
    pub fn top_k(&self, history: &[usize], k: usize) -> RecsysResult<Vec<(usize, f32)>> {
        if k == 0 || k > self.cfg.n_items {
            return Err(RecsysError::InvalidK {
                k,
                n: self.cfg.n_items,
            });
        }
        let interests = self.encode_user(history)?;
        let d = self.cfg.embed_dim;
        let k_cap = self.cfg.n_interests;

        // Score all items.
        let mut scores: Vec<(usize, f32)> = (0..self.cfg.n_items)
            .map(|item_id| {
                let target_emb = &self.item_emb[item_id * d..(item_id + 1) * d];
                let max_s = (0..k_cap)
                    .map(|ki| dot(&interests[ki * d..(ki + 1) * d], target_emb))
                    .fold(f32::NEG_INFINITY, f32::max);
                (item_id, max_s)
            })
            .collect();

        // Partial sort: bring top-k to the front.
        scores.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(k);
        scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scores)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn default_cfg() -> MindConfig {
        MindConfig {
            n_items: 20,
            embed_dim: 8,
            n_interests: 3,
            n_routing_iters: 3,
            history_len: 10,
        }
    }

    fn small_model(rng: &mut LcgRng) -> MindNetwork {
        MindNetwork::new(default_cfg(), rng).expect("MIND construction should succeed")
    }

    #[test]
    fn construction_succeeds() {
        let mut rng = make_rng();
        let model = small_model(&mut rng);
        assert_eq!(model.cfg.n_items, 20);
        assert_eq!(model.cfg.embed_dim, 8);
    }

    #[test]
    fn encode_user_output_shape() {
        let mut rng = make_rng();
        let model = small_model(&mut rng);
        let history = vec![0, 3, 7, 12];
        let interests = model
            .encode_user(&history)
            .expect("encode_user should succeed");
        assert_eq!(
            interests.len(),
            model.cfg.n_interests * model.cfg.embed_dim,
            "encode_user output length must be n_interests * embed_dim"
        );
    }

    #[test]
    fn encode_user_interests_unit_norm() {
        let mut rng = make_rng();
        let model = small_model(&mut rng);
        let history = vec![0, 3, 7, 12];
        let interests = model
            .encode_user(&history)
            .expect("encode_user should succeed");
        let d = model.cfg.embed_dim;
        let k_cap = model.cfg.n_interests;
        for k in 0..k_cap {
            let norm: f32 = interests[k * d..(k + 1) * d]
                .iter()
                .map(|&x| x * x)
                .sum::<f32>()
                .sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5 || norm < 1e-10,
                "interest capsule {k} must be unit-norm, got {norm:.6}"
            );
        }
    }

    #[test]
    fn score_is_finite() {
        let mut rng = make_rng();
        let model = small_model(&mut rng);
        let history = vec![0, 1, 2, 5];
        let s = model.score(&history, 10).expect("score should succeed");
        assert!(s.is_finite(), "MIND score must be finite, got {s}");
    }

    #[test]
    fn top_k_length_and_sorted() {
        let mut rng = make_rng();
        let model = small_model(&mut rng);
        let history = vec![0, 1, 5, 8, 15];
        let results = model.top_k(&history, 5).expect("top_k should succeed");
        assert_eq!(results.len(), 5, "top_k must return exactly 5 results");
        for w in results.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "top_k results must be sorted descending ({} < {})",
                w[0].1,
                w[1].1
            );
        }
    }

    #[test]
    fn err_encode_user_empty_history() {
        let mut rng = make_rng();
        let model = small_model(&mut rng);
        assert!(matches!(
            model.encode_user(&[]),
            Err(RecsysError::EmptyInput)
        ));
    }

    #[test]
    fn err_encode_user_unknown_item() {
        let mut rng = make_rng();
        let model = small_model(&mut rng);
        assert!(matches!(
            model.encode_user(&[0, 999]),
            Err(RecsysError::UnknownItem { .. })
        ));
    }

    #[test]
    fn err_score_unknown_target() {
        let mut rng = make_rng();
        let model = small_model(&mut rng);
        assert!(matches!(
            model.score(&[0, 1], 999),
            Err(RecsysError::UnknownItem { .. })
        ));
    }

    #[test]
    fn err_top_k_zero() {
        let mut rng = make_rng();
        let model = small_model(&mut rng);
        assert!(matches!(
            model.top_k(&[0, 1], 0),
            Err(RecsysError::InvalidK { .. })
        ));
    }

    #[test]
    fn err_top_k_exceeds_n_items() {
        let mut rng = make_rng();
        let model = small_model(&mut rng);
        // n_items = 20, requesting 100
        assert!(matches!(
            model.top_k(&[0, 1], 100),
            Err(RecsysError::InvalidK { .. })
        ));
    }

    #[test]
    fn err_n_items_zero() {
        let mut rng = make_rng();
        let cfg = MindConfig {
            n_items: 0,
            embed_dim: 8,
            n_interests: 2,
            n_routing_iters: 3,
            history_len: 5,
        };
        assert!(matches!(
            MindNetwork::new(cfg, &mut rng),
            Err(RecsysError::InvalidNumItems { .. })
        ));
    }

    #[test]
    fn err_embed_dim_zero() {
        let mut rng = make_rng();
        let cfg = MindConfig {
            n_items: 10,
            embed_dim: 0,
            n_interests: 2,
            n_routing_iters: 3,
            history_len: 5,
        };
        assert!(matches!(
            MindNetwork::new(cfg, &mut rng),
            Err(RecsysError::InvalidEmbeddingDim { .. })
        ));
    }

    #[test]
    fn err_n_interests_zero() {
        let mut rng = make_rng();
        let cfg = MindConfig {
            n_items: 10,
            embed_dim: 8,
            n_interests: 0,
            n_routing_iters: 3,
            history_len: 5,
        };
        assert!(matches!(
            MindNetwork::new(cfg, &mut rng),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn history_truncated_to_history_len() {
        let mut rng = make_rng();
        let cfg = MindConfig {
            n_items: 20,
            embed_dim: 8,
            n_interests: 2,
            n_routing_iters: 3,
            history_len: 3,
        };
        let model = MindNetwork::new(cfg, &mut rng).expect("construction should succeed");
        // Provide a history longer than history_len — should not error.
        let long_history: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let result = model.encode_user(&long_history);
        assert!(
            result.is_ok(),
            "long history should be truncated, not error"
        );
    }

    #[test]
    fn deterministic_same_seed() {
        let mut rng_a = LcgRng::new(17);
        let mut rng_b = LcgRng::new(17);
        let model_a = small_model(&mut rng_a);
        let model_b = small_model(&mut rng_b);
        let history = vec![2, 5, 9, 14];
        let ia = model_a
            .encode_user(&history)
            .expect("encode_user should succeed");
        let ib = model_b
            .encode_user(&history)
            .expect("encode_user should succeed");
        for (a, b) in ia.iter().zip(ib.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "same seed must yield identical capsules ({a} vs {b})"
            );
        }
    }
}
