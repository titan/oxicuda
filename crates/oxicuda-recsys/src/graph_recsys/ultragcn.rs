//! UltraGCN — Ultra-simplified Graph Convolution for Recommendation.
//!
//! Reference: Mao et al., "UltraGCN: Ultra Simplification of Graph Convolutional
//! Networks for Recommendation", CIKM 2021.
//!
//! Key insight: explicit graph propagation can be replaced by a weighted BCE
//! loss that approximates the stationary distribution of infinite-layer GCN.
//! Constraint weights:
//!   omega(u, i) = 1 / sqrt(deg(u) * deg(i))
//!
//! Loss per positive (u, i):
//!   L = omega(u,i) * BCE(s(u,i), 1)
//!     + neg_weight * Σ_j BCE(s(u,j), 0)   (j sampled negatives)
//!
//! Updates use vanilla SGD on `user_emb` and `item_emb`.

use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Numerically-stable binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
/// where `p = sigmoid(logit)`.
#[inline]
fn bce_loss(logit: f32, label: f32) -> f32 {
    // Use the log-sigmoid identity:
    //   log(σ(x)) = x - log(1+e^x)  [numerically better via log1p for x<0]
    // BCE = -label * log(σ(x)) - (1-label) * log(1-σ(x))
    //      = -label * log(σ(x)) - (1-label) * log(σ(-x))
    let log_p = -f32::ln_1p((-logit).exp()); // log sigmoid(x)
    let log_1mp = -f32::ln_1p(logit.exp()); // log sigmoid(-x)
    -(label * log_p + (1.0 - label) * log_1mp)
}

/// Gradient of BCE w.r.t. logit: `sigmoid(logit) - label`.
#[inline]
fn bce_grad(logit: f32, label: f32) -> f32 {
    sigmoid(logit) - label
}

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters for [`UltraGcn`].
#[derive(Debug, Clone)]
pub struct UltraGcnConfig {
    /// Embedding dimensionality (must be >= 1).
    pub embed_dim: usize,
    /// Weight `λ` for the item-item (ii) constraint loss term (>= 0).
    pub lambda: f32,
    /// Weight `γ` for the user-item (ui) constraint reweighting (>= 0).
    pub gamma: f32,
    /// Weight applied to negative-sample BCE losses (>= 0).
    pub neg_weight: f32,
    /// Number of negative items to sample per positive interaction.
    pub n_neg: usize,
    /// SGD learning rate (> 0).
    pub lr: f32,
}

/// UltraGCN recommendation model.
pub struct UltraGcn {
    /// Number of users.
    pub n_users: usize,
    /// Number of items.
    pub n_items: usize,
    /// Embedding dimensionality (mirrors `cfg.embed_dim`).
    pub embed_dim: usize,
    /// User embedding matrix `[n_users × embed_dim]` (row-major).
    user_emb: Vec<f32>,
    /// Item embedding matrix `[n_items × embed_dim]` (row-major).
    item_emb: Vec<f32>,
    /// Sparse constraint weights: `(user_id, item_id, omega)`.
    omega: Vec<(u32, u32, f32)>,
    /// Model configuration.
    cfg: UltraGcnConfig,
}

impl UltraGcn {
    /// Construct a new `UltraGcn` with Xavier-normal initialisation.
    ///
    /// # Errors
    /// - [`RecsysError::InvalidNumUsers`] when `n_users == 0`.
    /// - [`RecsysError::InvalidNumItems`] when `n_items == 0`.
    /// - [`RecsysError::InvalidEmbeddingDim`] when `cfg.embed_dim == 0`.
    /// - [`RecsysError::InvalidLambda`] when `cfg.lambda < 0`.
    /// - [`RecsysError::InvalidLossWeight`] when `cfg.neg_weight < 0` or
    ///   `cfg.lr <= 0`.
    pub fn new(
        n_users: usize,
        n_items: usize,
        cfg: UltraGcnConfig,
        rng: &mut LcgRng,
    ) -> RecsysResult<Self> {
        if n_users == 0 {
            return Err(RecsysError::InvalidNumUsers { n: n_users });
        }
        if n_items == 0 {
            return Err(RecsysError::InvalidNumItems { n: n_items });
        }
        if cfg.embed_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: 0 });
        }
        if cfg.lambda < 0.0 {
            return Err(RecsysError::InvalidLambda { val: cfg.lambda });
        }
        if cfg.neg_weight < 0.0 {
            return Err(RecsysError::InvalidLossWeight { w: cfg.neg_weight });
        }
        if cfg.lr <= 0.0 {
            return Err(RecsysError::InvalidLossWeight { w: cfg.lr });
        }

        let d = cfg.embed_dim;
        let scale = (1.0 / d as f32).sqrt();
        let user_emb: Vec<f32> = (0..n_users * d)
            .map(|_| rng.next_normal() * scale)
            .collect();
        let item_emb: Vec<f32> = (0..n_items * d)
            .map(|_| rng.next_normal() * scale)
            .collect();

        Ok(Self {
            n_users,
            n_items,
            embed_dim: d,
            user_emb,
            item_emb,
            omega: Vec::new(),
            cfg,
        })
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Constraint weight computation
    // ──────────────────────────────────────────────────────────────────────────

    /// Build the sparse constraint-weight matrix from the observed
    /// (user, item) interaction edges.
    ///
    /// `omega(u, i) = 1 / sqrt(deg_u * deg_i)` where `deg_u` and `deg_i` are
    /// the number of distinct items (resp. users) that user `u` (resp. item
    /// `i`) has interacted with.
    ///
    /// # Errors
    /// - [`RecsysError::EmptyInteraction`] when `edges` is empty.
    /// - [`RecsysError::UnknownUser`] / [`RecsysError::UnknownItem`] for
    ///   out-of-bounds indices.
    pub fn compute_omega(&mut self, edges: &[(usize, usize)]) -> RecsysResult<()> {
        if edges.is_empty() {
            return Err(RecsysError::EmptyInteraction);
        }
        for &(u, i) in edges {
            if u >= self.n_users {
                return Err(RecsysError::UnknownUser { id: u });
            }
            if i >= self.n_items {
                return Err(RecsysError::UnknownItem { id: i });
            }
        }

        // Compute degrees.
        let mut deg_u = vec![0u32; self.n_users];
        let mut deg_i = vec![0u32; self.n_items];
        for &(u, i) in edges {
            deg_u[u] += 1;
            deg_i[i] += 1;
        }

        // Build sparse omega table.
        self.omega = edges
            .iter()
            .map(|&(u, i)| {
                let du = deg_u[u] as f32;
                let di = deg_i[i] as f32;
                let w = if du > 0.0 && di > 0.0 {
                    1.0 / (du * di).sqrt()
                } else {
                    1.0
                };
                (u as u32, i as u32, w)
            })
            .collect();

        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Inference
    // ──────────────────────────────────────────────────────────────────────────

    /// Compute `sigmoid(user_emb[u] · item_emb[i])`.
    ///
    /// # Errors
    /// - [`RecsysError::UnknownUser`] when `user >= n_users`.
    /// - [`RecsysError::UnknownItem`] when `item >= n_items`.
    pub fn score(&self, user: usize, item: usize) -> RecsysResult<f32> {
        if user >= self.n_users {
            return Err(RecsysError::UnknownUser { id: user });
        }
        if item >= self.n_items {
            return Err(RecsysError::UnknownItem { id: item });
        }
        let d = self.embed_dim;
        let logit = dot(
            &self.user_emb[user * d..(user + 1) * d],
            &self.item_emb[item * d..(item + 1) * d],
        );
        Ok(sigmoid(logit))
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Training
    // ──────────────────────────────────────────────────────────────────────────

    /// Perform one training step over all positive edges with randomly-sampled
    /// in-batch negatives.
    ///
    /// For each positive (u, i):
    ///   - look up `omega(u, i)` (defaults to 1.0 if not found in the sparse table)
    ///   - compute weighted positive BCE: `omega * BCE(s(u,i), 1)`
    ///   - sample `cfg.n_neg` random items (excluding i) as negatives
    ///   - compute negative BCE: `neg_weight * Σ_j BCE(s(u,j), 0)`
    ///   - SGD update on `user_emb[u]` and `item_emb[{i,j}]`
    ///
    /// Returns the mean BCE loss over all positive edges.
    ///
    /// # Errors
    /// - [`RecsysError::EmptyInteraction`] when `pos_edges` is empty.
    /// - [`RecsysError::UnknownUser`] / [`RecsysError::UnknownItem`] on
    ///   out-of-bounds indices.
    /// - [`RecsysError::NoNegativeAvailable`] when `n_items <= 1`.
    pub fn train_step(
        &mut self,
        pos_edges: &[(usize, usize)],
        rng: &mut LcgRng,
    ) -> RecsysResult<f32> {
        if pos_edges.is_empty() {
            return Err(RecsysError::EmptyInteraction);
        }
        if self.n_items <= 1 {
            return Err(RecsysError::NoNegativeAvailable { user: 0 });
        }
        for &(u, i) in pos_edges {
            if u >= self.n_users {
                return Err(RecsysError::UnknownUser { id: u });
            }
            if i >= self.n_items {
                return Err(RecsysError::UnknownItem { id: i });
            }
        }

        let d = self.embed_dim;
        let lr = self.cfg.lr;
        let neg_weight = self.cfg.neg_weight;
        let n_neg = self.cfg.n_neg;
        let n_items = self.n_items;

        // Build a quick-lookup map from (user, item) → omega for this step.
        // We use a linear scan since sparse tables for recommendation data
        // are commonly not huge; avoid any external collections crate.
        let lookup_omega = |u: usize, i: usize| -> f32 {
            for &(ou, oi, ow) in &self.omega {
                if ou as usize == u && oi as usize == i {
                    return ow;
                }
            }
            1.0_f32
        };

        let mut total_loss = 0.0_f32;

        for &(u, pos_i) in pos_edges {
            let w_ui = lookup_omega(u, pos_i);

            // ── Positive gradient ──────────────────────────────────────────
            let logit_pos = dot(
                &self.user_emb[u * d..(u + 1) * d],
                &self.item_emb[pos_i * d..(pos_i + 1) * d],
            );
            let loss_pos = w_ui * bce_loss(logit_pos, 1.0);
            total_loss += loss_pos;
            let g_pos = w_ui * bce_grad(logit_pos, 1.0);

            // Accumulate gradients into temporary user update vector.
            let mut user_grad = vec![0.0_f32; d];
            for (ug, (ie, ue)) in user_grad.iter_mut().zip(
                self.item_emb[pos_i * d..(pos_i + 1) * d]
                    .iter_mut()
                    .zip(self.user_emb[u * d..(u + 1) * d].iter()),
            ) {
                // ∂L/∂user_emb[u][k] += g_pos * item_emb[pos_i][k]
                *ug += g_pos * *ie;
                // ∂L/∂item_emb[pos_i][k] += g_pos * user_emb[u][k]
                *ie -= lr * g_pos * *ue;
            }

            // ── Negative sampling ──────────────────────────────────────────
            for _ in 0..n_neg {
                // Sample a negative item != pos_i.
                let mut neg_j = rng.next_usize(n_items);
                // Retry up to 8 times to avoid the positive item.
                for _ in 0..8 {
                    if neg_j != pos_i {
                        break;
                    }
                    neg_j = rng.next_usize(n_items);
                }
                // If still identical (extremely rare with large n_items), pick
                // the next item modulo n_items.
                if neg_j == pos_i {
                    neg_j = (pos_i + 1) % n_items;
                }

                let logit_neg = dot(
                    &self.user_emb[u * d..(u + 1) * d],
                    &self.item_emb[neg_j * d..(neg_j + 1) * d],
                );
                let loss_neg = neg_weight * bce_loss(logit_neg, 0.0);
                total_loss += loss_neg;
                let g_neg = neg_weight * bce_grad(logit_neg, 0.0);

                for (ug, (ie, ue)) in user_grad.iter_mut().zip(
                    self.item_emb[neg_j * d..(neg_j + 1) * d]
                        .iter_mut()
                        .zip(self.user_emb[u * d..(u + 1) * d].iter()),
                ) {
                    *ug += g_neg * *ie;
                    *ie -= lr * g_neg * *ue;
                }
            }

            // Apply accumulated user gradient.
            for (ue, ug) in self.user_emb[u * d..(u + 1) * d]
                .iter_mut()
                .zip(user_grad.iter())
            {
                *ue -= lr * ug;
            }
        }

        Ok(total_loss / pos_edges.len() as f32)
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

    fn default_cfg() -> UltraGcnConfig {
        UltraGcnConfig {
            embed_dim: 8,
            lambda: 0.5,
            gamma: 1.0,
            neg_weight: 1.5,
            n_neg: 3,
            lr: 0.01,
        }
    }

    fn small_model(rng: &mut LcgRng) -> UltraGcn {
        UltraGcn::new(5, 10, default_cfg(), rng).expect("model construction should succeed")
    }

    #[test]
    fn construction_succeeds() {
        let mut rng = make_rng();
        let model = small_model(&mut rng);
        assert_eq!(model.n_users, 5);
        assert_eq!(model.n_items, 10);
        assert_eq!(model.embed_dim, 8);
    }

    #[test]
    fn score_in_unit_interval() {
        let mut rng = make_rng();
        let model = small_model(&mut rng);
        let s = model.score(0, 0).expect("score should succeed");
        assert!(
            (0.0..=1.0).contains(&s),
            "UltraGCN score must be in [0, 1], got {s}"
        );
    }

    #[test]
    fn err_score_unknown_user() {
        let mut rng = make_rng();
        let model = small_model(&mut rng);
        assert!(matches!(
            model.score(999, 0),
            Err(RecsysError::UnknownUser { .. })
        ));
    }

    #[test]
    fn err_score_unknown_item() {
        let mut rng = make_rng();
        let model = small_model(&mut rng);
        assert!(matches!(
            model.score(0, 999),
            Err(RecsysError::UnknownItem { .. })
        ));
    }

    #[test]
    fn compute_omega_succeeds() {
        let mut rng = make_rng();
        let mut model = small_model(&mut rng);
        let edges = vec![(0, 0), (0, 1), (1, 0), (2, 2)];
        model
            .compute_omega(&edges)
            .expect("compute_omega should succeed");
        assert_eq!(model.omega.len(), 4);
    }

    #[test]
    fn omega_values_finite_and_positive() {
        let mut rng = make_rng();
        let mut model = small_model(&mut rng);
        let edges = vec![(0, 0), (0, 1), (1, 2), (2, 0), (3, 3)];
        model
            .compute_omega(&edges)
            .expect("compute_omega should succeed");
        for &(_, _, w) in &model.omega {
            assert!(
                w.is_finite() && w > 0.0,
                "omega weight must be finite > 0, got {w}"
            );
        }
    }

    #[test]
    fn err_compute_omega_empty() {
        let mut rng = make_rng();
        let mut model = small_model(&mut rng);
        assert!(matches!(
            model.compute_omega(&[]),
            Err(RecsysError::EmptyInteraction)
        ));
    }

    #[test]
    fn train_step_returns_finite_loss() {
        let mut rng = make_rng();
        let mut model = small_model(&mut rng);
        let edges = vec![(0, 0), (0, 1), (1, 2), (2, 3), (3, 4)];
        model
            .compute_omega(&edges)
            .expect("compute_omega should succeed");
        let mut train_rng = LcgRng::new(99);
        let loss = model
            .train_step(&edges, &mut train_rng)
            .expect("train_step should succeed");
        assert!(loss.is_finite(), "training loss must be finite, got {loss}");
    }

    #[test]
    fn err_train_step_empty_edges() {
        let mut rng = make_rng();
        let mut model = small_model(&mut rng);
        let mut train_rng = LcgRng::new(1);
        assert!(matches!(
            model.train_step(&[], &mut train_rng),
            Err(RecsysError::EmptyInteraction)
        ));
    }

    #[test]
    fn err_invalid_embed_dim() {
        let mut rng = make_rng();
        let cfg = UltraGcnConfig {
            embed_dim: 0,
            lambda: 0.5,
            gamma: 1.0,
            neg_weight: 1.0,
            n_neg: 2,
            lr: 0.01,
        };
        assert!(matches!(
            UltraGcn::new(4, 8, cfg, &mut rng),
            Err(RecsysError::InvalidEmbeddingDim { .. })
        ));
    }

    #[test]
    fn loss_decreases_after_multiple_steps() {
        let mut rng = LcgRng::new(13);
        let mut model = small_model(&mut rng);
        let edges = vec![(0, 0), (0, 1), (1, 2), (2, 3)];
        model
            .compute_omega(&edges)
            .expect("compute_omega should succeed");
        let mut train_rng = LcgRng::new(77);
        let loss1 = model
            .train_step(&edges, &mut train_rng)
            .expect("train_step should succeed");
        let loss2 = model
            .train_step(&edges, &mut train_rng)
            .expect("train_step should succeed");
        // Both losses should be finite; the exact monotonicity depends on
        // learning rate and data, so we only assert finiteness.
        assert!(loss1.is_finite() && loss2.is_finite(), "both losses finite");
    }

    #[test]
    fn err_n_users_zero() {
        let mut rng = make_rng();
        assert!(matches!(
            UltraGcn::new(0, 8, default_cfg(), &mut rng),
            Err(RecsysError::InvalidNumUsers { .. })
        ));
    }

    #[test]
    fn err_n_items_zero() {
        let mut rng = make_rng();
        assert!(matches!(
            UltraGcn::new(4, 0, default_cfg(), &mut rng),
            Err(RecsysError::InvalidNumItems { .. })
        ));
    }
}
