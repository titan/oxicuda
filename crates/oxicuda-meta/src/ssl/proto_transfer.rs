//! ProtoTransfer / ProtoCLR — Self-Supervised Prototypical Transfer Learning
//! (Medina, Devos & Grossglauser 2020: "Self-Supervised Prototypical Transfer
//! Learning for Few-Shot Classification").
//!
//! ProtoTransfer pre-trains a backbone *without labels* using a contrastive,
//! prototype-based objective called **ProtoCLR**, then transfers the frozen
//! embedding to a downstream few-shot episode where a [`crate::metric_learning`]
//! ProtoNet classifier is applied (the "ProtoTune"/ProtoNet transfer stage).
//!
//! # ProtoCLR objective
//!
//! Each *instance* of an unlabelled batch is treated as its own class.  For
//! instance `i` we are given:
//!
//! * one **prototype view** `a_i` (e.g. the un-augmented or weakly-augmented
//!   image), and
//! * one or more **query views** `q_{i,r}` (strong augmentations of the same
//!   instance).
//!
//! The backbone maps every view to an embedding which is **L2-normalised**.
//! Per-instance prototypes are the embeddings of the prototype views.  The loss
//! pulls every query view towards the prototype of its own instance and pushes
//! it away from the prototypes of every other instance in the batch, via a
//! temperature-scaled softmax over the **negative squared Euclidean distance**
//! to all `B` prototypes:
//!
//! ```text
//!   p(c | q) = softmax_c( −‖q − a_c‖² / τ )
//!   L        = −(1 / (B·R)) Σ_i Σ_r log p(i | q_{i,r})
//! ```
//!
//! This is the prototypical analogue of NT-Xent / InfoNCE used by SimCLR, but
//! with class centroids (here single-shot prototypes) rather than raw pairwise
//! comparisons, matching Snell et al.'s prototypical formulation that the
//! downstream stage also uses — which is the property that makes the pretrained
//! features directly transferable to ProtoNet.
//!
//! This module provides a Pure-Rust, fully analytic implementation over a dense
//! linear embedding head (`embed = W x`), including:
//!
//! * [`l2_normalize`] — per-vector L2 normalisation with an `ε` guard;
//! * [`proto_clr_loss`] — the scalar ProtoCLR loss for a batch of views;
//! * [`ProtoTransferHead`] — a trainable embedding projection with an analytic
//!   ProtoCLR gradient step, plus a transfer helper that embeds a few-shot
//!   episode and classifies it with ProtoNet.

use crate::error::{MetaError, MetaResult};
use crate::handle::LcgRng;
use crate::metric_learning::proto_net::{compute_prototypes, proto_predict};

// ─────────────────────────────────────────────────────────────────────────────
// Normalisation + distance helpers
// ─────────────────────────────────────────────────────────────────────────────

/// L2-normalise a single embedding in place, guarding the denominator with `ε`.
/// Returns the normalised vector (a fresh allocation).
pub fn l2_normalize(v: &[f32], eps: f32) -> Vec<f32> {
    let norm = (v.iter().map(|&x| x * x).sum::<f32>() + eps).sqrt();
    v.iter().map(|&x| x / norm).collect()
}

/// Squared Euclidean distance between two equal-length vectors.
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

/// Row-major softmax over a slice of scores.
fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / scores.len() as f32; scores.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// ProtoCLR loss (functional form over precomputed embeddings)
// ─────────────────────────────────────────────────────────────────────────────

/// ProtoCLR contrastive loss over a batch of `b` instances.
///
/// * `prototypes`: `b · embed_dim` row-major — one prototype embedding per
///   instance (`a_i`).
/// * `queries`: `b · r · embed_dim` row-major — `r` query embeddings per
///   instance (`q_{i,0..r}`), grouped by instance.
/// * `temperature`: `τ > 0` — the softmax temperature on the negative squared
///   distance.
///
/// Embeddings are used **as supplied** (the caller is expected to L2-normalise
/// when emulating the paper).  Returns the mean cross-entropy of assigning each
/// query view to its own instance's prototype.
///
/// # Errors
/// * [`MetaError::InvalidFeatDim`] if `embed_dim == 0`.
/// * [`MetaError::InvalidNWay`] if `b < 2` (a contrastive loss needs negatives).
/// * [`MetaError::InvalidLr`] if `temperature <= 0` or non-finite.
/// * [`MetaError::DimensionMismatch`] if the buffer lengths are inconsistent.
pub fn proto_clr_loss(
    prototypes: &[f32],
    queries: &[f32],
    b: usize,
    r: usize,
    embed_dim: usize,
    temperature: f32,
) -> MetaResult<f32> {
    if embed_dim == 0 {
        return Err(MetaError::InvalidFeatDim { dim: embed_dim });
    }
    if b < 2 {
        return Err(MetaError::InvalidNWay { n_way: b });
    }
    if r == 0 {
        return Err(MetaError::InvalidQuerySize { size: r });
    }
    if temperature <= 0.0 || !temperature.is_finite() {
        return Err(MetaError::InvalidLr { lr: temperature });
    }
    if prototypes.len() != b * embed_dim {
        return Err(MetaError::DimensionMismatch {
            expected: b * embed_dim,
            got: prototypes.len(),
        });
    }
    if queries.len() != b * r * embed_dim {
        return Err(MetaError::DimensionMismatch {
            expected: b * r * embed_dim,
            got: queries.len(),
        });
    }

    let mut total = 0.0_f32;
    let n_queries = b * r;
    for qi in 0..n_queries {
        let inst = qi / r;
        let q = &queries[qi * embed_dim..(qi + 1) * embed_dim];
        // Scores over all prototypes: −‖q − a_c‖² / τ.
        let mut scores = vec![0.0_f32; b];
        for c in 0..b {
            let proto = &prototypes[c * embed_dim..(c + 1) * embed_dim];
            scores[c] = -sq_dist(q, proto) / temperature;
        }
        let probs = softmax(&scores);
        let p = probs[inst].max(1e-30);
        let lp = p.ln();
        if !lp.is_finite() {
            return Err(MetaError::NanEncountered {
                context: "proto_clr log-prob non-finite".into(),
            });
        }
        total -= lp;
    }
    Ok(total / n_queries as f32)
}

// ─────────────────────────────────────────────────────────────────────────────
// Trainable embedding head
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a [`ProtoTransferHead`].
#[derive(Debug, Clone)]
pub struct ProtoTransferConfig {
    /// Dimensionality of the raw input fed to the embedding projection.
    pub in_dim: usize,
    /// Embedding dimensionality produced by the projection.
    pub embed_dim: usize,
    /// Softmax temperature `τ > 0` for the ProtoCLR loss.
    pub temperature: f32,
    /// ε guard for L2 normalisation.
    pub norm_eps: f32,
}

/// A dense linear embedding projection `embed = W x` (no bias — ProtoCLR
/// operates on the *direction* of L2-normalised embeddings, so an additive bias
/// is redundant) with an analytic ProtoCLR gradient step.
pub struct ProtoTransferHead {
    /// Projection weights `[embed_dim × in_dim]` row-major.
    w: Vec<f32>,
    cfg: ProtoTransferConfig,
}

impl ProtoTransferHead {
    /// Construct a Xavier-initialised embedding head.
    ///
    /// # Errors
    /// * [`MetaError::InvalidFeatDim`] if `in_dim == 0` or `embed_dim == 0`.
    /// * [`MetaError::InvalidLr`] if `temperature <= 0` or non-finite.
    pub fn new(cfg: ProtoTransferConfig, rng: &mut LcgRng) -> MetaResult<Self> {
        if cfg.in_dim == 0 {
            return Err(MetaError::InvalidFeatDim { dim: cfg.in_dim });
        }
        if cfg.embed_dim == 0 {
            return Err(MetaError::InvalidFeatDim { dim: cfg.embed_dim });
        }
        if cfg.temperature <= 0.0 || !cfg.temperature.is_finite() {
            return Err(MetaError::InvalidLr {
                lr: cfg.temperature,
            });
        }
        let limit = (6.0_f32 / (cfg.in_dim + cfg.embed_dim) as f32).sqrt();
        let mut w = vec![0.0_f32; cfg.embed_dim * cfg.in_dim];
        for v in w.iter_mut() {
            *v = (rng.next_f32() * 2.0 - 1.0) * limit;
        }
        Ok(Self { w, cfg })
    }

    /// Read-only access to the configuration.
    pub fn config(&self) -> &ProtoTransferConfig {
        &self.cfg
    }

    /// Read-only view of the projection weights (`[embed_dim × in_dim]`).
    pub fn weights(&self) -> &[f32] {
        &self.w
    }

    /// Raw (un-normalised) embedding `W x` of a single input.
    fn embed_raw(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.cfg.embed_dim];
        for (o, out_o) in out.iter_mut().enumerate() {
            let row = &self.w[o * self.cfg.in_dim..(o + 1) * self.cfg.in_dim];
            *out_o = row
                .iter()
                .zip(x.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f32>();
        }
        out
    }

    /// L2-normalised embedding `W x / ‖W x‖` of a single input.
    ///
    /// # Errors
    /// [`MetaError::DimensionMismatch`] if `x.len() != in_dim`.
    pub fn embed(&self, x: &[f32]) -> MetaResult<Vec<f32>> {
        if x.len() != self.cfg.in_dim {
            return Err(MetaError::DimensionMismatch {
                expected: self.cfg.in_dim,
                got: x.len(),
            });
        }
        Ok(l2_normalize(&self.embed_raw(x), self.cfg.norm_eps))
    }

    /// Embed every row of a flat `(n · in_dim)` batch into a flat
    /// `(n · embed_dim)` L2-normalised batch.
    fn embed_batch(&self, x_flat: &[f32], n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n * self.cfg.embed_dim);
        for i in 0..n {
            let x = &x_flat[i * self.cfg.in_dim..(i + 1) * self.cfg.in_dim];
            out.extend_from_slice(&l2_normalize(&self.embed_raw(x), self.cfg.norm_eps));
        }
        out
    }

    /// Evaluate the ProtoCLR loss of this head on a batch of un-augmented
    /// prototype inputs and their query (augmented) inputs.
    ///
    /// * `proto_x`: `b · in_dim` — one prototype input per instance.
    /// * `query_x`: `b · r · in_dim` — `r` query inputs per instance.
    ///
    /// # Errors
    /// Validation errors as in [`proto_clr_loss`], plus
    /// [`MetaError::DimensionMismatch`] for inconsistent input lengths.
    pub fn loss(&self, proto_x: &[f32], query_x: &[f32], b: usize, r: usize) -> MetaResult<f32> {
        self.check_batch(proto_x, query_x, b, r)?;
        let proto_emb = self.embed_batch(proto_x, b);
        let query_emb = self.embed_batch(query_x, b * r);
        proto_clr_loss(
            &proto_emb,
            &query_emb,
            b,
            r,
            self.cfg.embed_dim,
            self.cfg.temperature,
        )
    }

    fn check_batch(&self, proto_x: &[f32], query_x: &[f32], b: usize, r: usize) -> MetaResult<()> {
        if b < 2 {
            return Err(MetaError::InvalidNWay { n_way: b });
        }
        if r == 0 {
            return Err(MetaError::InvalidQuerySize { size: r });
        }
        if proto_x.len() != b * self.cfg.in_dim {
            return Err(MetaError::DimensionMismatch {
                expected: b * self.cfg.in_dim,
                got: proto_x.len(),
            });
        }
        if query_x.len() != b * r * self.cfg.in_dim {
            return Err(MetaError::DimensionMismatch {
                expected: b * r * self.cfg.in_dim,
                got: query_x.len(),
            });
        }
        Ok(())
    }

    /// One ProtoCLR gradient-descent step on the embedding weights, returning
    /// the scalar loss measured *before* the update.
    ///
    /// The gradient is computed analytically.  Writing `e = W x` and the
    /// L2-normalisation `ê = e / ‖e‖`, the chain rule through the per-view
    /// softmax-cross-entropy is propagated back through the normalisation
    /// Jacobian `∂ê/∂e = (I − ê êᵀ) / ‖e‖` and then through the linear map to the
    /// weights.  Both the query views and the prototype views receive gradients
    /// (a prototype `a_c` appears in the score of every query, so it accumulates
    /// contributions across the batch).
    ///
    /// # Errors
    /// Validation errors as in [`Self::loss`].
    pub fn proto_clr_step(
        &mut self,
        proto_x: &[f32],
        query_x: &[f32],
        b: usize,
        r: usize,
        lr: f32,
    ) -> MetaResult<f32> {
        self.check_batch(proto_x, query_x, b, r)?;
        if lr <= 0.0 || !lr.is_finite() {
            return Err(MetaError::InvalidLr { lr });
        }
        let embed_dim = self.cfg.embed_dim;
        let in_dim = self.cfg.in_dim;
        let tau = self.cfg.temperature;
        let eps = self.cfg.norm_eps;
        let inv_total = 1.0 / (b * r) as f32;

        // Forward: raw + normalised embeddings + per-vector norms.
        let mut proto_raw = Vec::with_capacity(b * embed_dim);
        let mut proto_norm = Vec::with_capacity(b);
        let mut proto_hat = Vec::with_capacity(b * embed_dim);
        for c in 0..b {
            let x = &proto_x[c * in_dim..(c + 1) * in_dim];
            let e = self.embed_raw(x);
            let n = (e.iter().map(|&v| v * v).sum::<f32>() + eps).sqrt();
            let hat: Vec<f32> = e.iter().map(|&v| v / n).collect();
            proto_raw.extend_from_slice(&e);
            proto_norm.push(n);
            proto_hat.extend_from_slice(&hat);
        }
        let n_queries = b * r;
        let mut query_raw = Vec::with_capacity(n_queries * embed_dim);
        let mut query_norm = Vec::with_capacity(n_queries);
        let mut query_hat = Vec::with_capacity(n_queries * embed_dim);
        for qi in 0..n_queries {
            let x = &query_x[qi * in_dim..(qi + 1) * in_dim];
            let e = self.embed_raw(x);
            let n = (e.iter().map(|&v| v * v).sum::<f32>() + eps).sqrt();
            let hat: Vec<f32> = e.iter().map(|&v| v / n).collect();
            query_raw.extend_from_slice(&e);
            query_norm.push(n);
            query_hat.extend_from_slice(&hat);
        }

        // Gradients w.r.t. the *normalised* embeddings (then mapped back through
        // the normalisation Jacobian and the linear weights).
        let mut d_proto_hat = vec![0.0_f32; b * embed_dim];
        let mut d_query_hat = vec![0.0_f32; n_queries * embed_dim];
        let mut loss = 0.0_f32;

        for qi in 0..n_queries {
            let inst = qi / r;
            let qh = &query_hat[qi * embed_dim..(qi + 1) * embed_dim];
            // scores_c = −‖qh − ph_c‖² / τ ; probs = softmax(scores).
            let mut scores = vec![0.0_f32; b];
            for c in 0..b {
                let ph = &proto_hat[c * embed_dim..(c + 1) * embed_dim];
                scores[c] = -sq_dist(qh, ph) / tau;
            }
            let probs = softmax(&scores);
            loss -= probs[inst].max(1e-30).ln() * inv_total;

            // dL/dscore_c = (probs_c − 1[c==inst]) · inv_total.
            // score_c = −‖qh − ph_c‖²/τ.
            //   ∂score_c/∂qh   = −2 (qh − ph_c) / τ
            //   ∂score_c/∂ph_c = +2 (qh − ph_c) / τ
            for c in 0..b {
                let g_score = (probs[c] - if c == inst { 1.0 } else { 0.0 }) * inv_total;
                let ph = &proto_hat[c * embed_dim..(c + 1) * embed_dim];
                let coef = g_score * 2.0 / tau;
                let dq = &mut d_query_hat[qi * embed_dim..(qi + 1) * embed_dim];
                for k in 0..embed_dim {
                    let diff = qh[k] - ph[k];
                    dq[k] += coef * (-diff);
                }
                let dp = &mut d_proto_hat[c * embed_dim..(c + 1) * embed_dim];
                for k in 0..embed_dim {
                    let diff = qh[k] - ph[k];
                    dp[k] += coef * diff;
                }
            }
        }

        // Back-prop through the normalisation Jacobian and accumulate weight
        // gradients.  For a view with raw embedding `e`, norm `n`, normalised
        // `ê = e/n`, upstream `d_hat`:
        //   d_e = (d_hat − (d_hat · ê) ê) / n
        //   dW[k, :] += d_e[k] · xᵀ
        let mut grad_w = vec![0.0_f32; embed_dim * in_dim];

        let accumulate =
            |raw: &[f32], norm: f32, hat: &[f32], d_hat: &[f32], x: &[f32], grad_w: &mut [f32]| {
                let _ = raw; // raw kept for symmetry/readability; norm+hat suffice.
                let dot: f32 = d_hat.iter().zip(hat.iter()).map(|(&a, &b)| a * b).sum();
                for k in 0..embed_dim {
                    let d_e = (d_hat[k] - dot * hat[k]) / norm;
                    if d_e == 0.0 {
                        continue;
                    }
                    let row = &mut grad_w[k * in_dim..(k + 1) * in_dim];
                    for (gw, &xi) in row.iter_mut().zip(x.iter()) {
                        *gw += d_e * xi;
                    }
                }
            };

        for qi in 0..n_queries {
            let x = &query_x[qi * in_dim..(qi + 1) * in_dim];
            accumulate(
                &query_raw[qi * embed_dim..(qi + 1) * embed_dim],
                query_norm[qi],
                &query_hat[qi * embed_dim..(qi + 1) * embed_dim],
                &d_query_hat[qi * embed_dim..(qi + 1) * embed_dim],
                x,
                &mut grad_w,
            );
        }
        for c in 0..b {
            let x = &proto_x[c * in_dim..(c + 1) * in_dim];
            accumulate(
                &proto_raw[c * embed_dim..(c + 1) * embed_dim],
                proto_norm[c],
                &proto_hat[c * embed_dim..(c + 1) * embed_dim],
                &d_proto_hat[c * embed_dim..(c + 1) * embed_dim],
                x,
                &mut grad_w,
            );
        }

        for (w, g) in self.w.iter_mut().zip(grad_w.iter()) {
            *w -= lr * g;
        }
        Ok(loss)
    }

    /// Transfer the pretrained embedding to a few-shot episode and classify the
    /// queries with ProtoNet over the embeddings.
    ///
    /// * `support_x`: `n_way · k_shot · in_dim` row-major raw support inputs.
    /// * `support_y`: `n_way · k_shot` labels in `0..n_way`.
    /// * `query_x`: `(n_query · in_dim)` row-major raw query inputs.
    ///
    /// Returns the predicted class for each query.
    ///
    /// # Errors
    /// * [`MetaError::InvalidNWay`] / [`MetaError::InvalidKShot`] for degenerate
    ///   episode shapes.
    /// * [`MetaError::DimensionMismatch`] for inconsistent buffer lengths.
    /// * any error from the ProtoNet stage.
    pub fn transfer_classify(
        &self,
        support_x: &[f32],
        support_y: &[u32],
        query_x: &[f32],
        n_way: usize,
        k_shot: usize,
    ) -> MetaResult<Vec<u32>> {
        if n_way < 2 {
            return Err(MetaError::InvalidNWay { n_way });
        }
        if k_shot == 0 {
            return Err(MetaError::InvalidKShot { k_shot });
        }
        let n_support = n_way * k_shot;
        if support_x.len() != n_support * self.cfg.in_dim {
            return Err(MetaError::DimensionMismatch {
                expected: n_support * self.cfg.in_dim,
                got: support_x.len(),
            });
        }
        if support_y.len() != n_support {
            return Err(MetaError::DimensionMismatch {
                expected: n_support,
                got: support_y.len(),
            });
        }
        if query_x.is_empty() || !query_x.len().is_multiple_of(self.cfg.in_dim) {
            return Err(MetaError::DimensionMismatch {
                expected: self.cfg.in_dim,
                got: query_x.len(),
            });
        }
        let n_query = query_x.len() / self.cfg.in_dim;

        let support_emb = self.embed_batch(support_x, n_support);
        let query_emb = self.embed_batch(query_x, n_query);
        let protos =
            compute_prototypes(&support_emb, support_y, n_way, k_shot, self.cfg.embed_dim)?;
        proto_predict(&query_emb, &protos, n_way, self.cfg.embed_dim)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ProtoTransferConfig {
        ProtoTransferConfig {
            in_dim: 8,
            embed_dim: 6,
            temperature: 0.5,
            norm_eps: 1e-8,
        }
    }

    fn make() -> ProtoTransferHead {
        let mut rng = LcgRng::new(2026);
        ProtoTransferHead::new(cfg(), &mut rng).expect("valid ProtoTransfer cfg")
    }

    /// Build a ProtoCLR mini-batch where each instance is a distinct random
    /// centre and its query views are small jitters of that centre.
    fn make_batch(b: usize, r: usize, in_dim: usize, rng: &mut LcgRng) -> (Vec<f32>, Vec<f32>) {
        let mut centres = Vec::with_capacity(b * in_dim);
        for _ in 0..b * in_dim {
            centres.push(rng.next_f32() * 2.0 - 1.0);
        }
        let mut queries = Vec::with_capacity(b * r * in_dim);
        for c in 0..b {
            let centre = &centres[c * in_dim..(c + 1) * in_dim];
            for _ in 0..r {
                for &v in centre.iter() {
                    queries.push(v + (rng.next_f32() - 0.5) * 0.1);
                }
            }
        }
        (centres, queries)
    }

    // ── l2_normalize ─────────────────────────────────────────────────────────

    #[test]
    fn l2_normalize_unit_norm() {
        let v = vec![3.0_f32, 4.0];
        let n = l2_normalize(&v, 0.0);
        let len = (n[0] * n[0] + n[1] * n[1]).sqrt();
        assert!((len - 1.0).abs() < 1e-6);
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
    }

    // ── proto_clr_loss functional checks ─────────────────────────────────────

    #[test]
    fn proto_clr_loss_perfect_match_is_small() {
        // If each query equals its own prototype and prototypes are well
        // separated, the loss should be near zero.
        let embed_dim = 3;
        let b = 3;
        let r = 1;
        // Orthonormal prototypes.
        let prototypes = vec![
            1.0, 0.0, 0.0, // inst 0
            0.0, 1.0, 0.0, // inst 1
            0.0, 0.0, 1.0, // inst 2
        ];
        let queries = prototypes.clone();
        let loss = proto_clr_loss(&prototypes, &queries, b, r, embed_dim, 0.05).expect("loss ok");
        assert!(
            loss < 0.05,
            "perfect-match ProtoCLR loss should be small, got {loss}"
        );
    }

    #[test]
    fn proto_clr_loss_requires_two_instances() {
        let prototypes = vec![1.0_f32, 0.0];
        let queries = vec![1.0_f32, 0.0];
        assert!(matches!(
            proto_clr_loss(&prototypes, &queries, 1, 1, 2, 0.5),
            Err(MetaError::InvalidNWay { .. })
        ));
    }

    #[test]
    fn proto_clr_loss_bad_temperature_errs() {
        let prototypes = vec![1.0_f32, 0.0, 0.0, 1.0];
        let queries = prototypes.clone();
        assert!(matches!(
            proto_clr_loss(&prototypes, &queries, 2, 1, 2, 0.0),
            Err(MetaError::InvalidLr { .. })
        ));
    }

    #[test]
    fn proto_clr_loss_dim_mismatch_errs() {
        let prototypes = vec![1.0_f32, 0.0, 0.0, 1.0];
        let queries = vec![1.0_f32, 0.0];
        assert!(matches!(
            proto_clr_loss(&prototypes, &queries, 2, 1, 2, 0.5),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    // ── head construction validation ─────────────────────────────────────────

    #[test]
    fn new_valid_succeeds() {
        let mut rng = LcgRng::new(1);
        assert!(ProtoTransferHead::new(cfg(), &mut rng).is_ok());
    }

    #[test]
    fn new_zero_embed_errs() {
        let mut c = cfg();
        c.embed_dim = 0;
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            ProtoTransferHead::new(c, &mut rng),
            Err(MetaError::InvalidFeatDim { .. })
        ));
    }

    #[test]
    fn new_bad_temperature_errs() {
        let mut c = cfg();
        c.temperature = -1.0;
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            ProtoTransferHead::new(c, &mut rng),
            Err(MetaError::InvalidLr { .. })
        ));
    }

    #[test]
    fn embed_is_unit_norm() {
        let head = make();
        let mut rng = LcgRng::new(3);
        let x: Vec<f32> = (0..head.config().in_dim)
            .map(|_| rng.next_f32() * 2.0 - 1.0)
            .collect();
        let e = head.embed(&x).expect("embed ok");
        let n = e.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            (n - 1.0).abs() < 1e-4,
            "embedding must be L2-unit, got norm {n}"
        );
    }

    #[test]
    fn embed_wrong_dim_errs() {
        let head = make();
        assert!(matches!(
            head.embed(&[0.0; 2]),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    // ── training reduces the ProtoCLR loss ───────────────────────────────────

    #[test]
    fn proto_clr_step_reduces_loss() {
        let mut head = make();
        let mut rng = LcgRng::new(77);
        let b = 5;
        let r = 2;
        let (proto_x, query_x) = make_batch(b, r, head.config().in_dim, &mut rng);
        let first = head
            .proto_clr_step(&proto_x, &query_x, b, r, 0.5)
            .expect("step 1");
        let mut last = first;
        for _ in 0..60 {
            last = head
                .proto_clr_step(&proto_x, &query_x, b, r, 0.5)
                .expect("step");
        }
        assert!(
            last < first,
            "ProtoCLR training must reduce the contrastive loss: {first} -> {last}"
        );
        assert!(last.is_finite());
    }

    #[test]
    fn loss_matches_step_pre_loss() {
        // The loss returned by `loss` must equal the pre-update loss reported by
        // `proto_clr_step` for the same weights.
        let mut head = make();
        let mut rng = LcgRng::new(11);
        let b = 4;
        let r = 2;
        let (proto_x, query_x) = make_batch(b, r, head.config().in_dim, &mut rng);
        let l_eval = head.loss(&proto_x, &query_x, b, r).expect("loss");
        let l_step = head
            .proto_clr_step(&proto_x, &query_x, b, r, 0.1)
            .expect("step");
        assert!(
            (l_eval - l_step).abs() < 1e-5,
            "loss() {l_eval} should match proto_clr_step pre-loss {l_step}"
        );
    }

    #[test]
    fn proto_clr_step_bad_lr_errs() {
        let mut head = make();
        let mut rng = LcgRng::new(5);
        let (proto_x, query_x) = make_batch(3, 1, head.config().in_dim, &mut rng);
        assert!(matches!(
            head.proto_clr_step(&proto_x, &query_x, 3, 1, 0.0),
            Err(MetaError::InvalidLr { .. })
        ));
    }

    // ── numerical gradient check ─────────────────────────────────────────────

    #[test]
    fn analytic_gradient_matches_finite_difference() {
        // Verify the analytic ProtoCLR gradient (encoded by the weight update)
        // against a central finite-difference estimate of the loss gradient on a
        // single weight.
        let mut rng = LcgRng::new(404);
        let head = make();
        let b = 3;
        let r = 2;
        let (proto_x, query_x) = make_batch(b, r, head.config().in_dim, &mut rng);

        // Analytic gradient: take a step with a tiny lr and read back −Δw/lr per
        // weight is awkward; instead compute the gradient by replicating the
        // step's accumulation via two evaluations of `loss` around w[idx].
        let idx = 7usize;
        let h = 1e-3_f32;

        let mut head_plus = make();
        let mut head_minus = make();
        // Both `make()` use the same seed, so weights are identical to `head`.
        head_plus.w[idx] += h;
        head_minus.w[idx] -= h;
        let l_plus = head_plus.loss(&proto_x, &query_x, b, r).expect("l+");
        let l_minus = head_minus.loss(&proto_x, &query_x, b, r).expect("l-");
        let fd = (l_plus - l_minus) / (2.0 * h);

        // Analytic: a single step with lr `g` moves w[idx] by −lr·grad[idx]; do a
        // step with a small lr and recover grad[idx] = (w_before − w_after)/lr.
        let mut head_step = make();
        let lr = 1e-2_f32;
        let w_before = head_step.w[idx];
        head_step
            .proto_clr_step(&proto_x, &query_x, b, r, lr)
            .expect("step");
        let grad = (w_before - head_step.w[idx]) / lr;

        assert!(
            (grad - fd).abs() < 5e-2 * (1.0 + fd.abs()),
            "analytic grad {grad} vs finite-difference {fd}"
        );
    }

    // ── transfer to ProtoNet ─────────────────────────────────────────────────

    #[test]
    fn transfer_classify_separable_episode() {
        // Pre-train the head on a contrastive batch built from class centres,
        // then check that the transferred ProtoNet classifies a clean episode
        // drawn from the same well-separated centres.
        let mut head = make();
        let in_dim = head.config().in_dim;
        let n_way = 3;
        let k_shot = 2;
        let n_query = 3;

        // Distinct, well-separated class centres on coordinate axes.
        let mut centres = vec![0.0_f32; n_way * in_dim];
        for c in 0..n_way {
            centres[c * in_dim + c] = 1.0;
        }

        // Contrastive pretraining batch: treat the centres as instances.
        let mut rng = LcgRng::new(31);
        let r = 3;
        let mut query_views = Vec::new();
        for c in 0..n_way {
            let centre = &centres[c * in_dim..(c + 1) * in_dim];
            for _ in 0..r {
                for &v in centre.iter() {
                    query_views.push(v + (rng.next_f32() - 0.5) * 0.05);
                }
            }
        }
        for _ in 0..80 {
            head.proto_clr_step(&centres, &query_views, n_way, r, 0.3)
                .expect("pretrain step");
        }

        // Downstream episode from the same centres with small jitter.
        let mut support_x = Vec::new();
        let mut support_y = Vec::new();
        for c in 0..n_way {
            let centre = &centres[c * in_dim..(c + 1) * in_dim];
            for _ in 0..k_shot {
                for &v in centre.iter() {
                    support_x.push(v + (rng.next_f32() - 0.5) * 0.02);
                }
                support_y.push(c as u32);
            }
        }
        let mut query_x = Vec::new();
        let mut query_y = Vec::new();
        for c in 0..n_way {
            let centre = &centres[c * in_dim..(c + 1) * in_dim];
            for _ in 0..n_query {
                for &v in centre.iter() {
                    query_x.push(v + (rng.next_f32() - 0.5) * 0.02);
                }
                query_y.push(c as u32);
            }
        }

        let preds = head
            .transfer_classify(&support_x, &support_y, &query_x, n_way, k_shot)
            .expect("transfer classify");
        let correct = preds
            .iter()
            .zip(query_y.iter())
            .filter(|(p, t)| p == t)
            .count();
        let acc = correct as f32 / query_y.len() as f32;
        assert!(
            acc >= 0.8,
            "ProtoTransfer should classify a separable episode well, acc={acc}"
        );
    }

    #[test]
    fn transfer_classify_bad_shape_errs() {
        let head = make();
        let in_dim = head.config().in_dim;
        // n_way=1 invalid.
        assert!(matches!(
            head.transfer_classify(&vec![0.0; in_dim], &[0], &vec![0.0; in_dim], 1, 1),
            Err(MetaError::InvalidNWay { .. })
        ));
    }
}
