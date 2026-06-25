//! DuoRec — Contrastive Learning for the Representation Degeneration Problem in
//! Sequential Recommendation.
//!
//! Reference: Ruihong Qiu, Zi Huang, Hongzhi Yin, Zijian Wang, "Contrastive
//! Learning for Representation Degeneration Problem in Sequential
//! Recommendation", WSDM 2022.
//!
//! # Idea
//!
//! Where CL4SRec augments the *input* sequence (crop / mask / reorder), DuoRec
//! observes that those edits can change sequence semantics, and instead builds
//! two correlated views at the **model level**:
//!
//! * **Unsupervised view** — feed the *same* sequence through the encoder twice
//!   with two independent **dropout** masks. The two views share semantics but
//!   differ in their stochastic feature subset (`x` and `x⁺`).
//! * **Supervised view** — two *different* sequences that share the same
//!   **next-target item** are treated as a semantic positive pair (a *duo*),
//!   pulling their representations together.
//!
//! Both objectives are NT-Xent / InfoNCE losses with in-batch negatives; their
//! sum (weighted by `λ`) is the DuoRec contrastive regulariser added on top of
//! the usual next-item prediction loss. Contrastive regularisation spreads the
//! sequence embeddings over the hypersphere and alleviates the *representation
//! degeneration* (anisotropy) that plagues softmax-trained sequence models.
//!
//! The encoder here is a compact mean-pooling encoder over an item-embedding
//! table (mirroring [`crate::sequential::cl4srec`]); dropout is applied to the
//! pooled representation. This keeps the contrastive behaviour analytic while
//! exercising the full DuoRec objective.

use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;
use crate::sequential::cl4srec::info_nce_loss;

/// Configuration for [`DuoRec`].
#[derive(Debug, Clone)]
pub struct DuoRecConfig {
    /// Number of items (ids `0..n_items`).
    pub n_items: usize,
    /// Embedding / sequence-representation width.
    pub embed_dim: usize,
    /// Feature-dropout probability `ρ ∈ [0, 1)` for the unsupervised views.
    pub dropout: f32,
    /// InfoNCE temperature `τ > 0`.
    pub temperature: f32,
    /// Weight `λ ≥ 0` of the supervised term relative to the unsupervised term
    /// in the combined DuoRec objective.
    pub lambda_sup: f32,
}

impl Default for DuoRecConfig {
    fn default() -> Self {
        Self {
            n_items: 1,
            embed_dim: 1,
            dropout: 0.2,
            temperature: 0.2,
            lambda_sup: 0.1,
        }
    }
}

/// DuoRec contrastive sequential recommender (mean-pool encoder + dropout).
#[derive(Debug, Clone)]
pub struct DuoRec {
    /// Configuration the model was built from.
    pub cfg: DuoRecConfig,
    /// Item embedding table `n_items × embed_dim` (row-major).
    pub item_emb: Vec<f32>,
}

impl DuoRec {
    /// Construct a DuoRec with `1/√d`-scaled normal initialisation.
    ///
    /// # Errors
    /// - [`RecsysError::InvalidNumItems`] when `n_items == 0`.
    /// - [`RecsysError::InvalidEmbeddingDim`] when `embed_dim == 0`.
    /// - [`RecsysError::InvalidConfig`] for an out-of-range dropout/temperature
    ///   or a negative `lambda_sup`.
    pub fn new(cfg: DuoRecConfig, rng: &mut LcgRng) -> RecsysResult<Self> {
        if cfg.n_items == 0 {
            return Err(RecsysError::InvalidNumItems { n: 0 });
        }
        if cfg.embed_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: 0 });
        }
        if !(cfg.dropout >= 0.0 && cfg.dropout < 1.0) {
            return Err(RecsysError::InvalidConfig {
                msg: "dropout must be in [0, 1)".into(),
            });
        }
        if cfg.temperature <= 0.0 {
            return Err(RecsysError::InvalidConfig {
                msg: "temperature must be > 0".into(),
            });
        }
        if cfg.lambda_sup < 0.0 {
            return Err(RecsysError::InvalidConfig {
                msg: "lambda_sup must be >= 0".into(),
            });
        }
        let d = cfg.embed_dim;
        let scale = (1.0 / d as f32).sqrt();
        let item_emb: Vec<f32> = (0..cfg.n_items * d)
            .map(|_| rng.next_normal() * scale)
            .collect();
        Ok(Self { cfg, item_emb })
    }

    /// Mean-pool encoder over the item embeddings (no dropout).
    ///
    /// # Errors
    /// - [`RecsysError::EmptyInput`] for an empty sequence.
    /// - [`RecsysError::ItemOutOfBounds`] for any id `>= n_items`.
    pub fn encode(&self, seq: &[usize]) -> RecsysResult<Vec<f32>> {
        if seq.is_empty() {
            return Err(RecsysError::EmptyInput);
        }
        let d = self.cfg.embed_dim;
        let mut acc = vec![0.0_f32; d];
        for &it in seq {
            if it >= self.cfg.n_items {
                return Err(RecsysError::ItemOutOfBounds {
                    idx: it,
                    n: self.cfg.n_items,
                });
            }
            let row = &self.item_emb[it * d..(it + 1) * d];
            for (a, &e) in acc.iter_mut().zip(row.iter()) {
                *a += e;
            }
        }
        let inv = 1.0 / seq.len() as f32;
        for a in &mut acc {
            *a *= inv;
        }
        Ok(acc)
    }

    /// Encode with inverted dropout: each feature is zeroed with probability
    /// `dropout` and surviving features are scaled by `1/(1−dropout)` so the
    /// expected activation is preserved. This is the *unsupervised* DuoRec view.
    ///
    /// # Errors
    /// Propagates [`Self::encode`].
    pub fn encode_dropout(&self, seq: &[usize], rng: &mut LcgRng) -> RecsysResult<Vec<f32>> {
        let mut h = self.encode(seq)?;
        let p = self.cfg.dropout;
        if p <= 0.0 {
            return Ok(h);
        }
        let scale = 1.0 / (1.0 - p);
        for x in &mut h {
            let u = rng.next_u32() as f64 / 2f64.powi(32);
            if (u as f32) < p {
                *x = 0.0;
            } else {
                *x *= scale;
            }
        }
        Ok(h)
    }

    /// Score a candidate `target_item` against an encoded sequence via dot
    /// product (the next-item logit before softmax).
    ///
    /// # Errors
    /// - [`RecsysError::ItemOutOfBounds`] if `target_item >= n_items`.
    /// - Propagates [`Self::encode`].
    pub fn score(&self, seq: &[usize], target_item: usize) -> RecsysResult<f32> {
        if target_item >= self.cfg.n_items {
            return Err(RecsysError::ItemOutOfBounds {
                idx: target_item,
                n: self.cfg.n_items,
            });
        }
        let h = self.encode(seq)?;
        let d = self.cfg.embed_dim;
        let row = &self.item_emb[target_item * d..(target_item + 1) * d];
        Ok(h.iter().zip(row.iter()).map(|(&a, &b)| a * b).sum())
    }

    /// **Unsupervised** DuoRec loss: build two dropout views of every sequence
    /// and return their in-batch InfoNCE. Identical (no-dropout) views collapse
    /// the loss toward 0 on a 1-element batch.
    ///
    /// # Errors
    /// - [`RecsysError::EmptyInput`] when `seqs` is empty.
    /// - Propagates [`Self::encode_dropout`] and [`info_nce_loss`].
    pub fn unsupervised_loss(&self, seqs: &[Vec<usize>], rng: &mut LcgRng) -> RecsysResult<f32> {
        if seqs.is_empty() {
            return Err(RecsysError::EmptyInput);
        }
        let d = self.cfg.embed_dim;
        let n = seqs.len();
        let mut view_a = Vec::with_capacity(n * d);
        let mut view_b = Vec::with_capacity(n * d);
        for s in seqs {
            view_a.extend(self.encode_dropout(s, rng)?);
            view_b.extend(self.encode_dropout(s, rng)?);
        }
        info_nce_loss(&view_a, &view_b, n, d, self.cfg.temperature)
    }

    /// **Supervised** DuoRec loss: sequences carrying the same `target` form
    /// semantic positives. For each anchor we pair it with another sequence that
    /// shares its target (a *duo*); when no such partner exists the anchor's own
    /// (dropout) view is used as its positive. Negatives are the remaining
    /// in-batch sequences.
    ///
    /// # Errors
    /// - [`RecsysError::EmptyInput`] when `seqs` is empty.
    /// - [`RecsysError::DimensionMismatch`] when `targets.len() != seqs.len()`.
    /// - Propagates the encoders and [`info_nce_loss`].
    pub fn supervised_loss(
        &self,
        seqs: &[Vec<usize>],
        targets: &[usize],
        rng: &mut LcgRng,
    ) -> RecsysResult<f32> {
        if seqs.is_empty() {
            return Err(RecsysError::EmptyInput);
        }
        if targets.len() != seqs.len() {
            return Err(RecsysError::DimensionMismatch {
                expected: seqs.len(),
                got: targets.len(),
            });
        }
        let d = self.cfg.embed_dim;
        let n = seqs.len();

        // Anchor views (dropout) and per-anchor supervised-positive views.
        let mut view_a = Vec::with_capacity(n * d);
        let mut view_b = Vec::with_capacity(n * d);
        for i in 0..n {
            view_a.extend(self.encode_dropout(&seqs[i], rng)?);
            // Find a partner j != i with the same target.
            let partner = (0..n).find(|&j| j != i && targets[j] == targets[i]);
            match partner {
                Some(j) => view_b.extend(self.encode_dropout(&seqs[j], rng)?),
                None => view_b.extend(self.encode_dropout(&seqs[i], rng)?),
            }
        }
        info_nce_loss(&view_a, &view_b, n, d, self.cfg.temperature)
    }

    /// Combined DuoRec contrastive objective
    /// `L = L_unsup + λ · L_sup` over a mini-batch.
    ///
    /// # Errors
    /// Propagates [`Self::unsupervised_loss`] and [`Self::supervised_loss`].
    pub fn duo_loss(
        &self,
        seqs: &[Vec<usize>],
        targets: &[usize],
        rng: &mut LcgRng,
    ) -> RecsysResult<f32> {
        let unsup = self.unsupervised_loss(seqs, rng)?;
        let sup = self.supervised_loss(seqs, targets, rng)?;
        Ok(unsup + self.cfg.lambda_sup * sup)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(seed: u64) -> DuoRec {
        let cfg = DuoRecConfig {
            n_items: 12,
            embed_dim: 8,
            dropout: 0.3,
            temperature: 0.2,
            lambda_sup: 0.5,
        };
        let mut rng = LcgRng::new(seed);
        DuoRec::new(cfg, &mut rng).expect("new ok")
    }

    #[test]
    fn rejects_invalid_config() {
        let mut rng = LcgRng::new(1);
        let bad = DuoRecConfig {
            n_items: 0,
            ..Default::default()
        };
        assert!(DuoRec::new(bad, &mut rng).is_err());
        let bad2 = DuoRecConfig {
            n_items: 4,
            embed_dim: 4,
            dropout: 1.0,
            ..Default::default()
        };
        assert!(DuoRec::new(bad2, &mut rng).is_err());
        let bad3 = DuoRecConfig {
            n_items: 4,
            embed_dim: 4,
            temperature: 0.0,
            ..Default::default()
        };
        assert!(DuoRec::new(bad3, &mut rng).is_err());
    }

    #[test]
    fn encode_rejects_oob_and_empty() {
        let m = model(2);
        assert!(m.encode(&[]).is_err());
        assert!(m.encode(&[99]).is_err());
        assert!(m.score(&[0, 1], 99).is_err());
    }

    #[test]
    fn no_dropout_is_deterministic_and_unscaled() {
        let cfg = DuoRecConfig {
            n_items: 6,
            embed_dim: 4,
            dropout: 0.0,
            temperature: 0.2,
            lambda_sup: 0.1,
        };
        let mut rng = LcgRng::new(9);
        let m = DuoRec::new(cfg, &mut rng).expect("ok");
        let plain = m.encode(&[1, 2, 3]).expect("enc");
        let mut r2 = LcgRng::new(123);
        let dropped = m.encode_dropout(&[1, 2, 3], &mut r2).expect("enc");
        for (a, b) in plain.iter().zip(dropped.iter()) {
            assert!((a - b).abs() < 1e-6, "no-dropout must match plain encode");
        }
    }

    #[test]
    fn dropout_preserves_expected_scale() {
        // Average over many masks of a constant vector ≈ original (inverted
        // dropout is unbiased in expectation).
        let m = model(3);
        let seq = vec![0usize, 1, 2, 3];
        let base = m.encode(&seq).expect("enc");
        let mut rng = LcgRng::new(77);
        let d = base.len();
        let mut acc = vec![0.0_f32; d];
        let trials = 5000usize;
        for _ in 0..trials {
            let v = m.encode_dropout(&seq, &mut rng).expect("enc");
            for (a, x) in acc.iter_mut().zip(v.iter()) {
                *a += x;
            }
        }
        for a in &mut acc {
            *a /= trials as f32;
        }
        for (mean, b) in acc.iter().zip(base.iter()) {
            assert!(
                (mean - b).abs() < 0.05_f32.max(b.abs() * 0.2),
                "inverted dropout mean {mean} should track base {b}"
            );
        }
    }

    #[test]
    fn losses_are_finite_and_nonnegative() {
        let m = model(4);
        let seqs = vec![
            vec![0usize, 1, 2],
            vec![3, 4, 5],
            vec![1, 2, 6],
            vec![7, 8, 9],
        ];
        let targets = vec![10usize, 11, 10, 11];
        let mut rng = LcgRng::new(555);
        let unsup = m.unsupervised_loss(&seqs, &mut rng).expect("unsup");
        let sup = m.supervised_loss(&seqs, &targets, &mut rng).expect("sup");
        let duo = m.duo_loss(&seqs, &targets, &mut rng).expect("duo");
        assert!(unsup.is_finite() && unsup >= -1e-5, "unsup {unsup}");
        assert!(sup.is_finite() && sup >= -1e-5, "sup {sup}");
        assert!(duo.is_finite() && duo >= -1e-5, "duo {duo}");
    }

    #[test]
    fn supervised_mismatch_errors() {
        let m = model(6);
        let seqs = vec![vec![0usize, 1], vec![2, 3]];
        let targets = vec![5usize];
        let mut rng = LcgRng::new(1);
        assert!(m.supervised_loss(&seqs, &targets, &mut rng).is_err());
        assert!(m.unsupervised_loss(&[], &mut rng).is_err());
    }
}
