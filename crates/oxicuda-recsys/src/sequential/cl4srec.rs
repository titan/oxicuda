//! CL4SRec — Contrastive Learning for Sequential Recommendation.
//!
//! Reference: Xu Xie, Fei Sun, Zhaoyang Liu, Shiwen Wu, Jinyang Gao, Jiandong
//! Zhang, Bolin Ding, Bin Cui, "Contrastive Learning for Sequential
//! Recommendation", ICDE 2022.
//!
//! # Idea
//!
//! Two stochastic **augmentations** are applied to a user's item-ID sequence to
//! produce two correlated *views*. An encoder maps each view to a sequence
//! embedding, and an **InfoNCE / NT-Xent** loss pulls the two views of the same
//! sequence together while pushing apart the views of *different* sequences in
//! the same mini-batch (in-batch negatives).
//!
//! Three augmentation operators are provided (Xie et al. §3.2):
//!
//! * **item crop** — keep a contiguous sub-sequence of length
//!   `max(1, ⌊η·L⌋)`;
//! * **item mask** — replace `⌊γ·L⌋` randomly-chosen positions with a reserved
//!   *mask* item id (`mask_id == n_items`, i.e. the extra row of the embedding
//!   table);
//! * **item reorder** — shuffle a contiguous span of length `max(1, ⌊β·L⌋)`.
//!
//! The encoder is a compact mean-pooling encoder over the (mask-augmented) item
//! embedding table; this keeps the InfoNCE behaviour analytic (identical views
//! ⇒ near-zero loss; a representation collapse ⇒ loss ≈ `ln(batch)`).

use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

/// Inner product of two equal-length slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// InfoNCE (NT-Xent) loss between two batches of view embeddings using
/// in-batch negatives and **cosine** similarity scaled by `temperature`.
///
/// For anchor `i` (taken from `view_a`) the positive is `view_b[i]` and the
/// negatives are `{ view_b[j] : j ≠ i }`. The loss is the mean over the batch
/// of `-log softmax_i( cos(a_i, b_·)/τ )`, which is always `≥ 0`. When all
/// embeddings collapse to a single direction every logit is equal and the loss
/// equals `ln(n_batch)`.
///
/// # Errors
/// - [`RecsysError::InvalidConfig`] when `temperature <= 0`.
/// - [`RecsysError::EmptyInput`] when `n_batch == 0` or `dim == 0`.
/// - [`RecsysError::DimensionMismatch`] when either view does not have exactly
///   `n_batch · dim` elements.
pub fn info_nce_loss(
    view_a: &[f32],
    view_b: &[f32],
    n_batch: usize,
    dim: usize,
    temperature: f32,
) -> RecsysResult<f32> {
    if temperature <= 0.0 {
        return Err(RecsysError::InvalidConfig {
            msg: "temperature must be > 0".into(),
        });
    }
    if n_batch == 0 || dim == 0 {
        return Err(RecsysError::EmptyInput);
    }
    if view_a.len() != n_batch * dim {
        return Err(RecsysError::DimensionMismatch {
            expected: n_batch * dim,
            got: view_a.len(),
        });
    }
    if view_b.len() != n_batch * dim {
        return Err(RecsysError::DimensionMismatch {
            expected: n_batch * dim,
            got: view_b.len(),
        });
    }

    let inv_tau = 1.0 / temperature;
    let mut loss = 0.0_f32;
    for i in 0..n_batch {
        let a = &view_a[i * dim..(i + 1) * dim];
        let na = dot(a, a).sqrt().max(1e-12);
        let mut logits = Vec::with_capacity(n_batch);
        for j in 0..n_batch {
            let b = &view_b[j * dim..(j + 1) * dim];
            let nb = dot(b, b).sqrt().max(1e-12);
            logits.push(dot(a, b) / (na * nb) * inv_tau);
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = max + logits.iter().map(|&l| (l - max).exp()).sum::<f32>().ln();
        loss += lse - logits[i];
    }
    Ok(loss / n_batch as f32)
}

/// Configuration for [`Cl4sRec`].
#[derive(Debug, Clone)]
pub struct Cl4sRecConfig {
    /// Number of real items (ids `0..n_items`); `mask_id == n_items`.
    pub n_items: usize,
    /// Embedding / sequence-representation width.
    pub embed_dim: usize,
    /// Crop keep-ratio `η ∈ (0, 1]`.
    pub crop_ratio: f32,
    /// Mask ratio `γ ∈ [0, 1)`.
    pub mask_ratio: f32,
    /// Reorder span-ratio `β ∈ (0, 1]`.
    pub reorder_ratio: f32,
    /// InfoNCE temperature `τ > 0`.
    pub temperature: f32,
}

impl Default for Cl4sRecConfig {
    fn default() -> Self {
        Self {
            n_items: 1,
            embed_dim: 1,
            crop_ratio: 0.6,
            mask_ratio: 0.3,
            reorder_ratio: 0.6,
            temperature: 0.2,
        }
    }
}

/// Three contrastive augmentation operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Augmentation {
    /// Keep a contiguous sub-sequence.
    Crop,
    /// Replace random positions with the mask id.
    Mask,
    /// Shuffle a contiguous span.
    Reorder,
}

/// CL4SRec contrastive sequential recommender (mean-pool encoder).
pub struct Cl4sRec {
    /// Configuration the model was built from.
    pub cfg: Cl4sRecConfig,
    /// Item embedding table including the mask row:
    /// `(n_items + 1) × embed_dim` (row-major). Row `n_items` is the mask
    /// embedding.
    pub item_emb: Vec<f32>,
}

impl Cl4sRec {
    /// Construct a CL4SRec with Kaiming-style normal initialisation.
    ///
    /// # Errors
    /// - [`RecsysError::InvalidNumItems`] when `n_items == 0`.
    /// - [`RecsysError::InvalidEmbeddingDim`] when `embed_dim == 0`.
    /// - [`RecsysError::InvalidConfig`] for an out-of-range ratio or a
    ///   non-positive temperature.
    pub fn new(cfg: Cl4sRecConfig, rng: &mut LcgRng) -> RecsysResult<Self> {
        if cfg.n_items == 0 {
            return Err(RecsysError::InvalidNumItems { n: 0 });
        }
        if cfg.embed_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: 0 });
        }
        Self::validate_ratios(&cfg)?;

        let rows = cfg.n_items + 1;
        let d = cfg.embed_dim;
        let scale = (1.0 / d as f32).sqrt();
        let item_emb: Vec<f32> = (0..rows * d).map(|_| rng.next_normal() * scale).collect();
        Ok(Self { cfg, item_emb })
    }

    fn validate_ratios(cfg: &Cl4sRecConfig) -> RecsysResult<()> {
        if !(cfg.crop_ratio > 0.0 && cfg.crop_ratio <= 1.0) {
            return Err(RecsysError::InvalidConfig {
                msg: "crop_ratio must be in (0, 1]".into(),
            });
        }
        if !(cfg.mask_ratio >= 0.0 && cfg.mask_ratio < 1.0) {
            return Err(RecsysError::InvalidConfig {
                msg: "mask_ratio must be in [0, 1)".into(),
            });
        }
        if !(cfg.reorder_ratio > 0.0 && cfg.reorder_ratio <= 1.0) {
            return Err(RecsysError::InvalidConfig {
                msg: "reorder_ratio must be in (0, 1]".into(),
            });
        }
        if cfg.temperature <= 0.0 {
            return Err(RecsysError::InvalidConfig {
                msg: "temperature must be > 0".into(),
            });
        }
        Ok(())
    }

    /// Reserved mask item id (`== n_items`).
    #[must_use]
    pub fn mask_id(&self) -> usize {
        self.cfg.n_items
    }

    /// Number of embedding-table rows (`n_items + 1`).
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.cfg.n_items + 1
    }

    /// Mean-pool encoder: average the embeddings of the (possibly
    /// mask-augmented) item ids into a single sequence representation.
    ///
    /// # Errors
    /// - [`RecsysError::EmptyInput`] for an empty sequence.
    /// - [`RecsysError::ItemOutOfBounds`] for any id `> mask_id`.
    pub fn encode(&self, seq: &[usize]) -> RecsysResult<Vec<f32>> {
        if seq.is_empty() {
            return Err(RecsysError::EmptyInput);
        }
        let d = self.cfg.embed_dim;
        let rows = self.n_rows();
        let mut acc = vec![0.0_f32; d];
        for &id in seq {
            if id >= rows {
                return Err(RecsysError::ItemOutOfBounds { idx: id, n: rows });
            }
            let e = self
                .item_emb
                .get(id * d..(id + 1) * d)
                .ok_or(RecsysError::ItemOutOfBounds { idx: id, n: rows })?;
            for (a, &v) in acc.iter_mut().zip(e.iter()) {
                *a += v;
            }
        }
        let inv = 1.0 / seq.len() as f32;
        for a in acc.iter_mut() {
            *a *= inv;
        }
        Ok(acc)
    }

    /// Encode a batch of sequences into a flat `n_batch · embed_dim` buffer.
    ///
    /// # Errors
    /// - [`RecsysError::EmptyInput`] when the batch is empty or any sequence is.
    /// - [`RecsysError::ItemOutOfBounds`] for any id `> mask_id`.
    pub fn encode_batch(&self, seqs: &[Vec<usize>]) -> RecsysResult<Vec<f32>> {
        if seqs.is_empty() {
            return Err(RecsysError::EmptyInput);
        }
        let d = self.cfg.embed_dim;
        let mut out = Vec::with_capacity(seqs.len() * d);
        for seq in seqs {
            out.extend_from_slice(&self.encode(seq)?);
        }
        Ok(out)
    }

    /// **Item crop**: keep a contiguous sub-sequence of length
    /// `max(1, ⌊η·L⌋)` starting at a random offset.
    ///
    /// # Errors
    /// [`RecsysError::EmptyInput`] for an empty sequence.
    pub fn item_crop(&self, seq: &[usize], rng: &mut LcgRng) -> RecsysResult<Vec<usize>> {
        if seq.is_empty() {
            return Err(RecsysError::EmptyInput);
        }
        let len = seq.len();
        let keep = ((self.cfg.crop_ratio * len as f32).floor() as usize).clamp(1, len);
        let start = rng.next_usize(len - keep + 1);
        Ok(seq[start..start + keep].to_vec())
    }

    /// **Item mask**: replace `⌊γ·L⌋` random positions with the mask id.
    ///
    /// # Errors
    /// [`RecsysError::EmptyInput`] for an empty sequence.
    pub fn item_mask(&self, seq: &[usize], rng: &mut LcgRng) -> RecsysResult<Vec<usize>> {
        if seq.is_empty() {
            return Err(RecsysError::EmptyInput);
        }
        let len = seq.len();
        let n_mask = ((self.cfg.mask_ratio * len as f32).floor() as usize).min(len);
        let mut out = seq.to_vec();
        let mask_id = self.mask_id();
        // Partial Fisher–Yates over a position index set picks `n_mask`
        // distinct positions without replacement.
        let mut idx: Vec<usize> = (0..len).collect();
        for k in 0..n_mask {
            let j = k + rng.next_usize(len - k);
            idx.swap(k, j);
            if let Some(slot) = idx.get(k).copied().and_then(|pos| out.get_mut(pos)) {
                *slot = mask_id;
            }
        }
        Ok(out)
    }

    /// **Item reorder**: shuffle a contiguous span of length
    /// `max(1, ⌊β·L⌋)` starting at a random offset; the multiset of ids is
    /// preserved.
    ///
    /// # Errors
    /// [`RecsysError::EmptyInput`] for an empty sequence.
    pub fn item_reorder(&self, seq: &[usize], rng: &mut LcgRng) -> RecsysResult<Vec<usize>> {
        if seq.is_empty() {
            return Err(RecsysError::EmptyInput);
        }
        let len = seq.len();
        let span = ((self.cfg.reorder_ratio * len as f32).floor() as usize).clamp(1, len);
        let start = rng.next_usize(len - span + 1);
        let mut out = seq.to_vec();
        // Fisher–Yates within `[start, start + span)`.
        for k in (1..span).rev() {
            let j = rng.next_usize(k + 1);
            out.swap(start + k, start + j);
        }
        Ok(out)
    }

    /// Apply one uniformly-chosen augmentation operator.
    ///
    /// # Errors
    /// [`RecsysError::EmptyInput`] for an empty sequence.
    pub fn augment(
        &self,
        seq: &[usize],
        rng: &mut LcgRng,
    ) -> RecsysResult<(Augmentation, Vec<usize>)> {
        match rng.next_usize(3) {
            0 => Ok((Augmentation::Crop, self.item_crop(seq, rng)?)),
            1 => Ok((Augmentation::Mask, self.item_mask(seq, rng)?)),
            _ => Ok((Augmentation::Reorder, self.item_reorder(seq, rng)?)),
        }
    }

    /// Full contrastive objective: build two augmented views of each input
    /// sequence, encode both, and return the InfoNCE loss with in-batch
    /// negatives at the configured temperature.
    ///
    /// # Errors
    /// - [`RecsysError::EmptyInput`] when `sequences` is empty.
    /// - Propagated augmentation / encoding / InfoNCE errors.
    pub fn contrastive_loss(
        &self,
        sequences: &[Vec<usize>],
        rng: &mut LcgRng,
    ) -> RecsysResult<f32> {
        if sequences.is_empty() {
            return Err(RecsysError::EmptyInput);
        }
        let d = self.cfg.embed_dim;
        let n = sequences.len();
        let mut view_a = Vec::with_capacity(n * d);
        let mut view_b = Vec::with_capacity(n * d);
        for seq in sequences {
            let (_, a) = self.augment(seq, rng)?;
            let (_, b) = self.augment(seq, rng)?;
            view_a.extend_from_slice(&self.encode(&a)?);
            view_b.extend_from_slice(&self.encode(&b)?);
        }
        info_nce_loss(&view_a, &view_b, n, d, self.cfg.temperature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn default_cfg() -> Cl4sRecConfig {
        Cl4sRecConfig {
            n_items: 8,
            embed_dim: 4,
            crop_ratio: 0.6,
            mask_ratio: 0.5,
            reorder_ratio: 0.6,
            temperature: 0.2,
        }
    }

    #[test]
    fn info_nce_non_negative() {
        let mut rng = LcgRng::new(1);
        let model = Cl4sRec::new(default_cfg(), &mut rng).expect("value should be present");
        let seqs = vec![vec![0_usize, 1, 2], vec![3, 4], vec![5, 6, 7]];
        let mut rng2 = LcgRng::new(2);
        let loss = model
            .contrastive_loss(&seqs, &mut rng2)
            .expect("contrastive_loss should succeed");
        assert!(loss >= 0.0, "InfoNCE must be >= 0, got {loss}");
    }

    #[test]
    fn info_nce_small_when_views_match_distinct_items() {
        // Three batch elements whose two views are identical and well
        // separated (orthogonal basis directions) → loss near 0.
        let dim = 3;
        let n = 3;
        let mut view_a = vec![0.0_f32; n * dim];
        for i in 0..n {
            view_a[i * dim + i] = 1.0;
        }
        let view_b = view_a.clone();
        let loss =
            info_nce_loss(&view_a, &view_b, n, dim, 0.1).expect("info_nce_loss should succeed");
        assert!(
            loss < 1e-2,
            "matching distinct views must give small loss, got {loss}"
        );
    }

    #[test]
    fn info_nce_collapse_equals_ln_batch() {
        // All embeddings identical → uniform soft-max → loss = ln(batch).
        let dim = 4;
        let n = 5;
        let view_a = vec![0.5_f32; n * dim];
        let view_b = vec![0.5_f32; n * dim];
        let loss =
            info_nce_loss(&view_a, &view_b, n, dim, 0.2).expect("info_nce_loss should succeed");
        let expected = (n as f32).ln();
        assert!(
            (loss - expected).abs() < 1e-4,
            "collapsed batch loss {loss} should equal ln({n}) = {expected}"
        );
    }

    #[test]
    fn crop_lengths_and_bounds() {
        let mut rng = LcgRng::new(5);
        let model = Cl4sRec::new(default_cfg(), &mut rng).expect("value should be present");
        let seq = vec![0_usize, 1, 2, 3, 4, 5];
        let expected_keep = ((0.6_f32 * 6.0).floor() as usize).clamp(1, 6);
        for s in 0..20 {
            let mut r = LcgRng::new(100 + s);
            let cropped = model
                .item_crop(&seq, &mut r)
                .expect("item_crop should succeed");
            assert_eq!(cropped.len(), expected_keep);
            assert!(cropped.iter().all(|&id| id < model.cfg.n_items));
        }
    }

    #[test]
    fn mask_replaces_with_mask_id() {
        let mut rng = LcgRng::new(6);
        let model = Cl4sRec::new(default_cfg(), &mut rng).expect("value should be present");
        let seq = vec![0_usize, 1, 2, 3, 4, 5];
        let n_mask = (0.5_f32 * 6.0).floor() as usize;
        for s in 0..20 {
            let mut r = LcgRng::new(200 + s);
            let masked = model
                .item_mask(&seq, &mut r)
                .expect("item_mask should succeed");
            assert_eq!(masked.len(), seq.len());
            let count = masked.iter().filter(|&&id| id == model.mask_id()).count();
            assert_eq!(count, n_mask, "exactly {n_mask} positions must be masked");
            // Non-masked positions keep their original id.
            for (orig, &got) in seq.iter().zip(masked.iter()) {
                assert!(got == *orig || got == model.mask_id());
            }
        }
    }

    #[test]
    fn reorder_preserves_multiset() {
        let mut rng = LcgRng::new(7);
        let model = Cl4sRec::new(default_cfg(), &mut rng).expect("value should be present");
        let seq = vec![0_usize, 1, 2, 3, 4, 5];
        for s in 0..20 {
            let mut r = LcgRng::new(300 + s);
            let reordered = model
                .item_reorder(&seq, &mut r)
                .expect("item_reorder should succeed");
            assert_eq!(reordered.len(), seq.len());
            let mut a = seq.clone();
            let mut b = reordered.clone();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "reorder must preserve the multiset of ids");
            assert!(reordered.iter().all(|&id| id < model.cfg.n_items));
        }
    }

    #[test]
    fn err_temperature_not_positive() {
        let dim = 2;
        let n = 2;
        let v = vec![1.0_f32; n * dim];
        assert!(matches!(
            info_nce_loss(&v, &v, n, dim, 0.0),
            Err(RecsysError::InvalidConfig { .. })
        ));
        assert!(matches!(
            info_nce_loss(&v, &v, n, dim, -1.0),
            Err(RecsysError::InvalidConfig { .. })
        ));
        let mut rng = LcgRng::new(9);
        let mut cfg = default_cfg();
        cfg.temperature = 0.0;
        assert!(matches!(
            Cl4sRec::new(cfg, &mut rng),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn err_shape_mismatch() {
        let v_ok = vec![1.0_f32; 6];
        let v_bad = vec![1.0_f32; 5];
        assert!(matches!(
            info_nce_loss(&v_bad, &v_ok, 3, 2, 0.2),
            Err(RecsysError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            info_nce_loss(&v_ok, &v_bad, 3, 2, 0.2),
            Err(RecsysError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_construction_validation() {
        let mut rng = LcgRng::new(10);
        let mut bad = default_cfg();
        bad.n_items = 0;
        assert!(matches!(
            Cl4sRec::new(bad, &mut rng),
            Err(RecsysError::InvalidNumItems { .. })
        ));
        let mut bad = default_cfg();
        bad.embed_dim = 0;
        assert!(matches!(
            Cl4sRec::new(bad, &mut rng),
            Err(RecsysError::InvalidEmbeddingDim { .. })
        ));
        let mut bad = default_cfg();
        bad.crop_ratio = 1.5;
        assert!(matches!(
            Cl4sRec::new(bad, &mut rng),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn encode_mean_pool_and_mask_row() {
        let mut rng = LcgRng::new(11);
        let model = Cl4sRec::new(default_cfg(), &mut rng).expect("value should be present");
        // Encoding a single id returns that row exactly.
        let e = model.encode(&[2]).expect("encode should succeed");
        let row = &model.item_emb[2 * 4..3 * 4];
        for (a, b) in e.iter().zip(row.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
        // The mask id is a valid encodable row.
        let m = model
            .encode(&[model.mask_id()])
            .expect("value should be present");
        assert_eq!(m.len(), 4);
        // Out-of-range id errors.
        assert!(matches!(
            model.encode(&[model.n_rows()]),
            Err(RecsysError::ItemOutOfBounds { .. })
        ));
        assert!(matches!(model.encode(&[]), Err(RecsysError::EmptyInput)));
    }

    #[test]
    fn single_item_sequence_augmentations_valid() {
        let mut rng = LcgRng::new(12);
        let model = Cl4sRec::new(default_cfg(), &mut rng).expect("value should be present");
        let seq = vec![3_usize];
        let mut r = LcgRng::new(99);
        assert_eq!(
            model
                .item_crop(&seq, &mut r)
                .expect("item_crop should succeed")
                .len(),
            1
        );
        assert_eq!(
            model
                .item_reorder(&seq, &mut r)
                .expect("item_reorder should succeed")
                .len(),
            1
        );
        let masked = model
            .item_mask(&seq, &mut r)
            .expect("item_mask should succeed");
        assert_eq!(masked.len(), 1);
    }
}
