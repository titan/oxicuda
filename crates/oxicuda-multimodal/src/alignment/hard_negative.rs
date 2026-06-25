//! Hard-negative mining for contrastive image-text alignment.
//!
//! Plain InfoNCE ([`crate::alignment::contrastive::clip_loss`]) treats every
//! off-diagonal entry of the similarity matrix as an equally-weighted negative.
//! Retrieval-oriented training (VSE++, Faghri 2018; ALBEF's hard-negative ITM,
//! Li 2021) instead focuses the gradient on the **hardest** negatives — the
//! mismatched pairs that the model currently scores *most* similar — because
//! those carry the most informative signal.
//!
//! This module provides two complementary, fully CPU-deterministic tools:
//!
//! 1. [`mine_hard_negatives`] — for each anchor row, return the column index of
//!    the highest-similarity *negative* (off-diagonal) entry. This is exactly the
//!    sampling step ALBEF uses to draw one hard negative per positive for the ITM
//!    head.
//! 2. [`hard_negative_infonce`] — the **top-k** InfoNCE variant: each row's
//!    denominator keeps only the positive plus the `k` hardest negatives, so easy
//!    negatives stop diluting the loss. With `k = batch - 1` it reduces exactly to
//!    the standard full-softmax InfoNCE.
//!
//! The "max-violation" hinge of VSE++ ([`vse_plus_plus_loss`]) is also provided:
//! `Σ_a max_n [ α − s(a,a) + s(a,n) ]_+` summed over both retrieval directions.

use crate::alignment::contrastive::l2_normalise;
use crate::error::{MmResult, MultiModalError};

/// Cosine similarity matrix `sim[i*batch + j] = a_i · b_j` for L2-normalised
/// inputs.
fn sim_matrix(a_norm: &[f32], b_norm: &[f32], batch: usize, dim: usize) -> Vec<f32> {
    let mut sim = vec![0.0_f32; batch * batch];
    for i in 0..batch {
        for j in 0..batch {
            let mut dot = 0.0_f32;
            for d in 0..dim {
                dot += a_norm[i * dim + d] * b_norm[j * dim + d];
            }
            sim[i * batch + j] = dot;
        }
    }
    sim
}

/// For each anchor row `i`, find the index `j != i` with the largest similarity
/// `image_i · text_j` — the current hardest negative text for that image.
///
/// Returns a `Vec<usize>` of length `batch`. For `batch == 1` (no possible
/// negative) the single entry is `0` (the anchor itself), since there is no
/// off-diagonal column.
///
/// # Errors
/// - [`MultiModalError::InvalidBatchSize`] when `batch == 0`.
/// - [`MultiModalError::InvalidFeatureDim`] when `dim == 0`.
/// - [`MultiModalError::DimensionMismatch`] when either feature buffer length is
///   not `batch * dim`.
pub fn mine_hard_negatives(
    image_feats: &[f32],
    text_feats: &[f32],
    batch: usize,
    dim: usize,
) -> MmResult<Vec<usize>> {
    if batch == 0 {
        return Err(MultiModalError::InvalidBatchSize);
    }
    if dim == 0 {
        return Err(MultiModalError::InvalidFeatureDim);
    }
    let img = l2_normalise(image_feats, batch, dim)?;
    let txt = l2_normalise(text_feats, batch, dim)?;
    let sim = sim_matrix(&img, &txt, batch, dim);

    let mut hardest = vec![0_usize; batch];
    for i in 0..batch {
        let mut best_j = i;
        let mut best_s = f32::NEG_INFINITY;
        for j in 0..batch {
            if j == i {
                continue;
            }
            let s = sim[i * batch + j];
            if s > best_s {
                best_s = s;
                best_j = j;
            }
        }
        hardest[i] = best_j;
    }
    Ok(hardest)
}

/// Top-k InfoNCE: bidirectional symmetric contrastive loss whose per-row
/// denominator keeps only the positive plus the `k` hardest negatives.
///
/// `k` is clamped to `[1, batch - 1]`. With `k = batch - 1` the result equals
/// the standard full-softmax CLIP loss; smaller `k` sharpens the gradient onto
/// the most confusable mismatches.
///
/// # Errors
/// - [`MultiModalError::InvalidTemperature`] when `temperature` is not finite
///   and positive.
/// - [`MultiModalError::InvalidBatchSize`] when `batch == 0`.
/// - [`MultiModalError::InvalidFeatureDim`] when `dim == 0`.
/// - [`MultiModalError::DimensionMismatch`] from the L2-normalisation step.
/// - [`MultiModalError::NanEncountered`] when the loss is non-finite.
pub fn hard_negative_infonce(
    image_feats: &[f32],
    text_feats: &[f32],
    batch: usize,
    dim: usize,
    temperature: f32,
    k: usize,
) -> MmResult<f32> {
    if temperature <= 0.0 || !temperature.is_finite() {
        return Err(MultiModalError::InvalidTemperature { temp: temperature });
    }
    if batch == 0 {
        return Err(MultiModalError::InvalidBatchSize);
    }
    if dim == 0 {
        return Err(MultiModalError::InvalidFeatureDim);
    }
    let img = l2_normalise(image_feats, batch, dim)?;
    let txt = l2_normalise(text_feats, batch, dim)?;
    let sim_i2t = sim_matrix(&img, &txt, batch, dim);
    let sim_t2i = sim_matrix(&txt, &img, batch, dim);

    let k_eff = k.clamp(1, batch.saturating_sub(1).max(1));
    let loss_i2t = topk_nce_direction(&sim_i2t, batch, temperature, k_eff);
    let loss_t2i = topk_nce_direction(&sim_t2i, batch, temperature, k_eff);
    let loss = 0.5 * (loss_i2t + loss_t2i);

    if !loss.is_finite() {
        return Err(MultiModalError::NanEncountered {
            location: "hard_negative_infonce",
        });
    }
    Ok(loss)
}

/// One direction of the top-k InfoNCE. For each row keep the diagonal positive
/// plus the `k` largest off-diagonal logits, then `-log softmax` over that
/// restricted support.
fn topk_nce_direction(sim: &[f32], batch: usize, temperature: f32, k: usize) -> f32 {
    let inv_t = 1.0 / temperature;
    let mut loss = 0.0_f32;
    let mut negs: Vec<f32> = Vec::with_capacity(batch);
    for i in 0..batch {
        let row = &sim[i * batch..(i + 1) * batch];
        let pos = row[i] * inv_t;

        negs.clear();
        for (j, &s) in row.iter().enumerate() {
            if j != i {
                negs.push(s * inv_t);
            }
        }
        // Descending sort so the first `k` are the hardest negatives.
        negs.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let kept = &negs[..k.min(negs.len())];

        // Stable log-sum-exp over {pos} ∪ kept.
        let mut max_logit = pos;
        for &s in kept {
            if s > max_logit {
                max_logit = s;
            }
        }
        let mut sum_exp = (pos - max_logit).exp();
        for &s in kept {
            sum_exp += (s - max_logit).exp();
        }
        let log_sum = max_logit + sum_exp.ln();
        loss += log_sum - pos;
    }
    loss / batch as f32
}

/// VSE++ max-of-hinges ranking loss (Faghri 2018), bidirectional.
///
/// For margin `alpha`, with `s(a, p)` the matched (diagonal) similarity and
/// `s(a, n)` a negative,
///
/// ```text
/// L = Σ_a max_n [ α − s(a,a) + s(a,n) ]_+   (image→text)
///   + Σ_t max_n [ α − s(t,t) + s(n,t) ]_+   (text→image)
/// ```
///
/// averaged over the batch. Only the single hardest violating negative per
/// anchor contributes, which is the defining feature of VSE++.
///
/// # Errors
/// Mirrors [`hard_negative_infonce`] minus the temperature check; additionally
/// returns [`MultiModalError::Internal`] when `alpha` is negative or non-finite.
pub fn vse_plus_plus_loss(
    image_feats: &[f32],
    text_feats: &[f32],
    batch: usize,
    dim: usize,
    alpha: f32,
) -> MmResult<f32> {
    if batch == 0 {
        return Err(MultiModalError::InvalidBatchSize);
    }
    if dim == 0 {
        return Err(MultiModalError::InvalidFeatureDim);
    }
    if alpha < 0.0 || !alpha.is_finite() {
        return Err(MultiModalError::Internal(
            "VSE++ margin must be non-negative and finite".to_string(),
        ));
    }
    let img = l2_normalise(image_feats, batch, dim)?;
    let txt = l2_normalise(text_feats, batch, dim)?;
    let sim = sim_matrix(&img, &txt, batch, dim);

    let mut loss = 0.0_f32;
    for i in 0..batch {
        let pos = sim[i * batch + i];
        // image→text: hardest text negative for image i (row i, columns).
        let mut worst_i2t = 0.0_f32;
        // text→image: hardest image negative for text i (column i, rows).
        let mut worst_t2i = 0.0_f32;
        for j in 0..batch {
            if j == i {
                continue;
            }
            let v_i2t = (alpha - pos + sim[i * batch + j]).max(0.0);
            if v_i2t > worst_i2t {
                worst_i2t = v_i2t;
            }
            let v_t2i = (alpha - pos + sim[j * batch + i]).max(0.0);
            if v_t2i > worst_t2i {
                worst_t2i = v_t2i;
            }
        }
        loss += worst_i2t + worst_t2i;
    }
    let loss = loss / batch as f32;
    if !loss.is_finite() {
        return Err(MultiModalError::NanEncountered {
            location: "vse_plus_plus_loss",
        });
    }
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::contrastive::clip_loss;
    use crate::handle::LcgRng;

    fn random_feats(batch: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut v = vec![0.0_f32; batch * dim];
        rng.fill_normal(&mut v);
        v
    }

    #[test]
    fn mines_planted_hard_negative() {
        // dim=3 one-hot anchors so the diagonal is the positive. We deliberately
        // make text row 1 align with image row 0 (besides its own positive), and
        // verify image 0's hardest negative is column 1.
        let batch = 3;
        let dim = 3;
        let image = vec![
            1.0, 0.0, 0.0, // img0 = e0
            0.0, 1.0, 0.0, // img1 = e1
            0.0, 0.0, 1.0, // img2 = e2
        ];
        let text = vec![
            1.0, 0.0, 0.0, // txt0 = e0 (positive for img0)
            0.9, 0.1, 0.0, // txt1 ≈ e0 → strong negative for img0
            0.0, 0.0, 1.0, // txt2 = e2
        ];
        let hard = mine_hard_negatives(&image, &text, batch, dim).expect("mine");
        assert_eq!(hard[0], 1, "image 0's hardest negative must be text 1");
    }

    #[test]
    fn hard_negatives_are_off_diagonal() {
        let (batch, dim) = (6, 8);
        let image = random_feats(batch, dim, 1);
        let text = random_feats(batch, dim, 2);
        let hard = mine_hard_negatives(&image, &text, batch, dim).expect("mine");
        for (i, &j) in hard.iter().enumerate() {
            assert_ne!(i, j, "anchor {i} mined itself as a negative");
        }
    }

    #[test]
    fn topk_full_equals_clip_loss() {
        // k = batch - 1 keeps every negative → identical to full-softmax InfoNCE.
        let (batch, dim) = (5, 10);
        let image = random_feats(batch, dim, 3);
        let text = random_feats(batch, dim, 4);
        let t = 0.07;
        let full = hard_negative_infonce(&image, &text, batch, dim, t, batch - 1).expect("hn");
        let clip = clip_loss(&image, &text, batch, dim, t).expect("clip");
        assert!(
            (full - clip).abs() < 1e-4,
            "topk-full {full} vs clip {clip}"
        );
    }

    #[test]
    fn fewer_negatives_lowers_loss() {
        // Keeping only the single hardest negative gives a smaller denominator
        // than keeping all of them → strictly smaller cross-entropy (for a batch
        // with >1 negative whose easy negatives still carry mass).
        let (batch, dim) = (8, 16);
        let image = random_feats(batch, dim, 5);
        let text = random_feats(batch, dim, 6);
        let t = 0.1;
        let k1 = hard_negative_infonce(&image, &text, batch, dim, t, 1).expect("k1");
        let kall = hard_negative_infonce(&image, &text, batch, dim, t, batch - 1).expect("kall");
        assert!(k1 <= kall + 1e-6, "k=1 ({k1}) must be ≤ k=all ({kall})");
        assert!(k1 < kall, "expected strict decrease, k1={k1} kall={kall}");
    }

    #[test]
    fn topk_deterministic() {
        let (batch, dim) = (4, 8);
        let image = random_feats(batch, dim, 7);
        let text = random_feats(batch, dim, 8);
        let a = hard_negative_infonce(&image, &text, batch, dim, 0.07, 2).expect("a");
        let b = hard_negative_infonce(&image, &text, batch, dim, 0.07, 2).expect("b");
        assert_eq!(a.to_bits(), b.to_bits());
    }

    #[test]
    fn vse_zero_when_well_separated() {
        // Perfectly aligned, orthogonal one-hot pairs: positive sim = 1, every
        // negative sim = 0. With α = 0.2 the hinge α − 1 + 0 = −0.8 < 0 → loss 0.
        let batch = 4;
        let dim = 4;
        let mut feats = vec![0.0_f32; batch * dim];
        for i in 0..batch {
            feats[i * dim + i] = 1.0;
        }
        let loss = vse_plus_plus_loss(&feats, &feats, batch, dim, 0.2).expect("vse");
        assert!(
            loss.abs() < 1e-6,
            "well-separated loss should be 0, got {loss}"
        );
    }

    #[test]
    fn vse_positive_when_confused() {
        // image 0 and text 1 are identical → s(0,1)=1=s(0,0): hinge = α > 0.
        let batch = 2;
        let dim = 2;
        let image = vec![1.0, 0.0, 0.0, 1.0];
        let text = vec![1.0, 0.0, 1.0, 0.0]; // txt1 == img0
        let loss = vse_plus_plus_loss(&image, &text, batch, dim, 0.2).expect("vse");
        assert!(loss > 0.0, "confused pair should produce a positive loss");
    }

    #[test]
    fn vse_negative_margin_errors() {
        let f = vec![0.0_f32; 2 * 4];
        assert!(matches!(
            vse_plus_plus_loss(&f, &f, 2, 4, -0.1),
            Err(MultiModalError::Internal(_))
        ));
    }

    #[test]
    fn mine_zero_batch_errors() {
        assert!(matches!(
            mine_hard_negatives(&[], &[], 0, 4),
            Err(MultiModalError::InvalidBatchSize)
        ));
    }

    #[test]
    fn infonce_invalid_temperature_errors() {
        let f = vec![0.0_f32; 2 * 4];
        assert!(matches!(
            hard_negative_infonce(&f, &f, 2, 4, 0.0, 1),
            Err(MultiModalError::InvalidTemperature { .. })
        ));
    }

    #[test]
    fn single_sample_batch_is_safe() {
        // batch == 1: no negatives. mine returns the anchor; top-k loss reduces to
        // -log(1) = 0 since the only logit is the positive.
        let dim = 4;
        let f = random_feats(1, dim, 11);
        let hard = mine_hard_negatives(&f, &f, 1, dim).expect("mine");
        assert_eq!(hard, vec![0]);
        let loss = hard_negative_infonce(&f, &f, 1, dim, 0.07, 1).expect("loss");
        assert!(
            loss.abs() < 1e-6,
            "single-sample loss should be ~0, got {loss}"
        );
    }
}
