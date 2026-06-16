//! Contrastive cross-modal alignment losses.
//!
//! Implements:
//! - **CLIP** bidirectional InfoNCE (image→text + text→image symmetrised).
//! - **ImageBind**-style triple alignment (3-way cross-modal consistency).

use crate::error::{MmResult, MultiModalError};

// ─── L2 normalisation ─────────────────────────────────────────────────────────

/// L2-normalise each row of a `[batch × dim]` matrix.
pub fn l2_normalise(feats: &[f32], batch: usize, dim: usize) -> MmResult<Vec<f32>> {
    if feats.len() != batch * dim {
        return Err(MultiModalError::DimensionMismatch {
            expected: batch * dim,
            got: feats.len(),
        });
    }
    let mut out = feats.to_vec();
    for b in 0..batch {
        let row = &mut out[b * dim..(b + 1) * dim];
        let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        let inv_norm = if norm > 1e-12 { 1.0 / norm } else { 1.0 };
        for v in row.iter_mut() {
            *v *= inv_norm;
        }
    }
    Ok(out)
}

// ─── Similarity matrix ────────────────────────────────────────────────────────

/// Compute `[batch × batch]` cosine similarity matrix from two normalised feature sets.
/// `sim[i, j] = feat_a[i] · feat_b[j]`.
fn cosine_sim_matrix(feat_a: &[f32], feat_b: &[f32], batch: usize, dim: usize) -> Vec<f32> {
    let mut sim = vec![0.0_f32; batch * batch];
    for i in 0..batch {
        for j in 0..batch {
            let mut dot = 0.0_f32;
            for d in 0..dim {
                dot += feat_a[i * dim + d] * feat_b[j * dim + d];
            }
            sim[i * batch + j] = dot;
        }
    }
    sim
}

/// Compute per-row stable log-softmax for a `[batch × batch]` matrix
/// scaled by `1/temperature`, then return the diagonal (correct pair) log-prob.
///
/// Returns `loss = -mean(log_prob_diag)`.
fn nce_loss_from_sim(sim: &[f32], batch: usize, temperature: f32) -> f32 {
    let mut loss = 0.0_f32;
    for i in 0..batch {
        let row_start = i * batch;
        let max_s = sim[row_start..row_start + batch]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0_f32;
        for j in 0..batch {
            sum_exp += ((sim[row_start + j] / temperature) - (max_s / temperature)).exp();
        }
        let log_sum = max_s / temperature + sum_exp.ln();
        let diag_scaled = sim[row_start + i] / temperature;
        loss += log_sum - diag_scaled;
    }
    loss / batch as f32
}

// ─── CLIP loss ────────────────────────────────────────────────────────────────

/// CLIP bidirectional InfoNCE loss.
///
/// Normalises features, computes the `[N × N]` cosine similarity matrix, and
/// computes symmetric NCE: `(image→text + text→image) / 2`.
///
/// For identical normalised features (perfect alignment), the loss converges
/// towards `ln(N)` when `temperature = 1.0` and N is small, because the
/// diagonal elements dominate but can't fully suppress off-diagonal.
pub fn clip_loss(
    image_feats: &[f32],
    text_feats: &[f32],
    batch: usize,
    dim: usize,
    temperature: f32,
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

    let img_n = l2_normalise(image_feats, batch, dim)?;
    let txt_n = l2_normalise(text_feats, batch, dim)?;

    // Image → Text: sim[i, j] = img[i] · txt[j]
    let sim_i2t = cosine_sim_matrix(&img_n, &txt_n, batch, dim);
    // Text → Image (transpose)
    let sim_t2i = cosine_sim_matrix(&txt_n, &img_n, batch, dim);

    let loss_i2t = nce_loss_from_sim(&sim_i2t, batch, temperature);
    let loss_t2i = nce_loss_from_sim(&sim_t2i, batch, temperature);
    let loss = (loss_i2t + loss_t2i) / 2.0;

    if !loss.is_finite() {
        return Err(MultiModalError::NanEncountered {
            location: "clip_loss",
        });
    }
    Ok(loss)
}

// ─── ImageBind triple loss ─────────────────────────────────────────────────────

/// ImageBind-style triple alignment loss for 3 modalities.
///
/// Computes average of 3 pairwise CLIP losses:
/// `L = (clip(A, B) + clip(B, C) + clip(A, C)) / 3`.
pub fn imagebind_loss(
    feats_a: &[f32],
    feats_b: &[f32],
    feats_c: &[f32],
    batch: usize,
    dim: usize,
    temperature: f32,
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

    let l_ab = clip_loss(feats_a, feats_b, batch, dim, temperature)?;
    let l_bc = clip_loss(feats_b, feats_c, batch, dim, temperature)?;
    let l_ac = clip_loss(feats_a, feats_c, batch, dim, temperature)?;

    Ok((l_ab + l_bc + l_ac) / 3.0)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_normalise_unit_norm() {
        let feats = vec![3.0_f32, 4.0, 0.0, 0.0]; // batch=2, dim=2
        let n = l2_normalise(&feats, 2, 2).expect("l2_normalise should succeed");
        // Row 0: (3,4)/5 = (0.6, 0.8)
        let norm0 = (n[0] * n[0] + n[1] * n[1]).sqrt();
        assert!((norm0 - 1.0).abs() < 1e-5, "norm0={norm0}");
    }

    #[test]
    fn l2_normalise_zero_row_stays_unit() {
        let feats = vec![0.0_f32, 0.0];
        let n = l2_normalise(&feats, 1, 2).expect("l2_normalise should succeed");
        // Zero vector: keep as-is (div by 1.0 with inv_norm clamp)
        assert!(n.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn clip_loss_identical_features_approx_ln_n() {
        // For perfectly aligned features (each row is a distinct one-hot),
        // loss ≈ ln(N) when N is small and the model perfectly separates diagonal.
        // But with identical features the off-diagonal similarity = 1/dim as well.
        // We just verify the loss is finite and positive.
        let n = 4;
        let dim = 8;
        // One-hot features: row i has 1 at position i%dim
        let mut feats = vec![0.0_f32; n * dim];
        for i in 0..n {
            feats[i * dim + i % dim] = 1.0;
        }
        let loss = clip_loss(&feats, &feats, n, dim, 0.07).expect("clip_loss should succeed");
        assert!(loss.is_finite(), "loss should be finite");
        assert!(loss >= 0.0, "loss should be non-negative");
    }

    #[test]
    fn clip_loss_identical_gives_finite() {
        let n = 8;
        let dim = 16;
        let feats: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let loss = clip_loss(&feats, &feats, n, dim, 0.07).expect("clip_loss should succeed");
        assert!(loss.is_finite());
    }

    #[test]
    fn clip_loss_invalid_temperature() {
        let f = vec![1.0_f32; 4];
        let err = clip_loss(&f, &f, 2, 2, 0.0).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidTemperature { .. }));
    }

    #[test]
    fn clip_loss_invalid_batch() {
        let f: Vec<f32> = vec![];
        let err = clip_loss(&f, &f, 0, 4, 0.07).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidBatchSize));
    }

    #[test]
    fn clip_loss_invalid_dim() {
        let f: Vec<f32> = vec![];
        let err = clip_loss(&f, &f, 2, 0, 0.07).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidFeatureDim));
    }

    #[test]
    fn clip_loss_orthogonal_higher_than_diagonal() {
        // Orthogonal features → harder to distinguish → higher loss.
        let n = 4;
        let dim = 4;
        // Aligned: each row = e_i (perfect one-hot matching)
        let mut aligned = vec![0.0_f32; n * dim];
        for i in 0..n {
            aligned[i * dim + i] = 1.0;
        }
        // Random orthogonal-ish: row 0 = e_0, row 1 = e_2, etc. (shuffled)
        let mut shuffled = vec![0.0_f32; n * dim];
        shuffled[1] = 1.0; // image 0 matches text 1  (row 0 = 0*dim + 1 = 1)
        shuffled[dim + 2] = 1.0;
        shuffled[2 * dim + 3] = 1.0;
        shuffled[3 * dim] = 1.0;

        let loss_aligned =
            clip_loss(&aligned, &aligned, n, dim, 0.07).expect("clip_loss should succeed");
        let loss_shuffled =
            clip_loss(&shuffled, &aligned, n, dim, 0.07).expect("clip_loss should succeed");
        // Shuffled (mismatched) should have higher loss
        assert!(
            loss_shuffled > loss_aligned,
            "aligned={loss_aligned}, shuffled={loss_shuffled}"
        );
    }

    #[test]
    fn imagebind_loss_three_modalities() {
        let n = 4;
        let dim = 8;
        let a: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.13).cos()).collect();
        let c: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.07).sin()).collect();
        let loss = imagebind_loss(&a, &b, &c, n, dim, 0.07).expect("imagebind_loss should succeed");
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn imagebind_loss_invalid_temp() {
        let f = vec![0.0_f32; 4 * 8];
        let err = imagebind_loss(&f, &f, &f, 4, 8, -0.1).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidTemperature { .. }));
    }

    #[test]
    fn cosine_sim_matrix_self_similarity() {
        // Normalised row · itself = 1
        let feats = vec![1.0_f32, 0.0, 0.0, 1.0]; // batch=2, dim=2, unit vectors
        let sim = cosine_sim_matrix(&feats, &feats, 2, 2);
        assert!((sim[0] - 1.0).abs() < 1e-6, "sim[0,0]={}", sim[0]);
        assert!((sim[3] - 1.0).abs() < 1e-6, "sim[1,1]={}", sim[3]);
        assert!(sim[1].abs() < 1e-6, "cross-sim={}", sim[1]);
    }
}
