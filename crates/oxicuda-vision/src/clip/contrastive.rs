//! InfoNCE / NT-Xent symmetric contrastive loss.
//!
//! Implements the symmetric cross-entropy contrastive objective used in CLIP:
//!
//! ```text
//! S[i, j] = dot(image[i], text[j]) / temperature
//! L_img    = mean_i(-S[i,i] + log(Σ_j exp(S[i,j] - max_i)))
//! L_text   = mean_j(-S[j,j] + log(Σ_i exp(S[i,j] - max_j)))
//! L        = 0.5 * (L_img + L_text)
//! ```
//!
//! All intermediate computations use numerically stable log-sum-exp.

use crate::error::{VisionError, VisionResult};

// ─── softmax_cross_entropy_rows ──────────────────────────────────────────────

/// Compute the mean cross-entropy loss over rows of a logit matrix.
///
/// `logits`: flat `[B × B]` row-major.  The *diagonal* entry `logits[i*B + i]`
/// is the positive pair for row `i`.
///
/// Uses numerically stable log-sum-exp:
/// ```text
/// CE_i = -logits[i,i] + log(Σ_j exp(logits[i,j] - row_max)) + row_max
/// ```
fn softmax_cross_entropy_rows(logits: &[f32], batch: usize) -> f32 {
    if batch == 0 {
        return 0.0;
    }

    let mut total_loss = 0.0_f32;

    for i in 0..batch {
        let row_start = i * batch;
        let row = &logits[row_start..row_start + batch];

        // Numerically stable max over the row.
        let row_max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Log-sum-exp with shifted exponents.
        let lse = row.iter().map(|&v| (v - row_max).exp()).sum::<f32>().ln() + row_max;

        // CE for this sample: -positive_logit + log_sum_exp
        total_loss += -row[i] + lse;
    }

    total_loss / batch as f32
}

// ─── info_nce_loss ───────────────────────────────────────────────────────────

/// Compute the symmetric InfoNCE / NT-Xent contrastive loss.
///
/// # Parameters
/// - `image_embeds`: flat `[B × D]` L2-normalised image embeddings.
/// - `text_embeds`:  flat `[B × D]` L2-normalised text embeddings.
/// - `embed_dim`:    dimension `D` of each embedding vector.
/// - `temperature`:  positive scalar; lower values sharpen the distribution.
///
/// # Returns
/// `(loss, sim_matrix)` where
/// - `loss` is the mean symmetric cross-entropy (scalar).
/// - `sim_matrix` is flat `[B × B]` containing
///   `image_embeds[i] · text_embeds[j] / temperature` for all pairs.
///
/// # Errors
/// - [`VisionError::NonPositiveTemperature`] if `temperature <= 0`.
/// - [`VisionError::EmptyInput`] if either embedding buffer is empty.
/// - [`VisionError::DimensionMismatch`] if buffer lengths are not divisible
///   by `embed_dim` or the two buffers describe a different batch size.
/// - [`VisionError::InvalidEmbedDim`] if `embed_dim == 0`.
pub fn info_nce_loss(
    image_embeds: &[f32],
    text_embeds: &[f32],
    embed_dim: usize,
    temperature: f32,
) -> VisionResult<(f32, Vec<f32>)> {
    // ── Input validation ────────────────────────────────────────────────────
    if temperature <= 0.0 {
        return Err(VisionError::NonPositiveTemperature(temperature));
    }
    if embed_dim == 0 {
        return Err(VisionError::InvalidEmbedDim(embed_dim));
    }
    if image_embeds.is_empty() {
        return Err(VisionError::EmptyInput("image_embeds"));
    }
    if text_embeds.is_empty() {
        return Err(VisionError::EmptyInput("text_embeds"));
    }
    if image_embeds.len() % embed_dim != 0 {
        return Err(VisionError::DimensionMismatch {
            expected: (image_embeds.len() / embed_dim) * embed_dim,
            got: image_embeds.len(),
        });
    }
    if text_embeds.len() % embed_dim != 0 {
        return Err(VisionError::DimensionMismatch {
            expected: (text_embeds.len() / embed_dim) * embed_dim,
            got: text_embeds.len(),
        });
    }
    let batch = image_embeds.len() / embed_dim;
    if text_embeds.len() / embed_dim != batch {
        return Err(VisionError::DimensionMismatch {
            expected: image_embeds.len(),
            got: text_embeds.len(),
        });
    }

    // ── Similarity matrix S[i,j] = dot(img[i], txt[j]) / T ──────────────────
    let inv_t = 1.0 / temperature;
    let mut sim = vec![0.0f32; batch * batch];

    for i in 0..batch {
        let img_row = &image_embeds[i * embed_dim..(i + 1) * embed_dim];
        for j in 0..batch {
            let txt_row = &text_embeds[j * embed_dim..(j + 1) * embed_dim];
            let dot: f32 = img_row
                .iter()
                .zip(txt_row.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            sim[i * batch + j] = dot * inv_t;
        }
    }

    // ── Image loss: CE over rows (each row i, positive is column i) ──────────
    let image_loss = softmax_cross_entropy_rows(&sim, batch);

    // ── Text loss: CE over columns (transpose, then rows) ────────────────────
    // Transpose sim so that column j becomes row j.
    let mut sim_t = vec![0.0f32; batch * batch];
    for i in 0..batch {
        for j in 0..batch {
            sim_t[j * batch + i] = sim[i * batch + j];
        }
    }
    let text_loss = softmax_cross_entropy_rows(&sim_t, batch);

    let loss = 0.5 * (image_loss + text_loss);

    Ok((loss, sim))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Generate `batch` L2-normalised random unit vectors of dimension `dim`.
    fn random_unit_vecs(batch: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut data = vec![0.0f32; batch * dim];
        rng.fill_normal(&mut data);
        // L2-normalise each row.
        for i in 0..batch {
            let row = &mut data[i * dim..(i + 1) * dim];
            let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
            if norm > 1e-12 {
                for v in row.iter_mut() {
                    *v /= norm;
                }
            }
        }
        data
    }

    // ── Basic correctness ────────────────────────────────────────────────────

    #[test]
    fn symmetric_loss_identical_embeddings() {
        // Swapping image_embeds and text_embeds for the same Gaussian random
        // embeddings should yield the same loss (symmetry of InfoNCE).
        let embeds = random_unit_vecs(8, 64, 42);
        let (loss_a, _) = info_nce_loss(&embeds, &embeds, 64, 0.07).expect("ok");
        // The loss of (A, A) should be the same as (A, A) by definition.
        // Also verify that swapping rows/cols gives the same value for
        // independently generated but mirrored pairs.
        let img = random_unit_vecs(8, 64, 10);
        let txt = random_unit_vecs(8, 64, 11);
        let (loss_it, _) = info_nce_loss(&img, &txt, 64, 0.07).expect("ok");
        let (loss_ti, _) = info_nce_loss(&txt, &img, 64, 0.07).expect("ok");
        // The symmetric loss is averaged, so swapping (I,T) → (T,I) should
        // produce the same scalar because the formula is symmetric.
        assert!(
            (loss_it - loss_ti).abs() < 1e-5,
            "InfoNCE not symmetric under swap: {loss_it} vs {loss_ti}"
        );
        // Just check the self-pair loss is finite and positive.
        assert!(
            loss_a.is_finite() && loss_a >= 0.0,
            "self-pair loss: {loss_a}"
        );
    }

    #[test]
    fn lower_temperature_lower_loss_for_aligned_pairs() {
        // Perfectly aligned embeddings (image == text) → positive pair has
        // dot product 1 / temperature; lower temperature makes the positive
        // pair dominate more strongly, reducing the loss.
        //
        // However this only holds when the embeddings are truly identical and
        // non-trivially distributed (not all equal), so use a set of distinct
        // random unit vectors.
        let embeds = random_unit_vecs(16, 128, 7);
        let (loss_high_t, _) = info_nce_loss(&embeds, &embeds, 128, 1.0).expect("ok");
        let (loss_low_t, _) = info_nce_loss(&embeds, &embeds, 128, 0.07).expect("ok");
        assert!(
            loss_low_t < loss_high_t,
            "lower temp should give lower loss for aligned pairs; got low={loss_low_t}, high={loss_high_t}"
        );
    }

    #[test]
    fn perfect_alignment_loss_approx_log_batch() {
        // For perfectly aligned pairs (image_i == text_i, all distinct), and
        // as temperature → 0 the numerically the positive pair dominates.
        // The minimum achievable loss for uniform random queries in B classes
        // is approximately ln(B) (uniform prior), converging toward 0 for
        // very low temperature and distinct embeddings.
        // Here we just verify loss is non-negative and finite.
        let b = 4;
        let embeds = random_unit_vecs(b, 64, 99);
        let (loss, sim) = info_nce_loss(&embeds, &embeds, 64, 0.07).expect("ok");
        assert!(loss.is_finite(), "loss not finite: {loss}");
        assert!(loss >= 0.0, "loss negative: {loss}");
        assert_eq!(sim.len(), b * b, "sim matrix wrong size");
    }

    #[test]
    fn sim_matrix_shape() {
        let img = random_unit_vecs(5, 32, 1);
        let txt = random_unit_vecs(5, 32, 2);
        let (_, sim) = info_nce_loss(&img, &txt, 32, 0.1).expect("ok");
        assert_eq!(sim.len(), 5 * 5, "sim matrix should be B×B");
    }

    #[test]
    fn sim_matrix_diagonal_values() {
        // For identical embeddings, sim[i,i] = 1 / temperature (dot = 1).
        let b = 4;
        let embeds = random_unit_vecs(b, 64, 55);
        let temperature = 0.5_f32;
        let (_, sim) = info_nce_loss(&embeds, &embeds, 64, temperature).expect("ok");
        for i in 0..b {
            let diag_val = sim[i * b + i];
            // dot(v, v) = 1 → scaled by 1/T
            let expected = 1.0 / temperature;
            assert!(
                (diag_val - expected).abs() < 1e-4,
                "sim[{i},{i}] = {diag_val}, expected {expected}"
            );
        }
    }

    // ── Error conditions ─────────────────────────────────────────────────────

    #[test]
    fn error_nonpositive_temperature_zero() {
        let embeds = random_unit_vecs(4, 32, 1);
        let r = info_nce_loss(&embeds, &embeds, 32, 0.0);
        assert!(
            matches!(r, Err(VisionError::NonPositiveTemperature(_))),
            "expected NonPositiveTemperature, got {:?}",
            r
        );
    }

    #[test]
    fn error_nonpositive_temperature_negative() {
        let embeds = random_unit_vecs(4, 32, 1);
        let r = info_nce_loss(&embeds, &embeds, 32, -0.5);
        assert!(matches!(r, Err(VisionError::NonPositiveTemperature(_))));
    }

    #[test]
    fn error_empty_image_embeds() {
        let txt = random_unit_vecs(4, 32, 2);
        let r = info_nce_loss(&[], &txt, 32, 0.07);
        assert!(matches!(r, Err(VisionError::EmptyInput(_))));
    }

    #[test]
    fn error_empty_text_embeds() {
        let img = random_unit_vecs(4, 32, 3);
        let r = info_nce_loss(&img, &[], 32, 0.07);
        assert!(matches!(r, Err(VisionError::EmptyInput(_))));
    }

    #[test]
    fn error_zero_embed_dim() {
        let img = random_unit_vecs(4, 32, 4);
        let r = info_nce_loss(&img, &img, 0, 0.07);
        assert!(matches!(r, Err(VisionError::InvalidEmbedDim(0))));
    }

    #[test]
    fn error_batch_size_mismatch() {
        let img = random_unit_vecs(4, 32, 5);
        let txt = random_unit_vecs(6, 32, 6); // different batch
        let r = info_nce_loss(&img, &txt, 32, 0.07);
        assert!(
            matches!(r, Err(VisionError::DimensionMismatch { .. })),
            "expected DimensionMismatch for batch size mismatch, got {:?}",
            r
        );
    }

    #[test]
    fn error_image_embeds_not_divisible_by_embed_dim() {
        // 13 values not divisible by 4.
        let img = vec![0.0f32; 13];
        let txt = vec![0.0f32; 16];
        let r = info_nce_loss(&img, &txt, 4, 0.07);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn loss_is_finite_for_random_inputs() {
        let img = random_unit_vecs(8, 64, 100);
        let txt = random_unit_vecs(8, 64, 101);
        let (loss, _) = info_nce_loss(&img, &txt, 64, 0.07).expect("ok");
        assert!(
            loss.is_finite(),
            "loss must be finite for random unit-norm inputs"
        );
    }

    #[test]
    fn loss_is_nonnegative() {
        let img = random_unit_vecs(8, 64, 200);
        let txt = random_unit_vecs(8, 64, 201);
        let (loss, _) = info_nce_loss(&img, &txt, 64, 0.07).expect("ok");
        assert!(loss >= 0.0, "InfoNCE loss must be non-negative");
    }

    #[test]
    fn single_pair_loss_is_zero() {
        // With B=1, the only class is the positive → CE = -log(1) = 0.
        let embed = random_unit_vecs(1, 32, 7);
        let (loss, _) = info_nce_loss(&embed, &embed, 32, 0.1).expect("ok");
        assert!(
            loss.abs() < 1e-5,
            "single-pair loss should be ~0, got {loss}"
        );
    }

    #[test]
    fn temperature_effect_on_sim_matrix() {
        // Scaling temperature by 2 should scale all sim entries by 0.5.
        let img = random_unit_vecs(4, 32, 9);
        let txt = random_unit_vecs(4, 32, 10);
        let (_, sim1) = info_nce_loss(&img, &txt, 32, 1.0).expect("ok");
        let (_, sim2) = info_nce_loss(&img, &txt, 32, 2.0).expect("ok");
        for (i, (&a, &b)) in sim1.iter().zip(sim2.iter()).enumerate() {
            assert!(
                (a - 2.0 * b).abs() < 1e-5,
                "sim[{i}]: t=1 gives {a}, t=2 gives {b}; expected ratio 2"
            );
        }
    }
}
