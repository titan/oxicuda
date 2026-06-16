//! SP — Similarity-Preserving Knowledge Distillation (Tung & Mori 2019).
//!
//! Reference: Tung, F., & Mori, G. (2019). *Similarity-Preserving Knowledge
//! Distillation*. ICCV 2019. <https://arxiv.org/abs/1907.09682>
//!
//! SP enforces that input pairs which produce *similar* activations in the teacher
//! also produce similar activations in the student — i.e. it transfers the **pairwise
//! similarity structure of a mini-batch** instead of individual activations. For a
//! batch of `B` activation vectors stacked as `A ∈ ℝ^{B × d}`:
//!
//! ```text
//!   G = A · Aᵀ          (B × B batch-similarity / Gram matrix)
//!   G̃[i, :] = G[i, :] / ‖G[i, :]‖₂      (row-wise L2 normalisation)
//!   L_SP = (1 / B²) · ‖G̃ₛ − G̃_t‖²_F
//! ```
//!
//! Because the loss compares the `B × B` Gram matrices, the student and teacher may
//! have **different feature dimensions** `d` — only the batch size `B` must match.
//! The row normalisation makes the target invariant to the absolute activation scale,
//! so SP captures *relative* similarity rather than magnitude.

use crate::error::{DistillError, DistillResult};

const EPS: f32 = 1e-8;

/// Compute the row-normalised batch-similarity matrix `G̃` for a `batch × dim`
/// row-major activation matrix.
///
/// `G = A·Aᵀ` followed by L2-normalisation of each row. Returns a flat
/// `batch × batch` row-major matrix.
#[must_use]
pub fn similarity_matrix(activations: &[f32], batch: usize, dim: usize) -> Vec<f32> {
    let mut g = vec![0.0_f32; batch * batch];
    for i in 0..batch {
        let ai = &activations[i * dim..(i + 1) * dim];
        for j in 0..batch {
            let aj = &activations[j * dim..(j + 1) * dim];
            let dot: f32 = ai.iter().zip(aj.iter()).map(|(&a, &b)| a * b).sum();
            g[i * batch + j] = dot;
        }
    }
    // Row-wise L2 normalisation (Eq. 4 of the paper).
    for i in 0..batch {
        let row = &mut g[i * batch..(i + 1) * batch];
        let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt().max(EPS);
        for v in row.iter_mut() {
            *v /= norm;
        }
    }
    g
}

/// Similarity-Preserving loss between a student and teacher activation batch.
///
/// `student` is `batch × student_dim`, `teacher` is `batch × teacher_dim`; both are
/// row-major. The batch size must match; the feature dims may differ.
///
/// # Errors
///
/// - [`DistillError::EmptyInput`] if any input is empty or `batch == 0` /
///   `*_dim == 0`.
/// - [`DistillError::DimensionMismatch`] if a slice length disagrees with
///   `batch · dim`.
/// - [`DistillError::NumericalError`] if the result is non-finite.
pub fn sp_loss(
    student: &[f32],
    student_dim: usize,
    teacher: &[f32],
    teacher_dim: usize,
    batch: usize,
) -> DistillResult<f32> {
    if student.is_empty() || teacher.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if batch == 0 || student_dim == 0 || teacher_dim == 0 {
        return Err(DistillError::EmptyInput);
    }
    if student.len() != batch * student_dim {
        return Err(DistillError::DimensionMismatch {
            expected: batch * student_dim,
            got: student.len(),
        });
    }
    if teacher.len() != batch * teacher_dim {
        return Err(DistillError::DimensionMismatch {
            expected: batch * teacher_dim,
            got: teacher.len(),
        });
    }
    let g_s = similarity_matrix(student, batch, student_dim);
    let g_t = similarity_matrix(teacher, batch, teacher_dim);
    let fro_sq: f32 = g_s
        .iter()
        .zip(g_t.iter())
        .map(|(&s, &t)| (s - t) * (s - t))
        .sum();
    let loss = fro_sq / (batch as f32 * batch as f32);
    if !loss.is_finite() {
        return Err(DistillError::NumericalError {
            msg: "SP loss produced a non-finite value".into(),
        });
    }
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_rows_unit_norm() {
        let a: Vec<f32> = (0..6).map(|i| (i as f32) + 1.0).collect(); // 2×3
        let g = similarity_matrix(&a, 2, 3);
        for i in 0..2 {
            let norm: f32 = g[i * 2..(i + 1) * 2]
                .iter()
                .map(|&v| v * v)
                .sum::<f32>()
                .sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "row {i} norm = {norm}");
        }
    }

    #[test]
    fn identical_batches_zero_loss() {
        let a: Vec<f32> = (0..12).map(|i| (i as f32) * 0.4 - 1.0).collect(); // 4×3
        let loss = sp_loss(&a, 3, &a, 3, 4).expect("sp_loss should succeed");
        assert!(
            loss < 1e-6,
            "identical activations must give 0 SP loss, got {loss}"
        );
    }

    #[test]
    fn loss_nonneg() {
        let s: Vec<f32> = (0..8).map(|i| i as f32).collect(); // 4×2
        let t: Vec<f32> = (0..8).map(|i| (8 - i) as f32).collect();
        let loss = sp_loss(&s, 2, &t, 2, 4).expect("sp_loss should succeed");
        assert!(loss >= 0.0 && loss.is_finite(), "loss={loss}");
    }

    #[test]
    fn different_feature_dims_ok() {
        // Student dim 2, teacher dim 5, same batch 3.
        let s: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let t: Vec<f32> = (0..15).map(|i| (i as f32) * 0.2).collect();
        let loss = sp_loss(&s, 2, &t, 5, 3).expect("sp_loss should succeed");
        assert!(loss.is_finite() && loss >= 0.0);
    }

    #[test]
    fn scale_invariant() {
        // Row normalisation ⇒ scaling student activations leaves the loss unchanged.
        let s: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let t: Vec<f32> = (1..=8).map(|i| (i as f32) * 0.3 + 1.0).collect();
        let s_scaled: Vec<f32> = s.iter().map(|&v| v * 10.0).collect();
        let l1 = sp_loss(&s, 2, &t, 2, 4).expect("sp_loss should succeed");
        let l2 = sp_loss(&s_scaled, 2, &t, 2, 4).expect("sp_loss should succeed");
        assert!(
            (l1 - l2).abs() < 1e-5,
            "SP loss must be scale-invariant: {l1} vs {l2}"
        );
    }

    #[test]
    fn symmetric_gram_for_real_features() {
        // G = A·Aᵀ is symmetric before normalisation; after row-normalisation it
        // generally is not, but the diagonal stays positive.
        let a: Vec<f32> = (1..=9).map(|i| i as f32).collect(); // 3×3
        let g = similarity_matrix(&a, 3, 3);
        for i in 0..3 {
            assert!(
                g[i * 3 + i] > 0.0,
                "diagonal self-similarity must be positive"
            );
        }
    }

    #[test]
    fn empty_input_errors() {
        assert!(matches!(
            sp_loss(&[], 0, &[1.0], 1, 1),
            Err(DistillError::EmptyInput)
        ));
        let a = vec![1.0_f32; 4];
        assert!(matches!(
            sp_loss(&a, 2, &a, 2, 0),
            Err(DistillError::EmptyInput)
        ));
    }

    #[test]
    fn dim_mismatch_student_errors() {
        let s = vec![1.0_f32; 7]; // not 4*2
        let t = vec![1.0_f32; 8];
        assert!(matches!(
            sp_loss(&s, 2, &t, 2, 4),
            Err(DistillError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn dim_mismatch_teacher_errors() {
        let s = vec![1.0_f32; 8];
        let t = vec![1.0_f32; 9]; // not 4*2
        assert!(matches!(
            sp_loss(&s, 2, &t, 2, 4),
            Err(DistillError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn batch_size_one_works() {
        // A single sample ⇒ 1×1 Gram = [1] after normalisation for both ⇒ 0 loss.
        let s = vec![1.0_f32, 2.0, 3.0];
        let t = vec![4.0_f32, 5.0, 6.0];
        let loss = sp_loss(&s, 3, &t, 3, 1).expect("sp_loss should succeed");
        assert!(
            loss < 1e-6,
            "batch=1 normalised Gram is [1] for both ⇒ loss 0, got {loss}"
        );
    }

    #[test]
    fn more_different_similarity_larger_loss() {
        // Teacher batch where rows 0,1 are similar; build two students, one matching
        // that structure and one with the opposite structure.
        let teacher = vec![
            1.0_f32, 0.0, // sample 0
            1.0, 0.0, // sample 1 (== 0)
            0.0, 1.0, // sample 2 (orthogonal)
        ]; // 3×2
        let student_match = teacher.clone();
        let student_off = vec![
            1.0_f32, 0.0, // 0
            0.0, 1.0, // 1 (now orthogonal to 0 — wrong)
            0.0, 1.0, // 2
        ];
        let l_match = sp_loss(&student_match, 2, &teacher, 2, 3).expect("sp_loss should succeed");
        let l_off = sp_loss(&student_off, 2, &teacher, 2, 3).expect("sp_loss should succeed");
        assert!(
            l_match < l_off,
            "matching structure must cost less: {l_match} < {l_off}"
        );
        assert!(l_match < 1e-6);
    }

    #[test]
    fn known_two_sample_value() {
        // Two orthogonal unit samples in both nets ⇒ G = I ⇒ normalised G = I ⇒
        // loss 0. Swap teacher to identical samples ⇒ G all-ones ⇒ differs.
        let s = vec![1.0_f32, 0.0, 0.0, 1.0]; // 2×2, orthogonal
        let t_same = vec![1.0_f32, 0.0, 1.0, 0.0]; // 2×2, identical samples
        let loss = sp_loss(&s, 2, &t_same, 2, 2).expect("sp_loss should succeed");
        // G_s normalised = I; G_t = [[1,1],[1,1]] → row-normalised = [[.707,.707],...]
        // Frobenius² of difference / 4 must be positive and finite.
        assert!(
            loss > 0.0 && loss.is_finite(),
            "expected positive loss, got {loss}"
        );
    }
}
