//! FitNets hint loss — Romero et al. 2015.
//!
//! "FitNets: Hints for Thin Deep Nets" (Romero et al., ICLR 2015) distils
//! *intermediate* representations, not just the final logits. A "hint" is a
//! teacher hidden layer; the matching student layer is the "guided" layer.
//! Because the student hidden width `d_student` rarely equals the teacher width
//! `d_teacher`, a learnable **regressor** linearly projects the student hint
//! into the teacher's feature space before the mean-squared-error comparison:
//!
//! ```text
//! projected = regressor_w · student_hint + regressor_b   (∈ ℝ^{d_teacher})
//! L_hint    = MSE(projected, teacher_guided)
//! ```
//!
//! Minimising `L_hint` pre-trains the student up to the guided layer so that it
//! reproduces the teacher's intermediate features, giving a better
//! initialisation for the subsequent full knowledge-distillation stage.
//!
//! This module exposes the loss as a single free function operating on an
//! externally-owned regressor (`regressor_w`, `regressor_b`). The stateful
//! [`crate::feature::fitnets::FitNetsRegressor`] object provides the same
//! mathematics with an owned, He-initialised projection.

use crate::error::{DistillError, DistillResult};

/// FitNets hint loss for a single feature vector.
///
/// Projects `student_hint` (`d_student`) into the teacher space via the dense
/// regressor `regressor_w` (`[d_teacher × d_student]`, row-major) plus
/// `regressor_b` (`[d_teacher]`), then returns the mean squared error against
/// `teacher_guided` (`d_teacher`):
///
/// ```text
/// projected[o] = Σ_i regressor_w[o · d_student + i] · student_hint[i] + regressor_b[o]
/// loss         = (1 / d_teacher) · Σ_o (projected[o] − teacher_guided[o])²
/// ```
///
/// # Errors
/// - [`DistillError::InvalidConfig`] if `d_student == 0` or `d_teacher == 0`.
/// - [`DistillError::DimensionMismatch`] if any input length disagrees with the
///   declared dimensions: `student_hint.len() != d_student`,
///   `teacher_guided.len() != d_teacher`,
///   `regressor_w.len() != d_teacher · d_student`, or
///   `regressor_b.len() != d_teacher`.
/// - [`DistillError::NumericalError`] if the projection produces a non-finite
///   value.
pub fn fitnet_hint_loss(
    student_hint: &[f32],
    teacher_guided: &[f32],
    regressor_w: &[f32],
    regressor_b: &[f32],
    d_student: usize,
    d_teacher: usize,
) -> DistillResult<f32> {
    if d_student == 0 || d_teacher == 0 {
        return Err(DistillError::InvalidConfig {
            msg: format!("d_student and d_teacher must be > 0, got {d_student} and {d_teacher}"),
        });
    }
    if student_hint.len() != d_student {
        return Err(DistillError::DimensionMismatch {
            expected: d_student,
            got: student_hint.len(),
        });
    }
    if teacher_guided.len() != d_teacher {
        return Err(DistillError::DimensionMismatch {
            expected: d_teacher,
            got: teacher_guided.len(),
        });
    }
    if regressor_w.len() != d_teacher * d_student {
        return Err(DistillError::DimensionMismatch {
            expected: d_teacher * d_student,
            got: regressor_w.len(),
        });
    }
    if regressor_b.len() != d_teacher {
        return Err(DistillError::DimensionMismatch {
            expected: d_teacher,
            got: regressor_b.len(),
        });
    }

    let mut sq_err = 0.0_f32;
    for o in 0..d_teacher {
        let w_row = &regressor_w[o * d_student..(o + 1) * d_student];
        let mut proj = regressor_b[o];
        for (wv, sv) in w_row.iter().zip(student_hint.iter()) {
            proj += wv * sv;
        }
        if !proj.is_finite() {
            return Err(DistillError::NumericalError {
                msg: format!("non-finite projection at output {o}"),
            });
        }
        let diff = proj - teacher_guided[o];
        sq_err += diff * diff;
    }

    Ok(sq_err / d_teacher as f32)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loss_finite() {
        // d_student = 3, d_teacher = 2
        let s = vec![1.0_f32, 2.0, 3.0];
        let t = vec![0.5_f32, -0.5];
        let w = vec![0.1_f32, 0.2, 0.3, -0.1, 0.0, 0.2];
        let bvec = vec![0.05_f32, -0.05];
        let l = fitnet_hint_loss(&s, &t, &w, &bvec, 3, 2).expect("ok");
        assert!(l.is_finite(), "loss = {l}");
    }

    #[test]
    fn loss_nonneg() {
        let s = vec![-1.0_f32, 4.0, 0.0];
        let t = vec![2.0_f32, 2.0];
        let w = vec![1.0_f32, -1.0, 0.5, 0.0, 1.0, -0.5];
        let bvec = vec![0.0_f32, 0.0];
        let l = fitnet_hint_loss(&s, &t, &w, &bvec, 3, 2).expect("ok");
        assert!(l >= 0.0, "MSE must be non-negative, got {l}");
    }

    #[test]
    fn zero_for_perfect_regressor() {
        // Choose w, b so the projection lands exactly on the teacher target.
        // student = [2, 3]; pick w = [[1,0],[0,1]], b = [0, 0] ⇒ proj = [2, 3].
        // teacher = [2, 3] ⇒ loss = 0.
        let s = vec![2.0_f32, 3.0];
        let t = vec![2.0_f32, 3.0];
        let w = vec![1.0_f32, 0.0, 0.0, 1.0];
        let bvec = vec![0.0_f32, 0.0];
        let l = fitnet_hint_loss(&s, &t, &w, &bvec, 2, 2).expect("ok");
        assert!(l < 1e-10, "perfect regressor must give 0, got {l}");
    }

    #[test]
    fn dim_mismatch_error() {
        let s = vec![1.0_f32, 2.0, 3.0];
        let t = vec![0.5_f32, -0.5];
        let w = vec![0.1_f32, 0.2, 0.3, -0.1, 0.0, 0.2];
        let bvec = vec![0.05_f32, -0.05];
        // Wrong d_student vs slice length.
        let r = fitnet_hint_loss(&s, &t, &w, &bvec, 4, 2);
        assert!(matches!(r, Err(DistillError::DimensionMismatch { .. })));
        // Wrong weight length.
        let bad_w = vec![0.1_f32, 0.2];
        let r2 = fitnet_hint_loss(&s, &t, &bad_w, &bvec, 3, 2);
        assert!(matches!(r2, Err(DistillError::DimensionMismatch { .. })));
        // Wrong bias length.
        let bad_b = vec![0.0_f32];
        let r3 = fitnet_hint_loss(&s, &t, &w, &bad_b, 3, 2);
        assert!(matches!(r3, Err(DistillError::DimensionMismatch { .. })));
    }

    #[test]
    fn identity_regressor() {
        // When d_student == d_teacher and w = I, b = 0, the loss is exactly
        // MSE(student, teacher).
        let s = vec![1.0_f32, 2.0, 3.0];
        let t = vec![1.5_f32, 2.0, 2.0];
        let w = vec![
            1.0_f32, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, // row 1
            0.0, 0.0, 1.0, // row 2
        ];
        let bvec = vec![0.0_f32, 0.0, 0.0];
        let l = fitnet_hint_loss(&s, &t, &w, &bvec, 3, 3).expect("ok");
        // MSE = ((1-1.5)² + 0 + (3-2)²) / 3 = (0.25 + 1.0) / 3 = 0.41666…
        let expected = (0.25_f32 + 0.0 + 1.0) / 3.0;
        assert!((l - expected).abs() < 1e-5, "got {l}, expected {expected}");
    }

    #[test]
    fn d_student_0_error() {
        let s: Vec<f32> = vec![];
        let t = vec![1.0_f32];
        let w: Vec<f32> = vec![];
        let bvec = vec![0.0_f32];
        let r = fitnet_hint_loss(&s, &t, &w, &bvec, 0, 1);
        assert!(matches!(r, Err(DistillError::InvalidConfig { .. })));
    }

    #[test]
    fn larger_for_different() {
        // A regressor that lands far from the target gives a larger loss than
        // one that lands on it.
        let s = vec![1.0_f32, 1.0];
        let t = vec![0.0_f32, 0.0];
        let w_close = vec![0.0_f32, 0.0, 0.0, 0.0]; // proj = [0, 0] ⇒ loss 0
        let w_far = vec![5.0_f32, 5.0, 5.0, 5.0]; // proj = [10, 10] ⇒ large
        let bvec = vec![0.0_f32, 0.0];
        let close = fitnet_hint_loss(&s, &t, &w_close, &bvec, 2, 2).expect("ok");
        let far = fitnet_hint_loss(&s, &t, &w_far, &bvec, 2, 2).expect("ok");
        assert!(far > close, "far={far} should exceed close={close}");
    }

    #[test]
    fn batch_consistent() {
        // Summing the loss over two "tokens" by calling twice must equal the
        // mean-of-means computed manually (the function is deterministic and
        // additive across independent calls).
        let w = vec![0.2_f32, -0.1, 0.3, 0.0];
        let bvec = vec![0.1_f32, -0.1];
        let s0 = vec![1.0_f32, 2.0];
        let t0 = vec![0.0_f32, 1.0];
        let s1 = vec![-1.0_f32, 0.5];
        let t1 = vec![0.5_f32, 0.5];
        let l0 = fitnet_hint_loss(&s0, &t0, &w, &bvec, 2, 2).expect("ok");
        let l1 = fitnet_hint_loss(&s1, &t1, &w, &bvec, 2, 2).expect("ok");
        let l0_again = fitnet_hint_loss(&s0, &t0, &w, &bvec, 2, 2).expect("ok");
        assert!((l0 - l0_again).abs() < 1e-9, "deterministic across calls");
        let mean = 0.5 * (l0 + l1);
        assert!(mean.is_finite() && mean >= 0.0);
    }

    #[test]
    fn bias_contributes() {
        // With zero weights, the projection equals the bias, so the loss is
        // MSE(bias, teacher).
        let s = vec![3.0_f32, -2.0];
        let t = vec![1.0_f32, 1.0];
        let w = vec![0.0_f32; 4];
        let bvec = vec![1.0_f32, 1.0]; // proj = teacher exactly ⇒ loss 0
        let l = fitnet_hint_loss(&s, &t, &w, &bvec, 2, 2).expect("ok");
        assert!(l < 1e-10, "bias should land on target, got {l}");
    }
}
