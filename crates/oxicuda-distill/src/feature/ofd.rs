//! OFD — Overhaul of Feature Distillation (Heo et al. 2019).
//!
//! Reference: Heo, B., Kim, J., Yun, S., Park, H., Kwak, N., & Choi, J. Y. (2019).
//! *A Comprehensive Overhaul of Feature Distillation*. ICCV 2019.
//! <https://arxiv.org/abs/1904.01866>
//!
//! Heo et al. revisit every design choice of feature distillation and propose:
//!
//! 1. **Distillation position** — distill *before* the ReLU of a block (the "pre-activation"
//!    feature) so that information lost by the activation can still be transferred.
//!
//! 2. **Margin ReLU teacher transform** — instead of passing the teacher feature through a
//!    plain ReLU, use a *margin ReLU* with a per-channel negative margin `m_c < 0`:
//!
//!    ```text
//!      σ_m(x)_c = max(x_c, m_c)        with  m_c = E[ x_c | x_c < 0 ]   (expected negative)
//!    ```
//!
//!    Negative teacher responses below the margin carry little useful signal, so they are
//!    clamped to the margin rather than to zero, preserving "beneficial" negative information
//!    while discarding noise.
//!
//! 3. **Partial-L2 distance** — an asymmetric distance that penalises the student only when it
//!    disagrees with the *sign-aware* target. Concretely, for a teacher target `t` (after
//!    margin ReLU) and a student response `s`, the per-element cost is
//!
//!    ```text
//!      d(s, t) = 0                if  t ≤ 0  and  s ≤ t        (both "off"; no penalty)
//!              = (s − t)²         otherwise
//!    ```
//!
//!    This stops the student being pushed to reproduce large negative teacher values that the
//!    margin ReLU has already deemed unimportant.
//!
//! A 1×1 connector ([`OfdConnector`]) maps the student channel dimension to the teacher's.
//!
//! This module operates on flat per-channel feature vectors of length `channels` (one spatial
//! position, or a global-pooled descriptor); batched helpers iterate over rows.

use crate::error::{DistillError, DistillResult};
use crate::handle::LcgRng;

/// Margin ReLU: `max(x, margin)` applied element-wise with a per-channel margin.
///
/// `x` and `margin` must have equal length (`channels`).
///
/// # Errors
/// [`DistillError::DimensionMismatch`] if the lengths differ.
pub fn margin_relu(x: &[f32], margin: &[f32]) -> DistillResult<Vec<f32>> {
    if x.len() != margin.len() {
        return Err(DistillError::DimensionMismatch {
            expected: x.len(),
            got: margin.len(),
        });
    }
    Ok(x.iter()
        .zip(margin.iter())
        .map(|(&xi, &mi)| xi.max(mi))
        .collect())
}

/// Estimate the per-channel negative margin `m_c = E[x_c | x_c < 0]` from a batch.
///
/// `feats` is a batch of `channels`-length rows. For each channel the mean of all strictly
/// negative entries is taken; channels with no negative entries get margin `0.0`.
///
/// # Errors
/// - [`DistillError::EmptyInput`] if `feats` is empty.
/// - [`DistillError::DimensionMismatch`] if any row length differs from the first.
pub fn estimate_margins(feats: &[Vec<f32>]) -> DistillResult<Vec<f32>> {
    let first = feats.first().ok_or(DistillError::EmptyInput)?;
    let channels = first.len();
    let mut sums = vec![0.0_f32; channels];
    let mut counts = vec![0_usize; channels];
    for row in feats {
        if row.len() != channels {
            return Err(DistillError::DimensionMismatch {
                expected: channels,
                got: row.len(),
            });
        }
        for (c, &v) in row.iter().enumerate() {
            if v < 0.0 {
                sums[c] += v;
                counts[c] += 1;
            }
        }
    }
    Ok((0..channels)
        .map(|c| {
            if counts[c] == 0 {
                0.0
            } else {
                sums[c] / counts[c] as f32
            }
        })
        .collect())
}

/// Partial-L2 distance between a student response `s` and a teacher target `t`.
///
/// Per element: `0` when `t ≤ 0 && s ≤ t`, else `(s − t)²`. The result is the **sum** over
/// elements (not the mean), matching the original OFD formulation.
///
/// # Errors
/// [`DistillError::DimensionMismatch`] if the lengths differ.
pub fn partial_l2(s: &[f32], t: &[f32]) -> DistillResult<f32> {
    if s.len() != t.len() {
        return Err(DistillError::DimensionMismatch {
            expected: s.len(),
            got: t.len(),
        });
    }
    let mut total = 0.0_f32;
    for (&si, &ti) in s.iter().zip(t.iter()) {
        if ti <= 0.0 && si <= ti {
            // Both "off" and the student is already at-or-below the (negative) target:
            // no penalty.
            continue;
        }
        let d = si - ti;
        total += d * d;
    }
    Ok(total)
}

/// A 1×1 connector mapping `in_channels` student channels to `out_channels` teacher channels.
///
/// Applied to a single feature descriptor of length `in_channels`.
#[derive(Debug, Clone)]
pub struct OfdConnector {
    /// Number of input (student) channels.
    pub in_channels: usize,
    /// Number of output (teacher) channels.
    pub out_channels: usize,
    /// Weight matrix `[out_channels × in_channels]` (row-major).
    pub w: Vec<f32>,
}

impl OfdConnector {
    /// Construct a connector with weights `~ N(0, 1/√in_channels)`.
    #[must_use]
    pub fn new(in_channels: usize, out_channels: usize, rng: &mut LcgRng) -> Self {
        let scale = if in_channels == 0 {
            1.0
        } else {
            1.0 / (in_channels as f32).sqrt()
        };
        let mut w = vec![0.0_f32; out_channels * in_channels];
        for wi in w.iter_mut() {
            *wi = rng.next_normal() * scale;
        }
        Self {
            in_channels,
            out_channels,
            w,
        }
    }

    /// Apply the connector to a single descriptor of length `in_channels`.
    ///
    /// # Errors
    /// - [`DistillError::InvalidConfig`] if `in_channels == 0`.
    /// - [`DistillError::DimensionMismatch`] if `x.len() != in_channels`.
    pub fn forward(&self, x: &[f32]) -> DistillResult<Vec<f32>> {
        if self.in_channels == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "OfdConnector in_channels is zero".into(),
            });
        }
        if x.len() != self.in_channels {
            return Err(DistillError::DimensionMismatch {
                expected: self.in_channels,
                got: x.len(),
            });
        }
        Ok((0..self.out_channels)
            .map(|o| {
                let row = &self.w[o * self.in_channels..(o + 1) * self.in_channels];
                row.iter().zip(x.iter()).map(|(&w, &xi)| w * xi).sum()
            })
            .collect())
    }
}

/// Full OFD loss for a single descriptor.
///
/// 1. Connect the student feature: `s' = connector(student)`.
/// 2. Transform the teacher feature with the margin ReLU: `t' = σ_m(teacher)`.
/// 3. Return the partial-L2 distance `d(s', t')`.
///
/// `margin` is the per-channel teacher margin (length `connector.out_channels`).
///
/// # Errors
/// Propagates errors from [`OfdConnector::forward`], [`margin_relu`], and [`partial_l2`].
pub fn ofd_loss(
    student: &[f32],
    teacher: &[f32],
    connector: &OfdConnector,
    margin: &[f32],
) -> DistillResult<f32> {
    let s_proj = connector.forward(student)?;
    if teacher.len() != connector.out_channels {
        return Err(DistillError::DimensionMismatch {
            expected: connector.out_channels,
            got: teacher.len(),
        });
    }
    let t_marg = margin_relu(teacher, margin)?;
    partial_l2(&s_proj, &t_marg)
}

/// Mean OFD loss over a batch of descriptors.
///
/// `students` and `teachers` are equal-length batches; the loss is averaged over rows.
///
/// # Errors
/// - [`DistillError::EmptyInput`] if `students` is empty.
/// - [`DistillError::DimensionMismatch`] if batch sizes differ.
/// - Propagates per-row errors from [`ofd_loss`].
pub fn ofd_loss_batch(
    students: &[Vec<f32>],
    teachers: &[Vec<f32>],
    connector: &OfdConnector,
    margin: &[f32],
) -> DistillResult<f32> {
    if students.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if students.len() != teachers.len() {
        return Err(DistillError::DimensionMismatch {
            expected: students.len(),
            got: teachers.len(),
        });
    }
    let mut total = 0.0_f32;
    for (s, t) in students.iter().zip(teachers.iter()) {
        total += ofd_loss(s, t, connector, margin)?;
    }
    Ok(total / students.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn margin_relu_clamps_to_margin() {
        let x = vec![-3.0_f32, -0.5, 2.0];
        let m = vec![-1.0_f32, -1.0, -1.0];
        let out = margin_relu(&x, &m).expect("margin_relu should succeed");
        assert_eq!(out, vec![-1.0, -0.5, 2.0]);
    }

    #[test]
    fn margin_relu_dimension_mismatch_errors() {
        let x = vec![1.0_f32, 2.0];
        let m = vec![-1.0_f32];
        assert!(matches!(
            margin_relu(&x, &m),
            Err(DistillError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn estimate_margins_negative_mean() {
        let feats = vec![
            vec![-2.0_f32, 1.0],
            vec![-4.0_f32, -2.0],
            vec![1.0_f32, 3.0],
        ];
        let m = estimate_margins(&feats).expect("estimate_margins should succeed");
        // channel 0 negatives: -2, -4 → mean -3 ; channel 1 negatives: -2 → -2
        assert!((m[0] - (-3.0)).abs() < 1e-6, "m0={}", m[0]);
        assert!((m[1] - (-2.0)).abs() < 1e-6, "m1={}", m[1]);
    }

    #[test]
    fn estimate_margins_no_negatives_is_zero() {
        let feats = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 4.0]];
        let m = estimate_margins(&feats).expect("estimate_margins should succeed");
        assert_eq!(m, vec![0.0, 0.0]);
    }

    #[test]
    fn estimate_margins_empty_errors() {
        let feats: Vec<Vec<f32>> = vec![];
        assert!(matches!(
            estimate_margins(&feats),
            Err(DistillError::EmptyInput)
        ));
    }

    #[test]
    fn estimate_margins_all_nonneg_always_le_zero() {
        let feats = vec![vec![-1.0_f32, 5.0, -3.0], vec![2.0_f32, -1.0, 4.0]];
        let m = estimate_margins(&feats).expect("estimate_margins should succeed");
        for &mc in &m {
            assert!(mc <= 0.0, "margin must be ≤ 0, got {mc}");
        }
    }

    #[test]
    fn partial_l2_no_penalty_when_both_off() {
        // t ≤ 0 and s ≤ t → no penalty
        let s = vec![-5.0_f32, -3.0];
        let t = vec![-2.0_f32, -1.0];
        let d = partial_l2(&s, &t).expect("partial_l2 should succeed");
        assert!(d < 1e-6, "no penalty expected, got {d}");
    }

    #[test]
    fn partial_l2_penalizes_positive_mismatch() {
        let s = vec![1.0_f32];
        let t = vec![3.0_f32];
        let d = partial_l2(&s, &t).expect("partial_l2 should succeed");
        assert!((d - 4.0).abs() < 1e-6, "expected (1-3)^2=4, got {d}");
    }

    #[test]
    fn partial_l2_penalizes_student_above_negative_target() {
        // t < 0 but s > t → penalty (s − t)²
        let s = vec![0.0_f32];
        let t = vec![-2.0_f32];
        let d = partial_l2(&s, &t).expect("partial_l2 should succeed");
        assert!((d - 4.0).abs() < 1e-6, "expected 4, got {d}");
    }

    #[test]
    fn partial_l2_identical_is_zero_for_positive() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let d = partial_l2(&v, &v).expect("partial_l2 should succeed");
        assert!(d < 1e-6, "identical positive → 0, got {d}");
    }

    #[test]
    fn partial_l2_dimension_mismatch_errors() {
        let s = vec![1.0_f32, 2.0];
        let t = vec![1.0_f32];
        assert!(matches!(
            partial_l2(&s, &t),
            Err(DistillError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn connector_forward_shape() {
        let mut rng = LcgRng::new(1);
        let conn = OfdConnector::new(4, 6, &mut rng);
        let x: Vec<f32> = (0..4).map(|i| i as f32).collect();
        let out = conn.forward(&x).expect("forward should succeed");
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn connector_dimension_mismatch_errors() {
        let mut rng = LcgRng::new(2);
        let conn = OfdConnector::new(4, 6, &mut rng);
        assert!(matches!(
            conn.forward(&[1.0, 2.0]),
            Err(DistillError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn ofd_loss_finite_and_nonneg() {
        let mut rng = LcgRng::new(7);
        let conn = OfdConnector::new(3, 4, &mut rng);
        let margin = vec![-1.0_f32; 4];
        let student: Vec<f32> = vec![0.5, -0.5, 1.0];
        let teacher: Vec<f32> = vec![1.0, -2.0, 0.0, 0.5];
        let loss = ofd_loss(&student, &teacher, &conn, &margin).expect("ofd_loss should succeed");
        assert!(loss.is_finite() && loss >= 0.0, "loss={loss}");
    }

    #[test]
    fn ofd_loss_teacher_dim_mismatch_errors() {
        let mut rng = LcgRng::new(8);
        let conn = OfdConnector::new(3, 4, &mut rng);
        let margin = vec![-1.0_f32; 4];
        let student = vec![0.0_f32; 3];
        let teacher = vec![0.0_f32; 5]; // != out_channels 4
        assert!(matches!(
            ofd_loss(&student, &teacher, &conn, &margin),
            Err(DistillError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn ofd_loss_batch_mean() {
        let mut rng = LcgRng::new(9);
        let conn = OfdConnector::new(2, 2, &mut rng);
        let margin = vec![-1.0_f32; 2];
        let students = vec![vec![0.5_f32, -0.5], vec![1.0_f32, 0.0]];
        let teachers = vec![vec![1.0_f32, -2.0], vec![0.5_f32, 0.5]];
        let l0 =
            ofd_loss(&students[0], &teachers[0], &conn, &margin).expect("ofd_loss should succeed");
        let l1 =
            ofd_loss(&students[1], &teachers[1], &conn, &margin).expect("ofd_loss should succeed");
        let mean = ofd_loss_batch(&students, &teachers, &conn, &margin)
            .expect("ofd_loss_batch should succeed");
        assert!(
            (mean - (l0 + l1) / 2.0).abs() < 1e-5,
            "mean mismatch: {mean}"
        );
    }

    #[test]
    fn ofd_loss_batch_empty_errors() {
        let mut rng = LcgRng::new(10);
        let conn = OfdConnector::new(2, 2, &mut rng);
        let margin = vec![-1.0_f32; 2];
        let empty: Vec<Vec<f32>> = vec![];
        assert!(matches!(
            ofd_loss_batch(&empty, &empty, &conn, &margin),
            Err(DistillError::EmptyInput)
        ));
    }
}
