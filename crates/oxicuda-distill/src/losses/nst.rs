//! NST — Neuron Selectivity Transfer (Huang & Wang 2017).
//!
//! Reference: Huang, Z., & Wang, N. (2017). *Like What You Like: Knowledge Distill
//! via Neuron Selectivity Transfer*. arXiv:1707.01219.
//! <https://arxiv.org/abs/1707.01219>
//!
//! NST aligns the *spatial activation patterns* of teacher and student feature maps
//! by matching the **distribution of neuron selectivities** rather than their raw
//! values. Each channel of a `C × (H·W)` feature map is one "neuron"; its
//! `H·W`-dimensional spatial response is treated as a sample drawn from a
//! per-network distribution. The loss is the squared Maximum Mean Discrepancy (MMD)
//! between the student's and teacher's selectivity samples:
//!
//! ```text
//!   L_MMD² =  1/Cₛ²  Σ_{i,j} k(fₛⁱ, fₛʲ)
//!           + 1/C_t² Σ_{i,j} k(f_tⁱ, f_tʲ)
//!           − 2/(Cₛ·C_t) Σ_{i,j} k(fₛⁱ, f_tʲ)
//! ```
//!
//! Each channel response is first **L2-normalised across the spatial dimension** so
//! that the kernel compares activation *shapes*, not magnitudes (the paper applies
//! this normalisation before the kernel). Two kernels are supported:
//!
//! - [`NstKernel::Linear`]: `k(x, y) = ⟨x, y⟩`. With unit-norm inputs the MMD has the
//!   closed form `‖mean(fₛ) − mean(f_t)‖²` (squared distance of the mean normalised
//!   channel), which is fast and convex.
//! - [`NstKernel::Polynomial`]: `k(x, y) = (⟨x, y⟩ · scale + bias)^degree`, the
//!   second-order kernel recommended in the paper for richer selectivity matching.
//!
//! Student and teacher may have different channel counts but **must share the same
//! spatial size** `H·W`.

use crate::error::{DistillError, DistillResult};

const EPS: f32 = 1e-8;

/// Kernel used inside the NST MMD computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NstKernel {
    /// Linear kernel `k(x, y) = ⟨x, y⟩`.
    Linear,
    /// Polynomial kernel `k(x, y) = (⟨x, y⟩ · scale + bias)^degree`.
    Polynomial {
        /// Polynomial degree (≥ 1).
        degree: u32,
        /// Inner-product scale.
        scale: f32,
        /// Additive bias.
        bias: f32,
    },
}

impl NstKernel {
    /// Evaluate the kernel on two equal-length channel responses.
    #[must_use]
    fn eval(self, x: &[f32], y: &[f32]) -> f32 {
        let dot: f32 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
        match self {
            NstKernel::Linear => dot,
            NstKernel::Polynomial {
                degree,
                scale,
                bias,
            } => (dot * scale + bias).powi(degree as i32),
        }
    }
}

/// L2-normalise each of the `n_channels` rows (length `spatial`) of a `C × (H·W)`
/// row-major feature map, returning a fresh matrix.
#[must_use]
pub fn normalize_channels(feat: &[f32], n_channels: usize, spatial: usize) -> Vec<f32> {
    let mut out = feat.to_vec();
    for c in 0..n_channels {
        let row = &mut out[c * spatial..(c + 1) * spatial];
        let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt().max(EPS);
        for v in row.iter_mut() {
            *v /= norm;
        }
    }
    out
}

/// Mean of the pairwise kernel `1/(Ca·Cb) Σ_{i,j} k(aⁱ, bʲ)` between two
/// channel-normalised feature maps `a` (`ca × spatial`) and `b` (`cb × spatial`).
fn mean_kernel(
    a: &[f32],
    ca: usize,
    b: &[f32],
    cb: usize,
    spatial: usize,
    kernel: NstKernel,
) -> f32 {
    let mut acc = 0.0_f32;
    for i in 0..ca {
        let ai = &a[i * spatial..(i + 1) * spatial];
        for j in 0..cb {
            let bj = &b[j * spatial..(j + 1) * spatial];
            acc += kernel.eval(ai, bj);
        }
    }
    acc / (ca as f32 * cb as f32)
}

/// NST loss: squared MMD between the student's and teacher's neuron-selectivity
/// distributions.
///
/// `student` is `student_channels × spatial`, `teacher` is `teacher_channels ×
/// spatial`; both are row-major. The two maps must agree on `spatial = H·W`.
///
/// # Errors
///
/// - [`DistillError::EmptyInput`] if either feature map is empty or `spatial == 0`.
/// - [`DistillError::DimensionMismatch`] if a slice length does not equal
///   `channels · spatial`.
/// - [`DistillError::InvalidConfig`] for a polynomial kernel with `degree == 0`.
/// - [`DistillError::NumericalError`] if the result is non-finite.
#[allow(clippy::too_many_arguments)]
pub fn nst_loss(
    student: &[f32],
    student_channels: usize,
    teacher: &[f32],
    teacher_channels: usize,
    spatial: usize,
    kernel: NstKernel,
) -> DistillResult<f32> {
    if student.is_empty() || teacher.is_empty() || spatial == 0 {
        return Err(DistillError::EmptyInput);
    }
    if student_channels == 0 || teacher_channels == 0 {
        return Err(DistillError::EmptyInput);
    }
    if student.len() != student_channels * spatial {
        return Err(DistillError::DimensionMismatch {
            expected: student_channels * spatial,
            got: student.len(),
        });
    }
    if teacher.len() != teacher_channels * spatial {
        return Err(DistillError::DimensionMismatch {
            expected: teacher_channels * spatial,
            got: teacher.len(),
        });
    }
    if let NstKernel::Polynomial { degree, .. } = kernel
        && degree == 0
    {
        return Err(DistillError::InvalidConfig {
            msg: "polynomial kernel degree must be >= 1".into(),
        });
    }
    let s = normalize_channels(student, student_channels, spatial);
    let t = normalize_channels(teacher, teacher_channels, spatial);

    let ss = mean_kernel(&s, student_channels, &s, student_channels, spatial, kernel);
    let tt = mean_kernel(&t, teacher_channels, &t, teacher_channels, spatial, kernel);
    let st = mean_kernel(&s, student_channels, &t, teacher_channels, spatial, kernel);

    let loss = ss + tt - 2.0 * st;
    if !loss.is_finite() {
        return Err(DistillError::NumericalError {
            msg: "NST MMD produced a non-finite value".into(),
        });
    }
    // Numerical noise can make a theoretically-zero MMD slightly negative; clamp.
    Ok(loss.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear() -> NstKernel {
        NstKernel::Linear
    }

    fn poly() -> NstKernel {
        NstKernel::Polynomial {
            degree: 2,
            scale: 1.0,
            bias: 0.0,
        }
    }

    #[test]
    fn normalize_gives_unit_rows() {
        let feat = vec![3.0_f32, 4.0, 0.0, 0.0, 6.0, 8.0]; // 2 channels × 3 spatial
        let n = normalize_channels(&feat, 2, 3);
        let n0: f32 = n[0..3].iter().map(|&v| v * v).sum::<f32>().sqrt();
        let n1: f32 = n[3..6].iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!((n0 - 1.0).abs() < 1e-5 && (n1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn normalize_zero_row_safe() {
        let feat = vec![0.0_f32, 0.0, 0.0];
        let n = normalize_channels(&feat, 1, 3);
        assert!(
            n.iter().all(|v| v.is_finite()),
            "zero row must not produce NaN"
        );
    }

    #[test]
    fn identical_maps_zero_loss_linear() {
        let feat: Vec<f32> = (0..12).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let loss = nst_loss(&feat, 3, &feat, 3, 4, linear()).expect("value should be present");
        assert!(
            loss < 1e-5,
            "identical maps must give ~0 NST loss, got {loss}"
        );
    }

    #[test]
    fn identical_maps_zero_loss_poly() {
        let feat: Vec<f32> = (0..12).map(|i| (i as f32) * 0.2 + 0.5).collect();
        let loss = nst_loss(&feat, 3, &feat, 3, 4, poly()).expect("value should be present");
        assert!(
            loss < 1e-5,
            "identical maps must give ~0 poly NST loss, got {loss}"
        );
    }

    #[test]
    fn loss_nonneg() {
        let s: Vec<f32> = (0..8).map(|i| (i as f32) * 0.5).collect();
        let t: Vec<f32> = (0..8).map(|i| (8 - i) as f32 * 0.3).collect();
        let loss = nst_loss(&s, 2, &t, 2, 4, linear()).expect("value should be present");
        assert!(
            loss >= 0.0 && loss.is_finite(),
            "MMD must be >= 0, got {loss}"
        );
    }

    #[test]
    fn different_channel_counts_ok() {
        // Student has 2 channels, teacher has 3, same spatial size 4.
        let s: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let t: Vec<f32> = (0..12).map(|i| (i as f32) * 0.5).collect();
        let loss = nst_loss(&s, 2, &t, 3, 4, linear()).expect("value should be present");
        assert!(loss.is_finite() && loss >= 0.0);
    }

    #[test]
    fn scale_invariant_after_normalization() {
        // Scaling a feature map by a positive constant must not change the loss,
        // because channels are L2-normalised first.
        let s: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let t: Vec<f32> = (1..=8).map(|i| (i as f32) * 0.7 + 1.0).collect();
        let s_scaled: Vec<f32> = s.iter().map(|&v| v * 5.0).collect();
        let l1 = nst_loss(&s, 2, &t, 2, 4, linear()).expect("value should be present");
        let l2 = nst_loss(&s_scaled, 2, &t, 2, 4, linear()).expect("value should be present");
        assert!(
            (l1 - l2).abs() < 1e-5,
            "loss must be scale-invariant: {l1} vs {l2}"
        );
    }

    #[test]
    fn polynomial_degree_zero_errors() {
        let feat = vec![1.0_f32; 4];
        let bad = NstKernel::Polynomial {
            degree: 0,
            scale: 1.0,
            bias: 0.0,
        };
        let r = nst_loss(&feat, 1, &feat, 1, 4, bad);
        assert!(matches!(r, Err(DistillError::InvalidConfig { .. })));
    }

    #[test]
    fn empty_input_errors() {
        assert!(matches!(
            nst_loss(&[], 0, &[1.0], 1, 1, linear()),
            Err(DistillError::EmptyInput)
        ));
        let feat = vec![1.0_f32; 4];
        assert!(matches!(
            nst_loss(&feat, 1, &feat, 1, 0, linear()),
            Err(DistillError::EmptyInput)
        ));
    }

    #[test]
    fn dim_mismatch_student_errors() {
        let s = vec![1.0_f32; 7]; // not 2*4
        let t = vec![1.0_f32; 8];
        let r = nst_loss(&s, 2, &t, 2, 4, linear());
        assert!(matches!(r, Err(DistillError::DimensionMismatch { .. })));
    }

    #[test]
    fn dim_mismatch_teacher_errors() {
        let s = vec![1.0_f32; 8];
        let t = vec![1.0_f32; 9]; // not 2*4
        let r = nst_loss(&s, 2, &t, 2, 4, linear());
        assert!(matches!(r, Err(DistillError::DimensionMismatch { .. })));
    }

    #[test]
    fn linear_mmd_equals_mean_distance() {
        // For the linear kernel with unit-norm channels, MMD² == ‖μ_s − μ_t‖².
        let s: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0]; // 2 ch × 2 spatial, already ~unit
        let t: Vec<f32> = vec![0.0, 1.0, 1.0, 0.0];
        let loss = nst_loss(&s, 2, &t, 2, 2, linear()).expect("value should be present");
        // μ_s = mean of normalised rows; compute reference directly.
        let sn = normalize_channels(&s, 2, 2);
        let tn = normalize_channels(&t, 2, 2);
        let mut mu_s = [0.0_f32; 2];
        let mut mu_t = [0.0_f32; 2];
        for c in 0..2 {
            for j in 0..2 {
                mu_s[j] += sn[c * 2 + j] / 2.0;
                mu_t[j] += tn[c * 2 + j] / 2.0;
            }
        }
        let ref_loss: f32 = (0..2).map(|j| (mu_s[j] - mu_t[j]).powi(2)).sum();
        assert!((loss - ref_loss).abs() < 1e-5, "loss={loss} ref={ref_loss}");
    }

    #[test]
    fn more_different_maps_larger_loss() {
        // The polynomial kernel captures higher-order selectivity moments, so a map
        // that more strongly disagrees with the teacher costs more. (The linear
        // kernel only compares channel means and is intentionally insensitive here.)
        let base: Vec<f32> = vec![1.0, 0.2, 0.2, 1.0, 0.5, 0.5];
        let near: Vec<f32> = vec![0.95, 0.25, 0.25, 0.95, 0.5, 0.5];
        let far: Vec<f32> = vec![0.2, 1.0, 1.0, 0.2, 0.9, 0.1];
        let l_near = nst_loss(&base, 3, &near, 3, 2, poly()).expect("value should be present");
        let l_far = nst_loss(&base, 3, &far, 3, 2, poly()).expect("value should be present");
        assert!(
            l_far > l_near,
            "more different patterns must cost more: {l_far} vs {l_near}"
        );
    }
}
