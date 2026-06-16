//! CC — Correlation Congruence for Knowledge Distillation (Peng et al. 2019).
//!
//! Reference: Peng, B., Jin, X., Liu, J., Li, D., Wu, Y., Liu, Y., Zhou, S., &
//! Zhang, Z. (2019). *Correlation Congruence for Knowledge Distillation*. ICCV
//! 2019. <https://arxiv.org/abs/1904.01802>
//!
//! # Idea
//!
//! Instance-level distillation (e.g. plain logit/KD) transfers each sample
//! independently. Correlation Congruence additionally transfers the **correlation
//! between instances** in a mini-batch: it builds an `n × n` *correlation matrix*
//! `Ψ` whose entry `Ψ[i, j]` measures how related samples `i` and `j` are in the
//! embedding space, and minimises the discrepancy between the student's and
//! teacher's correlation matrices:
//!
//! ```text
//!   Ψ[i, j] = κ(f_i, f_j)                     (instance correlation kernel)
//!   L_CC = (1 / n²) · Σ_{i,j} ( Ψ_s[i,j] − Ψ_t[i,j] )²
//! ```
//!
//! The paper's key contribution is the **Taylor-series-expanded Gaussian RBF
//! kernel**. Writing the standard RBF over L2-normalised embeddings as
//! `exp(−γ‖f_i − f_j‖²) = exp(−2γ(1 − f_iᵀf_j))`, a finite Taylor expansion of the
//! exponential in the inner product `z = f_iᵀf_j` gives a cheap polynomial
//! kernel that captures higher-order correlations:
//!
//! ```text
//!   κ_Taylor(z) = Σ_{p=0..P}  (2γ)^p / p! · z^p        (× exp(−2γ) prefactor)
//! ```
//!
//! This module implements the bilinear (linear), Gaussian-RBF, and the paper's
//! Taylor-expanded RBF correlation kernels, plus the congruence loss. Embeddings
//! are L2-normalised before the kernel so the correlation reflects *direction*
//! (relative similarity) rather than magnitude.

use crate::error::{DistillError, DistillResult};

const EPS: f32 = 1e-8;

/// Instance-correlation kernel used to build the `n × n` correlation matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CcKernel {
    /// Bilinear / linear correlation: `κ(f_i, f_j) = f_iᵀ f_j` on unit vectors.
    Bilinear,
    /// Gaussian RBF correlation with bandwidth `gamma`:
    /// `κ = exp(−γ‖f_i − f_j‖²)`.
    Gaussian {
        /// RBF bandwidth `γ > 0`.
        gamma: f32,
    },
    /// Taylor-expanded Gaussian RBF (the paper's kernel) to polynomial order
    /// `order` with bandwidth `gamma`.
    TaylorRbf {
        /// RBF bandwidth `γ > 0`.
        gamma: f32,
        /// Truncation order `P ≥ 1` of the Taylor expansion.
        order: usize,
    },
}

/// L2-normalise each row of an `n × dim` row-major embedding matrix in place into
/// a freshly-allocated buffer.
#[must_use]
pub fn l2_normalize_rows(feats: &[f32], n: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n * dim];
    for i in 0..n {
        let row = &feats[i * dim..(i + 1) * dim];
        let norm = row.iter().map(|&v| v * v).sum::<f32>().sqrt().max(EPS);
        let out_row = &mut out[i * dim..(i + 1) * dim];
        for (o, &v) in out_row.iter_mut().zip(row.iter()) {
            *o = v / norm;
        }
    }
    out
}

/// Evaluate the correlation kernel between two **unit-norm** embedding rows.
#[must_use]
fn kernel_value(a: &[f32], b: &[f32], kernel: CcKernel) -> f32 {
    // Inner product of unit vectors lies in [−1, 1].
    let z: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    match kernel {
        CcKernel::Bilinear => z,
        CcKernel::Gaussian { gamma } => {
            // ‖a − b‖² = 2(1 − z) for unit vectors.
            let dist_sq = 2.0 * (1.0 - z);
            (-gamma * dist_sq).exp()
        }
        CcKernel::TaylorRbf { gamma, order } => {
            // exp(−γ‖a−b‖²) = exp(−2γ) · exp(2γ z); Taylor-expand exp(2γ z).
            let prefactor = (-2.0 * gamma).exp();
            let a2 = 2.0 * gamma;
            let mut term = 1.0_f32; // (a2·z)^0 / 0!
            let mut sum = 1.0_f32;
            for p in 1..=order {
                term *= a2 * z / p as f32;
                sum += term;
            }
            prefactor * sum
        }
    }
}

/// Build the `n × n` instance-correlation matrix `Ψ` (flat row-major) for a batch
/// of `n` embeddings of dimension `dim`.
///
/// Rows are L2-normalised first, then the chosen [`CcKernel`] is applied to every
/// pair. The matrix is symmetric (`Ψ[i,j] == Ψ[j,i]`).
///
/// # Errors
///
/// - [`DistillError::EmptyInput`] if `n == 0` or `dim == 0`.
/// - [`DistillError::DimensionMismatch`] if `feats.len() != n * dim`.
/// - [`DistillError::InvalidConfig`] if a kernel parameter is invalid (non-positive
///   `gamma`, or `order == 0`).
pub fn correlation_matrix(
    feats: &[f32],
    n: usize,
    dim: usize,
    kernel: CcKernel,
) -> DistillResult<Vec<f32>> {
    if n == 0 || dim == 0 {
        return Err(DistillError::EmptyInput);
    }
    if feats.len() != n * dim {
        return Err(DistillError::DimensionMismatch {
            expected: n * dim,
            got: feats.len(),
        });
    }
    match kernel {
        CcKernel::Gaussian { gamma } if gamma <= 0.0 => {
            return Err(DistillError::InvalidConfig {
                msg: "Gaussian kernel gamma must be > 0".into(),
            });
        }
        CcKernel::TaylorRbf { gamma, order } if gamma <= 0.0 || order == 0 => {
            return Err(DistillError::InvalidConfig {
                msg: "TaylorRbf requires gamma > 0 and order >= 1".into(),
            });
        }
        _ => {}
    }
    let unit = l2_normalize_rows(feats, n, dim);
    let mut psi = vec![0.0_f32; n * n];
    for i in 0..n {
        let ri = &unit[i * dim..(i + 1) * dim];
        // Diagonal then upper triangle, mirrored — guarantees exact symmetry.
        psi[i * n + i] = kernel_value(ri, ri, kernel);
        for j in (i + 1)..n {
            let rj = &unit[j * dim..(j + 1) * dim];
            let v = kernel_value(ri, rj, kernel);
            psi[i * n + j] = v;
            psi[j * n + i] = v;
        }
    }
    Ok(psi)
}

/// Correlation Congruence loss between a student and teacher embedding batch.
///
/// Both are `n × *` row-major; the **batch size `n` must match**, but the student
/// and teacher feature dimensions may differ (the kernel operates on unit-norm
/// embeddings, comparing only the `n × n` correlation structure). The loss is the
/// mean squared difference of the two correlation matrices.
///
/// `L = (1 / n²) · ‖ Ψ_s − Ψ_t ‖²_F`.
///
/// # Errors
///
/// - [`DistillError::EmptyInput`] if any input is empty or a batch size is zero.
/// - [`DistillError::DimensionMismatch`] if a slice length disagrees with
///   `n · dim`.
/// - [`DistillError::NumericalError`] if the result is non-finite.
pub fn cc_loss(
    student: &[f32],
    student_dim: usize,
    teacher: &[f32],
    teacher_dim: usize,
    n: usize,
    kernel: CcKernel,
) -> DistillResult<f32> {
    if student.is_empty() || teacher.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    let psi_s = correlation_matrix(student, n, student_dim, kernel)?;
    let psi_t = correlation_matrix(teacher, n, teacher_dim, kernel)?;
    let fro_sq: f32 = psi_s
        .iter()
        .zip(psi_t.iter())
        .map(|(&s, &t)| (s - t) * (s - t))
        .sum();
    let loss = fro_sq / (n as f32 * n as f32);
    if !loss.is_finite() {
        return Err(DistillError::NumericalError {
            msg: "CC loss produced a non-finite value".into(),
        });
    }
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear() -> CcKernel {
        CcKernel::Bilinear
    }
    fn gauss() -> CcKernel {
        CcKernel::Gaussian { gamma: 0.5 }
    }
    fn taylor() -> CcKernel {
        CcKernel::TaylorRbf {
            gamma: 0.5,
            order: 4,
        }
    }

    // L2 normalisation produces unit rows.
    #[test]
    fn rows_normalised_to_unit() {
        let feats: Vec<f32> = vec![3.0, 4.0, 0.0, 5.0]; // 2×2
        let u = l2_normalize_rows(&feats, 2, 2);
        for i in 0..2 {
            let norm: f32 = u[i * 2..(i + 1) * 2]
                .iter()
                .map(|&v| v * v)
                .sum::<f32>()
                .sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "row {i} norm {norm}");
        }
    }

    // Correlation matrix is symmetric for every kernel.
    #[test]
    fn correlation_matrix_symmetric() {
        let feats: Vec<f32> = (0..12).map(|i| (i as f32) * 0.3 - 1.0).collect(); // 4×3
        for kernel in [linear(), gauss(), taylor()] {
            let psi = correlation_matrix(&feats, 4, 3, kernel).expect("ok");
            for i in 0..4 {
                for j in 0..4 {
                    let a = psi[i * 4 + j];
                    let b = psi[j * 4 + i];
                    assert!(
                        (a - b).abs() < 1e-6,
                        "kernel {kernel:?} asymmetric at ({i},{j}): {a} vs {b}"
                    );
                }
            }
        }
    }

    // Self-correlation on the diagonal equals 1 for the exact kernels (a sample is
    // maximally correlated with itself): bilinear (unit vectors, z=1 ⇒ 1) and the
    // true Gaussian (z=1 ⇒ exp(0)=1).
    #[test]
    fn diagonal_self_correlation_is_one() {
        let feats: Vec<f32> = (1..=9).map(|i| i as f32).collect(); // 3×3
        for kernel in [linear(), gauss()] {
            let psi = correlation_matrix(&feats, 3, 3, kernel).expect("ok");
            for i in 0..3 {
                let d = psi[i * 3 + i];
                assert!(
                    (d - 1.0).abs() < 1e-4,
                    "kernel {kernel:?} diagonal {i} = {d}, expected 1"
                );
            }
        }
    }

    // The truncated Taylor-RBF diagonal equals the *exact* truncated-series value
    // (z=1 ⇒ exp(−2γ)·Σ_{p=0..P} (2γ)^p/p!), approaching 1 from below as P grows.
    #[test]
    fn taylor_diagonal_matches_truncated_series() {
        let feats: Vec<f32> = (1..=9).map(|i| i as f32).collect(); // 3×3
        let gamma = 0.5_f32;
        let order = 4usize;
        let psi =
            correlation_matrix(&feats, 3, 3, CcKernel::TaylorRbf { gamma, order }).expect("ok");
        // Closed-form truncated value at z = 1.
        let prefactor = (-2.0 * gamma).exp();
        let a2 = 2.0 * gamma;
        let mut term = 1.0_f32;
        let mut series = 1.0_f32;
        for p in 1..=order {
            term *= a2 / p as f32;
            series += term;
        }
        let expected = prefactor * series;
        for i in 0..3 {
            let d = psi[i * 3 + i];
            assert!(
                (d - expected).abs() < 1e-5,
                "Taylor diagonal {i} = {d}, expected truncated value {expected}"
            );
        }
        // Higher order pushes the diagonal strictly closer to 1.
        let psi_hi =
            correlation_matrix(&feats, 3, 3, CcKernel::TaylorRbf { gamma, order: 16 }).expect("ok");
        assert!(
            (psi_hi[0] - 1.0).abs() < (psi[0] - 1.0).abs(),
            "higher Taylor order must bring the diagonal closer to 1"
        );
        assert!(psi_hi[0] <= 1.0 + 1e-5, "diagonal approaches 1 from below");
    }

    // Identical activations → 0 CC loss (Ψ_s == Ψ_t).
    #[test]
    fn identical_activations_zero_loss() {
        let feats: Vec<f32> = (0..15).map(|i| (i as f32) * 0.21 - 1.4).collect(); // 5×3
        for kernel in [linear(), gauss(), taylor()] {
            let loss = cc_loss(&feats, 3, &feats, 3, 5, kernel).expect("ok");
            assert!(loss < 1e-6, "kernel {kernel:?} identical loss {loss}");
        }
    }

    // Loss ≥ 0 and finite for mismatched batches.
    #[test]
    fn loss_nonneg_finite() {
        let s: Vec<f32> = (0..8).map(|i| i as f32).collect(); // 4×2
        let t: Vec<f32> = (0..8).map(|i| (8 - i) as f32).collect();
        for kernel in [linear(), gauss(), taylor()] {
            let loss = cc_loss(&s, 2, &t, 2, 4, kernel).expect("ok");
            assert!(
                loss >= 0.0 && loss.is_finite(),
                "kernel {kernel:?} loss {loss}"
            );
        }
    }

    // Captures which samples are treated similarly: two parallel samples have
    // correlation ~1, an orthogonal one has a smaller correlation.
    #[test]
    fn captures_relative_similarity() {
        // Sample 0 and 1 point the same way; sample 2 is orthogonal.
        let feats: Vec<f32> = vec![
            1.0, 0.0, // 0
            2.0, 0.0, // 1 (same direction as 0)
            0.0, 1.0, // 2 (orthogonal)
        ]; // 3×2
        let psi = correlation_matrix(&feats, 3, 2, linear()).expect("ok");
        // Bilinear on unit vectors: cos angle. ⟨0,1⟩ = 1, ⟨0,2⟩ = 0.
        assert!((psi[1] - 1.0).abs() < 1e-5, "parallel corr ~1");
        assert!(psi[2].abs() < 1e-5, "orthogonal corr ~0");
        assert!(
            psi[1] > psi[2],
            "similar samples must have larger correlation"
        );
        // Gaussian: parallel → exp(0) = 1; orthogonal → exp(−2γ) < 1.
        let psi_g = correlation_matrix(&feats, 3, 2, gauss()).expect("ok");
        assert!((psi_g[1] - 1.0).abs() < 1e-5);
        assert!(psi_g[2] < psi_g[1], "RBF: similar > dissimilar");
    }

    // Structural mismatch increases the loss: a student that preserves the
    // teacher's correlation structure costs less than one that destroys it.
    #[test]
    fn structure_mismatch_increases_loss() {
        let teacher: Vec<f32> = vec![
            1.0, 0.0, // 0
            1.0, 0.0, // 1 (== 0 direction)
            0.0, 1.0, // 2 (orthogonal)
        ];
        let student_match = teacher.clone();
        let student_wrong: Vec<f32> = vec![
            1.0, 0.0, // 0
            0.0, 1.0, // 1 (now orthogonal to 0 — wrong)
            0.0, 1.0, // 2
        ];
        for kernel in [linear(), gauss(), taylor()] {
            let l_match = cc_loss(&student_match, 2, &teacher, 2, 3, kernel).expect("ok");
            let l_wrong = cc_loss(&student_wrong, 2, &teacher, 2, 3, kernel).expect("ok");
            assert!(l_match < 1e-6, "kernel {kernel:?} matching loss {l_match}");
            assert!(
                l_wrong > l_match,
                "kernel {kernel:?}: wrong structure must cost more ({l_wrong} > {l_match})"
            );
        }
    }

    // The Taylor-RBF kernel approximates the true Gaussian RBF closely at modest
    // order — exercises the paper's core kernel construction with an exact oracle.
    #[test]
    fn taylor_approximates_gaussian() {
        let gamma = 0.5_f32;
        let feats: Vec<f32> = vec![1.0, 0.2, -0.3, 0.8, 0.1, 0.4]; // 2×3
        let psi_g = correlation_matrix(&feats, 2, 3, CcKernel::Gaussian { gamma }).expect("ok");
        let psi_t =
            correlation_matrix(&feats, 2, 3, CcKernel::TaylorRbf { gamma, order: 8 }).expect("ok");
        for (g, t) in psi_g.iter().zip(psi_t.iter()) {
            assert!(
                (g - t).abs() < 1e-4,
                "Taylor(order=8) must approximate Gaussian: {g} vs {t}"
            );
        }
    }

    // Higher Taylor order more closely matches the true exponential at z extremes.
    #[test]
    fn taylor_order_improves_accuracy() {
        let gamma = 1.0_f32;
        // Two anti-parallel unit samples: z = −1, the hardest point for the series.
        let feats: Vec<f32> = vec![1.0, 0.0, -1.0, 0.0]; // 2×2
        let true_val = (-gamma * 2.0 * (1.0 - (-1.0_f32))).exp(); // exp(−4γ) for z=−1
        let low =
            correlation_matrix(&feats, 2, 2, CcKernel::TaylorRbf { gamma, order: 2 }).expect("ok");
        let high =
            correlation_matrix(&feats, 2, 2, CcKernel::TaylorRbf { gamma, order: 12 }).expect("ok");
        let err_low = (low[1] - true_val).abs();
        let err_high = (high[1] - true_val).abs();
        assert!(
            err_high <= err_low + 1e-7,
            "higher order must not be worse: low={err_low} high={err_high}"
        );
    }

    // Different feature dimensions are allowed (only n must match).
    #[test]
    fn different_dims_ok() {
        let s: Vec<f32> = (0..6).map(|i| i as f32).collect(); // 3×2
        let t: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1).collect(); // 3×4
        let loss = cc_loss(&s, 2, &t, 4, 3, gauss()).expect("ok");
        assert!(loss.is_finite() && loss >= 0.0);
    }

    // Validation paths.
    #[test]
    fn empty_and_mismatch_errors() {
        assert!(matches!(
            cc_loss(&[], 0, &[1.0], 1, 1, linear()),
            Err(DistillError::EmptyInput)
        ));
        let s = vec![1.0_f32; 7]; // not 4*2
        let t = vec![1.0_f32; 8];
        assert!(matches!(
            cc_loss(&s, 2, &t, 2, 4, linear()),
            Err(DistillError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn bad_kernel_params_error() {
        let f = vec![1.0_f32; 4];
        assert!(matches!(
            correlation_matrix(&f, 2, 2, CcKernel::Gaussian { gamma: 0.0 }),
            Err(DistillError::InvalidConfig { .. })
        ));
        assert!(matches!(
            correlation_matrix(
                &f,
                2,
                2,
                CcKernel::TaylorRbf {
                    gamma: 0.5,
                    order: 0
                }
            ),
            Err(DistillError::InvalidConfig { .. })
        ));
        assert!(matches!(
            correlation_matrix(
                &f,
                2,
                2,
                CcKernel::TaylorRbf {
                    gamma: -1.0,
                    order: 3
                }
            ),
            Err(DistillError::InvalidConfig { .. })
        ));
    }
}
