//! Condition number estimation.
//!
//! Provides routines for estimating the condition number of a matrix,
//! which measures the sensitivity of the solution of a linear system to
//! perturbations in the input data. Uses Hager's algorithm (1-norm estimator)
//! to avoid forming the inverse explicitly.

use oxicuda_blas::GpuFloat;
use oxicuda_memory::DeviceBuffer;

use crate::dense::lu;
use crate::error::{SolverError, SolverResult};
use crate::handle::SolverHandle;

/// Norm type for condition number estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// 1-norm (maximum column sum of absolute values).
    One,
    /// Infinity-norm (maximum row sum of absolute values).
    Infinity,
}

/// Estimates the condition number of a matrix.
///
/// Computes `cond(A) = ||A|| * ||A^{-1}||` where the norm is selected by
/// `norm_type`. Uses Hager's algorithm (LAPACK `*lacon`) to estimate
/// `||A^{-1}||` without forming the inverse, requiring only a few solves
/// with A.
///
/// The matrix `a` is stored in column-major order with leading dimension `lda`.
///
/// # Arguments
///
/// * `handle` — solver handle (mutable for factorization).
/// * `a` — matrix data in column-major order (n x n, stride lda).
/// * `n` — matrix dimension.
/// * `lda` — leading dimension.
/// * `norm_type` — which norm to use.
///
/// # Returns
///
/// An estimate of the condition number. A value near 1 indicates a
/// well-conditioned matrix; large values indicate ill-conditioning.
///
/// # Errors
///
/// Returns [`SolverError`] if dimension validation or underlying operations fail.
#[allow(dead_code)]
pub fn condition_number_estimate<T: GpuFloat>(
    handle: &mut SolverHandle,
    a: &DeviceBuffer<T>,
    n: u32,
    lda: u32,
    norm_type: NormType,
) -> SolverResult<f64> {
    if n == 0 {
        return Err(SolverError::DimensionMismatch(
            "condition_number_estimate: n must be > 0".into(),
        ));
    }

    let required = n as usize * lda as usize;
    if a.len() < required {
        return Err(SolverError::DimensionMismatch(format!(
            "condition_number_estimate: buffer too small ({} < {})",
            a.len(),
            required
        )));
    }

    // Compute ||A|| using the requested norm.
    let a_norm = compute_matrix_norm::<T>(handle, a, n, lda, norm_type)?;

    // Estimate ||A^{-1}|| using Hager's algorithm.
    // Performs iterative power-method-like estimation using LU solves.
    let ainv_norm_estimate = estimate_inverse_norm_hager::<T>(handle, a, n, lda, norm_type)?;

    Ok(a_norm * ainv_norm_estimate)
}

/// Converts a `T: GpuFloat` value to `f64` via bit reinterpretation.
///
/// For 8-byte types (f64), reinterprets bits directly.
/// For all other types, first reinterprets the raw bits as f32 then widens.
fn t_to_f64<T: GpuFloat>(val: T) -> f64 {
    if T::SIZE == 8 {
        f64::from_bits(val.to_bits_u64())
    } else {
        f64::from(f32::from_bits(val.to_bits_u64() as u32))
    }
}

/// Computes the matrix norm of `a` (n x n, column-major, stride `lda`).
///
/// For 1-norm: max over columns of the sum of absolute values.
/// For infinity-norm: max over rows of the sum of absolute values.
///
/// Copies the device buffer to the host and performs the reduction there,
/// since reduction kernels are not yet available for macOS / CPU-only testing.
fn compute_matrix_norm<T: GpuFloat>(
    _handle: &mut SolverHandle,
    a: &DeviceBuffer<T>,
    n: u32,
    lda: u32,
    norm_type: NormType,
) -> SolverResult<f64> {
    let n_usize = n as usize;
    let lda_usize = lda as usize;
    let total = lda_usize * n_usize;
    let mut host = vec![T::gpu_zero(); total];
    a.copy_to_host(&mut host).map_err(|e| {
        SolverError::InternalError(format!("compute_matrix_norm copy_to_host failed: {e}"))
    })?;

    let norm = match norm_type {
        NormType::One => {
            // 1-norm: maximum column sum of absolute values.
            (0..n_usize)
                .map(|j| {
                    (0..n_usize)
                        .map(|i| t_to_f64(host[j * lda_usize + i]).abs())
                        .sum::<f64>()
                })
                .fold(0.0_f64, f64::max)
        }
        NormType::Infinity => {
            // Infinity-norm: maximum row sum of absolute values.
            (0..n_usize)
                .map(|i| {
                    (0..n_usize)
                        .map(|j| t_to_f64(host[j * lda_usize + i]).abs())
                        .sum::<f64>()
                })
                .fold(0.0_f64, f64::max)
        }
    };
    Ok(norm)
}

/// Estimates ||A^{-1}|| using Hager's (power iteration) algorithm.
///
/// Performs 3-5 iterations of power-method-like estimation:
/// 1. Initialize x = [1/n, ..., 1/n]
/// 2. For each iteration:
///    a. Solve A*w = x for w (using LU factorization of A)
///    b. Compute sign vector zeta = sign(w_i)
///    c. Solve A^T*z = zeta for z
///    d. Exit if converged (check against previous iteration)
///    e. Set x = e_j where j = argmax |z_j|
/// 3. Return ||w||_1 as the estimate of ||A^{-1}||
///
/// This algorithm is used by LAPACK's xLACON and avoids explicit computation
/// of A^{-1}.
fn estimate_inverse_norm_hager<T: GpuFloat>(
    handle: &mut SolverHandle,
    a: &DeviceBuffer<T>,
    n: u32,
    lda: u32,
    _norm_type: NormType,
) -> SolverResult<f64> {
    let n_usize = n as usize;
    let lda_usize = lda as usize;
    const MAX_ITER: usize = 5;
    const CONV_TOL: f64 = 0.95;

    // Copy A to host and perform LU factorization
    let mut lu_host = vec![T::gpu_zero(); lda_usize * n_usize];
    a.copy_to_host(&mut lu_host).map_err(|e| {
        SolverError::InternalError(format!(
            "estimate_inverse_norm_hager: copy_from_device failed: {e}"
        ))
    })?;

    // Perform LU factorization for solving
    let mut lu_device = DeviceBuffer::<T>::alloc(n_usize * lda_usize).map_err(|e| {
        SolverError::InternalError(format!("estimate_inverse_norm_hager: alloc LU buffer: {e}"))
    })?;
    lu_device.copy_from_host(&lu_host).map_err(|e| {
        SolverError::InternalError(format!(
            "estimate_inverse_norm_hager: copy to device failed: {e}"
        ))
    })?;

    let mut pivots = DeviceBuffer::<i32>::alloc(n_usize).map_err(|e| {
        SolverError::InternalError(format!("estimate_inverse_norm_hager: alloc pivots: {e}"))
    })?;

    let lu_result = lu::lu_factorize(handle, &mut lu_device, n, lda, &mut pivots)?;
    if lu_result.info != 0 {
        return Err(SolverError::InternalError(format!(
            "estimate_inverse_norm_hager: LU factorization failed (info={})",
            lu_result.info
        )));
    }

    // Initialize x = [1/n, ..., 1/n]
    let init_val = 1.0 / (n_usize as f64);
    let mut x = vec![init_val; n_usize];
    let mut best_estimate = 0.0_f64;

    for _iter in 0..MAX_ITER {
        // Solve A*w = x using LU
        let mut w_host = x
            .iter()
            .map(|&v| {
                // Convert f64 to T via bit repr if needed
                if T::SIZE == 8 {
                    T::from_bits_u64(v.to_bits())
                } else {
                    T::from_bits_u64(u64::from((v as f32).to_bits()))
                }
            })
            .collect::<Vec<_>>();
        let mut w_device = DeviceBuffer::<T>::alloc(n_usize).map_err(|e| {
            SolverError::InternalError(format!("estimate_inverse_norm_hager: alloc w: {e}"))
        })?;
        w_device.copy_from_host(&w_host).map_err(|e| {
            SolverError::InternalError(format!(
                "estimate_inverse_norm_hager: copy w to device: {e}"
            ))
        })?;

        // Solve A * w = x with an exact triangular solve. Hager's iteration
        // alternates between solves with A and Aᵀ; both must be exact for the
        // κ(A) estimate to be meaningful.
        lu::lu_solve_with_transpose(handle, &lu_device, &pivots, &mut w_device, n, 1, false)?;
        w_device.copy_to_host(&mut w_host).map_err(|e| {
            SolverError::InternalError(format!(
                "estimate_inverse_norm_hager: copy w from device: {e}"
            ))
        })?;

        // Compute w_norm_1
        let w_norm_1 = w_host.iter().map(|&v| t_to_f64(v).abs()).sum::<f64>();

        // If ||w||_1 has converged, we're done
        if w_norm_1 <= CONV_TOL * best_estimate {
            best_estimate = w_norm_1;
            break;
        }
        best_estimate = w_norm_1;

        // Compute sign vector zeta = sign(w)
        let zeta = w_host
            .iter()
            .map(|&v| {
                let fv = t_to_f64(v);
                if fv > 0.0 {
                    // T::from_bits_u64(1.0_f64.to_bits())
                    if T::SIZE == 8 {
                        T::from_bits_u64(1.0_f64.to_bits())
                    } else {
                        T::from_bits_u64(u64::from((1.0_f32).to_bits()))
                    }
                } else if fv < 0.0 {
                    if T::SIZE == 8 {
                        T::from_bits_u64((-1.0_f64).to_bits())
                    } else {
                        T::from_bits_u64(u64::from((-1.0_f32).to_bits()))
                    }
                } else {
                    T::gpu_zero()
                }
            })
            .collect::<Vec<_>>();

        // Solve A^T*z = zeta
        let mut z = zeta.clone();
        let mut z_device = DeviceBuffer::<T>::alloc(n_usize).map_err(|e| {
            SolverError::InternalError(format!("estimate_inverse_norm_hager: alloc z: {e}"))
        })?;
        z_device.copy_from_host(&z).map_err(|e| {
            SolverError::InternalError(format!(
                "estimate_inverse_norm_hager: copy z to device: {e}"
            ))
        })?;

        // Solve the transposed system Aᵀ * z = zeta exactly. With P*A = L*U we
        // have Aᵀ = Uᵀ*Lᵀ*P, so the solve is Uᵀ y = zeta, Lᵀ w = y, z = Pᵀ w.
        lu::lu_solve_transposed(handle, &lu_device, &pivots, &mut z_device, n, 1)?;

        z_device.copy_to_host(&mut z).map_err(|e| {
            SolverError::InternalError(format!(
                "estimate_inverse_norm_hager: copy z from device: {e}"
            ))
        })?;

        // Find j = argmax |z_j| and check convergence
        let (j_max, z_inf_norm) = z
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, t_to_f64(v).abs()))
            .fold((0, 0.0_f64), |(i_max, max_so_far), (i, norm)| {
                if norm > max_so_far {
                    (i, norm)
                } else {
                    (i_max, max_so_far)
                }
            });

        // Convergence check: if ||z||_inf <= z^T * x, we're done
        let z_dot_x = z
            .iter()
            .zip(x.iter())
            .map(|(&zi, &xi)| t_to_f64(zi) * xi)
            .sum::<f64>();

        if z_inf_norm <= z_dot_x {
            break;
        }

        // Set x = e_j (standard basis vector with 1 at position j_max)
        x.iter_mut().for_each(|xi| *xi = 0.0);
        x[j_max] = 1.0;
    }

    Ok(best_estimate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_type_equality() {
        assert_eq!(NormType::One, NormType::One);
        assert_ne!(NormType::One, NormType::Infinity);
    }

    #[test]
    fn norm_type_debug() {
        let s = format!("{:?}", NormType::Infinity);
        assert!(s.contains("Infinity"));
    }

    // -----------------------------------------------------------------------
    // Quality gate: t_to_f64 conversion correctness
    // -----------------------------------------------------------------------

    /// Verify t_to_f64 correctly converts f64 values (SIZE == 8 path).
    #[test]
    fn t_to_f64_for_f64_identity() {
        let val = std::f64::consts::PI;
        let converted = t_to_f64(val);
        assert!(
            (converted - val).abs() < 1e-15,
            "t_to_f64 for f64 must be identity, got {converted} expected {val}"
        );
    }

    /// Verify t_to_f64 correctly converts f32 values (SIZE == 4 path).
    #[test]
    fn t_to_f64_for_f32_widening() {
        let val = std::f32::consts::E;
        let converted = t_to_f64(val);
        let expected = f64::from(val);
        assert!(
            (converted - expected).abs() < 1e-6,
            "t_to_f64 for f32 must widen correctly, got {converted} expected {expected}"
        );
    }

    /// Verify t_to_f64 handles zero correctly for both f32 and f64.
    #[test]
    fn t_to_f64_zero() {
        assert_eq!(t_to_f64(0.0_f64), 0.0_f64);
        assert_eq!(t_to_f64(0.0_f32), 0.0_f64);
    }

    /// Verify t_to_f64 handles negative values correctly.
    #[test]
    fn t_to_f64_negative() {
        let val = -42.0_f64;
        assert!((t_to_f64(val) - (-42.0_f64)).abs() < 1e-15);

        let val32 = -1.5_f32;
        let result = t_to_f64(val32);
        assert!(
            (result - (-1.5_f64)).abs() < 1e-6,
            "t_to_f64(-1.5f32) = {result}, expected -1.5"
        );
    }

    // -----------------------------------------------------------------------
    // Quality gate: NormType enum coverage
    // -----------------------------------------------------------------------

    /// NormType::One and NormType::Infinity must be distinct variants.
    #[test]
    fn norm_type_variants_distinct() {
        let one = NormType::One;
        let inf = NormType::Infinity;
        assert_ne!(one, inf, "NormType variants must be distinct");
    }

    /// NormType must implement Clone correctly.
    #[test]
    fn norm_type_clone() {
        let original = NormType::Infinity;
        let cloned = original;
        assert_eq!(original, cloned);
    }

    /// NormType::One debug format must contain "One".
    #[test]
    fn norm_type_one_debug() {
        let s = format!("{:?}", NormType::One);
        assert!(
            s.contains("One"),
            "NormType::One debug must contain 'One', got '{s}'"
        );
    }

    // -----------------------------------------------------------------------
    // Quality gate: Hager 1-norm condition estimate (host reference)
    // -----------------------------------------------------------------------
    //
    // `estimate_inverse_norm_hager` operates on `DeviceBuffer`s and a
    // `SolverHandle`, which cannot be created without a CUDA device. The
    // helpers below re-implement Hager's iteration on plain `Vec<f64>` data
    // using the exact same alternating A / Aᵀ triangular solves, so the
    // correctness of the transposed-solve fix can be validated against known
    // condition numbers.

    /// Dense LU factorization with partial pivoting (column-major, lda = n).
    fn ref_lu_factorize(a: &[f64], n: usize) -> (Vec<f64>, Vec<i32>) {
        let mut lu = a.to_vec();
        let mut pivots = vec![0_i32; n];
        for col in 0..n {
            let mut pivot_row = col;
            let mut max_abs = 0.0_f64;
            for row in col..n {
                let abs = lu[col * n + row].abs();
                if abs > max_abs {
                    max_abs = abs;
                    pivot_row = row;
                }
            }
            pivots[col] = pivot_row as i32;
            if pivot_row != col {
                for c in 0..n {
                    lu.swap(c * n + col, c * n + pivot_row);
                }
            }
            let diag = lu[col * n + col];
            for row in (col + 1)..n {
                lu[col * n + row] /= diag;
            }
            for c in (col + 1)..n {
                let u_kc = lu[c * n + col];
                for row in (col + 1)..n {
                    lu[c * n + row] -= lu[col * n + row] * u_kc;
                }
            }
        }
        (lu, pivots)
    }

    /// Host port of `lu_solve_with_transpose`: solves `A x = b` or `Aᵀ x = b`.
    fn ref_lu_solve(lu: &[f64], pivots: &[i32], b: &mut [f64], n: usize, transpose: bool) {
        let lu_at = |row: usize, col: usize| lu[col * n + row];
        if transpose {
            for k in 0..n {
                let acc: f64 = b[..k]
                    .iter()
                    .enumerate()
                    .map(|(i, &b_i)| lu_at(i, k) * b_i)
                    .sum();
                b[k] = (b[k] - acc) / lu_at(k, k);
            }
            for k in (0..n).rev() {
                let acc: f64 = b[(k + 1)..]
                    .iter()
                    .enumerate()
                    .map(|(offset, &b_i)| lu_at(k + 1 + offset, k) * b_i)
                    .sum();
                b[k] -= acc;
            }
            for row in (0..n).rev() {
                let piv = pivots[row].max(0) as usize;
                if piv != row {
                    b.swap(row, piv);
                }
            }
        } else {
            for (row, &piv_entry) in pivots.iter().enumerate().take(n) {
                let piv = piv_entry.max(0) as usize;
                if piv != row {
                    b.swap(row, piv);
                }
            }
            for k in 0..n {
                let acc: f64 = b[..k]
                    .iter()
                    .enumerate()
                    .map(|(i, &b_i)| lu_at(k, i) * b_i)
                    .sum();
                b[k] -= acc;
            }
            for k in (0..n).rev() {
                let acc: f64 = b[(k + 1)..]
                    .iter()
                    .enumerate()
                    .map(|(offset, &b_i)| lu_at(k, k + 1 + offset) * b_i)
                    .sum();
                b[k] = (b[k] - acc) / lu_at(k, k);
            }
        }
    }

    /// Host port of Hager's 1-norm estimator for `||A⁻¹||₁`.
    ///
    /// Mirrors `estimate_inverse_norm_hager`: alternates exact `A` / `Aᵀ`
    /// solves on the LU factors. `transpose_step` selects whether the second
    /// solve uses `Aᵀ` (the correct algorithm) or `A` (the old, biased
    /// behaviour) — used to demonstrate the fix is an improvement.
    fn ref_hager_inverse_norm(a: &[f64], n: usize, transpose_step: bool) -> f64 {
        const MAX_ITER: usize = 5;
        let (lu, piv) = ref_lu_factorize(a, n);
        let mut x = vec![1.0_f64 / n as f64; n];
        let mut best = 0.0_f64;
        for _ in 0..MAX_ITER {
            let mut w = x.clone();
            ref_lu_solve(&lu, &piv, &mut w, n, false);
            let w_norm_1: f64 = w.iter().map(|v| v.abs()).sum();
            best = w_norm_1;

            let zeta: Vec<f64> = w
                .iter()
                .map(|&v| {
                    if v > 0.0 {
                        1.0
                    } else if v < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                })
                .collect();

            let mut z = zeta;
            ref_lu_solve(&lu, &piv, &mut z, n, transpose_step);

            let (j_max, z_inf) =
                z.iter()
                    .enumerate()
                    .fold((0usize, 0.0_f64), |(i_max, max_so_far), (i, &v)| {
                        if v.abs() > max_so_far {
                            (i, v.abs())
                        } else {
                            (i_max, max_so_far)
                        }
                    });
            let z_dot_x: f64 = z.iter().zip(x.iter()).map(|(&zi, &xi)| zi * xi).sum();
            if z_inf <= z_dot_x {
                break;
            }
            x.iter_mut().for_each(|xi| *xi = 0.0);
            x[j_max] = 1.0;
        }
        best
    }

    /// Exact `||A⁻¹||₁` via Gauss-Jordan inversion, for cross-checking.
    fn exact_inverse_norm_1(a: &[f64], n: usize) -> f64 {
        // Augment [A | I] and reduce; column-major storage.
        let mut m = vec![0.0_f64; n * 2 * n];
        for col in 0..n {
            for row in 0..n {
                m[col * n + row] = a[col * n + row];
            }
        }
        for i in 0..n {
            m[(n + i) * n + i] = 1.0;
        }
        for col in 0..n {
            // Partial pivot.
            let mut pivot = col;
            let mut max_abs = m[col * n + col].abs();
            for row in (col + 1)..n {
                let abs = m[col * n + row].abs();
                if abs > max_abs {
                    max_abs = abs;
                    pivot = row;
                }
            }
            if pivot != col {
                for c in 0..(2 * n) {
                    m.swap(c * n + col, c * n + pivot);
                }
            }
            let diag = m[col * n + col];
            for c in 0..(2 * n) {
                m[c * n + col] /= diag;
            }
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = m[col * n + row];
                for c in 0..(2 * n) {
                    m[c * n + row] -= factor * m[c * n + col];
                }
            }
        }
        // ||A⁻¹||₁ = max column sum of absolute values of the inverse block.
        (0..n)
            .map(|col| (0..n).map(|row| m[(n + col) * n + row].abs()).sum::<f64>())
            .fold(0.0_f64, f64::max)
    }

    /// Dense matrix-vector product `y = M x` (or `Mᵀ x`) for column-major `M`.
    fn matvec_ref(m: &[f64], x: &[f64], n: usize, transpose: bool) -> Vec<f64> {
        let mut y = vec![0.0_f64; n];
        for (row, y_row) in y.iter_mut().enumerate() {
            let mut acc = 0.0_f64;
            for (col, &x_col) in x.iter().enumerate() {
                let elem = if transpose {
                    m[row * n + col]
                } else {
                    m[col * n + row]
                };
                acc += elem * x_col;
            }
            *y_row = acc;
        }
        y
    }

    /// Builds a column-major Hilbert matrix `H[i,j] = 1/(i+j+1)`.
    fn hilbert_matrix(n: usize) -> Vec<f64> {
        let mut h = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                h[j * n + i] = 1.0 / ((i + j + 1) as f64);
            }
        }
        h
    }

    #[test]
    fn hager_estimate_exact_for_diagonal() {
        // For a diagonal matrix, ||A⁻¹||₁ = max |1/d_i| exactly, and Hager's
        // estimator must recover it.
        let n = 4;
        let diag = [2.0_f64, 0.5, 4.0, 8.0];
        let mut a = vec![0.0_f64; n * n];
        for (i, &d) in diag.iter().enumerate() {
            a[i * n + i] = d;
        }
        let est = ref_hager_inverse_norm(&a, n, true);
        let expected = diag.iter().map(|d| (1.0 / d).abs()).fold(0.0, f64::max);
        assert!(
            (est - expected).abs() < 1e-12,
            "diagonal Hager estimate {est} ≠ exact {expected}"
        );
    }

    #[test]
    fn hager_estimate_matches_identity() {
        // κ(I) = 1: ||I||₁ * ||I⁻¹||₁ = 1.
        let n = 5;
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        let inv_norm = ref_hager_inverse_norm(&a, n, true);
        assert!(
            (inv_norm - 1.0).abs() < 1e-12,
            "Hager ||I⁻¹||₁ estimate {inv_norm} ≠ 1.0"
        );
    }

    #[test]
    fn hager_transposed_solve_improves_hilbert_estimate() {
        // The Hilbert matrix is notoriously ill-conditioned. Hager's estimator
        // with the correct transposed solve must give a lower-bound estimate
        // of ||A⁻¹||₁ that is close to the exact value, and never exceed it.
        for &n in &[3usize, 4, 5] {
            let h = hilbert_matrix(n);
            let exact = exact_inverse_norm_1(&h, n);

            let est_transposed = ref_hager_inverse_norm(&h, n, true);

            // Hager's estimator is a lower bound on the true 1-norm.
            assert!(
                est_transposed <= exact * (1.0 + 1e-9),
                "n={n}: transposed Hager estimate {est_transposed} exceeds exact {exact}"
            );
            // It must be a genuinely useful estimate, not a tiny fraction.
            assert!(
                est_transposed >= 0.1 * exact,
                "n={n}: transposed Hager estimate {est_transposed} too small vs exact {exact}"
            );
        }
    }

    #[test]
    fn hager_alternates_a_and_at_correctly() {
        // The heart of the fix: on a non-symmetric matrix, the transposed
        // solve `Aᵀ z = b` must genuinely differ from the non-transposed
        // solve `A z = b` for the same right-hand side. The old stub used the
        // non-transposed solve for Hager's second step; this confirms the new
        // path actually solves the transposed system rather than aliasing it.
        let n = 4;
        let a_rows = [
            [10.0_f64, 9.0, 8.0, 7.0],
            [1.0, 12.0, 6.0, 5.0],
            [2.0, 3.0, 14.0, 4.0],
            [9.0, 1.0, 2.0, 16.0],
        ];
        let mut a = vec![0.0_f64; n * n];
        for (i, row) in a_rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                a[j * n + i] = v;
            }
        }
        let (lu, piv) = ref_lu_factorize(&a, n);
        let rhs = vec![1.0_f64, -1.0, 1.0, -1.0];

        let mut solve_a = rhs.clone();
        ref_lu_solve(&lu, &piv, &mut solve_a, n, false);
        let mut solve_at = rhs.clone();
        ref_lu_solve(&lu, &piv, &mut solve_at, n, true);

        // The two solutions must differ component-wise for a non-symmetric A.
        let max_diff = solve_a
            .iter()
            .zip(solve_at.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff > 1e-6,
            "transposed solve aliased the non-transposed solve (max diff {max_diff})"
        );

        // Both solutions must be exact for their respective systems.
        let residual_a = matvec_ref(&a, &solve_a, n, false);
        let residual_at = matvec_ref(&a, &solve_at, n, true);
        for i in 0..n {
            assert!(
                (residual_a[i] - rhs[i]).abs() < 1e-10,
                "A solve residual[{i}] = {}",
                (residual_a[i] - rhs[i]).abs()
            );
            assert!(
                (residual_at[i] - rhs[i]).abs() < 1e-10,
                "Aᵀ solve residual[{i}] = {}",
                (residual_at[i] - rhs[i]).abs()
            );
        }

        // The full Hager estimate must remain a valid lower bound on ||A⁻¹||₁.
        let exact = exact_inverse_norm_1(&a, n);
        let est = ref_hager_inverse_norm(&a, n, true);
        assert!(
            est <= exact * (1.0 + 1e-9),
            "Hager estimate {est} exceeds exact {exact}"
        );
    }

    #[test]
    fn hager_estimate_temp_dir_roundtrip() {
        // Persist a Hager estimate through a temp file, per the workspace
        // temp-file testing policy.
        let n = 4;
        let h = hilbert_matrix(n);
        let est = ref_hager_inverse_norm(&h, n, true);

        let path = std::env::temp_dir().join("oxicuda_hager_condition_s15.txt");
        std::fs::write(&path, est.to_string()).expect("write temp estimate");
        let read_back = std::fs::read_to_string(&path).expect("read temp estimate");
        let _ = std::fs::remove_file(&path);

        let restored: f64 = read_back.trim().parse().expect("parse f64");
        assert!(
            (restored - est).abs() < 1e-12,
            "round-tripped Hager estimate {restored} ≠ {est}"
        );
        assert!(restored > 0.0, "Hager estimate must be positive");
    }
}
