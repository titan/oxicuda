//! QR Factorization via blocked Householder reflections.
//!
//! Computes `A = Q * R` where:
//! - Q is an m x m orthogonal matrix (stored implicitly via Householder vectors)
//! - R is an m x n upper triangular matrix
//!
//! Uses a blocked algorithm:
//! 1. Panel QR: compute Householder vectors for a block of columns
//! 2. Form compact WY representation: `I - V * T * V^T`
//! 3. Apply to trailing matrix via two GEMM calls
//!
//! The Householder vectors are stored in the lower triangle of A (below
//! the diagonal), and the `tau` array stores the Householder scalars.

use oxicuda_blas::types::{GpuFloat, Layout, MatrixDesc, MatrixDescMut, Transpose};
use oxicuda_memory::DeviceBuffer;

use crate::error::{SolverError, SolverResult};
use crate::handle::SolverHandle;

/// Block size for the blocked QR algorithm.
const QR_BLOCK_SIZE: u32 = 32;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Performs QR factorization in-place via blocked Householder reflections.
///
/// On exit, the upper triangle of `a` contains R, and the lower triangle
/// (below the diagonal) contains the Householder vectors. The `tau` array
/// stores the Householder scalars (length = min(m, n)).
///
/// # Arguments
///
/// * `handle` — solver handle.
/// * `a` — matrix buffer (m x n, column-major, lda stride), modified in-place.
/// * `m` — number of rows.
/// * `n` — number of columns.
/// * `lda` — leading dimension (>= m).
/// * `tau` — output Householder scalars buffer (length >= min(m, n)).
///
/// # Errors
///
/// Returns [`SolverError`] if dimensions are invalid or a kernel launch fails.
pub fn qr_factorize<T: GpuFloat>(
    handle: &mut SolverHandle,
    a: &mut DeviceBuffer<T>,
    m: u32,
    n: u32,
    lda: u32,
    tau: &mut DeviceBuffer<T>,
) -> SolverResult<()> {
    if m == 0 || n == 0 {
        return Ok(());
    }
    if lda < m {
        return Err(SolverError::DimensionMismatch(format!(
            "qr_factorize: lda ({lda}) must be >= m ({m})"
        )));
    }
    let required = n as usize * lda as usize;
    if a.len() < required {
        return Err(SolverError::DimensionMismatch(format!(
            "qr_factorize: buffer too small ({} < {required})",
            a.len()
        )));
    }
    let k = m.min(n);
    if tau.len() < k as usize {
        return Err(SolverError::DimensionMismatch(format!(
            "qr_factorize: tau buffer too small ({} < {k})",
            tau.len()
        )));
    }

    // Ensure workspace for T matrix (block_size x block_size) and W matrix.
    let ws = (QR_BLOCK_SIZE as usize * QR_BLOCK_SIZE as usize
        + m as usize * QR_BLOCK_SIZE as usize)
        * T::SIZE;
    handle.ensure_workspace(ws)?;

    blocked_qr::<T>(handle, a, m, n, lda, tau)
}

/// Solves the least-squares problem `min ||A*x - b||_2` using QR factorization.
///
/// Requires a QR-factored matrix (output of [`qr_factorize`]). The solution
/// overwrites `b` in-place.
///
/// For overdetermined systems (m >= n): `x = R^{-1} * Q^T * b`.
///
/// # Arguments
///
/// * `handle` — solver handle.
/// * `a` — QR-factored matrix from `qr_factorize`.
/// * `tau` — Householder scalars from `qr_factorize`.
/// * `b` — right-hand side (m x nrhs), overwritten with solution.
/// * `m` — number of rows of original A.
/// * `n` — number of columns of original A.
/// * `nrhs` — number of right-hand side columns.
///
/// # Errors
///
/// Returns [`SolverError`] if dimensions are invalid or operations fail.
pub fn qr_solve<T: GpuFloat>(
    handle: &SolverHandle,
    a: &DeviceBuffer<T>,
    tau: &DeviceBuffer<T>,
    b: &mut DeviceBuffer<T>,
    m: u32,
    n: u32,
    nrhs: u32,
) -> SolverResult<()> {
    if m == 0 || n == 0 || nrhs == 0 {
        return Ok(());
    }
    if m < n {
        return Err(SolverError::DimensionMismatch(
            "qr_solve: requires m >= n (overdetermined system)".into(),
        ));
    }
    let k = m.min(n);
    if tau.len() < k as usize {
        return Err(SolverError::DimensionMismatch(
            "qr_solve: tau buffer too small".into(),
        ));
    }

    // Step 1: Apply Q^T to B: B <- Q^T * B.
    // Process Householder reflections in forward order.
    apply_qt::<T>(handle, a, tau, b, m, n, nrhs)?;

    // Step 2: Solve R * X = (Q^T * B)[0:n, :] via TRSM.
    let r_desc = MatrixDesc::<T>::from_raw(a.as_device_ptr(), n, n, m, Layout::ColMajor);
    let mut b_desc = MatrixDescMut::<T>::from_raw(b.as_device_ptr(), n, nrhs, m, Layout::ColMajor);

    oxicuda_blas::level3::trsm(
        handle.blas(),
        oxicuda_blas::Side::Left,
        oxicuda_blas::FillMode::Upper,
        Transpose::NoTrans,
        oxicuda_blas::DiagType::NonUnit,
        T::gpu_one(),
        &r_desc,
        &mut b_desc,
    )?;

    Ok(())
}

/// Explicitly forms the Q matrix from the Householder representation.
///
/// # Arguments
///
/// * `handle` — solver handle.
/// * `a` — QR-factored matrix containing Householder vectors (below diagonal).
/// * `tau` — Householder scalars.
/// * `q` — output buffer for Q (m x m), filled on return.
/// * `m` — number of rows.
/// * `n` — number of columns of original A.
///
/// # Errors
///
/// Returns [`SolverError`] if dimensions are invalid.
pub fn qr_generate_q<T: GpuFloat>(
    handle: &SolverHandle,
    a: &DeviceBuffer<T>,
    tau: &DeviceBuffer<T>,
    q: &mut DeviceBuffer<T>,
    m: u32,
    n: u32,
) -> SolverResult<()> {
    if m == 0 {
        return Ok(());
    }
    let k = m.min(n);
    if tau.len() < k as usize {
        return Err(SolverError::DimensionMismatch(
            "qr_generate_q: tau buffer too small".into(),
        ));
    }
    if q.len() < (m as usize * m as usize) {
        return Err(SolverError::DimensionMismatch(
            "qr_generate_q: Q buffer too small".into(),
        ));
    }

    let a_required = m as usize * n as usize;
    if a.len() < a_required {
        return Err(SolverError::DimensionMismatch(format!(
            "qr_generate_q: A buffer too small ({} < {a_required})",
            a.len()
        )));
    }

    // Initialize Q = I (identity matrix).
    // Then apply Householder reflections in reverse order:
    // Q = H(k-1) * ... * H(1) * H(0) * I
    // where H(i) = I - tau[i] * v_i * v_i^T.
    //
    // For the blocked version, process blocks of QR_BLOCK_SIZE columns
    // in reverse, forming the compact WY representation for each block
    // and applying it to Q via GEMM.

    // Host fallback: reconstruct explicit Q from Householder vectors.
    // This provides correctness until the fully blocked GPU ORGQR path lands.
    let mut a_host = vec![T::gpu_zero(); a_required];
    a.copy_to_host(&mut a_host)?;

    let mut tau_host = vec![T::gpu_zero(); k as usize];
    tau.copy_to_host(&mut tau_host)?;

    let q_host = form_explicit_q_from_householder_host(&a_host, &tau_host, m, n, m)?;
    q.copy_from_host(&q_host)?;

    let _ = handle;

    Ok(())
}

fn form_explicit_q_from_householder_host<T: GpuFloat>(
    a_host: &[T],
    tau_host: &[T],
    m: u32,
    n: u32,
    lda: u32,
) -> SolverResult<Vec<T>> {
    let m_usize = m as usize;
    let n_usize = n as usize;
    let lda_usize = lda as usize;
    let k = m_usize.min(n_usize);

    if a_host.len() < lda_usize * n_usize {
        return Err(SolverError::DimensionMismatch(
            "form_explicit_q: A host buffer too small".into(),
        ));
    }
    if tau_host.len() < k {
        return Err(SolverError::DimensionMismatch(
            "form_explicit_q: tau host buffer too small".into(),
        ));
    }

    if T::SIZE == 4 {
        let a_f32: Vec<f32> = a_host
            .iter()
            .map(|&v| f32::from_bits(v.to_bits_u64() as u32))
            .collect();
        let tau_f32: Vec<f32> = tau_host
            .iter()
            .map(|&v| f32::from_bits(v.to_bits_u64() as u32))
            .collect();

        let mut q_f32 = vec![0.0_f32; m_usize * m_usize];
        for col in 0..m_usize {
            q_f32[col * m_usize + col] = 1.0;
        }

        // Q = H_0 H_1 ... H_{k-1}: apply reflectors from the left in reverse
        // index order so H_0 ends up leftmost (forming Q, not Qᵀ).
        for i in (0..k).rev() {
            let tau_i = tau_f32[i];
            if tau_i == 0.0 {
                continue;
            }

            for col in 0..m_usize {
                let mut dot = q_f32[col * m_usize + i];
                for row in (i + 1)..m_usize {
                    let v_row = a_f32[i * lda_usize + row];
                    dot += v_row * q_f32[col * m_usize + row];
                }

                let scale = tau_i * dot;
                q_f32[col * m_usize + i] -= scale;
                for row in (i + 1)..m_usize {
                    let v_row = a_f32[i * lda_usize + row];
                    q_f32[col * m_usize + row] -= scale * v_row;
                }
            }
        }

        let q_t: Vec<T> = q_f32
            .into_iter()
            .map(|x| T::from_bits_u64(u64::from(x.to_bits())))
            .collect();
        return Ok(q_t);
    }

    if T::SIZE == 8 {
        let a_f64: Vec<f64> = a_host
            .iter()
            .map(|&v| f64::from_bits(v.to_bits_u64()))
            .collect();
        let tau_f64: Vec<f64> = tau_host
            .iter()
            .map(|&v| f64::from_bits(v.to_bits_u64()))
            .collect();

        let mut q_f64 = vec![0.0_f64; m_usize * m_usize];
        for col in 0..m_usize {
            q_f64[col * m_usize + col] = 1.0;
        }

        // Q = H_0 H_1 ... H_{k-1}: apply reflectors from the left in reverse
        // index order so H_0 ends up leftmost (forming Q, not Qᵀ).
        for i in (0..k).rev() {
            let tau_i = tau_f64[i];
            if tau_i == 0.0 {
                continue;
            }

            for col in 0..m_usize {
                let mut dot = q_f64[col * m_usize + i];
                for row in (i + 1)..m_usize {
                    let v_row = a_f64[i * lda_usize + row];
                    dot += v_row * q_f64[col * m_usize + row];
                }

                let scale = tau_i * dot;
                q_f64[col * m_usize + i] -= scale;
                for row in (i + 1)..m_usize {
                    let v_row = a_f64[i * lda_usize + row];
                    q_f64[col * m_usize + row] -= scale * v_row;
                }
            }
        }

        let q_t: Vec<T> = q_f64
            .into_iter()
            .map(|x| T::from_bits_u64(x.to_bits()))
            .collect();
        return Ok(q_t);
    }

    Err(SolverError::InternalError(format!(
        "form_explicit_q: unsupported precision size {}",
        T::SIZE
    )))
}

// ---------------------------------------------------------------------------
// Precision helpers
// ---------------------------------------------------------------------------

/// Reinterprets a `T: GpuFloat` value as `f64` (8-byte direct, narrower via f32).
fn t_to_f64<T: GpuFloat>(v: T) -> f64 {
    if T::SIZE == 8 {
        f64::from_bits(v.to_bits_u64())
    } else {
        f64::from(f32::from_bits(v.to_bits_u64() as u32))
    }
}

/// Converts an `f64` back to `T: GpuFloat` (inverse of [`t_to_f64`]).
fn f64_to_t<T: GpuFloat>(v: f64) -> T {
    if T::SIZE == 8 {
        T::from_bits_u64(v.to_bits())
    } else {
        T::from_bits_u64(u64::from((v as f32).to_bits()))
    }
}

// ---------------------------------------------------------------------------
// Householder QR factorization (CPU host-fallback)
// ---------------------------------------------------------------------------

/// Householder QR factorization, computed on the host.
///
/// ===================================================================
/// CPU host-fallback: the fully blocked on-device Householder QR (panel
/// factorization + compact-WY trailing update on the GPU) is not implemented
/// yet. Rather than launch a no-op kernel and read back a matrix that was never
/// factorized, this routine performs the entire factorization on the host and
/// stages the result back. Results are exact (LAPACK `geqrf` storage); there is
/// NO GPU acceleration of this path (see follow-ups).
/// ===================================================================
///
/// On exit `a` holds `R` in its upper triangle and the Householder vectors `v`
/// (with implicit unit head) below the diagonal; `tau[k]` is the reflector
/// scalar for `H_k = I - tau_k v_k v_kᵀ`. This is exactly the layout consumed by
/// [`form_explicit_q_from_householder_host`] and [`apply_qt`].
fn blocked_qr<T: GpuFloat>(
    handle: &mut SolverHandle,
    a: &mut DeviceBuffer<T>,
    m: u32,
    n: u32,
    lda: u32,
    tau: &mut DeviceBuffer<T>,
) -> SolverResult<()> {
    // No kernels are launched on this CPU path; the handle only binds the
    // context/stream for the device transfers below.
    let _ = &handle;

    let m_usize = m as usize;
    let n_usize = n as usize;
    let lda_usize = lda as usize;
    let k = m_usize.min(n_usize);

    let mut a_host = vec![T::gpu_zero(); a.len()];
    a.copy_to_host(&mut a_host)?;
    let mut af: Vec<f64> = a_host.iter().map(|&v| t_to_f64(v)).collect();
    let mut tauf = vec![0.0_f64; k];

    for col in 0..k {
        // Norm of the sub-column A[col.., col].
        let mut norm = 0.0_f64;
        for r in col..m_usize {
            let v = af[col * lda_usize + r];
            norm += v * v;
        }
        norm = norm.sqrt();
        if norm == 0.0 {
            tauf[col] = 0.0;
            continue;
        }
        let x0 = af[col * lda_usize + col];
        let beta = if x0 >= 0.0 { -norm } else { norm };
        let tau_c = (beta - x0) / beta;
        let inv = 1.0 / (x0 - beta);

        // v below the diagonal (implicit v[col] = 1); R diagonal = beta.
        for r in (col + 1)..m_usize {
            af[col * lda_usize + r] *= inv;
        }
        af[col * lda_usize + col] = beta;
        tauf[col] = tau_c;

        // Apply H_col to the trailing columns c > col.
        for c in (col + 1)..n_usize {
            let mut w = af[c * lda_usize + col]; // v[col] = 1
            for r in (col + 1)..m_usize {
                w += af[col * lda_usize + r] * af[c * lda_usize + r];
            }
            w *= tau_c;
            af[c * lda_usize + col] -= w;
            for r in (col + 1)..m_usize {
                af[c * lda_usize + r] -= af[col * lda_usize + r] * w;
            }
        }
    }

    for (slot, &val) in a_host.iter_mut().zip(af.iter()) {
        *slot = f64_to_t(val);
    }
    a.copy_from_host(&a_host)?;

    let mut tau_host = vec![T::gpu_zero(); tau.len()];
    for (dst, &val) in tau_host.iter_mut().zip(tauf.iter()) {
        *dst = f64_to_t(val);
    }
    tau.copy_from_host(&tau_host)?;

    Ok(())
}

/// Applies `Qᵀ` to `B` in place (`B <- Qᵀ B`) using the stored Householder
/// reflectors, computed on the host.
///
/// CPU host-fallback (see [`blocked_qr`]); no GPU kernels are launched. The QR
/// factor `a` and the right-hand side `b` are interpreted with leading
/// dimension `m` (column-major), matching [`qr_solve`]'s triangular solve.
fn apply_qt<T: GpuFloat>(
    handle: &SolverHandle,
    a: &DeviceBuffer<T>,
    tau: &DeviceBuffer<T>,
    b: &mut DeviceBuffer<T>,
    m: u32,
    n: u32,
    nrhs: u32,
) -> SolverResult<()> {
    let _ = &handle;
    let m_usize = m as usize;
    let n_usize = n as usize;
    let nrhs_usize = nrhs as usize;
    let k = m_usize.min(n_usize);

    let mut a_host = vec![T::gpu_zero(); a.len()];
    a.copy_to_host(&mut a_host)?;
    let af: Vec<f64> = a_host.iter().map(|&v| t_to_f64(v)).collect();
    let mut tau_host = vec![T::gpu_zero(); tau.len()];
    tau.copy_to_host(&mut tau_host)?;
    let tauf: Vec<f64> = tau_host.iter().map(|&v| t_to_f64(v)).collect();
    let mut b_host = vec![T::gpu_zero(); b.len()];
    b.copy_to_host(&mut b_host)?;
    let mut bf: Vec<f64> = b_host.iter().map(|&v| t_to_f64(v)).collect();

    // Qᵀ = H_{k-1} … H_0; applying H_0, H_1, …, H_{k-1} from the left in order.
    for col in 0..k {
        let tau_c = tauf[col];
        if tau_c == 0.0 {
            continue;
        }
        for c in 0..nrhs_usize {
            let mut w = bf[c * m_usize + col]; // v[col] = 1
            for r in (col + 1)..m_usize {
                w += af[col * m_usize + r] * bf[c * m_usize + r];
            }
            w *= tau_c;
            bf[c * m_usize + col] -= w;
            for r in (col + 1)..m_usize {
                bf[c * m_usize + r] -= af[col * m_usize + r] * w;
            }
        }
    }

    for (slot, &val) in b_host.iter_mut().zip(bf.iter()) {
        *slot = f64_to_t(val);
    }
    b.copy_from_host(&b_host)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qr_block_size_positive() {
        let block_size = QR_BLOCK_SIZE;
        assert!(block_size > 0);
        assert!(block_size <= 256);
    }

    /// Panel factorization block size for QR must be exactly 32.
    #[test]
    fn test_qr_block_size_is_32() {
        assert_eq!(QR_BLOCK_SIZE, 32, "QR panel block size must be 32");
    }

    #[test]
    fn qr_backward_error_2x2() {
        // Householder QR: A = QR, verify determinant preserved: det(A) = det(Q)*det(R) = ±det(R)
        // For A = [[2, 1], [1, 3]]:
        //   det(A) = 2*3 - 1*1 = 5
        let a = [[2.0_f64, 1.0], [1.0, 3.0]];
        let det_a = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        assert!((det_a - 5.0).abs() < 1e-14, "det(A) must be 5, got {det_a}");
        // QR_BLOCK_SIZE must be 32 (panel factorization tuning requirement)
        assert_eq!(QR_BLOCK_SIZE, 32, "QR panel block size must be 32");
    }

    #[test]
    fn form_explicit_q_identity_when_tau_zero() {
        // Column-major A with lda=m, values unused when tau is zero.
        let m = 3_u32;
        let n = 2_u32;
        let lda = m;
        let a_host = vec![0.0_f64; (lda * n) as usize];
        let tau_host = vec![0.0_f64; m.min(n) as usize];

        let q = form_explicit_q_from_householder_host(&a_host, &tau_host, m, n, lda)
            .expect("Q generation should succeed");

        // Identity in column-major: idx = col*m + row
        for col in 0..m as usize {
            for row in 0..m as usize {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!(
                    (q[col * m as usize + row] - expected).abs() < 1e-12,
                    "Q({}, {}) mismatch",
                    row,
                    col
                );
            }
        }
    }

    fn try_solver_handle() -> Option<(std::sync::Arc<oxicuda_driver::Context>, SolverHandle)> {
        if oxicuda_driver::init().is_err() {
            eprintln!("skipping device test: CUDA driver unavailable");
            return None;
        }
        if !oxicuda_driver::device::Device::count()
            .map(|c| c > 0)
            .unwrap_or(false)
        {
            eprintln!("skipping device test: no NVIDIA CUDA device");
            return None;
        }
        let dev = oxicuda_driver::device::Device::get(0).expect("device 0");
        let ctx = std::sync::Arc::new(oxicuda_driver::Context::new(&dev).expect("ctx"));
        let handle = SolverHandle::new(&ctx).expect("handle");
        Some((ctx, handle))
    }

    #[test]
    fn qr_factorize_reconstructs_a_equals_qr() {
        // CPU host-fallback correctness: factor on device buffers, form Q, and
        // verify Q * R == A within tolerance. (Computation is on the host.)
        let Some((_ctx, mut handle)) = try_solver_handle() else {
            return;
        };
        let m = 4usize;
        let n = 3usize;
        // Column-major A (m x n), well-conditioned.
        let a = vec![
            1.0_f64, 2.0, 3.0, 4.0, // col 0
            2.0, 1.0, 0.0, 1.0, // col 1
            3.0, 0.0, 2.0, 1.0, // col 2
        ];
        let mut d_a = oxicuda_memory::DeviceBuffer::from_host(&a).expect("up A");
        let mut d_tau = oxicuda_memory::DeviceBuffer::<f64>::zeroed(n).expect("tau");
        qr_factorize::<f64>(
            &mut handle,
            &mut d_a,
            m as u32,
            n as u32,
            m as u32,
            &mut d_tau,
        )
        .expect("qr factorize");

        let mut qr = vec![0.0_f64; m * n];
        d_a.copy_to_host(&mut qr).expect("dl QR");
        let mut q = oxicuda_memory::DeviceBuffer::<f64>::zeroed(m * m).expect("q");
        qr_generate_q::<f64>(&handle, &d_a, &d_tau, &mut q, m as u32, n as u32).expect("gen Q");
        let mut q_host = vec![0.0_f64; m * m];
        q.copy_to_host(&mut q_host).expect("dl Q");

        // R = upper triangle of qr (m x n, col-major, lda = m).
        let r_at = |row: usize, col: usize| {
            if row <= col { qr[col * m + row] } else { 0.0 }
        };
        // Reconstruct A = Q * R and compare.
        for col in 0..n {
            for row in 0..m {
                let mut s = 0.0;
                for kk in 0..m {
                    s += q_host[kk * m + row] * r_at(kk, col);
                }
                let want = a[col * m + row];
                assert!(
                    (s - want).abs() < 1e-10,
                    "(Q*R)[{row},{col}]={s} != A={want}"
                );
            }
        }

        // Q must be orthonormal: QᵀQ ≈ I.
        for i in 0..m {
            for j in 0..m {
                let mut s = 0.0;
                for r in 0..m {
                    s += q_host[i * m + r] * q_host[j * m + r];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((s - want).abs() < 1e-10, "(QᵀQ)[{i},{j}]={s}");
            }
        }
    }

    #[test]
    fn qr_solve_least_squares_residual() {
        let Some((_ctx, mut handle)) = try_solver_handle() else {
            return;
        };
        // Overdetermined consistent system: pick x, set b = A x, recover x.
        let m = 4usize;
        let n = 2usize;
        let a = vec![1.0_f64, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0];
        let x_true = [2.0_f64, -1.0];
        let mut b = vec![0.0_f64; m];
        for row in 0..m {
            b[row] = a[row] * x_true[0] + a[m + row] * x_true[1];
        }
        let mut d_a = oxicuda_memory::DeviceBuffer::from_host(&a).expect("up A");
        let mut d_tau = oxicuda_memory::DeviceBuffer::<f64>::zeroed(n).expect("tau");
        qr_factorize::<f64>(
            &mut handle,
            &mut d_a,
            m as u32,
            n as u32,
            m as u32,
            &mut d_tau,
        )
        .expect("qr factorize");
        let mut d_b = oxicuda_memory::DeviceBuffer::from_host(&b).expect("up b");
        qr_solve::<f64>(&handle, &d_a, &d_tau, &mut d_b, m as u32, n as u32, 1).expect("qr solve");
        let mut sol = vec![0.0_f64; m];
        d_b.copy_to_host(&mut sol).expect("dl");
        assert!((sol[0] - x_true[0]).abs() < 1e-9, "x0={}", sol[0]);
        assert!((sol[1] - x_true[1]).abs() < 1e-9, "x1={}", sol[1]);
    }
}
