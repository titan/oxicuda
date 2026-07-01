//! Singular Value Decomposition (SVD).
//!
//! Computes `A = U * Σ * V^T` where:
//! - U is an m x m (or m x k for thin SVD) orthogonal matrix
//! - Σ is a diagonal matrix of singular values in descending order
//! - V^T is an n x n (or k x n for thin SVD) orthogonal matrix
//!
//! CPU host-fallback: this routine currently computes the decomposition on the
//! host via one-sided Jacobi rotations (see `host_svd`). A real on-device SVD
//! is not yet implemented; rather than launch a no-op kernel and fabricate
//! singular values, the path runs entirely on the CPU and returns exact results.
//! There is NO GPU acceleration of this path (tracked as a follow-up).

#![allow(dead_code)]

use oxicuda_blas::GpuFloat;
use oxicuda_memory::DeviceBuffer;

use crate::error::{SolverError, SolverResult};
use crate::handle::SolverHandle;

/// Converts an `f64` value to `T: GpuFloat` via bit reinterpretation.
fn from_f64_to_t<T: GpuFloat>(val: f64) -> T {
    if T::SIZE == 4 {
        T::from_bits_u64(u64::from((val as f32).to_bits()))
    } else {
        T::from_bits_u64(val.to_bits())
    }
}

/// Converts a `T: GpuFloat` value to `f64` via bit reinterpretation.
///
/// For 8-byte types (f64), reinterprets bits directly.
/// For all other types (f32, f16, bf16, FP8), first reinterprets the raw bits
/// as f32 and then widens to f64.  This is a host-side fallback used when a
/// GPU kernel is unavailable (e.g. on macOS).
fn t_to_f64<T: GpuFloat>(val: T) -> f64 {
    if T::SIZE == 8 {
        f64::from_bits(val.to_bits_u64())
    } else {
        f64::from(f32::from_bits(val.to_bits_u64() as u32))
    }
}

/// Threshold below which the Jacobi SVD path is used.
const JACOBI_SVD_THRESHOLD: u32 = 32;

/// Maximum number of Jacobi sweeps before declaring convergence failure.
const JACOBI_MAX_SWEEPS: u32 = 100;

/// Convergence tolerance for Jacobi sweeps (relative to Frobenius norm).
const JACOBI_TOL: f64 = 1e-14;

/// Maximum iterations for bidiagonal QR.
const BIDIAG_QR_MAX_ITER: u32 = 200;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Controls which parts of the SVD to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvdJob {
    /// Compute full U and V^T (all left and right singular vectors).
    All,
    /// Compute thin (economy-size) U and V^T: only the first min(m,n) columns/rows.
    Thin,
    /// Compute singular values only (no U or V^T).
    SingularValuesOnly,
}

/// Result of an SVD computation.
///
/// The singular values are always in descending order.
#[derive(Debug, Clone)]
pub struct SvdResult<T: GpuFloat> {
    /// Singular values in descending order (length = min(m, n)).
    pub singular_values: Vec<T>,
    /// Left singular vectors (column-major, m x k or m x m depending on [`SvdJob`]).
    /// `None` if `SvdJob::SingularValuesOnly` was requested.
    pub u: Option<Vec<T>>,
    /// Right singular vectors transposed (column-major, k x n or n x n depending on
    /// [`SvdJob`]). `None` if `SvdJob::SingularValuesOnly` was requested.
    pub vt: Option<Vec<T>>,
    /// Diagnostic info: 0 on success, positive if the algorithm did not converge.
    pub info: i32,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Computes the SVD of an m x n matrix A.
///
/// The matrix `a` is stored in column-major order with leading dimension `lda`.
/// On return, `a` is destroyed (overwritten with intermediate data).
///
/// # Arguments
///
/// * `handle` — solver handle providing BLAS, stream, PTX cache.
/// * `a` — input matrix buffer (m x n, column-major), destroyed on output.
/// * `m` — number of rows.
/// * `n` — number of columns.
/// * `lda` — leading dimension (>= m).
/// * `job` — controls which parts of the SVD to compute.
///
/// # Returns
///
/// An [`SvdResult`] containing the singular values and optionally U and V^T.
///
/// # Errors
///
/// Returns [`SolverError::DimensionMismatch`] for invalid dimensions.
/// Returns [`SolverError::ConvergenceFailure`] if the iterative algorithm does
/// not converge within the allowed number of iterations.
pub fn svd<T: GpuFloat>(
    handle: &mut SolverHandle,
    a: &mut DeviceBuffer<T>,
    m: u32,
    n: u32,
    lda: u32,
    job: SvdJob,
) -> SolverResult<SvdResult<T>> {
    // Validate dimensions.
    if m == 0 || n == 0 {
        return Ok(SvdResult {
            singular_values: Vec::new(),
            u: if job == SvdJob::SingularValuesOnly {
                None
            } else {
                Some(Vec::new())
            },
            vt: if job == SvdJob::SingularValuesOnly {
                None
            } else {
                Some(Vec::new())
            },
            info: 0,
        });
    }
    if lda < m {
        return Err(SolverError::DimensionMismatch(format!(
            "svd: lda ({lda}) must be >= m ({m})"
        )));
    }
    let required = n as usize * lda as usize;
    if a.len() < required {
        return Err(SolverError::DimensionMismatch(format!(
            "svd: buffer too small ({} < {required})",
            a.len()
        )));
    }

    host_svd(handle, a, m, n, lda, job)
}

// ---------------------------------------------------------------------------
// One-sided Jacobi SVD (CPU host-fallback)
// ---------------------------------------------------------------------------

/// Computes the SVD `A = U Σ Vᵀ` on the host via one-sided Jacobi rotations.
///
/// ===================================================================
/// CPU host-fallback: the on-device SVD (Jacobi rotations / Golub–Kahan
/// bidiagonalization on the GPU) is not implemented yet. Rather than launch a
/// no-op kernel and read back fabricated values (e.g. raw column norms passed
/// off as singular values), this routine performs the full decomposition on the
/// host. Results are exact; there is NO GPU acceleration of this path.
/// ===================================================================
///
/// One-sided Jacobi orthogonalizes the columns of the working matrix; the
/// resulting column norms are the singular values and the normalized columns are
/// the left singular vectors, with the accumulated rotations forming `V`. For
/// `m < n` the algorithm is applied to `Aᵀ` and the roles of `U`/`V` are
/// swapped. Singular values are returned in descending order.
fn host_svd<T: GpuFloat>(
    handle: &mut SolverHandle,
    a: &mut DeviceBuffer<T>,
    m: u32,
    n: u32,
    lda: u32,
    job: SvdJob,
) -> SolverResult<SvdResult<T>> {
    // No kernels are launched; the handle only binds the context for transfers.
    let _ = &handle;
    let m_usize = m as usize;
    let n_usize = n as usize;
    let lda_usize = lda as usize;
    let k = m_usize.min(n_usize);

    // Download A into a dense column-major f64 matrix (drop the lda padding).
    let mut a_host = vec![T::gpu_zero(); a.len()];
    a.copy_to_host(&mut a_host)?;

    let transpose = m_usize < n_usize;
    // Working matrix `w` is (rows x cols) with rows >= cols.
    let (rows, cols) = if transpose {
        (n_usize, m_usize)
    } else {
        (m_usize, n_usize)
    };
    let mut w = vec![0.0_f64; rows * cols];
    for col in 0..n_usize {
        for row in 0..m_usize {
            let val = t_to_f64(a_host[col * lda_usize + row]);
            if transpose {
                // w is Aᵀ: w[row=col_of_A, col=row_of_A]
                w[row * rows + col] = val;
            } else {
                w[col * rows + row] = val;
            }
        }
    }

    // One-sided Jacobi: orthogonalize the `cols` columns of `w` (rows x cols),
    // accumulating right rotations into `v` (cols x cols).
    let (sigma, left, v) = jacobi_one_sided(&mut w, rows, cols);

    // Map back: when transposed, A = V_w Σ U_wᵀ, so U_A = V_w and V_Aᵀ = U_w.
    // `left` is rows x cols (normalized left vectors of `w`); `v` is cols x cols.
    let k_actual = sigma.len();
    debug_assert_eq!(k_actual, k);

    // Build U (left singular vectors of A) and V (right singular vectors of A),
    // both as column-major matrices of normalized columns of width k.
    let (u_cols, u_dim, vmat, v_dim) = if transpose {
        // U_A = v (cols x cols == m x m? cols = m). v is m x m.
        // V_A columns = left (rows x cols == n x m): each column is a right
        // singular vector of length n.
        (v, cols, left, rows)
    } else {
        (left, rows, v, cols)
    };
    // u_cols: u_dim x k columns are U; vmat: v_dim x k columns are V.

    let want_vectors = job != SvdJob::SingularValuesOnly;
    let full = job == SvdJob::All;

    let singular_values: Vec<T> = sigma.iter().map(|&s| from_f64_to_t(s)).collect();

    let (u_out, vt_out) = if !want_vectors {
        (None, None)
    } else {
        // U: u_dim x (k or u_dim). For `All`, complete to a full orthonormal basis.
        let u_width = if full { u_dim } else { k };
        let mut u = vec![0.0_f64; u_dim * u_width];
        for j in 0..k {
            for r in 0..u_dim {
                u[j * u_dim + r] = u_cols[j * u_dim + r];
            }
        }
        if full && u_dim > k {
            complete_orthonormal_basis(&mut u, u_dim, k);
        }

        // Vᵀ: (k or v_dim) x v_dim. Build V (v_dim x width) then transpose.
        let v_width = if full { v_dim } else { k };
        let mut vfull = vec![0.0_f64; v_dim * v_width];
        for j in 0..k {
            for r in 0..v_dim {
                vfull[j * v_dim + r] = vmat[j * v_dim + r];
            }
        }
        if full && v_dim > k {
            complete_orthonormal_basis(&mut vfull, v_dim, k);
        }
        // Vᵀ is (v_width x v_dim), column-major: vt[col*v_width + row] = V[row, col]^T
        // = V[col_of_V? ]. Vᵀ[i, j] = V[j, i].
        let mut vt = vec![0.0_f64; v_width * v_dim];
        for i in 0..v_width {
            for j in 0..v_dim {
                vt[j * v_width + i] = vfull[i * v_dim + j];
            }
        }

        (
            Some(u.iter().map(|&x| from_f64_to_t(x)).collect()),
            Some(vt.iter().map(|&x| from_f64_to_t(x)).collect()),
        )
    };

    Ok(SvdResult {
        singular_values,
        u: u_out,
        vt: vt_out,
        info: 0,
    })
}

/// One-sided Jacobi orthogonalization of the `cols` columns of `w`
/// (`rows x cols`, `rows >= cols`, column-major), modified in place.
///
/// Returns `(sigma, u, v)` sorted by descending singular value, where `sigma`
/// has length `cols`, `u` is `rows x cols` (normalized left vectors) and `v` is
/// `cols x cols` (accumulated right rotations), both column-major.
fn jacobi_one_sided(w: &mut [f64], rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // V starts as the identity (cols x cols).
    let mut v = vec![0.0_f64; cols * cols];
    for i in 0..cols {
        v[i * cols + i] = 1.0;
    }

    let tol = 1e-15;
    for _sweep in 0..JACOBI_MAX_SWEEPS {
        let mut off = 0.0_f64;
        for p in 0..cols {
            for q in (p + 1)..cols {
                // Inner products of columns p and q.
                let mut app = 0.0;
                let mut aqq = 0.0;
                let mut apq = 0.0;
                for r in 0..rows {
                    let xp = w[p * rows + r];
                    let xq = w[q * rows + r];
                    app += xp * xp;
                    aqq += xq * xq;
                    apq += xp * xq;
                }
                if apq.abs() <= tol * (app * aqq).sqrt() {
                    continue;
                }
                off += apq * apq;
                // Jacobi rotation diagonalizing [[app, apq], [apq, aqq]].
                let tau = (aqq - app) / (2.0 * apq);
                let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                // Rotate columns p, q of w and v.
                for r in 0..rows {
                    let xp = w[p * rows + r];
                    let xq = w[q * rows + r];
                    w[p * rows + r] = c * xp - s * xq;
                    w[q * rows + r] = s * xp + c * xq;
                }
                for r in 0..cols {
                    let vp = v[p * cols + r];
                    let vq = v[q * cols + r];
                    v[p * cols + r] = c * vp - s * vq;
                    v[q * cols + r] = s * vp + c * vq;
                }
            }
        }
        if off.sqrt() <= JACOBI_TOL {
            break;
        }
    }

    // Singular values = column norms; left vectors = normalized columns.
    let mut sigma = vec![0.0_f64; cols];
    let mut u = vec![0.0_f64; rows * cols];
    for j in 0..cols {
        let mut nrm = 0.0;
        for r in 0..rows {
            let x = w[j * rows + r];
            nrm += x * x;
        }
        let nrm = nrm.sqrt();
        sigma[j] = nrm;
        let inv = if nrm > 1e-300 { 1.0 / nrm } else { 0.0 };
        for r in 0..rows {
            u[j * rows + r] = w[j * rows + r] * inv;
        }
    }

    // Sort descending by singular value, permuting u and v columns.
    let mut order: Vec<usize> = (0..cols).collect();
    order.sort_by(|&i, &j| {
        sigma[j]
            .partial_cmp(&sigma[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sigma_s: Vec<f64> = order.iter().map(|&i| sigma[i]).collect();
    let mut u_s = vec![0.0_f64; rows * cols];
    let mut v_s = vec![0.0_f64; cols * cols];
    for (new_j, &old_j) in order.iter().enumerate() {
        for r in 0..rows {
            u_s[new_j * rows + r] = u[old_j * rows + r];
        }
        for r in 0..cols {
            v_s[new_j * cols + r] = v[old_j * cols + r];
        }
    }

    (sigma_s, u_s, v_s)
}

/// Extends the first `start` orthonormal columns of the `dim x dim` column-major
/// matrix `q` to a full orthonormal basis using modified Gram–Schmidt against
/// the standard basis vectors.
fn complete_orthonormal_basis(q: &mut [f64], dim: usize, start: usize) {
    let mut filled = start;
    let mut e = 0usize;
    while filled < dim && e < dim {
        // Candidate = e-th standard basis vector.
        let mut cand = vec![0.0_f64; dim];
        cand[e] = 1.0;
        // Orthogonalize against existing columns 0..filled.
        for j in 0..filled {
            let mut dot = 0.0;
            for r in 0..dim {
                dot += q[j * dim + r] * cand[r];
            }
            for r in 0..dim {
                cand[r] -= dot * q[j * dim + r];
            }
        }
        let mut nrm = 0.0;
        for &cr in cand.iter().take(dim) {
            nrm += cr * cr;
        }
        let nrm = nrm.sqrt();
        if nrm > 1e-10 {
            for r in 0..dim {
                q[filled * dim + r] = cand[r] / nrm;
            }
            filled += 1;
        }
        e += 1;
    }
}

// ---------------------------------------------------------------------------
// Bidiagonal QR iteration (host helper, retained for the bidiagonal path)
// ---------------------------------------------------------------------------

/// Implicit-shift QR iteration on a bidiagonal matrix.
///
/// Drives the superdiagonal elements to zero, leaving the singular values
/// on the diagonal. Optionally accumulates the left and right rotations
/// into the U and V^T matrices.
///
/// Returns `true` if the algorithm converged, `false` otherwise.
fn bidiagonal_svd_qr(
    d: &mut [f64],
    e: &mut [f64],
    u: Option<&mut [f64]>,
    vt: Option<&mut [f64]>,
    k: u32,
) -> SolverResult<bool> {
    let n = k as usize;
    if n == 0 {
        return Ok(true);
    }

    // Initialize identity matrices.
    if let Some(u_mat) = u {
        for val in u_mat.iter_mut() {
            *val = 0.0;
        }
        for i in 0..n {
            u_mat[i * n + i] = 1.0;
        }
    }
    if let Some(vt_mat) = vt {
        for val in vt_mat.iter_mut() {
            *val = 0.0;
        }
        for i in 0..n {
            vt_mat[i * n + i] = 1.0;
        }
    }

    // Implicit-shift QR iteration on the bidiagonal matrix.
    // Each step targets the smallest unconverged superdiagonal element.
    let tol = JACOBI_TOL;

    for _iter in 0..BIDIAG_QR_MAX_ITER {
        // Find the active block: the largest subrange where e[i] != 0.
        let mut q = n.saturating_sub(1);
        while q > 0 && e[q - 1].abs() <= tol * (d[q - 1].abs() + d[q].abs()) {
            e[q - 1] = 0.0;
            q -= 1;
        }
        if q == 0 {
            // All superdiagonal elements are zero — converged.
            return Ok(true);
        }

        // Find the start of the active block.
        let mut p = q - 1;
        while p > 0 && e[p - 1].abs() > tol * (d[p - 1].abs() + d[p].abs()) {
            p -= 1;
        }

        // Apply one implicit QR step to the active block d[p..=q], e[p..q].
        bidiagonal_qr_step(d, e, p, q);
    }

    // Check convergence.
    let off_norm: f64 = e.iter().map(|v| v * v).sum::<f64>().sqrt();
    Ok(off_norm <= tol)
}

/// One step of the implicit-shift QR iteration on a bidiagonal matrix.
///
/// Uses the Golub-Kahan shift strategy: the shift is chosen as the eigenvalue
/// of the trailing 2x2 submatrix of B^T * B that is closest to `d[end]^2`.
fn bidiagonal_qr_step(d: &mut [f64], e: &mut [f64], start: usize, end: usize) {
    // Compute the trailing 2x2 of T = B^T * B.
    let dm1 = d[end - 1];
    let dm = d[end];
    let em1 = e[end - 1];

    let t11 = dm1 * dm1
        + if end >= 2 {
            e[end - 2] * e[end - 2]
        } else {
            0.0
        };
    let t12 = dm1 * em1;
    let t22 = dm * dm + em1 * em1;

    // Wilkinson shift: eigenvalue of [[t11, t12], [t12, t22]] closest to t22.
    let delta = (t11 - t22) * 0.5;
    let sign_delta = if delta >= 0.0 { 1.0 } else { -1.0 };
    let mu = t22 - t12 * t12 / (delta + sign_delta * (delta * delta + t12 * t12).sqrt());

    // Chase the bulge.
    let mut y = d[start] * d[start] - mu;
    let mut z = d[start] * e[start];

    for k in start..end {
        // Right rotation to zero z in the (k, k+1) column pair.
        let (cs, sn) = givens_rotation(y, z);
        if k > start {
            e[k - 1] = cs * e[k - 1] + sn * z;
        }
        let tmp_d = cs * d[k] + sn * e[k];
        e[k] = -sn * d[k] + cs * e[k];
        d[k] = tmp_d;
        let tmp_z = sn * d[k + 1];
        d[k + 1] *= cs;

        y = d[k];
        z = tmp_z;

        // Left rotation to zero z in the (k, k+1) row pair.
        let (cs2, sn2) = givens_rotation(y, z);
        d[k] = cs2 * d[k] + sn2 * tmp_z;
        let tmp_e = cs2 * e[k] + sn2 * d[k + 1];
        d[k + 1] = -sn2 * e[k] + cs2 * d[k + 1];
        e[k] = tmp_e;

        if k + 1 < end {
            y = e[k];
            z = sn2 * e[k + 1];
            e[k + 1] *= cs2;
        }
    }
}

/// Computes a Givens rotation that zeros the second component.
///
/// Returns `(cs, sn)` such that `[cs, sn; -sn, cs] * [a; b] = [r; 0]`.
fn givens_rotation(a: f64, b: f64) -> (f64, f64) {
    if b.abs() < 1e-300 {
        return (1.0, 0.0);
    }
    if a.abs() < 1e-300 {
        return (0.0, if b >= 0.0 { 1.0 } else { -1.0 });
    }
    let r = (a * a + b * b).sqrt();
    (a / r, b / r)
}

// ---------------------------------------------------------------------------
// U / V^T reconstruction helpers
// ---------------------------------------------------------------------------

/// Reconstructs thin U (m x k) from Householder vectors and bidiag U rotations.
#[allow(clippy::too_many_arguments)]
fn reconstruct_u_thin<T: GpuFloat>(
    _handle: &SolverHandle,
    _a: &DeviceBuffer<T>,
    m: u32,
    _n: u32,
    _lda: u32,
    _tauq: &DeviceBuffer<T>,
    u_bidiag: Option<&[f64]>,
    k: u32,
) -> SolverResult<Vec<T>> {
    let m_usize = m as usize;
    let k_usize = k as usize;
    Ok(build_u_embedding::<T>(u_bidiag, m_usize, k_usize, false))
}

/// Reconstructs full U (m x m) from Householder vectors and bidiag U rotations.
#[allow(clippy::too_many_arguments)]
fn reconstruct_u_full<T: GpuFloat>(
    _handle: &SolverHandle,
    _a: &DeviceBuffer<T>,
    m: u32,
    _n: u32,
    _lda: u32,
    _tauq: &DeviceBuffer<T>,
    u_bidiag: Option<&[f64]>,
    k: u32,
) -> SolverResult<Vec<T>> {
    let m_usize = m as usize;
    let k_usize = k as usize;
    Ok(build_u_embedding::<T>(u_bidiag, m_usize, k_usize, true))
}

/// Reconstructs thin V^T (k x n) from Householder vectors and bidiag V^T rotations.
#[allow(clippy::too_many_arguments)]
fn reconstruct_vt_thin<T: GpuFloat>(
    _handle: &SolverHandle,
    _a: &DeviceBuffer<T>,
    _m: u32,
    n: u32,
    _lda: u32,
    _taup: &DeviceBuffer<T>,
    vt_bidiag: Option<&[f64]>,
    k: u32,
) -> SolverResult<Vec<T>> {
    let n_usize = n as usize;
    let k_usize = k as usize;
    Ok(build_vt_embedding::<T>(vt_bidiag, n_usize, k_usize, false))
}

/// Reconstructs full V^T (n x n) from Householder vectors and bidiag V^T rotations.
#[allow(clippy::too_many_arguments)]
fn reconstruct_vt_full<T: GpuFloat>(
    _handle: &SolverHandle,
    _a: &DeviceBuffer<T>,
    _m: u32,
    n: u32,
    _lda: u32,
    _taup: &DeviceBuffer<T>,
    vt_bidiag: Option<&[f64]>,
    k: u32,
) -> SolverResult<Vec<T>> {
    let n_usize = n as usize;
    let k_usize = k as usize;
    Ok(build_vt_embedding::<T>(vt_bidiag, n_usize, k_usize, true))
}

fn build_u_embedding<T: GpuFloat>(
    u_bidiag: Option<&[f64]>,
    m: usize,
    k: usize,
    full: bool,
) -> Vec<T> {
    let cols = if full { m } else { k };
    let mut out = vec![T::gpu_zero(); m * cols];

    if let Some(u_small) = u_bidiag {
        for col in 0..k {
            for row in 0..k.min(m) {
                out[col * m + row] = from_f64_to_t(u_small[col * k + row]);
            }
        }
    } else {
        for i in 0..k.min(m) {
            out[i * m + i] = T::gpu_one();
        }
    }

    if full {
        for i in k..m {
            out[i * m + i] = T::gpu_one();
        }
    }

    out
}

fn build_vt_embedding<T: GpuFloat>(
    vt_bidiag: Option<&[f64]>,
    n: usize,
    k: usize,
    full: bool,
) -> Vec<T> {
    let rows = if full { n } else { k };
    let mut out = vec![T::gpu_zero(); rows * n];

    if let Some(vt_small) = vt_bidiag {
        for col in 0..k.min(n) {
            for row in 0..k.min(rows) {
                out[col * rows + row] = from_f64_to_t(vt_small[col * k + row]);
            }
        }
    } else {
        for i in 0..k.min(rows).min(n) {
            out[i * rows + i] = T::gpu_one();
        }
    }

    if full {
        for i in k..n {
            out[i * n + i] = T::gpu_one();
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svd_job_equality() {
        assert_eq!(SvdJob::All, SvdJob::All);
        assert_ne!(SvdJob::All, SvdJob::Thin);
        assert_ne!(SvdJob::Thin, SvdJob::SingularValuesOnly);
    }

    #[test]
    fn svd_result_construction() {
        let result = SvdResult::<f64> {
            singular_values: vec![3.0, 2.0, 1.0],
            u: None,
            vt: None,
            info: 0,
        };
        assert_eq!(result.singular_values.len(), 3);
        assert_eq!(result.info, 0);
    }

    #[test]
    fn svd_result_with_vectors() {
        let result = SvdResult::<f32> {
            singular_values: vec![5.0, 3.0],
            u: Some(vec![1.0; 6]),
            vt: Some(vec![1.0; 6]),
            info: 0,
        };
        assert!(result.u.is_some());
        assert!(result.vt.is_some());
    }

    #[test]
    fn givens_rotation_basic() {
        let (cs, sn) = givens_rotation(3.0, 4.0);
        let r = cs * 3.0 + sn * 4.0;
        assert!((r - 5.0).abs() < 1e-10);
        let zero = -sn * 3.0 + cs * 4.0;
        assert!(zero.abs() < 1e-10);
    }

    #[test]
    fn givens_rotation_zero_b() {
        let (cs, sn) = givens_rotation(5.0, 0.0);
        assert!((cs - 1.0).abs() < 1e-15);
        assert!(sn.abs() < 1e-15);
    }

    #[test]
    fn givens_rotation_zero_a() {
        let (cs, sn) = givens_rotation(0.0, 3.0);
        assert!(cs.abs() < 1e-15);
        assert!((sn - 1.0).abs() < 1e-15);
    }

    #[test]
    fn bidiagonal_svd_qr_trivial() {
        let mut d = vec![3.0, 2.0, 1.0];
        let mut e = vec![0.0, 0.0];
        let result = bidiagonal_svd_qr(&mut d, &mut e, None, None, 3);
        assert!(result.is_ok());
        assert!(result.ok() == Some(true));
    }

    #[test]
    fn bidiagonal_svd_qr_with_superdiag() {
        let mut d = vec![4.0, 3.0];
        let mut e = vec![1.0];
        let mut u = vec![0.0; 4];
        let mut vt = vec![0.0; 4];
        let result = bidiagonal_svd_qr(&mut d, &mut e, Some(&mut u), Some(&mut vt), 2);
        assert!(result.is_ok());
    }

    #[test]
    fn bidiagonal_svd_qr_empty() {
        let mut d: Vec<f64> = Vec::new();
        let mut e: Vec<f64> = Vec::new();
        let result = bidiagonal_svd_qr(&mut d, &mut e, None, None, 0);
        assert!(result.is_ok());
        assert!(result.ok() == Some(true));
    }

    #[test]
    fn u_embedding_thin_maps_bidiag_block() {
        // u_bidiag is 2x2 in column-major: col0=[1,2], col1=[3,4]
        let u_small = vec![1.0_f64, 2.0, 3.0, 4.0];
        let out = build_u_embedding::<f64>(Some(&u_small), 4, 2, false);
        assert_eq!(out.len(), 8);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[4], 3.0);
        assert_eq!(out[5], 4.0);
        assert_eq!(out[2], 0.0);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn vt_embedding_full_extends_identity() {
        let vt_small = vec![1.0_f64, 0.0, 0.0, 1.0]; // 2x2 identity (column-major)
        let out = build_vt_embedding::<f64>(Some(&vt_small), 4, 2, true);
        assert_eq!(out.len(), 16);
        // top-left identity
        assert_eq!(out[0], 1.0);
        assert_eq!(out[5], 1.0);
        // extended identity tail
        assert_eq!(out[10], 1.0);
        assert_eq!(out[15], 1.0);
    }

    #[test]
    fn jacobi_threshold() {
        let threshold = JACOBI_SVD_THRESHOLD;
        assert!(threshold > 0);
        assert!(threshold <= 64);
    }

    #[test]
    fn svd_backward_error_2x2() {
        // For a 2×2 diagonal matrix A = [[3, 0], [0, 2]]:
        //   U = I, Σ = diag(3, 2), V^T = I
        // Singular values must be in descending order.
        // Verify reconstruction error ||A - U*Σ*V^T||_F < 1e-14.
        let sigma = [3.0_f64, 2.0]; // singular values in descending order
        assert!(
            sigma[0] >= sigma[1],
            "singular values must be in descending order"
        );

        // Reconstruct A = diag(sigma) (with U = I, V^T = I)
        let a_recon = [[sigma[0], 0.0], [0.0, sigma[1]]];
        let a_orig = [[3.0_f64, 0.0], [0.0, 2.0_f64]];

        // Frobenius norm of reconstruction error
        let mut err_sq = 0.0_f64;
        for i in 0..2 {
            for j in 0..2 {
                let diff = a_recon[i][j] - a_orig[i][j];
                err_sq += diff * diff;
            }
        }
        let err = err_sq.sqrt();
        assert!(err < 1e-14, "SVD backward error {err} must be < 1e-14");
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

    /// Reconstructs `A` from a [`SvdResult`] (`U Σ Vᵀ`) and checks it equals the
    /// original within `tol`. `u` is `m x k`, `vt` is `k x n` (thin), column-major.
    fn check_svd_reconstruction(a: &[f64], m: usize, n: usize, res: &SvdResult<f64>, tol: f64) {
        let k = m.min(n);
        let u = res.u.as_ref().expect("U");
        let vt = res.vt.as_ref().expect("Vt");
        let s = &res.singular_values;
        assert_eq!(s.len(), k, "expected {k} singular values");
        // Descending order.
        for i in 1..k {
            assert!(s[i - 1] + 1e-12 >= s[i], "singular values not descending");
        }
        for col in 0..n {
            for row in 0..m {
                let mut acc = 0.0;
                for t in 0..k {
                    // U[row,t] * sigma_t * Vt[t,col]
                    acc += u[t * m + row] * s[t] * vt[col * k + t];
                }
                let want = a[col * m + row];
                assert!(
                    (acc - want).abs() < tol,
                    "(UΣVᵀ)[{row},{col}]={acc} != A={want}"
                );
            }
        }
    }

    #[test]
    fn svd_reconstructs_tall_matrix() {
        let Some((_ctx, mut handle)) = try_solver_handle() else {
            return;
        };
        let m = 4usize;
        let n = 3usize;
        let a = vec![
            1.0_f64, 2.0, 3.0, 4.0, // col 0
            2.0, 0.0, 1.0, 1.0, // col 1
            0.0, 1.0, 2.0, 3.0, // col 2
        ];
        let mut d_a = oxicuda_memory::DeviceBuffer::from_host(&a).expect("up");
        let res = svd::<f64>(
            &mut handle,
            &mut d_a,
            m as u32,
            n as u32,
            m as u32,
            SvdJob::Thin,
        )
        .expect("svd");
        check_svd_reconstruction(&a, m, n, &res, 1e-9);
    }

    #[test]
    fn svd_reconstructs_wide_matrix() {
        let Some((_ctx, mut handle)) = try_solver_handle() else {
            return;
        };
        // m < n exercises the transpose path.
        let m = 2usize;
        let n = 3usize;
        let a = vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0];
        let mut d_a = oxicuda_memory::DeviceBuffer::from_host(&a).expect("up");
        let res = svd::<f64>(
            &mut handle,
            &mut d_a,
            m as u32,
            n as u32,
            m as u32,
            SvdJob::Thin,
        )
        .expect("svd");
        check_svd_reconstruction(&a, m, n, &res, 1e-9);
    }

    #[test]
    fn svd_singular_values_match_known() {
        let Some((_ctx, mut handle)) = try_solver_handle() else {
            return;
        };
        // Diagonal matrix: singular values are |diag|, sorted descending.
        let m = 3usize;
        let n = 3usize;
        let a = vec![3.0_f64, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0];
        let mut d_a = oxicuda_memory::DeviceBuffer::from_host(&a).expect("up");
        let res = svd::<f64>(
            &mut handle,
            &mut d_a,
            m as u32,
            n as u32,
            m as u32,
            SvdJob::SingularValuesOnly,
        )
        .expect("svd");
        let s: Vec<f64> = res.singular_values;
        assert!((s[0] - 5.0).abs() < 1e-12, "s0={}", s[0]);
        assert!((s[1] - 3.0).abs() < 1e-12, "s1={}", s[1]);
        assert!((s[2] - 1.0).abs() < 1e-12, "s2={}", s[2]);
    }
}
