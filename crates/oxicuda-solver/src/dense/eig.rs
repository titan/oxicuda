//! Symmetric eigenvalue decomposition.
//!
//! Computes `A = Q * Λ * Q^T` for a real symmetric matrix A, where:
//! - Q is an orthogonal matrix whose columns are eigenvectors
//! - Λ is a diagonal matrix of eigenvalues in ascending order
//!
//! The algorithm proceeds in two stages:
//! 1. **Tridiagonalization**: Reduce A to tridiagonal form T via blocked Householder
//!    reflections: `A = Q_1 * T * Q_1^T`.
//! 2. **Tridiagonal QR iteration**: Apply implicit-shift QR iteration to T to
//!    compute eigenvalues (and optionally eigenvectors).
//! 3. **Back-transformation**: If eigenvectors are requested, accumulate the
//!    Householder reflections and QR rotations: `Q = Q_1 * Q_2`.

#![allow(dead_code)]

use oxicuda_blas::GpuFloat;
use oxicuda_memory::DeviceBuffer;

use crate::error::{SolverError, SolverResult};
use crate::handle::SolverHandle;

/// Maximum iterations for the tridiagonal QR algorithm.
const TRIDIAG_QR_MAX_ITER: u32 = 300;

/// Convergence tolerance for off-diagonal elements.
const TRIDIAG_QR_TOL: f64 = 1e-14;

/// Block size for the tridiagonalization step.
const TRIDIAG_BLOCK_SIZE: u32 = 64;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Controls what to compute in the eigendecomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EigJob {
    /// Compute eigenvalues only.
    ValuesOnly,
    /// Compute both eigenvalues and eigenvectors.
    ValuesAndVectors,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Computes eigenvalues (and optionally eigenvectors) of a symmetric matrix.
///
/// The matrix `a` is stored in column-major order with leading dimension `lda`.
/// Only the lower triangle is accessed. On exit:
/// - `eigenvalues` contains the eigenvalues in ascending order.
/// - If `job == ValuesAndVectors`, `a` is overwritten with the orthogonal
///   eigenvector matrix Q (column-major).
///
/// # Arguments
///
/// * `handle` — solver handle.
/// * `a` — symmetric matrix (n x n, column-major), destroyed/overwritten on output.
/// * `n` — matrix dimension.
/// * `lda` — leading dimension (>= n).
/// * `eigenvalues` — output buffer for eigenvalues (length >= n).
/// * `job` — controls what to compute.
///
/// # Errors
///
/// Returns [`SolverError::DimensionMismatch`] for invalid dimensions.
/// Returns [`SolverError::ConvergenceFailure`] if QR iteration does not converge.
pub fn syevd<T: GpuFloat>(
    handle: &mut SolverHandle,
    a: &mut DeviceBuffer<T>,
    n: u32,
    lda: u32,
    eigenvalues: &mut DeviceBuffer<T>,
    job: EigJob,
) -> SolverResult<()> {
    // Validate dimensions.
    if n == 0 {
        return Ok(());
    }
    if lda < n {
        return Err(SolverError::DimensionMismatch(format!(
            "syevd: lda ({lda}) must be >= n ({n})"
        )));
    }
    let required = n as usize * lda as usize;
    if a.len() < required {
        return Err(SolverError::DimensionMismatch(format!(
            "syevd: buffer too small ({} < {required})",
            a.len()
        )));
    }
    if eigenvalues.len() < n as usize {
        return Err(SolverError::DimensionMismatch(format!(
            "syevd: eigenvalues buffer too small ({} < {n})",
            eigenvalues.len()
        )));
    }

    // ===================================================================
    // CPU host-fallback computation.
    //
    // The full on-device symmetric eigensolver (blocked Householder
    // tridiagonalization + implicit-shift QR on the GPU) is not implemented
    // yet. Rather than launch a no-op kernel and read back fabricated values,
    // this routine performs the entire decomposition on the host and stages the
    // results back to the device buffers. Results are exact; there is NO
    // GPU acceleration of this path (see crate docs / follow-ups).
    // ===================================================================
    // The solver handle binds the device context/stream used for the buffer
    // transfers below; no kernels are launched on this CPU path.
    let _ = &handle;
    let n_usize = n as usize;
    let lda_usize = lda as usize;

    // Download the (symmetric) matrix and build a dense column-major f64 copy.
    let mut a_host = vec![T::gpu_zero(); a.len()];
    a.copy_to_host(&mut a_host)?;
    let mut sym = vec![0.0_f64; n_usize * n_usize];
    for col in 0..n_usize {
        for row in 0..n_usize {
            sym[col * n_usize + row] = t_to_f64(a_host[col * lda_usize + row]);
        }
    }
    // Symmetrize from the lower triangle (only the lower triangle is defined).
    for col in 0..n_usize {
        for row in (col + 1)..n_usize {
            let lower = sym[col * n_usize + row];
            sym[row * n_usize + col] = lower;
        }
    }

    // Step 1 (host): Householder reduction to symmetric tridiagonal form,
    // accumulating the orthogonal reduction matrix Q1 when vectors are wanted.
    let want_vectors = job == EigJob::ValuesAndVectors;
    let (mut d, mut e, q1) = host_tridiagonalize(&sym, n_usize, want_vectors);

    // Step 2 (host): implicit-shift QR iteration on the tridiagonal matrix,
    // accumulating the rotations into Q1 to form the full eigenvectors.
    let mut vectors = q1;
    let converged = tridiagonal_qr(&mut d, &mut e, n, vectors.as_deref_mut())?;
    if !converged {
        return Err(SolverError::ConvergenceFailure {
            iterations: TRIDIAG_QR_MAX_ITER,
            residual: e.iter().map(|v| v * v).sum::<f64>().sqrt(),
        });
    }

    // Sort eigenvalues ascending (and reorder eigenvector columns).
    sort_eigenvalues(&mut d, vectors.as_deref_mut(), n_usize);

    // Write eigenvalues back to the device buffer.
    let eig_stage = stage_eigenvalues_to_device::<T>(eigenvalues.len(), &d);
    eigenvalues.copy_from_host(&eig_stage)?;

    // Write eigenvectors (column-major) into A when requested.
    if let Some(vecs) = vectors {
        let stage = stage_eigenvectors_col_major_to_lda::<T>(&vecs, n_usize, lda_usize, a.len())?;
        a.copy_from_host(&stage)?;
    }

    Ok(())
}

/// Householder reduction of a dense symmetric matrix to tridiagonal form (host).
///
/// `sym` is the full `n x n` symmetric matrix in column-major order. Returns
/// `(d, e, q)` where `d[i]` is the tridiagonal diagonal, `e[i] = T[i+1, i]` is
/// the subdiagonal, and `q` (when `want_vectors`) is the column-major orthogonal
/// matrix `Q1` with `A = Q1 · T · Q1ᵀ`; `q` is `None` otherwise.
///
/// Each reflector `H_k = I - β v vᵀ` is applied symmetrically (`A ← H_k A H_k`)
/// using the rank-2 update `A ← A − v wᵀ − w vᵀ`, which is the standard,
/// numerically stable symmetric tridiagonalization (EISPACK `tred2`-style).
fn host_tridiagonalize(
    sym: &[f64],
    n: usize,
    want_vectors: bool,
) -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>) {
    // Working dense matrix (column-major); `at(i,j) = a[j*n + i]`.
    let mut a = sym.to_vec();
    let idx = |row: usize, col: usize| col * n + row;

    // Q accumulator (column-major), initialised to the identity.
    let mut q = if want_vectors {
        let mut m = vec![0.0_f64; n * n];
        for i in 0..n {
            m[idx(i, i)] = 1.0;
        }
        Some(m)
    } else {
        None
    };

    let mut v = vec![0.0_f64; n];
    let mut p = vec![0.0_f64; n];

    // Reflect columns 0..n-2 (the trailing 2x2 block is already tridiagonal).
    for k in 0..n.saturating_sub(2) {
        // Norm of the sub-column A[k+1.., k].
        let mut norm = 0.0_f64;
        for i in (k + 1)..n {
            norm += a[idx(i, k)] * a[idx(i, k)];
        }
        norm = norm.sqrt();
        if norm == 0.0 {
            continue;
        }
        let a1 = a[idx(k + 1, k)];
        let alpha = if a1 >= 0.0 { -norm } else { norm };

        // Householder vector v (support on rows k+1..n).
        for vi in v.iter_mut().take(n) {
            *vi = 0.0;
        }
        v[k + 1] = a1 - alpha;
        for i in (k + 2)..n {
            v[i] = a[idx(i, k)];
        }
        let mut vnorm2 = 0.0_f64;
        for &vi in v.iter().take(n).skip(k + 1) {
            vnorm2 += vi * vi;
        }
        if vnorm2 == 0.0 {
            continue;
        }
        let beta = 2.0 / vnorm2;

        // p = beta * A * v  (restricted to indices k+1..n).
        for pi in p.iter_mut().take(n) {
            *pi = 0.0;
        }
        for i in (k + 1)..n {
            let mut s = 0.0_f64;
            for j in (k + 1)..n {
                s += a[idx(i, j)] * v[j];
            }
            p[i] = beta * s;
        }
        // w = p - (beta/2)(vᵀ p) v  (reuse p as w).
        let mut vp = 0.0_f64;
        for i in (k + 1)..n {
            vp += v[i] * p[i];
        }
        let kk = 0.5 * beta * vp;
        for i in (k + 1)..n {
            p[i] -= kk * v[i];
        }
        // Rank-2 symmetric update A -= v wᵀ + w vᵀ.
        for j in (k + 1)..n {
            let vj = v[j];
            let wj = p[j];
            for i in (k + 1)..n {
                a[idx(i, j)] -= v[i] * wj + p[i] * vj;
            }
        }
        // Restore the explicit tridiagonal column/row k.
        a[idx(k + 1, k)] = alpha;
        a[idx(k, k + 1)] = alpha;
        for i in (k + 2)..n {
            a[idx(i, k)] = 0.0;
            a[idx(k, i)] = 0.0;
        }

        // Accumulate Q <- Q * H_k = Q - beta (Q v) vᵀ.
        if let Some(ref mut qm) = q {
            let mut qv = vec![0.0_f64; n];
            for r in 0..n {
                let mut s = 0.0_f64;
                for j in (k + 1)..n {
                    s += qm[idx(r, j)] * v[j];
                }
                qv[r] = s;
            }
            for col in (k + 1)..n {
                let vc = v[col] * beta;
                for r in 0..n {
                    qm[idx(r, col)] -= qv[r] * vc;
                }
            }
        }
    }

    let mut d = vec![0.0_f64; n];
    let mut e = vec![0.0_f64; n.saturating_sub(1)];
    for i in 0..n {
        d[i] = a[idx(i, i)];
    }
    for i in 0..n.saturating_sub(1) {
        e[i] = a[idx(i + 1, i)];
    }

    (d, e, q)
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

fn from_f64_to_t<T: GpuFloat>(val: f64) -> T {
    if T::SIZE == 8 {
        T::from_bits_u64(val.to_bits())
    } else {
        T::from_bits_u64(u64::from((val as f32).to_bits()))
    }
}

// ---------------------------------------------------------------------------
// Tridiagonal QR iteration
// ---------------------------------------------------------------------------

/// QR iteration with implicit Wilkinson shift for symmetric tridiagonal matrices.
///
/// Drives the subdiagonal elements to zero, leaving eigenvalues on the diagonal.
/// Optionally accumulates the rotation matrices into `vectors`.
///
/// Returns `true` if the algorithm converged within the iteration limit.
fn tridiagonal_qr(
    d: &mut [f64],
    e: &mut [f64],
    n: u32,
    mut vectors: Option<&mut [f64]>,
) -> SolverResult<bool> {
    let n_usize = n as usize;
    if n_usize <= 1 {
        return Ok(true);
    }

    let tol = TRIDIAG_QR_TOL;

    for _iter in 0..TRIDIAG_QR_MAX_ITER {
        // Find the active unreduced block.
        let mut q = n_usize - 1;
        while q > 0 && e[q - 1].abs() <= tol * (d[q - 1].abs() + d[q].abs()) {
            e[q - 1] = 0.0;
            q -= 1;
        }
        if q == 0 {
            return Ok(true);
        }

        let mut p = q - 1;
        while p > 0 && e[p - 1].abs() > tol * (d[p - 1].abs() + d[p].abs()) {
            p -= 1;
        }

        // Apply one implicit QR step with Wilkinson shift.
        implicit_qr_step(d, e, p, q, vectors.as_deref_mut(), n_usize);
    }

    // Check convergence.
    let off_norm: f64 = e.iter().map(|v| v * v).sum::<f64>().sqrt();
    Ok(off_norm <= tol)
}

/// One step of implicit QR with Wilkinson shift on T[start..=end, start..=end].
///
/// The Wilkinson shift is the eigenvalue of the trailing 2x2 block of T
/// that is closest to `T[end, end]`.
fn implicit_qr_step(
    d: &mut [f64],
    e: &mut [f64],
    start: usize,
    end: usize,
    mut vectors: Option<&mut [f64]>,
    n: usize,
) {
    // Compute Wilkinson shift.
    let delta = (d[end - 1] - d[end]) * 0.5;
    let sign_delta = if delta >= 0.0 { 1.0 } else { -1.0 };
    let e_sq = e[end - 1] * e[end - 1];
    let mu = d[end] - e_sq / (delta + sign_delta * (delta * delta + e_sq).sqrt());

    // Bulge chase using Givens rotations.
    let mut x = d[start] - mu;
    let mut z = e[start];

    for k in start..end {
        // Compute Givens rotation.
        let (cs, sn) = givens_rotation(x, z);

        // Apply rotation to T.
        if k > start {
            e[k - 1] = cs * x + sn * z;
        }
        let dk = d[k];
        let dk1 = d[k + 1];
        let ek = e[k];

        d[k] = cs * cs * dk + 2.0 * cs * sn * ek + sn * sn * dk1;
        d[k + 1] = sn * sn * dk - 2.0 * cs * sn * ek + cs * cs * dk1;
        e[k] = cs * sn * (dk1 - dk) + (cs * cs - sn * sn) * ek;

        // Create bulge for next step.
        if k + 1 < end {
            x = e[k];
            z = sn * e[k + 1];
            e[k + 1] *= cs;
        }

        // Accumulate rotation into eigenvector matrix.
        if let Some(ref mut vecs) = vectors.as_deref_mut() {
            for i in 0..n {
                let vi_k = vecs[k * n + i];
                let vi_k1 = vecs[(k + 1) * n + i];
                vecs[k * n + i] = cs * vi_k + sn * vi_k1;
                vecs[(k + 1) * n + i] = -sn * vi_k + cs * vi_k1;
            }
        }
    }
}

/// Computes a Givens rotation that zeros the second component.
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

/// Sorts eigenvalues in ascending order, rearranging eigenvectors accordingly.
fn sort_eigenvalues(d: &mut [f64], mut vectors: Option<&mut [f64]>, n: usize) {
    // Simple selection sort (n is typically small after tridiagonal reduction).
    for i in 0..n {
        let mut min_idx = i;
        let mut min_val = d[i];
        for (offset, &val) in d[(i + 1)..n].iter().enumerate() {
            if val < min_val {
                min_val = val;
                min_idx = i + 1 + offset;
            }
        }
        if min_idx != i {
            d.swap(i, min_idx);
            if let Some(ref mut vecs) = vectors.as_deref_mut() {
                // Swap columns i and min_idx.
                for row in 0..n {
                    let a = i * n + row;
                    let b = min_idx * n + row;
                    vecs.swap(a, b);
                }
            }
        }
    }
}

fn stage_eigenvalues_to_device<T: GpuFloat>(dst_len: usize, d: &[f64]) -> Vec<T> {
    let mut out = vec![T::gpu_zero(); dst_len];
    for (idx, &val) in d.iter().enumerate() {
        if idx >= dst_len {
            break;
        }
        out[idx] = from_f64_to_t(val);
    }
    out
}

fn stage_eigenvectors_col_major_to_lda<T: GpuFloat>(
    vectors: &[f64],
    n: usize,
    lda: usize,
    dst_len: usize,
) -> SolverResult<Vec<T>> {
    if vectors.len() < n * n {
        return Err(SolverError::DimensionMismatch(format!(
            "stage_eigenvectors_col_major_to_lda: vectors too small ({} < {})",
            vectors.len(),
            n * n
        )));
    }
    if dst_len < n * lda {
        return Err(SolverError::DimensionMismatch(format!(
            "stage_eigenvectors_col_major_to_lda: destination too small ({} < {})",
            dst_len,
            n * lda
        )));
    }

    let mut out = vec![T::gpu_zero(); dst_len];
    for col in 0..n {
        for row in 0..n {
            // vectors is n x n in column-major order.
            out[col * lda + row] = from_f64_to_t(vectors[col * n + row]);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eig_job_equality() {
        assert_eq!(EigJob::ValuesOnly, EigJob::ValuesOnly);
        assert_ne!(EigJob::ValuesOnly, EigJob::ValuesAndVectors);
    }

    #[test]
    fn givens_rotation_basic() {
        let (cs, sn) = givens_rotation(3.0, 4.0);
        let r = cs * 3.0 + sn * 4.0;
        assert!((r - 5.0).abs() < 1e-10);
    }

    #[test]
    fn givens_rotation_zero_b() {
        let (cs, sn) = givens_rotation(5.0, 0.0);
        assert!((cs - 1.0).abs() < 1e-15);
        assert!(sn.abs() < 1e-15);
    }

    #[test]
    fn sort_eigenvalues_basic() {
        let mut d = vec![3.0, 1.0, 2.0];
        sort_eigenvalues(&mut d, None, 3);
        assert!((d[0] - 1.0).abs() < 1e-15);
        assert!((d[1] - 2.0).abs() < 1e-15);
        assert!((d[2] - 3.0).abs() < 1e-15);
    }

    #[test]
    fn sort_eigenvalues_already_sorted() {
        let mut d = vec![1.0, 2.0, 3.0];
        sort_eigenvalues(&mut d, None, 3);
        assert!((d[0] - 1.0).abs() < 1e-15);
        assert!((d[2] - 3.0).abs() < 1e-15);
    }

    #[test]
    fn tridiag_qr_trivial() {
        let mut d = vec![1.0, 2.0, 3.0];
        let mut e = vec![0.0, 0.0];
        let result = tridiagonal_qr(&mut d, &mut e, 3, None);
        assert!(result.is_ok());
        assert!(result.ok() == Some(true));
    }

    #[test]
    fn tridiag_qr_single() {
        let mut d = vec![5.0];
        let mut e: Vec<f64> = vec![];
        let result = tridiagonal_qr(&mut d, &mut e, 1, None);
        assert!(result.is_ok());
    }

    #[test]
    fn host_tridiagonalize_reduces_and_reconstructs() {
        // Symmetric 4x4 (column-major == row-major for symmetric).
        let n = 4usize;
        let sym = vec![
            4.0_f64, 1.0, 2.0, 0.5, // col 0
            1.0, 3.0, 0.5, 1.5, // col 1
            2.0, 0.5, 5.0, 1.0, // col 2
            0.5, 1.5, 1.0, 6.0, // col 3
        ];
        let (d, e, q) = host_tridiagonalize(&sym, n, true);
        let q = q.expect("vectors requested");
        // Build T (tridiagonal) and verify Q * T * Q^T == sym.
        let mut t = vec![0.0_f64; n * n];
        for i in 0..n {
            t[i * n + i] = d[i];
        }
        for i in 0..n - 1 {
            t[i * n + (i + 1)] = e[i];
            t[(i + 1) * n + i] = e[i];
        }
        // qt = Q * T  (col-major).
        let at = |m: &[f64], r: usize, c: usize| m[c * n + r];
        let mut qt = vec![0.0_f64; n * n];
        for c in 0..n {
            for r in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += at(&q, r, k) * at(&t, k, c);
                }
                qt[c * n + r] = s;
            }
        }
        // rec = qt * Q^T.
        for c in 0..n {
            for r in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += qt[k * n + r] * at(&q, c, k); // Q^T[k,c] = Q[c,k]
                }
                let want = sym[c * n + r];
                assert!(
                    (s - want).abs() < 1e-10,
                    "reconstruct [{r},{c}] {s} != {want}"
                );
            }
        }
        // Off-tridiagonal entries of T must be (near) zero by construction.
        for i in 0..n {
            for j in 0..n {
                if j > i + 1 || (i > 0 && j + 1 < i) {
                    assert!(t[j * n + i].abs() < 1e-10, "T not tridiagonal at [{i},{j}]");
                }
            }
        }
    }

    #[test]
    fn host_eig_values_match_known_2x2() {
        // [[2,1],[1,2]] has eigenvalues 1 and 3.
        let sym = vec![2.0_f64, 1.0, 1.0, 2.0];
        let (mut d, mut e, _q) = host_tridiagonalize(&sym, 2, false);
        let ok = tridiagonal_qr(&mut d, &mut e, 2, None).expect("qr");
        assert!(ok);
        sort_eigenvalues(&mut d, None, 2);
        assert!((d[0] - 1.0).abs() < 1e-12, "lambda0 = {}", d[0]);
        assert!((d[1] - 3.0).abs() < 1e-12, "lambda1 = {}", d[1]);
    }

    #[test]
    fn stage_eigenvalues_prefix_copy() {
        let d = vec![1.5_f64, 2.5, 3.5];
        let out = stage_eigenvalues_to_device::<f64>(5, &d);
        assert_eq!(out.len(), 5);
        assert_eq!(out[0], 1.5);
        assert_eq!(out[1], 2.5);
        assert_eq!(out[2], 3.5);
        assert_eq!(out[3], 0.0);
        assert_eq!(out[4], 0.0);
    }

    #[test]
    fn stage_eigenvectors_to_lda_maps_columns() {
        // 2x2 column-major: col0=[1,2], col1=[3,4]
        let vecs = vec![1.0_f64, 2.0, 3.0, 4.0];
        let out = stage_eigenvectors_col_major_to_lda::<f64>(&vecs, 2, 3, 6);
        assert!(out.is_ok());
        let out = out.unwrap_or_default();
        assert_eq!(out.len(), 6);
        // col0 rows 0,1
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 2.0);
        // col1 rows 0,1 start at lda=3
        assert_eq!(out[3], 3.0);
        assert_eq!(out[4], 4.0);
        // padded lda rows remain zero
        assert_eq!(out[2], 0.0);
        assert_eq!(out[5], 0.0);
    }
}
