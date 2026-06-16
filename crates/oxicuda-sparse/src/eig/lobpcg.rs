//! LOBPCG eigensolver for the smallest eigenpairs of a sparse SPD matrix.
//!
//! Implements the Locally Optimal Block Preconditioned Conjugate Gradient
//! method (Knyazev, 2001) for the standard symmetric eigenproblem `A x = λ x`
//! with `A` symmetric positive-definite. The method computes the `k` smallest
//! eigenpairs simultaneously by minimising the block Rayleigh quotient over the
//! locally optimal subspace spanned by the current iterate block `X`, the
//! (preconditioned) residual block `W`, and the previous search-direction block
//! `P`.
//!
//! ## Outline
//!
//! Starting from a deterministic random block `X` (n × k), each iteration:
//! 1. forms `A X` (column-wise sparse mat-vecs);
//! 2. computes the Ritz values `Λ` and residual block `R = A X − X Λ`;
//! 3. checks per-column convergence `‖r‖ / (|λ| ‖x‖) < tol`;
//! 4. applies the (diagonal Jacobi) preconditioner `W = T R`;
//! 5. assembles the trial subspace `S = [X | W | P]` and orthonormalises it,
//!    discarding numerically dependent columns;
//! 6. solves the small dense symmetric Rayleigh-Ritz problem `Sᵀ A S` for its
//!    `k` smallest eigenpairs (via the inline cyclic-Jacobi solver);
//! 7. updates `X = S C` and forms the new search direction from the `W`/`P`
//!    portions of the Ritz coefficients (the locally-optimal CG recurrence).
//!
//! Because `B = I`, the `B`-orthonormalisation reduces to ordinary modified
//! Gram-Schmidt orthonormalisation.

use crate::eig::dense_sym::jacobi_eigh;
use crate::error::{SparseError, SparseResult};
use crate::host_csr::HostCsr;

/// Multiplier `state * MULT + INCR (mod 2⁶⁴)` for the deterministic initial
/// block. Matches the crate's `LcgRng` recipe used across the OxiCUDA handles.
const LCG_MULT: u64 = 6_364_136_223_846_793_005;
/// Increment for the deterministic LCG (see [`LCG_MULT`]).
const LCG_INCR: u64 = 1_442_695_040_888_963_407;
/// Threshold below which a Gram-Schmidt column is treated as dependent.
const ORTHO_DROP: f64 = 1e-10;

/// Result of a LOBPCG solve.
pub struct LobpcgResult {
    /// The computed eigenvalues in ascending order (length up to `k`).
    pub eigenvalues: Vec<f64>,
    /// The computed eigenvectors, `eigenvectors[j]` matching `eigenvalues[j]`.
    /// Each inner vector has length `n` and is `A`-orthonormal in exact
    /// arithmetic (Euclidean-orthonormal as returned).
    pub eigenvectors: Vec<Vec<f64>>,
    /// Number of iterations performed.
    pub iters: usize,
    /// Whether all requested eigenpairs reached the requested tolerance.
    pub converged: bool,
}

/// A simple linear congruential generator mirroring the crate's `LcgRng`.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(LCG_MULT).wrapping_add(LCG_INCR);
        (self.state >> 32) as u32
    }

    /// Uniform sample in `[-1, 1)`.
    fn next_signed(&mut self) -> f64 {
        let u = self.next_u32() as f64 / (u32::MAX as f64 + 1.0);
        2.0 * u - 1.0
    }
}

/// Computes the `k` smallest eigenpairs of a sparse SPD matrix `a`.
///
/// # Arguments
///
/// * `a` -- a square symmetric positive-definite host CSR matrix.
/// * `k` -- the number of smallest eigenpairs to compute (`1 ≤ k ≤ n`).
/// * `max_iter` -- the maximum number of LOBPCG iterations.
/// * `tol` -- the relative residual tolerance for convergence.
/// * `precond` -- an optional diagonal (Jacobi) preconditioner of length `n`,
///   typically `1 / diag(A)`; when `None`, no preconditioning is applied.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `a` is not square, if `precond`
/// has the wrong length; [`SparseError::InvalidArgument`] if `k` is zero or
/// exceeds `n`; or propagates errors from the dense Rayleigh-Ritz solve.
pub fn lobpcg(
    a: &HostCsr,
    k: usize,
    max_iter: usize,
    tol: f64,
    precond: Option<&[f64]>,
) -> SparseResult<LobpcgResult> {
    if a.nrows != a.ncols {
        return Err(SparseError::DimensionMismatch(format!(
            "LOBPCG requires a square matrix, got {}x{}",
            a.nrows, a.ncols
        )));
    }
    let n = a.nrows;
    if k == 0 {
        return Err(SparseError::InvalidArgument(
            "LOBPCG requires k >= 1".to_string(),
        ));
    }
    if k > n {
        return Err(SparseError::InvalidArgument(format!(
            "requested k = {k} exceeds matrix dimension n = {n}"
        )));
    }
    if let Some(t) = precond {
        if t.len() != n {
            return Err(SparseError::DimensionMismatch(format!(
                "preconditioner length {} must equal n = {n}",
                t.len()
            )));
        }
    }

    // Scale used to relativise residuals. For the standard problem the natural
    // scale is the spectral norm; we approximate it cheaply by the largest
    // absolute row sum (an upper bound on ‖A‖₂). Using this rather than |λ|
    // avoids the spurious convergence floor `ε‖A‖ / λ` that otherwise prevents
    // small eigenvalues from ever meeting a relative tolerance.
    let norm_scale = a_norm_estimate(a);

    // ---- Initial block X (n columns laid out as Vec of length-n columns) ----
    let mut x: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut rng = Lcg::new(0x5eed_1234_abcd_ef01);
    for _ in 0..k {
        let mut col = vec![0.0f64; n];
        for slot in col.iter_mut() {
            *slot = rng.next_signed();
        }
        x.push(col);
    }
    // Seed extra deterministic structure so degenerate all-equal columns are
    // avoided for tiny matrices: add a unit spike per column.
    for (j, col) in x.iter_mut().enumerate() {
        col[j % n] += 1.0;
    }
    orthonormalize(&mut x);

    let mut p: Vec<Vec<f64>> = Vec::new();
    let mut eigenvalues = vec![0.0f64; k];
    let mut converged = false;
    let mut iters_done = 0usize;

    for iter in 0..max_iter {
        iters_done = iter + 1;

        // A X and Ritz values via Rayleigh quotient of current X.
        let ax: Vec<Vec<f64>> = x.iter().map(|col| a.matvec(col)).collect();
        for j in 0..k {
            eigenvalues[j] = dot(&x[j], &ax[j]);
        }

        // Residual R_j = A x_j − λ_j x_j and convergence test.
        let mut residuals: Vec<Vec<f64>> = Vec::with_capacity(k);
        let mut all_converged = true;
        for j in 0..k {
            let mut r = vec![0.0f64; n];
            for i in 0..n {
                r[i] = ax[j][i] - eigenvalues[j] * x[j][i];
            }
            let rnorm = norm(&r);
            // Relative residual measured against the spectral scale of A plus
            // the eigenvalue magnitude (so both large and small eigenvalues are
            // handled). This is the standard ARPACK-style stopping rule.
            let denom = (norm_scale + eigenvalues[j].abs()).max(1e-30);
            if rnorm / denom >= tol {
                all_converged = false;
            }
            residuals.push(r);
        }
        if all_converged {
            converged = true;
            break;
        }

        // Preconditioned residual block W = T R.
        let mut w: Vec<Vec<f64>> = Vec::with_capacity(k);
        for r in &residuals {
            let mut wc = r.clone();
            if let Some(t) = precond {
                for i in 0..n {
                    wc[i] = t[i] * r[i];
                }
            }
            w.push(wc);
        }

        // Assemble the trial subspace S = [X | W | P], tracking the block each
        // column belongs to so the search-direction update can exclude the X
        // contribution.
        let mut s_cols: Vec<Vec<f64>> = Vec::new();
        let mut block_tag: Vec<u8> = Vec::new(); // 0 = X, 1 = W, 2 = P
        for col in &x {
            s_cols.push(col.clone());
            block_tag.push(0);
        }
        for col in &w {
            s_cols.push(col.clone());
            block_tag.push(1);
        }
        for col in &p {
            s_cols.push(col.clone());
            block_tag.push(2);
        }

        // Orthonormalise S, dropping dependent columns. The retained columns
        // keep their original block tags so we can split the Ritz coefficients.
        let kept = orthonormalize_tagged(&mut s_cols, &mut block_tag);
        let m = kept; // subspace dimension after pruning

        // Rayleigh-Ritz: build the small dense projected matrix Sᵀ A S.
        let as_cols: Vec<Vec<f64>> = s_cols.iter().map(|col| a.matvec(col)).collect();
        let mut proj = vec![0.0f64; m * m];
        for i in 0..m {
            for jj in i..m {
                let val = dot(&s_cols[i], &as_cols[jj]);
                proj[i * m + jj] = val;
                proj[jj * m + i] = val;
            }
        }

        // Solve the projected eigenproblem; take the k smallest Ritz pairs.
        let (ritz_vals, ritz_vecs) = jacobi_eigh(&proj, m)?;
        let take = k.min(m);

        // New iterate X = S C_k and new search direction P from the W/P part.
        let mut new_x: Vec<Vec<f64>> = Vec::with_capacity(k);
        let mut new_p: Vec<Vec<f64>> = Vec::with_capacity(k);
        for sel in 0..take {
            // Coefficient column `sel` of the Ritz vectors.
            let mut xcol = vec![0.0f64; n];
            let mut pcol = vec![0.0f64; n];
            for row in 0..m {
                let coeff = ritz_vecs[row * m + sel];
                if coeff == 0.0 {
                    continue;
                }
                let src = &s_cols[row];
                for i in 0..n {
                    xcol[i] += coeff * src[i];
                }
                // The new direction excludes the X block (tag 0).
                if block_tag[row] != 0 {
                    for i in 0..n {
                        pcol[i] += coeff * src[i];
                    }
                }
            }
            new_x.push(xcol);
            new_p.push(pcol);
            eigenvalues[sel] = ritz_vals[sel];
        }
        // If the subspace collapsed below k, pad by carrying over old vectors.
        for old in x.iter().take(k).skip(take) {
            new_x.push(old.clone());
            new_p.push(vec![0.0f64; n]);
        }

        // Re-orthonormalise X for numerical hygiene; normalise P columns.
        orthonormalize(&mut new_x);
        for col in new_p.iter_mut() {
            let nrm = norm(col);
            if nrm > ORTHO_DROP {
                let inv = 1.0 / nrm;
                for v in col.iter_mut() {
                    *v *= inv;
                }
            } else {
                col.iter_mut().for_each(|v| *v = 0.0);
            }
        }

        x = new_x;
        p = new_p;
    }

    // Final Ritz refinement of eigenvalues from the converged X.
    for j in 0..k {
        let axj = a.matvec(&x[j]);
        let num = dot(&x[j], &axj);
        let den = dot(&x[j], &x[j]).max(1e-300);
        eigenvalues[j] = num / den;
    }

    Ok(LobpcgResult {
        eigenvalues,
        eigenvectors: x,
        iters: iters_done,
        converged,
    })
}

// ---------------------------------------------------------------------------
// Linear algebra helpers on column vectors.
// ---------------------------------------------------------------------------

/// Estimates `‖A‖₂` cheaply via the maximum absolute row sum (the infinity
/// norm), which is an upper bound on the spectral norm and is exact for the
/// diagonal of a symmetric M-matrix scale.
fn a_norm_estimate(a: &HostCsr) -> f64 {
    let mut max_sum = 0.0f64;
    for i in 0..a.nrows {
        let start = a.row_ptr[i];
        let end = a.row_ptr[i + 1];
        let mut s = 0.0f64;
        for k in start..end {
            s += a.values[k].abs();
        }
        if s > max_sum {
            max_sum = s;
        }
    }
    max_sum.max(1.0)
}

/// Euclidean inner product.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Euclidean norm.
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Subtracts `scale * src` from `dst` element-wise.
fn subtract_scaled(dst: &mut [f64], scale: f64, src: &[f64]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d -= scale * s;
    }
}

/// Removes the components of `target` along each already-orthonormal column in
/// `prior`, in place (one modified Gram-Schmidt pass).
fn project_out(target: &mut [f64], prior: &[Vec<f64>]) {
    for basis in prior {
        let proj = dot(target, basis);
        subtract_scaled(target, proj, basis);
    }
}

/// Modified Gram-Schmidt orthonormalisation in place. Columns that become
/// numerically dependent are replaced by a deterministic unit vector so the
/// block keeps full rank `k` (important for the initial iterate).
fn orthonormalize(cols: &mut [Vec<f64>]) {
    let k = cols.len();
    if k == 0 {
        return;
    }
    let n = cols[0].len();
    for j in 0..k {
        // Split so that earlier columns are an immutable prior and column j is
        // mutable; j is the first element of the right half.
        let (prior, rest) = cols.split_at_mut(j);
        let target = &mut rest[0];
        project_out(target, prior);

        let nrm = norm(target);
        if nrm > ORTHO_DROP {
            let inv = 1.0 / nrm;
            target.iter_mut().for_each(|v| *v *= inv);
        } else {
            // Replace with a fresh canonical direction and re-orthonormalise.
            target.iter_mut().for_each(|v| *v = 0.0);
            target[j % n] = 1.0;
            project_out(target, prior);
            let nrm2 = norm(target);
            let inv = if nrm2 > ORTHO_DROP { 1.0 / nrm2 } else { 0.0 };
            target.iter_mut().for_each(|v| *v *= inv);
        }
    }
}

/// Modified Gram-Schmidt orthonormalisation that prunes dependent columns,
/// keeping `block_tag` in sync. Returns the number of retained columns; the
/// retained columns occupy the prefix of `cols`/`block_tag` after the call.
fn orthonormalize_tagged(cols: &mut Vec<Vec<f64>>, block_tag: &mut Vec<u8>) -> usize {
    let original = cols.len();
    if original == 0 {
        return 0;
    }
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(original);
    let mut tags: Vec<u8> = Vec::with_capacity(original);

    for (j, col) in cols.iter().enumerate() {
        let mut v = col.clone();
        // Two passes of MGS for stability against the existing basis.
        project_out(&mut v, &basis);
        project_out(&mut v, &basis);
        let nrm = norm(&v);
        if nrm > ORTHO_DROP {
            let inv = 1.0 / nrm;
            v.iter_mut().for_each(|val| *val *= inv);
            basis.push(v);
            tags.push(block_tag[j]);
        }
    }

    let kept = basis.len();
    *cols = basis;
    *block_tag = tags;
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host_csr::{laplacian_1d, laplacian_2d};
    use std::f64::consts::PI;

    /// Analytic eigenvalues of the 1-D Laplacian tridiag(-1, 2, -1) of order n.
    fn laplacian_1d_eigs(n: usize, count: usize) -> Vec<f64> {
        (1..=count)
            .map(|j| 2.0 - 2.0 * ((j as f64) * PI / ((n + 1) as f64)).cos())
            .collect()
    }

    #[test]
    fn smallest_eigs_of_1d_laplacian() {
        let n = 50;
        let a = laplacian_1d(n);
        let res = lobpcg(&a, 3, 300, 1e-7, None).expect("lobpcg");
        let exact = laplacian_1d_eigs(n, 3);
        assert!(res.converged, "LOBPCG did not converge");
        for (j, &ex) in exact.iter().enumerate() {
            assert!(
                (res.eigenvalues[j] - ex).abs() < 1e-6,
                "eig {j}: got {} expected {ex}",
                res.eigenvalues[j],
            );
        }
    }

    #[test]
    fn eigenvectors_a_orthonormal() {
        let n = 40;
        let a = laplacian_1d(n);
        let res = lobpcg(&a, 3, 300, 1e-8, None).expect("lobpcg");
        // x_i^T A x_j ≈ λ_i δ_ij and Rayleigh quotient equals eigenvalue.
        for i in 0..3 {
            let ax = a.matvec(&res.eigenvectors[i]);
            let rq =
                dot(&res.eigenvectors[i], &ax) / dot(&res.eigenvectors[i], &res.eigenvectors[i]);
            assert!(
                (rq - res.eigenvalues[i]).abs() < 1e-6,
                "Rayleigh quotient mismatch"
            );
            for j in 0..3 {
                let cross = dot(&res.eigenvectors[i], &a.matvec(&res.eigenvectors[j]));
                if i == j {
                    assert!((cross - res.eigenvalues[i]).abs() < 1e-5);
                } else {
                    assert!(cross.abs() < 1e-4, "A-orthogonality failed: {cross}");
                }
            }
        }
    }

    #[test]
    fn smallest_eig_2d_laplacian() {
        // 2-D 5-point Laplacian on a g×g grid. Smallest eigenvalue is
        // 4 − 2cos(π/(g+1)) − 2cos(π/(g+1)).
        let g = 8;
        let a = laplacian_2d(g, g);
        let res = lobpcg(&a, 1, 400, 1e-7, None).expect("lobpcg");
        let c = (PI / ((g + 1) as f64)).cos();
        let expected = 4.0 - 2.0 * c - 2.0 * c;
        assert!(
            (res.eigenvalues[0] - expected).abs() < 1e-5,
            "got {} expected {}",
            res.eigenvalues[0],
            expected
        );
    }

    #[test]
    fn diagonal_preconditioner_helps() {
        // A Jacobi preconditioner only accelerates convergence when the
        // diagonal varies; for the constant-diagonal Laplacian it is a mere
        // scalar. We therefore build an SPD matrix with a strongly graded
        // diagonal: D^{1/2} L D^{1/2} where D has entries growing along the
        // index. Jacobi preconditioning then genuinely improves conditioning.
        let n = 50;
        let lap = laplacian_1d(n);
        let scale: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.5).collect();
        // Form the scaled matrix entrywise: a_ij = sqrt(d_i) * l_ij * sqrt(d_j).
        let mut row_ptr = vec![0usize; n + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            let start = lap.row_ptr[i];
            let end = lap.row_ptr[i + 1];
            for kk in start..end {
                let j = lap.col_indices[kk];
                let v = scale[i].sqrt() * lap.values[kk] * scale[j].sqrt();
                col_indices.push(j);
                values.push(v);
            }
            row_ptr[i + 1] = col_indices.len();
        }
        let a = HostCsr::new(n, n, row_ptr, col_indices, values).expect("scaled spd");

        let plain = lobpcg(&a, 2, 500, 1e-7, None).expect("plain");
        let diag = a.diagonal();
        let t: Vec<f64> = diag.iter().map(|&d| 1.0 / d).collect();
        let prec = lobpcg(&a, 2, 500, 1e-7, Some(&t)).expect("prec");
        assert!(
            plain.converged && prec.converged,
            "both should converge: plain={} prec={}",
            plain.converged,
            prec.converged
        );
        assert!(
            prec.iters <= plain.iters,
            "preconditioned iters {} should not exceed plain {}",
            prec.iters,
            plain.iters
        );
    }

    #[test]
    fn rejects_bad_k() {
        let a = laplacian_1d(5);
        assert!(lobpcg(&a, 0, 10, 1e-6, None).is_err());
        assert!(lobpcg(&a, 6, 10, 1e-6, None).is_err());
    }

    #[test]
    fn rejects_bad_precond_length() {
        let a = laplacian_1d(5);
        let t = vec![1.0; 4];
        assert!(lobpcg(&a, 1, 10, 1e-6, Some(&t)).is_err());
    }

    #[test]
    fn single_eigenpair_smallest() {
        let n = 30;
        let a = laplacian_1d(n);
        let res = lobpcg(&a, 1, 300, 1e-8, None).expect("lobpcg");
        let exact = laplacian_1d_eigs(n, 1)[0];
        assert!((res.eigenvalues[0] - exact).abs() < 1e-6);
    }
}
