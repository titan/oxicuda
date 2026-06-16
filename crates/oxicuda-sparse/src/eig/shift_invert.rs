//! Shift-invert power iteration for the eigenvalue nearest a target shift.
//!
//! Ordinary power iteration converges to the eigenvalue of largest magnitude.
//! The **shift-invert** spectral transformation turns the *interior* eigenvalue
//! search into a dominant-eigenvalue search: for a shift `σ` the operator
//!
//! ```text
//!     B = (A − σI)^{-1}
//! ```
//!
//! has eigenvalues `μ_i = 1 / (λ_i − σ)`, so the eigenvalue `λ_i` *closest* to
//! `σ` becomes the one of largest magnitude `|μ_i|`. Power iteration on `B`
//! therefore homes in on the eigenpair nearest `σ` -- precisely the eigenvalue
//! ordinary power iteration would *miss*.
//!
//! Each iteration solves `(A − σI) w = v_k` rather than forming `B` explicitly.
//! The solve uses a single dense LU factorisation of the (small) shifted matrix
//! `A − σI`, computed once and reused across all iterations (the crate carries
//! no general *sparse* direct factorisation; a dense LU on the assembled shifted
//! matrix is the documented fallback and is exact for the small problems this
//! routine targets). From the Rayleigh quotient of the inverted operator,
//!
//! ```text
//!     μ = v_kᵀ B v_k / v_kᵀ v_k = v_kᵀ w   (v_k unit-norm),
//! ```
//!
//! the eigenvalue of `A` is recovered as `λ = σ + 1/μ`. Convergence is declared
//! when the eigenpair residual `‖A v − λ v‖` falls below the tolerance.

use crate::error::{SparseError, SparseResult};
use crate::host_csr::HostCsr;

/// Multiplier of the deterministic LCG used to seed the starting vector,
/// matching the crate's `LcgRng` recipe.
const LCG_MULT: u64 = 6_364_136_223_846_793_005;
/// Increment of the deterministic LCG (see [`LCG_MULT`]).
const LCG_INCR: u64 = 1_442_695_040_888_963_407;
/// Floor below which the Rayleigh quotient `μ` is treated as an underflow
/// (`v` essentially orthogonal to every eigenvector with finite `1/(λ−σ)`).
const MU_FLOOR: f64 = 1e-300;
/// Floor below which an un-normalised iterate is treated as collapsed.
const VEC_FLOOR: f64 = 1e-300;

/// Outcome of a shift-invert solve.
#[derive(Debug, Clone)]
pub struct ShiftInvertResult {
    /// The converged eigenvalue of `A` nearest the requested shift `σ`.
    pub eigenvalue: f64,
    /// The associated unit-norm eigenvector (length `n`).
    pub eigenvector: Vec<f64>,
    /// Number of iterations performed.
    pub iters: usize,
    /// Whether the residual tolerance was met within `max_iter` iterations.
    pub converged: bool,
    /// The final eigenpair residual `‖A v − λ v‖`.
    pub residual: f64,
}

/// Finds the eigenvalue of `a` closest to the shift `sigma` via shift-invert
/// power iteration.
///
/// The matrix `a` must be square. It need not be symmetric, though the Rayleigh
/// quotient recovery is most accurate (and convergence cleanest) for symmetric
/// `a`. The shift `sigma` must not coincide exactly with an eigenvalue, since
/// `A − σI` would then be singular; a shift merely *close* to an eigenvalue is
/// ideal and accelerates convergence.
///
/// # Arguments
///
/// * `a` -- the square host CSR matrix.
/// * `sigma` -- the target shift; the eigenvalue nearest this value is sought.
/// * `max_iter` -- maximum number of power iterations (`≥ 1`).
/// * `tol` -- residual tolerance `‖A v − λ v‖` for convergence.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `a` is not square,
/// [`SparseError::InvalidArgument`] if `a` is empty or `max_iter == 0`,
/// [`SparseError::SingularMatrix`] if `A − σI` is numerically singular (the
/// shift hit an eigenvalue exactly), and [`SparseError::ConvergenceFailure`] if
/// the iterate collapses (e.g. an unreachable spectrum component).
pub fn shift_invert(
    a: &HostCsr,
    sigma: f64,
    max_iter: usize,
    tol: f64,
) -> SparseResult<ShiftInvertResult> {
    if a.nrows != a.ncols {
        return Err(SparseError::DimensionMismatch(format!(
            "shift-invert requires a square matrix, got {}x{}",
            a.nrows, a.ncols
        )));
    }
    let n = a.nrows;
    if n == 0 {
        return Err(SparseError::InvalidArgument(
            "shift-invert requires a non-empty matrix".to_string(),
        ));
    }
    if max_iter == 0 {
        return Err(SparseError::InvalidArgument(
            "shift-invert requires max_iter >= 1".to_string(),
        ));
    }

    // Assemble and factor the shifted operator A − σI once.
    let mut shifted = a.to_dense();
    for i in 0..n {
        shifted[i * n + i] -= sigma;
    }
    let lu = DenseLu::factor(shifted, n)?;

    // Deterministic unit-norm starting vector.
    let mut rng = Lcg::new(0x51f7_1234_abcd_9e01);
    let mut v = vec![0.0f64; n];
    for slot in v.iter_mut() {
        *slot = rng.next_signed();
    }
    if !normalize(&mut v) {
        // Degenerate all-zero draw: fall back to the first canonical axis.
        v.iter_mut().for_each(|x| *x = 0.0);
        v[0] = 1.0;
    }

    let mut eigenvalue = sigma;
    let mut residual = f64::INFINITY;
    let mut converged = false;
    let mut iters = 0usize;
    // `pending` carries the un-normalised next direction `w = B v` from the
    // previous iteration so that the eigenvalue, vector, and residual reported
    // on the iteration we stop are all consistent with the same `v`.
    let mut pending: Option<Vec<f64>> = None;

    for it in 0..max_iter {
        iters = it + 1;

        if let Some(w) = pending.take() {
            let mut next = w;
            if !normalize(&mut next) {
                return Err(SparseError::ConvergenceFailure(
                    "shift-invert iterate collapsed to zero".to_string(),
                ));
            }
            v = next;
        }

        // w = (A − σI)^{-1} v.
        let w = lu.solve(&v);

        // Rayleigh quotient of the inverted operator B at the unit vector v.
        let mu = dot(&v, &w);
        if mu.abs() < MU_FLOOR {
            return Err(SparseError::ConvergenceFailure(
                "shift-invert Rayleigh quotient underflowed".to_string(),
            ));
        }
        eigenvalue = sigma + 1.0 / mu;

        // Eigenpair residual on the *same* v used for the Rayleigh quotient.
        let av = a.matvec(&v);
        residual = av
            .iter()
            .zip(v.iter())
            .map(|(&av_i, &v_i)| {
                let d = av_i - eigenvalue * v_i;
                d * d
            })
            .sum::<f64>()
            .sqrt();

        if residual < tol {
            converged = true;
            break;
        }

        // Guard against a collapsing solve before deferring the normalisation.
        if norm(&w) < VEC_FLOOR {
            return Err(SparseError::ConvergenceFailure(
                "shift-invert solve produced a zero vector".to_string(),
            ));
        }
        pending = Some(w);
    }

    Ok(ShiftInvertResult {
        eigenvalue,
        eigenvector: v,
        iters,
        converged,
        residual,
    })
}

// ---------------------------------------------------------------------------
// Dense LU factorisation (factor once, solve many) used for the shifted solves.
// ---------------------------------------------------------------------------

/// A dense LU factorisation with partial pivoting (`PA = LU`).
///
/// Factored once from the assembled shifted matrix and reused for every
/// shift-invert solve, which is the entire computational point of the
/// transformation: the expensive `O(n³)` factorisation happens a single time
/// while each iteration costs only an `O(n²)` triangular pair of solves.
struct DenseLu {
    /// Matrix order.
    n: usize,
    /// Combined `L\U` factors, row-major; the unit-diagonal of `L` is implicit.
    lu: Vec<f64>,
    /// Row permutation: `piv[i]` is the original row now occupying position `i`.
    piv: Vec<usize>,
}

impl DenseLu {
    /// Factors the row-major `n × n` matrix `a` in place.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::SingularMatrix`] if a pivot is numerically zero.
    fn factor(mut a: Vec<f64>, n: usize) -> SparseResult<Self> {
        let mut piv: Vec<usize> = (0..n).collect();
        for col in 0..n {
            // Partial pivot: largest magnitude entry at or below the diagonal.
            let mut pivot_row = col;
            let mut pivot_mag = a[col * n + col].abs();
            for r in (col + 1)..n {
                let mag = a[r * n + col].abs();
                if mag > pivot_mag {
                    pivot_mag = mag;
                    pivot_row = r;
                }
            }
            if pivot_mag < 1e-300 {
                return Err(SparseError::SingularMatrix);
            }
            if pivot_row != col {
                for c in 0..n {
                    a.swap(col * n + c, pivot_row * n + c);
                }
                piv.swap(col, pivot_row);
            }
            let pivot = a[col * n + col];
            for r in (col + 1)..n {
                let factor = a[r * n + col] / pivot;
                a[r * n + col] = factor; // store the multiplier in L
                if factor != 0.0 {
                    for c in (col + 1)..n {
                        a[r * n + c] -= factor * a[col * n + c];
                    }
                }
            }
        }
        Ok(Self { n, lu: a, piv })
    }

    /// Solves `A x = b` using the stored factors.
    fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.n;
        // Apply the row permutation: x = P b.
        let mut x: Vec<f64> = self.piv.iter().map(|&p| b[p]).collect();
        // Forward substitution with unit-lower L: subtract the dot product of
        // the sub-diagonal row entries with the already-solved prefix of x.
        for i in 0..n {
            let lrow = &self.lu[i * n..i * n + i];
            let dotp: f64 = lrow.iter().zip(x[..i].iter()).map(|(&l, &xj)| l * xj).sum();
            x[i] -= dotp;
        }
        // Backward substitution with upper U.
        for i in (0..n).rev() {
            let urow = &self.lu[i * n + i + 1..i * n + n];
            let dotp: f64 = urow
                .iter()
                .zip(x[i + 1..].iter())
                .map(|(&u, &xj)| u * xj)
                .sum();
            x[i] = (x[i] - dotp) / self.lu[i * n + i];
        }
        x
    }
}

// ---------------------------------------------------------------------------
// Small vector helpers and the deterministic LCG.
// ---------------------------------------------------------------------------

/// Euclidean inner product.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Euclidean norm.
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Normalises `v` to unit length in place, returning `false` if `v` is too
/// small to normalise (left unchanged in that case).
fn normalize(v: &mut [f64]) -> bool {
    let nrm = norm(v);
    if nrm < VEC_FLOOR {
        return false;
    }
    let inv = 1.0 / nrm;
    for x in v.iter_mut() {
        *x *= inv;
    }
    true
}

/// Deterministic linear congruential generator mirroring the crate's `LcgRng`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host_csr::laplacian_1d;
    use std::f64::consts::PI;

    /// Analytic eigenvalues of `tridiag(-1, 2, -1)` of order `n`:
    /// `λ_j = 2 − 2 cos(jπ / (n+1))`, `j = 1..=n`.
    fn laplacian_1d_eigs(n: usize) -> Vec<f64> {
        (1..=n)
            .map(|j| 2.0 - 2.0 * ((j as f64) * PI / ((n + 1) as f64)).cos())
            .collect()
    }

    #[test]
    fn finds_interior_eigenvalue_not_dominant() {
        // n = 5 Laplacian eigenvalues ≈ {0.268, 1.0, 2.0, 3.0, 3.732}.
        let n = 5;
        let a = laplacian_1d(n);
        let eigs = laplacian_1d_eigs(n);
        let dominant = eigs[n - 1];

        // Shift near the interior eigenvalue 2.0 (index 2).
        let res = shift_invert(&a, 1.9, 200, 1e-10).expect("shift-invert");
        assert!(res.converged, "did not converge");
        assert!(
            (res.eigenvalue - eigs[2]).abs() < 1e-7,
            "expected {} got {}",
            eigs[2],
            res.eigenvalue
        );
        // It must NOT have drifted to the dominant eigenvalue.
        assert!(
            (res.eigenvalue - dominant).abs() > 1.0,
            "shift-invert returned the dominant eigenvalue"
        );
        // Residual must be tiny and finite.
        assert!(res.residual < 1e-9);
        assert!(res.eigenvalue.is_finite());
        assert!(res.eigenvector.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn eigenvector_residual_small() {
        let n = 5;
        let a = laplacian_1d(n);
        let res = shift_invert(&a, 0.9, 200, 1e-11).expect("shift-invert");
        // Independently recompute ‖A v − λ v‖.
        let av = a.matvec(&res.eigenvector);
        let r: f64 = av
            .iter()
            .zip(res.eigenvector.iter())
            .map(|(&av_i, &v_i)| {
                let d = av_i - res.eigenvalue * v_i;
                d * d
            })
            .sum::<f64>()
            .sqrt();
        assert!(r < 1e-9, "residual too large: {r}");
        // Unit-norm eigenvector.
        let nrm: f64 = res.eigenvector.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!((nrm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn different_shifts_target_different_eigenvalues() {
        let n = 5;
        let a = laplacian_1d(n);
        let eigs = laplacian_1d_eigs(n);

        // Each shift is placed nearest a distinct eigenvalue.
        let cases = [
            (0.3, eigs[0]), // ≈ 0.268
            (0.9, eigs[1]), // 1.0
            (1.9, eigs[2]), // 2.0
            (2.9, eigs[3]), // 3.0
            (3.8, eigs[4]), // ≈ 3.732
        ];
        for (sigma, expected) in cases {
            let res = shift_invert(&a, sigma, 300, 1e-10).expect("shift-invert");
            assert!(res.converged, "sigma {sigma} did not converge");
            assert!(
                (res.eigenvalue - expected).abs() < 1e-6,
                "sigma {sigma}: expected {expected} got {}",
                res.eigenvalue
            );
        }
    }

    #[test]
    fn finds_smallest_eigenvalue() {
        // Shift below the spectrum targets the smallest eigenvalue, which
        // ordinary power iteration would never find.
        let n = 6;
        let a = laplacian_1d(n);
        let eigs = laplacian_1d_eigs(n);
        let res = shift_invert(&a, -0.5, 300, 1e-10).expect("shift-invert");
        assert!(res.converged);
        assert!(
            (res.eigenvalue - eigs[0]).abs() < 1e-6,
            "expected smallest {} got {}",
            eigs[0],
            res.eigenvalue
        );
    }

    #[test]
    fn converges_within_max_iter() {
        let n = 5;
        let a = laplacian_1d(n);
        let res = shift_invert(&a, 1.9, 50, 1e-9).expect("shift-invert");
        assert!(res.converged);
        assert!(res.iters <= 50);
    }

    #[test]
    fn rejects_non_square() {
        let a = HostCsr::new(2, 3, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]).expect("rect");
        assert!(shift_invert(&a, 0.0, 10, 1e-8).is_err());
    }

    #[test]
    fn rejects_zero_max_iter() {
        let a = laplacian_1d(4);
        assert!(shift_invert(&a, 0.5, 0, 1e-8).is_err());
    }

    #[test]
    fn singular_shift_errors() {
        // Diagonal matrix diag(1, 2, 3); shift exactly on an eigenvalue makes
        // A − σI singular.
        let a =
            HostCsr::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 2.0, 3.0]).expect("diag");
        assert!(shift_invert(&a, 2.0, 50, 1e-8).is_err());
    }

    #[test]
    fn diagonal_matrix_exact() {
        // For a diagonal matrix the eigenvalues are the diagonal entries.
        let a = HostCsr::new(
            4,
            4,
            vec![0, 1, 2, 3, 4],
            vec![0, 1, 2, 3],
            vec![10.0, 20.0, 30.0, 40.0],
        )
        .expect("diag");
        let res = shift_invert(&a, 19.0, 100, 1e-10).expect("shift-invert");
        assert!(res.converged);
        assert!((res.eigenvalue - 20.0).abs() < 1e-8);
    }

    #[test]
    fn dense_lu_solves_correctly() {
        // [[2,1],[1,3]] x = [3,5] -> x = [0.8, 1.4].
        let lu = DenseLu::factor(vec![2.0, 1.0, 1.0, 3.0], 2).expect("factor");
        let x = lu.solve(&[3.0, 5.0]);
        assert!((x[0] - 0.8).abs() < 1e-12);
        assert!((x[1] - 1.4).abs() < 1e-12);
    }
}
