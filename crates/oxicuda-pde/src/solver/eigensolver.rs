//! Symmetric eigensolver via the Lanczos iteration with full reorthogonalisation.
//!
//! Computes a few extreme eigenpairs (smallest or largest) of a symmetric
//! linear operator `A`, supplied either as a closure `x ↦ A x` or as a
//! [`SparseCsr`] matrix. The operator is never formed densely, so the method
//! scales to large sparse discretised Laplacians while only the small
//! Krylov-space tridiagonal projection is diagonalised directly.
//!
//! # Algorithm
//!
//! Starting from a unit vector `q₀`, the Lanczos three-term recurrence builds an
//! orthonormal Krylov basis `Q = [q₀, …, q_{m-1}]` such that `Qᵀ A Q = T` is a
//! symmetric tridiagonal matrix with diagonal `α` and off-diagonal `β`:
//!
//! ```text
//! w        = A q_j
//! α_j      = ⟨w, q_j⟩
//! w        = w − α_j q_j − β_{j-1} q_{j-1}
//! (full reorthogonalisation: w ← w − Σ_i ⟨w, q_i⟩ q_i, twice)
//! β_j      = ‖w‖
//! q_{j+1}  = w / β_j
//! ```
//!
//! Loss of orthogonality is the classical failure mode of finite-precision
//! Lanczos; for the modest problem sizes targeted here we apply **full
//! reorthogonalisation** (modified Gram–Schmidt against every stored basis
//! vector, repeated twice — "twice is enough", Parlett & Scott 1979) which
//! keeps `Q` orthonormal to working precision.
//!
//! The Ritz values (approximate eigenvalues) are the eigenvalues `θ` of `T`,
//! and the Ritz vectors are `y = Q s` where `T s = θ s`. The `m × m`
//! tridiagonal eigenproblem is solved with a cyclic Jacobi rotation sweep,
//! which is robust and yields orthonormal eigenvectors for small symmetric
//! matrices.
//!
//! # References
//!
//! * Lanczos, *An iteration method for the solution of the eigenvalue problem
//!   of linear differential and integral operators*, J. Res. NBS 45 (1950).
//! * Golub & Van Loan, *Matrix Computations*, 4th ed., JHU Press 2013, §10.1.
//! * Parlett, *The Symmetric Eigenvalue Problem*, SIAM 1998.

use crate::error::{PdeError, PdeResult};
use crate::handle::LcgRng;
use crate::solver::sparse::{SparseCsr, dot, norm2};

/// Which end of the spectrum to target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Which {
    /// The algebraically smallest eigenvalues.
    Smallest,
    /// The algebraically largest eigenvalues.
    Largest,
}

/// Configuration for the Lanczos iteration.
#[derive(Debug, Clone)]
pub struct LanczosConfig {
    /// Maximum Krylov dimension (number of Lanczos steps); capped at the
    /// problem dimension `n`.
    pub max_iter: usize,
    /// Number of extreme eigenpairs to return.
    pub n_eigenpairs: usize,
    /// Which end of the spectrum to compute.
    pub which: Which,
    /// Residual tolerance `‖A y − θ y‖` used to flag convergence.
    pub tol: f64,
    /// Lanczos breakdown threshold: if `β_j` falls below this the iteration has
    /// found an invariant subspace and stops.
    pub breakdown_tol: f64,
    /// Whether to apply full reorthogonalisation (recommended; default `true`).
    pub reorthogonalize: bool,
    /// Seed for the deterministic starting vector.
    pub seed: u64,
}

impl Default for LanczosConfig {
    fn default() -> Self {
        Self {
            max_iter: 50,
            n_eigenpairs: 1,
            which: Which::Smallest,
            tol: 1.0e-8,
            breakdown_tol: 1.0e-12,
            reorthogonalize: true,
            seed: 0x5EED_1234,
        }
    }
}

/// A converged (or candidate) eigenpair.
#[derive(Debug, Clone)]
pub struct EigenPair {
    /// Ritz value (approximate eigenvalue).
    pub value: f64,
    /// Ritz vector (approximate eigenvector), unit Euclidean norm.
    pub vector: Vec<f64>,
    /// Residual norm `‖A v − λ v‖`.
    pub residual: f64,
}

/// Result of a Lanczos run.
#[derive(Debug, Clone)]
pub struct LanczosResult {
    /// The requested eigenpairs, ordered by increasing value for
    /// [`Which::Smallest`] and decreasing value for [`Which::Largest`].
    pub pairs: Vec<EigenPair>,
    /// Number of Lanczos steps actually performed (Krylov dimension).
    pub iterations: usize,
    /// `true` when every returned pair has residual below `tol`.
    pub converged: bool,
}

/// Normalise `v` to unit Euclidean norm in place, returning the old norm.
fn normalize(v: &mut [f64]) -> PdeResult<f64> {
    let nrm = norm2(v);
    if nrm <= 1.0e-300 {
        return Err(PdeError::NumericalInstability(
            "lanczos: encountered a zero-norm vector".into(),
        ));
    }
    let inv = 1.0 / nrm;
    for x in v.iter_mut() {
        *x *= inv;
    }
    Ok(nrm)
}

/// Diagonalise a dense symmetric `n × n` matrix with cyclic Jacobi rotations.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors` is row-major
/// `n × n` with column `j` holding the unit eigenvector for `eigenvalues[j]`.
/// Input `a` is consumed.
fn jacobi_symmetric_eig(mut a: Vec<f64>, n: usize) -> PdeResult<(Vec<f64>, Vec<f64>)> {
    if a.len() != n * n {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n * n],
            got: vec![a.len()],
        });
    }
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    if n <= 1 {
        let eigvals = if n == 1 { vec![a[0]] } else { Vec::new() };
        return Ok((eigvals, v));
    }
    let max_sweeps = 100;
    for _sweep in 0..max_sweeps {
        let mut off = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off <= 1.0e-30 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() <= 1.0e-300 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Update the cross terms with rows/columns other than p, q.
                for i in 0..n {
                    if i == p || i == q {
                        continue;
                    }
                    let aip = a[i * n + p];
                    let aiq = a[i * n + q];
                    let nip = c * aip - s * aiq;
                    let niq = s * aip + c * aiq;
                    a[i * n + p] = nip;
                    a[p * n + i] = nip;
                    a[i * n + q] = niq;
                    a[q * n + i] = niq;
                }
                // Update the 2 × 2 pivot block (off-diagonal becomes exactly 0).
                a[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                a[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;
                // Accumulate the rotation into the eigenvector matrix.
                for i in 0..n {
                    let vip = v[i * n + p];
                    let viq = v[i * n + q];
                    v[i * n + p] = c * vip - s * viq;
                    v[i * n + q] = s * vip + c * viq;
                }
            }
        }
    }
    let eigvals: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    Ok((eigvals, v))
}

/// Run the Lanczos iteration on a symmetric operator `apply_op : x ↦ A x`.
///
/// `n` is the dimension of the operator's domain.
///
/// # Errors
///
/// * [`PdeError::InvalidGrid`] if `n == 0`.
/// * [`PdeError::InvalidParameter`] if `n_eigenpairs == 0`.
/// * [`PdeError::DimensionMismatch`] if `apply_op` returns a vector of the wrong
///   length.
/// * [`PdeError::NumericalInstability`] if the starting vector degenerates.
pub fn lanczos<F>(apply_op: F, n: usize, cfg: &LanczosConfig) -> PdeResult<LanczosResult>
where
    F: Fn(&[f64]) -> PdeResult<Vec<f64>>,
{
    if n == 0 {
        return Err(PdeError::InvalidGrid(
            "lanczos: operator dimension n is 0".into(),
        ));
    }
    if cfg.n_eigenpairs == 0 {
        return Err(PdeError::InvalidParameter {
            name: "n_eigenpairs".into(),
            reason: "must be at least 1".into(),
        });
    }
    let m_max = cfg.max_iter.clamp(1, n);

    // Deterministic starting vector with components along (generically) every
    // eigenvector, then normalised.
    let mut rng = LcgRng::new(cfg.seed);
    let mut q_curr: Vec<f64> = (0..n).map(|_| rng.next_range(-1.0, 1.0)).collect();
    normalize(&mut q_curr)?;

    let mut q_vectors: Vec<Vec<f64>> = Vec::with_capacity(m_max);
    let mut alphas: Vec<f64> = Vec::with_capacity(m_max);
    let mut betas: Vec<f64> = Vec::with_capacity(m_max);

    let mut q_prev = vec![0.0_f64; n];
    let mut beta_prev = 0.0_f64;
    let mut used = 0_usize;

    for j in 0..m_max {
        q_vectors.push(q_curr.clone());
        let mut w = apply_op(&q_curr)?;
        if w.len() != n {
            return Err(PdeError::DimensionMismatch { a: w.len(), b: n });
        }
        let alpha = dot(&w, &q_curr)?;
        for i in 0..n {
            w[i] -= alpha * q_curr[i] + beta_prev * q_prev[i];
        }
        if cfg.reorthogonalize {
            for _ in 0..2 {
                for qv in &q_vectors {
                    let proj = dot(&w, qv)?;
                    for i in 0..n {
                        w[i] -= proj * qv[i];
                    }
                }
            }
        }
        let beta = norm2(&w);
        alphas.push(alpha);
        used = j + 1;
        if beta <= cfg.breakdown_tol || j + 1 == m_max {
            break;
        }
        betas.push(beta);
        q_prev.copy_from_slice(&q_curr);
        beta_prev = beta;
        let inv = 1.0 / beta;
        for i in 0..n {
            q_curr[i] = w[i] * inv;
        }
    }

    let m = used;
    // Assemble the dense symmetric tridiagonal projection T (m × m).
    let mut t = vec![0.0_f64; m * m];
    for (i, &a_i) in alphas.iter().enumerate().take(m) {
        t[i * m + i] = a_i;
    }
    for (i, &b_i) in betas.iter().enumerate().take(m.saturating_sub(1)) {
        t[i * m + (i + 1)] = b_i;
        t[(i + 1) * m + i] = b_i;
    }
    let (theta, s) = jacobi_symmetric_eig(t, m)?;

    // Order the Ritz values and pick the requested end of the spectrum.
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| theta[a].total_cmp(&theta[b]));
    let k = cfg.n_eigenpairs.min(m);
    let selected: Vec<usize> = match cfg.which {
        Which::Smallest => order[..k].to_vec(),
        Which::Largest => order[m - k..].iter().rev().copied().collect(),
    };

    let mut pairs = Vec::with_capacity(k);
    let mut all_converged = true;
    for &col in &selected {
        let lambda = theta[col];
        // Ritz vector y = Q · s_col.
        let mut y = vec![0.0_f64; n];
        for (r, qr) in q_vectors.iter().enumerate().take(m) {
            let coeff = s[r * m + col];
            for i in 0..n {
                y[i] += coeff * qr[i];
            }
        }
        normalize(&mut y)?;
        // Residual ‖A y − λ y‖.
        let ay = apply_op(&y)?;
        if ay.len() != n {
            return Err(PdeError::DimensionMismatch { a: ay.len(), b: n });
        }
        let mut res = 0.0_f64;
        for i in 0..n {
            let d = ay[i] - lambda * y[i];
            res += d * d;
        }
        let res = res.sqrt();
        if res > cfg.tol {
            all_converged = false;
        }
        pairs.push(EigenPair {
            value: lambda,
            vector: y,
            residual: res,
        });
    }

    Ok(LanczosResult {
        pairs,
        iterations: m,
        converged: all_converged,
    })
}

/// Convenience wrapper running [`lanczos`] on a [`SparseCsr`] matrix.
///
/// # Errors
///
/// Returns [`PdeError::DimensionMismatch`] if `a` is not square, otherwise the
/// same errors as [`lanczos`].
pub fn lanczos_csr(a: &SparseCsr, cfg: &LanczosConfig) -> PdeResult<LanczosResult> {
    if a.n_rows != a.n_cols {
        return Err(PdeError::DimensionMismatch {
            a: a.n_rows,
            b: a.n_cols,
        });
    }
    lanczos(|x| a.matvec(x), a.n_rows, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PI: f64 = std::f64::consts::PI;

    /// 1-D Dirichlet Laplacian on `m` interior nodes with spacing `h`, stored
    /// as a CSR tridiagonal `(1/h²)·tridiag(-1, 2, -1)`.
    fn laplacian_1d_csr(m: usize, h: f64) -> SparseCsr {
        let ih2 = 1.0 / (h * h);
        let mut row_ptr = vec![0_usize];
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..m {
            if i > 0 {
                cols.push(i - 1);
                vals.push(-ih2);
            }
            cols.push(i);
            vals.push(2.0 * ih2);
            if i + 1 < m {
                cols.push(i + 1);
                vals.push(-ih2);
            }
            row_ptr.push(cols.len());
        }
        SparseCsr::new(m, m, row_ptr, cols, vals).expect("valid csr")
    }

    /// Analytic eigenvalues of the discrete 1-D Dirichlet Laplacian.
    fn laplacian_eigenvalue(k: usize, m: usize, h: f64) -> f64 {
        2.0 * (1.0 - (k as f64 * PI / (m as f64 + 1.0)).cos()) / (h * h)
    }

    #[test]
    fn smallest_eigenvalues_match_analytic() {
        let m = 24;
        let h = 1.0 / (m as f64 + 1.0);
        let a = laplacian_1d_csr(m, h);
        let cfg = LanczosConfig {
            max_iter: m,
            n_eigenpairs: 5,
            which: Which::Smallest,
            tol: 1.0e-6,
            ..LanczosConfig::default()
        };
        let res = lanczos_csr(&a, &cfg).expect("lanczos ok");
        assert!(res.converged, "expected convergence with full Krylov space");
        for (idx, pair) in res.pairs.iter().enumerate() {
            let analytic = laplacian_eigenvalue(idx + 1, m, h);
            let rel = (pair.value - analytic).abs() / analytic;
            assert!(
                rel < 1.0e-4,
                "eig {idx}: got {} analytic {analytic} rel {rel}",
                pair.value
            );
        }
    }

    #[test]
    fn largest_eigenvalue_match_analytic() {
        let m = 20;
        let h = 1.0 / (m as f64 + 1.0);
        let a = laplacian_1d_csr(m, h);
        let cfg = LanczosConfig {
            max_iter: m,
            n_eigenpairs: 2,
            which: Which::Largest,
            tol: 1.0e-6,
            ..LanczosConfig::default()
        };
        let res = lanczos_csr(&a, &cfg).expect("lanczos ok");
        // Largest first.
        let analytic_top = laplacian_eigenvalue(m, m, h);
        let analytic_2nd = laplacian_eigenvalue(m - 1, m, h);
        assert!((res.pairs[0].value - analytic_top).abs() / analytic_top < 1.0e-4);
        assert!((res.pairs[1].value - analytic_2nd).abs() / analytic_2nd < 1.0e-4);
        assert!(res.pairs[0].value > res.pairs[1].value);
    }

    #[test]
    fn ritz_vectors_orthonormal() {
        let m = 20;
        let h = 1.0 / (m as f64 + 1.0);
        let a = laplacian_1d_csr(m, h);
        let cfg = LanczosConfig {
            max_iter: m,
            n_eigenpairs: 4,
            which: Which::Smallest,
            ..LanczosConfig::default()
        };
        let res = lanczos_csr(&a, &cfg).expect("lanczos ok");
        for (i, pi) in res.pairs.iter().enumerate() {
            let nrm = norm2(&pi.vector);
            assert!((nrm - 1.0).abs() < 1.0e-9, "vector {i} norm {nrm}");
            for pj in res.pairs.iter().skip(i + 1) {
                let d = dot(&pi.vector, &pj.vector).expect("dot ok");
                assert!(d.abs() < 1.0e-8, "non-orthogonal pair, dot = {d}");
            }
        }
    }

    #[test]
    fn residuals_below_tol_when_converged() {
        let m = 24;
        let h = 1.0 / (m as f64 + 1.0);
        let a = laplacian_1d_csr(m, h);
        let cfg = LanczosConfig {
            max_iter: m,
            n_eigenpairs: 3,
            which: Which::Smallest,
            tol: 1.0e-6,
            ..LanczosConfig::default()
        };
        let res = lanczos_csr(&a, &cfg).expect("lanczos ok");
        for pair in &res.pairs {
            assert!(
                pair.residual < 1.0e-6,
                "residual {} not below tol",
                pair.residual
            );
            assert!(pair.residual.is_finite());
        }
    }

    #[test]
    fn more_iterations_improve_convergence() {
        // The smallest Ritz value bounds the smallest eigenvalue from above and
        // decreases monotonically as the Krylov dimension grows (Cauchy
        // interlacing): more iterations ⇒ smaller error, and a full-dimension
        // Krylov space recovers the eigenvalue to working precision.
        let m = 40;
        let h = 1.0 / (m as f64 + 1.0);
        let a = laplacian_1d_csr(m, h);
        let analytic = laplacian_eigenvalue(1, m, h);
        let run = |iters: usize| -> f64 {
            let cfg = LanczosConfig {
                max_iter: iters,
                n_eigenpairs: 1,
                which: Which::Smallest,
                tol: 1.0e-12,
                ..LanczosConfig::default()
            };
            let res = lanczos_csr(&a, &cfg).expect("lanczos ok");
            (res.pairs[0].value - analytic).abs()
        };
        let err_few = run(5);
        let err_mid = run(20);
        let err_full = run(m);
        assert!(
            err_mid < err_few,
            "expected better convergence: err(5)={err_few}, err(20)={err_mid}"
        );
        assert!(
            err_full < err_mid,
            "expected better convergence: err(20)={err_mid}, err(40)={err_full}"
        );
        assert!(err_full < 1.0e-6, "full-Krylov err {err_full}");
    }

    #[test]
    fn closure_operator_diagonal() {
        // Diagonal operator A = diag(d): eigenvalues are exactly d_i.
        let d = vec![5.0_f64, 1.0, 3.0, 9.0, 2.0, 7.0];
        let n = d.len();
        let d_clone = d.clone();
        let apply = move |x: &[f64]| -> PdeResult<Vec<f64>> {
            Ok(x.iter().zip(&d_clone).map(|(xi, di)| xi * di).collect())
        };
        let cfg = LanczosConfig {
            max_iter: n,
            n_eigenpairs: 2,
            which: Which::Smallest,
            tol: 1.0e-8,
            ..LanczosConfig::default()
        };
        let res = lanczos(apply, n, &cfg).expect("lanczos ok");
        assert!(
            (res.pairs[0].value - 1.0).abs() < 1.0e-7,
            "{}",
            res.pairs[0].value
        );
        assert!(
            (res.pairs[1].value - 2.0).abs() < 1.0e-7,
            "{}",
            res.pairs[1].value
        );
    }

    #[test]
    fn all_outputs_finite() {
        let m = 16;
        let h = 1.0 / (m as f64 + 1.0);
        let a = laplacian_1d_csr(m, h);
        let cfg = LanczosConfig {
            max_iter: m,
            n_eigenpairs: 4,
            which: Which::Smallest,
            ..LanczosConfig::default()
        };
        let res = lanczos_csr(&a, &cfg).expect("lanczos ok");
        for pair in &res.pairs {
            assert!(pair.value.is_finite());
            assert!(pair.vector.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn errors_on_bad_input() {
        let a = laplacian_1d_csr(8, 0.1);
        let bad = LanczosConfig {
            n_eigenpairs: 0,
            ..LanczosConfig::default()
        };
        assert!(matches!(
            lanczos_csr(&a, &bad),
            Err(PdeError::InvalidParameter { .. })
        ));
        let cfg = LanczosConfig::default();
        assert!(matches!(
            lanczos(|x| Ok(x.to_vec()), 0, &cfg),
            Err(PdeError::InvalidGrid(_))
        ));
        // Non-square CSR.
        let rect = SparseCsr::new(2, 3, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]).expect("ok");
        assert!(matches!(
            lanczos_csr(&rect, &cfg),
            Err(PdeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn jacobi_eig_known_2x2() {
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3.
        let (vals, _vecs) = jacobi_symmetric_eig(vec![2.0, 1.0, 1.0, 2.0], 2).expect("ok");
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        assert!((sorted[0] - 1.0).abs() < 1.0e-12, "{}", sorted[0]);
        assert!((sorted[1] - 3.0).abs() < 1.0e-12, "{}", sorted[1]);
    }
}
