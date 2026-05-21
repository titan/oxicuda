//! Cross-decomposition: Canonical Correlation Analysis (CCA) and Partial Least Squares (PLS).
//!
//! # Algorithms implemented
//!
//! - **CCA** (Borga 2001): SVD of the whitened cross-covariance matrix.
//!   `M = L_X^{-T} C_XY L_Y^{-1}`, then SVD gives canonical directions.
//! - **PLS2** (de Jong 1993 NIPALS): Iterative deflation algorithm finding latent
//!   factors that maximally co-vary between X and Y.
//! - **PLSSVD**: Simplified variant that directly SVD-decomposes the cross-covariance
//!   matrix without deflation.
//!
//! All routines operate on row-major `&[f64]` matrices and return `ManifoldResult`.

use crate::error::{ManifoldError, ManifoldResult};
use crate::linalg::jacobi_eig::{jacobi_eigh, sort_eigen_descending};

// ─────────────────────────────────────────────────────────────────────────────
// Internal numeric helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute `C = A^T B` where A is `(m × p)` and B is `(m × q)`, row-major.
/// Returns `C` of shape `(p × q)` row-major.
fn matmul_atb(a: &[f64], b: &[f64], m: usize, p: usize, q: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; p * q];
    for k in 0..m {
        for i in 0..p {
            let a_ki = a[k * p + i];
            for j in 0..q {
                c[i * q + j] += a_ki * b[k * q + j];
            }
        }
    }
    c
}

/// Compute `C = A B` where A is `(m × k)` and B is `(k × n)`, row-major.
/// Returns `C` of shape `(m × n)`.
fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * n];
    for i in 0..m {
        for r in 0..k {
            let a_ir = a[i * k + r];
            for j in 0..n {
                c[i * n + j] += a_ir * b[r * n + j];
            }
        }
    }
    c
}

/// Euclidean norm of a slice.
#[inline]
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |acc, &x| acc + x * x).sqrt()
}

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |acc, (&ai, &bi)| acc + ai * bi)
}

/// Cholesky factorization of a symmetric positive-definite `n × n` matrix `a`.
/// Returns `L` (lower triangular, row-major) such that `A = L L^T`.
/// Returns `None` if the matrix is not positive-definite.
fn cholesky_l(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Forward substitution: solve `L x = b` where `L` is `n × n` lower-triangular (row-major).
fn triangular_solve_lower(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        let mut acc = b[i];
        for j in 0..i {
            acc -= l[i * n + j] * x[j];
        }
        // l[i*n+i] is guaranteed non-zero because Cholesky succeeded
        x[i] = acc / l[i * n + i];
    }
    x
}

/// Back substitution: solve `L^T x = b` where `L^T` is upper-triangular.
/// `lt` is the lower triangular `L` stored row-major; we interpret `L^T` by swapping indices.
fn triangular_solve_upper(lt: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut acc = b[i];
        for j in (i + 1)..n {
            // L^T[i,j] = L[j,i]
            acc -= lt[j * n + i] * x[j];
        }
        x[i] = acc / lt[i * n + i];
    }
    x
}

/// SVD of a general `m × n` matrix via eigendecomposition of `A^T A`.
///
/// Returns `(U, sigma, V)` where:
/// - `U` is `m × k` (column-orthonormal, row-major)
/// - `sigma` has length `k` (non-negative, descending)
/// - `V` is `n × k` (column-orthonormal, row-major)
///
/// `k = min(m, n)` but we keep only the top-`rank` components.
fn svd_via_ata(
    a: &[f64],
    m: usize,
    n: usize,
    rank: usize,
) -> ManifoldResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let rank = rank.min(m).min(n);
    if rank == 0 {
        return Ok((vec![], vec![], vec![]));
    }
    // Compute A^T A (n × n symmetric)
    let ata = matmul_atb(a, a, m, n, n);
    // Eigendecompose A^T A — eigenvalues are σ²
    let (mut w, mut v_mat) = jacobi_eigh(&ata, n)?;
    sort_eigen_descending(&mut w, &mut v_mat, n);
    // Singular values σ_i = sqrt(max(0, λ_i))
    let sigma: Vec<f64> = (0..rank).map(|i| w[i].max(0.0).sqrt()).collect();
    // V columns are the right singular vectors (n × rank)
    let mut v = vec![0.0_f64; n * rank];
    for j in 0..n {
        for c in 0..rank {
            v[j * rank + c] = v_mat[j * n + c];
        }
    }
    // U = A V / σ  (shape m × rank)
    let av = matmul(a, &v, m, n, rank); // m × rank
    let mut u = vec![0.0_f64; m * rank];
    for c in 0..rank {
        let s = sigma[c];
        if s < 1e-14 {
            // Zero singular value — U column is undefined; set to zero
            for i in 0..m {
                u[i * rank + c] = 0.0;
            }
        } else {
            for i in 0..m {
                u[i * rank + c] = av[i * rank + c] / s;
            }
        }
    }
    Ok((u, sigma, v))
}

/// Compute the standard deviation (with Bessel's correction) of each column of a
/// centered matrix and return the std vector. Stores zero for constant columns.
fn col_std(xc: &[f64], n: usize, p: usize) -> Vec<f64> {
    let denom = (n.saturating_sub(1)).max(1) as f64;
    let mut std = vec![0.0_f64; p];
    for j in 0..p {
        let mut ss = 0.0;
        for i in 0..n {
            let v = xc[i * p + j];
            ss += v * v;
        }
        std[j] = (ss / denom).sqrt();
    }
    std
}

// ─────────────────────────────────────────────────────────────────────────────
// CCA — Canonical Correlation Analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Canonical Correlation Analysis.
#[derive(Debug, Clone)]
pub struct CcaConfig {
    /// Number of canonical components to extract. Capped at `min(n_x, n_y)`.
    pub n_components: usize,
    /// Tikhonov regularization added to both covariance matrices before Cholesky.
    /// A small positive value (e.g. `1e-4`) stabilizes near-singular matrices.
    pub regularization: f64,
}

impl Default for CcaConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            regularization: 1e-4,
        }
    }
}

/// Fitted Canonical Correlation Analysis model.
#[derive(Debug)]
pub struct CcaFit {
    /// X canonical directions `A`, row-major `[n_x × n_components]`.
    /// Column `k` is the k-th canonical weight vector for X.
    pub x_weights: Vec<f64>,
    /// Y canonical directions `B`, row-major `[n_y × n_components]`.
    pub y_weights: Vec<f64>,
    /// X loadings: `X_c^T T / n_samples`, shape `[n_x × n_components]`.
    pub x_loadings: Vec<f64>,
    /// Y loadings: `Y_c^T U / n_samples`, shape `[n_y × n_components]`.
    pub y_loadings: Vec<f64>,
    /// Canonical correlations (σ₁ ≥ σ₂ ≥ … ≥ σ_k), clipped to `[0, 1]`.
    pub correlations: Vec<f64>,
    /// Column means of X (length `n_x`).
    pub x_mean: Vec<f64>,
    /// Column means of Y (length `n_y`).
    pub y_mean: Vec<f64>,
    /// Number of canonical components stored.
    pub n_components: usize,
}

/// Fit Canonical Correlation Analysis on paired datasets X and Y.
///
/// X is `(n_samples × n_x)` and Y is `(n_samples × n_y)`, both row-major.
///
/// # Algorithm (Borga 2001)
///
/// 1. Center X and Y.
/// 2. Compute cross-covariance matrices `C_XX`, `C_YY`, `C_XY`.
/// 3. Regularize diagonals: `C_XX += α I`, `C_YY += α I`.
/// 4. Cholesky-factorize: `C_XX = L_X L_X^T`, `C_YY = L_Y L_Y^T`.
/// 5. Form `M = L_X^{-T} C_XY L_Y^{-1}` and compute its SVD `U Σ V^T`.
/// 6. Canonical directions: `A = L_X^{-1} U`, `B = L_Y^{-1} V`.
pub fn cca_fit(
    x: &[f64],
    n_samples: usize,
    n_x: usize,
    y: &[f64],
    n_y: usize,
    cfg: &CcaConfig,
) -> ManifoldResult<CcaFit> {
    // ── input validation ───────────────────────────────────────────────────
    if n_samples == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if x.len() != n_samples * n_x {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, n_x],
            got: vec![x.len()],
        });
    }
    if y.len() != n_samples * n_y {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, n_y],
            got: vec![y.len()],
        });
    }
    let k = cfg.n_components.min(n_x).min(n_y);
    if k == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: "must be ≥ 1".into(),
        });
    }

    // ── 1. Compute column means ────────────────────────────────────────────
    let mut x_mean = vec![0.0_f64; n_x];
    let mut y_mean = vec![0.0_f64; n_y];
    for i in 0..n_samples {
        for j in 0..n_x {
            x_mean[j] += x[i * n_x + j];
        }
        for j in 0..n_y {
            y_mean[j] += y[i * n_y + j];
        }
    }
    for v in &mut x_mean {
        *v /= n_samples as f64;
    }
    for v in &mut y_mean {
        *v /= n_samples as f64;
    }

    // ── 2. Center the data ─────────────────────────────────────────────────
    let mut xc = x.to_vec();
    let mut yc = y.to_vec();
    for i in 0..n_samples {
        for j in 0..n_x {
            xc[i * n_x + j] -= x_mean[j];
        }
        for j in 0..n_y {
            yc[i * n_y + j] -= y_mean[j];
        }
    }

    // ── 3. Covariance matrices ─────────────────────────────────────────────
    let denom = (n_samples.saturating_sub(1)).max(1) as f64;
    // C_XX = X_c^T X_c / (n-1)  [n_x × n_x]
    let mut c_xx = vec![0.0_f64; n_x * n_x];
    for j in 0..n_x {
        for k_idx in j..n_x {
            let mut acc = 0.0;
            for i in 0..n_samples {
                acc += xc[i * n_x + j] * xc[i * n_x + k_idx];
            }
            let v = acc / denom;
            c_xx[j * n_x + k_idx] = v;
            c_xx[k_idx * n_x + j] = v;
        }
    }
    // C_YY = Y_c^T Y_c / (n-1)  [n_y × n_y]
    let mut c_yy = vec![0.0_f64; n_y * n_y];
    for j in 0..n_y {
        for k_idx in j..n_y {
            let mut acc = 0.0;
            for i in 0..n_samples {
                acc += yc[i * n_y + j] * yc[i * n_y + k_idx];
            }
            let v = acc / denom;
            c_yy[j * n_y + k_idx] = v;
            c_yy[k_idx * n_y + j] = v;
        }
    }
    // C_XY = X_c^T Y_c / (n-1)  [n_x × n_y]
    let mut c_xy = vec![0.0_f64; n_x * n_y];
    for j in 0..n_x {
        for l in 0..n_y {
            let mut acc = 0.0;
            for i in 0..n_samples {
                acc += xc[i * n_x + j] * yc[i * n_y + l];
            }
            c_xy[j * n_y + l] = acc / denom;
        }
    }

    // ── 4. Regularize and Cholesky-factorize ──────────────────────────────
    let alpha = cfg.regularization.max(0.0);
    for j in 0..n_x {
        c_xx[j * n_x + j] += alpha;
    }
    for j in 0..n_y {
        c_yy[j * n_y + j] += alpha;
    }

    let l_x = cholesky_l(&c_xx, n_x).ok_or_else(|| {
        ManifoldError::NumericalInstability(
            "C_XX is not positive-definite; increase regularization".into(),
        )
    })?;
    let l_y = cholesky_l(&c_yy, n_y).ok_or_else(|| {
        ManifoldError::NumericalInstability(
            "C_YY is not positive-definite; increase regularization".into(),
        )
    })?;

    // ── 5. Form M = L_X^{-T} C_XY L_Y^{-1} ──────────────────────────────
    //
    // Step 5a: solve L_X^T M_tmp = C_XY  column by column (M_tmp = L_X^{-T} C_XY)
    // L_X is lower-triangular, so L_X^T is upper-triangular.
    // For each column c of C_XY, solve L_X^T z = C_XY[:,c].
    //
    // C_XY has shape n_x × n_y, stored row-major.
    let mut m_tmp = vec![0.0_f64; n_x * n_y]; // n_x × n_y
    for col in 0..n_y {
        // Extract column `col` of C_XY (length n_x)
        let b: Vec<f64> = (0..n_x).map(|row| c_xy[row * n_y + col]).collect();
        let z = triangular_solve_upper(&l_x, &b, n_x);
        for row in 0..n_x {
            m_tmp[row * n_y + col] = z[row];
        }
    }

    // Step 5b: solve M = M_tmp L_Y^{-1}  row by row
    // Equivalently solve L_Y^T M^T = M_tmp^T, i.e. for each row r of M_tmp,
    // solve L_Y x = M_tmp[r] (lower triangular).
    let mut m_mat = vec![0.0_f64; n_x * n_y]; // n_x × n_y
    for row in 0..n_x {
        let b: Vec<f64> = (0..n_y).map(|col| m_tmp[row * n_y + col]).collect();
        let z = triangular_solve_lower(&l_y, &b, n_y);
        for col in 0..n_y {
            m_mat[row * n_y + col] = z[col];
        }
    }

    // ── 6. SVD of M ───────────────────────────────────────────────────────
    let (u_svd, sigma_raw, v_svd) = svd_via_ata(&m_mat, n_x, n_y, k)?;
    // u_svd: n_x × k, v_svd: n_y × k, sigma: k

    // ── 7. Canonical directions: A = L_X^{-1} U, B = L_Y^{-1} V ─────────
    // Solve L_X a_k = u_k for each column k
    let mut x_weights = vec![0.0_f64; n_x * k];
    for c in 0..k {
        let u_col: Vec<f64> = (0..n_x).map(|r| u_svd[r * k + c]).collect();
        let a = triangular_solve_lower(&l_x, &u_col, n_x);
        for r in 0..n_x {
            x_weights[r * k + c] = a[r];
        }
    }
    // Solve L_Y b_k = v_k for each column k
    let mut y_weights = vec![0.0_f64; n_y * k];
    for c in 0..k {
        let v_col: Vec<f64> = (0..n_y).map(|r| v_svd[r * k + c]).collect();
        let b = triangular_solve_lower(&l_y, &v_col, n_y);
        for r in 0..n_y {
            y_weights[r * k + c] = b[r];
        }
    }

    // ── 8. Canonical variates: T = X_c A, U = Y_c B ──────────────────────
    // T: n_samples × k
    let t_scores = matmul(&xc, &x_weights, n_samples, n_x, k);
    // U: n_samples × k
    let u_scores = matmul(&yc, &y_weights, n_samples, n_y, k);

    // ── 9. Loadings: X_c^T T / n, Y_c^T U / n ────────────────────────────
    let x_loadings_raw = matmul_atb(&xc, &t_scores, n_samples, n_x, k);
    let y_loadings_raw = matmul_atb(&yc, &u_scores, n_samples, n_y, k);
    let n_f = n_samples as f64;
    let x_loadings: Vec<f64> = x_loadings_raw.iter().map(|&v| v / n_f).collect();
    let y_loadings: Vec<f64> = y_loadings_raw.iter().map(|&v| v / n_f).collect();

    // ── 10. Canonical correlations, clipped to [0, 1] ─────────────────────
    let correlations: Vec<f64> = sigma_raw.iter().map(|&s| s.clamp(0.0, 1.0)).collect();

    Ok(CcaFit {
        x_weights,
        y_weights,
        x_loadings,
        y_loadings,
        correlations,
        x_mean,
        y_mean,
        n_components: k,
    })
}

/// Transform new observations through a fitted CCA model.
///
/// Returns `(T, U)` where `T = (X - μ_X) A` and `U = (Y - μ_Y) B`,
/// each of shape `[n_samples × n_components]` row-major.
pub fn cca_transform(
    fit: &CcaFit,
    x: &[f64],
    n_samples: usize,
    y: &[f64],
) -> ManifoldResult<(Vec<f64>, Vec<f64>)> {
    let n_x = fit.x_mean.len();
    let n_y = fit.y_mean.len();
    let k = fit.n_components;
    if n_samples == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if x.len() != n_samples * n_x {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, n_x],
            got: vec![x.len()],
        });
    }
    if y.len() != n_samples * n_y {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, n_y],
            got: vec![y.len()],
        });
    }

    // Center
    let mut xc = x.to_vec();
    let mut yc = y.to_vec();
    for i in 0..n_samples {
        for j in 0..n_x {
            xc[i * n_x + j] -= fit.x_mean[j];
        }
        for j in 0..n_y {
            yc[i * n_y + j] -= fit.y_mean[j];
        }
    }

    let t = matmul(&xc, &fit.x_weights, n_samples, n_x, k);
    let u = matmul(&yc, &fit.y_weights, n_samples, n_y, k);
    Ok((t, u))
}

// ─────────────────────────────────────────────────────────────────────────────
// PLS — Partial Least Squares (NIPALS PLS2)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for PLS (NIPALS PLS2).
#[derive(Debug, Clone)]
pub struct PlsConfig {
    /// Number of latent components to extract.
    pub n_components: usize,
    /// Maximum number of inner NIPALS iterations per component.
    pub max_iter: usize,
    /// Convergence tolerance for the Y-score vector `u`.
    pub tol: f64,
    /// Whether to scale X and Y (divide each column by its standard deviation after centering).
    pub scale: bool,
}

impl Default for PlsConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            max_iter: 500,
            tol: 1e-8,
            scale: true,
        }
    }
}

/// Fitted PLS2 model.
#[derive(Debug)]
pub struct PlsFit {
    /// X weights `W`, row-major `[n_x × n_components]`.
    pub x_weights: Vec<f64>,
    /// Y weights `Q`, row-major `[n_y × n_components]`.
    pub y_weights: Vec<f64>,
    /// X loadings `P`, row-major `[n_x × n_components]`.
    pub x_loadings: Vec<f64>,
    /// Y loadings (regression), row-major `[n_y × n_components]`.
    pub y_loadings: Vec<f64>,
    /// Training X scores `T`, row-major `[n_samples × n_components]`.
    pub x_scores: Vec<f64>,
    /// Training Y scores `U`, row-major `[n_samples × n_components]`.
    pub y_scores: Vec<f64>,
    /// Regression coefficients `B = W (P^T W)^{-1} Q^T`, shape `[n_x × n_y]`.
    pub coef: Vec<f64>,
    /// Column means of X (before scaling), length `n_x`.
    pub x_mean: Vec<f64>,
    /// Column means of Y (before scaling), length `n_y`.
    pub y_mean: Vec<f64>,
    /// Column standard deviations of X used for scaling (1.0 if `scale = false`).
    pub x_std: Vec<f64>,
    /// Column standard deviations of Y used for scaling (1.0 if `scale = false`).
    pub y_std: Vec<f64>,
    /// Number of latent components fitted.
    pub n_components: usize,
    /// Fraction of X variance explained per component (cumulative-free, per-component).
    pub r_sq_x: Vec<f64>,
    /// Fraction of Y variance explained per component.
    pub r_sq_y: Vec<f64>,
}

/// Fit PLS2 (NIPALS) on paired datasets X `[n_samples × n_x]` and Y `[n_samples × n_y]`.
pub fn pls_fit(
    x: &[f64],
    n_samples: usize,
    n_x: usize,
    y: &[f64],
    n_y: usize,
    cfg: &PlsConfig,
) -> ManifoldResult<PlsFit> {
    // ── validation ────────────────────────────────────────────────────────
    if n_samples == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if x.len() != n_samples * n_x {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, n_x],
            got: vec![x.len()],
        });
    }
    if y.len() != n_samples * n_y {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, n_y],
            got: vec![y.len()],
        });
    }
    let k = cfg.n_components.min(n_x).min(n_samples);
    if k == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: "must be ≥ 1".into(),
        });
    }

    // ── 1. Center ─────────────────────────────────────────────────────────
    let mut x_mean = vec![0.0_f64; n_x];
    let mut y_mean = vec![0.0_f64; n_y];
    for i in 0..n_samples {
        for j in 0..n_x {
            x_mean[j] += x[i * n_x + j];
        }
        for j in 0..n_y {
            y_mean[j] += y[i * n_y + j];
        }
    }
    for v in &mut x_mean {
        *v /= n_samples as f64;
    }
    for v in &mut y_mean {
        *v /= n_samples as f64;
    }

    let mut xr = x.to_vec(); // working X residual
    let mut yr = y.to_vec(); // working Y residual
    for i in 0..n_samples {
        for j in 0..n_x {
            xr[i * n_x + j] -= x_mean[j];
        }
        for j in 0..n_y {
            yr[i * n_y + j] -= y_mean[j];
        }
    }

    // ── 2. Scale ──────────────────────────────────────────────────────────
    let x_std = if cfg.scale {
        let mut s = col_std(&xr, n_samples, n_x);
        for v in &mut s {
            if *v < 1e-14 {
                *v = 1.0;
            }
        }
        s
    } else {
        vec![1.0_f64; n_x]
    };
    let y_std = if cfg.scale {
        let mut s = col_std(&yr, n_samples, n_y);
        for v in &mut s {
            if *v < 1e-14 {
                *v = 1.0;
            }
        }
        s
    } else {
        vec![1.0_f64; n_y]
    };
    if cfg.scale {
        for i in 0..n_samples {
            for j in 0..n_x {
                xr[i * n_x + j] /= x_std[j];
            }
            for j in 0..n_y {
                yr[i * n_y + j] /= y_std[j];
            }
        }
    }

    // ── 3. Total variance for R² ──────────────────────────────────────────
    let total_var_x: f64 = xr.iter().map(|&v| v * v).sum();
    let total_var_y: f64 = yr.iter().map(|&v| v * v).sum();

    // ── 4. NIPALS PLS2 ────────────────────────────────────────────────────
    let mut w_mat = vec![0.0_f64; n_x * k]; // X weights W
    let mut q_mat = vec![0.0_f64; n_y * k]; // Y weights Q
    let mut p_mat = vec![0.0_f64; n_x * k]; // X loadings P
    let mut y_load_mat = vec![0.0_f64; n_y * k]; // Y loadings
    let mut t_mat = vec![0.0_f64; n_samples * k]; // X scores T
    let mut u_mat = vec![0.0_f64; n_samples * k]; // Y scores U
    let mut r_sq_x = vec![0.0_f64; k];
    let mut r_sq_y = vec![0.0_f64; k];

    for comp in 0..k {
        // Initialize u_score = first column of yr
        let mut u_score: Vec<f64> = (0..n_samples).map(|i| yr[i * n_y]).collect();

        let mut w_vec = vec![0.0_f64; n_x];
        let mut t_score = vec![0.0_f64; n_samples];
        let mut q_vec = vec![0.0_f64; n_y];

        let mut converged = false;
        for _iter in 0..cfg.max_iter {
            // a. w = X^T u / ||X^T u||
            for j in 0..n_x {
                let mut acc = 0.0;
                for i in 0..n_samples {
                    acc += xr[i * n_x + j] * u_score[i];
                }
                w_vec[j] = acc;
            }
            let w_norm = l2_norm(&w_vec);
            if w_norm < 1e-14 {
                break;
            }
            for v in &mut w_vec {
                *v /= w_norm;
            }

            // b. t = X w
            for i in 0..n_samples {
                let mut acc = 0.0;
                for j in 0..n_x {
                    acc += xr[i * n_x + j] * w_vec[j];
                }
                t_score[i] = acc;
            }

            // c. q = Y^T t / ||Y^T t||
            for j in 0..n_y {
                let mut acc = 0.0;
                for i in 0..n_samples {
                    acc += yr[i * n_y + j] * t_score[i];
                }
                q_vec[j] = acc;
            }
            let q_norm = l2_norm(&q_vec);
            if q_norm < 1e-14 {
                break;
            }
            for v in &mut q_vec {
                *v /= q_norm;
            }

            // d. u_new = Y q
            let mut u_new = vec![0.0_f64; n_samples];
            for i in 0..n_samples {
                let mut acc = 0.0;
                for j in 0..n_y {
                    acc += yr[i * n_y + j] * q_vec[j];
                }
                u_new[i] = acc;
            }

            // e. convergence check
            let diff: f64 = u_new
                .iter()
                .zip(u_score.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            u_score = u_new;
            if diff < cfg.tol {
                converged = true;
                break;
            }
        }
        if !converged {
            // Non-fatal: continue with unconverged estimate (warn silently by design)
        }

        // Store X scores and Y scores for this component
        for i in 0..n_samples {
            t_mat[i * k + comp] = t_score[i];
            u_mat[i * k + comp] = u_score[i];
        }

        // X loadings: p = X^T t / (t^T t)
        let t_sq: f64 = dot(&t_score, &t_score);
        let mut p_vec = vec![0.0_f64; n_x];
        if t_sq > 1e-28 {
            for j in 0..n_x {
                let mut acc = 0.0;
                for i in 0..n_samples {
                    acc += xr[i * n_x + j] * t_score[i];
                }
                p_vec[j] = acc / t_sq;
            }
        }

        // Y loadings (regression): g = Y^T t / (t^T t)
        let mut y_load_vec = vec![0.0_f64; n_y];
        if t_sq > 1e-28 {
            for j in 0..n_y {
                let mut acc = 0.0;
                for i in 0..n_samples {
                    acc += yr[i * n_y + j] * t_score[i];
                }
                y_load_vec[j] = acc / t_sq;
            }
        }

        // R² explained variance per component
        let t_norm = l2_norm(&t_score);
        let p_norm = l2_norm(&p_vec);
        let var_x_explained = (t_norm * p_norm).powi(2);
        r_sq_x[comp] = if total_var_x > 1e-28 {
            (var_x_explained / total_var_x).min(1.0)
        } else {
            0.0
        };
        let u_norm = l2_norm(&u_score);
        let q_norm_final = l2_norm(&q_vec);
        let var_y_explained = (u_norm * q_norm_final).powi(2);
        r_sq_y[comp] = if total_var_y > 1e-28 {
            (var_y_explained / total_var_y).min(1.0)
        } else {
            0.0
        };

        // Store weights and loadings
        for j in 0..n_x {
            w_mat[j * k + comp] = w_vec[j];
            p_mat[j * k + comp] = p_vec[j];
        }
        for j in 0..n_y {
            q_mat[j * k + comp] = q_vec[j];
            y_load_mat[j * k + comp] = y_load_vec[j];
        }

        // ── Deflation ────────────────────────────────────────────────────
        // X ← X - t p^T
        for i in 0..n_samples {
            for j in 0..n_x {
                xr[i * n_x + j] -= t_score[i] * p_vec[j];
            }
        }
        // Y ← Y - t q^T  (simplified PLS2 inner-relation deflation)
        for i in 0..n_samples {
            for j in 0..n_y {
                yr[i * n_y + j] -= t_score[i] * y_load_vec[j];
            }
        }
    }

    // ── 5. Regression coefficients: B = W (P^T W)^{-1} Q^T ───────────────
    //
    // Compute P^T W (k × k matrix)
    // P and W both stored as [n_x × k] row-major
    let mut ptw = vec![0.0_f64; k * k]; // P^T W
    for i in 0..k {
        for j in 0..k {
            let mut acc = 0.0;
            for r in 0..n_x {
                acc += p_mat[r * k + i] * w_mat[r * k + j];
            }
            ptw[i * k + j] = acc;
        }
    }
    // Invert (P^T W) — for small k, use Gauss-Jordan elimination
    let ptw_inv = invert_small_matrix(&ptw, k)?;
    // W (P^T W)^{-1}  — shape n_x × k
    let w_ptw_inv = matmul(&w_mat, &ptw_inv, n_x, k, k);
    // B = W (P^T W)^{-1} Q^T  — shape n_x × n_y
    // Q is [n_y × k], so Q^T is [k × n_y]
    // w_ptw_inv [n_x × k] × Q^T [k × n_y]
    let mut q_t = vec![0.0_f64; k * n_y];
    for j in 0..n_y {
        for c in 0..k {
            q_t[c * n_y + j] = q_mat[j * k + c];
        }
    }
    let coef_scaled = matmul(&w_ptw_inv, &q_t, n_x, k, n_y);

    // If scaling was applied, adjust coefficients back to original space:
    // B_orig[i,j] = B_scaled[i,j] * y_std[j] / x_std[i]
    let mut coef = coef_scaled;
    if cfg.scale {
        for i in 0..n_x {
            for j in 0..n_y {
                coef[i * n_y + j] *= y_std[j] / x_std[i];
            }
        }
    }

    Ok(PlsFit {
        x_weights: w_mat,
        y_weights: q_mat,
        x_loadings: p_mat,
        y_loadings: y_load_mat,
        x_scores: t_mat,
        y_scores: u_mat,
        coef,
        x_mean,
        y_mean,
        x_std,
        y_std,
        n_components: k,
        r_sq_x,
        r_sq_y,
    })
}

/// Predict Y from new X observations using a fitted PLS model.
///
/// Returns predicted Y of shape `[n_new × n_y]` row-major (in the original Y scale).
pub fn pls_predict(fit: &PlsFit, x_new: &[f64], n_new: usize) -> ManifoldResult<Vec<f64>> {
    let n_x = fit.x_mean.len();
    let n_y = fit.y_mean.len();
    if n_new == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if x_new.len() != n_new * n_x {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_new, n_x],
            got: vec![x_new.len()],
        });
    }
    // Center using training means
    let mut xc = x_new.to_vec();
    for i in 0..n_new {
        for j in 0..n_x {
            xc[i * n_x + j] -= fit.x_mean[j];
        }
    }
    // Y_pred = X_c B + μ_Y  where B is coef [n_x × n_y]
    let yp = matmul(&xc, &fit.coef, n_new, n_x, n_y);
    let mut y_pred = yp;
    for i in 0..n_new {
        for j in 0..n_y {
            y_pred[i * n_y + j] += fit.y_mean[j];
        }
    }
    Ok(y_pred)
}

/// Transform new X observations to PLS latent space.
///
/// Returns X scores of shape `[n_new × n_components]` row-major.
pub fn pls_transform(fit: &PlsFit, x_new: &[f64], n_new: usize) -> ManifoldResult<Vec<f64>> {
    let n_x = fit.x_mean.len();
    let k = fit.n_components;
    if n_new == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if x_new.len() != n_new * n_x {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_new, n_x],
            got: vec![x_new.len()],
        });
    }
    let mut xc = x_new.to_vec();
    for i in 0..n_new {
        for j in 0..n_x {
            xc[i * n_x + j] -= fit.x_mean[j];
        }
    }
    if fit.x_std.iter().any(|&s| (s - 1.0).abs() > 1e-14) {
        for i in 0..n_new {
            for j in 0..n_x {
                xc[i * n_x + j] /= fit.x_std[j];
            }
        }
    }
    let scores = matmul(&xc, &fit.x_weights, n_new, n_x, k);
    Ok(scores)
}

/// Invert a small `k × k` matrix via Gauss-Jordan elimination (row-major).
fn invert_small_matrix(a: &[f64], k: usize) -> ManifoldResult<Vec<f64>> {
    if k == 0 {
        return Ok(vec![]);
    }
    // Augmented matrix [A | I]
    let mut m = vec![0.0_f64; k * (2 * k)];
    for i in 0..k {
        for j in 0..k {
            m[i * (2 * k) + j] = a[i * k + j];
        }
        m[i * (2 * k) + k + i] = 1.0;
    }
    for col in 0..k {
        // Partial pivot
        let mut max_row = col;
        let mut max_val = m[col * (2 * k) + col].abs();
        for row in (col + 1)..k {
            let v = m[row * (2 * k) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(ManifoldError::SingularMatrix(
                "P^T W is singular; reduce n_components".into(),
            ));
        }
        if max_row != col {
            for j in 0..(2 * k) {
                m.swap(col * (2 * k) + j, max_row * (2 * k) + j);
            }
        }
        let pivot = m[col * (2 * k) + col];
        for j in 0..(2 * k) {
            m[col * (2 * k) + j] /= pivot;
        }
        for row in 0..k {
            if row == col {
                continue;
            }
            let factor = m[row * (2 * k) + col];
            for j in 0..(2 * k) {
                let delta = factor * m[col * (2 * k) + j];
                m[row * (2 * k) + j] -= delta;
            }
        }
    }
    // Extract right half
    let mut inv = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..k {
            inv[i * k + j] = m[i * (2 * k) + k + j];
        }
    }
    Ok(inv)
}

// ─────────────────────────────────────────────────────────────────────────────
// PLSSVD — PLS via SVD of the cross-covariance matrix
// ─────────────────────────────────────────────────────────────────────────────

/// Fitted PLSSVD model.
#[derive(Debug)]
pub struct PlsSvdFit {
    /// X weights `U`, row-major `[n_x × n_components]` (left singular vectors of C_XY).
    pub x_weights: Vec<f64>,
    /// Y weights `V`, row-major `[n_y × n_components]` (right singular vectors of C_XY).
    pub y_weights: Vec<f64>,
    /// Singular values of the cross-covariance matrix (descending).
    pub singular_values: Vec<f64>,
    /// Column means of X.
    pub x_mean: Vec<f64>,
    /// Column means of Y.
    pub y_mean: Vec<f64>,
}

/// Fit PLSSVD: decompose the cross-covariance matrix `C_XY` via SVD.
///
/// Returns up to `n_components` latent directions without deflation.
/// Useful for two-view feature extraction (not for regression).
pub fn pls_svd_fit(
    x: &[f64],
    n_samples: usize,
    n_x: usize,
    y: &[f64],
    n_y: usize,
    n_components: usize,
) -> ManifoldResult<PlsSvdFit> {
    if n_samples == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if x.len() != n_samples * n_x {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, n_x],
            got: vec![x.len()],
        });
    }
    if y.len() != n_samples * n_y {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, n_y],
            got: vec![y.len()],
        });
    }
    let k = n_components.min(n_x).min(n_y);
    if k == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: "must be ≥ 1".into(),
        });
    }

    // Center
    let mut x_mean = vec![0.0_f64; n_x];
    let mut y_mean = vec![0.0_f64; n_y];
    for i in 0..n_samples {
        for j in 0..n_x {
            x_mean[j] += x[i * n_x + j];
        }
        for j in 0..n_y {
            y_mean[j] += y[i * n_y + j];
        }
    }
    for v in &mut x_mean {
        *v /= n_samples as f64;
    }
    for v in &mut y_mean {
        *v /= n_samples as f64;
    }

    let mut xc = x.to_vec();
    let mut yc = y.to_vec();
    for i in 0..n_samples {
        for j in 0..n_x {
            xc[i * n_x + j] -= x_mean[j];
        }
        for j in 0..n_y {
            yc[i * n_y + j] -= y_mean[j];
        }
    }

    // C_XY = X_c^T Y_c / (n-1)  [n_x × n_y]
    let denom = (n_samples.saturating_sub(1)).max(1) as f64;
    let mut c_xy = vec![0.0_f64; n_x * n_y];
    for j in 0..n_x {
        for l in 0..n_y {
            let mut acc = 0.0;
            for i in 0..n_samples {
                acc += xc[i * n_x + j] * yc[i * n_y + l];
            }
            c_xy[j * n_y + l] = acc / denom;
        }
    }

    // SVD of C_XY
    let (u_svd, sigma, v_svd) = svd_via_ata(&c_xy, n_x, n_y, k)?;
    // u_svd: n_x × k, v_svd: n_y × k

    Ok(PlsSvdFit {
        x_weights: u_svd,
        y_weights: v_svd,
        singular_values: sigma,
        x_mean,
        y_mean,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Generate a random matrix of shape (rows × cols) with entries in [-1, 1].
    fn rand_matrix(rng: &mut LcgRng, rows: usize, cols: usize) -> Vec<f64> {
        (0..rows * cols)
            .map(|_| rng.next_range(-1.0, 1.0))
            .collect()
    }

    /// Root mean square error between two equal-length slices.
    fn rmse(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let ss: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).powi(2))
            .sum();
        (ss / n).sqrt()
    }

    // ── Test 1: cca_empty_error ───────────────────────────────────────────
    #[test]
    fn cca_empty_error() {
        let cfg = CcaConfig::default();
        let result = cca_fit(&[], 0, 3, &[], 2, &cfg);
        assert!(matches!(result, Err(ManifoldError::EmptyInput)));
    }

    // ── Test 2: cca_output_shape ─────────────────────────────────────────
    #[test]
    fn cca_output_shape() {
        let mut rng = LcgRng::new(42);
        let n = 30;
        let p = 5;
        let q = 4;
        let k = 3;
        let x = rand_matrix(&mut rng, n, p);
        let y = rand_matrix(&mut rng, n, q);
        let cfg = CcaConfig {
            n_components: k,
            regularization: 1e-3,
        };
        let fit = cca_fit(&x, n, p, &y, q, &cfg).expect("fit");
        assert_eq!(fit.x_weights.len(), p * k, "x_weights shape");
        assert_eq!(fit.y_weights.len(), q * k, "y_weights shape");
        assert_eq!(fit.correlations.len(), k, "correlations length");
        assert_eq!(fit.n_components, k);
    }

    // ── Test 3: cca_max_correlation — X = Y gives corr ≈ 1 ──────────────
    #[test]
    fn cca_max_correlation() {
        let mut rng = LcgRng::new(7);
        let n = 20;
        let p = 4;
        // Generate X, set Y = X (identical views)
        let x = rand_matrix(&mut rng, n, p);
        let y = x.clone();
        let cfg = CcaConfig {
            n_components: 1,
            regularization: 1e-4,
        };
        let fit = cca_fit(&x, n, p, &y, p, &cfg).expect("fit");
        // First canonical correlation should be ≈ 1
        assert!(
            (fit.correlations[0] - 1.0).abs() < 1e-4,
            "first corr = {} (expected ≈ 1)",
            fit.correlations[0]
        );
    }

    // ── Test 4: cca_correlations_sorted ──────────────────────────────────
    #[test]
    fn cca_correlations_sorted() {
        let mut rng = LcgRng::new(13);
        let n = 40;
        let p = 5;
        let q = 4;
        let k = 3;
        let x = rand_matrix(&mut rng, n, p);
        let y = rand_matrix(&mut rng, n, q);
        let cfg = CcaConfig {
            n_components: k,
            regularization: 1e-3,
        };
        let fit = cca_fit(&x, n, p, &y, q, &cfg).expect("fit");
        for i in 1..k {
            assert!(
                fit.correlations[i - 1] >= fit.correlations[i] - 1e-10,
                "correlations not sorted: {} < {}",
                fit.correlations[i - 1],
                fit.correlations[i]
            );
        }
    }

    // ── Test 5: cca_correlations_range ───────────────────────────────────
    #[test]
    fn cca_correlations_range() {
        let mut rng = LcgRng::new(99);
        let n = 25;
        let p = 3;
        let q = 3;
        let k = 2;
        let x = rand_matrix(&mut rng, n, p);
        let y = rand_matrix(&mut rng, n, q);
        let cfg = CcaConfig {
            n_components: k,
            regularization: 1e-3,
        };
        let fit = cca_fit(&x, n, p, &y, q, &cfg).expect("fit");
        for &c in &fit.correlations {
            assert!(
                (0.0..=1.0 + 1e-12).contains(&c),
                "correlation out of range: {c}"
            );
        }
    }

    // ── Test 6: cca_transform_shape ───────────────────────────────────────
    #[test]
    fn cca_transform_shape() {
        let mut rng = LcgRng::new(55);
        let n = 20;
        let p = 4;
        let q = 3;
        let k = 2;
        let x = rand_matrix(&mut rng, n, p);
        let y = rand_matrix(&mut rng, n, q);
        let cfg = CcaConfig {
            n_components: k,
            regularization: 1e-3,
        };
        let fit = cca_fit(&x, n, p, &y, q, &cfg).expect("fit");
        let (t, u) = cca_transform(&fit, &x, n, &y).expect("transform");
        assert_eq!(t.len(), n * k, "T shape");
        assert_eq!(u.len(), n * k, "U shape");
    }

    // ── Test 7: cca_orthogonal_variates ──────────────────────────────────
    #[test]
    fn cca_orthogonal_variates() {
        let mut rng = LcgRng::new(21);
        let n = 50;
        let p = 5;
        let q = 4;
        let k = 3;
        let x = rand_matrix(&mut rng, n, p);
        let y = rand_matrix(&mut rng, n, q);
        let cfg = CcaConfig {
            n_components: k,
            regularization: 1e-3,
        };
        let fit = cca_fit(&x, n, p, &y, q, &cfg).expect("fit");
        let (t, _u) = cca_transform(&fit, &x, n, &y).expect("transform");
        // T^T T should be approximately diagonal (orthogonal canonical variates)
        for i in 0..k {
            for j in (i + 1)..k {
                let t_i: Vec<f64> = (0..n).map(|r| t[r * k + i]).collect();
                let t_j: Vec<f64> = (0..n).map(|r| t[r * k + j]).collect();
                let inner = dot(&t_i, &t_j);
                let nrm_i = l2_norm(&t_i);
                let nrm_j = l2_norm(&t_j);
                let cos = inner / (nrm_i * nrm_j + 1e-14);
                assert!(
                    cos.abs() < 0.15,
                    "canonical variates {i} and {j} not orthogonal: cos = {cos}"
                );
            }
        }
    }

    // ── Test 8: pls_output_shape ──────────────────────────────────────────
    #[test]
    fn pls_output_shape() {
        let mut rng = LcgRng::new(11);
        let n = 30;
        let p = 5;
        let q = 3;
        let k = 2;
        let x = rand_matrix(&mut rng, n, p);
        let y = rand_matrix(&mut rng, n, q);
        let cfg = PlsConfig {
            n_components: k,
            ..Default::default()
        };
        let fit = pls_fit(&x, n, p, &y, q, &cfg).expect("fit");
        assert_eq!(fit.x_weights.len(), p * k, "W shape");
        assert_eq!(fit.y_weights.len(), q * k, "Q shape");
        assert_eq!(fit.x_loadings.len(), p * k, "P shape");
        assert_eq!(fit.coef.len(), p * q, "coef shape");
        assert_eq!(fit.x_scores.len(), n * k, "T shape");
        assert_eq!(fit.y_scores.len(), n * k, "U shape");
        assert_eq!(fit.n_components, k);
    }

    // ── Test 9: pls_predict_shape ─────────────────────────────────────────
    #[test]
    fn pls_predict_shape() {
        let mut rng = LcgRng::new(33);
        let n = 20;
        let p = 4;
        let q = 2;
        let x = rand_matrix(&mut rng, n, p);
        let y = rand_matrix(&mut rng, n, q);
        let x_new = rand_matrix(&mut rng, 5, p);
        let cfg = PlsConfig {
            n_components: 2,
            ..Default::default()
        };
        let fit = pls_fit(&x, n, p, &y, q, &cfg).expect("fit");
        let y_pred = pls_predict(&fit, &x_new, 5).expect("predict");
        assert_eq!(y_pred.len(), 5 * q, "predicted Y shape");
    }

    // ── Test 10: pls_fit_r_sq_range ──────────────────────────────────────
    #[test]
    fn pls_fit_r_sq_range() {
        let mut rng = LcgRng::new(77);
        let n = 30;
        let p = 4;
        let q = 2;
        let x = rand_matrix(&mut rng, n, p);
        let y = rand_matrix(&mut rng, n, q);
        let cfg = PlsConfig {
            n_components: 2,
            ..Default::default()
        };
        let fit = pls_fit(&x, n, p, &y, q, &cfg).expect("fit");
        for &r in &fit.r_sq_x {
            assert!((0.0..=1.0 + 1e-10).contains(&r), "r_sq_x out of range: {r}");
        }
        for &r in &fit.r_sq_y {
            assert!((0.0..=1.0 + 1e-10).contains(&r), "r_sq_y out of range: {r}");
        }
    }

    // ── Test 11: pls_predict_recovery — linear Y=XB ──────────────────────
    #[test]
    fn pls_predict_recovery() {
        let mut rng = LcgRng::new(123);
        let n = 80;
        let p = 4;
        let q = 3;
        // Construct Y = X[:, 0:2] @ B_true + small noise
        let x: Vec<f64> = (0..n * p).map(|_| rng.next_normal()).collect();
        // B_true: 2×3
        let b_true = [1.5_f64, -0.5, 0.8, -1.0, 2.0, 0.3];
        let mut y = vec![0.0_f64; n * q];
        for i in 0..n {
            for j in 0..q {
                // Use first 2 columns of X
                y[i * q + j] =
                    x[i * p] * b_true[j] + x[i * p + 1] * b_true[3 + j] + rng.next_normal() * 0.05;
            }
        }
        let cfg = PlsConfig {
            n_components: 2,
            scale: true,
            ..Default::default()
        };
        let fit = pls_fit(&x, n, p, &y, q, &cfg).expect("fit");
        let y_pred = pls_predict(&fit, &x, n).expect("predict");
        let err = rmse(&y_pred, &y);
        assert!(err < 0.5, "RMSE = {err:.4} (expected < 0.5)");
    }

    // ── Test 12: pls_transform_shape ──────────────────────────────────────
    #[test]
    fn pls_transform_shape() {
        let mut rng = LcgRng::new(44);
        let n = 20;
        let p = 4;
        let q = 2;
        let k = 2;
        let x = rand_matrix(&mut rng, n, p);
        let y = rand_matrix(&mut rng, n, q);
        let cfg = PlsConfig {
            n_components: k,
            ..Default::default()
        };
        let fit = pls_fit(&x, n, p, &y, q, &cfg).expect("fit");
        let scores = pls_transform(&fit, &x, n).expect("transform");
        assert_eq!(scores.len(), n * k, "X scores shape");
    }

    // ── Test 13: pls_1_component_simple ──────────────────────────────────
    #[test]
    fn pls_1_component_simple() {
        // X has a dominant direction; Y is linearly related to that direction.
        let mut rng = LcgRng::new(88);
        let n = 60;
        let p = 5;
        let q = 2;
        // Latent variable t ~ N(0,1), shape (n,)
        let t_latent: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
        // X = t * w_x + noise,  Y = t * w_y + noise
        let w_x = [1.0_f64, 0.5, -0.3, 0.2, 0.8];
        let w_y = [1.2_f64, -0.7];
        let mut x = vec![0.0_f64; n * p];
        let mut y = vec![0.0_f64; n * q];
        for i in 0..n {
            for j in 0..p {
                x[i * p + j] = t_latent[i] * w_x[j] + rng.next_normal() * 0.1;
            }
            for j in 0..q {
                y[i * q + j] = t_latent[i] * w_y[j] + rng.next_normal() * 0.1;
            }
        }
        let cfg = PlsConfig {
            n_components: 1,
            ..Default::default()
        };
        let fit = pls_fit(&x, n, p, &y, q, &cfg).expect("fit");
        // The X score should correlate highly with t_latent
        let t_score: Vec<f64> = (0..n).map(|i| fit.x_scores[i]).collect();
        let corr = pearson_corr(&t_score, &t_latent);
        assert!(
            corr.abs() > 0.90,
            "PLS 1-component score should correlate with latent variable, got r={corr:.4}"
        );
    }

    // ── Test 14: pls_svd_singular_values_positive ─────────────────────────
    #[test]
    fn pls_svd_singular_values_positive() {
        let mut rng = LcgRng::new(66);
        let n = 30;
        let p = 4;
        let q = 3;
        let k = 2;
        let x = rand_matrix(&mut rng, n, p);
        let y = rand_matrix(&mut rng, n, q);
        let fit = pls_svd_fit(&x, n, p, &y, q, k).expect("fit");
        for &sv in &fit.singular_values {
            assert!(sv >= 0.0, "singular value negative: {sv}");
        }
        // At least one positive (non-trivial data)
        assert!(fit.singular_values.iter().any(|&sv| sv > 1e-6));
    }

    // ── Test 15: pls_svd_output_shape ────────────────────────────────────
    #[test]
    fn pls_svd_output_shape() {
        let mut rng = LcgRng::new(200);
        let n = 25;
        let p = 4;
        let q = 3;
        let k = 2;
        let x = rand_matrix(&mut rng, n, p);
        let y = rand_matrix(&mut rng, n, q);
        let fit = pls_svd_fit(&x, n, p, &y, q, k).expect("fit");
        assert_eq!(fit.x_weights.len(), p * k, "x_weights shape");
        assert_eq!(fit.y_weights.len(), q * k, "y_weights shape");
        assert_eq!(fit.singular_values.len(), k, "singular values length");
    }

    // ── Test 16: pls_svd_matches_cross_cov_svd ───────────────────────────
    #[test]
    fn pls_svd_matches_cross_cov_svd() {
        let mut rng = LcgRng::new(300);
        let n = 30;
        let p = 4;
        let q = 3;
        let k = 2;
        let x = rand_matrix(&mut rng, n, p);
        let y = rand_matrix(&mut rng, n, q);
        let fit = pls_svd_fit(&x, n, p, &y, q, k).expect("fit");
        // Compute C_XY manually and verify singular values match
        let x_mean: Vec<f64> = (0..p)
            .map(|j| (0..n).map(|i| x[i * p + j]).sum::<f64>() / n as f64)
            .collect();
        let y_mean: Vec<f64> = (0..q)
            .map(|j| (0..n).map(|i| y[i * q + j]).sum::<f64>() / n as f64)
            .collect();
        let mut xc = x.clone();
        let mut yc = y.clone();
        for i in 0..n {
            for j in 0..p {
                xc[i * p + j] -= x_mean[j];
            }
            for j in 0..q {
                yc[i * q + j] -= y_mean[j];
            }
        }
        let denom = (n - 1) as f64;
        let mut c_xy = vec![0.0_f64; p * q];
        for r in 0..p {
            for c in 0..q {
                let mut acc = 0.0;
                for i in 0..n {
                    acc += xc[i * p + r] * yc[i * q + c];
                }
                c_xy[r * q + c] = acc / denom;
            }
        }
        let (_u, sigma_manual, _v) = svd_via_ata(&c_xy, p, q, k).expect("svd");
        for (i, (&s_fit, &s_man)) in fit
            .singular_values
            .iter()
            .zip(sigma_manual.iter())
            .enumerate()
        {
            assert!(
                (s_fit - s_man).abs() < 1e-8,
                "singular value {i}: fit={s_fit:.6}, manual={s_man:.6}"
            );
        }
    }

    // ── Test 17: cca_regularization_effect ───────────────────────────────
    #[test]
    fn cca_regularization_effect() {
        let mut rng = LcgRng::new(500);
        let n = 20;
        let p = 3;
        let q = 3;
        let x = rand_matrix(&mut rng, n, p);
        let y = rand_matrix(&mut rng, n, q);
        let cfg_low = CcaConfig {
            n_components: 1,
            regularization: 1e-8,
        };
        let cfg_high = CcaConfig {
            n_components: 1,
            regularization: 10.0,
        };
        let fit_low = cca_fit(&x, n, p, &y, q, &cfg_low).expect("low reg");
        let fit_high = cca_fit(&x, n, p, &y, q, &cfg_high).expect("high reg");
        // Higher regularization shrinks the effective cross-covariance contribution,
        // leading to smaller or equal first canonical correlation.
        assert!(
            fit_high.correlations[0] <= fit_low.correlations[0] + 0.1,
            "high-reg corr {} should be ≤ low-reg corr {} (with tolerance)",
            fit_high.correlations[0],
            fit_low.correlations[0]
        );
    }

    /// Pearson correlation between two equal-length slices.
    fn pearson_corr(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let ma = a.iter().sum::<f64>() / n;
        let mb = b.iter().sum::<f64>() / n;
        let num: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - ma) * (bi - mb))
            .sum();
        let da: f64 = a.iter().map(|&ai| (ai - ma).powi(2)).sum::<f64>().sqrt();
        let db: f64 = b.iter().map(|&bi| (bi - mb).powi(2)).sum::<f64>().sqrt();
        if da < 1e-14 || db < 1e-14 {
            return 0.0;
        }
        num / (da * db)
    }
}
