//! Generalised Additive Models (GAM) via backfitting with penalised cubic B-splines.
//!
//! # Algorithm
//!
//! The additive model is
//!   `y_i = α + Σ_j f_j(x_{ij}) + ε_i`
//! where each smooth component f_j is estimated as a penalised cubic B-spline.
//!
//! **Basis construction** — for each term j, an augmented knot vector is built
//! as `[x_min × 4, interior_knots..., x_max × 4]` and B-spline basis functions
//! are evaluated via the Cox-de Boor recursion (degree 3).
//!
//! **Penalty** — a second-difference penalty matrix P = DᵀD (where D is the
//! second-difference operator) penalises curvature: roughness = λ γᵀPγ.
//!
//! **Backfitting** (Wood §4.4) — iterates over terms, fitting each by penalised
//! least squares with the partial residual as response, until convergence.
//!
//! **Effective degrees of freedom** — edf_j = trace(H_j) where
//! H_j = B_j (BᵀB_j + λ_j P_j)⁻¹ Bᵀ_j is the hat matrix for term j.
//!
//! # References
//! - Wood, S.N. (2017) *Generalised Additive Models: An Introduction with R*,
//!   2nd ed., Chapter 4.
//! - de Boor, C. (2001) *A Practical Guide to Splines*, Revised Edition.

use crate::error::{StatsError, StatsResult};
use crate::regression::linear::{matrix_inverse_lu, matrix_mul, matrix_transpose};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration & Result structs
// ─────────────────────────────────────────────────────────────────────────────

/// Per-term B-spline smoothing configuration.
#[derive(Debug, Clone)]
pub struct GamSmoothConfig {
    /// Number of interior knots. 0 = use 3 automatically.
    pub n_knots: usize,
    /// Smoothing penalty λ. 0 = OLS spline (no penalty).
    pub lambda: f64,
}

/// Overall GAM configuration.
#[derive(Debug, Clone)]
pub struct GamConfig {
    /// One `GamSmoothConfig` per term.
    pub terms: Vec<GamSmoothConfig>,
    /// Maximum backfitting iterations (default 200).
    pub max_iter: usize,
    /// Convergence tolerance on maximum relative γ change (default 1e-6).
    pub tol: f64,
    /// Whether to fit a global intercept (default true).
    pub include_intercept: bool,
}

impl Default for GamConfig {
    fn default() -> Self {
        Self {
            terms: Vec::new(),
            max_iter: 200,
            tol: 1e-6,
            include_intercept: true,
        }
    }
}

/// Fitted GAM model.
#[derive(Debug, Clone)]
pub struct GamFit {
    /// Global intercept.
    pub intercept: f64,
    /// B-spline coefficients γ_j for each term.
    pub smooth_coefs: Vec<Vec<f64>>,
    /// Full extended knot vector for each term (length n_basis + 4 = n_interior + 8).
    pub knots: Vec<Vec<f64>>,
    /// Effective degrees of freedom per term.
    pub edf: Vec<f64>,
    /// In-sample fitted values.
    pub fitted: Vec<f64>,
    /// In-sample residuals.
    pub residuals: Vec<f64>,
    /// Residual sum of squares.
    pub rss: f64,
    /// Whether backfitting converged.
    pub converged: bool,
    /// Number of backfitting iterations performed.
    pub iterations: usize,
    /// Number of smooth terms.
    pub n_terms: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// B-spline basis: de Boor / Cox-de Boor recursion (degree 3)
// ─────────────────────────────────────────────────────────────────────────────

/// Place `n_knots` interior knots at quantiles of the sorted covariate values.
///
/// Positions: quantile at ranks 1/(n+1), 2/(n+1), …, n/(n+1).
fn place_interior_knots(x_sorted: &[f64], n_knots: usize) -> Vec<f64> {
    if n_knots == 0 || x_sorted.is_empty() {
        return vec![];
    }
    let n = x_sorted.len();
    (1..=n_knots)
        .map(|k| {
            // position in [0, 1]: k / (n_knots + 1)
            let frac = k as f64 / (n_knots + 1) as f64;
            let pos = frac * (n - 1) as f64;
            let lo = pos.floor() as usize;
            let hi = lo + 1;
            if hi >= n {
                x_sorted[n - 1]
            } else {
                let t = pos - lo as f64;
                x_sorted[lo] * (1.0 - t) + x_sorted[hi] * t
            }
        })
        .collect()
}

/// Build the augmented knot vector `[x_min × 4, interior..., x_max × 4]`.
///
/// This gives C² cubic B-splines with natural boundary conditions.
fn build_knot_vector(x_min: f64, x_max: f64, interior: &[f64]) -> Vec<f64> {
    let mut knots = Vec::with_capacity(8 + interior.len());
    // 4 repeated left boundary knots
    knots.extend([x_min; 4]);
    knots.extend_from_slice(interior);
    // 4 repeated right boundary knots
    knots.extend([x_max; 4]);
    knots
}

/// Evaluate all cubic B-spline basis functions at `x` using the Cox-de Boor recursion.
///
/// `knots` must be the full augmented knot vector of length n_basis + 4.
/// Returns a `Vec` of length `n_basis = knots.len() - 4`.
fn bspline_eval(x: f64, knots: &[f64]) -> Vec<f64> {
    let n_knots = knots.len();
    let degree = 3usize;
    // n_basis = n_knots - degree - 1
    let n_basis = n_knots.saturating_sub(degree + 1);
    if n_basis == 0 || n_knots < 2 {
        return vec![];
    }

    // Number of B-spline intervals at degree 0 = n_knots - 1
    let n_intervals = n_knots - 1;

    // We need a 2D table b[i][k] for i=0..n_intervals, k=0..=degree
    // Use flat Vec of size n_intervals * (degree + 1)
    let d = degree + 1;
    let mut b = vec![0.0_f64; n_intervals * d];

    // Degree 0 indicator: B_{i,0}(x)
    // Special case for the last knot: we include the right boundary
    let x_max = *knots.last().unwrap_or(&f64::INFINITY);
    for i in 0..n_intervals {
        let in_interval = if i == n_intervals - 1 {
            // Last interval: include right boundary
            x >= knots[i] && x <= x_max
        } else {
            x >= knots[i] && x < knots[i + 1]
        };
        b[i * d] = if in_interval { 1.0 } else { 0.0 };
    }

    // Recursion for degrees 1, 2, 3
    for k in 1..=degree {
        for i in 0..(n_intervals - k) {
            // Left term: (x - knots[i]) / (knots[i+k] - knots[i]) * B_{i,k-1}(x)
            let left = {
                let denom = knots[i + k] - knots[i];
                if denom.abs() < 1e-300 {
                    0.0
                } else {
                    (x - knots[i]) / denom * b[i * d + (k - 1)]
                }
            };
            // Right term: (knots[i+k+1] - x) / (knots[i+k+1] - knots[i+1]) * B_{i+1,k-1}(x)
            let right = {
                let denom = knots[i + k + 1] - knots[i + 1];
                if denom.abs() < 1e-300 {
                    0.0
                } else {
                    (knots[i + k + 1] - x) / denom * b[(i + 1) * d + (k - 1)]
                }
            };
            b[i * d + k] = left + right;
        }
    }

    // Extract B_{i, degree} for i = 0..n_basis
    (0..n_basis).map(|i| b[i * d + degree]).collect()
}

/// Build basis matrix B of shape (n_samples, n_basis) for covariate x_j.
fn build_basis_matrix(x_j: &[f64], knots: &[f64]) -> Vec<f64> {
    let n = x_j.len();
    let n_basis = knots.len().saturating_sub(4); // degree 3
    let mut b = vec![0.0_f64; n * n_basis];
    for (i, &xi) in x_j.iter().enumerate() {
        let row = bspline_eval(xi, knots);
        for (j, &val) in row.iter().enumerate().take(n_basis) {
            b[i * n_basis + j] = val;
        }
    }
    b
}

// ─────────────────────────────────────────────────────────────────────────────
// Penalty matrix
// ─────────────────────────────────────────────────────────────────────────────

/// Build second-difference penalty matrix P = DᵀD of size (n_basis × n_basis).
///
/// D is the (n_basis-2) × n_basis second-difference operator:
///   D[k, k] = 1, D[k, k+1] = -2, D[k, k+2] = 1, for k = 0..n_basis-2.
fn build_penalty_matrix(n_basis: usize) -> Vec<f64> {
    if n_basis < 3 {
        return vec![0.0; n_basis * n_basis];
    }
    let n_d = n_basis - 2; // number of rows in D
    // D: n_d × n_basis
    let mut d_mat = vec![0.0_f64; n_d * n_basis];
    for k in 0..n_d {
        d_mat[k * n_basis + k] = 1.0;
        d_mat[k * n_basis + k + 1] = -2.0;
        d_mat[k * n_basis + k + 2] = 1.0;
    }
    // P = Dᵀ D
    // Dt: n_basis × n_d
    let dt = matrix_transpose(&d_mat, n_d, n_basis);
    // P = Dt · D (n_basis × n_basis)
    // Manual multiply to avoid propagating StatsResult here
    let mut p = vec![0.0_f64; n_basis * n_basis];
    for i in 0..n_basis {
        for j in 0..n_basis {
            let mut acc = 0.0_f64;
            for k in 0..n_d {
                acc += dt[i * n_d + k] * d_mat[k * n_basis + j];
            }
            p[i * n_basis + j] = acc;
        }
    }
    p
}

// ─────────────────────────────────────────────────────────────────────────────
// Effective degrees of freedom
// ─────────────────────────────────────────────────────────────────────────────

/// Compute trace(H_j) where H_j = B_j (BᵀB_j + λ P_j)⁻¹ Bᵀ_j.
///
/// Efficiently: tr(H_j) = tr(A_inv BᵀB) where A = BᵀB + λP.
/// Since A A_inv = I, tr(A_inv BᵀB) = tr(I - λ A_inv P) = n_basis - λ tr(A_inv P).
fn compute_edf(btb: &[f64], a_inv: &[f64], p: &[f64], n_basis: usize, lambda: f64) -> f64 {
    // tr(A_inv P): sum of diagonal of A_inv · P
    let mut trace_inv_p = 0.0_f64;
    for i in 0..n_basis {
        for k in 0..n_basis {
            trace_inv_p += a_inv[i * n_basis + k] * p[k * n_basis + i];
        }
    }
    // edf = trace(A_inv BᵀB) = n_basis - lambda * trace(A_inv P)
    // Alternatively: directly compute trace(A_inv BtB)
    let mut trace_h = 0.0_f64;
    for i in 0..n_basis {
        for k in 0..n_basis {
            trace_h += a_inv[i * n_basis + k] * btb[k * n_basis + i];
        }
    }
    let _ = (trace_inv_p, lambda); // both available but we use the direct formula
    trace_h.clamp(1.0, n_basis as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Fit a Generalised Additive Model with penalised cubic B-spline smooth terms.
///
/// # Parameters
/// - `x` — row-major covariate matrix of shape `(n_samples, n_terms)`.
///   Access covariate j at observation i as `x[i * n_terms + j]`.
/// - `y` — response vector, length `n_samples`.
/// - `n_samples` — number of observations.
/// - `n_terms` — number of smooth terms.
/// - `config` — model configuration including per-term smoothing parameters.
///
/// # Errors
/// See [`StatsError`] variants for specific failure modes.
pub fn gam_fit(
    x: &[f64],
    y: &[f64],
    n_samples: usize,
    n_terms: usize,
    config: &GamConfig,
) -> StatsResult<GamFit> {
    // ── Validation ───────────────────────────────────────────────────────────
    if n_terms == 0 {
        return Err(StatsError::InvalidParameter {
            name: "n_terms".to_string(),
            reason: "must be >= 1".to_string(),
        });
    }
    if config.terms.len() != n_terms {
        return Err(StatsError::DimensionMismatch {
            a: config.terms.len(),
            b: n_terms,
        });
    }
    if x.len() != n_samples * n_terms {
        return Err(StatsError::DimensionMismatch {
            a: x.len(),
            b: n_samples * n_terms,
        });
    }
    if y.len() != n_samples {
        return Err(StatsError::DimensionMismatch {
            a: y.len(),
            b: n_samples,
        });
    }

    // ── Per-term effective knot count (0 → 3) ────────────────────────────────
    let effective_n_knots: Vec<usize> = config
        .terms
        .iter()
        .map(|tc| if tc.n_knots == 0 { 3 } else { tc.n_knots })
        .collect();

    // Validate that n_samples >= n_knots + 5 for each term
    for &nk in effective_n_knots.iter() {
        if n_samples < nk + 5 {
            return Err(StatsError::InsufficientSampleSize {
                got: n_samples,
                need: nk + 5,
            });
        }
    }

    // ── Extract per-term covariates ──────────────────────────────────────────
    // x_j[i] = x[i * n_terms + j]
    let extract_col =
        |j: usize| -> Vec<f64> { (0..n_samples).map(|i| x[i * n_terms + j]).collect() };

    // ── Build knot vectors and basis matrices for all terms ──────────────────
    // knot_vecs[j] = full extended knot vector (length n_interior + 8)
    let mut knot_vecs: Vec<Vec<f64>> = Vec::with_capacity(n_terms);
    let mut basis_mats: Vec<Vec<f64>> = Vec::with_capacity(n_terms); // n_samples × n_basis_j
    let mut n_bases: Vec<usize> = Vec::with_capacity(n_terms);
    let mut penalty_mats: Vec<Vec<f64>> = Vec::with_capacity(n_terms);

    for (j, &n_ki) in effective_n_knots.iter().enumerate() {
        let x_j = extract_col(j);

        // Sort for quantile placement
        let mut x_sorted = x_j.clone();
        x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let x_min = *x_sorted.first().unwrap_or(&0.0);
        let x_max = *x_sorted.last().unwrap_or(&1.0);
        // Nudge boundaries slightly outward so boundary points are inside
        let x_lo = x_min - 1e-8 * (x_max - x_min + 1e-8);
        let x_hi = x_max + 1e-8 * (x_max - x_min + 1e-8);

        let interior = place_interior_knots(&x_sorted, n_ki);
        let knots = build_knot_vector(x_lo, x_hi, &interior);

        let n_basis = knots.len().saturating_sub(4); // = n_ki + 4
        n_bases.push(n_basis);

        let b_mat = build_basis_matrix(&x_j, &knots);
        basis_mats.push(b_mat);

        let p_mat = build_penalty_matrix(n_basis);
        penalty_mats.push(p_mat);

        knot_vecs.push(knots);
    }

    // ── Backfitting ──────────────────────────────────────────────────────────
    let mut intercept = if config.include_intercept {
        y.iter().sum::<f64>() / n_samples as f64
    } else {
        0.0
    };

    // f_j: smooth contribution of term j at each observation
    let mut f: Vec<Vec<f64>> = (0..n_terms).map(|_| vec![0.0_f64; n_samples]).collect();

    // gamma_j: B-spline coefficients for term j
    let mut gamma: Vec<Vec<f64>> = (0..n_terms).map(|j| vec![0.0_f64; n_bases[j]]).collect();

    // Inverse matrices cached per term (built inside iteration)
    let mut a_inv_cache: Vec<Vec<f64>> = (0..n_terms)
        .map(|j| vec![0.0_f64; n_bases[j] * n_bases[j]])
        .collect();

    // BtB matrices cached per term
    let mut btb_cache: Vec<Vec<f64>> = (0..n_terms)
        .map(|j| vec![0.0_f64; n_bases[j] * n_bases[j]])
        .collect();

    // Pre-compute BtB and A_inv for each term (they don't change across iterations
    // since x and lambda are fixed)
    for j in 0..n_terms {
        let nb = n_bases[j];
        let b_j = &basis_mats[j];
        let lambda_j = config.terms[j].lambda;
        let p_j = &penalty_mats[j];

        // BᵀB (nb × nb)
        let bt = matrix_transpose(b_j, n_samples, nb);
        let btb = matrix_mul(&bt, b_j, nb, n_samples, nb)?;

        // A = BᵀB + λP
        let mut a = btb.clone();
        for k in 0..(nb * nb) {
            a[k] += lambda_j * p_j[k];
        }

        let a_inv = matrix_inverse_lu(&a, nb)?;
        btb_cache[j] = btb;
        a_inv_cache[j] = a_inv;
    }

    let mut converged = false;
    let mut n_iter = 0usize;

    for iter in 0..config.max_iter {
        n_iter = iter + 1;
        let mut max_rel_delta = 0.0_f64;

        for j in 0..n_terms {
            let nb = n_bases[j];
            let b_j = &basis_mats[j];
            let a_inv = &a_inv_cache[j];

            // Partial residual: y - intercept - sum_{k != j} f_k
            let partial: Vec<f64> = (0..n_samples)
                .map(|i| {
                    let sum_other: f64 = (0..n_terms).filter(|&k| k != j).map(|k| f[k][i]).sum();
                    y[i] - intercept - sum_other
                })
                .collect();

            // Bᵀ partial (nb × 1)
            let bt = matrix_transpose(b_j, n_samples, nb);
            let btr = matrix_mul(&bt, &partial, nb, n_samples, 1)?;

            // gamma_new = A_inv · btr
            let gamma_new = matrix_mul(a_inv, &btr, nb, nb, 1)?;

            // Relative change for convergence check
            for k in 0..nb {
                let denom = gamma[j][k].abs().max(1e-8);
                let rel = (gamma_new[k] - gamma[j][k]).abs() / denom;
                max_rel_delta = max_rel_delta.max(rel);
            }
            gamma[j] = gamma_new.clone();

            // f_j = B_j * gamma_j (n_samples × 1)
            let f_new = matrix_mul(b_j, &gamma[j], n_samples, nb, 1)?;
            let mean_f: f64 = f_new.iter().sum::<f64>() / n_samples as f64;

            // Center: absorb mean into intercept
            if config.include_intercept {
                intercept += mean_f;
                f[j] = f_new.iter().map(|&v| v - mean_f).collect();
            } else {
                f[j] = f_new;
            }
        }

        if max_rel_delta < config.tol {
            converged = true;
            break;
        }
    }

    // ── Fitted values, residuals, RSS ────────────────────────────────────────
    let fitted: Vec<f64> = (0..n_samples)
        .map(|i| intercept + (0..n_terms).map(|j| f[j][i]).sum::<f64>())
        .collect();
    let residuals: Vec<f64> = y.iter().zip(&fitted).map(|(yi, fi)| yi - fi).collect();
    let rss: f64 = residuals.iter().map(|r| r * r).sum();

    // ── Effective DF per term ────────────────────────────────────────────────
    let edf: Vec<f64> = (0..n_terms)
        .map(|j| {
            compute_edf(
                &btb_cache[j],
                &a_inv_cache[j],
                &penalty_mats[j],
                n_bases[j],
                config.terms[j].lambda,
            )
        })
        .collect();

    Ok(GamFit {
        intercept,
        smooth_coefs: gamma,
        knots: knot_vecs,
        edf,
        fitted,
        residuals,
        rss,
        converged,
        iterations: n_iter,
        n_terms,
    })
}

/// Predict on new covariate data using a fitted GAM.
///
/// `x_new` — row-major shape `(n_new, fit.n_terms)`.
pub fn gam_predict(fit: &GamFit, x_new: &[f64], n_new: usize) -> StatsResult<Vec<f64>> {
    let n_terms = fit.n_terms;
    if x_new.len() != n_new * n_terms {
        return Err(StatsError::DimensionMismatch {
            a: x_new.len(),
            b: n_new * n_terms,
        });
    }

    let mut preds = vec![fit.intercept; n_new];
    for j in 0..n_terms {
        let x_j: Vec<f64> = (0..n_new).map(|i| x_new[i * n_terms + j]).collect();
        let nb = fit.smooth_coefs[j].len();
        if nb == 0 {
            continue;
        }
        let b_j = build_basis_matrix(&x_j, &fit.knots[j]);
        let f_j = matrix_mul(&b_j, &fit.smooth_coefs[j], n_new, nb, 1)?;
        for i in 0..n_new {
            preds[i] += f_j[i];
        }
    }
    Ok(preds)
}

/// Compute the partial effect of term `term` at new covariate values.
///
/// Returns a vector of length `n_new` containing `f_term(x_new_term)`.
pub fn gam_partial_effects(
    fit: &GamFit,
    x_new: &[f64],
    n_new: usize,
    term: usize,
) -> StatsResult<Vec<f64>> {
    if term >= fit.n_terms {
        return Err(StatsError::InvalidParameter {
            name: "term".to_string(),
            reason: format!("term index {term} >= n_terms {}", fit.n_terms),
        });
    }
    if x_new.len() != n_new * fit.n_terms {
        return Err(StatsError::DimensionMismatch {
            a: x_new.len(),
            b: n_new * fit.n_terms,
        });
    }

    let nb = fit.smooth_coefs[term].len();
    let x_j: Vec<f64> = (0..n_new).map(|i| x_new[i * fit.n_terms + term]).collect();

    if nb == 0 {
        return Ok(vec![0.0; n_new]);
    }

    let b_j = build_basis_matrix(&x_j, &fit.knots[term]);
    let f_j = matrix_mul(&b_j, &fit.smooth_coefs[term], n_new, nb, 1)?;
    Ok(f_j)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
        if n <= 1 {
            return vec![start];
        }
        (0..n)
            .map(|i| start + (stop - start) * i as f64 / (n - 1) as f64)
            .collect()
    }

    fn make_config_1term(n_knots: usize, lambda: f64) -> GamConfig {
        GamConfig {
            terms: vec![GamSmoothConfig { n_knots, lambda }],
            max_iter: 200,
            tol: 1e-6,
            include_intercept: true,
        }
    }

    // ── Test 1: Linear fit ─────────────────────────────────────────────────────

    #[test]
    fn linear_fit_low_lambda() {
        let n = 50usize;
        let x: Vec<f64> = linspace(0.0, 1.0, n);
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let config = make_config_1term(4, 0.0);
        let fit = gam_fit(&x, &y, n, 1, &config).expect("gam_fit linear");
        let max_err = fit
            .fitted
            .iter()
            .zip(&y)
            .map(|(fi, yi)| (fi - yi).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 0.01,
            "Linear fit max error = {max_err}, expected < 0.01"
        );
    }

    // ── Test 2: Sine recovery ──────────────────────────────────────────────────

    #[test]
    fn sine_recovery() {
        let n = 100usize;
        let x: Vec<f64> = linspace(0.0, 1.0, n);
        let y: Vec<f64> = x.iter().map(|&xi| (2.0 * PI * xi).sin()).collect();
        let config = make_config_1term(8, 0.01);
        let fit = gam_fit(&x, &y, n, 1, &config).expect("gam_fit sine");
        let mse: f64 = fit
            .fitted
            .iter()
            .zip(&y)
            .map(|(fi, yi)| (fi - yi).powi(2))
            .sum::<f64>()
            / n as f64;
        assert!(mse < 0.01, "Sine MSE = {mse}, expected < 0.01");
    }

    // ── Test 3: Quadratic ─────────────────────────────────────────────────────

    #[test]
    fn quadratic_fit() {
        let n = 50usize;
        let x: Vec<f64> = linspace(0.0, 1.0, n);
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let config = make_config_1term(4, 0.0);
        let fit = gam_fit(&x, &y, n, 1, &config).expect("gam_fit quadratic");
        let mse: f64 = fit
            .fitted
            .iter()
            .zip(&y)
            .map(|(fi, yi)| (fi - yi).powi(2))
            .sum::<f64>()
            / n as f64;
        assert!(mse < 1e-3, "Quadratic MSE = {mse}, expected < 1e-3");
    }

    // ── Test 4: Two-term additive ─────────────────────────────────────────────

    #[test]
    fn two_term_additive() {
        let n = 200usize;
        // x0 ∈ [0, 2π], x1 ∈ [0, 1]
        let x0: Vec<f64> = linspace(0.0, 2.0 * PI, n);
        let x1: Vec<f64> = linspace(0.0, 1.0, n);
        let y: Vec<f64> = x0
            .iter()
            .zip(&x1)
            .map(|(&a, &b)| a.sin() + 2.0 * b)
            .collect();
        // Interleave into row-major [x0_0, x1_0, x0_1, x1_1, ...]
        let x: Vec<f64> = (0..n).flat_map(|i| [x0[i], x1[i]]).collect();
        let config = GamConfig {
            terms: vec![
                GamSmoothConfig {
                    n_knots: 6,
                    lambda: 0.001,
                },
                GamSmoothConfig {
                    n_knots: 4,
                    lambda: 0.0,
                },
            ],
            max_iter: 200,
            tol: 1e-6,
            include_intercept: true,
        };
        let fit = gam_fit(&x, &y, n, 2, &config).expect("gam_fit 2-term");
        let mse: f64 = fit
            .fitted
            .iter()
            .zip(&y)
            .map(|(fi, yi)| (fi - yi).powi(2))
            .sum::<f64>()
            / n as f64;
        assert!(mse < 0.05, "Two-term MSE = {mse}, expected < 0.05");
    }

    // ── Test 5: Intercept recovery for constant series ────────────────────────

    #[test]
    fn intercept_recovery_constant() {
        let n = 50usize;
        let x: Vec<f64> = linspace(0.0, 1.0, n);
        let y = vec![5.0_f64; n];
        let config = make_config_1term(4, 0.0);
        let fit = gam_fit(&x, &y, n, 1, &config).expect("fit constant");
        assert!(
            (fit.intercept - 5.0).abs() < 0.01,
            "Intercept = {}, expected ≈ 5.0",
            fit.intercept
        );
    }

    // ── Test 6: Larger lambda → smaller edf ──────────────────────────────────

    #[test]
    fn larger_lambda_smaller_edf() {
        let n = 100usize;
        let x: Vec<f64> = linspace(0.0, 1.0, n);
        let y: Vec<f64> = x.iter().map(|&xi| (2.0 * PI * xi).sin()).collect();

        let config_low = make_config_1term(6, 0.001);
        let config_high = make_config_1term(6, 1000.0);

        let fit_low = gam_fit(&x, &y, n, 1, &config_low).expect("fit low lambda");
        let fit_high = gam_fit(&x, &y, n, 1, &config_high).expect("fit high lambda");

        assert!(
            fit_high.edf[0] < fit_low.edf[0],
            "edf with high lambda ({}) should be < edf with low lambda ({})",
            fit_high.edf[0],
            fit_low.edf[0]
        );
    }

    // ── Test 7: 1 ≤ edf_j ≤ n_knots + 4 ──────────────────────────────────────

    #[test]
    fn edf_within_bounds() {
        let n = 100usize;
        let x: Vec<f64> = linspace(0.0, 1.0, n);
        let y: Vec<f64> = x.iter().map(|&xi| (2.0 * PI * xi).sin()).collect();
        let n_knots = 5usize;
        let config = make_config_1term(n_knots, 0.1);
        let fit = gam_fit(&x, &y, n, 1, &config).expect("fit");
        let edf = fit.edf[0];
        assert!(
            edf >= 1.0 && edf <= (n_knots + 4) as f64,
            "edf = {edf} out of bounds [1, {}]",
            n_knots + 4
        );
    }

    // ── Test 8: converged on smooth data ─────────────────────────────────────

    #[test]
    fn converges_on_smooth_data() {
        let n = 100usize;
        let x: Vec<f64> = linspace(0.0, 1.0, n);
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let config = make_config_1term(4, 0.01);
        let fit = gam_fit(&x, &y, n, 1, &config).expect("fit");
        assert!(fit.converged, "GAM should converge on smooth data");
    }

    // ── Test 9: gam_predict ≈ fit.fitted on training data ────────────────────

    #[test]
    fn predict_matches_fitted() {
        let n = 60usize;
        let x: Vec<f64> = linspace(0.0, 1.0, n);
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi + 0.1).collect();
        let config = make_config_1term(4, 0.01);
        let fit = gam_fit(&x, &y, n, 1, &config).expect("fit");
        let preds = gam_predict(&fit, &x, n).expect("predict");
        let max_diff = preds
            .iter()
            .zip(&fit.fitted)
            .map(|(p, f)| (p - f).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-8,
            "gam_predict vs fitted max diff = {max_diff}"
        );
    }

    // ── Test 10: partial effects sum ≈ fitted - intercept ────────────────────

    #[test]
    fn partial_effects_sum_to_fitted() {
        let n = 80usize;
        let x0: Vec<f64> = linspace(0.0, PI, n);
        let x1: Vec<f64> = linspace(0.0, 1.0, n);
        let y: Vec<f64> = x0.iter().zip(&x1).map(|(&a, &b)| a.sin() + b).collect();
        let x: Vec<f64> = (0..n).flat_map(|i| [x0[i], x1[i]]).collect();
        let config = GamConfig {
            terms: vec![
                GamSmoothConfig {
                    n_knots: 5,
                    lambda: 0.01,
                },
                GamSmoothConfig {
                    n_knots: 4,
                    lambda: 0.01,
                },
            ],
            max_iter: 200,
            tol: 1e-6,
            include_intercept: true,
        };
        let fit = gam_fit(&x, &y, n, 2, &config).expect("fit");
        let f0 = gam_partial_effects(&fit, &x, n, 0).expect("f0");
        let f1 = gam_partial_effects(&fit, &x, n, 1).expect("f1");
        let max_err: f64 = (0..n)
            .map(|i| {
                let sum = f0[i] + f1[i];
                let expected = fit.fitted[i] - fit.intercept;
                (sum - expected).abs()
            })
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1e-8, "Partial effects sum deviation = {max_err}");
    }

    // ── Test 11: config.terms.len() != n_terms → DimensionMismatch ───────────

    #[test]
    fn wrong_terms_length_error() {
        let n = 50usize;
        let x: Vec<f64> = linspace(0.0, 1.0, n);
        let y = vec![1.0_f64; n];
        let config = GamConfig {
            terms: vec![
                GamSmoothConfig {
                    n_knots: 3,
                    lambda: 0.1,
                },
                GamSmoothConfig {
                    n_knots: 3,
                    lambda: 0.1,
                },
            ],
            max_iter: 10,
            tol: 1e-4,
            include_intercept: true,
        };
        let result = gam_fit(&x, &y, n, 1, &config); // n_terms=1 but config has 2
        assert!(
            matches!(result, Err(StatsError::DimensionMismatch { .. })),
            "Expected DimensionMismatch"
        );
    }

    // ── Test 12: n_terms=0 → InvalidParameter ────────────────────────────────

    #[test]
    fn zero_n_terms_error() {
        let n = 50usize;
        let x: Vec<f64> = linspace(0.0, 1.0, n);
        let y = vec![1.0_f64; n];
        let config = GamConfig {
            terms: vec![],
            max_iter: 10,
            tol: 1e-4,
            include_intercept: true,
        };
        let result = gam_fit(&x, &y, n, 0, &config);
        assert!(
            matches!(result, Err(StatsError::InvalidParameter { .. })),
            "Expected InvalidParameter for n_terms=0"
        );
    }

    // ── Test 13: x.len() != n_samples * n_terms → DimensionMismatch ──────────

    #[test]
    fn wrong_x_length_error() {
        let n = 50usize;
        let x = vec![1.0_f64; n + 5]; // wrong length
        let y = vec![1.0_f64; n];
        let config = make_config_1term(3, 0.1);
        let result = gam_fit(&x, &y, n, 1, &config);
        assert!(
            matches!(result, Err(StatsError::DimensionMismatch { .. })),
            "Expected DimensionMismatch"
        );
    }

    // ── Test 14: n_samples < n_knots + 5 → InsufficientSampleSize ────────────

    #[test]
    fn insufficient_samples_error() {
        let n = 5usize;
        let x: Vec<f64> = linspace(0.0, 1.0, n);
        let y = vec![1.0_f64; n];
        let config = make_config_1term(10, 0.1); // need >= 10+5=15 samples
        let result = gam_fit(&x, &y, n, 1, &config);
        assert!(
            matches!(result, Err(StatsError::InsufficientSampleSize { .. })),
            "Expected InsufficientSampleSize"
        );
    }

    // ── Test 15: n_knots=0 → auto 3 interior knots, no error ─────────────────

    #[test]
    fn zero_n_knots_auto_three() {
        let n = 50usize;
        let x: Vec<f64> = linspace(0.0, 1.0, n);
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let config = make_config_1term(0, 0.1); // n_knots=0 → use 3
        let result = gam_fit(&x, &y, n, 1, &config);
        assert!(
            result.is_ok(),
            "n_knots=0 should succeed: {:?}",
            result.err()
        );
        let fit = result.expect("result should be present");
        // With 3 interior knots, n_basis = 3 + 4 = 7
        assert_eq!(fit.smooth_coefs[0].len(), 7);
    }

    // ── Test 16: gam_partial_effects returns length n_new ────────────────────

    #[test]
    fn partial_effects_length() {
        let n = 60usize;
        let x: Vec<f64> = linspace(0.0, 1.0, n);
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let config = make_config_1term(4, 0.01);
        let fit = gam_fit(&x, &y, n, 1, &config).expect("fit");
        let x_new: Vec<f64> = linspace(0.1, 0.9, 20);
        let effects = gam_partial_effects(&fit, &x_new, 20, 0).expect("effects");
        assert_eq!(effects.len(), 20);
    }

    // ── Test 17: gam_partial_effects with term >= n_terms → error ─────────────

    #[test]
    fn partial_effects_out_of_bounds() {
        let n = 50usize;
        let x: Vec<f64> = linspace(0.0, 1.0, n);
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let config = make_config_1term(4, 0.01);
        let fit = gam_fit(&x, &y, n, 1, &config).expect("fit");
        let result = gam_partial_effects(&fit, &x, n, 5); // term=5 >= n_terms=1
        assert!(
            matches!(result, Err(StatsError::InvalidParameter { .. })),
            "Expected error for term out of bounds"
        );
    }

    // ── Test 18: Predictions on training data match fitted ────────────────────

    #[test]
    fn predict_on_training_matches_fitted_exactly() {
        let n = 40usize;
        let x: Vec<f64> = linspace(0.0, 1.0, n);
        let y: Vec<f64> = x.iter().map(|&xi| (PI * xi).sin()).collect();
        let config = make_config_1term(5, 0.01);
        let fit = gam_fit(&x, &y, n, 1, &config).expect("fit");
        let pred = gam_predict(&fit, &x, n).expect("predict");
        for (i, (&p, &f)) in pred.iter().zip(&fit.fitted).enumerate() {
            assert!((p - f).abs() < 1e-8, "predict[{i}] = {p} vs fitted = {f}");
        }
    }

    // ── Test 19: RSS is non-negative ─────────────────────────────────────────

    #[test]
    fn rss_non_negative() {
        let n = 80usize;
        let x: Vec<f64> = linspace(0.0, 1.0, n);
        let y: Vec<f64> = x.iter().map(|&xi| (2.0 * PI * xi).sin()).collect();
        let config = make_config_1term(6, 0.1);
        let fit = gam_fit(&x, &y, n, 1, &config).expect("fit");
        assert!(fit.rss >= 0.0, "RSS must be non-negative, got {}", fit.rss);
    }

    // ── Test 20: heavy smoothing → near-linear effect (edf ≈ 2) ─────────────

    #[test]
    fn heavy_smoothing_near_linear_edf() {
        let n = 100usize;
        let x: Vec<f64> = linspace(0.0, 1.0, n);
        let y: Vec<f64> = x.iter().map(|&xi| (2.0 * PI * xi).sin()).collect();
        let config = make_config_1term(6, 1000.0);
        let fit = gam_fit(&x, &y, n, 1, &config).expect("fit");
        // Under heavy penalty the spline should be nearly linear: edf close to 2
        assert!(
            fit.edf[0] < 3.0,
            "edf with lambda=1000 should be < 3, got {}",
            fit.edf[0]
        );
    }

    // ── Test 21: Determinism ─────────────────────────────────────────────────

    #[test]
    fn deterministic_output() {
        let n = 60usize;
        let x: Vec<f64> = linspace(0.0, 1.0, n);
        let y: Vec<f64> = x.iter().map(|&xi| (2.0 * PI * xi).sin()).collect();
        let config = make_config_1term(5, 0.1);
        let fit1 = gam_fit(&x, &y, n, 1, &config).expect("fit1");
        let fit2 = gam_fit(&x, &y, n, 1, &config).expect("fit2");
        for (i, (&f1, &f2)) in fit1.fitted.iter().zip(&fit2.fitted).enumerate() {
            assert!((f1 - f2).abs() < 1e-14, "fitted[{i}] differs: {f1} vs {f2}");
        }
    }

    // ── Test 22: All fitted values finite ─────────────────────────────────────

    #[test]
    fn all_fitted_values_finite() {
        let n = 100usize;
        let x: Vec<f64> = linspace(0.0, 1.0, n);
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| (2.0 * PI * xi).sin() + 0.01 * i as f64)
            .collect();
        let config = make_config_1term(7, 0.05);
        let fit = gam_fit(&x, &y, n, 1, &config).expect("fit");
        for (i, &v) in fit.fitted.iter().enumerate() {
            assert!(v.is_finite(), "fitted[{i}] = {v} is not finite");
        }
    }
}
