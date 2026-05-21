//! Heckman two-step sample-selection model (Heckman 1979).
//!
//! Heckman JJ. *Sample selection bias as a specification error.* Econometrica
//! 47(1): 153–161 (1979).
//!
//! # Problem
//!
//! We observe a continuous outcome `y_i` only when a binary *selection*
//! indicator `D_i = 1`. The structural equations are
//!
//! ```text
//!   y_i  = x_i^T β + u_i,                    (outcome equation)
//!   D_i  = 1 { z_i^T γ + v_i  > 0 },        (selection equation)
//!   (u_i, v_i) ~ N_2(0, Σ),    var(v_i)=1.
//! ```
//!
//! Naively running OLS on `y` over `{i : D_i = 1}` produces a biased
//! estimate of `β` because `E[u_i | D_i = 1] = ρ σ_u λ(z_i^T γ) ≠ 0`,
//! where `λ(t) = φ(t) / Φ(t)` is the inverse Mills ratio. Heckman's
//! two-step correction adds `λ̂_i` as a regressor in stage 2:
//!
//! ```text
//!   y_i = x_i^T β + (ρ σ_u) · λ̂_i + η_i,    i ∈ selected.
//! ```
//!
//! # Algorithm
//!
//! 1. **Stage 1.** Fit `P(D=1 | Z) = Φ(γ^T z)` via Newton-Raphson on the
//!    probit log-likelihood. We add a ridge to the Hessian for numerical
//!    stability and iterate until `|Δγ|_∞ < probit_tol` or
//!    `probit_max_iters` is reached.
//! 2. **Inverse Mills ratio.** For each selected `i`, compute
//!    `λ̂_i = φ(γ̂^T z_i) / Φ(γ̂^T z_i)`. Φ is approximated by the
//!    Abramowitz-Stegun 26.2.17 rational fit; we clip Φ to
//!    `[1e-12, 1 - 1e-12]` before forming the ratio.
//! 3. **Stage 2.** Solve `[1, x, λ̂] β̃ = y` by OLS (with the same ridge as
//!    stage 1) over the selected rows only.
//! 4. **σ_u estimate.** Residual MSE on the selected rows; the residual
//!    variance is reduced by `(n_sel - p)` degrees of freedom.
//! 5. **Robust standard errors.** White's heteroskedasticity-consistent
//!    sandwich `Var(β̂) = (X^T X)^{-1} X^T Ω X (X^T X)^{-1}` with
//!    `Ω = diag(residual_i^2)`.
//! 6. **Implied correlation.** `ρ̂ = lambda_coef / σ̂_u`, clipped to
//!    `(-0.9999, 0.9999)` to avoid degenerate variance reports.

use crate::error::{CausalError, CausalResult};

/// Configuration knobs for [`Heckman::estimate`].
#[derive(Clone, Debug)]
pub struct HeckmanConfig {
    /// Ridge added to the diagonal of `X^T X` and to the probit Hessian for
    /// numerical stability. Must be strictly positive (a tiny value like
    /// `1e-6` is fine).
    pub ridge_lambda: f64,
    /// Convergence tolerance on `|Δγ|_∞` for the stage-1 probit Newton step.
    /// Must be strictly positive.
    pub probit_tol: f64,
    /// Cap on Newton iterations for the stage-1 probit. Must be at least 1.
    pub probit_max_iters: usize,
}

impl Default for HeckmanConfig {
    fn default() -> Self {
        Self {
            ridge_lambda: 1e-6,
            probit_tol: 1e-6,
            probit_max_iters: 100,
        }
    }
}

/// Result of the Heckman two-step procedure.
#[derive(Clone, Debug)]
pub struct HeckmanResult {
    /// Stage-2 coefficients, in the order `[intercept, x_1, ..., x_d]`.
    /// Length is `d + 1`.
    pub beta: Vec<f64>,
    /// Coefficient on the inverse Mills ratio `λ̂` from stage 2; equal to
    /// `ρ · σ_u` under the bivariate-normal assumption.
    pub lambda_coef: f64,
    /// Implied correlation `ρ̂ = lambda_coef / σ̂_u`, clipped to
    /// `(-0.9999, 0.9999)`.
    pub rho: f64,
    /// Residual standard deviation `σ̂_u` estimated on the selected sample
    /// with `(n_sel − p)` degrees of freedom.
    pub sigma_e: f64,
    /// Heteroskedasticity-consistent (White) standard errors for `beta`.
    /// Length is `d + 1`.
    pub se: Vec<f64>,
    /// Number of selected observations (`Σ_i D_i = 1`).
    pub n_selected: usize,
}

/// Stateless namespace for the Heckman estimator.
pub struct Heckman;

impl Heckman {
    /// Estimate `(β, λ_coef, ρ, σ_u)` from observed `(y, selected, x, z)`.
    ///
    /// `y` is only used at indices where `selected[i] = true`. Both `x`
    /// and `z` are row-oriented: `x[i]` is the outcome-covariate vector for
    /// observation `i` (length `d_x`), and `z[i]` the selection-covariate
    /// vector (length `d_z`). Intercepts are added internally.
    pub fn estimate(
        y: &[f64],
        selected: &[bool],
        x: &[Vec<f64>],
        z: &[Vec<f64>],
        cfg: &HeckmanConfig,
    ) -> CausalResult<HeckmanResult> {
        validate(y, selected, x, z, cfg)?;
        let n = y.len();
        let d_x = x[0].len();
        let d_z = z[0].len();

        // --- Stage 1: probit on D | Z (with intercept in Z) ---
        let dz_p1 = d_z + 1;
        let z_aug = augment_with_intercept(z, n, d_z);
        let d_vec: Vec<f64> = selected
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect();
        let gamma = probit_newton(&z_aug, &d_vec, n, dz_p1, cfg)?;

        // --- Stage 2: form selected sub-design [1, x, λ̂] and solve OLS ---
        let n_sel = selected.iter().filter(|&&b| b).count();
        if n_sel < 2 {
            return Err(CausalError::EmptyInput);
        }
        let p = d_x + 2; // intercept + d_x + lambda
        if n_sel < p {
            return Err(CausalError::EmptyInput);
        }

        let mut x_sel = vec![0.0_f64; n_sel * p];
        let mut y_sel = vec![0.0_f64; n_sel];
        let mut row = 0_usize;
        for i in 0..n {
            if !selected[i] {
                continue;
            }
            x_sel[row * p] = 1.0; // intercept
            for j in 0..d_x {
                x_sel[row * p + 1 + j] = x[i][j];
            }
            // Inverse Mills ratio at the selected observation.
            let mut linear = gamma[d_z]; // intercept term
            for j in 0..d_z {
                linear += gamma[j] * z[i][j];
            }
            x_sel[row * p + 1 + d_x] = inverse_mills_ratio(linear);
            y_sel[row] = y[i];
            row += 1;
        }

        let coeffs = ols_ridge(&x_sel, &y_sel, n_sel, p, cfg.ridge_lambda)?;

        // beta is the (intercept + x) portion; lambda_coef is the last.
        let lambda_coef = coeffs[p - 1];
        let beta_full: Vec<f64> = coeffs[..p - 1].to_vec();

        // --- σ_u via residual MSE ---
        let mut sse = 0.0_f64;
        for r in 0..n_sel {
            let mut pred = 0.0_f64;
            for k in 0..p {
                pred += x_sel[r * p + k] * coeffs[k];
            }
            let residual = y_sel[r] - pred;
            sse += residual * residual;
        }
        let df = if n_sel > p { n_sel - p } else { 1 };
        let sigma_e = (sse / df as f64).sqrt().max(1e-12);

        // --- robust (White) standard errors ---
        let se = white_se(&x_sel, &y_sel, &coeffs, n_sel, p, cfg.ridge_lambda)?;
        // We keep only the SE for [intercept, x_1, ..., x_d]; the λ_coef SE
        // is not part of the public contract.
        let se_beta: Vec<f64> = se[..p - 1].to_vec();

        let rho = (lambda_coef / sigma_e).clamp(-0.9999, 0.9999);

        Ok(HeckmanResult {
            beta: beta_full,
            lambda_coef,
            rho,
            sigma_e,
            se: se_beta,
            n_selected: n_sel,
        })
    }
}

// ---------------------------------------------------------------------------
// validation
// ---------------------------------------------------------------------------

fn validate(
    y: &[f64],
    selected: &[bool],
    x: &[Vec<f64>],
    z: &[Vec<f64>],
    cfg: &HeckmanConfig,
) -> CausalResult<()> {
    if !(cfg.ridge_lambda.is_finite() && cfg.ridge_lambda > 0.0) {
        return Err(CausalError::IncompatibleData);
    }
    if !(cfg.probit_tol.is_finite() && cfg.probit_tol > 0.0) {
        return Err(CausalError::IncompatibleData);
    }
    if cfg.probit_max_iters == 0 {
        return Err(CausalError::IncompatibleData);
    }
    if y.is_empty() || selected.is_empty() || x.is_empty() || z.is_empty() {
        return Err(CausalError::EmptyInput);
    }
    let n = y.len();
    if selected.len() != n {
        return Err(CausalError::DimensionMismatch {
            expected: n,
            got: selected.len(),
        });
    }
    if x.len() != n {
        return Err(CausalError::DimensionMismatch {
            expected: n,
            got: x.len(),
        });
    }
    if z.len() != n {
        return Err(CausalError::DimensionMismatch {
            expected: n,
            got: z.len(),
        });
    }
    let d_x = x[0].len();
    let d_z = z[0].len();
    if d_x == 0 || d_z == 0 {
        return Err(CausalError::EmptyInput);
    }
    for row in x.iter() {
        if row.len() != d_x {
            return Err(CausalError::DimensionMismatch {
                expected: d_x,
                got: row.len(),
            });
        }
        for &v in row.iter() {
            if !v.is_finite() {
                return Err(CausalError::IncompatibleData);
            }
        }
    }
    for row in z.iter() {
        if row.len() != d_z {
            return Err(CausalError::DimensionMismatch {
                expected: d_z,
                got: row.len(),
            });
        }
        for &v in row.iter() {
            if !v.is_finite() {
                return Err(CausalError::IncompatibleData);
            }
        }
    }
    for (i, &b) in selected.iter().enumerate() {
        if b && !y[i].is_finite() {
            return Err(CausalError::IncompatibleData);
        }
    }
    let n_sel = selected.iter().filter(|&&b| b).count();
    if n_sel < 2 {
        return Err(CausalError::EmptyInput);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// numerical helpers
// ---------------------------------------------------------------------------

/// Standard-normal probability density `φ(z) = (1/√(2π)) · e^{-z²/2}`.
fn phi(z: f64) -> f64 {
    let inv_sqrt_two_pi = 0.398_942_280_401_432_7_f64;
    inv_sqrt_two_pi * (-0.5 * z * z).exp()
}

/// Standard-normal cumulative `Φ(z)` via Abramowitz-Stegun 26.2.17. Maximum
/// absolute error is roughly `7.5e-8`. We exploit the symmetry
/// `Φ(-z) = 1 − Φ(z)` so the rational fit is only evaluated for `z ≥ 0`.
fn cap_phi(z: f64) -> f64 {
    let sign = if z >= 0.0 { 1.0_f64 } else { -1.0_f64 };
    let abs_z = z.abs();
    let p = 0.231_641_900_0_f64;
    let b1 = 0.319_381_530_0_f64;
    let b2 = -0.356_563_782_0_f64;
    let b3 = 1.781_477_937_0_f64;
    let b4 = -1.821_255_978_0_f64;
    let b5 = 1.330_274_429_0_f64;
    let t = 1.0 / (1.0 + p * abs_z);
    let poly = b1 * t + b2 * t * t + b3 * t.powi(3) + b4 * t.powi(4) + b5 * t.powi(5);
    let pdf = phi(abs_z);
    let cdf_pos = 1.0 - pdf * poly;
    if sign > 0.0 { cdf_pos } else { 1.0 - cdf_pos }
}

/// `λ(t) = φ(t) / Φ(t)`. Φ is clipped to `[1e-12, 1 - 1e-12]` to avoid the
/// degeneracy at the far-left tail of the selection distribution.
fn inverse_mills_ratio(t: f64) -> f64 {
    let cap = cap_phi(t).clamp(1e-12, 1.0 - 1e-12);
    phi(t) / cap
}

/// Augment a row-oriented covariate matrix with a trailing intercept column.
/// Output is row-major `n × (d + 1)`.
fn augment_with_intercept(rows: &[Vec<f64>], n: usize, d: usize) -> Vec<f64> {
    let p = d + 1;
    let mut out = vec![0.0_f64; n * p];
    for (i, row) in rows.iter().enumerate() {
        for j in 0..d {
            out[i * p + j] = row[j];
        }
        out[i * p + d] = 1.0; // intercept last so γ[d] is the intercept coef
    }
    out
}

/// Probit Newton-Raphson on `D | Z`. Returns the parameter vector γ of
/// length `dz_p1` (covariates first, intercept last).
fn probit_newton(
    z: &[f64],
    d_vec: &[f64],
    n: usize,
    dz_p1: usize,
    cfg: &HeckmanConfig,
) -> CausalResult<Vec<f64>> {
    let mut gamma = vec![0.0_f64; dz_p1];
    // Seed the intercept with logit(π̂) where π̂ = mean(D), to avoid
    // pathological starts when D is near constant.
    let mean_d = d_vec.iter().sum::<f64>() / n as f64;
    let p0 = mean_d.clamp(1e-6, 1.0 - 1e-6);
    // Approximate Φ⁻¹(p0): probit inverse via Newton on Φ; cheap closed-form
    // surrogate `ln(p0 / (1 - p0)) / 1.6` is good enough for an initial seed.
    gamma[dz_p1 - 1] = (p0 / (1.0 - p0)).ln() / 1.6;

    for _ in 0..cfg.probit_max_iters {
        let mut grad = vec![0.0_f64; dz_p1];
        let mut hess = vec![0.0_f64; dz_p1 * dz_p1];
        let mut prev_log_lik = 0.0_f64;
        for i in 0..n {
            let mut linear = 0.0_f64;
            for k in 0..dz_p1 {
                linear += z[i * dz_p1 + k] * gamma[k];
            }
            let phi_v = phi(linear);
            let cap_v = cap_phi(linear).clamp(1e-12, 1.0 - 1e-12);
            let one_m_cap = (1.0 - cap_v).max(1e-12);
            let d_i = d_vec[i];
            let factor = d_i * phi_v / cap_v - (1.0 - d_i) * phi_v / one_m_cap;
            // Hessian weight: derivative of `factor` w.r.t. `linear`.
            let w_d1 = d_i * (phi_v * phi_v / (cap_v * cap_v) + linear * phi_v / cap_v);
            let w_d0 = (1.0 - d_i)
                * (phi_v * phi_v / (one_m_cap * one_m_cap) - linear * phi_v / one_m_cap);
            let h_weight = w_d1 + w_d0;
            prev_log_lik += d_i * cap_v.ln() + (1.0 - d_i) * one_m_cap.ln();
            for k in 0..dz_p1 {
                grad[k] += factor * z[i * dz_p1 + k];
                for l in 0..dz_p1 {
                    hess[k * dz_p1 + l] += h_weight * z[i * dz_p1 + k] * z[i * dz_p1 + l];
                }
            }
        }
        let _ = prev_log_lik;
        // Ridge for numerical stability.
        for k in 0..dz_p1 {
            hess[k * dz_p1 + k] += cfg.ridge_lambda;
        }
        let delta =
            gauss_jordan_solve(&hess, &grad, dz_p1).map_err(|_| CausalError::IncompatibleData)?;
        let mut max_abs = 0.0_f64;
        for (k, gamma_k) in gamma.iter_mut().enumerate() {
            *gamma_k += delta[k];
            if delta[k].abs() > max_abs {
                max_abs = delta[k].abs();
            }
        }
        if max_abs < cfg.probit_tol {
            break;
        }
    }
    Ok(gamma)
}

/// OLS with a tiny ridge: solve `(X^T X + λI) β = X^T y`. `x_mat` is row-
/// major `n × p`.
fn ols_ridge(x_mat: &[f64], y: &[f64], n: usize, p: usize, lambda: f64) -> CausalResult<Vec<f64>> {
    let mut xtx = vec![0.0_f64; p * p];
    let mut xty = vec![0.0_f64; p];
    for row in 0..n {
        for i in 0..p {
            let xri = x_mat[row * p + i];
            for j in 0..p {
                xtx[i * p + j] += xri * x_mat[row * p + j];
            }
            xty[i] += xri * y[row];
        }
    }
    for k in 0..p {
        xtx[k * p + k] += lambda;
    }
    gauss_jordan_solve(&xtx, &xty, p)
}

/// White heteroskedasticity-consistent standard errors for the ridge-OLS
/// fit. `coeffs` are the previously solved β. Returns `sqrt(diag(Var))`.
fn white_se(
    x_mat: &[f64],
    y: &[f64],
    coeffs: &[f64],
    n: usize,
    p: usize,
    lambda: f64,
) -> CausalResult<Vec<f64>> {
    // residuals
    let mut residual_sq = vec![0.0_f64; n];
    for row in 0..n {
        let mut pred = 0.0_f64;
        for k in 0..p {
            pred += x_mat[row * p + k] * coeffs[k];
        }
        let r = y[row] - pred;
        residual_sq[row] = r * r;
    }
    // X^T X + λI
    let mut xtx = vec![0.0_f64; p * p];
    for row in 0..n {
        for i in 0..p {
            let xri = x_mat[row * p + i];
            for j in 0..p {
                xtx[i * p + j] += xri * x_mat[row * p + j];
            }
        }
    }
    for k in 0..p {
        xtx[k * p + k] += lambda;
    }
    let xtx_inv = gauss_jordan_invert(&xtx, p)?;
    // meat = X^T Ω X
    let mut meat = vec![0.0_f64; p * p];
    for row in 0..n {
        let r2 = residual_sq[row];
        for i in 0..p {
            let xri = x_mat[row * p + i];
            for j in 0..p {
                meat[i * p + j] += r2 * xri * x_mat[row * p + j];
            }
        }
    }
    // var = (xtx_inv) · meat · (xtx_inv)
    let mut tmp = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in 0..p {
            let mut s = 0.0_f64;
            for k in 0..p {
                s += xtx_inv[i * p + k] * meat[k * p + j];
            }
            tmp[i * p + j] = s;
        }
    }
    let mut var = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in 0..p {
            let mut s = 0.0_f64;
            for k in 0..p {
                s += tmp[i * p + k] * xtx_inv[k * p + j];
            }
            var[i * p + j] = s;
        }
    }
    let mut se = vec![0.0_f64; p];
    for k in 0..p {
        let v = var[k * p + k].max(0.0);
        se[k] = v.sqrt();
    }
    Ok(se)
}

/// Gauss-Jordan solve `A β = b` with partial pivoting.
fn gauss_jordan_solve(a: &[f64], b: &[f64], p: usize) -> CausalResult<Vec<f64>> {
    let cols = p + 1;
    let mut m = vec![0.0_f64; p * cols];
    for i in 0..p {
        for j in 0..p {
            m[i * cols + j] = a[i * p + j];
        }
        m[i * cols + p] = b[i];
    }
    for col in 0..p {
        let mut piv = col;
        let mut best = m[col * cols + col].abs();
        for r in (col + 1)..p {
            let v = m[r * cols + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-15 {
            return Err(CausalError::MatrixSingular);
        }
        if piv != col {
            for k in 0..cols {
                m.swap(col * cols + k, piv * cols + k);
            }
        }
        let pv = m[col * cols + col];
        for k in 0..cols {
            m[col * cols + k] /= pv;
        }
        for r in 0..p {
            if r == col {
                continue;
            }
            let f = m[r * cols + col];
            if f.abs() < 1e-18 {
                continue;
            }
            for k in 0..cols {
                let v = m[col * cols + k];
                m[r * cols + k] -= f * v;
            }
        }
    }
    let mut x = vec![0.0_f64; p];
    for i in 0..p {
        x[i] = m[i * cols + p];
    }
    Ok(x)
}

/// Gauss-Jordan inversion. Used by [`white_se`].
fn gauss_jordan_invert(a: &[f64], n: usize) -> CausalResult<Vec<f64>> {
    let mut m = vec![0.0_f64; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            m[i * 2 * n + j] = a[i * n + j];
        }
        m[i * 2 * n + n + i] = 1.0;
    }
    for col in 0..n {
        let mut pivot = col;
        for row in (col + 1)..n {
            if m[row * 2 * n + col].abs() > m[pivot * 2 * n + col].abs() {
                pivot = row;
            }
        }
        if m[pivot * 2 * n + col].abs() < 1e-14 {
            return Err(CausalError::MatrixSingular);
        }
        if pivot != col {
            for k in 0..(2 * n) {
                m.swap(col * 2 * n + k, pivot * 2 * n + k);
            }
        }
        let div = m[col * 2 * n + col];
        for k in 0..(2 * n) {
            m[col * 2 * n + k] /= div;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = m[row * 2 * n + col];
            if factor.abs() < 1e-18 {
                continue;
            }
            for k in 0..(2 * n) {
                let v = m[col * 2 * n + k] * factor;
                m[row * 2 * n + k] -= v;
            }
        }
    }
    let mut inv = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = m[i * 2 * n + n + j];
        }
    }
    Ok(inv)
}
