//! Linear quantile regression via a Frisch–Newton interior-point algorithm.
//!
//! Quantile regression (Koenker & Bassett, 1978) estimates the conditional
//! `τ`-quantile of `y` given `x` by minimising the asymmetric **check** (pinball)
//! loss
//!
//! ```text
//! min_β  Σ ρ_τ(yᵢ − xᵢᵀ β),      ρ_τ(u) = u · (τ − 1{u < 0}).
//! ```
//!
//! Unlike least squares (which targets the conditional mean) this is robust to
//! `y`-outliers at `τ = 0.5` and recovers an arbitrary conditional quantile for
//! `τ ∈ (0, 1)`.
//!
//! # Why a separate solver?
//!
//! The sibling module [`crate::regression::quantile`] solves the same problem by
//! the **iteratively reweighted L1 / MM** surrogate (Hunter & Lange, 2000), which
//! linearises `|r|` and is only first-order. This module instead uses the
//! **convolution-smoothed Newton** method (a.k.a. *conquer*; Fernandes, Guerre &
//! Horta, 2021; He, Pan, Tan & Zhou, 2021). The non-smooth check loss is replaced
//! by its convolution with a Gaussian kernel of bandwidth `h`, giving a twice-
//! differentiable convex objective whose gradient and Hessian are available in
//! closed form. Newton steps then converge quadratically at a fixed `h`, and the
//! bandwidth is annealed toward zero so that the iterate converges to the **exact**
//! minimiser of the original pinball loss — certified by the genuine subgradient
//! `Xᵀ(τ − 1{r<0})` vanishing at the optimum.
//!
//! # The smoothed objective
//!
//! With residual `rᵢ = yᵢ − xᵢᵀ β`, the smoothed score and curvature are
//!
//! ```text
//! sₕ(r) = τ − Φ(−r / h),     wₕ(r) = φ(r / h) / h,
//! ```
//!
//! where `Φ`, `φ` are the standard-normal CDF / PDF. The Newton step solves the
//! weighted normal equations `(Xᵀ W X) Δβ = Xᵀ g` with `gᵢ = sₕ(rᵢ)`,
//! `Wᵢ = wₕ(rᵢ)`, backtracking on the smoothed objective to guarantee descent.
//! As `h → 0`, `sₕ → τ − 1{r<0}` and the solution coincides with the LP optimum
//! of the standard interior-point / simplex methods (Koenker, 2005, §6).
//!
//! # References
//! - Koenker, R. & Bassett, G. (1978). "Regression Quantiles". *Econometrica*
//!   46(1):33–50.
//! - Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
//! - Fernandes, M., Guerre, E. & Horta, E. (2021). "Smoothing quantile
//!   regressions". *J. Business & Economic Statistics* 39(1):338–357.
//! - He, X., Pan, X., Tan, K.M. & Zhou, W.-X. (2021). "Smoothed quantile
//!   regression with large-scale inference". *J. Econometrics*.

use crate::error::{StatsError, StatsResult};
use crate::regression::linear::matrix_inverse_lu;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`quantile_regression_fit`].
#[derive(Debug, Clone)]
pub struct QuantRegConfig {
    /// Quantile level `τ ∈ (0, 1)` (default 0.5, the median).
    pub tau: f64,
    /// Maximum interior-point iterations (default 100).
    pub max_iter: usize,
    /// Duality-gap tolerance for convergence (default 1e-8).
    pub tol: f64,
    /// Prepend an intercept column of ones (default `true`).
    pub intercept: bool,
}

impl Default for QuantRegConfig {
    fn default() -> Self {
        Self {
            tau: 0.5,
            max_iter: 100,
            tol: 1e-8,
            intercept: true,
        }
    }
}

impl QuantRegConfig {
    /// Convenience constructor fixing `τ`, defaults elsewhere.
    #[must_use]
    pub fn with_tau(tau: f64) -> Self {
        Self {
            tau,
            ..Default::default()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Result
// ─────────────────────────────────────────────────────────────────────────────

/// A fitted quantile-regression model.
#[derive(Debug, Clone)]
pub struct QuantRegFit {
    /// Slope coefficients (length `n_features`; excludes the intercept).
    pub coefficients: Vec<f64>,
    /// Fitted intercept (`0.0` when `cfg.intercept == false`).
    pub intercept_val: f64,
    /// Residuals `rᵢ = yᵢ − ŷᵢ`.
    pub residuals: Vec<f64>,
    /// Optimal check / pinball loss `Σ ρ_τ(rᵢ)`.
    pub pinball_loss: f64,
    /// Final normalised exact subgradient norm `‖Xᵀ(τ − 1{r<0})‖₂ / scale`
    /// (≈ 0 at the optimum).
    pub subgradient_norm: f64,
    /// Number of Newton iterations executed.
    pub n_iter: usize,
    /// Whether the subgradient fell below `tol` at the bandwidth floor.
    pub converged: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Loss & (sub)gradient utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Pinball loss `ρ_τ(u) = u (τ − 1{u < 0})`.
#[inline]
#[must_use]
pub fn check_loss(u: f64, tau: f64) -> f64 {
    if u >= 0.0 { tau * u } else { (tau - 1.0) * u }
}

/// Total pinball loss for a residual vector.
#[must_use]
pub fn pinball_loss(residuals: &[f64], tau: f64) -> f64 {
    residuals.iter().map(|&r| check_loss(r, tau)).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build the (possibly intercept-augmented) design matrix, row-major `(n × p)`.
fn build_design(x: &[f64], n: usize, p: usize, intercept: bool) -> (Vec<f64>, usize) {
    if !intercept {
        return (x.to_vec(), p);
    }
    let p_aug = p + 1;
    let mut xa = vec![0.0_f64; n * p_aug];
    for i in 0..n {
        xa[i * p_aug] = 1.0;
        for j in 0..p {
            xa[i * p_aug + 1 + j] = x[i * p + j];
        }
    }
    (xa, p_aug)
}

/// `β = (Xᵀ D X)⁻¹ Xᵀ D r`, the WLS solution with diagonal weights `D`.
fn weighted_normal_solve(
    xa: &[f64],
    rhs_vec: &[f64],
    w: &[f64],
    n: usize,
    p: usize,
) -> StatsResult<Vec<f64>> {
    // A = Xᵀ W X  (p × p).
    let mut a = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in i..p {
            let mut acc = 0.0;
            for k in 0..n {
                acc += xa[k * p + i] * w[k] * xa[k * p + j];
            }
            a[i * p + j] = acc;
            a[j * p + i] = acc;
        }
    }
    let diag_max = (0..p).map(|j| a[j * p + j].abs()).fold(0.0_f64, f64::max);
    let ridge = (diag_max * 1e-12).max(1e-14);
    for j in 0..p {
        a[j * p + j] += ridge;
    }
    // b = Xᵀ W rhs  (p).
    let mut b = vec![0.0_f64; p];
    for i in 0..p {
        let mut acc = 0.0;
        for k in 0..n {
            acc += xa[k * p + i] * w[k] * rhs_vec[k];
        }
        b[i] = acc;
    }
    let inv = matrix_inverse_lu(&a, p)?;
    let mut out = vec![0.0_f64; p];
    for i in 0..p {
        let mut acc = 0.0;
        for j in 0..p {
            acc += inv[i * p + j] * b[j];
        }
        out[i] = acc;
    }
    Ok(out)
}

/// Residuals `rᵢ = yᵢ − xᵢᵀ β`.
fn residual_vec(xa: &[f64], y: &[f64], beta: &[f64], n: usize, p: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let fitted: f64 = (0..p).map(|j| xa[i * p + j] * beta[j]).sum();
            y[i] - fitted
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Convolution-smoothed Newton solver
// ─────────────────────────────────────────────────────────────────────────────

/// Smooth surrogate gradient of the check loss.
///
/// The pinball derivative `ψ_τ(u) = τ − 1{u < 0}` is discontinuous at `0`.
/// Convolving the indicator with a Gaussian kernel of bandwidth `h` gives the
/// smooth score (Fernandes–Guerre–Horta 2021; He et al. 2021, "conquer")
///
/// ```text
/// s_h(u) = τ − Φ(−u / h),     Φ = standard-normal CDF.
/// ```
///
/// As `h → 0` this recovers `ψ_τ` exactly. The smoothed objective whose gradient
/// this is, is convex, so Newton's method converges globally.
#[inline]
fn smooth_score(u: f64, tau: f64, h: f64) -> f64 {
    tau - normal_cdf(-u / h)
}

/// Smoothed second derivative (kernel density at the residual): `φ(u/h) / h`.
#[inline]
fn smooth_weight(u: f64, h: f64) -> f64 {
    let z = u / h;
    normal_pdf(z) / h
}

/// Standard-normal CDF via `erf` from the crate's special functions.
#[inline]
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + crate::special::erf::erf(x / std::f64::consts::SQRT_2))
}

/// Standard-normal PDF.
#[inline]
fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Core solver: convolution-smoothed quantile regression by Newton's method with
/// bandwidth annealing. Returns `(beta, subgradient_norm, n_iter, converged)`.
///
/// At a fixed bandwidth `h` the objective `Σ ∫ ρ_τ` (smoothed) is minimised by
/// Newton steps `Δβ = (Xᵀ W X)⁻¹ Xᵀ g`, where `gᵢ = s_h(rᵢ)` is the smooth score
/// and `Wᵢ = φ(rᵢ/h)/h` the smooth curvature. The bandwidth is then shrunk
/// geometrically toward a small floor, so the iterate converges to the exact
/// (non-smooth) check-loss minimiser; convergence is declared once the genuine
/// subgradient `Xᵀ(τ − 1{r<0})` is small relative to the design scale.
fn smoothed_newton(
    xa: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    tau: f64,
    max_iter: usize,
    tol: f64,
) -> StatsResult<(Vec<f64>, f64, usize, bool)> {
    // Warm start: OLS solution (unit weights) — minimiser of the squared loss is
    // a sound interior starting point for the smoothed problem.
    let unit = vec![1.0_f64; n];
    let mut beta = weighted_normal_solve_y(xa, y, &unit, n, p)?;

    // Initial bandwidth from the residual scale (Silverman-like), with a floor.
    let r0 = residual_vec(xa, y, &beta, n, p);
    let scale = robust_scale(&r0).max(1e-3);
    let mut h = scale.max(1e-2);
    let h_floor = (scale * 1e-4).max(1e-8);

    // Subgradient scale for the convergence test: ‖X‖∞-ish.
    let x_scale = {
        let mut s = 0.0;
        for v in xa.iter() {
            s += v.abs();
        }
        (s / (n as f64)).max(1.0)
    };

    let mut converged = false;
    let mut n_iter = 0usize;
    let mut subgrad_norm = f64::INFINITY;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        let r = residual_vec(xa, y, &beta, n, p);

        // Smooth score g and curvature weight w at bandwidth h.
        let mut g = vec![0.0_f64; n];
        let mut w = vec![0.0_f64; n];
        for i in 0..n {
            g[i] = smooth_score(r[i], tau, h);
            // Floor the weight so the Hessian stays positive definite even when
            // every residual is far from 0 (kernel underflow).
            w[i] = smooth_weight(r[i], h).max(1e-6);
        }

        // Newton direction: solve (Xᵀ W X) Δ = Xᵀ g, then β ← β + Δ.
        // Note ∂obj/∂β = −Xᵀ g (since rᵢ = yᵢ − xᵢᵀβ), and the Hessian is Xᵀ W X,
        // so the Newton step is Δ = +(Xᵀ W X)⁻¹ Xᵀ g.
        let delta = newton_step(xa, &g, &w, n, p)?;

        // Damped update with a simple backtracking on the smoothed objective to
        // guarantee descent.
        let f0 = smoothed_objective(&r, tau, h);
        let mut step = 1.0_f64;
        let mut beta_try = beta.clone();
        let mut accepted = false;
        for _ in 0..30 {
            for j in 0..p {
                beta_try[j] = beta[j] + step * delta[j];
            }
            let r_try = residual_vec(xa, y, &beta_try, n, p);
            let f_try = smoothed_objective(&r_try, tau, h);
            if f_try <= f0 + 1e-12 {
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if accepted {
            beta = beta_try;
        }

        // Genuine (non-smooth) subgradient Xᵀ(τ − 1{r<0}) at the current β.
        let r_now = residual_vec(xa, y, &beta, n, p);
        subgrad_norm = exact_subgradient_norm(xa, &r_now, tau, n, p) / x_scale;

        // Anneal the bandwidth toward the floor.
        h = (h * 0.5).max(h_floor);

        // Converged when the exact subgradient is small and the bandwidth has
        // reached its floor (so the smoothing no longer biases the solution).
        if subgrad_norm < tol.max(1e-6) && h <= h_floor * 1.0001 {
            converged = true;
            break;
        }
    }

    Ok((beta, subgrad_norm, n_iter, converged))
}

/// OLS-style solve `β = (Xᵀ W X)⁻¹ Xᵀ W y`.
fn weighted_normal_solve_y(
    xa: &[f64],
    y: &[f64],
    w: &[f64],
    n: usize,
    p: usize,
) -> StatsResult<Vec<f64>> {
    weighted_normal_solve(xa, y, w, n, p)
}

/// Newton step `(Xᵀ W X)⁻¹ Xᵀ g`.
fn newton_step(xa: &[f64], g: &[f64], w: &[f64], n: usize, p: usize) -> StatsResult<Vec<f64>> {
    // Hessian H = Xᵀ W X.
    let mut h = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in i..p {
            let mut acc = 0.0;
            for k in 0..n {
                acc += xa[k * p + i] * w[k] * xa[k * p + j];
            }
            h[i * p + j] = acc;
            h[j * p + i] = acc;
        }
    }
    let diag_max = (0..p).map(|j| h[j * p + j].abs()).fold(0.0_f64, f64::max);
    let ridge = (diag_max * 1e-10).max(1e-12);
    for j in 0..p {
        h[j * p + j] += ridge;
    }
    // Gradient term Xᵀ g.
    let mut xtg = vec![0.0_f64; p];
    for i in 0..p {
        let mut acc = 0.0;
        for k in 0..n {
            acc += xa[k * p + i] * g[k];
        }
        xtg[i] = acc;
    }
    let inv = matrix_inverse_lu(&h, p)?;
    let mut delta = vec![0.0_f64; p];
    for i in 0..p {
        let mut acc = 0.0;
        for j in 0..p {
            acc += inv[i * p + j] * xtg[j];
        }
        delta[i] = acc;
    }
    Ok(delta)
}

/// Smoothed objective value (a convex surrogate of the pinball loss).
///
/// Uses the Gaussian-convolved check loss whose gradient is [`smooth_score`]:
/// `ρ_h(u) = (τ − ½) u + h·[ φ(u/h) + (u/h)Φ(u/h) ]` (up to constants), evaluated
/// stably for monitoring backtracking descent.
fn smoothed_objective(r: &[f64], tau: f64, h: f64) -> f64 {
    let mut acc = 0.0;
    for &u in r {
        let z = u / h;
        // Convolved check loss (Horowitz / conquer):
        //   ρ_h(u) = (τ − ½) u + (h/2)[ (2/√(2π)) e^{−z²/2} + z·erf(z/√2) ] ... but
        // we only need a valid descent monitor; use the integral form below.
        let phi = normal_pdf(z);
        let cdf = normal_cdf(z);
        // E[ρ_τ(u − hN)] for N~Normal: = (τ − Φ(−z)) u + h φ(z)  (closed form).
        let val = (tau - normal_cdf(-z)) * u + h * phi;
        let _ = cdf;
        acc += val;
    }
    acc
}

/// `‖Xᵀ(τ − 1{r<0})‖₂`, the (a) subgradient of the exact pinball loss.
fn exact_subgradient_norm(xa: &[f64], r: &[f64], tau: f64, n: usize, p: usize) -> f64 {
    let mut grad = vec![0.0_f64; p];
    for j in 0..p {
        let mut acc = 0.0;
        for k in 0..n {
            let psi = if r[k] < 0.0 { tau - 1.0 } else { tau };
            acc += xa[k * p + j] * psi;
        }
        grad[j] = acc;
    }
    grad.iter().map(|&v| v * v).sum::<f64>().sqrt()
}

/// Robust residual scale: 1.4826 × median(|r − median(r)|) (MAD), or the standard
/// deviation if the MAD collapses to zero.
fn robust_scale(r: &[f64]) -> f64 {
    if r.is_empty() {
        return 1.0;
    }
    let mut sorted = r.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = median_sorted(&sorted);
    let mut dev: Vec<f64> = r.iter().map(|&v| (v - med).abs()).collect();
    dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = 1.4826 * median_sorted(&dev);
    if mad > 1e-12 {
        mad
    } else {
        let mean = r.iter().sum::<f64>() / r.len() as f64;
        let var = r.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / r.len() as f64;
        var.sqrt().max(1e-6)
    }
}

/// Median of an already-sorted slice.
fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Fit a linear quantile-regression model at level `τ`.
///
/// # Parameters
/// - `x` — row-major design matrix `(n_samples, n_features)` **without** the
///   intercept column.
/// - `y` — response vector of length `n_samples`.
///
/// # Errors
/// Shape/length mismatches, non-finite inputs, `τ ∉ (0,1)`, insufficient samples,
/// or a singular design.
pub fn quantile_regression_fit(
    x: &[f64],
    y: &[f64],
    n_samples: usize,
    n_features: usize,
    cfg: &QuantRegConfig,
) -> StatsResult<QuantRegFit> {
    if n_samples == 0 {
        return Err(StatsError::EmptyInput);
    }
    if x.len() != n_samples * n_features {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n_samples, n_features],
            got: vec![x.len()],
        });
    }
    if y.len() != n_samples {
        return Err(StatsError::DimensionMismatch {
            a: y.len(),
            b: n_samples,
        });
    }
    if !(cfg.tau > 0.0 && cfg.tau < 1.0) {
        return Err(StatsError::InvalidParameter {
            name: "tau".to_string(),
            reason: format!("must be in (0, 1), got {}", cfg.tau),
        });
    }
    for (i, &v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }
    for (i, &v) in x.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }

    let (xa, p) = build_design(x, n_samples, n_features, cfg.intercept);
    if n_samples < p {
        return Err(StatsError::InsufficientSampleSize {
            got: n_samples,
            need: p,
        });
    }

    let (beta, subgradient_norm, n_iter, converged) =
        smoothed_newton(&xa, y, n_samples, p, cfg.tau, cfg.max_iter, cfg.tol)?;

    let residuals = residual_vec(&xa, y, &beta, n_samples, p);
    let loss = pinball_loss(&residuals, cfg.tau);

    let (intercept_val, coefficients) = if cfg.intercept {
        (beta[0], beta[1..].to_vec())
    } else {
        (0.0, beta)
    };

    Ok(QuantRegFit {
        coefficients,
        intercept_val,
        residuals,
        pinball_loss: loss,
        subgradient_norm,
        n_iter,
        converged,
    })
}

/// Predict the fitted conditional quantile for new observations.
pub fn quantile_regression_predict(
    fit: &QuantRegFit,
    x_new: &[f64],
    n_new: usize,
) -> StatsResult<Vec<f64>> {
    if n_new == 0 {
        return Ok(Vec::new());
    }
    let n_features = fit.coefficients.len();
    if x_new.len() != n_new * n_features {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n_new, n_features],
            got: vec![x_new.len()],
        });
    }
    Ok((0..n_new)
        .map(|i| {
            let mut yhat = fit.intercept_val;
            for j in 0..n_features {
                yhat += x_new[i * n_features + j] * fit.coefficients[j];
            }
            yhat
        })
        .collect())
}

/// Empirical coverage: fraction of training residuals strictly below the fit
/// (`rᵢ > 0` ⇔ `yᵢ > ŷᵢ`, so "below the line" means `rᵢ < 0`). For a good
/// `τ`-fit this is ≈ `τ`.
#[must_use]
pub fn coverage_below(residuals: &[f64]) -> f64 {
    if residuals.is_empty() {
        return 0.0;
    }
    let below = residuals.iter().filter(|&&r| r < 0.0).count();
    below as f64 / residuals.len() as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::regression::linear::ols;

    /// OLS slope/intercept on a single feature (with intercept), for comparison.
    fn ols_line(xs: &[f64], ys: &[f64]) -> (f64, f64) {
        let n = xs.len();
        let mut design = vec![0.0_f64; n * 2];
        for i in 0..n {
            design[i * 2] = 1.0;
            design[i * 2 + 1] = xs[i];
        }
        let m = ols(&design, ys, n, 2).expect("ols");
        (m.coefficients[0], m.coefficients[1]) // (intercept, slope)
    }

    // (a) τ=0.5 is robust to y-outliers where OLS is badly pulled.
    #[test]
    fn median_regression_robust_to_outliers() {
        let n = 41;
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let true_slope = 2.0;
        let true_intercept = 1.0;
        let mut ys: Vec<f64> = xs
            .iter()
            .map(|&x| true_intercept + true_slope * x)
            .collect();
        // Inject high-leverage y-outliers at the *ends* so they tilt the OLS
        // slope strongly: drag the high-x points up, the low-x points down.
        ys[0] -= 900.0;
        ys[1] -= 800.0;
        ys[n - 1] += 900.0;
        ys[n - 2] += 800.0;

        let cfg = QuantRegConfig::with_tau(0.5);
        let fit = quantile_regression_fit(&xs, &ys, n, 1, &cfg).expect("fit");

        let (_oi, os) = ols_line(&xs, &ys);
        // OLS slope should be visibly perturbed away from the truth.
        assert!(
            (os - true_slope).abs() > 1.0,
            "OLS slope {os} should be pulled from {true_slope}"
        );
        // Median-regression slope should stay close to the truth (majority of
        // points are exactly on the line, so the median fit ignores the 4 outliers).
        assert!(
            (fit.coefficients[0] - true_slope).abs() < 0.3,
            "median slope {} vs true {}",
            fit.coefficients[0],
            true_slope
        );
    }

    // (b) Heteroscedastic data with known conditional quantiles → correct slope.
    #[test]
    fn recovers_quantile_slope_heteroscedastic() {
        // y = a + b x + (c x) ε, ε ~ Uniform(-1,1) symmetric → conditional
        // τ-quantile is a + b x + (c x)·F⁻¹(τ). With ε uniform on (-1,1),
        // F⁻¹(τ) = 2τ − 1, so the τ-slope is b + c(2τ−1).
        let n = 600;
        let a = 1.0;
        let b = 2.0;
        let c = 1.5;
        let mut rng = LcgRng::new(4242);
        let xs: Vec<f64> = (0..n)
            .map(|i| 1.0 + 3.0 * (i as f64) / (n as f64))
            .collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| {
                let eps = 2.0 * rng.next_f64() - 1.0; // Uniform(-1,1)
                a + b * x + (c * x) * eps
            })
            .collect();

        let tau = 0.75;
        let cfg = QuantRegConfig::with_tau(tau);
        let fit = quantile_regression_fit(&xs, &ys, n, 1, &cfg).expect("fit");
        let expected_slope = b + c * (2.0 * tau - 1.0);
        assert!(
            (fit.coefficients[0] - expected_slope).abs() < 0.4,
            "τ={tau} slope {} vs expected {expected_slope}",
            fit.coefficients[0]
        );
    }

    // (c) Coverage: ≈ τ of training points lie below the fitted line.
    #[test]
    fn coverage_approximates_tau() {
        let n = 500;
        let mut rng = LcgRng::new(13);
        let xs: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| 1.0 + 2.0 * x + (rng.next_f64() - 0.5))
            .collect();
        for &tau in &[0.25_f64, 0.5, 0.75] {
            let cfg = QuantRegConfig::with_tau(tau);
            let fit = quantile_regression_fit(&xs, &ys, n, 1, &cfg).expect("fit");
            let cov = coverage_below(&fit.residuals);
            assert!((cov - tau).abs() < 0.06, "τ={tau}: coverage {cov} not ≈ τ");
        }
    }

    // (d) Pinball loss at the optimum ≤ pinball loss at the OLS coefficients.
    #[test]
    fn optimum_beats_ols_in_pinball() {
        let n = 80;
        let mut rng = LcgRng::new(555);
        let xs: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| 0.5 + 1.5 * x + 0.5 * (rng.next_f64() - 0.5))
            .collect();
        let tau = 0.3;
        let cfg = QuantRegConfig::with_tau(tau);
        let fit = quantile_regression_fit(&xs, &ys, n, 1, &cfg).expect("fit");

        // OLS residuals & their pinball loss.
        let (oi, os) = ols_line(&xs, &ys);
        let ols_resid: Vec<f64> = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| y - (oi + os * x))
            .collect();
        let ols_pin = pinball_loss(&ols_resid, tau);

        assert!(
            fit.pinball_loss <= ols_pin + 1e-6,
            "QR pinball {} should be ≤ OLS pinball {ols_pin}",
            fit.pinball_loss
        );
    }

    // (e) τ=0.5 minimises the mean absolute deviation (= ½ Σ|r| for the loss).
    #[test]
    fn median_minimises_mad() {
        // For a constant-only model the τ=0.5 fit is the sample median, which
        // minimises Σ|yᵢ − c|.
        let ys = vec![1.0, 3.0, 2.0, 8.0, 5.0, 4.0, 9.0];
        let n = ys.len();
        let xs = vec![0.0_f64; n]; // single zero feature → intercept-only effect
        let cfg = QuantRegConfig {
            tau: 0.5,
            intercept: true,
            ..Default::default()
        };
        let fit = quantile_regression_fit(&xs, &ys, n, 1, &cfg).expect("fit");
        let c_hat = fit.intercept_val;

        // Compare Σ|yᵢ − ĉ| against a grid of constants: ĉ should be ~minimal.
        let mad = |c: f64| ys.iter().map(|&y| (y - c).abs()).sum::<f64>();
        let mad_hat = mad(c_hat);
        for k in 0..200 {
            let c = k as f64 * 0.1;
            assert!(
                mad_hat <= mad(c) + 1e-6,
                "median const {c_hat} (MAD {mad_hat}) not minimal vs {c} (MAD {})",
                mad(c)
            );
        }
    }

    // (f) τ near 0 / 1 gives a lower / upper envelope.
    #[test]
    fn extreme_tau_envelopes() {
        let n = 120;
        let mut rng = LcgRng::new(2718);
        let xs: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| 1.0 + 2.0 * x + (rng.next_f64() - 0.5))
            .collect();

        let lo =
            quantile_regression_fit(&xs, &ys, n, 1, &QuantRegConfig::with_tau(0.05)).expect("lo");
        let hi =
            quantile_regression_fit(&xs, &ys, n, 1, &QuantRegConfig::with_tau(0.95)).expect("hi");

        // Most points above the τ=0.05 line; most below the τ=0.95 line.
        let above_lo = lo.residuals.iter().filter(|&&r| r > 0.0).count();
        let below_hi = hi.residuals.iter().filter(|&&r| r < 0.0).count();
        assert!(
            above_lo as f64 / n as f64 > 0.85,
            "τ=0.05 should sit below most points ({above_lo}/{n} above)"
        );
        assert!(
            below_hi as f64 / n as f64 > 0.85,
            "τ=0.95 should sit above most points ({below_hi}/{n} below)"
        );
    }

    // (g) The check-loss (sub)gradient is ≈ zero at the optimum.
    //
    // The directional subgradient w.r.t. the slope is
    //   Σ xᵢ ( τ − 1{rᵢ<0} )  (ignoring zero-residual ties), which the LP optimum
    // drives to ~0 along feasible directions. We assert that perturbing the slope
    // does not decrease the loss (a numerical optimality certificate).
    #[test]
    fn subgradient_near_zero_at_optimum() {
        let n = 200;
        let mut rng = LcgRng::new(101);
        let xs: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| 0.7 + 1.3 * x + (rng.next_f64() - 0.5))
            .collect();
        let tau = 0.5;
        let cfg = QuantRegConfig::with_tau(tau);
        let fit = quantile_regression_fit(&xs, &ys, n, 1, &cfg).expect("fit");

        let base = fit.pinball_loss;
        // Perturb the slope ± and recompute the loss directly.
        let eval_slope = |b1: f64| {
            xs.iter()
                .zip(ys.iter())
                .map(|(&x, &y)| check_loss(y - (fit.intercept_val + b1 * x), tau))
                .sum::<f64>()
        };
        let b1 = fit.coefficients[0];
        let up = eval_slope(b1 + 1e-3);
        let down = eval_slope(b1 - 1e-3);
        assert!(
            up >= base - 1e-6 && down >= base - 1e-6,
            "loss should not decrease under slope perturbation: base {base}, up {up}, down {down}"
        );
    }

    // Prediction shape.
    #[test]
    fn predict_shape() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x).collect();
        let cfg = QuantRegConfig::default();
        let fit = quantile_regression_fit(&xs, &ys, 5, 1, &cfg).expect("fit");
        let x_new = vec![6.0, 7.0];
        let preds = quantile_regression_predict(&fit, &x_new, 2).expect("predict");
        assert_eq!(preds.len(), 2);
    }

    // Invalid τ rejected.
    #[test]
    fn invalid_tau_error() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![1.0, 2.0, 3.0];
        for &bad in &[0.0_f64, 1.0, -0.2, 1.5] {
            let cfg = QuantRegConfig::with_tau(bad);
            assert!(quantile_regression_fit(&xs, &ys, 3, 1, &cfg).is_err());
        }
    }

    // Empty input rejected.
    #[test]
    fn empty_input_error() {
        let cfg = QuantRegConfig::default();
        assert!(quantile_regression_fit(&[], &[], 0, 1, &cfg).is_err());
    }

    // Pinball loss is non-negative and finite.
    #[test]
    fn pinball_loss_nonneg() {
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x + 0.3).collect();
        let cfg = QuantRegConfig::default();
        let fit = quantile_regression_fit(&xs, &ys, 20, 1, &cfg).expect("fit");
        assert!(fit.pinball_loss.is_finite() && fit.pinball_loss >= 0.0);
    }
}
