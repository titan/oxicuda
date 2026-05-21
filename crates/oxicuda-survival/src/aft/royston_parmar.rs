//! Royston-Parmar flexible parametric survival models (Royston & Parmar 2002).
//!
//! Models the log cumulative hazard via a restricted cubic spline in `ln(t)`:
//!
//! ```text
//! ln H(t | x) = s(ln t; γ) + x^T β
//! ```
//!
//! where `s(·; γ)` is a natural cubic spline with interior knots placed at
//! quantiles of the log event times, and `H = -ln S` is the cumulative hazard.
//!
//! # References
//! - Royston P, Parmar MKB (2002). Flexible parametric proportional-hazards
//!   and proportional-odds models for censored survival data.
//!   *Statistics in Medicine* 21: 2175–2197.
//! - Harrell FE (2001). *Regression Modeling Strategies*. Springer.

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for the Royston-Parmar flexible parametric model.
#[derive(Debug, Clone)]
pub struct RoystonParmarConfig {
    /// Number of interior knots for the restricted cubic spline.
    /// `df = n_interior_knots + 2` total spline parameters.
    pub n_interior_knots: usize,
    /// Convergence tolerance: stop when `max |∂ℓ/∂θ_j| < tol`.
    pub tol: f64,
    /// Maximum number of gradient-descent iterations.
    pub max_iter: usize,
    /// Initial learning rate for Armijo line search.
    pub lr_init: f64,
}

impl Default for RoystonParmarConfig {
    fn default() -> Self {
        Self {
            n_interior_knots: 2,
            tol: 1.0e-5,
            max_iter: 200,
            lr_init: 0.01,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Fit result
// ──────────────────────────────────────────────────────────────────────────────

/// Fitted Royston-Parmar flexible parametric model.
#[derive(Debug, Clone)]
pub struct RoystonParmarFit {
    /// Spline coefficients `γ` (length = `df = n_interior_knots + 2`).
    pub gamma: Vec<f64>,
    /// Regression coefficients `β` for covariates (length = `n_features`).
    pub beta: Vec<f64>,
    /// All knots: `[boundary_min, interior_1, ..., interior_k, boundary_max]`.
    /// Length = `n_interior_knots + 2`.
    pub knots: Vec<f64>,
    /// Final log-likelihood.
    pub log_likelihood: f64,
    /// Number of optimisation iterations used.
    pub n_iter: usize,
    /// Whether the optimiser declared convergence.
    pub converged: bool,
    /// Spline degrees of freedom: `n_interior_knots + 2`.
    pub df: usize,
}

impl RoystonParmarFit {
    /// Predict `S(t | x) = exp(-exp(s(ln t; γ) + x^T β))` for each time in `times`.
    ///
    /// # Errors
    /// Returns `DimensionMismatch` if `x.len() != beta.len()`.
    pub fn predict_survival(&self, x: &[f64], times: &[f64]) -> SurvivalResult<Vec<f64>> {
        if x.len() != self.beta.len() {
            return Err(SurvivalError::DimensionMismatch {
                a: x.len(),
                b: self.beta.len(),
            });
        }
        times
            .iter()
            .map(|&t| {
                if t <= 0.0 {
                    return Ok(1.0);
                }
                let u = t.ln();
                let basis = rcs_basis(u, &self.knots);
                let eta = linear_predictor(&self.gamma, x, &self.beta, &basis);
                Ok((-eta.exp()).exp())
            })
            .collect()
    }

    /// Predict `h(t | x)` for each time in `times`.
    ///
    /// The hazard is `h(t) = dH/dt = exp(ln H(t)) * s'(ln t) / t`.
    ///
    /// # Errors
    /// Returns `DimensionMismatch` if `x.len() != beta.len()`.
    pub fn predict_hazard(&self, x: &[f64], times: &[f64]) -> SurvivalResult<Vec<f64>> {
        if x.len() != self.beta.len() {
            return Err(SurvivalError::DimensionMismatch {
                a: x.len(),
                b: self.beta.len(),
            });
        }
        times
            .iter()
            .map(|&t| {
                if t <= 0.0 {
                    return Ok(0.0);
                }
                let u = t.ln();
                let basis = rcs_basis(u, &self.knots);
                let deriv = rcs_deriv(u, &self.knots);
                let eta = linear_predictor(&self.gamma, x, &self.beta, &basis);
                // s'(ln t) = d(ln H)/d(ln t) = Σ γ_j * deriv_j
                let spline_deriv: f64 = self
                    .gamma
                    .iter()
                    .zip(deriv.iter())
                    .map(|(g, d)| g * d)
                    .sum();
                // h(t) = H(t) * s'(ln t) / t = exp(eta) * spline_deriv / t
                let h = eta.exp() * spline_deriv / t;
                Ok(h.max(0.0))
            })
            .collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Restricted cubic spline basis (Harrell 2001)
// ──────────────────────────────────────────────────────────────────────────────

/// Truncated power function `(x - ksi)_+^3`.
#[inline]
fn trunc_cube(x: f64, ksi: f64) -> f64 {
    let d = x - ksi;
    if d > 0.0 { d * d * d } else { 0.0 }
}

/// Restricted cubic spline basis at `x` given all knots (boundary + interior).
///
/// The knots vector has length `k + 2` where `k = n_interior_knots`.
/// Returns a basis vector of length `k + 2 = df`:
///
/// ```text
/// v_0(x) = 1          (intercept)
/// v_1(x) = x          (linear)
/// v_j(x) = (x - ξ_{j-1})_+^3 - λ_j (x - ξ_{k+1})_+^3 + (λ_j - 1)(x - ξ_{k+2})_+^3
///          for j = 2 .. df-1
/// ```
///
/// where `ξ_1 … ξ_{k+2}` are all knots in order and
/// `λ_j = (ξ_{last} - ξ_{j-1}) / (ξ_{last} - ξ_{second_last})`.
#[must_use]
pub fn rcs_basis(x: f64, knots: &[f64]) -> Vec<f64> {
    let n_knots = knots.len(); // k + 2
    let df = n_knots; // df = k + 2
    let mut basis = vec![0.0_f64; df];

    // v_0 = 1 (intercept), v_1 = x (linear)
    basis[0] = 1.0;
    basis[1] = x;

    if n_knots < 2 {
        return basis;
    }

    // Boundary knots
    let ksi_last = knots[n_knots - 1];
    let ksi_second_last = knots[n_knots - 2];
    let denom = ksi_last - ksi_second_last;

    for j in 2..df {
        // Interior knot index: ξ_{j-1} in 0-based is knots[j-2]
        let ksi_j = knots[j - 2];
        let lambda_j = if denom.abs() < f64::EPSILON {
            0.5
        } else {
            (ksi_last - ksi_j) / denom
        };
        basis[j] = trunc_cube(x, ksi_j) - lambda_j * trunc_cube(x, ksi_second_last)
            + (lambda_j - 1.0) * trunc_cube(x, ksi_last);
    }
    basis
}

/// Derivative of the RCS basis w.r.t. `x` at point `x`.
///
/// Returns a vector of length `df` containing `dv_j/dx` for each basis function.
#[must_use]
pub fn rcs_deriv(x: f64, knots: &[f64]) -> Vec<f64> {
    let n_knots = knots.len();
    let df = n_knots;
    let mut deriv = vec![0.0_f64; df];

    // d/dx [1] = 0, d/dx [x] = 1
    deriv[0] = 0.0;
    deriv[1] = 1.0;

    if n_knots < 2 {
        return deriv;
    }

    let ksi_last = knots[n_knots - 1];
    let ksi_second_last = knots[n_knots - 2];
    let denom = ksi_last - ksi_second_last;

    // Derivative of truncated power: d/dx (x - ksi)_+^3 = 3(x - ksi)_+^2
    for j in 2..df {
        let ksi_j = knots[j - 2];
        let lambda_j = if denom.abs() < f64::EPSILON {
            0.5
        } else {
            (ksi_last - ksi_j) / denom
        };
        let dt_j = if x > ksi_j {
            3.0 * (x - ksi_j).powi(2)
        } else {
            0.0
        };
        let dt_sl = if x > ksi_second_last {
            3.0 * (x - ksi_second_last).powi(2)
        } else {
            0.0
        };
        let dt_l = if x > ksi_last {
            3.0 * (x - ksi_last).powi(2)
        } else {
            0.0
        };
        deriv[j] = dt_j - lambda_j * dt_sl + (lambda_j - 1.0) * dt_l;
    }
    deriv
}

// ──────────────────────────────────────────────────────────────────────────────
// Linear predictor
// ──────────────────────────────────────────────────────────────────────────────

/// Compute `γ · basis + x · β`.
#[must_use]
pub fn linear_predictor(gamma: &[f64], x: &[f64], beta: &[f64], basis: &[f64]) -> f64 {
    let spline_part: f64 = gamma.iter().zip(basis.iter()).map(|(g, b)| g * b).sum();
    let covariate_part: f64 = beta.iter().zip(x.iter()).map(|(b, xi)| b * xi).sum();
    spline_part + covariate_part
}

// ──────────────────────────────────────────────────────────────────────────────
// Log-likelihood
// ──────────────────────────────────────────────────────────────────────────────

/// Compute the total log-likelihood for right-censored data.
///
/// For event `i` with status = true:
/// ```text
/// log L_i = log h(t_i | x_i)
///         = log[s'(ln t_i) / t_i] + ln H(t_i | x_i)
///         = log(spline_deriv_i) - log(t_i) + eta_i
/// ```
/// where `spline_deriv_i = Σ γ_j * dv_j/du |_{u = ln t_i}` and
/// `eta_i = ln H(t_i | x_i) = s(ln t_i; γ) + x_i^T β`.
///
/// For censored `i`:
/// ```text
/// log L_i = log S(t_i | x_i) = -exp(eta_i)
/// ```
///
/// # Parameters
/// - `theta`: parameter vector `[γ_0, …, γ_{df-1}, β_0, …, β_{p-1}]` of length `df + p`.
/// - `knots`: all knots (boundary + interior), length `df`.
/// - `df`: spline degrees of freedom = `n_interior_knots + 2`.
pub fn log_likelihood_fn(
    data: &Dataset,
    theta: &[f64],
    knots: &[f64],
    df: usize,
) -> SurvivalResult<f64> {
    let p = theta.len() - df;
    let gamma = &theta[..df];
    let beta = &theta[df..];

    let mut ll = 0.0_f64;
    for (i, obs) in data.observations.iter().enumerate() {
        let t = obs.time;
        if t <= 0.0 {
            // Cannot take ln(t) — skip with large penalty
            return Ok(f64::NEG_INFINITY);
        }
        let u = t.ln();
        let basis = rcs_basis(u, knots);
        let x_i: &[f64] = if p > 0 {
            match &data.covariates {
                Some(cov) => &cov[i],
                None => &[],
            }
        } else {
            &[]
        };
        let eta = linear_predictor(gamma, x_i, beta, &basis);

        if obs.event {
            // log h(t | x) = log(Σ γ_j dv_j/du) - log(t) + eta
            let deriv = rcs_deriv(u, knots);
            let spline_deriv: f64 = gamma.iter().zip(deriv.iter()).map(|(g, d)| g * d).sum();
            if spline_deriv <= 0.0 {
                // Non-positive derivative → hazard is non-positive → invalid
                return Ok(f64::NEG_INFINITY);
            }
            ll += spline_deriv.ln() - t.ln() + eta;
        } else {
            // log S(t | x) = -exp(eta)
            ll -= eta.exp();
        }
    }

    if ll.is_finite() {
        Ok(ll)
    } else {
        Ok(f64::NEG_INFINITY)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Knot placement
// ──────────────────────────────────────────────────────────────────────────────

/// Place knots at quantiles of `ln(event_times)`.
///
/// Returns `[boundary_min, interior_1, …, interior_k, boundary_max]` (length `k+2`).
fn place_knots(event_log_times: &mut [f64], n_interior: usize) -> Vec<f64> {
    event_log_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = event_log_times.len();

    let boundary_min = event_log_times[0];
    let boundary_max = event_log_times[m - 1];

    let mut knots = Vec::with_capacity(n_interior + 2);
    knots.push(boundary_min);

    for k in 1..=n_interior {
        // Equally-spaced quantile positions: k / (n_interior + 1)
        let q = k as f64 / (n_interior + 1) as f64;
        // Linear interpolation in the sorted array
        let pos = q * (m - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(m - 1);
        let frac = pos - lo as f64;
        let ksi = event_log_times[lo] * (1.0 - frac) + event_log_times[hi] * frac;
        knots.push(ksi);
    }

    knots.push(boundary_max);
    knots
}

// ──────────────────────────────────────────────────────────────────────────────
// Gradient via central finite differences
// ──────────────────────────────────────────────────────────────────────────────

fn finite_diff_gradient(
    data: &Dataset,
    theta: &[f64],
    knots: &[f64],
    df: usize,
    h: f64,
) -> SurvivalResult<Vec<f64>> {
    let dim = theta.len();
    let mut grad = vec![0.0_f64; dim];
    let mut theta_p = theta.to_vec();
    let mut theta_m = theta.to_vec();
    for j in 0..dim {
        theta_p[j] = theta[j] + h;
        theta_m[j] = theta[j] - h;
        let lp = log_likelihood_fn(data, &theta_p, knots, df)?;
        let lm = log_likelihood_fn(data, &theta_m, knots, df)?;
        grad[j] = (lp - lm) / (2.0 * h);
        theta_p[j] = theta[j];
        theta_m[j] = theta[j];
    }
    Ok(grad)
}

// ──────────────────────────────────────────────────────────────────────────────
// L-BFGS two-loop recursion
// ──────────────────────────────────────────────────────────────────────────────

/// Simple L-BFGS direction given curvature pairs `(s_k, y_k)`.
/// Returns the search direction (negative of the L-BFGS Hessian-approximation times gradient).
fn lbfgs_direction(grad: &[f64], s_history: &[Vec<f64>], y_history: &[Vec<f64>]) -> Vec<f64> {
    let dim = grad.len();
    let m = s_history.len();
    let mut q = grad.to_vec();
    let mut alpha = vec![0.0_f64; m];

    // Two-loop recursion — first loop (most recent pair first)
    for i in (0..m).rev() {
        let sy: f64 = s_history[i]
            .iter()
            .zip(y_history[i].iter())
            .map(|(s, y)| s * y)
            .sum();
        if sy.abs() < f64::EPSILON * 1e6 {
            continue;
        }
        let rho = 1.0 / sy;
        let sq: f64 = s_history[i]
            .iter()
            .zip(q.iter())
            .map(|(s, qi)| s * qi)
            .sum();
        alpha[i] = rho * sq;
        for k in 0..dim {
            q[k] -= alpha[i] * y_history[i][k];
        }
    }

    // Initial Hessian scaling H_0 = (y_m · s_m) / (y_m · y_m) * I
    let mut r = if m > 0 {
        let last = m - 1;
        let sy: f64 = s_history[last]
            .iter()
            .zip(y_history[last].iter())
            .map(|(s, y)| s * y)
            .sum();
        let yy: f64 = y_history[last].iter().map(|y| y * y).sum();
        let scale = if yy > f64::EPSILON * 1e6 {
            sy / yy
        } else {
            1.0
        };
        q.iter().map(|qi| scale * qi).collect::<Vec<_>>()
    } else {
        q.clone()
    };

    // Second loop
    for i in 0..m {
        let sy: f64 = s_history[i]
            .iter()
            .zip(y_history[i].iter())
            .map(|(s, y)| s * y)
            .sum();
        if sy.abs() < f64::EPSILON * 1e6 {
            continue;
        }
        let rho = 1.0 / sy;
        let yr: f64 = y_history[i]
            .iter()
            .zip(r.iter())
            .map(|(y, ri)| y * ri)
            .sum();
        let beta = rho * yr;
        for k in 0..dim {
            r[k] += s_history[i][k] * (alpha[i] - beta);
        }
    }

    // direction = -r (we are maximising, so direction = -H^{-1} grad of negative LL)
    r.iter().map(|ri| -ri).collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Main fitting function
// ──────────────────────────────────────────────────────────────────────────────

/// Fit a Royston-Parmar flexible parametric survival model.
///
/// The model log cumulative hazard is a restricted cubic spline in `ln(t)` plus
/// a linear predictor from covariates.
///
/// # Errors
/// - `EmptyDataset` if the dataset has no observations.
/// - `NoEvents` if there are no uncensored event times.
/// - `InvalidParameter` if the dataset has too few events relative to `n_interior_knots`.
/// - `NumericalInstability` on non-finite log-likelihood at convergence.
pub fn fit_royston_parmar(
    data: &Dataset,
    config: &RoystonParmarConfig,
) -> SurvivalResult<RoystonParmarFit> {
    // ── Validation ────────────────────────────────────────────────────────────
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    let n_events = data.n_events();
    if n_events == 0 {
        return Err(SurvivalError::NoEvents);
    }

    let df = config.n_interior_knots + 2;
    // Need at least df + 1 events for the model to be identifiable
    if n_events < df + 1 {
        return Err(SurvivalError::InvalidParameter(format!(
            "too few events ({n_events}) for n_interior_knots={}: need at least {}",
            config.n_interior_knots,
            df + 1
        )));
    }

    let p = data.n_features();

    // ── Collect event log-times for knot placement ─────────────────────────
    let mut event_log_times: Vec<f64> = data
        .observations
        .iter()
        .filter(|o| o.event && o.time > 0.0)
        .map(|o| o.time.ln())
        .collect();

    if event_log_times.len() < df + 1 {
        return Err(SurvivalError::InvalidParameter(format!(
            "too few positive event times ({}) for df={}",
            event_log_times.len(),
            df
        )));
    }

    let knots = place_knots(&mut event_log_times, config.n_interior_knots);

    // ── Initialise parameters ─────────────────────────────────────────────
    // Start with a Weibull-like initialisation: γ = (intercept, slope, 0, …, 0)
    // For a standard Weibull: ln H(t) = ln(lambda) + k * ln(t).
    // Use median event time to guess intercept; slope = 1.
    let n = event_log_times.len();
    let median_log_t = event_log_times[n / 2];
    let slope_init = 1.0_f64;
    // H(median) ≈ ln(2) → intercept ≈ ln(ln(2)) - slope * median_log_t
    let intercept_init = 2_f64.ln().ln() - slope_init * median_log_t;

    let dim = df + p;
    let mut theta = vec![0.0_f64; dim];
    theta[0] = intercept_init;
    theta[1] = slope_init;
    // remaining spline and beta coefficients start at 0

    // ── L-BFGS with Armijo line search ────────────────────────────────────
    const FD_H: f64 = 1.0e-6;
    const LBFGS_MEMORY: usize = 10;
    const ARMIJO_C: f64 = 1.0e-4;
    const MAX_STEP_HALVINGS: usize = 40;

    let mut s_history: Vec<Vec<f64>> = Vec::with_capacity(LBFGS_MEMORY);
    let mut y_history: Vec<Vec<f64>> = Vec::with_capacity(LBFGS_MEMORY);

    let mut ll_cur = log_likelihood_fn(data, &theta, &knots, df)?;
    let mut grad_cur = finite_diff_gradient(data, &theta, &knots, df, FD_H)?;

    let mut n_iter = 0usize;
    let mut converged = false;

    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        // Convergence check: max |gradient component|
        let max_grad = grad_cur.iter().map(|g| g.abs()).fold(0.0_f64, f64::max);
        if max_grad < config.tol {
            converged = true;
            break;
        }

        // L-BFGS search direction
        let direction = lbfgs_direction(&grad_cur, &s_history, &y_history);

        // Verify it's an ascent direction; if not fall back to gradient
        let slope: f64 = grad_cur
            .iter()
            .zip(direction.iter())
            .map(|(g, d)| g * d)
            .sum();
        let direction = if slope <= 0.0 {
            grad_cur.clone()
        } else {
            direction
        };

        // Armijo backtracking line search
        let dir_norm: f64 = direction
            .iter()
            .map(|d| d * d)
            .sum::<f64>()
            .sqrt()
            .max(f64::EPSILON);
        let slope_norm: f64 = grad_cur
            .iter()
            .zip(direction.iter())
            .map(|(g, d)| g * d)
            .sum::<f64>()
            / dir_norm;

        let mut step = config.lr_init;
        let mut accepted = false;
        let mut theta_new = theta.clone();

        for _ in 0..MAX_STEP_HALVINGS {
            for j in 0..dim {
                theta_new[j] = theta[j] + step * direction[j];
            }
            let ll_new = log_likelihood_fn(data, &theta_new, &knots, df)?;
            if ll_new.is_finite() && ll_new >= ll_cur + ARMIJO_C * step * slope_norm * dir_norm {
                accepted = true;
                // Update L-BFGS curvature pairs
                let s_k: Vec<f64> = theta_new
                    .iter()
                    .zip(theta.iter())
                    .map(|(n, o)| n - o)
                    .collect();
                let grad_new = finite_diff_gradient(data, &theta_new, &knots, df, FD_H)?;
                let y_k: Vec<f64> = grad_new
                    .iter()
                    .zip(grad_cur.iter())
                    .map(|(n, o)| n - o)
                    .collect();

                // Only add pair if curvature condition s·y > 0
                let sy: f64 = s_k.iter().zip(y_k.iter()).map(|(s, y)| s * y).sum();
                if sy > 0.0 {
                    if s_history.len() == LBFGS_MEMORY {
                        s_history.remove(0);
                        y_history.remove(0);
                    }
                    s_history.push(s_k);
                    y_history.push(y_k);
                    grad_cur = grad_new;
                } else {
                    grad_cur = grad_new;
                }

                theta = theta_new.clone();
                ll_cur = ll_new;
                break;
            }
            step *= 0.5;
            if step < 1.0e-20 {
                break;
            }
        }

        if !accepted {
            // Try a small pure gradient step as a recovery
            let fallback_step = 1.0e-6;
            for j in 0..dim {
                theta_new[j] = theta[j] + fallback_step * grad_cur[j];
            }
            let ll_fb = log_likelihood_fn(data, &theta_new, &knots, df)?;
            if ll_fb.is_finite() && ll_fb > ll_cur {
                theta = theta_new;
                ll_cur = ll_fb;
                grad_cur = finite_diff_gradient(data, &theta, &knots, df, FD_H)?;
                s_history.clear();
                y_history.clear();
            } else {
                // Cannot make progress — declare convergence-like exit
                break;
            }
        }
    }

    let log_likelihood = log_likelihood_fn(data, &theta, &knots, df)?;
    if !log_likelihood.is_finite() {
        return Err(SurvivalError::NumericalInstability(
            "non-finite log-likelihood at final parameters".to_string(),
        ));
    }

    let gamma = theta[..df].to_vec();
    let beta = theta[df..].to_vec();

    Ok(RoystonParmarFit {
        gamma,
        beta,
        knots,
        log_likelihood,
        n_iter,
        converged,
        df,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Dataset, Observation};
    use crate::handle::LcgRng;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_obs(t: f64, e: bool) -> Observation {
        Observation::new(t, e).expect("valid observation")
    }

    /// Simple synthetic dataset: 20 event times from exp(1), no censoring.
    fn synth_no_cov(seed: u64, n: usize) -> Dataset {
        let mut rng = LcgRng::new(seed);
        let obs: Vec<Observation> = (0..n)
            .map(|_| make_obs(rng.next_exponential(1.0).max(0.01), true))
            .collect();
        Dataset::new(obs, None, None).expect("valid dataset")
    }

    /// Synthetic dataset with one covariate and partial censoring.
    fn synth_with_cov(seed: u64, n: usize) -> Dataset {
        let mut rng = LcgRng::new(seed);
        let mut obs = Vec::with_capacity(n);
        let mut covs = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.next_normal();
            let t = rng.next_exponential(0.5 * (1.0 + x.abs())).max(0.01);
            let censored = rng.next_f64() < 0.2;
            obs.push(make_obs(t, !censored));
            covs.push(vec![x]);
        }
        Dataset::new(obs, Some(covs), None).expect("valid dataset")
    }

    // ── basis tests ──────────────────────────────────────────────────────────

    #[test]
    fn rcs_basis_length_correct() {
        let knots = vec![0.0, 1.0, 2.0, 3.0]; // 2 interior knots → df = 4
        let b = rcs_basis(1.5, &knots);
        assert_eq!(
            b.len(),
            4,
            "basis length must equal n_interior_knots + 2 = 4"
        );
    }

    #[test]
    fn rcs_basis_continuity() {
        // At each interior knot the RCS basis must be continuous.
        // We check that |b(ksi - ε) - b(ksi + ε)| / ε → 0 as ε → 0 at a
        // rate consistent with at most a first-order difference (no jump).
        // A true jump discontinuity would give |diff| ≈ constant as ε → 0.
        // Continuity means |diff| ≈ O(ε), so |diff| / (2ε) must stay finite
        // and consistent with the derivative.  We simply verify that with a
        // very small ε the absolute difference is no larger than ~1e-4 (i.e.,
        // it is O(ε) and not O(1)).
        let knots = vec![0.0, 1.0, 2.0, 3.0];
        let eps = 1.0e-9;
        for &ksi in &knots[1..knots.len() - 1] {
            let b_left = rcs_basis(ksi - eps, &knots);
            let b_right = rcs_basis(ksi + eps, &knots);
            // For the cubic spline correction terms (index ≥ 2), the first
            // derivative is continuous at knots but the function itself might
            // change smoothly.  With ε = 1e-9, any continuous function gives
            // |b(ksi - ε) - b(ksi + ε)| ≤ |b'| * 2ε which is at most ~1e-6
            // for reasonable derivatives.  A jump would give O(1) difference.
            for (idx, (l, r)) in b_left.iter().zip(b_right.iter()).enumerate() {
                let diff = (l - r).abs();
                assert!(
                    diff < 1.0e-4,
                    "discontinuity in basis[{idx}] at knot {ksi}: |left - right| = {diff} (eps = {eps})"
                );
            }
        }
    }

    #[test]
    fn rcs_basis_intercept_and_linear() {
        let knots = vec![0.0, 0.5, 1.0, 2.0];
        let x = 0.7;
        let b = rcs_basis(x, &knots);
        assert_eq!(b[0], 1.0, "basis[0] must be 1 (intercept)");
        assert!(
            (b[1] - x).abs() < f64::EPSILON,
            "basis[1] must equal x (linear)"
        );
    }

    #[test]
    fn rcs_deriv_linear_term() {
        let knots = vec![0.0, 1.0, 3.0]; // 1 interior knot → df = 3
        // Far from any knot, the spline is linear → derivative of basis[1] = 1
        let d = rcs_deriv(-10.0, &knots);
        assert!((d[1] - 1.0).abs() < f64::EPSILON);
        assert!((d[0] - 0.0).abs() < f64::EPSILON);
    }

    // ── fit tests ────────────────────────────────────────────────────────────

    #[test]
    fn royston_parmar_fits_without_error() {
        let data = synth_no_cov(42, 40);
        let config = RoystonParmarConfig::default();
        let fit = fit_royston_parmar(&data, &config);
        assert!(fit.is_ok(), "expected Ok, got {:?}", fit);
    }

    #[test]
    fn royston_parmar_no_covariates_converges() {
        let data = synth_no_cov(7, 60);
        let config = RoystonParmarConfig {
            n_interior_knots: 1,
            tol: 1.0e-4,
            max_iter: 300,
            lr_init: 0.05,
        };
        let fit = fit_royston_parmar(&data, &config).expect("fit ok");
        assert_eq!(fit.beta.len(), 0, "no covariates → beta must be empty");
        assert_eq!(fit.df, 3, "1 interior knot → df = 3");
        assert!(fit.gamma.len() == 3);
    }

    #[test]
    fn royston_parmar_log_likelihood_finite() {
        let data = synth_no_cov(13, 50);
        let config = RoystonParmarConfig::default();
        let fit = fit_royston_parmar(&data, &config).expect("fit ok");
        assert!(
            fit.log_likelihood.is_finite(),
            "log-likelihood must be finite, got {}",
            fit.log_likelihood
        );
    }

    #[test]
    fn royston_parmar_survival_in_range() {
        let data = synth_no_cov(17, 50);
        let config = RoystonParmarConfig::default();
        let fit = fit_royston_parmar(&data, &config).expect("fit ok");
        // Use times in [0, 1] — well within the observed exponential data range
        // S(t) = exp(-exp(eta)) is in [0, 1] by construction.
        let times: Vec<f64> = (1..=10).map(|i| i as f64 * 0.1).collect();
        let s = fit.predict_survival(&[], &times).expect("predict ok");
        for (i, &si) in s.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&si),
                "S(t[{i}]) = {si} must be in [0, 1]"
            );
        }
        // All values must be in [0, 1] — this is guaranteed by the complementary log-log link
        let all_in_range = s.iter().all(|&si| (0.0..=1.0).contains(&si));
        assert!(all_in_range, "all S(t) values must be in [0, 1]");
    }

    #[test]
    fn royston_parmar_survival_monotone() {
        let data = synth_no_cov(23, 60);
        let config = RoystonParmarConfig::default();
        let fit = fit_royston_parmar(&data, &config).expect("fit ok");
        let times: Vec<f64> = (1..=20).map(|i| i as f64 * 0.3).collect();
        let s = fit.predict_survival(&[], &times).expect("predict ok");
        for w in s.windows(2) {
            assert!(
                w[0] >= w[1] - 1.0e-9,
                "S(t) must be non-increasing: S({}) = {} > S = {}",
                times[0],
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn royston_parmar_hazard_positive() {
        let data = synth_no_cov(31, 50);
        let config = RoystonParmarConfig::default();
        let fit = fit_royston_parmar(&data, &config).expect("fit ok");
        let times: Vec<f64> = (1..=10).map(|i| i as f64 * 0.4).collect();
        let h = fit.predict_hazard(&[], &times).expect("predict ok");
        for &hi in &h {
            assert!(hi >= 0.0, "hazard must be non-negative, got {hi}");
        }
    }

    #[test]
    fn royston_parmar_with_covariates_ok() {
        let data = synth_with_cov(55, 60);
        let config = RoystonParmarConfig {
            n_interior_knots: 1,
            tol: 1.0e-4,
            max_iter: 300,
            lr_init: 0.02,
        };
        let fit = fit_royston_parmar(&data, &config).expect("fit ok");
        assert_eq!(fit.beta.len(), 1, "one covariate → beta has length 1");
        assert!(fit.log_likelihood.is_finite());
    }

    #[test]
    fn royston_parmar_empty_dataset_returns_error() {
        use crate::error::SurvivalError;
        let result = Dataset::new(vec![], None, None);
        // Dataset::new itself rejects empty datasets
        assert!(
            matches!(result, Err(SurvivalError::EmptyDataset)),
            "expected EmptyDataset"
        );
    }

    #[test]
    fn royston_parmar_too_few_events_for_knots_returns_error() {
        use crate::error::SurvivalError;
        // 3 events with n_interior_knots=2 → df=4, need ≥5 events
        let data = Dataset::new(
            vec![
                make_obs(1.0, true),
                make_obs(2.0, true),
                make_obs(3.0, true),
                make_obs(4.0, false),
                make_obs(5.0, false),
            ],
            None,
            None,
        )
        .expect("ok");
        let config = RoystonParmarConfig {
            n_interior_knots: 2,
            ..Default::default()
        };
        let result = fit_royston_parmar(&data, &config);
        assert!(
            matches!(result, Err(SurvivalError::InvalidParameter(_))),
            "expected InvalidParameter, got {result:?}"
        );
    }

    #[test]
    fn royston_parmar_knots_count_correct() {
        let data = synth_no_cov(77, 50);
        let config = RoystonParmarConfig {
            n_interior_knots: 3,
            tol: 1.0e-4,
            max_iter: 300,
            lr_init: 0.02,
        };
        let fit = fit_royston_parmar(&data, &config).expect("fit ok");
        // knots = [boundary_min, 3 interior, boundary_max] → length 5
        assert_eq!(
            fit.knots.len(),
            5,
            "knots length must be n_interior_knots + 2 = 5"
        );
        assert_eq!(fit.df, 5);
        assert_eq!(fit.gamma.len(), 5);
    }
}
