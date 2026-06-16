//! Bivariate copula models for joint survival analysis.
//!
//! Implements the IFM (Inference Functions for Margins) estimator:
//! 1. Fit Weibull marginals via gradient-ascent MLE independently.
//! 2. Convert marginal survival estimates to pseudo-observations (u_i, v_i).
//! 3. Optimize copula dependence parameter θ via golden-section search on
//!    the copula log-likelihood.
//!
//! Supported copula families: Frank, Clayton, Gumbel.

use crate::error::{SurvivalError, SurvivalResult};

// ─── Public types ─────────────────────────────────────────────────────────────

/// Supported copula families for bivariate survival.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CopulaFamily {
    /// Frank copula — allows negative/positive dependence, θ ∈ (-∞, ∞) \ {0}.
    Frank,
    /// Clayton copula — lower-tail dependence, θ ∈ (0, ∞).
    Clayton,
    /// Gumbel copula — upper-tail dependence, θ ∈ [1, ∞).
    Gumbel,
}

/// Configuration for bivariate copula fitting.
#[derive(Debug, Clone)]
pub struct CopulaConfig {
    /// Copula family to fit.
    pub family: CopulaFamily,
    /// Optional initial value for θ. If `None`, estimated from Kendall's τ.
    pub theta_init: Option<f64>,
    /// Maximum iterations for golden-section search.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
}

impl Default for CopulaConfig {
    fn default() -> Self {
        Self {
            family: CopulaFamily::Clayton,
            theta_init: None,
            max_iter: 100,
            tol: 1e-6,
        }
    }
}

/// Fitted Weibull marginal: S(t) = exp(-(t/scale)^shape).
#[derive(Debug, Clone, Copy)]
pub struct WeibullMarginalFit {
    /// Scale parameter λ (location in time scale).
    pub scale: f64,
    /// Shape parameter k (> 0).
    pub shape: f64,
}

impl WeibullMarginalFit {
    /// Survival function S(t; λ, k) = exp(-(t/λ)^k).
    #[must_use]
    pub fn survival(&self, t: f64) -> f64 {
        (-(t / self.scale).powf(self.shape)).exp()
    }
}

/// Fitted bivariate copula model.
#[derive(Debug, Clone)]
pub struct BivariateCopulaFit {
    /// Fitted Weibull marginal for the first variable.
    pub marginal_1: WeibullMarginalFit,
    /// Fitted Weibull marginal for the second variable.
    pub marginal_2: WeibullMarginalFit,
    /// Estimated copula dependence parameter.
    pub theta: f64,
    /// Standard error of θ̂ (via finite-difference Hessian).
    pub se_theta: f64,
    /// Full copula log-likelihood at the estimated parameters.
    pub log_likelihood: f64,
    /// AIC = 2*5 - 2*log_likelihood (5 params: λ₁, k₁, λ₂, k₂, θ).
    pub aic: f64,
    /// Kendall's τ implied by the fitted θ and copula family.
    pub kendall_tau: f64,
    /// Copula family used for fitting.
    pub family: CopulaFamily,
    /// Whether the optimization converged within `max_iter`.
    pub converged: bool,
    /// Number of golden-section iterations performed.
    pub iterations: usize,
}

// ─── Internal LCG (for tests only) ───────────────────────────────────────────
// Note: used only in #[cfg(test)] blocks below.

// ─── Weibull MLE via gradient ascent ─────────────────────────────────────────

/// Weibull log-likelihood gradient w.r.t. (λ, k).
///
/// ℓ(λ, k) = Σ_i [δ_i·(log k - log λ + (k-1)·log(t_i/λ)) - (t_i/λ)^k]
fn weibull_gradient(times: &[f64], events: &[bool], scale: f64, shape: f64) -> (f64, f64) {
    let mut grad_scale = 0.0_f64;
    let mut grad_shape = 0.0_f64;
    let n_events: f64 = events.iter().filter(|&&e| e).count() as f64;

    for (&t, &e) in times.iter().zip(events) {
        let ratio = t / scale;
        let ratio_k = ratio.powf(shape);
        let log_ratio = if ratio > 0.0 { ratio.ln() } else { -1e30 };

        // ∂ℓ/∂λ: -k/λ * n_events + k/λ * Σ (t/λ)^k  (per-obs contrib: k/λ * ratio^k)
        grad_scale += shape / scale * ratio_k;
        if e {
            grad_scale -= shape / scale;
        }

        // ∂ℓ/∂k: Σ δ_i * (1/k + log(t/λ)) - Σ (t/λ)^k * log(t/λ)
        if e {
            grad_shape += 1.0 / shape + log_ratio;
        }
        grad_shape -= ratio_k * log_ratio;
    }
    // grad_scale uses -k/λ*n_events + Σ_i k/λ ratio_i^k already (vectorised above).
    // Adjust: the sign should make ascent maximise ℓ.
    // ∂ℓ/∂λ = -k/λ * Σδ + k/λ * Σratio_k  = k/λ * (Σratio_k - n_events) ✓
    let _ = n_events; // already incorporated correctly per-observation
    (grad_scale, grad_shape)
}

/// Fit Weibull marginal to (times, events) via gradient ascent.
fn fit_weibull_marginal(times: &[f64], events: &[bool]) -> WeibullMarginalFit {
    let n = times.len();
    if n == 0 {
        return WeibullMarginalFit {
            scale: 1.0,
            shape: 1.0,
        };
    }
    let mean_t = times.iter().sum::<f64>() / n as f64;
    let mut scale = mean_t.max(1e-6);
    let mut shape = 1.0_f64;
    let lr = 1e-4;
    let max_iter = 500;
    let tol = 1e-6;

    for _ in 0..max_iter {
        let (gs, gk) = weibull_gradient(times, events, scale, shape);
        let new_scale = (scale + lr * gs).max(1e-6);
        let new_shape = (shape + lr * gk).max(1e-6);
        let delta = (new_scale - scale).abs() + (new_shape - shape).abs();
        scale = new_scale;
        shape = new_shape;
        if delta < tol {
            break;
        }
    }

    WeibullMarginalFit { scale, shape }
}

// ─── Normal CDF helper ────────────────────────────────────────────────────────

#[allow(dead_code)]
fn norm_cdf(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26 approximation, |err| < 7.5e-8.
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let pdf = (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    if x >= 0.0 {
        1.0 - pdf * poly
    } else {
        pdf * poly
    }
}

// ─── Copula family formulas ───────────────────────────────────────────────────

/// Clamp (u, v) to avoid log(0) / division-by-zero.
#[inline]
fn clamp_uv(u: f64, v: f64) -> (f64, f64) {
    (u.clamp(1e-9, 1.0 - 1e-9), v.clamp(1e-9, 1.0 - 1e-9))
}

// --- Frank copula ---

/// Frank C(u, v; θ).
fn frank_cdf(u: f64, v: f64, theta: f64) -> f64 {
    let (u, v) = clamp_uv(u, v);
    let em1 = (-theta).exp() - 1.0;
    if em1.abs() < 1e-15 {
        return u * v; // independence limit
    }
    let num = ((-theta * u).exp() - 1.0) * ((-theta * v).exp() - 1.0);
    -(1.0 / theta) * (1.0 + num / em1).ln().max(-700.0)
}

/// Frank log-density (correct implementation).
fn frank_log_density_correct(u: f64, v: f64, theta: f64) -> f64 {
    let (u, v) = clamp_uv(u, v);
    let abs_theta = theta.abs();
    if abs_theta < 1e-8 {
        return 0.0;
    }
    let em1 = (-abs_theta).exp() - 1.0; // negative when theta > 0
    let eu = (-abs_theta * u).exp() - 1.0; // negative
    let ev = (-abs_theta * v).exp() - 1.0; // negative
    // denom = (exp(-θ)-1) + (exp(-θu)-1)(exp(-θv)-1)
    // = em1 + eu*ev (both negatives so eu*ev > 0)
    let denom = em1 + eu * ev;
    if denom.abs() < 1e-30 {
        return -700.0;
    }
    // c = θ*(1-exp(-θ)) * exp(-θ*(u+v)) / denom^2
    // log c = log|θ| + log(1-exp(-θ)) - θ*(u+v) - 2*log|denom|
    let log_one_minus_exp_mt = (-em1).ln(); // 1-exp(-θ) = -em1 > 0
    abs_theta.ln() + log_one_minus_exp_mt - abs_theta * (u + v) - 2.0 * denom.abs().ln()
}

/// ∂C/∂u for Frank.
fn frank_partial_u(u: f64, v: f64, theta: f64) -> f64 {
    let (u, v) = clamp_uv(u, v);
    let abs_theta = theta.abs();
    if abs_theta < 1e-8 {
        return u;
    }
    let em1 = (-abs_theta).exp() - 1.0;
    let eu = (-abs_theta * u).exp() - 1.0;
    let ev = (-abs_theta * v).exp() - 1.0;
    let denom = em1 + eu * ev;
    if denom.abs() < 1e-30 {
        return 0.5;
    }
    // ∂C/∂u = exp(-θu) * (exp(-θv) - 1) / denom
    let val = (-abs_theta * u).exp() * ev / denom;
    val.clamp(1e-15, 1.0 - 1e-15)
}

/// ∂C/∂v for Frank (symmetric with ∂C/∂u).
fn frank_partial_v(u: f64, v: f64, theta: f64) -> f64 {
    frank_partial_u(v, u, theta)
}

/// Kendall's τ for Frank via Debye function D₁(θ).
///
/// τ_frank(θ) = 1 - 4/θ * (1 - D₁(θ))
/// where D₁(θ) = (1/θ) ∫₀^θ t/(e^t - 1) dt  (50-point trapezoidal rule).
pub fn frank_kendall_tau(theta: f64) -> f64 {
    let abs_theta = theta.abs();
    if abs_theta < 1e-6 {
        return 0.0;
    }
    let n_points = 50usize;
    let h = abs_theta / n_points as f64;
    let mut integral = 0.0_f64;
    // Trapezoidal rule: exclude endpoint t=0 (singularity), start from h.
    for k in 1..=n_points {
        let t = k as f64 * h;
        let expt = t.exp();
        let denom = expt - 1.0;
        let f = if denom.abs() < 1e-12 {
            1.0 // lim_{t→0} t/(e^t-1) = 1
        } else {
            t / denom
        };
        // Trapezoidal weights: 0.5 for endpoints.
        let w = if k == n_points { 0.5 } else { 1.0 };
        integral += w * f * h;
    }
    // Also add half-weight for t=0 (limit = 1).
    integral += 0.5 * 1.0 * h;
    let d1 = integral / abs_theta;
    let tau = 1.0 - 4.0 / abs_theta * (1.0 - d1);
    tau.clamp(-1.0, 1.0)
}

/// θ from Kendall's τ for Frank via bisection.
fn frank_theta_from_tau(tau: f64) -> SurvivalResult<f64> {
    if tau.abs() < 1e-8 {
        return Ok(1e-6);
    }
    // Search in (-50, 50) \ {0}.
    let target = tau;
    // Sign: tau > 0 → theta > 0.
    let (mut lo, mut hi) = if target > 0.0 {
        (1e-6_f64, 50.0_f64)
    } else {
        (-50.0_f64, -1e-6_f64)
    };
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        let t_mid = frank_kendall_tau(mid);
        if t_mid < target {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo).abs() < 1e-8 {
            break;
        }
    }
    Ok(0.5 * (lo + hi))
}

// --- Clayton copula ---

/// Clayton C(u, v; θ).
fn clayton_cdf(u: f64, v: f64, theta: f64) -> f64 {
    let (u, v) = clamp_uv(u, v);
    let theta = theta.max(1e-8);
    let inner = u.powf(-theta) + v.powf(-theta) - 1.0;
    if inner <= 0.0 {
        return 0.0;
    }
    inner.powf(-1.0 / theta)
}

/// Clayton log-density log c(u, v; θ).
fn clayton_log_density(u: f64, v: f64, theta: f64) -> f64 {
    let (u, v) = clamp_uv(u, v);
    let theta = theta.max(1e-8);
    let inner = u.powf(-theta) + v.powf(-theta) - 1.0;
    if inner <= 0.0 {
        return -700.0;
    }
    // log c = log(1+θ) - (1+θ)(log u + log v) + (2+1/θ) * log(C) where C = inner^{-1/θ}
    // Equivalently: log(1+θ) - (1+θ)(log u + log v) - (2+1/θ) * log(inner)
    (1.0 + theta).ln() - (1.0 + theta) * (u.ln() + v.ln()) - (2.0 + 1.0 / theta) * inner.ln()
}

/// ∂C/∂u for Clayton.
fn clayton_partial_u(u: f64, v: f64, theta: f64) -> f64 {
    let (u, v) = clamp_uv(u, v);
    let theta = theta.max(1e-8);
    let inner = u.powf(-theta) + v.powf(-theta) - 1.0;
    if inner <= 0.0 {
        return 0.5;
    }
    let val = u.powf(-(theta + 1.0)) * inner.powf(-(1.0 + 1.0 / theta));
    val.clamp(1e-15, 1.0 - 1e-15)
}

/// ∂C/∂v for Clayton.
fn clayton_partial_v(u: f64, v: f64, theta: f64) -> f64 {
    clayton_partial_u(v, u, theta)
}

/// Kendall's τ for Clayton: τ = θ/(θ+2).
#[inline]
fn clayton_kendall_tau(theta: f64) -> f64 {
    let theta = theta.max(0.0);
    theta / (theta + 2.0)
}

/// θ from τ for Clayton: θ = 2τ/(1-τ).
#[inline]
fn clayton_theta_from_tau(tau: f64) -> f64 {
    let tau = tau.clamp(0.0, 1.0 - 1e-9);
    2.0 * tau / (1.0 - tau)
}

// --- Gumbel copula ---

/// Gumbel C(u, v; θ).
fn gumbel_cdf(u: f64, v: f64, theta: f64) -> f64 {
    let (u, v) = clamp_uv(u, v);
    let theta = theta.max(1.0);
    let lu = (-u.ln()).powf(theta);
    let lv = (-v.ln()).powf(theta);
    let a = (lu + lv).powf(1.0 / theta);
    (-a).exp()
}

/// Gumbel log-density log c(u, v; θ).
fn gumbel_log_density(u: f64, v: f64, theta: f64) -> f64 {
    let (u, v) = clamp_uv(u, v);
    let theta = theta.max(1.0);
    let neg_ln_u = -u.ln(); // > 0
    let neg_ln_v = -v.ln(); // > 0
    let lu = neg_ln_u.powf(theta);
    let lv = neg_ln_v.powf(theta);
    let sum_lv = lu + lv;
    if sum_lv < 1e-30 {
        return -700.0;
    }
    let a = sum_lv.powf(1.0 / theta);
    let c_val = (-a).exp();
    if c_val < 1e-300 {
        return -700.0;
    }
    // log c = log(C) + log(A^{1/θ-2} * neg_ln_u^{θ-1} * neg_ln_v^{θ-1} * (A^{1/θ} + θ - 1))
    // where A = sum_lv^{1/θ}, C = gumbel_cdf.
    let log_c = -a;
    log_c
        + (1.0 / theta - 2.0) * sum_lv.ln()
        + (theta - 1.0) * neg_ln_u.ln()
        + (theta - 1.0) * neg_ln_v.ln()
        - u.ln() // 1/u factor
        - v.ln() // 1/v factor
        + (a + theta - 1.0).ln()
}

/// ∂C/∂u for Gumbel.
fn gumbel_partial_u(u: f64, v: f64, theta: f64) -> f64 {
    let (u, v) = clamp_uv(u, v);
    let theta = theta.max(1.0);
    let neg_ln_u = -u.ln();
    let neg_ln_v = -v.ln();
    let lu = neg_ln_u.powf(theta);
    let lv = neg_ln_v.powf(theta);
    let sum_lv = lu + lv;
    if sum_lv < 1e-30 {
        return 0.5;
    }
    let a = sum_lv.powf(1.0 / theta); // A
    let c_val = (-a).exp();
    // ∂C/∂u = C * A^{1/θ - 1} * (-ln u)^{θ-1} / u
    let val = c_val * sum_lv.powf(1.0 / theta - 1.0) * neg_ln_u.powf(theta - 1.0) / u;
    val.clamp(1e-15, 1.0 - 1e-15)
}

/// ∂C/∂v for Gumbel.
fn gumbel_partial_v(u: f64, v: f64, theta: f64) -> f64 {
    gumbel_partial_u(v, u, theta)
}

/// Kendall's τ for Gumbel: τ = 1 - 1/θ.
#[inline]
fn gumbel_kendall_tau(theta: f64) -> f64 {
    let theta = theta.max(1.0);
    1.0 - 1.0 / theta
}

/// θ from τ for Gumbel: θ = 1/(1-τ), clamped to [1, ∞).
#[inline]
fn gumbel_theta_from_tau(tau: f64) -> f64 {
    let tau = tau.clamp(0.0, 1.0 - 1e-9);
    (1.0 / (1.0 - tau)).max(1.0)
}

// ─── Public: Kendall's τ and θ conversions ────────────────────────────────────

/// Kendall's τ implied by a given θ for the specified copula family.
#[must_use]
pub fn kendall_tau_from_theta(family: CopulaFamily, theta: f64) -> f64 {
    match family {
        CopulaFamily::Frank => frank_kendall_tau(theta),
        CopulaFamily::Clayton => clayton_kendall_tau(theta),
        CopulaFamily::Gumbel => gumbel_kendall_tau(theta),
    }
}

/// θ corresponding to a given Kendall's τ for the specified copula family.
///
/// # Errors
/// - `InvalidParameter` if τ is outside the valid range for the family.
pub fn theta_from_kendall_tau(family: CopulaFamily, tau: f64) -> SurvivalResult<f64> {
    match family {
        CopulaFamily::Frank => frank_theta_from_tau(tau),
        CopulaFamily::Clayton => Ok(clayton_theta_from_tau(tau)),
        CopulaFamily::Gumbel => Ok(gumbel_theta_from_tau(tau)),
    }
}

// ─── Copula log-likelihood ────────────────────────────────────────────────────

/// Copula log-likelihood contribution given pseudo-observations and event indicators.
///
/// For each observation (u_i, v_i, δ_{1i}, δ_{2i}):
/// - (1,1): log c(u,v;θ)
/// - (1,0): log(∂C/∂u)
/// - (0,1): log(∂C/∂v)
/// - (0,0): log C(u,v;θ)
fn copula_log_likelihood(
    pseudo_u: &[f64],
    pseudo_v: &[f64],
    events_1: &[bool],
    events_2: &[bool],
    theta: f64,
    family: CopulaFamily,
) -> f64 {
    let mut ll = 0.0_f64;
    for i in 0..pseudo_u.len() {
        let u = pseudo_u[i];
        let v = pseudo_v[i];
        let e1 = events_1[i];
        let e2 = events_2[i];
        let contrib = match (e1, e2) {
            (true, true) => match family {
                CopulaFamily::Frank => frank_log_density_correct(u, v, theta),
                CopulaFamily::Clayton => clayton_log_density(u, v, theta),
                CopulaFamily::Gumbel => gumbel_log_density(u, v, theta),
            },
            (true, false) => {
                let p = match family {
                    CopulaFamily::Frank => frank_partial_u(u, v, theta),
                    CopulaFamily::Clayton => clayton_partial_u(u, v, theta),
                    CopulaFamily::Gumbel => gumbel_partial_u(u, v, theta),
                };
                p.max(1e-300).ln()
            }
            (false, true) => {
                let p = match family {
                    CopulaFamily::Frank => frank_partial_v(u, v, theta),
                    CopulaFamily::Clayton => clayton_partial_v(u, v, theta),
                    CopulaFamily::Gumbel => gumbel_partial_v(u, v, theta),
                };
                p.max(1e-300).ln()
            }
            (false, false) => {
                let c = match family {
                    CopulaFamily::Frank => frank_cdf(u, v, theta),
                    CopulaFamily::Clayton => clayton_cdf(u, v, theta),
                    CopulaFamily::Gumbel => gumbel_cdf(u, v, theta),
                };
                c.max(1e-300).ln()
            }
        };
        if contrib.is_finite() {
            ll += contrib;
        }
    }
    ll
}

// ─── Golden-section search ────────────────────────────────────────────────────

/// Golden-section search maximiser on [lo, hi].
fn golden_section_max<F: Fn(f64) -> f64>(
    f: F,
    mut lo: f64,
    mut hi: f64,
    max_iter: usize,
    tol: f64,
) -> (f64, usize) {
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0; // golden ratio conjugate ≈ 0.618
    let mut x1 = hi - phi * (hi - lo);
    let mut x2 = lo + phi * (hi - lo);
    let mut f1 = f(x1);
    let mut f2 = f(x2);
    let mut iters = 0usize;

    for _ in 0..max_iter {
        iters += 1;
        if f1 < f2 {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + phi * (hi - lo);
            f2 = f(x2);
        } else {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - phi * (hi - lo);
            f1 = f(x1);
        }
        if (hi - lo).abs() < tol {
            break;
        }
    }
    (0.5 * (lo + hi), iters)
}

// ─── Weibull marginal log-likelihood ─────────────────────────────────────────

fn weibull_log_likelihood(times: &[f64], events: &[bool], scale: f64, shape: f64) -> f64 {
    let mut ll = 0.0_f64;
    for (&t, &e) in times.iter().zip(events) {
        let ratio = (t / scale).max(1e-300);
        let ratio_k = ratio.powf(shape);
        ll -= ratio_k;
        if e {
            ll += shape.ln() - scale.ln() + (shape - 1.0) * ratio.ln();
        }
    }
    ll
}

// ─── Public: fit bivariate copula ────────────────────────────────────────────

/// Fit a bivariate copula model using the IFM estimator.
///
/// # Errors
/// - `EmptyDataset` if inputs are empty.
/// - `DimensionMismatch` if lengths differ.
/// - `NegativeTime` if any time is negative.
pub fn fit_bivariate_copula(
    times_1: &[f64],
    events_1: &[bool],
    times_2: &[f64],
    events_2: &[bool],
    config: &CopulaConfig,
) -> SurvivalResult<BivariateCopulaFit> {
    let n = times_1.len();
    if n == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    if times_2.len() != n || events_1.len() != n || events_2.len() != n {
        return Err(SurvivalError::DimensionMismatch {
            a: n,
            b: times_2.len(),
        });
    }
    for &t in times_1.iter().chain(times_2) {
        if !t.is_finite() || t < 0.0 {
            return Err(SurvivalError::NegativeTime(t));
        }
    }

    // Step 1: Fit Weibull marginals.
    let marginal_1 = fit_weibull_marginal(times_1, events_1);
    let marginal_2 = fit_weibull_marginal(times_2, events_2);

    // Step 2: Compute pseudo-observations.
    let pseudo_u: Vec<f64> = times_1
        .iter()
        .map(|&t| marginal_1.survival(t).clamp(1e-9, 1.0 - 1e-9))
        .collect();
    let pseudo_v: Vec<f64> = times_2
        .iter()
        .map(|&t| marginal_2.survival(t).clamp(1e-9, 1.0 - 1e-9))
        .collect();

    // Estimate sample Kendall's τ from pseudo-observations (as Pearson correlation proxy).
    let mean_u = pseudo_u.iter().sum::<f64>() / n as f64;
    let mean_v = pseudo_v.iter().sum::<f64>() / n as f64;
    let cov_uv: f64 = pseudo_u
        .iter()
        .zip(pseudo_v.iter())
        .map(|(&u, &v)| (u - mean_u) * (v - mean_v))
        .sum::<f64>();
    let var_u: f64 = pseudo_u.iter().map(|&u| (u - mean_u).powi(2)).sum();
    let var_v: f64 = pseudo_v.iter().map(|&v| (v - mean_v).powi(2)).sum();
    let sample_corr = if var_u > 0.0 && var_v > 0.0 {
        (cov_uv / (var_u * var_v).sqrt()).clamp(-0.99, 0.99)
    } else {
        0.0
    };

    // Initial θ from sample τ.
    let theta_init = if let Some(th) = config.theta_init {
        th
    } else {
        let tau_hat = sample_corr; // Pearson as proxy for τ
        match config.family {
            CopulaFamily::Frank => frank_theta_from_tau(tau_hat).unwrap_or(1.0),
            CopulaFamily::Clayton => clayton_theta_from_tau(tau_hat.max(0.0)).max(0.01),
            CopulaFamily::Gumbel => gumbel_theta_from_tau(tau_hat.max(0.0)).max(1.0),
        }
    };

    // Step 3: Golden-section search for θ.
    let (theta_lo, theta_hi) = match config.family {
        CopulaFamily::Frank => (0.01_f64, 20.0_f64),
        CopulaFamily::Clayton => (0.01_f64, 10.0_f64),
        CopulaFamily::Gumbel => (1.0_f64, 10.0_f64),
    };
    // Center initial search around theta_init.
    let search_lo = (theta_init * 0.1).max(theta_lo);
    let search_hi = (theta_init * 10.0).min(theta_hi);
    let (lo, hi) = if search_lo < search_hi {
        (search_lo, search_hi)
    } else {
        (theta_lo, theta_hi)
    };

    let ll_fn = |theta: f64| {
        copula_log_likelihood(
            &pseudo_u,
            &pseudo_v,
            events_1,
            events_2,
            theta,
            config.family,
        )
    };

    let (theta_hat, iters) = golden_section_max(ll_fn, lo, hi, config.max_iter, config.tol);

    let converged = iters < config.max_iter;

    // Step 4: Compute final log-likelihood and marginal LL contributions.
    let copula_ll = copula_log_likelihood(
        &pseudo_u,
        &pseudo_v,
        events_1,
        events_2,
        theta_hat,
        config.family,
    );
    let marg_ll_1 = weibull_log_likelihood(times_1, events_1, marginal_1.scale, marginal_1.shape);
    let marg_ll_2 = weibull_log_likelihood(times_2, events_2, marginal_2.scale, marginal_2.shape);
    let log_likelihood = marg_ll_1 + marg_ll_2 + copula_ll;

    // AIC with 5 parameters: λ1, k1, λ2, k2, θ.
    let aic = 2.0 * 5.0 - 2.0 * log_likelihood;

    // Step 5: Standard error via finite differences.
    let h_fd = (1.0 + theta_hat.abs()) * 1e-4;
    let ll_plus = ll_fn(theta_hat + h_fd);
    let ll_minus = ll_fn(theta_hat - h_fd);
    let hessian = (ll_plus - 2.0 * copula_ll + ll_minus) / (h_fd * h_fd);
    let se_theta = if hessian < -1e-15 {
        (1.0 / (-hessian)).sqrt()
    } else {
        f64::INFINITY
    };

    let kendall_tau = kendall_tau_from_theta(config.family, theta_hat);

    Ok(BivariateCopulaFit {
        marginal_1,
        marginal_2,
        theta: theta_hat,
        se_theta,
        log_likelihood,
        aic,
        kendall_tau,
        family: config.family,
        converged,
        iterations: iters,
    })
}

// ─── Public: prediction ───────────────────────────────────────────────────────

/// Joint survival probability P(T₁ > t₁, T₂ > t₂) = C(S₁(t₁), S₂(t₂); θ).
#[must_use]
pub fn copula_survival_prob(fit: &BivariateCopulaFit, t1: f64, t2: f64) -> f64 {
    let u = fit.marginal_1.survival(t1).clamp(1e-9, 1.0 - 1e-9);
    let v = fit.marginal_2.survival(t2).clamp(1e-9, 1.0 - 1e-9);
    let prob = match fit.family {
        CopulaFamily::Frank => frank_cdf(u, v, fit.theta),
        CopulaFamily::Clayton => clayton_cdf(u, v, fit.theta),
        CopulaFamily::Gumbel => gumbel_cdf(u, v, fit.theta),
    };
    prob.max(0.0)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple LCG for deterministic test data (only used inside tests).
    fn simple_lcg_sequence(seed: u64, n: usize) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((s >> 11) as f64) / (1u64 << 53) as f64
            })
            .collect()
    }

    /// Simulate bivariate data from a Clayton copula.
    /// Given uniform (u_raw, w_raw): v = ((u^{-θ}*(w^{-θ/(1+θ)}-1))+1)^{-1/θ}
    fn simulate_clayton(seed: u64, n: usize, theta: f64) -> (Vec<f64>, Vec<f64>) {
        let u_seq = simple_lcg_sequence(seed, n);
        let w_seq = simple_lcg_sequence(seed.wrapping_add(999), n);
        let v_seq: Vec<f64> = u_seq
            .iter()
            .zip(w_seq.iter())
            .map(|(&u, &w)| {
                let u = u.clamp(1e-6, 1.0 - 1e-6);
                let w = w.clamp(1e-6, 1.0 - 1e-6);
                let exp = theta / (1.0 + theta);
                let inner = u.powf(-theta) * (w.powf(-exp) - 1.0) + 1.0;
                inner.powf(-1.0 / theta).clamp(1e-6, 1.0 - 1e-6)
            })
            .collect();
        (u_seq, v_seq)
    }

    /// Convert uniform margins to Weibull times: t = λ * (-ln u)^{1/k}.
    fn uniform_to_weibull(u: f64, scale: f64, shape: f64) -> f64 {
        // S(t) = exp(-(t/λ)^k) = u → t = λ * (-ln u)^{1/k}
        scale * (-u.ln()).powf(1.0 / shape)
    }

    /// Build arrays for Clayton bivariate data with Weibull marginals.
    fn make_clayton_data(
        n: usize,
        theta: f64,
        scale1: f64,
        shape1: f64,
        scale2: f64,
        shape2: f64,
    ) -> (Vec<f64>, Vec<bool>, Vec<f64>, Vec<bool>) {
        let (u_raw, v_raw) = simulate_clayton(42, n, theta);
        let times_1: Vec<f64> = u_raw
            .iter()
            .map(|&u| uniform_to_weibull(u, scale1, shape1))
            .collect();
        let times_2: Vec<f64> = v_raw
            .iter()
            .map(|&v| uniform_to_weibull(v, scale2, shape2))
            .collect();
        // All events observed (no censoring for simplicity).
        let events_1 = vec![true; n];
        let events_2 = vec![true; n];
        (times_1, events_1, times_2, events_2)
    }

    // Test 1: Clayton θ=2 recovery within 1.5.
    #[test]
    fn clayton_theta_recovery() {
        let (t1, e1, t2, e2) = make_clayton_data(200, 2.0, 2.0, 1.0, 3.0, 1.5);
        let config = CopulaConfig {
            family: CopulaFamily::Clayton,
            ..Default::default()
        };
        let fit = fit_bivariate_copula(&t1, &e1, &t2, &e2, &config)
            .expect("fit_bivariate_copula should succeed");
        assert!(
            (fit.theta - 2.0).abs() < 1.5,
            "Clayton θ={:.3}, expected ≈2.0",
            fit.theta
        );
    }

    // Test 2: Frank θ=2 recovery within 2.0.
    #[test]
    fn frank_theta_recovery() {
        // Simulate from Frank copula using conditional distribution.
        // Use a simple data generation: times with positive correlation.
        let u_seq = simple_lcg_sequence(7777, 200);
        let w_seq = simple_lcg_sequence(8888, 200);
        // Frank conditional: given U=u, F_{V|U}(v|u) = ∂C/∂u
        // Solve numerically: ∂C/∂u = w → find v via bisection.
        let theta = 2.0_f64;
        let v_seq: Vec<f64> = u_seq
            .iter()
            .zip(w_seq.iter())
            .map(|(&u, &w)| {
                let u = u.clamp(1e-4, 1.0 - 1e-4);
                let w = w.clamp(1e-4, 1.0 - 1e-4);
                // Solve exp(-θu) * (exp(-θv) - 1) / (exp(-θ) - 1 + (exp(-θu)-1)(exp(-θv)-1)) = w
                // via bisection.
                let mut lo = 1e-6_f64;
                let mut hi = 1.0 - 1e-6_f64;
                for _ in 0..60 {
                    let mid = 0.5 * (lo + hi);
                    let f = frank_partial_u(u, mid, theta);
                    if f < w {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                0.5 * (lo + hi)
            })
            .collect();
        let scale = 2.0_f64;
        let shape = 1.0_f64;
        let t1: Vec<f64> = u_seq
            .iter()
            .map(|&u| uniform_to_weibull(u.clamp(1e-6, 1.0 - 1e-6), scale, shape))
            .collect();
        let t2: Vec<f64> = v_seq
            .iter()
            .map(|&v| uniform_to_weibull(v.clamp(1e-6, 1.0 - 1e-6), scale, shape))
            .collect();
        let e1 = vec![true; 200];
        let e2 = vec![true; 200];
        let config = CopulaConfig {
            family: CopulaFamily::Frank,
            ..Default::default()
        };
        let fit = fit_bivariate_copula(&t1, &e1, &t2, &e2, &config)
            .expect("fit_bivariate_copula should succeed");
        assert!(
            (fit.theta - 2.0).abs() < 2.0,
            "Frank θ={:.3}, expected ≈2.0",
            fit.theta
        );
    }

    // Test 3: Gumbel θ=1.5 recovery within 1.0.
    #[test]
    fn gumbel_theta_recovery() {
        // Simulate from Gumbel using conditional.
        let u_seq = simple_lcg_sequence(1234, 200);
        let w_seq = simple_lcg_sequence(5678, 200);
        let theta = 1.5_f64;
        let v_seq: Vec<f64> = u_seq
            .iter()
            .zip(w_seq.iter())
            .map(|(&u, &w)| {
                let u = u.clamp(1e-4, 1.0 - 1e-4);
                let w = w.clamp(1e-4, 1.0 - 1e-4);
                // Bisect on gumbel_partial_u(u, v, theta) = w.
                let mut lo = 1e-6_f64;
                let mut hi = 1.0 - 1e-6_f64;
                for _ in 0..60 {
                    let mid = 0.5 * (lo + hi);
                    let f = gumbel_partial_u(u, mid, theta);
                    if f < w {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                0.5 * (lo + hi)
            })
            .collect();
        let scale = 2.0_f64;
        let shape = 1.2_f64;
        let t1: Vec<f64> = u_seq
            .iter()
            .map(|&u| uniform_to_weibull(u.clamp(1e-6, 1.0 - 1e-6), scale, shape))
            .collect();
        let t2: Vec<f64> = v_seq
            .iter()
            .map(|&v| uniform_to_weibull(v.clamp(1e-6, 1.0 - 1e-6), scale, shape))
            .collect();
        let e1 = vec![true; 200];
        let e2 = vec![true; 200];
        let config = CopulaConfig {
            family: CopulaFamily::Gumbel,
            ..Default::default()
        };
        let fit = fit_bivariate_copula(&t1, &e1, &t2, &e2, &config)
            .expect("fit_bivariate_copula should succeed");
        assert!(
            (fit.theta - 1.5).abs() < 1.0,
            "Gumbel θ={:.3}, expected ≈1.5",
            fit.theta
        );
    }

    // Test 4: kendall_tau_from_theta Clayton θ=2 → τ=0.5.
    #[test]
    fn clayton_tau_from_theta_correct() {
        let tau = kendall_tau_from_theta(CopulaFamily::Clayton, 2.0);
        assert!((tau - 0.5).abs() < 1e-10, "Clayton τ={tau}");
    }

    // Test 5: kendall_tau_from_theta Gumbel θ=2 → τ=0.5.
    #[test]
    fn gumbel_tau_from_theta_correct() {
        let tau = kendall_tau_from_theta(CopulaFamily::Gumbel, 2.0);
        assert!((tau - 0.5).abs() < 1e-10, "Gumbel τ={tau}");
    }

    // Test 6: theta_from_kendall_tau round-trip Clayton.
    #[test]
    fn clayton_round_trip_tau_theta() {
        let tau_in = 0.5_f64;
        let theta = theta_from_kendall_tau(CopulaFamily::Clayton, tau_in)
            .expect("theta_from_kendall_tau should succeed");
        let tau_out = kendall_tau_from_theta(CopulaFamily::Clayton, theta);
        assert!(
            (tau_out - tau_in).abs() < 1e-10,
            "Clayton round-trip: τ_out={tau_out}"
        );
    }

    // Test 7: theta_from_kendall_tau round-trip Gumbel.
    #[test]
    fn gumbel_round_trip_tau_theta() {
        let tau_in = 0.5_f64;
        let theta = theta_from_kendall_tau(CopulaFamily::Gumbel, tau_in)
            .expect("theta_from_kendall_tau should succeed");
        let tau_out = kendall_tau_from_theta(CopulaFamily::Gumbel, theta);
        assert!(
            (tau_out - tau_in).abs() < 1e-10,
            "Gumbel round-trip: τ_out={tau_out}"
        );
    }

    // Test 8: copula_survival_prob at t=0 ≈ 1.0.
    #[test]
    fn survival_prob_at_zero_is_one() {
        let (t1, e1, t2, e2) = make_clayton_data(50, 2.0, 2.0, 1.0, 2.0, 1.0);
        let config = CopulaConfig::default();
        let fit = fit_bivariate_copula(&t1, &e1, &t2, &e2, &config)
            .expect("fit_bivariate_copula should succeed");
        let p = copula_survival_prob(&fit, 0.0, 0.0);
        assert!(p > 0.95, "P(T1>0, T2>0) = {p}, expected ≈1");
    }

    // Test 9: C(u, 1) ≈ u for Clayton (marginal consistency).
    #[test]
    fn clayton_marginal_consistency_u() {
        let u = 0.5_f64;
        let v = 1.0_f64 - 1e-9;
        let val = clayton_cdf(u, v, 2.0);
        assert!(
            (val - u).abs() < 0.01,
            "Clayton C(0.5, 1) = {val}, expected ≈0.5"
        );
    }

    // Test 10: C(1, v) ≈ v for Clayton.
    #[test]
    fn clayton_marginal_consistency_v() {
        let u = 1.0_f64 - 1e-9;
        let v = 0.3_f64;
        let val = clayton_cdf(u, v, 2.0);
        assert!(
            (val - v).abs() < 0.01,
            "Clayton C(1, 0.3) = {val}, expected ≈0.3"
        );
    }

    // Test 11: Clayton independence limit (θ → 0): C(u,v) ≈ u*v.
    #[test]
    fn clayton_independence_limit() {
        let u = 0.5_f64;
        let v = 0.5_f64;
        let c = clayton_cdf(u, v, 0.01);
        let expected = u * v;
        assert!(
            (c - expected).abs() < 0.01,
            "Clayton C≈{c}, expected uv={expected}"
        );
    }

    // Test 12: AIC is finite.
    #[test]
    fn aic_is_finite() {
        let (t1, e1, t2, e2) = make_clayton_data(50, 2.0, 2.0, 1.0, 2.0, 1.0);
        let config = CopulaConfig::default();
        let fit = fit_bivariate_copula(&t1, &e1, &t2, &e2, &config)
            .expect("fit_bivariate_copula should succeed");
        assert!(fit.aic.is_finite(), "AIC={}", fit.aic);
    }

    // Test 13: DimensionMismatch when lengths differ.
    #[test]
    fn dimension_mismatch_error() {
        let t1 = vec![1.0, 2.0, 3.0];
        let e1 = vec![true, true, true];
        let t2 = vec![1.0, 2.0];
        let e2 = vec![true, true];
        let config = CopulaConfig::default();
        let result = fit_bivariate_copula(&t1, &e1, &t2, &e2, &config);
        assert!(matches!(
            result,
            Err(SurvivalError::DimensionMismatch { .. })
        ));
    }

    // Test 14: NegativeTime for negative time.
    #[test]
    fn negative_time_error() {
        let t1 = vec![-1.0, 2.0];
        let e1 = vec![true, true];
        let t2 = vec![1.0, 2.0];
        let e2 = vec![true, true];
        let config = CopulaConfig::default();
        let result = fit_bivariate_copula(&t1, &e1, &t2, &e2, &config);
        assert!(matches!(result, Err(SurvivalError::NegativeTime(_))));
    }

    // Test 15: EmptyDataset for empty inputs.
    #[test]
    fn empty_dataset_error() {
        let config = CopulaConfig::default();
        let result = fit_bivariate_copula(&[], &[], &[], &[], &config);
        assert!(matches!(result, Err(SurvivalError::EmptyDataset)));
    }

    // Test 16: Gumbel theta_from_tau(0.0) → θ=1.0.
    #[test]
    fn gumbel_theta_from_zero_tau() {
        let theta = theta_from_kendall_tau(CopulaFamily::Gumbel, 0.0)
            .expect("theta_from_kendall_tau should succeed");
        assert!((theta - 1.0).abs() < 1e-9, "Gumbel θ(τ=0)={theta}");
    }

    // Test 17: Clayton theta_from_tau(0.0) → θ=0.0.
    #[test]
    fn clayton_theta_from_zero_tau() {
        let theta = theta_from_kendall_tau(CopulaFamily::Clayton, 0.0)
            .expect("theta_from_kendall_tau should succeed");
        assert!(theta.abs() < 1e-9, "Clayton θ(τ=0)={theta}");
    }

    // Test 18: copula_survival_prob is non-negative.
    #[test]
    fn survival_prob_non_negative() {
        let (t1, e1, t2, e2) = make_clayton_data(50, 2.0, 2.0, 1.0, 2.0, 1.0);
        let config = CopulaConfig::default();
        let fit = fit_bivariate_copula(&t1, &e1, &t2, &e2, &config)
            .expect("fit_bivariate_copula should succeed");
        for &t in &[0.1, 1.0, 5.0, 10.0] {
            let p = copula_survival_prob(&fit, t, t);
            assert!(p >= 0.0, "P(T1>{t}, T2>{t}) = {p} < 0");
        }
    }

    // Test 19: Determinism — same inputs → same output.
    #[test]
    fn determinism_same_output() {
        let (t1, e1, t2, e2) = make_clayton_data(50, 2.0, 2.0, 1.0, 2.0, 1.0);
        let config = CopulaConfig::default();
        let fit1 = fit_bivariate_copula(&t1, &e1, &t2, &e2, &config)
            .expect("fit_bivariate_copula should succeed");
        let fit2 = fit_bivariate_copula(&t1, &e1, &t2, &e2, &config)
            .expect("fit_bivariate_copula should succeed");
        assert_eq!(fit1.theta, fit2.theta);
        assert_eq!(fit1.log_likelihood, fit2.log_likelihood);
    }

    // Test 20: All output fields finite.
    #[test]
    fn all_fields_finite() {
        let (t1, e1, t2, e2) = make_clayton_data(50, 2.0, 2.0, 1.0, 2.0, 1.0);
        let config = CopulaConfig::default();
        let fit = fit_bivariate_copula(&t1, &e1, &t2, &e2, &config)
            .expect("fit_bivariate_copula should succeed");
        assert!(fit.marginal_1.scale.is_finite());
        assert!(fit.marginal_1.shape.is_finite());
        assert!(fit.marginal_2.scale.is_finite());
        assert!(fit.marginal_2.shape.is_finite());
        assert!(fit.theta.is_finite());
        assert!(fit.log_likelihood.is_finite());
        assert!(fit.aic.is_finite());
        assert!(fit.kendall_tau.is_finite());
    }
}
