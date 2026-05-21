//! Generalised Linear Models (GLMs) with exponential-family distributions and
//! arbitrary link functions, fitted by Iteratively Reweighted Least Squares (IRLS).
//!
//! # Supported families
//! Gaussian, Poisson, Binomial, Gamma, InverseGaussian
//!
//! # Supported links
//! Identity, Log, Logit, Probit, ClogLog, Sqrt, Inverse, InverseSquared
//!
//! # Reference
//! McCullagh & Nelder (1989), *Generalized Linear Models* (2nd ed.);
//! Dobson & Barnett (2008), *An Introduction to Generalized Linear Models* (3rd ed.)

use crate::error::{StatsError, StatsResult};
use crate::special::betainc::gammp;
use crate::special::erf::{erf, erfinv};
use crate::special::gammaln::lgamma;

// ─────────────────────────────── Public enums ────────────────────────────────

/// Exponential-family distribution for the response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmFamily {
    Gaussian,
    Poisson,
    Binomial,
    Gamma,
    InverseGaussian,
}

/// Link function g such that g(μ) = η = Xβ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmLink {
    /// g(μ) = μ  (canonical for Gaussian)
    Identity,
    /// g(μ) = log(μ)  (canonical for Poisson)
    Log,
    /// g(μ) = log(μ/(1-μ))  (canonical for Binomial)
    Logit,
    /// g(μ) = Φ⁻¹(μ)  (alternative for Binomial)
    Probit,
    /// g(μ) = log(-log(1-μ))  (complementary log-log, alternative for Binomial)
    ClogLog,
    /// g(μ) = √μ  (alternative for Poisson)
    Sqrt,
    /// g(μ) = 1/μ  (canonical for Gamma)
    Inverse,
    /// g(μ) = 1/μ²  (canonical for InverseGaussian)
    InverseSquared,
}

// ───────────────────────────── Configuration ─────────────────────────────────

/// Configuration for GLM fitting.
#[derive(Debug, Clone)]
pub struct GlmConfig {
    pub family: GlmFamily,
    pub link: GlmLink,
    /// Maximum number of IRLS iterations (default 100).
    pub max_iter: usize,
    /// Convergence tolerance on the Euclidean norm of Δβ (default 1e-8).
    pub tol: f64,
    /// Whether to prepend an intercept column to the design matrix (default true).
    pub intercept: bool,
}

impl Default for GlmConfig {
    fn default() -> Self {
        Self {
            family: GlmFamily::Gaussian,
            link: GlmLink::Identity,
            max_iter: 100,
            tol: 1e-8,
            intercept: true,
        }
    }
}

impl GlmConfig {
    /// Construct a GLM config using the canonical link for `family`.
    pub fn canonical(family: GlmFamily) -> Self {
        let link = match family {
            GlmFamily::Gaussian => GlmLink::Identity,
            GlmFamily::Poisson => GlmLink::Log,
            GlmFamily::Binomial => GlmLink::Logit,
            GlmFamily::Gamma => GlmLink::Inverse,
            GlmFamily::InverseGaussian => GlmLink::InverseSquared,
        };
        Self {
            family,
            link,
            ..Default::default()
        }
    }
}

// ──────────────────────────── Fitted model ───────────────────────────────────

/// The result of fitting a GLM.
#[derive(Debug, Clone)]
pub struct GlmFit {
    /// Estimated coefficients β (intercept first, if `cfg.intercept == true`).
    pub coefficients: Vec<f64>,
    /// Fitted mean values μ̂ = g⁻¹(Xβ̂), length n_samples.
    pub fitted_values: Vec<f64>,
    /// Pearson residuals (y - μ) / √V(μ), length n_samples.
    pub residuals: Vec<f64>,
    /// Deviance residuals (signed √(2 × d_i)), length n_samples.
    pub deviance_residuals: Vec<f64>,
    /// Null deviance: deviance when all μ = ȳ.
    pub null_deviance: f64,
    /// Residual deviance of fitted model.
    pub residual_deviance: f64,
    /// Pearson dispersion estimate φ̂ = Σ(y-μ)² / (V(μ)(n-p)).
    pub dispersion: f64,
    /// McFadden pseudo-R² = 1 - residual_deviance / null_deviance.
    pub pseudo_r2: f64,
    /// Akaike information criterion: AIC = -2 log L̂ + 2p.
    pub aic: f64,
    /// Number of IRLS iterations taken.
    pub n_iter: usize,
    /// Whether IRLS converged within `max_iter`.
    pub converged: bool,
    /// Asymptotic standard errors √diag((X^T W X)⁻¹), length p.
    pub std_errors: Vec<f64>,
    /// Wald z-statistics: coefficient / std_error, length p.
    pub z_stats: Vec<f64>,
    /// Two-sided p-values from the standard-normal Wald test, length p.
    pub p_values: Vec<f64>,
}

// ─────────────────────────── Link functions ──────────────────────────────────

/// Apply the link function: compute η = g(μ).
#[inline]
fn apply_link(mu: f64, link: GlmLink) -> f64 {
    match link {
        GlmLink::Identity => mu,
        GlmLink::Log => {
            let mu_safe = mu.max(f64::EPSILON);
            mu_safe.ln()
        }
        GlmLink::Logit => {
            let mu_c = mu.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
            (mu_c / (1.0 - mu_c)).ln()
        }
        GlmLink::Probit => {
            let mu_c = mu.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
            probit_quantile(mu_c)
        }
        GlmLink::ClogLog => {
            let mu_c = mu.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
            (-(1.0 - mu_c).ln()).ln()
        }
        GlmLink::Sqrt => mu.max(0.0).sqrt(),
        GlmLink::Inverse => {
            let mu_safe = if mu.abs() < 1e-8 {
                1e-8_f64.copysign(mu + 1e-15)
            } else {
                mu
            };
            1.0 / mu_safe
        }
        GlmLink::InverseSquared => {
            let mu_safe = mu.max(f64::EPSILON);
            1.0 / (mu_safe * mu_safe)
        }
    }
}

/// Apply the inverse link: compute μ = g⁻¹(η).
#[inline]
fn apply_inv_link(eta: f64, link: GlmLink) -> f64 {
    match link {
        GlmLink::Identity => eta,
        GlmLink::Log => eta.exp(),
        GlmLink::Logit => {
            // numerically stable sigmoid
            if eta >= 0.0 {
                1.0 / (1.0 + (-eta).exp())
            } else {
                let e = eta.exp();
                e / (1.0 + e)
            }
        }
        GlmLink::Probit => {
            // Φ(η) via erf
            0.5 * (1.0 + erf(eta / std::f64::consts::SQRT_2))
        }
        GlmLink::ClogLog => 1.0 - (-eta.exp()).exp(),
        GlmLink::Sqrt => {
            let eta_safe = eta.max(0.0);
            eta_safe * eta_safe
        }
        GlmLink::Inverse => {
            // clip away from zero to avoid singularities
            if eta.abs() < 1e-8 {
                1.0 / 1e-8_f64.copysign(eta + 1e-15)
            } else {
                1.0 / eta
            }
        }
        GlmLink::InverseSquared => {
            // μ = 1/√η; η must be > 0
            let eta_safe = eta.max(f64::EPSILON);
            1.0 / eta_safe.sqrt()
        }
    }
}

/// Derivative ∂μ/∂η (the reciprocal of the link derivative evaluated at η).
#[inline]
fn dmu_deta(eta: f64, link: GlmLink) -> f64 {
    match link {
        GlmLink::Identity => 1.0,
        GlmLink::Log => eta.exp(),
        GlmLink::Logit => {
            let mu = apply_inv_link(eta, link);
            mu * (1.0 - mu)
        }
        GlmLink::Probit => probit_pdf(eta),
        GlmLink::ClogLog => {
            // g⁻¹(η) = 1 - exp(-exp(η)), so dμ/dη = exp(η - exp(η))
            (eta - eta.exp()).exp()
        }
        GlmLink::Sqrt => {
            // μ = η², dμ/dη = 2η
            2.0 * eta.max(0.0)
        }
        GlmLink::Inverse => {
            // dμ/dη = -1/η²
            let eta_safe = if eta.abs() < 1e-8 {
                1e-8_f64.copysign(eta + 1e-15)
            } else {
                eta
            };
            -1.0 / (eta_safe * eta_safe)
        }
        GlmLink::InverseSquared => {
            // μ = η^{-1/2}, dμ/dη = -1/(2 η^{3/2})
            let eta_safe = eta.max(f64::EPSILON);
            -0.5 / (eta_safe * eta_safe.sqrt())
        }
    }
}

// ─────────────────── Probit approximation (Acklam) ───────────────────────────

/// Standard normal quantile Φ⁻¹(p) via erfinv: Φ⁻¹(p) = √2 · erf⁻¹(2p-1).
fn probit_quantile(p: f64) -> f64 {
    // Use the erfinv already implemented in special::erf
    let x = 2.0 * p - 1.0;
    // erfinv may fail only for |x|>=1 which we clipped away
    match erfinv(x) {
        Ok(v) => std::f64::consts::SQRT_2 * v,
        Err(_) => {
            // fallback rational (Acklam) for extreme p
            acklam_probit(p)
        }
    }
}

/// Acklam rational approximation for Φ⁻¹(p). Accurate to ~1e-9.
fn acklam_probit(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    let p_lo = 0.02425;
    let p_hi = 1.0 - p_lo;

    if p < p_lo {
        let q = (-2.0 * p.ln()).sqrt();
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        num / den
    } else if p <= p_hi {
        let q = p - 0.5;
        let r = q * q;
        let num = ((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5];
        let den = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0;
        q * num / den
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        -(num / den)
    }
}

/// Standard normal PDF φ(x) = (2π)^{-1/2} exp(-x²/2).
#[inline]
fn probit_pdf(x: f64) -> f64 {
    let ln_val = -0.5 * x * x - 0.5 * (2.0 * std::f64::consts::PI).ln();
    ln_val.exp()
}

// ─────────────────── Family-specific functions ────────────────────────────────

/// Variance function V(μ) for the given family.
#[inline]
fn variance(mu: f64, family: GlmFamily) -> f64 {
    match family {
        GlmFamily::Gaussian => 1.0,
        GlmFamily::Poisson => mu.max(f64::EPSILON),
        GlmFamily::Binomial => {
            let mu_c = mu.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
            mu_c * (1.0 - mu_c)
        }
        GlmFamily::Gamma => {
            let mu_safe = mu.max(f64::EPSILON);
            mu_safe * mu_safe
        }
        GlmFamily::InverseGaussian => {
            let mu_safe = mu.max(f64::EPSILON);
            mu_safe * mu_safe * mu_safe
        }
    }
}

/// Per-observation deviance contribution d(y, μ) (the raw, not doubled, version).
fn deviance_one(y: f64, mu: f64, family: GlmFamily) -> f64 {
    match family {
        GlmFamily::Gaussian => {
            let d = y - mu;
            d * d
        }
        GlmFamily::Poisson => {
            let mu_s = mu.max(f64::EPSILON);
            if y <= 0.0 {
                2.0 * mu_s
            } else {
                2.0 * (y * (y / mu_s).ln() - (y - mu_s))
            }
        }
        GlmFamily::Binomial => {
            let mu_c = mu.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
            let term1 = if y > f64::EPSILON {
                y * (y / mu_c).ln()
            } else {
                0.0
            };
            let term2 = if (1.0 - y) > f64::EPSILON {
                (1.0 - y) * ((1.0 - y) / (1.0 - mu_c)).ln()
            } else {
                0.0
            };
            2.0 * (term1 + term2)
        }
        GlmFamily::Gamma => {
            let mu_s = mu.max(f64::EPSILON);
            let y_s = y.max(f64::EPSILON);
            2.0 * (-(y_s / mu_s).ln() + (y_s - mu_s) / mu_s)
        }
        GlmFamily::InverseGaussian => {
            let mu_s = mu.max(f64::EPSILON);
            let y_s = y.max(f64::EPSILON);
            (y_s - mu_s) * (y_s - mu_s) / (mu_s * mu_s * y_s)
        }
    }
}

/// Log-likelihood contribution log p(y; μ, φ) for one observation.
fn log_likelihood_one(y: f64, mu: f64, phi: f64, family: GlmFamily) -> f64 {
    let phi_safe = phi.max(f64::EPSILON);
    match family {
        GlmFamily::Gaussian => {
            // -½ log(2πφ) - (y-μ)²/(2φ)
            let d = y - mu;
            -0.5 * (2.0 * std::f64::consts::PI * phi_safe).ln() - d * d / (2.0 * phi_safe)
        }
        GlmFamily::Poisson => {
            // y log(μ) - μ - log(y!)  (φ=1 canonical)
            let mu_s = mu.max(f64::EPSILON);
            let log_fac_y = lgamma(y + 1.0); // Γ(y+1) = y!
            y * mu_s.ln() - mu_s - log_fac_y
        }
        GlmFamily::Binomial => {
            // y log(μ) + (1-y) log(1-μ)  (n=1)
            let mu_c = mu.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
            let t1 = if y > f64::EPSILON { y * mu_c.ln() } else { 0.0 };
            let t2 = if (1.0 - y) > f64::EPSILON {
                (1.0 - y) * (1.0 - mu_c).ln()
            } else {
                0.0
            };
            t1 + t2
        }
        GlmFamily::Gamma => {
            // Gamma with mean μ and dispersion φ (shape = 1/φ):
            // log p = (1/φ)(log(y/μ) - y/μ + 1)/φ + constant  → use deviance
            // Full: -(1/φ) d(y,μ)/2 - log Γ(1/φ) + (1/φ-1)log(y) - (1/φ)log(φ)
            let mu_s = mu.max(f64::EPSILON);
            let y_s = y.max(f64::EPSILON);
            let inv_phi = 1.0 / phi_safe;
            inv_phi * (inv_phi.ln() + (y_s / mu_s).ln() - y_s / mu_s) - lgamma(inv_phi)
                + (inv_phi - 1.0) * y_s.ln()
                - y_s / mu_s
                + inv_phi * mu_s.ln()
        }
        GlmFamily::InverseGaussian => {
            // IG(μ, λ=1/φ): log p = ½ log(λ/(2πy³)) - λ(y-μ)²/(2μ²y)
            let mu_s = mu.max(f64::EPSILON);
            let y_s = y.max(f64::EPSILON);
            let lambda = 1.0 / phi_safe;
            0.5 * (lambda / (2.0 * std::f64::consts::PI * y_s * y_s * y_s)).ln()
                - lambda * (y_s - mu_s) * (y_s - mu_s) / (2.0 * mu_s * mu_s * y_s)
        }
    }
}

// ─────────────────────── Initialization of μ ─────────────────────────────────

fn init_mu(y: f64, family: GlmFamily) -> f64 {
    match family {
        GlmFamily::Gaussian => y,
        GlmFamily::Poisson => y + 0.1,
        GlmFamily::Binomial => ((y + 0.5) / 2.0).clamp(0.05, 0.95),
        GlmFamily::Gamma => y.max(1e-6),
        GlmFamily::InverseGaussian => y.max(1e-6),
    }
}

// ─────────────────────── WLS solver (Cholesky) ───────────────────────────────

/// Solve the WLS normal equations (X^T W X) β = X^T W z using Cholesky factorisation.
///
/// A small ridge penalty (1e-10 × max diagonal) is added to X^T W X to ensure positive
/// definiteness in near-rank-deficient cases (e.g. with all-zero feature columns or at
/// separation in Binomial models).
///
/// Returns `None` if the matrix is numerically singular even after regularisation.
fn wls_solve(x: &[f64], z: &[f64], w: &[f64], n: usize, p: usize) -> Option<Vec<f64>> {
    // Build A = X^T W X  (p × p, symmetric positive definite)
    let mut a = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in i..p {
            let mut acc = 0.0;
            for k in 0..n {
                acc += x[k * p + i] * w[k] * x[k * p + j];
            }
            a[i * p + j] = acc;
            a[j * p + i] = acc;
        }
    }
    // Ridge regularisation: add λ I where λ = 1e-10 × max|diag(A)|
    let diag_max = (0..p).map(|j| a[j * p + j].abs()).fold(0.0_f64, f64::max);
    let ridge = (diag_max * 1e-10).max(1e-14);
    for j in 0..p {
        a[j * p + j] += ridge;
    }
    // Build b = X^T W z  (p-vector)
    let mut b = vec![0.0_f64; p];
    for i in 0..p {
        let mut acc = 0.0;
        for k in 0..n {
            acc += x[k * p + i] * w[k] * z[k];
        }
        b[i] = acc;
    }
    // Cholesky factorisation of A (in-place lower-triangular L).
    // L stored in the lower triangle of `a`.
    for j in 0..p {
        // Diagonal element
        let mut s = a[j * p + j];
        for k in 0..j {
            s -= a[j * p + k] * a[j * p + k];
        }
        if s <= 0.0 {
            return None; // not positive definite even after ridge
        }
        let l_jj = s.sqrt();
        a[j * p + j] = l_jj;
        // Off-diagonal elements in column j, rows j+1..p
        for i in (j + 1)..p {
            let mut t = a[i * p + j];
            for k in 0..j {
                t -= a[i * p + k] * a[j * p + k];
            }
            a[i * p + j] = t / l_jj;
        }
    }
    // Forward substitution: L y = b
    let mut y_vec = vec![0.0_f64; p];
    for i in 0..p {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * p + k] * y_vec[k];
        }
        y_vec[i] = s / a[i * p + i];
    }
    // Back substitution: L^T beta = y
    let mut beta = vec![0.0_f64; p];
    for i in (0..p).rev() {
        let mut s = y_vec[i];
        for k in (i + 1)..p {
            s -= a[k * p + i] * beta[k]; // L^T[i,k] = L[k,i]
        }
        beta[i] = s / a[i * p + i];
    }
    Some(beta)
}

/// Invert (X^T W X) to obtain (X^T W X)^{-1}, using the Cholesky L already computed.
///
/// `l` is the lower-triangular factor (stored in lower triangle of a p×p column-major matrix).
/// Returns the full inverse (p×p row-major).
fn cholesky_inverse(l: &[f64], p: usize) -> Option<Vec<f64>> {
    // Invert L by forward substitution (L Linv = I column by column)
    let mut linv = vec![0.0_f64; p * p];
    for j in 0..p {
        linv[j * p + j] = 1.0 / l[j * p + j];
        for i in (j + 1)..p {
            let mut s = 0.0;
            for k in j..i {
                s -= l[i * p + k] * linv[k * p + j];
            }
            linv[i * p + j] = s / l[i * p + i];
        }
    }
    // (X^T W X)^{-1} = (L L^T)^{-1} = L^{-T} L^{-1}
    // = (Linv^T) (Linv)
    let mut inv = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in i..p {
            let mut s = 0.0;
            for k in j..p {
                // Linv^T[i,k] = Linv[k,i]; Linv[k,j] stored
                s += linv[k * p + i] * linv[k * p + j];
            }
            inv[i * p + j] = s;
            inv[j * p + i] = s;
        }
    }
    Some(inv)
}

// ───────────────────────────── Normal SF ─────────────────────────────────────

/// Standard normal survival function P(Z > |z|) for two-sided p-value.
fn normal_two_sided_p(z: f64) -> f64 {
    let az = z.abs() / std::f64::consts::SQRT_2;
    // P(|Z| > |z|) = erfc(|z|/√2)
    let erfc_val = 1.0 - erf(az);
    erfc_val.clamp(0.0, 1.0)
}

/// Chi-squared survival function P(χ²_df > x) for LRT p-value.
fn chi2_sf(x: f64, df: usize) -> StatsResult<f64> {
    if x <= 0.0 {
        return Ok(1.0);
    }
    // Q(df/2, x/2) = 1 - P(df/2, x/2)
    let p = gammp(df as f64 / 2.0, x / 2.0)?;
    Ok((1.0 - p).clamp(0.0, 1.0))
}

// ─────────────────────── Build extended design matrix ────────────────────────

/// Prepend intercept column (all-ones) to x if `intercept == true`.
/// Returns (extended_x, extended_p).
fn build_design(x: &[f64], n: usize, n_features: usize, intercept: bool) -> (Vec<f64>, usize) {
    if !intercept {
        return (x.to_vec(), n_features);
    }
    let p = n_features + 1;
    let mut xd = vec![0.0_f64; n * p];
    for k in 0..n {
        xd[k * p] = 1.0; // intercept
        for j in 0..n_features {
            xd[k * p + j + 1] = x[k * n_features + j];
        }
    }
    (xd, p)
}

// ─────────────────────────── Main GLM fit ────────────────────────────────────

/// Fit a GLM by Iteratively Reweighted Least Squares (IRLS).
///
/// `x` — design matrix, n×n_features row-major **without** the intercept column.
/// `y` — response vector of length n.
pub fn glm_fit(
    x: &[f64],
    y: &[f64],
    n_samples: usize,
    n_features: usize,
    cfg: &GlmConfig,
) -> StatsResult<GlmFit> {
    // Input validation
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
    // Check for non-finite values
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

    // Build (possibly extended) design matrix
    let (xd, p) = build_design(x, n_samples, n_features, cfg.intercept);

    if n_samples < p {
        return Err(StatsError::InsufficientSampleSize {
            got: n_samples,
            need: p,
        });
    }

    // ── IRLS ──────────────────────────────────────────────────────────────────
    // Initialise μ
    let mut mu: Vec<f64> = y.iter().map(|&yi| init_mu(yi, cfg.family)).collect();

    // Initialise β from the link-transformed mean
    let mut beta = vec![0.0_f64; p];
    // Start with β₀ = g(mean(μ)) for intercept, others zero
    if cfg.intercept && p > 0 {
        let mu_bar = mu.iter().sum::<f64>() / n_samples as f64;
        beta[0] = apply_link(mu_bar, cfg.link);
    }

    let mut converged = false;
    let mut n_iter = 0_usize;
    let mut last_xtwx_chol = vec![0.0_f64; p * p]; // for SE computation

    for iter in 0..cfg.max_iter {
        n_iter = iter + 1;

        // Step 1: linear predictor η = Xβ
        let mut eta = vec![0.0_f64; n_samples];
        for k in 0..n_samples {
            let mut acc = 0.0;
            for j in 0..p {
                acc += xd[k * p + j] * beta[j];
            }
            eta[k] = acc;
        }

        // Step 2: μ = g⁻¹(η) — keep μ from previous iter for first step
        for k in 0..n_samples {
            mu[k] = apply_inv_link(eta[k], cfg.link);
        }

        // Step 3: working weights  w_i = (∂μ/∂η)² / V(μ)
        let mut w = vec![0.0_f64; n_samples];
        for k in 0..n_samples {
            let dm = dmu_deta(eta[k], cfg.link);
            let v = variance(mu[k], cfg.family);
            w[k] = (dm * dm) / v.max(f64::EPSILON);
            if !w[k].is_finite() || w[k] < 0.0 {
                w[k] = 0.0;
            }
        }

        // Step 4: adjusted response z_i = η_i + (y_i - μ_i) * (∂η/∂μ)
        //         ∂η/∂μ = 1 / (∂μ/∂η)
        let mut z_adj = vec![0.0_f64; n_samples];
        for k in 0..n_samples {
            let dm = dmu_deta(eta[k], cfg.link);
            let deta_dmu = if dm.abs() < 1e-15 {
                1.0 / 1e-15
            } else {
                1.0 / dm
            };
            z_adj[k] = eta[k] + (y[k] - mu[k]) * deta_dmu;
            if !z_adj[k].is_finite() {
                z_adj[k] = eta[k];
            }
        }

        // Step 5: solve WLS (X^T W X) β_new = X^T W z
        let beta_new = wls_solve(&xd, &z_adj, &w, n_samples, p).ok_or_else(|| {
            StatsError::SingularMatrix(format!(
                "IRLS WLS step at iteration {}: X^T W X is singular",
                iter + 1
            ))
        })?;

        // Check convergence: ‖β_new - β_old‖₂
        let diff_norm: f64 = beta_new
            .iter()
            .zip(beta.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();

        // Re-build X^T W X Cholesky for SE estimation.
        // Apply the same ridge regularisation as in wls_solve for consistency.
        {
            let mut a = vec![0.0_f64; p * p];
            for i in 0..p {
                for j in i..p {
                    let mut acc = 0.0;
                    for k in 0..n_samples {
                        acc += xd[k * p + i] * w[k] * xd[k * p + j];
                    }
                    a[i * p + j] = acc;
                    a[j * p + i] = acc;
                }
            }
            // Same ridge as wls_solve
            let diag_max_se = (0..p).map(|j| a[j * p + j].abs()).fold(0.0_f64, f64::max);
            let ridge_se = (diag_max_se * 1e-10).max(1e-14);
            for j in 0..p {
                a[j * p + j] += ridge_se;
            }
            // Compute Cholesky
            let mut ok = true;
            for j in 0..p {
                let mut s = a[j * p + j];
                for k in 0..j {
                    s -= a[j * p + k] * a[j * p + k];
                }
                if s <= 0.0 {
                    ok = false;
                    break;
                }
                a[j * p + j] = s.sqrt();
                for i in (j + 1)..p {
                    let mut t = a[i * p + j];
                    for k in 0..j {
                        t -= a[i * p + k] * a[j * p + k];
                    }
                    a[i * p + j] = t / a[j * p + j];
                }
            }
            if ok {
                last_xtwx_chol = a;
            }
        }

        beta = beta_new;

        if diff_norm < cfg.tol {
            converged = true;
            break;
        }
    }

    // ── Final fitted values and residuals ─────────────────────────────────────
    let mut eta_final = vec![0.0_f64; n_samples];
    for k in 0..n_samples {
        let mut acc = 0.0;
        for j in 0..p {
            acc += xd[k * p + j] * beta[j];
        }
        eta_final[k] = acc;
        mu[k] = apply_inv_link(eta_final[k], cfg.link);
    }

    // Pearson residuals
    let pearson_resid: Vec<f64> = y
        .iter()
        .zip(mu.iter())
        .map(|(&yi, &mui)| {
            let v = variance(mui, cfg.family);
            (yi - mui) / v.max(f64::EPSILON).sqrt()
        })
        .collect();

    // Deviance residuals: sign(y-μ) √d_i
    let dev_resid: Vec<f64> = y
        .iter()
        .zip(mu.iter())
        .map(|(&yi, &mui)| {
            let di = deviance_one(yi, mui, cfg.family).max(0.0);
            let sign = if yi >= mui { 1.0 } else { -1.0 };
            sign * di.sqrt()
        })
        .collect();

    // Residual deviance = Σ d_i
    let residual_deviance: f64 = y
        .iter()
        .zip(mu.iter())
        .map(|(&yi, &mui)| deviance_one(yi, mui, cfg.family))
        .sum();

    // Null deviance: fit intercept-only model (μ_null = ȳ for all obs)
    let y_bar = y.iter().sum::<f64>() / n_samples as f64;
    let mu_null = match cfg.family {
        GlmFamily::Poisson => y_bar.max(f64::EPSILON),
        GlmFamily::Binomial => y_bar.clamp(f64::EPSILON, 1.0 - f64::EPSILON),
        GlmFamily::Gamma => y_bar.max(f64::EPSILON),
        GlmFamily::InverseGaussian => y_bar.max(f64::EPSILON),
        GlmFamily::Gaussian => y_bar,
    };
    let null_deviance: f64 = y
        .iter()
        .map(|&yi| deviance_one(yi, mu_null, cfg.family))
        .sum();

    // Dispersion: Pearson chi² / (n - p)
    let df = n_samples.saturating_sub(p);
    let pearson_chi2: f64 = y
        .iter()
        .zip(mu.iter())
        .map(|(&yi, &mui)| {
            let v = variance(mui, cfg.family);
            (yi - mui) * (yi - mui) / v.max(f64::EPSILON)
        })
        .sum();
    let dispersion = if df > 0 {
        pearson_chi2 / df as f64
    } else {
        1.0
    };

    // Pseudo-R² (McFadden: 1 - L_fit / L_null = 1 - ResidDev / NullDev)
    let pseudo_r2 = if null_deviance > f64::EPSILON {
        (1.0 - residual_deviance / null_deviance).clamp(0.0, 1.0)
    } else {
        1.0
    };

    // AIC = -2 Σ log_lik + 2p  (using dispersion=1 for Poisson/Binomial)
    let phi_for_ll = match cfg.family {
        GlmFamily::Poisson | GlmFamily::Binomial => 1.0,
        _ => dispersion.max(f64::EPSILON),
    };
    let log_lik: f64 = y
        .iter()
        .zip(mu.iter())
        .map(|(&yi, &mui)| log_likelihood_one(yi, mui, phi_for_ll, cfg.family))
        .sum();
    let aic = -2.0 * log_lik + 2.0 * p as f64;

    // ── Standard errors from (X^T W X)^{-1} ──────────────────────────────────
    let inv_opt = cholesky_inverse(&last_xtwx_chol, p);
    let (std_errors, z_stats, p_values) = if let Some(inv) = inv_opt {
        let se: Vec<f64> = (0..p).map(|j| inv[j * p + j].max(0.0).sqrt()).collect();
        let z: Vec<f64> = beta
            .iter()
            .zip(se.iter())
            .map(|(&b, &s)| if s > f64::EPSILON { b / s } else { 0.0 })
            .collect();
        let pv: Vec<f64> = z.iter().map(|&zi| normal_two_sided_p(zi)).collect();
        (se, z, pv)
    } else {
        // Fallback: all zeros
        (vec![0.0; p], vec![0.0; p], vec![1.0; p])
    };

    Ok(GlmFit {
        coefficients: beta,
        fitted_values: mu,
        residuals: pearson_resid,
        deviance_residuals: dev_resid,
        null_deviance,
        residual_deviance,
        dispersion,
        pseudo_r2,
        aic,
        n_iter,
        converged,
        std_errors,
        z_stats,
        p_values,
    })
}

// ─────────────────────── Prediction ─────────────────────────────────────────

/// Predict on new data using a fitted GLM.
///
/// `x_new` — new design matrix, n_new×n_features row-major (WITHOUT intercept column).
/// `on_link_scale` — if `true` return linear predictor η; if `false` return μ = g⁻¹(η).
pub fn glm_predict(
    fit: &GlmFit,
    x_new: &[f64],
    n_new: usize,
    n_features: usize,
    cfg: &GlmConfig,
    on_link_scale: bool,
) -> StatsResult<Vec<f64>> {
    if n_new == 0 {
        return Err(StatsError::EmptyInput);
    }
    if x_new.len() != n_new * n_features {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n_new, n_features],
            got: vec![x_new.len()],
        });
    }
    // Determine p from fit
    let p = fit.coefficients.len();
    // Build extended design for new data
    let (xd_new, p_new) = build_design(x_new, n_new, n_features, cfg.intercept);
    if p_new != p {
        return Err(StatsError::DimensionMismatch { a: p_new, b: p });
    }
    let mut out = vec![0.0_f64; n_new];
    for k in 0..n_new {
        let mut eta = 0.0;
        for j in 0..p {
            eta += xd_new[k * p + j] * fit.coefficients[j];
        }
        out[k] = if on_link_scale {
            eta
        } else {
            apply_inv_link(eta, cfg.link)
        };
    }
    Ok(out)
}

// ──────────────────────── Likelihood-Ratio Test ───────────────────────────────

/// Likelihood-ratio test comparing a reduced model against a full model.
///
/// Returns `(chi2_statistic, p_value)`.
/// `df_diff` should be `p_full - p_reduced`.
pub fn glm_lrt(fit_reduced: &GlmFit, fit_full: &GlmFit, df_diff: usize) -> StatsResult<(f64, f64)> {
    if df_diff == 0 {
        return Err(StatsError::DegreesOfFreedomZero);
    }
    // LRT statistic = residual_deviance_reduced - residual_deviance_full
    let chi2 = (fit_reduced.residual_deviance - fit_full.residual_deviance).max(0.0);
    let p_val = chi2_sf(chi2, df_diff)?;
    Ok((chi2, p_val))
}

// ────────────────────── Score (Rao) test ─────────────────────────────────────

/// Score (Rao) test statistic for each covariate evaluated under the null hypothesis β = 0.
///
/// Returns a vector of score statistics (one per covariate, excluding intercept if present).
/// Score statistic for covariate j: S_j = (U_j / √I_jj)  where U = X^T (y - μ_0) is the score
/// and I = X^T W_0 X is the information matrix under H_0.
pub fn glm_score_test(
    x: &[f64],
    y: &[f64],
    n_samples: usize,
    n_features: usize,
    cfg: &GlmConfig,
) -> StatsResult<Vec<f64>> {
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

    // Fit null model (intercept only, or constant mean)
    let y_bar = y.iter().sum::<f64>() / n_samples as f64;
    let mu0 = init_mu(y_bar, cfg.family);

    // Under H_0: μ = μ₀ for all obs; η₀ = g(μ₀)
    let eta0 = apply_link(mu0, cfg.link);
    let dm0 = dmu_deta(eta0, cfg.link);
    let v0 = variance(mu0, cfg.family);
    // w₀ = (∂μ/∂η)² / V(μ₀)
    let w0 = (dm0 * dm0) / v0.max(f64::EPSILON);
    // ∂η/∂μ at μ₀
    let deta_dmu0 = if dm0.abs() < 1e-15 { 0.0 } else { 1.0 / dm0 };

    // Score vector:
    // U_j = ∂ℓ/∂β_j = Σ_i x_{ij} (y_i - μ₀) * dm0 / V(μ₀)
    // (evaluated at β=0, hence μ=μ₀ for all i)
    //
    // Information diagonal:
    // I_jj = Σ_i x_{ij}² * w₀   where w₀ = dm0² / V(μ₀)
    //
    // Score statistic: S_j = U_j / √I_jj
    let _ = deta_dmu0; // used indirectly via dm0 in the score formula below

    let mut scores = vec![0.0_f64; n_features];
    for j in 0..n_features {
        let u_j: f64 = y
            .iter()
            .enumerate()
            .map(|(k, &yi)| x[k * n_features + j] * (yi - mu0) * dm0 / v0.max(f64::EPSILON))
            .sum();
        let info_jj: f64 = (0..n_samples)
            .map(|k| x[k * n_features + j] * x[k * n_features + j] * w0)
            .sum();
        scores[j] = if info_jj > f64::EPSILON {
            u_j / info_jj.sqrt()
        } else {
            0.0
        };
    }
    Ok(scores)
}

// ─────────────────────────────── Tests ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1. Gaussian(identity) ≈ OLS ──────────────────────────────────────────
    #[test]
    fn glm_gaussian_identity_matches_ols() {
        // y = 1 + 2x + noise-free
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 1.0 + 2.0 * x).collect();
        let x_mat: Vec<f64> = xs.to_vec(); // n×1 (no intercept col)
        let cfg = GlmConfig::canonical(GlmFamily::Gaussian);
        let fit = glm_fit(&x_mat, &ys, 10, 1, &cfg).expect("fit ok");
        // coefficients[0]=intercept ≈ 1, coefficients[1]=slope ≈ 2
        assert!(
            (fit.coefficients[0] - 1.0).abs() < 1e-4,
            "intercept mismatch"
        );
        assert!((fit.coefficients[1] - 2.0).abs() < 1e-4, "slope mismatch");
        assert!(fit.converged, "should converge");
    }

    // ── 2. Poisson(log) on count data ─────────────────────────────────────────
    #[test]
    fn glm_poisson_log_count_data() {
        // Simulate: log(μ) = 0.5 + 0.3 x
        let xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| (0.5 + 0.3 * x).exp().round()).collect();
        let x_mat: Vec<f64> = xs.clone();
        let cfg = GlmConfig::canonical(GlmFamily::Poisson);
        let fit = glm_fit(&x_mat, &ys, 20, 1, &cfg).expect("poisson fit");
        // Slope should be positive
        assert!(fit.coefficients[1] > 0.0, "slope should be positive");
        assert!(fit.converged);
    }

    // ── 3. Binomial(logit) reproduces logistic regression ─────────────────────
    #[test]
    fn glm_binomial_logit_matches_logistic() {
        // Use soft (probability) response values to avoid complete separation.
        // logit(p) = -1 + 0.8 x  →  p = σ(-1 + 0.8x)
        // We supply the probability directly (GLM treats this as fractional response).
        let xs: Vec<f64> = (0..30).map(|i| (i as f64 - 15.0) * 0.3).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| {
                let eta = -0.5 + 0.4 * x;
                // Clamp away from 0/1 to prevent exact separation
                (1.0 / (1.0 + (-eta).exp())).clamp(0.05, 0.95)
            })
            .collect();
        let x_mat: Vec<f64> = xs.clone();
        let cfg = GlmConfig::canonical(GlmFamily::Binomial);
        let fit = glm_fit(&x_mat, &ys, 30, 1, &cfg).expect("binomial fit");
        assert!(fit.converged, "binomial logit should converge");
        // Slope positive (more x → higher probability)
        assert!(fit.coefficients[1] > 0.0, "slope should be positive");
    }

    // ── 4. Gamma(inverse) on positive skewed data ─────────────────────────────
    #[test]
    fn glm_gamma_inverse_link() {
        // g(μ) = 1/μ = α + β x → μ = 1/(α + βx)
        let xs: Vec<f64> = (1..=20).map(|i| i as f64 * 0.3).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 1.0 / (0.5 + 0.2 * x)).collect();
        let x_mat: Vec<f64> = xs.clone();
        let cfg = GlmConfig::canonical(GlmFamily::Gamma);
        let fit = glm_fit(&x_mat, &ys, 20, 1, &cfg).expect("gamma fit");
        assert!(fit.converged, "Gamma should converge");
        assert!(fit.residual_deviance >= 0.0);
    }

    // ── 5. InverseGaussian convergence ───────────────────────────────────────
    #[test]
    fn glm_inverse_gaussian_fit() {
        // g(μ) = 1/μ² = α + βx
        let xs: Vec<f64> = (1..=15).map(|i| i as f64 * 0.2).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| (1.0 / (0.5 + 0.1 * x)).sqrt()).collect();
        let x_mat: Vec<f64> = xs.clone();
        let cfg = GlmConfig::canonical(GlmFamily::InverseGaussian);
        let fit = glm_fit(&x_mat, &ys, 15, 1, &cfg).expect("ig fit");
        assert!(fit.converged);
        assert!(fit.coefficients.len() == 2);
    }

    // ── 6. Poisson with sqrt link (non-canonical) ─────────────────────────────
    #[test]
    fn glm_poisson_sqrt_link() {
        // √μ = 1 + 0.5 x → μ = (1 + 0.5x)²
        let xs: Vec<f64> = (0..15).map(|i| i as f64 * 0.4).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| (1.0 + 0.5 * x) * (1.0 + 0.5 * x))
            .collect();
        let x_mat: Vec<f64> = xs.clone();
        let cfg = GlmConfig {
            family: GlmFamily::Poisson,
            link: GlmLink::Sqrt,
            ..Default::default()
        };
        let fit = glm_fit(&x_mat, &ys, 15, 1, &cfg).expect("poisson sqrt fit");
        assert!(fit.converged);
        assert!(fit.fitted_values.iter().all(|&v| v >= 0.0));
    }

    // ── 7. Binomial with cloglog link ─────────────────────────────────────────
    #[test]
    fn glm_binomial_cloglog() {
        // cloglog(μ) = log(-log(1-μ)) = α + βx
        // Use soft probability responses to avoid perfect separation.
        let xs: Vec<f64> = (0..20).map(|i| (i as f64 - 10.0) * 0.25).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| {
                let eta = -0.5 + 0.3 * x;
                // Clamp to avoid 0/1
                (1.0 - (-eta.exp()).exp()).clamp(0.05, 0.95)
            })
            .collect();
        let x_mat: Vec<f64> = xs.clone();
        let cfg = GlmConfig {
            family: GlmFamily::Binomial,
            link: GlmLink::ClogLog,
            ..Default::default()
        };
        let fit = glm_fit(&x_mat, &ys, 20, 1, &cfg).expect("cloglog fit");
        assert!(fit.converged, "cloglog should converge");
    }

    // ── 8. fitted_values length == n_samples ─────────────────────────────────
    #[test]
    fn glm_fitted_values_shape() {
        let x_mat: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let ys: Vec<f64> = (0..8).map(|i| i as f64 * 2.0).collect();
        let cfg = GlmConfig::canonical(GlmFamily::Gaussian);
        let fit = glm_fit(&x_mat, &ys, 8, 1, &cfg).expect("fit ok");
        assert_eq!(fit.fitted_values.len(), 8);
        assert_eq!(fit.residuals.len(), 8);
        assert_eq!(fit.deviance_residuals.len(), 8);
    }

    // ── 9. residual_deviance ≥ 0 ─────────────────────────────────────────────
    #[test]
    fn glm_residual_deviance_nonneg() {
        for family in [
            GlmFamily::Gaussian,
            GlmFamily::Poisson,
            GlmFamily::Binomial,
            GlmFamily::Gamma,
        ] {
            let xs: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
            let ys: Vec<f64> = match family {
                GlmFamily::Poisson => xs.iter().map(|&x| (x + 1.0).round()).collect(),
                GlmFamily::Binomial => (0..10)
                    .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
                    .collect(),
                GlmFamily::Gamma => xs.iter().map(|&x| x + 1.0).collect(),
                _ => xs.iter().map(|&x| x * 2.0 + 1.0).collect(),
            };
            let cfg = GlmConfig::canonical(family);
            let fit = glm_fit(&xs, &ys, 10, 1, &cfg).expect("fit ok");
            assert!(
                fit.residual_deviance >= -1e-10,
                "residual_deviance negative for {family:?}"
            );
        }
    }

    // ── 10. pseudo_r2 ∈ [0, 1] ───────────────────────────────────────────────
    #[test]
    fn glm_pseudo_r2_range() {
        let xs: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 1.0 + 3.0 * x).collect();
        let cfg = GlmConfig::canonical(GlmFamily::Gaussian);
        let fit = glm_fit(&xs, &ys, 12, 1, &cfg).expect("fit ok");
        assert!(fit.pseudo_r2 >= 0.0 && fit.pseudo_r2 <= 1.0);
    }

    // ── 11. perfect fit → pseudo_r2 close to 1 ───────────────────────────────
    #[test]
    fn glm_perfect_fit_r2_1() {
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 + 5.0 * x).collect();
        let cfg = GlmConfig::canonical(GlmFamily::Gaussian);
        let fit = glm_fit(&xs, &ys, 20, 1, &cfg).expect("fit ok");
        assert!(
            fit.pseudo_r2 > 0.99,
            "expected pseudo_r2 ≈ 1, got {}",
            fit.pseudo_r2
        );
    }

    // ── 12. glm_predict returns n_new values ─────────────────────────────────
    #[test]
    fn glm_predict_shape() {
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 1.0 + 2.0 * x).collect();
        let cfg = GlmConfig::canonical(GlmFamily::Gaussian);
        let fit = glm_fit(&xs, &ys, 10, 1, &cfg).expect("fit ok");
        let x_new: Vec<f64> = vec![1.0, 2.0, 3.0];
        let pred = glm_predict(&fit, &x_new, 3, 1, &cfg, false).expect("predict ok");
        assert_eq!(pred.len(), 3);
    }

    // ── 13. link-scale vs response-scale predictions differ ───────────────────
    #[test]
    fn glm_predict_link_vs_response() {
        let xs: Vec<f64> = (0..15).map(|i| i as f64 * 0.3).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| (0.5 + 0.4 * x).exp()).collect();
        let cfg = GlmConfig::canonical(GlmFamily::Poisson);
        let fit = glm_fit(&xs, &ys, 15, 1, &cfg).expect("fit ok");
        let x_new = vec![2.0, 3.0];
        let eta_preds = glm_predict(&fit, &x_new, 2, 1, &cfg, true).expect("link scale");
        let mu_preds = glm_predict(&fit, &x_new, 2, 1, &cfg, false).expect("response scale");
        // For log link, μ = exp(η) ≠ η for typical η
        for (&eta, &mu) in eta_preds.iter().zip(mu_preds.iter()) {
            assert!((mu - eta.exp()).abs() < 1e-6, "exp(η)≠μ: eta={eta} mu={mu}");
        }
    }

    // ── 14. std_errors all positive ──────────────────────────────────────────
    #[test]
    fn glm_std_errors_positive() {
        let xs: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 1.0 + 2.5 * x).collect();
        let cfg = GlmConfig::canonical(GlmFamily::Gaussian);
        let fit = glm_fit(&xs, &ys, 12, 1, &cfg).expect("fit ok");
        for &se in &fit.std_errors {
            assert!(se > 0.0, "std_error should be positive, got {se}");
        }
    }

    // ── 15. z_stats has length p (n_features + intercept) ────────────────────
    #[test]
    fn glm_z_stats_shape() {
        let xs: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 1.0 + x).collect();
        // 1 feature + intercept → p = 2
        let cfg = GlmConfig {
            intercept: true,
            ..GlmConfig::canonical(GlmFamily::Gaussian)
        };
        let fit = glm_fit(&xs, &ys, 12, 1, &cfg).expect("fit ok");
        assert_eq!(
            fit.z_stats.len(),
            2,
            "expected 2 z_stats (intercept + slope)"
        );
        assert_eq!(fit.p_values.len(), 2);
    }

    // ── 16. LRT chi2 > 0 for fuller model ────────────────────────────────────
    #[test]
    fn glm_lrt_nested() {
        // Reduced model: intercept-only (pass a constant-1 column as the only feature,
        // intercept=false so it's literally β₀ * 1).
        // Full model: intercept + x (intercept=true, 1 feature column).
        let n = 20_usize;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 1.0 + 2.0 * x).collect();

        // Null/reduced: intercept-only → pass ones column, intercept=false
        let x_null: Vec<f64> = vec![1.0_f64; n];
        let cfg_null = GlmConfig {
            intercept: false,
            ..GlmConfig::canonical(GlmFamily::Gaussian)
        };
        // Full: intercept=true, one real feature
        let cfg_full = GlmConfig::canonical(GlmFamily::Gaussian);

        let fit_null = glm_fit(&x_null, &ys, n, 1, &cfg_null).expect("null fit");
        let fit_full = glm_fit(&xs, &ys, n, 1, &cfg_full).expect("full fit");
        let (chi2, pval) = glm_lrt(&fit_null, &fit_full, 1).expect("lrt ok");
        assert!(chi2 >= 0.0, "chi2 must be non-negative");
        // Full model has much lower deviance on this strong linear signal
        assert!(
            fit_full.residual_deviance < fit_null.residual_deviance,
            "full deviance {} should be < null deviance {}",
            fit_full.residual_deviance,
            fit_null.residual_deviance,
        );
        // p_value in [0,1]
        assert!((0.0..=1.0).contains(&pval));
    }

    // ── 17. converges in < max_iter for simple data ───────────────────────────
    #[test]
    fn glm_convergence_flag() {
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 1.0 + x).collect();
        let cfg = GlmConfig {
            max_iter: 100,
            ..GlmConfig::canonical(GlmFamily::Gaussian)
        };
        let fit = glm_fit(&xs, &ys, 10, 1, &cfg).expect("fit ok");
        assert!(fit.converged, "should converge");
        assert!(fit.n_iter < 100, "should be fast, got {}", fit.n_iter);
    }

    // ── 18. empty data returns Err ────────────────────────────────────────────
    #[test]
    fn glm_empty_data_error() {
        let cfg = GlmConfig::canonical(GlmFamily::Gaussian);
        let result = glm_fit(&[], &[], 0, 1, &cfg);
        assert!(result.is_err(), "empty data should return error");
    }

    // ── 19. Probit link (Binomial) ────────────────────────────────────────────
    #[test]
    fn glm_binomial_probit() {
        // Use soft probabilities to avoid quasi-separation.
        let xs: Vec<f64> = (0..20).map(|i| (i as f64 - 10.0) * 0.3).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| {
                // p = Φ(-0.3 + 0.4x), clamped
                let eta = -0.3 + 0.4 * x;
                let p = 0.5 * (1.0 + erf(eta / std::f64::consts::SQRT_2));
                p.clamp(0.05, 0.95)
            })
            .collect();
        let cfg = GlmConfig {
            family: GlmFamily::Binomial,
            link: GlmLink::Probit,
            ..Default::default()
        };
        let fit = glm_fit(&xs, &ys, 20, 1, &cfg).expect("probit fit");
        assert!(fit.converged, "probit should converge");
        assert!(fit.coefficients.len() == 2);
    }

    // ── 20. score test returns n_features statistics ──────────────────────────
    #[test]
    fn glm_score_test_shape() {
        let xs: Vec<f64> = (0..15).map(|i| i as f64 * 0.4).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 1.0 + x).collect();
        let cfg = GlmConfig::canonical(GlmFamily::Gaussian);
        let scores = glm_score_test(&xs, &ys, 15, 1, &cfg).expect("score test ok");
        assert_eq!(scores.len(), 1);
    }

    // ── 21. canonical() convenience constructor ───────────────────────────────
    #[test]
    fn glm_canonical_constructor() {
        let cfg = GlmConfig::canonical(GlmFamily::Poisson);
        assert_eq!(cfg.family, GlmFamily::Poisson);
        assert_eq!(cfg.link, GlmLink::Log);
        assert_eq!(cfg.max_iter, 100);
        assert!(cfg.intercept);
    }

    // ── 22. dispersion ≥ 0 ───────────────────────────────────────────────────
    #[test]
    fn glm_dispersion_nonneg() {
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| (0.3 * x).exp()).collect();
        let cfg = GlmConfig::canonical(GlmFamily::Poisson);
        let fit = glm_fit(&xs, &ys, 10, 1, &cfg).expect("fit ok");
        assert!(fit.dispersion >= 0.0);
    }
}
