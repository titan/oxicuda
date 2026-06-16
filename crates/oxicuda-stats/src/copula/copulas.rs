//! Copula models for bivariate dependence modelling.
//!
//! Implements four copula families, each fitted by method-of-moments via Kendall's τ:
//! - **Gaussian**: elliptical, no tail dependence
//! - **Clayton**: Archimedean, lower tail dependence
//! - **Frank**: Archimedean, no tail dependence, symmetric
//! - **Gumbel**: Archimedean, upper tail dependence
//!
//! # References
//! - Nelsen, R.B. (2006). *An Introduction to Copulas*, 2nd ed. Springer.
//! - Genest, C. & MacKay, J. (1986). *The joy of copulas*. Am. Statistician 40(4):280-283.
//! - Joe, H. (1997). *Multivariate Models and Dependence Concepts*. Chapman & Hall.

use crate::error::{StatsError, StatsResult};
use crate::handle::LcgRng;
use std::f64::consts::{PI, SQRT_2};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Supported bivariate copula families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopulaFamily {
    /// Gaussian (normal) copula — no tail dependence.
    Gaussian,
    /// Clayton copula — lower tail dependence, θ > 0.
    Clayton,
    /// Frank copula — no tail dependence, θ ≠ 0.
    Frank,
    /// Gumbel copula — upper tail dependence, θ ≥ 1.
    Gumbel,
}

/// Fitted bivariate copula model.
#[derive(Debug, Clone)]
pub struct CopulaFit {
    /// Copula family.
    pub family: CopulaFamily,
    /// Copula parameter: ρ for Gaussian, θ for Archimedean families.
    pub theta: f64,
    /// Kendall's τ estimated from the data.
    pub kendall_tau: f64,
    /// Number of observations used for fitting.
    pub n_samples: usize,
}

// ---------------------------------------------------------------------------
// Helper: standard normal CDF, PDF, and quantile
// ---------------------------------------------------------------------------

/// Standard normal CDF: Φ(x) = (1 + erf(x / √2)) / 2.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / SQRT_2))
}

/// Standard normal PDF: φ(x) = exp(-x²/2) / √(2π).
fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Standard normal quantile Φ⁻¹(p) via Newton iteration starting from 0.
///
/// Converges in ~50 iterations for all p ∈ (1e-15, 1−1e-15).
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    // Rational seed: Acklam's rational approximation for speed of convergence.
    let mut x = acklam_seed(p);
    // Newton refinement: x ← x - (Φ(x) - p) / φ(x)
    for _ in 0..50 {
        let fx = normal_cdf(x) - p;
        let fpx = normal_pdf(x);
        if fpx.abs() < 1e-300 {
            break;
        }
        let dx = fx / fpx;
        x -= dx;
        if dx.abs() < 1e-13 {
            break;
        }
    }
    x
}

/// Peter Acklam's rational approximation seed for Φ⁻¹(p).
///
/// Gives ~9 significant digits without further refinement; used as Newton seed.
fn acklam_seed(p: f64) -> f64 {
    // Coefficients for central region
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_374_269e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    // Coefficients for tail regions
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
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        // p > P_HIGH: use symmetry x(p) = -x(1-p)
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Abramowitz-Stegun 7.1.26 erf approximation (max error ~1.5e-7).
fn erf_approx(x: f64) -> f64 {
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;
    if x == 0.0 {
        return 0.0;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + P * ax);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-ax * ax).exp();
    sign * y
}

// ---------------------------------------------------------------------------
// Bivariate normal CDF via 200-point Gaussian quadrature (single substitution)
// ---------------------------------------------------------------------------

/// Bivariate normal CDF Φ₂(x, y; ρ) via numerical integration.
///
/// Uses the representation C(Φ(x), Φ(y); ρ) and reduces to a 1-D integral
/// using the formula from Drezner (1978) / Vasicek (1977):
/// Φ₂(x,y;ρ) = ∫_{-∞}^{x} φ(s) · Φ((y − ρs)/√(1-ρ²)) ds
fn bivar_normal_cdf(x: f64, y: f64, rho: f64) -> f64 {
    if x == f64::NEG_INFINITY || y == f64::NEG_INFINITY {
        return 0.0;
    }
    if x == f64::INFINITY {
        return normal_cdf(y);
    }
    if y == f64::INFINITY {
        return normal_cdf(x);
    }
    let rho2 = (1.0 - rho * rho).max(1e-15);
    let sqrt_rho2 = rho2.sqrt();
    // Use 200-point trapezoidal rule on the variable u = Φ(s) ∈ [0, Φ(x)]
    // i.e. integrate φ(s)·Φ((y-ρs)/√(1-ρ²)) ds from -∞ to x
    // by substituting s from -8 to x across N_QUAD points.
    const N: usize = 200;
    let lo = (-8.0_f64).max(x - 20.0).min(x - 1e-10);
    let hi = x;
    let h = (hi - lo) / N as f64;
    let mut integral = 0.0_f64;
    for k in 0..=N {
        let s = lo + k as f64 * h;
        let inner = normal_cdf((y - rho * s) / sqrt_rho2);
        let w = if k == 0 || k == N { 0.5 } else { 1.0 };
        integral += w * normal_pdf(s) * inner;
    }
    integral * h
}

// ---------------------------------------------------------------------------
// Debye function D₁(x) for Frank copula τ computation
// ---------------------------------------------------------------------------

/// Debye function D₁(x) = (1/x) ∫₀^x t/(e^t − 1) dt, evaluated via 200-pt trapezoidal rule.
///
/// For small x (|x| < 1e-9) use the series D₁(x) ≈ 1 − x/4 + x²/36 − ...
fn debye1(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        return 1.0 - x / 4.0 + x * x / 36.0;
    }
    const N: usize = 200;
    let h = x / N as f64;
    let mut integral = 0.0_f64;
    for k in 1..=N {
        // Skip t=0 (integrand → 1 by L'Hôpital); start at h.
        let t = k as f64 * h;
        let integrand = if t.abs() < 1e-10 {
            1.0
        } else {
            t / (t.exp() - 1.0)
        };
        let w = if k == N { 0.5 } else { 1.0 };
        integral += w * integrand;
    }
    // Add boundary t→0 term (value 1) with weight 0.5 for trapezoidal
    integral += 0.5 * 1.0;
    integral * h / x
}

// ---------------------------------------------------------------------------
// Frank copula: invert τ = 1 + 4(D₁(θ) − 1)/θ
// ---------------------------------------------------------------------------

/// Kendall's τ as a function of Frank parameter θ (Nelsen, *An Introduction to
/// Copulas*).
///
/// τ(θ) = 1 + 4·(D₁(θ) − 1) / θ , where D₁ is the order-1 Debye function.
/// (Equivalently 1 − (4/θ)(1 − D₁).)
fn frank_tau_from_theta(theta: f64) -> f64 {
    if theta.abs() < 1e-9 {
        return 0.0; // θ→0 limit: τ→0
    }
    1.0 + 4.0 * (debye1(theta) - 1.0) / theta
}

/// Solve Frank τ(θ) = target_tau via bisection on θ ∈ (0, 50) for τ > 0.
fn frank_theta_from_tau(tau: f64) -> StatsResult<f64> {
    if tau.abs() < 1e-10 {
        // θ → 0 corresponds to independence; return small value
        return Ok(1e-6);
    }
    // Sign of theta matches sign of tau
    let (lo, hi) = if tau > 0.0 {
        (1e-6_f64, 50.0_f64)
    } else {
        (-50.0_f64, -1e-6_f64)
    };
    let target = tau;
    let f_lo = frank_tau_from_theta(lo) - target;
    let f_hi = frank_tau_from_theta(hi) - target;
    if f_lo * f_hi > 0.0 {
        return Err(StatsError::NotConverged {
            iter: 0,
            residual: f_lo.abs().min(f_hi.abs()),
        });
    }
    let mut a = lo;
    let mut b = hi;
    for _ in 0..100 {
        let mid = 0.5 * (a + b);
        let f_mid = frank_tau_from_theta(mid) - target;
        if f_mid.abs() < 1e-12 || (b - a).abs() < 1e-12 {
            return Ok(mid);
        }
        if f_lo * f_mid <= 0.0 {
            b = mid;
        } else {
            a = mid;
        }
    }
    Ok(0.5 * (a + b))
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_copula_data(u: &[f64], v: &[f64], n: usize) -> StatsResult<()> {
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    if u.len() != n {
        return Err(StatsError::DimensionMismatch { a: u.len(), b: n });
    }
    if v.len() != n {
        return Err(StatsError::DimensionMismatch { a: v.len(), b: n });
    }
    for (i, (&ui, &vi)) in u.iter().zip(v.iter()).enumerate() {
        if !(ui > 0.0 && ui < 1.0) {
            return Err(StatsError::InvalidParameter {
                name: format!("u[{i}]"),
                reason: format!("must be in (0, 1), got {ui}"),
            });
        }
        if !(vi > 0.0 && vi < 1.0) {
            return Err(StatsError::InvalidParameter {
                name: format!("v[{i}]"),
                reason: format!("must be in (0, 1), got {vi}"),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Kendall's τ
// ---------------------------------------------------------------------------

/// Estimate Kendall's τ from paired (u, v) samples via O(n²) concordance counting.
///
/// # Errors
/// - [`StatsError::InsufficientSampleSize`] if `n < 2`.
/// - [`StatsError::DimensionMismatch`] if slice lengths ≠ n.
/// - [`StatsError::InvalidParameter`] if any `u[i]` or `v[i]` ∉ (0, 1).
pub fn kendall_tau_pairs(u: &[f64], v: &[f64], n: usize) -> StatsResult<f64> {
    validate_copula_data(u, v, n)?;
    let mut concordant: i64 = 0;
    let mut discordant: i64 = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let du = u[j] - u[i];
            let dv = v[j] - v[i];
            let prod = du * dv;
            if prod > 0.0 {
                concordant += 1;
            } else if prod < 0.0 {
                discordant += 1;
            }
            // ties contribute 0
        }
    }
    let total = (n * (n - 1) / 2) as f64;
    Ok((concordant - discordant) as f64 / total)
}

// ---------------------------------------------------------------------------
// Copula fitting by method-of-moments
// ---------------------------------------------------------------------------

/// Fit a copula to paired uniform pseudo-observations via Kendall's τ.
///
/// Converts τ to the copula parameter θ using the family-specific relationship:
/// - Gaussian: ρ = sin(π τ / 2)
/// - Clayton: θ = 2τ/(1−τ)
/// - Frank: solve τ(θ) = τ numerically
/// - Gumbel: θ = 1/(1−τ)
///
/// # Errors
/// - [`StatsError::InsufficientSampleSize`] if `n < 2`.
/// - [`StatsError::DimensionMismatch`] if slice lengths ≠ n.
/// - [`StatsError::InvalidParameter`] if any `u[i]` or `v[i]` ∉ (0, 1),
///   or if the estimated parameter is invalid for the chosen family.
pub fn copula_fit(u: &[f64], v: &[f64], n: usize, family: CopulaFamily) -> StatsResult<CopulaFit> {
    validate_copula_data(u, v, n)?;
    let tau = kendall_tau_pairs(u, v, n)?;

    let theta = match family {
        CopulaFamily::Gaussian => {
            let rho = (PI * tau / 2.0).sin();
            // Clamp to valid range (−1, 1) with small margin
            rho.clamp(-0.9999, 0.9999)
        }
        CopulaFamily::Clayton => {
            let theta = 2.0 * tau / (1.0 - tau);
            if theta <= 0.0 {
                return Err(StatsError::InvalidParameter {
                    name: "theta".to_owned(),
                    reason: format!(
                        "Clayton copula requires θ > 0 (got τ={tau:.4}, θ={theta:.4}); \
                         data must be positively dependent"
                    ),
                });
            }
            theta
        }
        CopulaFamily::Frank => frank_theta_from_tau(tau)?,
        CopulaFamily::Gumbel => {
            let theta = if (1.0 - tau).abs() < 1e-12 {
                1e6 // extremely strong upper tail dependence
            } else {
                1.0 / (1.0 - tau)
            };
            if theta < 1.0 {
                return Err(StatsError::InvalidParameter {
                    name: "theta".to_owned(),
                    reason: format!(
                        "Gumbel copula requires θ ≥ 1 (got τ={tau:.4}, θ={theta:.4}); \
                         data must have non-negative upper tail dependence"
                    ),
                });
            }
            theta
        }
    };

    Ok(CopulaFit {
        family,
        theta,
        kendall_tau: tau,
        n_samples: n,
    })
}

// ---------------------------------------------------------------------------
// Copula CDF
// ---------------------------------------------------------------------------

/// Evaluate the copula CDF C(u, v) at a single point.
///
/// # Errors
/// - [`StatsError::InvalidParameter`] if u or v ∉ (0, 1).
pub fn copula_cdf(u: f64, v: f64, fit: &CopulaFit) -> StatsResult<f64> {
    if !(0.0..=1.0).contains(&u) {
        return Err(StatsError::InvalidParameter {
            name: "u".to_owned(),
            reason: format!("must be in [0, 1], got {u}"),
        });
    }
    if !(0.0..=1.0).contains(&v) {
        return Err(StatsError::InvalidParameter {
            name: "v".to_owned(),
            reason: format!("must be in [0, 1], got {v}"),
        });
    }
    // Boundary conditions
    if u == 0.0 || v == 0.0 {
        return Ok(0.0);
    }
    if u == 1.0 {
        return Ok(v);
    }
    if v == 1.0 {
        return Ok(u);
    }
    let theta = fit.theta;
    let c = match fit.family {
        CopulaFamily::Gaussian => {
            let x = normal_quantile(u);
            let y = normal_quantile(v);
            bivar_normal_cdf(x, y, theta)
        }
        CopulaFamily::Clayton => {
            // C(u,v;θ) = (u^{-θ} + v^{-θ} − 1)^{-1/θ}
            let inner = u.powf(-theta) + v.powf(-theta) - 1.0;
            if inner <= 0.0 {
                0.0
            } else {
                inner.powf(-1.0 / theta)
            }
        }
        CopulaFamily::Frank => {
            // C(u,v;θ) = −(1/θ) ln(1 + (e^{-θu}−1)(e^{-θv}−1)/(e^{-θ}−1))
            let em1 = (-theta).exp() - 1.0;
            if em1.abs() < 1e-14 {
                // θ ≈ 0 ⇒ independence
                return Ok(u * v);
            }
            let num = ((-theta * u).exp() - 1.0) * ((-theta * v).exp() - 1.0);
            let inner = 1.0 + num / em1;
            if inner <= 0.0 {
                return Ok(0.0);
            }
            -(1.0 / theta) * inner.ln()
        }
        CopulaFamily::Gumbel => {
            // C(u,v;θ) = exp(−((−ln u)^θ + (−ln v)^θ)^{1/θ})
            let a = (-u.ln()).powf(theta) + (-v.ln()).powf(theta);
            (-a.powf(1.0 / theta)).exp()
        }
    };
    Ok(c.clamp(0.0, 1.0))
}

// ---------------------------------------------------------------------------
// Copula PDF (density)
// ---------------------------------------------------------------------------

/// Evaluate the copula density c(u, v) at a single interior point.
///
/// # Errors
/// - [`StatsError::InvalidParameter`] if u or v ∉ (0, 1) (strict interior required).
pub fn copula_pdf(u: f64, v: f64, fit: &CopulaFit) -> StatsResult<f64> {
    if !(u > 0.0 && u < 1.0) {
        return Err(StatsError::InvalidParameter {
            name: "u".to_owned(),
            reason: format!("must be in (0, 1) for PDF, got {u}"),
        });
    }
    if !(v > 0.0 && v < 1.0) {
        return Err(StatsError::InvalidParameter {
            name: "v".to_owned(),
            reason: format!("must be in (0, 1) for PDF, got {v}"),
        });
    }
    let theta = fit.theta;
    let density = match fit.family {
        CopulaFamily::Gaussian => {
            let x = normal_quantile(u);
            let y = normal_quantile(v);
            let rho2 = theta * theta;
            // c(u,v;ρ) = (1/√(1-ρ²)) · exp((2ρxy - ρ²(x²+y²)) / (2(1-ρ²)))
            let one_minus_rho2 = (1.0 - rho2).max(1e-15);
            let exponent = (2.0 * theta * x * y - rho2 * (x * x + y * y)) / (2.0 * one_minus_rho2);
            exponent.exp() / one_minus_rho2.sqrt()
        }
        CopulaFamily::Clayton => {
            // c(u,v;θ) = (θ+1)·(uv)^{-(θ+1)}·(u^{-θ}+v^{-θ}−1)^{-(2+1/θ)}
            let inner = u.powf(-theta) + v.powf(-theta) - 1.0;
            if inner <= 0.0 {
                return Ok(0.0);
            }
            (theta + 1.0) * (u * v).powf(-(theta + 1.0)) * inner.powf(-(2.0 + 1.0 / theta))
        }
        CopulaFamily::Frank => {
            // c(u,v;θ) = −θ·(e^{-θ}−1)·e^{-θ(u+v)} / (e^{-θ}−1 + (e^{-θu}−1)(e^{-θv}−1))²
            let em1 = (-theta).exp() - 1.0;
            if em1.abs() < 1e-14 {
                return Ok(1.0); // independence
            }
            let eu = (-theta * u).exp() - 1.0;
            let ev = (-theta * v).exp() - 1.0;
            let denom = em1 + eu * ev;
            if denom.abs() < 1e-300 {
                return Ok(0.0);
            }
            let num = -theta * em1 * ((-theta * (u + v)).exp());
            (num / (denom * denom)).abs()
        }
        CopulaFamily::Gumbel => {
            // Let A = (−ln u)^θ + (−ln v)^θ, s = A^{1/θ}
            // c(u,v;θ) = C(u,v;θ) / (u·v) · (A^{1/θ-2}) · ((ln u)(ln v))^{θ-1} · (s + θ−1)
            // This is the standard Gumbel copula density formula (= ∂²C/∂u∂v).
            let neg_ln_u = -u.ln();
            let neg_ln_v = -v.ln();
            let a_u = neg_ln_u.powf(theta);
            let a_v = neg_ln_v.powf(theta);
            let big_a = a_u + a_v;
            if big_a < 1e-300 {
                return Ok(0.0);
            }
            let s = big_a.powf(1.0 / theta);
            let c_val = (-s).exp();
            // Density: c · (1/(u*v)) · A^{1/θ - 2} · (neg_ln_u·neg_ln_v)^{θ-1} · (s + θ - 1)
            // NOTE: the trailing factor is (s+θ−1), NOT (s+θ−1)/s. The earlier `/s` was a
            // latent bug making the density integrate to ≈2.9; see gumbel_pdf_* regression tests.
            let factor1 = 1.0 / (u * v);
            let factor2 = big_a.powf(1.0 / theta - 2.0);
            let factor3 = (neg_ln_u * neg_ln_v).powf(theta - 1.0);
            let factor4 = s + theta - 1.0;
            c_val * factor1 * factor2 * factor3 * factor4
        }
    };
    Ok(density.max(0.0))
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

/// Generate `n` bivariate samples from the fitted copula.
///
/// Returns a flat vector of length 2n: [u₀, v₀, u₁, v₁, …].
///
/// # Errors
/// - [`StatsError::InsufficientSampleSize`] if `n == 0`.
pub fn copula_sample(fit: &CopulaFit, n: usize, rng: &mut LcgRng) -> StatsResult<Vec<f64>> {
    if n == 0 {
        return Err(StatsError::InsufficientSampleSize { got: 0, need: 1 });
    }
    let mut out = Vec::with_capacity(2 * n);
    let theta = fit.theta;
    match fit.family {
        CopulaFamily::Gaussian => {
            // Cholesky: [[1,0],[ρ,√(1-ρ²)]] applied to (Z₁,Z₂) ~ N(0,I)
            let rho = theta;
            let sqrt_one_minus_rho2 = (1.0 - rho * rho).max(0.0).sqrt();
            for _ in 0..n {
                let z1 = rng.next_normal();
                let z2 = rng.next_normal();
                let w1 = z1;
                let w2 = rho * z1 + sqrt_one_minus_rho2 * z2;
                out.push(normal_cdf(w1));
                out.push(normal_cdf(w2));
            }
        }
        CopulaFamily::Clayton => {
            // Conditional CDF inversion (correct formula):
            // ∂C/∂u = u^{-(θ+1)} · (u^{-θ} + v^{-θ} − 1)^{-(1+1/θ)} = t
            // Solving for v^{-θ}:
            //   (u^{-θ} + v^{-θ} − 1) = (u^{-(θ+1)} / t)^{θ/(θ+1)}
            //   v^{-θ} = (u^{-(θ+1)} / t)^{θ/(θ+1)} + 1 − u^{-θ}
            //   v = max(0, above)^{-1/θ}
            for _ in 0..n {
                let u = rng.next_f64().clamp(1e-15, 1.0 - 1e-15);
                let t = rng.next_f64().clamp(1e-15, 1.0 - 1e-15);
                let a = (u.powf(-(theta + 1.0)) / t).powf(theta / (theta + 1.0));
                let b = a + 1.0 - u.powf(-theta);
                let v = if b <= 0.0 {
                    1e-15_f64
                } else {
                    b.powf(-1.0 / theta).clamp(1e-15, 1.0 - 1e-15)
                };
                out.push(u);
                out.push(v);
            }
        }
        CopulaFamily::Frank => {
            // Conditional CDF inversion:
            // v = -(1/θ) ln(1 + t·(e^{-θ}−1) / (e^{-θu}·(1−t) + t))
            let em1 = (-theta).exp() - 1.0;
            for _ in 0..n {
                let u = rng.next_f64().clamp(1e-15, 1.0 - 1e-15);
                let t = rng.next_f64().clamp(1e-15, 1.0 - 1e-15);
                let denom = (-theta * u).exp() * (1.0 - t) + t;
                let v = if denom.abs() < 1e-300 {
                    0.5
                } else {
                    let inner = 1.0 + t * em1 / denom;
                    if inner <= 0.0 {
                        1e-15
                    } else {
                        (-(1.0 / theta) * inner.ln()).clamp(1e-15, 1.0 - 1e-15)
                    }
                };
                out.push(u);
                out.push(v);
            }
        }
        CopulaFamily::Gumbel => {
            // Numerical conditional CDF inversion via bisection:
            // ∂C/∂u = t  ⟹  solve for v given u and t ~ U(0,1)
            // ∂C/∂u = C(u,v;θ) · (−ln u)^{θ-1} · ((−ln u)^θ + (−ln v)^θ)^{1/θ-1} / u
            for _ in 0..n {
                let u = rng.next_f64().clamp(1e-15, 1.0 - 1e-15);
                let t = rng.next_f64().clamp(1e-15, 1.0 - 1e-15);
                // Bisect for v: F(v) = ∂C/∂u - t = 0
                let v = gumbel_conditional_sample(u, t, theta);
                out.push(u);
                out.push(v);
            }
        }
    }
    Ok(out)
}

/// Gumbel conditional CDF inversion: find v such that ∂C/∂u(u, v; θ) = t.
///
/// ∂C/∂u = C · (−ln u)^{θ−1} · (A)^{1/θ−1} / u,
/// where A = (−ln u)^θ + (−ln v)^θ.
fn gumbel_conditional_sample(u: f64, t: f64, theta: f64) -> f64 {
    let neg_ln_u = -u.ln();
    let a_u = neg_ln_u.powf(theta);

    let cdf_u_given_v = |v: f64| -> f64 {
        if v <= 0.0 || v >= 1.0 {
            return if v <= 0.0 { 0.0 } else { 1.0 };
        }
        let neg_ln_v = -v.ln();
        let a_v = neg_ln_v.powf(theta);
        let big_a = a_u + a_v;
        if big_a < 1e-300 {
            return 0.0;
        }
        let s = big_a.powf(1.0 / theta);
        let c_val = (-s).exp();
        // ∂C/∂u = C · (−ln u)^{θ-1} · A^{1/θ - 1} / u
        let deriv = c_val * neg_ln_u.powf(theta - 1.0) * big_a.powf(1.0 / theta - 1.0) / u;
        deriv.clamp(0.0, 1.0)
    };

    // Bisect on v ∈ (1e-15, 1-1e-15)
    let mut lo = 1e-15_f64;
    let mut hi = 1.0 - 1e-15_f64;
    let f_lo = cdf_u_given_v(lo) - t;
    let f_hi = cdf_u_given_v(hi) - t;
    if f_lo * f_hi > 0.0 {
        // Degenerate: return midpoint
        return 0.5;
    }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        let f_mid = cdf_u_given_v(mid) - t;
        if f_mid.abs() < 1e-12 || (hi - lo) < 1e-12 {
            return mid;
        }
        if f_lo * f_mid <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

// ---------------------------------------------------------------------------
// Tail dependence
// ---------------------------------------------------------------------------

/// Compute the bivariate tail dependence coefficients (λ_lower, λ_upper).
///
/// Returns `(lower_tail, upper_tail)` where 0 means no tail dependence.
#[must_use]
pub fn copula_tail_dependence(fit: &CopulaFit) -> (f64, f64) {
    let theta = fit.theta;
    match fit.family {
        CopulaFamily::Gaussian => (0.0, 0.0),
        CopulaFamily::Clayton => {
            // λ_lower = 2^{-1/θ}, λ_upper = 0
            let lower = (2.0_f64).powf(-1.0 / theta);
            (lower, 0.0)
        }
        CopulaFamily::Frank => (0.0, 0.0),
        CopulaFamily::Gumbel => {
            // λ_upper = 2 − 2^{1/θ}, λ_lower = 0
            let upper = 2.0 - (2.0_f64).powf(1.0 / theta);
            (0.0, upper)
        }
    }
}

// ---------------------------------------------------------------------------
// Log-likelihood
// ---------------------------------------------------------------------------

/// Compute the log-likelihood of paired (u, v) observations under the copula model.
///
/// ℓ = Σᵢ ln c(uᵢ, vᵢ)
///
/// # Errors
/// - [`StatsError::InsufficientSampleSize`] if `n < 2`.
/// - [`StatsError::DimensionMismatch`] if slice lengths ≠ n.
/// - [`StatsError::InvalidParameter`] if any `u[i]` or `v[i]` ∉ (0, 1).
pub fn copula_log_likelihood(u: &[f64], v: &[f64], n: usize, fit: &CopulaFit) -> StatsResult<f64> {
    validate_copula_data(u, v, n)?;
    let mut ll = 0.0_f64;
    for i in 0..n {
        let d = copula_pdf(u[i], v[i], fit)?;
        let log_d = if d <= 0.0 { -1e300 } else { d.ln() };
        ll += log_d;
    }
    Ok(ll)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    // Helper: generate n independent U(0,1) pairs
    fn uniform_pairs(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = LcgRng::new(seed);
        let u: Vec<f64> = (0..n)
            .map(|_| rng.next_f64().clamp(EPS, 1.0 - EPS))
            .collect();
        let v: Vec<f64> = (0..n)
            .map(|_| rng.next_f64().clamp(EPS, 1.0 - EPS))
            .collect();
        (u, v)
    }

    // Helper: generate perfectly concordant pairs (u=v)
    fn concordant_pairs(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = LcgRng::new(seed);
        let u: Vec<f64> = (0..n)
            .map(|_| rng.next_f64().clamp(EPS, 1.0 - EPS))
            .collect();
        let v = u.clone();
        (u, v)
    }

    // 1. Gaussian CDF boundary: C(0, v) = 0
    #[test]
    fn gaussian_cdf_boundary_u0() {
        let fit = CopulaFit {
            family: CopulaFamily::Gaussian,
            theta: 0.5,
            kendall_tau: 0.333,
            n_samples: 100,
        };
        let c = copula_cdf(0.0, 0.5, &fit).expect("copula_cdf should succeed");
        assert!((c - 0.0).abs() < 1e-10);
    }

    // 2. Gaussian CDF: C(1, v) = v
    #[test]
    fn gaussian_cdf_boundary_u1() {
        let fit = CopulaFit {
            family: CopulaFamily::Gaussian,
            theta: 0.5,
            kendall_tau: 0.333,
            n_samples: 100,
        };
        let v = 0.6;
        let c = copula_cdf(1.0, v, &fit).expect("copula_cdf should succeed");
        assert!((c - v).abs() < 1e-10);
    }

    // 3. Gaussian CDF monotone in u
    #[test]
    fn gaussian_cdf_monotone_u() {
        let fit = CopulaFit {
            family: CopulaFamily::Gaussian,
            theta: 0.5,
            kendall_tau: 0.333,
            n_samples: 100,
        };
        let v = 0.5;
        let c1 = copula_cdf(0.3, v, &fit).expect("copula_cdf should succeed");
        let c2 = copula_cdf(0.7, v, &fit).expect("copula_cdf should succeed");
        assert!(
            c2 > c1,
            "CDF should be monotone: c(0.3,v)={c1}, c(0.7,v)={c2}"
        );
    }

    // 4. Gaussian PDF >= 0
    #[test]
    fn gaussian_pdf_nonneg() {
        let fit = CopulaFit {
            family: CopulaFamily::Gaussian,
            theta: 0.5,
            kendall_tau: 0.333,
            n_samples: 100,
        };
        let d = copula_pdf(0.4, 0.6, &fit).expect("copula_pdf should succeed");
        assert!(d >= 0.0, "PDF must be non-negative, got {d}");
    }

    // 4b. Gumbel PDF == ∂²C/∂u∂v (regression guarding the former `(s+θ−1)/s` latent bug).
    #[test]
    fn gumbel_pdf_matches_cdf_second_derivative() {
        let fit = CopulaFit {
            family: CopulaFamily::Gumbel,
            theta: 2.0,
            kendall_tau: 0.5,
            n_samples: 100,
        };
        let h = 1e-4;
        for &(u, v) in &[(0.4, 0.6), (0.3, 0.3), (0.65, 0.45), (0.5, 0.8)] {
            // central finite difference of the CDF gives ∂²C/∂u∂v = the copula density
            let cpp = copula_cdf(u + h, v + h, &fit).expect("copula_cdf should succeed");
            let cpm = copula_cdf(u + h, v - h, &fit).expect("copula_cdf should succeed");
            let cmp = copula_cdf(u - h, v + h, &fit).expect("copula_cdf should succeed");
            let cmm = copula_cdf(u - h, v - h, &fit).expect("copula_cdf should succeed");
            let fd = (cpp - cpm - cmp + cmm) / (4.0 * h * h);
            let pdf = copula_pdf(u, v, &fit).expect("copula_pdf should succeed");
            assert!(
                (pdf - fd).abs() / fd.abs() < 1e-3,
                "Gumbel pdf {pdf} must equal ∂²C/∂u∂v {fd} at ({u},{v})"
            );
        }
    }

    // 4c. Gumbel density integrates to ≈1 (the buggy `/s` form integrated to ≈2.9).
    #[test]
    fn gumbel_pdf_integrates_to_one() {
        let fit = CopulaFit {
            family: CopulaFamily::Gumbel,
            theta: 1.5,
            kendall_tau: 1.0 / 3.0,
            n_samples: 100,
        };
        let n = 400usize;
        let step = 1.0 / n as f64;
        let mut integral = 0.0;
        for i in 0..n {
            let u = (i as f64 + 0.5) * step;
            for j in 0..n {
                let v = (j as f64 + 0.5) * step;
                integral +=
                    copula_pdf(u, v, &fit).expect("copula_pdf should succeed") * step * step;
            }
        }
        assert!(
            (integral - 1.0).abs() < 0.05,
            "Gumbel density should integrate to ≈1, got {integral}"
        );
    }

    // 5. Clayton CDF boundaries
    #[test]
    fn clayton_cdf_boundaries() {
        let fit = CopulaFit {
            family: CopulaFamily::Clayton,
            theta: 2.0,
            kendall_tau: 0.5,
            n_samples: 100,
        };
        assert!((copula_cdf(0.0, 0.5, &fit).expect("copula_cdf should succeed")).abs() < 1e-10);
        assert!((copula_cdf(0.5, 0.0, &fit).expect("copula_cdf should succeed")).abs() < 1e-10);
        let c = copula_cdf(1.0, 0.7, &fit).expect("copula_cdf should succeed");
        assert!((c - 0.7).abs() < 1e-10, "C(1,v)=v; got {c}");
    }

    // 6. Frank CDF boundaries
    #[test]
    fn frank_cdf_boundaries() {
        let fit = CopulaFit {
            family: CopulaFamily::Frank,
            theta: 3.0,
            kendall_tau: 0.4,
            n_samples: 100,
        };
        assert!((copula_cdf(0.0, 0.5, &fit).expect("copula_cdf should succeed")).abs() < 1e-10);
        let c = copula_cdf(1.0, 0.4, &fit).expect("copula_cdf should succeed");
        assert!((c - 0.4).abs() < 1e-10, "C(1,v)=v; got {c}");
    }

    // 6b. Frank Kendall's τ(θ): correct sign + invertibility.
    #[test]
    fn frank_tau_matches_known_values_and_inverts() {
        // τ(5) ≈ 0.4567 (Nelsen). The old sign-flipped formula gave ≈1.54 (> 1).
        let tau5 = frank_tau_from_theta(5.0);
        assert!(
            (tau5 - 0.4567).abs() < 5e-3,
            "τ(5) should be ≈0.4567, got {tau5}"
        );
        // τ is odd through the origin and strictly increasing in θ.
        assert_eq!(frank_tau_from_theta(1e-12), 0.0); // stable θ→0 early-return
        assert!(frank_tau_from_theta(0.01) > 0.0 && frank_tau_from_theta(0.01) < 0.01);
        assert!(frank_tau_from_theta(-5.0) < 0.0, "τ(−θ) = −τ(θ)");
        assert!(frank_tau_from_theta(2.0) < frank_tau_from_theta(8.0));
        // τ ∈ (0, 1) for θ > 0 (the buggy formula violated this).
        for &th in &[0.5_f64, 2.0, 5.0, 12.0, 30.0] {
            let t = frank_tau_from_theta(th);
            assert!(t > 0.0 && t < 1.0, "τ({th}) = {t} must lie in (0,1)");
        }
        // Round-trip θ → τ → θ via the bisection inverter.
        let theta =
            frank_theta_from_tau(frank_tau_from_theta(7.0)).expect("value should be present");
        assert!(
            (theta - 7.0).abs() < 1e-2,
            "round-trip θ=7 recovered {theta}"
        );
    }

    // 7. Gumbel CDF boundaries
    #[test]
    fn gumbel_cdf_boundaries() {
        let fit = CopulaFit {
            family: CopulaFamily::Gumbel,
            theta: 2.0,
            kendall_tau: 0.5,
            n_samples: 100,
        };
        assert!((copula_cdf(0.0, 0.5, &fit).expect("copula_cdf should succeed")).abs() < 1e-10);
        let c = copula_cdf(1.0, 0.5, &fit).expect("copula_cdf should succeed");
        assert!((c - 0.5).abs() < 1e-10, "C(1,v)=v; got {c}");
    }

    // 8. copula_sample: all u,v in (0,1)
    #[test]
    fn sample_in_unit_interval() {
        let mut rng = LcgRng::new(42);
        let fit = CopulaFit {
            family: CopulaFamily::Gaussian,
            theta: 0.5,
            kendall_tau: 0.333,
            n_samples: 100,
        };
        let s = copula_sample(&fit, 100, &mut rng).expect("copula_sample should succeed");
        for &x in &s {
            assert!(x > 0.0 && x < 1.0, "sample {x} not in (0,1)");
        }
    }

    // 9. copula_sample: correct length = 2n
    #[test]
    fn sample_correct_length() {
        let mut rng = LcgRng::new(7);
        let fit = CopulaFit {
            family: CopulaFamily::Clayton,
            theta: 2.0,
            kendall_tau: 0.5,
            n_samples: 50,
        };
        let s = copula_sample(&fit, 50, &mut rng).expect("copula_sample should succeed");
        assert_eq!(s.len(), 100);
    }

    // 10. kendall_tau_pairs: τ ≈ 0 for independent samples
    #[test]
    fn tau_near_zero_independent() {
        let (u, v) = uniform_pairs(500, 99);
        let tau = kendall_tau_pairs(&u, &v, 500).expect("kendall_tau_pairs should succeed");
        assert!(
            tau.abs() < 0.1,
            "τ={tau:.4} should be near 0 for independent pairs"
        );
    }

    // 11. kendall_tau_pairs: τ ≈ 1 for perfectly concordant pairs
    #[test]
    fn tau_near_one_concordant() {
        let (u, v) = concordant_pairs(100, 17);
        let tau = kendall_tau_pairs(&u, &v, 100).expect("kendall_tau_pairs should succeed");
        assert!((tau - 1.0).abs() < 1e-10, "τ={tau:.6} for u=v should be 1");
    }

    // 12. copula_fit Gaussian: |theta| ≤ 1
    #[test]
    fn gaussian_theta_le_1() {
        let (u, v) = uniform_pairs(200, 42);
        let fit =
            copula_fit(&u, &v, 200, CopulaFamily::Gaussian).expect("copula_fit should succeed");
        assert!(fit.theta.abs() <= 1.0, "Gaussian ρ must be in [-1,1]");
    }

    // 13. copula_fit Clayton: theta > 0 (with strongly positively dependent data)
    #[test]
    fn clayton_theta_positive() {
        // Generate positively dependent pairs via Clayton sampling (corrected formula)
        let fit_base = CopulaFit {
            family: CopulaFamily::Clayton,
            theta: 3.0, // strong positive dependence, τ ≈ 0.6
            kendall_tau: 0.6,
            n_samples: 300,
        };
        let mut rng = LcgRng::new(55);
        let s = copula_sample(&fit_base, 300, &mut rng).expect("copula_sample should succeed");
        let u: Vec<f64> = s.iter().step_by(2).copied().collect();
        let v: Vec<f64> = s.iter().skip(1).step_by(2).copied().collect();
        let fit =
            copula_fit(&u, &v, 300, CopulaFamily::Clayton).expect("copula_fit should succeed");
        assert!(fit.theta > 0.0, "Clayton θ must be > 0, got {}", fit.theta);
    }

    // 14. copula_fit Gumbel: theta >= 1
    #[test]
    fn gumbel_theta_ge_1() {
        let fit_base = CopulaFit {
            family: CopulaFamily::Gumbel,
            theta: 2.0,
            kendall_tau: 0.5,
            n_samples: 200,
        };
        let mut rng = LcgRng::new(66);
        let s = copula_sample(&fit_base, 200, &mut rng).expect("copula_sample should succeed");
        let u: Vec<f64> = s.iter().step_by(2).copied().collect();
        let v: Vec<f64> = s.iter().skip(1).step_by(2).copied().collect();
        let fit = copula_fit(&u, &v, 200, CopulaFamily::Gumbel).expect("copula_fit should succeed");
        assert!(fit.theta >= 1.0, "Gumbel θ must be >= 1");
    }

    // 15. tail_dependence Gaussian: both tails = 0
    #[test]
    fn gaussian_no_tail_dependence() {
        let fit = CopulaFit {
            family: CopulaFamily::Gaussian,
            theta: 0.8,
            kendall_tau: 0.55,
            n_samples: 100,
        };
        let (lo, hi) = copula_tail_dependence(&fit);
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 0.0);
    }

    // 16. tail_dependence Clayton: lower > 0, upper = 0
    #[test]
    fn clayton_lower_tail_dependence() {
        let fit = CopulaFit {
            family: CopulaFamily::Clayton,
            theta: 2.0,
            kendall_tau: 0.5,
            n_samples: 100,
        };
        let (lo, hi) = copula_tail_dependence(&fit);
        assert!(lo > 0.0, "Clayton should have lower tail dependence");
        assert_eq!(hi, 0.0);
    }

    // 17. tail_dependence Gumbel: upper > 0, lower = 0
    #[test]
    fn gumbel_upper_tail_dependence() {
        let fit = CopulaFit {
            family: CopulaFamily::Gumbel,
            theta: 2.0,
            kendall_tau: 0.5,
            n_samples: 100,
        };
        let (lo, hi) = copula_tail_dependence(&fit);
        assert_eq!(lo, 0.0);
        assert!(hi > 0.0, "Gumbel should have upper tail dependence");
    }

    // 18. copula_log_likelihood finite for Gaussian
    #[test]
    fn gaussian_log_likelihood_finite() {
        let fit_base = CopulaFit {
            family: CopulaFamily::Gaussian,
            theta: 0.5,
            kendall_tau: 0.333,
            n_samples: 100,
        };
        let mut rng = LcgRng::new(77);
        let s = copula_sample(&fit_base, 100, &mut rng).expect("copula_sample should succeed");
        let u: Vec<f64> = s.iter().step_by(2).copied().collect();
        let v: Vec<f64> = s.iter().skip(1).step_by(2).copied().collect();
        let ll = copula_log_likelihood(&u, &v, 100, &fit_base)
            .expect("copula_log_likelihood should succeed");
        assert!(
            ll.is_finite(),
            "Gaussian log-likelihood should be finite; got {ll}"
        );
    }

    // 19. Frank copula log-likelihood finite and non-trivial
    #[test]
    fn frank_log_likelihood_finite() {
        let fit_base = CopulaFit {
            family: CopulaFamily::Frank,
            theta: 3.0,
            kendall_tau: 0.4,
            n_samples: 100,
        };
        let mut rng = LcgRng::new(88);
        let s = copula_sample(&fit_base, 100, &mut rng).expect("copula_sample should succeed");
        let u: Vec<f64> = s.iter().step_by(2).copied().collect();
        let v: Vec<f64> = s.iter().skip(1).step_by(2).copied().collect();
        let ll = copula_log_likelihood(&u, &v, 100, &fit_base)
            .expect("copula_log_likelihood should succeed");
        assert!(ll.is_finite() && ll != 0.0, "Frank ll={ll}");
    }

    // 20. n < 2 → InsufficientSampleSize
    #[test]
    fn insufficient_data_error() {
        let u = vec![0.5];
        let v = vec![0.5];
        assert!(matches!(
            copula_fit(&u, &v, 1, CopulaFamily::Gaussian),
            Err(StatsError::InsufficientSampleSize { .. })
        ));
    }

    // 21. u[i] outside (0,1) → InvalidParameter
    #[test]
    fn invalid_u_value_error() {
        let u = vec![0.5, 1.5]; // 1.5 is out of (0,1)
        let v = vec![0.3, 0.4];
        assert!(matches!(
            copula_fit(&u, &v, 2, CopulaFamily::Gaussian),
            Err(StatsError::InvalidParameter { .. })
        ));
    }

    // 22. Different seeds → different copula_sample outputs
    #[test]
    fn different_seeds_different_samples() {
        let fit = CopulaFit {
            family: CopulaFamily::Gaussian,
            theta: 0.5,
            kendall_tau: 0.333,
            n_samples: 50,
        };
        let mut rng1 = LcgRng::new(1);
        let mut rng2 = LcgRng::new(2);
        let s1 = copula_sample(&fit, 50, &mut rng1).expect("copula_sample should succeed");
        let s2 = copula_sample(&fit, 50, &mut rng2).expect("copula_sample should succeed");
        assert_ne!(s1, s2, "different seeds should give different samples");
    }

    // (bonus, test 22 second half) Same seed → identical samples
    #[test]
    fn same_seed_identical_samples() {
        let fit = CopulaFit {
            family: CopulaFamily::Frank,
            theta: 3.0,
            kendall_tau: 0.4,
            n_samples: 50,
        };
        let mut rng_a = LcgRng::new(42);
        let mut rng_b = LcgRng::new(42);
        let sa = copula_sample(&fit, 50, &mut rng_a).expect("copula_sample should succeed");
        let sb = copula_sample(&fit, 50, &mut rng_b).expect("copula_sample should succeed");
        assert_eq!(sa, sb, "same seed should reproduce identical samples");
    }
}
