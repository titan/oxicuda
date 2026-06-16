//! Archimedean copulas (Frank, Clayton, Gumbel) via generator characterisation.
//!
//! A bivariate Archimedean copula is built from a *generator*
//! `φ : (0, 1] → [0, ∞)` (continuous, strictly decreasing, convex, `φ(1) = 0`):
//!
//! ```text
//! C(u, v) = φ⁻¹( φ(u) + φ(v) )
//! ```
//!
//! This module exposes the generator and its inverse explicitly, derives the
//! CDF from them, provides the bivariate density and the closed-form Kendall's
//! τ(θ) relations, and — unlike the method-of-moments fit in
//! [`crate::copula::copulas`] — estimates the single parameter θ by **maximum
//! likelihood** via a 1-D golden-section search on the profile log-likelihood.
//!
//! | family  | generator φ(t)                                   | τ(θ)                  | valid θ |
//! |---------|--------------------------------------------------|-----------------------|---------|
//! | Clayton | `(t^{-θ} − 1) / θ`                                | `θ / (θ + 2)`         | θ > 0   |
//! | Gumbel  | `(−ln t)^θ`                                      | `1 − 1/θ`             | θ ≥ 1   |
//! | Frank   | `−ln((e^{−θt} − 1) / (e^{−θ} − 1))`              | `1 − 4(1 − D₁(θ))/θ`  | θ ≠ 0   |
//!
//! where `D₁` is the first Debye function.
//!
//! # References
//! - Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed. Springer.
//! - Genest, C. (1987). *Frank's family of bivariate distributions*.
//!   Biometrika 74(3): 549-555.
//! - Joe, H. (1997). *Multivariate Models and Dependence Concepts*. Chapman & Hall.

use crate::error::{StatsError, StatsResult};
use crate::handle::LcgRng;

/// Archimedean copula family with a single dependence parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchimedeanFamily {
    /// Clayton — lower tail dependence, θ > 0.
    Clayton,
    /// Frank — no tail dependence, symmetric, θ ≠ 0.
    Frank,
    /// Gumbel — upper tail dependence, θ ≥ 1.
    Gumbel,
}

/// A bivariate Archimedean copula with a fitted parameter θ.
#[derive(Debug, Clone, Copy)]
pub struct ArchimedeanCopula {
    /// Copula family.
    pub family: ArchimedeanFamily,
    /// Dependence parameter θ.
    pub theta: f64,
}

// ---------------------------------------------------------------------------
// First Debye function D₁(x) = (1/x) ∫₀ˣ t/(eᵗ − 1) dt
// ---------------------------------------------------------------------------

fn debye1(x: f64) -> f64 {
    if x.abs() < 1.0e-4 {
        // Maclaurin series D₁(x) = 1 − x/4 + x²/36 − x⁴/3600 + …
        return 1.0 - x / 4.0 + x * x / 36.0;
    }
    const N: usize = 400;
    let h = x / N as f64;
    let integrand = |t: f64| -> f64 {
        if t.abs() < 1.0e-12 {
            1.0
        } else {
            t / (t.exp() - 1.0)
        }
    };
    // Trapezoidal rule over [0, x] (handles x < 0 via signed h).
    let mut sum = 0.5 * (integrand(0.0) + integrand(x));
    for k in 1..N {
        sum += integrand(k as f64 * h);
    }
    sum * h / x
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_theta(family: ArchimedeanFamily, theta: f64) -> StatsResult<()> {
    if !theta.is_finite() {
        return Err(StatsError::InvalidParameter {
            name: "theta".to_owned(),
            reason: "θ must be finite".to_owned(),
        });
    }
    match family {
        ArchimedeanFamily::Clayton => {
            if theta <= 0.0 {
                return Err(StatsError::InvalidParameter {
                    name: "theta".to_owned(),
                    reason: format!("Clayton requires θ > 0, got {theta}"),
                });
            }
        }
        ArchimedeanFamily::Gumbel => {
            if theta < 1.0 {
                return Err(StatsError::InvalidParameter {
                    name: "theta".to_owned(),
                    reason: format!("Gumbel requires θ ≥ 1, got {theta}"),
                });
            }
        }
        ArchimedeanFamily::Frank => {
            if theta == 0.0 {
                return Err(StatsError::InvalidParameter {
                    name: "theta".to_owned(),
                    reason: "Frank requires θ ≠ 0".to_owned(),
                });
            }
        }
    }
    Ok(())
}

fn validate_pairs(u: &[f64], v: &[f64], n: usize) -> StatsResult<()> {
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    if u.len() != n {
        return Err(StatsError::DimensionMismatch { a: u.len(), b: n });
    }
    if v.len() != n {
        return Err(StatsError::DimensionMismatch { a: v.len(), b: n });
    }
    for (&ui, &vi) in u.iter().zip(v.iter()) {
        if !(ui > 0.0 && ui < 1.0) {
            return Err(StatsError::InvalidParameter {
                name: "u".to_owned(),
                reason: format!("must be in (0, 1), got {ui}"),
            });
        }
        if !(vi > 0.0 && vi < 1.0) {
            return Err(StatsError::InvalidParameter {
                name: "v".to_owned(),
                reason: format!("must be in (0, 1), got {vi}"),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Golden-section maximiser
// ---------------------------------------------------------------------------

fn golden_section_max<F: Fn(f64) -> f64>(f: F, mut a: f64, mut b: f64, max_iter: usize) -> f64 {
    let inv_phi = (5.0_f64.sqrt() - 1.0) / 2.0; // ≈ 0.618
    let mut c = b - inv_phi * (b - a);
    let mut d = a + inv_phi * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);
    for _ in 0..max_iter {
        if (b - a).abs() < 1.0e-7 {
            break;
        }
        if fc > fd {
            b = d;
            d = c;
            fd = fc;
            c = b - inv_phi * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + inv_phi * (b - a);
            fd = f(d);
        }
    }
    0.5 * (a + b)
}

// ---------------------------------------------------------------------------
// Kendall's τ from raw paired data (concordance counting)
// ---------------------------------------------------------------------------

fn kendall_tau_data(u: &[f64], v: &[f64], n: usize) -> f64 {
    let mut concordant: i64 = 0;
    let mut discordant: i64 = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let prod = (u[j] - u[i]) * (v[j] - v[i]);
            if prod > 0.0 {
                concordant += 1;
            } else if prod < 0.0 {
                discordant += 1;
            }
        }
    }
    let total = (n * (n - 1) / 2) as f64;
    (concordant - discordant) as f64 / total
}

/// Profile log-likelihood used by the optimiser (penalised, never NaN).
fn profile_ll(family: ArchimedeanFamily, theta: f64, u: &[f64], v: &[f64], n: usize) -> f64 {
    let cop = ArchimedeanCopula { family, theta };
    let mut acc = 0.0;
    for i in 0..n {
        let d = cop.density_raw(u[i], v[i]);
        acc += if d > 0.0 && d.is_finite() {
            d.ln()
        } else {
            -50.0
        };
    }
    acc
}

impl ArchimedeanCopula {
    /// Construct a copula, validating θ against the family's admissible range.
    ///
    /// # Errors
    /// [`StatsError::InvalidParameter`] if θ is non-finite or outside the
    /// family's valid range (Clayton θ > 0, Gumbel θ ≥ 1, Frank θ ≠ 0).
    pub fn new(family: ArchimedeanFamily, theta: f64) -> StatsResult<Self> {
        validate_theta(family, theta)?;
        Ok(Self { family, theta })
    }

    /// Generator `φ(t)` for `t ∈ (0, 1]`.
    #[must_use]
    pub fn generator(&self, t: f64) -> f64 {
        match self.family {
            ArchimedeanFamily::Clayton => (t.powf(-self.theta) - 1.0) / self.theta,
            ArchimedeanFamily::Gumbel => (-t.ln()).powf(self.theta),
            ArchimedeanFamily::Frank => {
                let num = (-self.theta * t).exp() - 1.0;
                let den = (-self.theta).exp() - 1.0;
                -(num / den).ln()
            }
        }
    }

    /// Inverse generator `φ⁻¹(s)` for `s ∈ [0, ∞)`.
    #[must_use]
    pub fn generator_inverse(&self, s: f64) -> f64 {
        match self.family {
            ArchimedeanFamily::Clayton => (1.0 + self.theta * s).powf(-1.0 / self.theta),
            ArchimedeanFamily::Gumbel => (-(s.powf(1.0 / self.theta))).exp(),
            ArchimedeanFamily::Frank => {
                let den = (-self.theta).exp() - 1.0;
                -(1.0 / self.theta) * (1.0 + (-s).exp() * den).ln()
            }
        }
    }

    /// Copula CDF `C(u, v) = φ⁻¹(φ(u) + φ(v))`.
    ///
    /// # Errors
    /// [`StatsError::InvalidParameter`] if `u` or `v` lies outside `[0, 1]`.
    pub fn cdf(&self, u: f64, v: f64) -> StatsResult<f64> {
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
        if u <= 0.0 || v <= 0.0 {
            return Ok(0.0);
        }
        if u >= 1.0 {
            return Ok(v);
        }
        if v >= 1.0 {
            return Ok(u);
        }
        if self.family == ArchimedeanFamily::Frank && self.theta.abs() < 1.0e-8 {
            return Ok(u * v); // independence limit
        }
        let c = self.generator_inverse(self.generator(u) + self.generator(v));
        Ok(c.clamp(0.0, 1.0))
    }

    /// Bivariate copula density (no validation; assumes `u, v ∈ (0, 1)`).
    fn density_raw(&self, u: f64, v: f64) -> f64 {
        let theta = self.theta;
        match self.family {
            ArchimedeanFamily::Clayton => {
                let inner = u.powf(-theta) + v.powf(-theta) - 1.0;
                if inner <= 0.0 {
                    return 0.0;
                }
                (theta + 1.0) * (u * v).powf(-(theta + 1.0)) * inner.powf(-(2.0 + 1.0 / theta))
            }
            ArchimedeanFamily::Gumbel => {
                let neg_ln_u = -u.ln();
                let neg_ln_v = -v.ln();
                let a_u = neg_ln_u.powf(theta);
                let a_v = neg_ln_v.powf(theta);
                let big_a = a_u + a_v;
                if big_a <= 0.0 {
                    return 0.0;
                }
                let s = big_a.powf(1.0 / theta);
                let c_val = (-s).exp();
                c_val / (u * v)
                    * big_a.powf(1.0 / theta - 2.0)
                    * (neg_ln_u * neg_ln_v).powf(theta - 1.0)
                    * (s + theta - 1.0)
                    / s
            }
            ArchimedeanFamily::Frank => {
                let em1 = (-theta).exp() - 1.0;
                if em1.abs() < 1.0e-14 {
                    return 1.0; // independence
                }
                let eu = (-theta * u).exp() - 1.0;
                let ev = (-theta * v).exp() - 1.0;
                let denom = em1 + eu * ev;
                if denom.abs() < 1.0e-300 {
                    return 0.0;
                }
                let num = -theta * em1 * (-theta * (u + v)).exp();
                (num / (denom * denom)).abs()
            }
        }
    }

    /// Bivariate copula density `c(u, v)` at an interior point.
    ///
    /// # Errors
    /// [`StatsError::InvalidParameter`] if `u` or `v` lies outside `(0, 1)`.
    pub fn pdf(&self, u: f64, v: f64) -> StatsResult<f64> {
        if !(u > 0.0 && u < 1.0) {
            return Err(StatsError::InvalidParameter {
                name: "u".to_owned(),
                reason: format!("must be in (0, 1), got {u}"),
            });
        }
        if !(v > 0.0 && v < 1.0) {
            return Err(StatsError::InvalidParameter {
                name: "v".to_owned(),
                reason: format!("must be in (0, 1), got {v}"),
            });
        }
        Ok(self.density_raw(u, v).max(0.0))
    }

    /// Closed-form Kendall's τ implied by the current θ.
    #[must_use]
    pub fn kendall_tau(&self) -> f64 {
        match self.family {
            ArchimedeanFamily::Clayton => self.theta / (self.theta + 2.0),
            ArchimedeanFamily::Gumbel => 1.0 - 1.0 / self.theta,
            ArchimedeanFamily::Frank => 1.0 - (4.0 / self.theta) * (1.0 - debye1(self.theta)),
        }
    }

    /// Convert a Kendall's τ into the family parameter θ.
    ///
    /// # Errors
    /// [`StatsError::InvalidParameter`] if τ is outside the family's reachable
    /// range, or [`StatsError::NotConverged`] if the Frank inversion fails.
    pub fn theta_from_tau(family: ArchimedeanFamily, tau: f64) -> StatsResult<f64> {
        match family {
            ArchimedeanFamily::Clayton => {
                if !(0.0..1.0).contains(&tau) {
                    return Err(StatsError::InvalidParameter {
                        name: "tau".to_owned(),
                        reason: format!("Clayton requires τ ∈ [0, 1), got {tau}"),
                    });
                }
                Ok((2.0 * tau / (1.0 - tau)).max(1.0e-8))
            }
            ArchimedeanFamily::Gumbel => {
                if !(0.0..1.0).contains(&tau) {
                    return Err(StatsError::InvalidParameter {
                        name: "tau".to_owned(),
                        reason: format!("Gumbel requires τ ∈ [0, 1), got {tau}"),
                    });
                }
                Ok((1.0 / (1.0 - tau)).max(1.0))
            }
            ArchimedeanFamily::Frank => {
                if !(-1.0 < tau && tau < 1.0) {
                    return Err(StatsError::InvalidParameter {
                        name: "tau".to_owned(),
                        reason: format!("Frank requires τ ∈ (-1, 1), got {tau}"),
                    });
                }
                if tau.abs() < 1.0e-10 {
                    return Ok(1.0e-6);
                }
                Self::frank_theta_bisect(tau)
            }
        }
    }

    fn frank_theta_bisect(target: f64) -> StatsResult<f64> {
        let frank_tau = |theta: f64| 1.0 - (4.0 / theta) * (1.0 - debye1(theta));
        let (mut a, mut b) = if target > 0.0 {
            (1.0e-6_f64, 60.0_f64)
        } else {
            (-60.0_f64, -1.0e-6_f64)
        };
        let mut fa = frank_tau(a) - target;
        if fa * (frank_tau(b) - target) > 0.0 {
            return Err(StatsError::NotConverged {
                iter: 0,
                residual: fa.abs(),
            });
        }
        for _ in 0..100 {
            let mid = 0.5 * (a + b);
            let fmid = frank_tau(mid) - target;
            if fmid.abs() < 1.0e-12 || (b - a).abs() < 1.0e-12 {
                return Ok(mid);
            }
            if fa * fmid <= 0.0 {
                b = mid;
            } else {
                a = mid;
                fa = fmid;
            }
        }
        Ok(0.5 * (a + b))
    }

    /// Total log-likelihood `Σ ln c(uᵢ, vᵢ)` for paired pseudo-observations.
    ///
    /// # Errors
    /// Validation errors from `validate_pairs`.
    pub fn log_likelihood(&self, u: &[f64], v: &[f64], n: usize) -> StatsResult<f64> {
        validate_pairs(u, v, n)?;
        let mut ll = 0.0;
        for i in 0..n {
            let d = self.density_raw(u[i], v[i]);
            ll += if d > 0.0 && d.is_finite() {
                d.ln()
            } else {
                -1.0e300
            };
        }
        Ok(ll)
    }

    /// Fit θ by **maximum likelihood** via golden-section search.
    ///
    /// The admissible θ range for the family is searched directly; the profile
    /// log-likelihood of these one-parameter families is unimodal in θ.
    ///
    /// # Errors
    /// Validation errors from `validate_pairs`, or [`StatsError::InvalidParameter`]
    /// if the optimum lands on an inadmissible θ.
    pub fn fit_mle(u: &[f64], v: &[f64], n: usize, family: ArchimedeanFamily) -> StatsResult<Self> {
        validate_pairs(u, v, n)?;
        let tau = kendall_tau_data(u, v, n);
        let (lo, hi) = match family {
            ArchimedeanFamily::Clayton => (1.0e-3, 40.0),
            ArchimedeanFamily::Gumbel => (1.0, 40.0),
            ArchimedeanFamily::Frank => {
                if tau >= 0.0 {
                    (1.0e-3, 40.0)
                } else {
                    (-40.0, -1.0e-3)
                }
            }
        };
        let theta_hat = golden_section_max(|t| profile_ll(family, t, u, v, n), lo, hi, 200);
        Self::new(family, theta_hat)
    }

    /// Theoretical tail-dependence coefficients `(λ_lower, λ_upper)`.
    #[must_use]
    pub fn tail_dependence(&self) -> (f64, f64) {
        match self.family {
            ArchimedeanFamily::Clayton => (2.0_f64.powf(-1.0 / self.theta), 0.0),
            ArchimedeanFamily::Frank => (0.0, 0.0),
            ArchimedeanFamily::Gumbel => (0.0, 2.0 - 2.0_f64.powf(1.0 / self.theta)),
        }
    }

    /// Generate `n` samples by conditional inversion.
    ///
    /// Returns a flat vector of length `2n`: `[u₀, v₀, u₁, v₁, …]`.
    ///
    /// # Errors
    /// [`StatsError::InsufficientSampleSize`] if `n == 0`.
    pub fn sample(&self, n: usize, rng: &mut LcgRng) -> StatsResult<Vec<f64>> {
        if n == 0 {
            return Err(StatsError::InsufficientSampleSize { got: 0, need: 1 });
        }
        let theta = self.theta;
        let mut out = Vec::with_capacity(2 * n);
        for _ in 0..n {
            let u = rng.next_f64().clamp(1.0e-12, 1.0 - 1.0e-12);
            let t = rng.next_f64().clamp(1.0e-12, 1.0 - 1.0e-12);
            let v = match self.family {
                ArchimedeanFamily::Clayton => {
                    let a = (u.powf(-(theta + 1.0)) / t).powf(theta / (theta + 1.0));
                    let b = a + 1.0 - u.powf(-theta);
                    if b <= 0.0 {
                        1.0e-12
                    } else {
                        b.powf(-1.0 / theta).clamp(1.0e-12, 1.0 - 1.0e-12)
                    }
                }
                ArchimedeanFamily::Frank => {
                    let em1 = (-theta).exp() - 1.0;
                    let denom = (-theta * u).exp() * (1.0 - t) + t;
                    if denom.abs() < 1.0e-300 {
                        0.5
                    } else {
                        let inner = 1.0 + t * em1 / denom;
                        if inner <= 0.0 {
                            1.0e-12
                        } else {
                            (-(1.0 / theta) * inner.ln()).clamp(1.0e-12, 1.0 - 1.0e-12)
                        }
                    }
                }
                ArchimedeanFamily::Gumbel => self.gumbel_conditional(u, t),
            };
            out.push(u);
            out.push(v);
        }
        Ok(out)
    }

    /// Solve `∂C/∂u(u, v) = t` for `v` by bisection (Gumbel sampling).
    fn gumbel_conditional(&self, u: f64, t: f64) -> f64 {
        let theta = self.theta;
        let neg_ln_u = -u.ln();
        let a_u = neg_ln_u.powf(theta);
        let cond = |v: f64| -> f64 {
            if v <= 0.0 {
                return 0.0;
            }
            if v >= 1.0 {
                return 1.0;
            }
            let a_v = (-v.ln()).powf(theta);
            let big_a = a_u + a_v;
            if big_a <= 0.0 {
                return 0.0;
            }
            let s = big_a.powf(1.0 / theta);
            let c_val = (-s).exp();
            (c_val * neg_ln_u.powf(theta - 1.0) * big_a.powf(1.0 / theta - 1.0) / u).clamp(0.0, 1.0)
        };
        let mut lo = 1.0e-12_f64;
        let mut hi = 1.0 - 1.0e-12_f64;
        if (cond(lo) - t) * (cond(hi) - t) > 0.0 {
            return 0.5;
        }
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            let fmid = cond(mid) - t;
            if fmid.abs() < 1.0e-12 || (hi - lo) < 1.0e-12 {
                return mid;
            }
            if (cond(lo) - t) * fmid <= 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        0.5 * (lo + hi)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const FAMILIES: [ArchimedeanFamily; 3] = [
        ArchimedeanFamily::Clayton,
        ArchimedeanFamily::Frank,
        ArchimedeanFamily::Gumbel,
    ];

    fn theta_for(family: ArchimedeanFamily) -> f64 {
        match family {
            ArchimedeanFamily::Clayton => 2.0,
            ArchimedeanFamily::Frank => 4.0,
            ArchimedeanFamily::Gumbel => 2.0,
        }
    }

    #[test]
    fn cdf_uniform_margins_and_zero() {
        for &fam in &FAMILIES {
            let cop = ArchimedeanCopula::new(fam, theta_for(fam)).expect("ok");
            // C(u, 1) = u and C(1, v) = v.
            for &x in &[0.2, 0.5, 0.85] {
                assert!((cop.cdf(x, 1.0).expect("ok") - x).abs() < 1e-12, "C(u,1)=u");
                assert!((cop.cdf(1.0, x).expect("ok") - x).abs() < 1e-12, "C(1,v)=v");
                assert!(cop.cdf(x, 0.0).expect("ok").abs() < 1e-12, "C(u,0)=0");
            }
        }
    }

    #[test]
    fn generator_roundtrip() {
        for &fam in &FAMILIES {
            let cop = ArchimedeanCopula::new(fam, theta_for(fam)).expect("ok");
            for &t in &[0.1, 0.4, 0.7, 0.95] {
                let back = cop.generator_inverse(cop.generator(t));
                assert!(
                    (back - t).abs() < 1e-8,
                    "{fam:?}: φ⁻¹(φ(t)) ≠ t ({back} vs {t})"
                );
            }
        }
    }

    #[test]
    fn independence_limits() {
        // Gumbel θ = 1 ⇒ C(u, v) = u·v exactly.
        let g = ArchimedeanCopula::new(ArchimedeanFamily::Gumbel, 1.0).expect("ok");
        assert!((g.cdf(0.3, 0.7).expect("ok") - 0.21).abs() < 1e-12);
        // Clayton and Frank with θ → 0 ⇒ C ≈ u·v.
        let c = ArchimedeanCopula::new(ArchimedeanFamily::Clayton, 1.0e-3).expect("ok");
        assert!((c.cdf(0.3, 0.7).expect("ok") - 0.21).abs() < 1e-2);
        let f = ArchimedeanCopula::new(ArchimedeanFamily::Frank, 1.0e-3).expect("ok");
        assert!((f.cdf(0.3, 0.7).expect("ok") - 0.21).abs() < 1e-2);
    }

    #[test]
    fn frechet_bounds_hold() {
        for &fam in &FAMILIES {
            let cop = ArchimedeanCopula::new(fam, theta_for(fam)).expect("ok");
            for &(u, v) in &[(0.3, 0.6), (0.5, 0.5), (0.8, 0.2)] {
                let c = cop.cdf(u, v).expect("ok");
                let lower = (u + v - 1.0).max(0.0);
                let upper = u.min(v);
                assert!(
                    c >= lower - 1e-9 && c <= upper + 1e-9,
                    "{fam:?}: {c} not in [{lower},{upper}]"
                );
            }
        }
    }

    #[test]
    fn kendall_tau_closed_form() {
        // Clayton θ=2 ⇒ τ = 2/4 = 0.5.
        let c = ArchimedeanCopula::new(ArchimedeanFamily::Clayton, 2.0).expect("ok");
        assert!((c.kendall_tau() - 0.5).abs() < 1e-12);
        // Gumbel θ=2 ⇒ τ = 1 - 1/2 = 0.5.
        let g = ArchimedeanCopula::new(ArchimedeanFamily::Gumbel, 2.0).expect("ok");
        assert!((g.kendall_tau() - 0.5).abs() < 1e-12);
        // Frank θ=5 ⇒ τ ≈ 0.4567 (known reference value).
        let f = ArchimedeanCopula::new(ArchimedeanFamily::Frank, 5.0).expect("ok");
        assert!(
            (f.kendall_tau() - 0.4567).abs() < 1e-3,
            "Frank τ(5) = {}",
            f.kendall_tau()
        );
    }

    #[test]
    fn tau_theta_roundtrip() {
        for &fam in &FAMILIES {
            let theta = theta_for(fam);
            let cop = ArchimedeanCopula::new(fam, theta).expect("ok");
            let tau = cop.kendall_tau();
            let recovered = ArchimedeanCopula::theta_from_tau(fam, tau).expect("ok");
            assert!(
                (recovered - theta).abs() < 1e-2 * theta.max(1.0),
                "{fam:?}: θ={theta} τ={tau} recovered={recovered}"
            );
        }
    }

    #[test]
    fn mle_recovers_clayton_theta() {
        let theta_true = 2.0;
        let cop = ArchimedeanCopula::new(ArchimedeanFamily::Clayton, theta_true).expect("ok");
        let mut rng = LcgRng::new(12345);
        let s = cop.sample(400, &mut rng).expect("ok");
        let u: Vec<f64> = s.iter().step_by(2).copied().collect();
        let v: Vec<f64> = s.iter().skip(1).step_by(2).copied().collect();
        let fit = ArchimedeanCopula::fit_mle(&u, &v, 400, ArchimedeanFamily::Clayton).expect("ok");
        assert!(fit.theta > 0.0, "θ̂ must be admissible");
        assert!(
            (fit.theta - theta_true).abs() < 1.5,
            "ML should recover θ≈2, got {}",
            fit.theta
        );
        // ML must dominate the method-of-moments seed.
        let tau = kendall_tau_data(&u, &v, 400);
        let theta_mom =
            ArchimedeanCopula::theta_from_tau(ArchimedeanFamily::Clayton, tau).expect("ok");
        let mom = ArchimedeanCopula::new(ArchimedeanFamily::Clayton, theta_mom).expect("ok");
        let ll_mle = fit.log_likelihood(&u, &v, 400).expect("ok");
        let ll_mom = mom.log_likelihood(&u, &v, 400).expect("ok");
        assert!(
            ll_mle >= ll_mom - 1e-6,
            "ML ll {ll_mle} should ≥ MoM ll {ll_mom}"
        );
    }

    #[test]
    fn mle_recovers_gumbel_theta() {
        let theta_true = 2.0;
        let cop = ArchimedeanCopula::new(ArchimedeanFamily::Gumbel, theta_true).expect("ok");
        let mut rng = LcgRng::new(999);
        let s = cop.sample(400, &mut rng).expect("ok");
        let u: Vec<f64> = s.iter().step_by(2).copied().collect();
        let v: Vec<f64> = s.iter().skip(1).step_by(2).copied().collect();
        let fit = ArchimedeanCopula::fit_mle(&u, &v, 400, ArchimedeanFamily::Gumbel).expect("ok");
        assert!(fit.theta >= 1.0, "Gumbel θ̂ ≥ 1");
        assert!(
            (fit.theta - theta_true).abs() < 1.5,
            "ML should recover θ≈2, got {}",
            fit.theta
        );
    }

    #[test]
    fn density_non_negative_and_samples_in_unit_square() {
        for &fam in &FAMILIES {
            let cop = ArchimedeanCopula::new(fam, theta_for(fam)).expect("ok");
            assert!(cop.pdf(0.4, 0.6).expect("ok") >= 0.0);
            let mut rng = LcgRng::new(7);
            let s = cop.sample(100, &mut rng).expect("ok");
            assert_eq!(s.len(), 200);
            for &x in &s {
                assert!(x > 0.0 && x < 1.0, "{fam:?}: sample {x} not in (0,1)");
            }
        }
    }

    #[test]
    fn tail_dependence_signature() {
        let c = ArchimedeanCopula::new(ArchimedeanFamily::Clayton, 2.0).expect("ok");
        let (lo, hi) = c.tail_dependence();
        assert!(lo > 0.0 && (hi - 0.0).abs() < 1e-15);
        let g = ArchimedeanCopula::new(ArchimedeanFamily::Gumbel, 2.0).expect("ok");
        let (lo, hi) = g.tail_dependence();
        assert!((lo - 0.0).abs() < 1e-15 && hi > 0.0);
    }

    #[test]
    fn invalid_theta_rejected() {
        assert!(ArchimedeanCopula::new(ArchimedeanFamily::Clayton, -1.0).is_err());
        assert!(ArchimedeanCopula::new(ArchimedeanFamily::Clayton, 0.0).is_err());
        assert!(ArchimedeanCopula::new(ArchimedeanFamily::Gumbel, 0.5).is_err());
        assert!(ArchimedeanCopula::new(ArchimedeanFamily::Frank, 0.0).is_err());
        assert!(ArchimedeanCopula::new(ArchimedeanFamily::Clayton, f64::NAN).is_err());
    }

    #[test]
    fn theta_from_tau_out_of_range() {
        assert!(ArchimedeanCopula::theta_from_tau(ArchimedeanFamily::Clayton, -0.1).is_err());
        assert!(ArchimedeanCopula::theta_from_tau(ArchimedeanFamily::Gumbel, 1.0).is_err());
        assert!(ArchimedeanCopula::theta_from_tau(ArchimedeanFamily::Frank, 1.5).is_err());
    }

    #[test]
    fn pdf_and_cdf_reject_out_of_range() {
        let cop = ArchimedeanCopula::new(ArchimedeanFamily::Clayton, 2.0).expect("ok");
        assert!(cop.pdf(1.5, 0.5).is_err());
        assert!(cop.cdf(-0.1, 0.5).is_err());
    }
}
