//! Directional-statistics estimators in the Mardia (1972) tradition: the
//! maximum-likelihood concentration parameter κ obtained by Newton iteration on
//! the Bessel ratio `A(κ) = I₁(κ)/I₀(κ) = R̄`, and the Watson-Williams `F`-test
//! for equality of mean directions across two or more samples.
//!
//! These complement [`crate::circular::circular`], which provides the von Mises
//! PDF/CDF, the Mardia *approximation* for κ, and the Rayleigh uniformity test.
//! The estimator here solves the κ equation exactly rather than via the piecewise
//! rational approximation.
//!
//! All angles are in radians.
//!
//! # References
//! - Mardia, K.V. (1972). *Statistics of Directional Data*. Academic Press.
//! - Mardia, K.V. & Jupp, P.E. (2000). *Directional Statistics*, 2nd ed. Wiley.
//! - Watson, G.S. & Williams, E.J. (1956). "On the construction of significance
//!   tests on the circle and the sphere". *Biometrika* 43(3-4):344-352.
//! - Fisher, N.I. (1993). *Statistical Analysis of Circular Data*. Cambridge.

use crate::circular::CircularError;
use crate::circular::circular::CircularResult;

// ─── Bessel helpers ─────────────────────────────────────────────────────────

/// Bessel ratio `A(κ) = I₁(κ) / I₀(κ)`, the mean resultant length of a von Mises
/// distribution with concentration `κ`. Monotone increasing from `A(0)=0` to
/// `A(∞)=1`.
///
/// Evaluated by the Perron/Gauss continued fraction
/// `I_{ν+1}(x)/I_ν(x) = 1 / (2(ν+1)/x + I_{ν+2}/I_{ν+1})`, run with the modified
/// Lentz algorithm. This is numerically stable for *all* `κ > 0`, unlike forming
/// the ratio of the two series (which overflow individually for large `κ`).
fn bessel_ratio(kappa: f64) -> f64 {
    if kappa <= 0.0 {
        return 0.0;
    }
    // Modified Lentz (Numerical Recipes §5.2) for the continued fraction
    //   h = b0 + a1/(b1 + a2/(b2 + ...)),  b0 = 0,  a_k = 1,  b_k = 2k/κ.
    // This converges to h = I₁(κ)/I₀(κ) directly.
    let tiny = 1e-300_f64;
    let mut f = tiny; // b0 = 0 → tiny
    let mut c = f;
    let mut d = 0.0_f64;
    for k in 1..=10_000_u32 {
        let b = 2.0 * k as f64 / kappa;
        d += b; // a_k = 1
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + 1.0 / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if k > 4 && (delta - 1.0).abs() < 1e-15 {
            break;
        }
    }
    f.clamp(0.0, 1.0 - 1e-300)
}

/// Derivative `A'(κ) = 1 - A(κ)² - A(κ)/κ` of the Bessel ratio (used in Newton).
#[inline]
fn bessel_ratio_deriv(kappa: f64, a: f64) -> f64 {
    if kappa <= 0.0 {
        // A'(0) = 1/2.
        return 0.5;
    }
    1.0 - a * a - a / kappa
}

// ─── Mean direction + resultant length ──────────────────────────────────────

/// Summary of the first trigonometric moment of a sample of angles.
#[derive(Debug, Clone, Copy)]
pub struct MeanDirection {
    /// Mean direction `θ̄ = atan2(Σ sin, Σ cos) ∈ (−π, π]`.
    pub theta_bar: f64,
    /// Mean resultant length `R̄ ∈ [0, 1]`.
    pub r_bar: f64,
    /// Resultant length `R = n · R̄`.
    pub r: f64,
    /// Sample size.
    pub n: usize,
}

/// Compute the mean direction and resultant length of a sample of angles.
///
/// # Errors
/// - [`CircularError::EmptyInput`] if `angles` is empty.
pub fn mean_direction(angles: &[f64]) -> CircularResult<MeanDirection> {
    if angles.is_empty() {
        return Err(CircularError::EmptyInput);
    }
    let n = angles.len();
    let sum_sin: f64 = angles.iter().map(|a| a.sin()).sum();
    let sum_cos: f64 = angles.iter().map(|a| a.cos()).sum();
    let r = (sum_sin * sum_sin + sum_cos * sum_cos).sqrt();
    let r_bar = r / n as f64;
    let theta_bar = sum_sin.atan2(sum_cos);
    Ok(MeanDirection {
        theta_bar,
        r_bar,
        r,
        n,
    })
}

// ─── MLE concentration κ via Newton ─────────────────────────────────────────

/// Maximum-likelihood estimate of the von Mises concentration `κ`, found by
/// solving `A(κ) = R̄` with Newton's method on the Bessel ratio.
///
/// The mean resultant length `r_bar` must lie in `[0, 1)`. For `R̄ = 0` the MLE
/// is `κ = 0`; as `R̄ → 1` the estimate diverges, so it is capped at a large but
/// finite value.
///
/// # Errors
/// - [`CircularError::InvalidKappa`] if `r_bar` is non-finite or outside `[0, 1)`.
/// - [`CircularError::NotConverged`] if Newton fails to converge.
pub fn kappa_mle(r_bar: f64) -> CircularResult<f64> {
    if !r_bar.is_finite() || !(0.0..1.0).contains(&r_bar) {
        return Err(CircularError::InvalidKappa(format!(
            "r_bar must be finite and in [0, 1), got {r_bar}"
        )));
    }
    if r_bar < 1e-12 {
        return Ok(0.0);
    }
    const KAPPA_MAX: f64 = 1.0e5;
    // If the data are so concentrated that even κ_max under-shoots R̄, the MLE is
    // effectively at the cap: report it rather than iterating fruitlessly.
    if r_bar >= bessel_ratio(KAPPA_MAX) {
        return Ok(KAPPA_MAX);
    }

    // Safeguarded Newton: because A(κ) is strictly increasing, we maintain a
    // bracket [lo, hi] with A(lo) < R̄ < A(hi) and fall back to bisection
    // whenever a Newton step would leave the bracket. This converges for every
    // admissible R̄ without the brittleness of bare Newton near A ≈ 1.
    let mut lo = 0.0_f64;
    let mut hi = KAPPA_MAX;
    let mut kappa = mardia_seed(r_bar).clamp(1e-6, KAPPA_MAX);
    for _ in 0..200 {
        let a = bessel_ratio(kappa);
        let f = a - r_bar;
        if f.abs() < 1e-12 {
            return Ok(kappa);
        }
        // Tighten the bracket using the sign of f (A monotone increasing).
        if f > 0.0 {
            hi = kappa;
        } else {
            lo = kappa;
        }
        let df = bessel_ratio_deriv(kappa, a);
        // Newton candidate; reject if non-finite, out of the bracket, or df→0.
        let next = if df.abs() < 1e-300 {
            f64::NAN
        } else {
            kappa - f / df
        };
        let candidate = if next.is_finite() && next > lo && next < hi {
            next
        } else {
            0.5 * (lo + hi)
        };
        if (candidate - kappa).abs() < 1e-12 * (1.0 + kappa) {
            return Ok(candidate);
        }
        kappa = candidate;
    }
    // Monotone bracket guarantees the midpoint is within (hi-lo)/2 of the root.
    if (hi - lo) < 1e-6 * (1.0 + kappa) {
        return Ok(0.5 * (lo + hi));
    }
    Err(CircularError::NotConverged)
}

/// Mardia & Jupp (2000) closed-form approximation used to seed Newton.
#[inline]
fn mardia_seed(r: f64) -> f64 {
    let r = r.clamp(0.0, 0.9999);
    if r < 0.53 {
        2.0 * r + r.powi(3) + 5.0 * r.powi(5) / 6.0
    } else if r < 0.85 {
        -0.4 + 1.39 * r + 0.43 / (1.0 - r)
    } else {
        let denom = (r.powi(3) - 4.0 * r.powi(2) + 3.0 * r).abs();
        if denom < 1e-10 { 5000.0 } else { 1.0 / denom }
    }
}

/// Fit the von Mises concentration `κ` directly from a sample of angles via the
/// exact (Newton) MLE.
///
/// Returns the [`MeanDirection`] summary together with `κ̂`.
///
/// # Errors
/// - [`CircularError::EmptyInput`] if `angles` is empty.
/// - propagates errors from [`kappa_mle`].
pub fn kappa_mle_from_angles(angles: &[f64]) -> CircularResult<(MeanDirection, f64)> {
    let md = mean_direction(angles)?;
    // Saturate just below 1 so the MLE remains finite for fully concentrated data.
    let r_bar = md.r_bar.min(1.0 - 1e-12);
    let kappa = kappa_mle(r_bar)?;
    Ok((md, kappa))
}

// ─── Watson-Williams F-test ─────────────────────────────────────────────────

/// Result of a Watson-Williams `F`-test for equality of mean directions.
#[derive(Debug, Clone, Copy)]
pub struct WatsonWilliamsResult {
    /// `F` statistic.
    pub statistic: f64,
    /// Numerator degrees of freedom `q − 1` (q = number of groups).
    pub df1: usize,
    /// Denominator degrees of freedom `N − q`.
    pub df2: usize,
    /// Pooled mean resultant length `R̄_pooled` (used for the κ correction).
    pub r_bar_pooled: f64,
    /// Upper-tail `p`-value `P(F_{df1,df2} > F)`.
    pub p_value: f64,
}

/// Watson-Williams high-concentration `F`-test of the null hypothesis that two
/// or more samples share a common mean direction.
///
/// For `q` groups with resultant lengths `R_i = n_i R̄_i` and pooled resultant
/// `R = |Σ vector means|`, the statistic is
///
/// ```text
/// F = (1 + 3/(8κ̂)) · (N − q)(Σ R_i − R) / ((q − 1)(N − Σ R_i))
/// ```
///
/// where `κ̂` is the MLE from the pooled `R̄`. The `(1 + 3/(8κ̂))` factor is the
/// Stephens (1972) small-`κ` correction. The test assumes reasonably large
/// concentration (`R̄ > 0.45` recommended); the result is still returned for
/// smaller `R̄` so callers can decide.
///
/// # Errors
/// - [`CircularError::InsufficientSamples`] if fewer than two groups, or any
///   group has fewer than two angles, or the total `N ≤ q`.
/// - propagates errors from [`kappa_mle`].
pub fn watson_williams_test(groups: &[&[f64]]) -> CircularResult<WatsonWilliamsResult> {
    let q = groups.len();
    if q < 2 {
        return Err(CircularError::InsufficientSamples { got: q, need: 2 });
    }
    let mut n_total = 0usize;
    let mut sum_ri = 0.0_f64; // Σ R_i
    let mut total_sin = 0.0_f64;
    let mut total_cos = 0.0_f64;
    for g in groups {
        if g.len() < 2 {
            return Err(CircularError::InsufficientSamples {
                got: g.len(),
                need: 2,
            });
        }
        n_total += g.len();
        let s: f64 = g.iter().map(|a| a.sin()).sum();
        let c: f64 = g.iter().map(|a| a.cos()).sum();
        sum_ri += (s * s + c * c).sqrt();
        total_sin += s;
        total_cos += c;
    }
    if n_total <= q {
        return Err(CircularError::InsufficientSamples {
            got: n_total,
            need: q + 1,
        });
    }

    // Pooled resultant.
    let r_pooled = (total_sin * total_sin + total_cos * total_cos).sqrt();
    let r_bar_pooled = r_pooled / n_total as f64;

    // κ̂ from the pooled mean resultant length (for the Stephens correction).
    let r_bar_clamped = r_bar_pooled.min(1.0 - 1e-12);
    let kappa = kappa_mle(r_bar_clamped)?;
    let correction = if kappa > 0.0 {
        1.0 + 3.0 / (8.0 * kappa)
    } else {
        1.0
    };

    let df1 = q - 1;
    let df2 = n_total - q;
    let between = sum_ri - r_pooled;
    let within = n_total as f64 - sum_ri;

    let statistic = if within.abs() < 1e-300 {
        f64::INFINITY
    } else {
        correction * (df2 as f64 * between) / (df1 as f64 * within)
    };

    let p_value = if statistic.is_finite() {
        f_sf(statistic, df1 as f64, df2 as f64)
    } else {
        0.0
    };

    Ok(WatsonWilliamsResult {
        statistic,
        df1,
        df2,
        r_bar_pooled,
        p_value,
    })
}

/// Upper-tail `F`-distribution survival function `P(F_{d1,d2} > x)` via the
/// regularised incomplete beta function.
fn f_sf(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    // P(F > x) = I_{d2/(d2+d1 x)}(d2/2, d1/2).
    let denom = d2 + d1 * x;
    if denom <= 0.0 {
        return 0.0;
    }
    let xb = d2 / denom;
    match crate::special::betainc::betainc(d2 / 2.0, d1 / 2.0, xb) {
        Ok(v) => v.clamp(0.0, 1.0),
        Err(_) => f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ── Bessel ratio ──────────────────────────────────────────────────────────

    #[test]
    fn bessel_ratio_zero_is_zero() {
        assert_eq!(bessel_ratio(0.0), 0.0);
    }

    #[test]
    fn bessel_ratio_known_value() {
        // A(1) = I₁(1)/I₀(1) ≈ 0.4463899658965346 (verified against the series).
        assert!((bessel_ratio(1.0) - 0.4463899658965346).abs() < 1e-10);
        // A(5) ≈ 0.8933831370440853.
        assert!((bessel_ratio(5.0) - 0.8933831370440853).abs() < 1e-10);
    }

    #[test]
    fn bessel_ratio_large_kappa_stable() {
        // For large κ, A(κ) ≈ 1 - 1/(2κ); must not overflow to NaN/Inf.
        let a = bessel_ratio(5000.0);
        assert!(a.is_finite());
        assert!((a - (1.0 - 1.0 / (2.0 * 5000.0))).abs() < 1e-4, "A={a}");
    }

    #[test]
    fn bessel_ratio_monotone_and_bounded() {
        let mut prev = bessel_ratio(0.01);
        for i in 1..=50 {
            let k = i as f64 * 0.2;
            let a = bessel_ratio(k);
            assert!(a > prev - 1e-12, "A not monotone at κ={k}");
            assert!((0.0..1.0).contains(&a), "A(κ)={a} out of [0,1)");
            prev = a;
        }
    }

    // ── Mean direction ──────────────────────────────────────────────────────────

    #[test]
    fn mean_direction_recovers_angle() {
        let mu = PI / 3.0;
        let angles: Vec<f64> = (0..100).map(|i| mu + 0.02 * (i as f64).sin()).collect();
        let md = mean_direction(&angles).expect("ok");
        assert!((md.theta_bar - mu).abs() < 0.05);
        assert!(md.r_bar > 0.99);
    }

    #[test]
    fn mean_direction_uniform_small_r() {
        let n = 360;
        let angles: Vec<f64> = (0..n).map(|i| 2.0 * PI * i as f64 / n as f64).collect();
        let md = mean_direction(&angles).expect("ok");
        assert!(md.r_bar < 1e-10, "r_bar={}", md.r_bar);
    }

    #[test]
    fn mean_direction_empty_errors() {
        assert!(matches!(
            mean_direction(&[]),
            Err(CircularError::EmptyInput)
        ));
    }

    // ── κ MLE ───────────────────────────────────────────────────────────────────

    #[test]
    fn kappa_mle_inverts_bessel_ratio() {
        // For several κ, A(κ)=R̄ and the MLE should recover κ.
        for &k_true in &[0.5_f64, 1.0, 2.0, 5.0, 10.0] {
            let r_bar = bessel_ratio(k_true);
            let k_hat = kappa_mle(r_bar).expect("ok");
            assert!(
                (k_hat - k_true).abs() < 1e-6 * (1.0 + k_true),
                "κ_true={k_true}, κ_hat={k_hat}"
            );
        }
    }

    #[test]
    fn kappa_mle_zero_for_zero_r() {
        assert_eq!(kappa_mle(0.0).expect("ok"), 0.0);
    }

    #[test]
    fn kappa_mle_large_for_high_r() {
        let k = kappa_mle(0.99).expect("ok");
        assert!(k > 40.0, "κ={k}");
    }

    #[test]
    fn kappa_mle_rejects_out_of_range() {
        assert!(kappa_mle(1.0).is_err());
        assert!(kappa_mle(-0.1).is_err());
        assert!(kappa_mle(f64::NAN).is_err());
    }

    #[test]
    fn kappa_mle_from_angles_concentrated() {
        let angles: Vec<f64> = (0..200)
            .map(|i| 0.3 + 0.05 * (i as f64 * 0.1).sin())
            .collect();
        let (md, kappa) = kappa_mle_from_angles(&angles).expect("ok");
        assert!(md.r_bar > 0.99);
        assert!(kappa > 50.0, "κ={kappa}");
    }

    // ── Watson-Williams ─────────────────────────────────────────────────────────

    #[test]
    fn watson_williams_same_mean_not_significant() {
        // Two concentrated samples around the SAME direction → large p-value.
        let g1: Vec<f64> = (0..60)
            .map(|i| 0.5 + 0.05 * (i as f64 * 0.3).sin())
            .collect();
        let g2: Vec<f64> = (0..60)
            .map(|i| 0.5 + 0.05 * (i as f64 * 0.7).cos())
            .collect();
        let res = watson_williams_test(&[&g1, &g2]).expect("ok");
        assert_eq!(res.df1, 1);
        assert_eq!(res.df2, 118);
        assert!(res.p_value > 0.05, "p={}", res.p_value);
    }

    #[test]
    fn watson_williams_different_means_significant() {
        // Two concentrated samples around clearly DIFFERENT directions.
        let g1: Vec<f64> = (0..80)
            .map(|i| 0.2 + 0.03 * (i as f64 * 0.3).sin())
            .collect();
        let g2: Vec<f64> = (0..80)
            .map(|i| 1.6 + 0.03 * (i as f64 * 0.5).cos())
            .collect();
        let res = watson_williams_test(&[&g1, &g2]).expect("ok");
        assert!(res.statistic > 10.0, "F={}", res.statistic);
        assert!(res.p_value < 0.01, "p={}", res.p_value);
    }

    #[test]
    fn watson_williams_three_groups_df() {
        let g1: Vec<f64> = (0..30).map(|i| 0.1 + 0.02 * (i as f64).sin()).collect();
        let g2: Vec<f64> = (0..40).map(|i| 0.1 + 0.02 * (i as f64).cos()).collect();
        let g3: Vec<f64> = (0..50)
            .map(|i| 0.1 + 0.02 * (i as f64 * 0.3).sin())
            .collect();
        let res = watson_williams_test(&[&g1, &g2, &g3]).expect("ok");
        assert_eq!(res.df1, 2);
        assert_eq!(res.df2, 30 + 40 + 50 - 3);
        assert!((0.0..=1.0).contains(&res.p_value));
    }

    #[test]
    fn watson_williams_too_few_groups_errors() {
        let g1 = [0.1_f64, 0.2, 0.3];
        assert!(matches!(
            watson_williams_test(&[&g1[..]]),
            Err(CircularError::InsufficientSamples { .. })
        ));
    }

    #[test]
    fn watson_williams_tiny_group_errors() {
        let g1 = [0.1_f64, 0.2, 0.3];
        let g2 = [0.5_f64];
        assert!(matches!(
            watson_williams_test(&[&g1[..], &g2[..]]),
            Err(CircularError::InsufficientSamples { got: 1, need: 2 })
        ));
    }

    #[test]
    fn f_sf_bounds() {
        assert!((f_sf(0.0, 2.0, 10.0) - 1.0).abs() < 1e-12);
        let p = f_sf(100.0, 2.0, 10.0);
        assert!(p < 0.001, "p={p}");
        let mid = f_sf(1.0, 5.0, 5.0);
        assert!((0.0..=1.0).contains(&mid));
    }
}
