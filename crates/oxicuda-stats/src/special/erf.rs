//! Error function `erf`, complementary error function `erfc`, and inverse `erfinv`.
//!
//! `erf` uses Abramowitz-Stegun 7.1.26 polynomial approximation (max error ~1.5e-7).
//! `erfinv` uses Newton iteration starting from a rational approximation seed.

use crate::error::{StatsError, StatsResult};

const A1: f64 = 0.254_829_592;
const A2: f64 = -0.284_496_736;
const A3: f64 = 1.421_413_741;
const A4: f64 = -1.453_152_027;
const A5: f64 = 1.061_405_429;
const P: f64 = 0.327_591_1;

/// Error function `erf(x)`.
///
/// Abramowitz-Stegun 7.1.26 (max error ~1.5e-7 in absolute terms).
#[must_use]
pub fn erf(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + P * ax);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-ax * ax).exp();
    sign * y
}

/// Complementary error function: `erfc(x) = 1 - erf(x)`.
#[must_use]
pub fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

/// Inverse error function. Uses an initial rational approximation followed by Newton iteration.
///
/// Returns `StatsError::ProbabilityOutOfRange` if `|p| >= 1`.
pub fn erfinv(p: f64) -> StatsResult<f64> {
    if !p.is_finite() || p <= -1.0 || p >= 1.0 {
        return Err(StatsError::ProbabilityOutOfRange { value: p });
    }
    if p == 0.0 {
        return Ok(0.0);
    }
    // Acklam's rational seed: simpler form via tabulated coefficients.
    let sign = if p < 0.0 { -1.0 } else { 1.0 };
    let pa = p.abs();
    // Initial seed from Winitzki approximation
    let ln1 = (1.0 - pa * pa).ln();
    let a = 0.147;
    let term = 2.0 / (std::f64::consts::PI * a) + ln1 / 2.0;
    let mut x = sign * (-(term) + (term * term - ln1 / a).sqrt()).sqrt();
    // Refine via Newton iteration: f(x) = erf(x) - p, f'(x) = 2/sqrt(pi) * exp(-x^2)
    let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
    for _ in 0..8 {
        let fx = erf(x) - p;
        let fpx = two_over_sqrt_pi * (-x * x).exp();
        if fpx.abs() < 1e-300 {
            break;
        }
        let dx = fx / fpx;
        x -= dx;
        if dx.abs() < 1e-15 {
            break;
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_known_values() {
        assert!((erf(0.0)).abs() < 1e-12);
        assert!((erf(1.0) - 0.842_700_793).abs() < 1e-6);
        assert!((erf(2.0) - 0.995_322_265).abs() < 1e-6);
        assert!((erf(-1.0) + 0.842_700_793).abs() < 1e-6);
    }

    #[test]
    fn erfc_complement() {
        assert!((erfc(0.0) - 1.0).abs() < 1e-12);
        assert!((erfc(1.0) + erf(1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn erfinv_roundtrip() {
        for &p in &[0.1, 0.5, 0.9, -0.3, -0.7] {
            let x = erfinv(p).expect("ok");
            assert!((erf(x) - p).abs() < 1e-6);
        }
    }

    #[test]
    fn erfinv_rejects_out_of_range() {
        assert!(erfinv(1.0).is_err());
        assert!(erfinv(-1.0).is_err());
        assert!(erfinv(2.0).is_err());
    }
}
