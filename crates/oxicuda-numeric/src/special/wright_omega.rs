//! Wright ω function on the real axis (`f64`).
//!
//! The Wright omega function `ω(z)` is the solution `w` of
//!
//! ```text
//! w + ln(w) = z
//! ```
//!
//! on the branch that is real-valued and strictly positive for real `z`. It is
//! related to the principal branch of the Lambert W function by the identity
//! `ω(z) = W₀(eᶻ)` (see [`crate::special::lambert_w`]). For real arguments `ω` is
//! smooth, strictly positive and strictly increasing, with limiting behaviour
//! `ω(z) → eᶻ` as `z → -∞` and `ω(z) → z - ln z` as `z → +∞`. Two notable values
//! are `ω(0) = Ω ≈ 0.5671432904` (the omega constant `W₀(1)`) and `ω(1) = 1`.
//!
//! Reference: R. M. Corless and D. J. Jeffrey, "The Wright ω Function",
//! in *Artificial Intelligence, Automated Reasoning, and Symbolic Computation*,
//! Lecture Notes in Computer Science 2385 (2002), pp. 76–89.
//!
//! The implementation seeds an asymptotic (`z - ln z`) or exponential (`eᶻ`)
//! initial guess and refines it with Halley's method on `f(w) = w + ln(w) - z`,
//! whose derivatives are `f'(w) = 1 + 1/w` and `f''(w) = -1/w²`.

use crate::error::{NumericError, NumericResult};

/// Maximum number of Halley refinement iterations.
const MAX_ITER: usize = 100;
/// Convergence tolerance on the (relative) Halley correction.
const TOL: f64 = 1.0e-15;

/// Evaluate the Wright ω function for a real argument `z`.
///
/// Returns the unique positive `w` satisfying `w + ln(w) = z`.
///
/// # Errors
/// Returns [`NumericError::InvalidParameter`] when `z` is not finite, and
/// [`NumericError::NumericalInstability`] / [`NumericError::NotConverged`] when the
/// Halley iteration cannot produce a positive, convergent root.
pub fn wright_omega(z: f64) -> NumericResult<f64> {
    if !z.is_finite() {
        return Err(NumericError::InvalidParameter(format!(
            "wright_omega: argument must be finite, got z={z}"
        )));
    }

    // Initial guess: asymptotic `z - ln z` for large z, exponential `eᶻ` otherwise.
    let mut w = if z >= 1.0 { z - z.ln() } else { z.exp() };
    // Guard against underflow / overflow of the seed to a non-positive value.
    if !w.is_finite() || w <= 0.0 {
        w = f64::MIN_POSITIVE;
    }

    for k in 0..MAX_ITER {
        // f(w) = w + ln(w) - z,  f'(w) = 1 + 1/w,  f''(w) = -1/w².
        let f = w + w.ln() - z;
        let fp = 1.0 + 1.0 / w;
        let fpp = -1.0 / (w * w);
        let denom = 2.0 * fp * fp - f * fpp;
        if denom.abs() < 1.0e-300 {
            return Err(NumericError::NumericalInstability(format!(
                "wright_omega: Halley denominator vanished at z={z}, iter={k}"
            )));
        }
        let step = 2.0 * f * fp / denom;
        let mut w_new = w - step;
        // `ln` is undefined for w ≤ 0: damp toward zero while staying positive.
        if !w_new.is_finite() || w_new <= 0.0 {
            w_new = 0.5 * w;
        }
        let delta = (w_new - w).abs();
        w = w_new;
        if !w.is_finite() {
            return Err(NumericError::NumericalInstability(format!(
                "wright_omega: iterate diverged at z={z}, iter={k}"
            )));
        }
        if delta <= TOL * w.abs().max(1.0) {
            return Ok(w);
        }
    }

    // Fall back on a residual check before declaring non-convergence.
    let residual = (w + w.ln() - z).abs();
    if residual <= 1.0e-11 {
        Ok(w)
    } else {
        Err(NumericError::NotConverged {
            iter: MAX_ITER,
            residual,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::special::lambert_w::lambert_w0;

    #[test]
    fn defining_equation_holds() {
        for &z in &[-2.0, -1.0, 0.0, 1.0, 5.0, 20.0, 100.0] {
            let w = wright_omega(z).expect("ok");
            assert!(w > 0.0, "omega must be positive: z={z}, w={w}");
            assert!(w.is_finite());
            let residual = w + w.ln() - z;
            assert!(
                residual.abs() < 1.0e-12,
                "z={z}, w={w}, residual={residual:e}"
            );
        }
    }

    #[test]
    fn omega_constant_at_zero() {
        // ω(0) = W₀(1) = Ω (the omega constant) ≈ 0.5671432904097838.
        let w = wright_omega(0.0).expect("ok");
        assert!((w - 0.567_143_290_409_783_8).abs() < 1.0e-12, "w={w}");
        // Defining check: w + ln(w) == 0.
        assert!((w + w.ln()).abs() < 1.0e-12);
    }

    #[test]
    fn omega_at_one_is_one() {
        // w + ln(w) = 1  ⇒  w = 1 (since 1 + ln 1 = 1).
        let w = wright_omega(1.0).expect("ok");
        assert!((w - 1.0).abs() < 1.0e-12, "w={w}");
    }

    #[test]
    fn matches_lambert_w0_of_exp_z() {
        // ω(z) = W₀(eᶻ) for real z (where eᶻ does not overflow).
        for &z in &[-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0] {
            let w = wright_omega(z).expect("ok");
            let lw = lambert_w0(z.exp()).expect("ok");
            assert!((w - lw).abs() < 1.0e-10, "z={z}, w={w}, W0(e^z)={lw}");
        }
    }

    #[test]
    fn monotone_increasing() {
        let mut prev = wright_omega(-5.0).expect("ok");
        let mut z = -4.5;
        while z <= 10.0 + 1.0e-9 {
            let cur = wright_omega(z).expect("ok");
            assert!(cur > prev, "not strictly increasing at z={z}");
            prev = cur;
            z += 0.5;
        }
    }

    #[test]
    fn very_large_argument_no_overflow() {
        let w = wright_omega(700.0).expect("ok");
        assert!(w.is_finite() && w > 0.0, "w={w}");
        let residual = w + w.ln() - 700.0;
        assert!(residual.abs() < 1.0e-9, "residual={residual:e}");
    }

    #[test]
    fn very_negative_argument() {
        let w = wright_omega(-30.0).expect("ok");
        assert!(w > 0.0 && w.is_finite(), "w={w}");
        let residual = w + w.ln() + 30.0;
        assert!(residual.abs() < 1.0e-10, "residual={residual:e}");
    }

    #[test]
    fn rejects_non_finite() {
        assert!(wright_omega(f64::NAN).is_err());
        assert!(wright_omega(f64::INFINITY).is_err());
        assert!(wright_omega(f64::NEG_INFINITY).is_err());
    }
}
