//! Exponential integral Ei(x) for real x.
//!
//! `Ei(x) = -∫_{-x}^∞ e^{-t}/t dt = γ + ln|x| + Σ_{k≥1} x^k / (k · k!)`.
//! For x > 0: use the series for |x| < 6, asymptotic for large x.
//! For x < 0: `Ei(-y) = -E_1(y)` with `E_1` computed via continued fraction.

use crate::error::{NumericError, NumericResult};

const EULER: f64 = 0.577_215_664_901_532_9;

fn ei_series(x: f64) -> f64 {
    // Ei(x) = γ + ln|x| + Σ x^k / (k k!)
    let mut term = 1.0_f64;
    let mut sum = 0.0_f64;
    for k in 1..200 {
        term *= x / k as f64;
        let add = term / k as f64;
        sum += add;
        if add.abs() < 1.0e-18 * sum.abs().max(1.0) {
            break;
        }
    }
    EULER + x.abs().ln() + sum
}

fn e1_continued_fraction(x: f64) -> f64 {
    // Modified Lentz: E_1(x) = e^{-x} / (x + 1 / (1 + 1 / (x + 2 / (1 + 2 / (x + 3 / ...)))))
    let mut b = x + 1.0;
    let tiny = 1.0e-300;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    for k in 1..200 {
        let a = -((k * k) as f64);
        b += 2.0;
        d = 1.0 / (a * d + b);
        c = b + a / c;
        let delta = c * d;
        h *= delta;
        if (delta - 1.0).abs() < 1.0e-15 {
            break;
        }
    }
    h * (-x).exp()
}

/// Ei(x).
pub fn exponential_integral(x: f64) -> NumericResult<f64> {
    if x == 0.0 {
        return Err(NumericError::OutOfDomain {
            value: x,
            function: "exponential_integral (singular at 0)".into(),
        });
    }
    if x > 0.0 {
        if x < 6.0 {
            Ok(ei_series(x))
        } else {
            // asymptotic: Ei(x) ≈ e^x/x · (1 + 1/x + 2/x² + 6/x³ + ...)
            let mut term = 1.0_f64;
            let mut sum = 1.0_f64;
            for k in 1..30 {
                term *= k as f64 / x;
                let next = sum + term;
                if (next - sum).abs() < 1.0e-15 * next.abs() {
                    return Ok(x.exp() / x * next);
                }
                sum = next;
            }
            Ok(x.exp() / x * sum)
        }
    } else {
        // Ei(-y) = -E_1(y)
        let y = -x;
        if y < 1.0 {
            // Series: E_1(y) = -γ - ln(y) - Σ_{k≥1} (-y)^k / (k k!)
            let mut term = 1.0_f64;
            let mut sum = 0.0_f64;
            for k in 1..200 {
                term *= -y / k as f64;
                let add = term / k as f64;
                sum += add;
                if add.abs() < 1.0e-18 {
                    break;
                }
            }
            Ok(-(-EULER - y.ln() - sum))
        } else {
            Ok(-e1_continued_fraction(y))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ei_one() {
        // Ei(1) ≈ 1.8951178163559367
        let r = exponential_integral(1.0).expect("ok");
        assert!((r - 1.895_117_816_355_936_7).abs() < 1.0e-8);
    }

    #[test]
    fn ei_two() {
        // Ei(2) ≈ 4.954234356001892
        let r = exponential_integral(2.0).expect("ok");
        assert!((r - 4.954_234_356_001_892).abs() < 1.0e-8);
    }

    #[test]
    fn ei_neg_one_e1() {
        // Ei(-1) = -E_1(1) ≈ -0.21938393439552...
        let r = exponential_integral(-1.0).expect("ok");
        assert!((r + 0.219_383_934_395_520_3).abs() < 1.0e-6);
    }

    #[test]
    fn ei_at_zero_err() {
        let r = exponential_integral(0.0);
        assert!(matches!(r, Err(NumericError::OutOfDomain { .. })));
    }
}
