//! Digamma function `psi(x) = d/dx ln(Gamma(x))`.
//!
//! Uses recurrence to push small arguments into the large-x regime,
//! then an asymptotic expansion.

/// Digamma function `psi(x)` for `x > 0`.
#[must_use]
pub fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        // Reflection: psi(1-x) - psi(x) = pi * cot(pi x).
        // We only support real positive x; for x<=0 with non-integer x, use reflection.
        if (x - x.round()).abs() < 1e-14 {
            // Pole; return NaN.
            return f64::NAN;
        }
        let pi = std::f64::consts::PI;
        return digamma(1.0 - x) - pi / (pi * x).tan();
    }
    let mut y = x;
    let mut result = 0.0;
    // Recurrence: psi(x) = psi(x+1) - 1/x
    while y < 6.0 {
        result -= 1.0 / y;
        y += 1.0;
    }
    // Asymptotic: psi(y) ~ ln(y) - 1/(2y) - 1/(12 y^2) + 1/(120 y^4) - 1/(252 y^6)
    let inv = 1.0 / y;
    let inv2 = inv * inv;
    result += y.ln() - 0.5 * inv;
    result -= inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0)));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digamma_1_equals_neg_gamma() {
        // psi(1) = -gamma (Euler-Mascheroni)
        let gamma_em = 0.577_215_664_901_532_9;
        assert!((digamma(1.0) + gamma_em).abs() < 1e-7);
    }

    #[test]
    fn digamma_2() {
        // psi(2) = 1 - gamma
        let gamma_em = 0.577_215_664_901_532_9;
        let expected = 1.0 - gamma_em;
        assert!((digamma(2.0) - expected).abs() < 1e-7);
    }

    #[test]
    fn digamma_recurrence() {
        // psi(x+1) - psi(x) = 1/x
        for x in [1.5, 2.7, 5.5, 10.0] {
            assert!((digamma(x + 1.0) - digamma(x) - 1.0 / x).abs() < 1e-8);
        }
    }
}
