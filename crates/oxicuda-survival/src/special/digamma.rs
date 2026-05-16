//! Digamma function `ψ(x) = d/dx ln Γ(x)`.

/// Compute digamma `ψ(x)` for `x > 0`.
///
/// Uses recurrence to push x >= 6 then an asymptotic expansion.
#[must_use]
pub fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        // reflection: ψ(1-x) - ψ(x) = π cot(πx) — fall back for negative.
        let pi = std::f64::consts::PI;
        return digamma(1.0 - x) - pi / (pi * x).tan();
    }
    let mut y = x;
    let mut result = 0.0_f64;
    while y < 6.0 {
        result -= 1.0 / y;
        y += 1.0;
    }
    let inv = 1.0 / y;
    let inv2 = inv * inv;
    // Asymptotic: ψ(y) ≈ ln y - 1/(2y) - Σ B_{2k}/(2k * y^{2k})
    let mut series = y.ln() - 0.5 * inv;
    series -= inv2
        * (1.0 / 12.0
            - inv2
                * (1.0 / 120.0
                    - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0 - inv2 * (1.0 / 132.0)))));
    result + series
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digamma_known_values() {
        // ψ(1) = -γ (Euler-Mascheroni constant ≈ -0.5772156649)
        let gamma = 0.577_215_664_901_532_9;
        assert!((digamma(1.0) + gamma).abs() < 1.0e-8);
    }

    #[test]
    fn digamma_recurrence() {
        // ψ(x+1) = ψ(x) + 1/x
        for k in 1..10 {
            let x = 0.5 + k as f64;
            let lhs = digamma(x + 1.0);
            let rhs = digamma(x) + 1.0 / x;
            assert!((lhs - rhs).abs() < 1.0e-9, "k={k}");
        }
    }

    #[test]
    fn digamma_two() {
        // ψ(2) = 1 - γ
        let gamma = 0.577_215_664_901_532_9;
        assert!((digamma(2.0) - (1.0 - gamma)).abs() < 1.0e-8);
    }
}
