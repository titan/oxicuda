//! Log-gamma function `ln Γ(x)` via the Lanczos approximation.

/// Compute `ln Γ(x)` for `x > 0` via Lanczos g=7, n=9.
/// Accurate to roughly 1e-14 across the positive reals.
#[must_use]
pub fn gammaln(x: f64) -> f64 {
    // Lanczos coefficients (g=7, n=9)
    const G: f64 = 7.0;
    const COEFS: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // reflection: Γ(x)Γ(1-x) = π/sin(πx)
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - gammaln(1.0 - x);
    }
    let xm = x - 1.0;
    let mut a = COEFS[0];
    for (i, c) in COEFS.iter().enumerate().skip(1) {
        a += c / (xm + i as f64);
    }
    let t = xm + G + 0.5;
    0.5 * std::f64::consts::TAU.ln() + (xm + 0.5) * t.ln() - t + a.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gammaln_integer_factorial() {
        // ln(n!) = gammaln(n+1)
        assert!((gammaln(1.0) - 0.0).abs() < 1.0e-10);
        assert!((gammaln(2.0) - 0.0).abs() < 1.0e-10);
        assert!((gammaln(3.0) - 2.0_f64.ln()).abs() < 1.0e-10);
        assert!((gammaln(4.0) - 6.0_f64.ln()).abs() < 1.0e-10);
        assert!((gammaln(5.0) - 24.0_f64.ln()).abs() < 1.0e-10);
    }

    #[test]
    fn gammaln_half() {
        // Γ(1/2) = √π
        let v = gammaln(0.5);
        let expected = std::f64::consts::PI.sqrt().ln();
        assert!((v - expected).abs() < 1.0e-8);
    }

    #[test]
    fn gammaln_monotone_for_large() {
        // For x >= 2, gammaln is monotone increasing
        let mut prev = gammaln(2.0);
        for k in 3..20 {
            let cur = gammaln(k as f64);
            assert!(cur > prev);
            prev = cur;
        }
    }
}
