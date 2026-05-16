//! Riemann zeta `ζ(s)` for real `s` via the Euler-Maclaurin sum (s > 1) and
//! the functional equation for s < 1.

use crate::error::NumericResult;

const PI: f64 = std::f64::consts::PI;

fn gamma_lanczos(x: f64) -> f64 {
    let p = [
        676.5203681218851_f64,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        PI / ((PI * x).sin() * gamma_lanczos(1.0 - x))
    } else {
        let x_shift = x - 1.0;
        let mut a = 0.999_999_999_999_809_93_f64;
        for (i, &pi) in p.iter().enumerate() {
            a += pi / (x_shift + (i as f64) + 1.0);
        }
        let t = x_shift + p.len() as f64 - 0.5;
        (2.0 * PI).sqrt() * t.powf(x_shift + 0.5) * (-t).exp() * a
    }
}

/// Riemann zeta `ζ(s)` for real s ≠ 1.
pub fn zeta(s: f64) -> NumericResult<f64> {
    if (s - 1.0).abs() < 1.0e-12 {
        return Ok(f64::INFINITY);
    }
    if s == 0.0 {
        return Ok(-0.5);
    }
    if s > 1.0 {
        // Euler-Maclaurin: Σ_{k=1}^N 1/k^s + N^{1-s}/(s-1) - 1/(2 N^s)
        //                  + Σ_{m=1}^M B_{2m}/(2m)! · (s)_{2m-1} / N^{s+2m-1}
        let n = 50_usize;
        let mut s_val: f64 = (1..=n).map(|k| (k as f64).powf(-s)).sum();
        s_val += (n as f64).powf(1.0 - s) / (s - 1.0);
        s_val -= 0.5 * (n as f64).powf(-s);
        // Bernoulli corrections — first 7 terms
        let bernoulli = [
            1.0 / 6.0,
            -1.0 / 30.0,
            1.0 / 42.0,
            -1.0 / 30.0,
            5.0 / 66.0,
            -691.0 / 2730.0,
            7.0 / 6.0,
        ];
        let mut pochhammer = s;
        let mut prev_term: f64 = f64::INFINITY;
        for (m, &b) in bernoulli.iter().enumerate() {
            let two_m = 2 * (m + 1);
            let fact = (1..=two_m).fold(1.0_f64, |acc, v| acc * v as f64);
            let term = b * pochhammer / (fact * (n as f64).powf(s + two_m as f64 - 1.0));
            // accumulate while term-by-term magnitude decreases (asymptotic expansion).
            if term.abs() > prev_term.abs() {
                break;
            }
            s_val += term;
            prev_term = term;
            pochhammer *= (s + two_m as f64) * (s + two_m as f64 + 1.0);
        }
        Ok(s_val)
    } else {
        // functional equation: ζ(s) = 2^s π^{s-1} sin(πs/2) Γ(1-s) ζ(1-s)
        let g = gamma_lanczos(1.0 - s);
        let z = zeta(1.0 - s)?;
        Ok(2.0_f64.powf(s) * PI.powf(s - 1.0) * (PI * s / 2.0).sin() * g * z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeta_two_is_pi_sq_six() {
        let r = zeta(2.0).expect("ok");
        assert!((r - PI * PI / 6.0).abs() < 1.0e-8);
    }

    #[test]
    fn zeta_four_is_pi_pow_four_ninety() {
        let r = zeta(4.0).expect("ok");
        assert!((r - PI.powi(4) / 90.0).abs() < 1.0e-8);
    }

    #[test]
    fn zeta_three_apery() {
        // ζ(3) ≈ 1.2020569031595942 (Apéry's constant)
        let r = zeta(3.0).expect("ok");
        assert!((r - 1.202_056_903_159_594_2).abs() < 1.0e-7);
    }

    #[test]
    fn zeta_zero_is_neg_half() {
        let r = zeta(0.0).expect("ok");
        assert!((r + 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn zeta_neg_one_is_neg_one_twelfth() {
        // ζ(-1) = -1/12
        let r = zeta(-1.0).expect("ok");
        assert!((r + 1.0 / 12.0).abs() < 1.0e-6);
    }
}
