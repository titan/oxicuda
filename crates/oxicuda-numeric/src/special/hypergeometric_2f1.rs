//! Gauss hypergeometric `₂F₁(a, b; c; z)` for real arguments.
//!
//! Direct Taylor series converges for `|z| < 1`. For `|z| ≥ 1` (handled here only when
//! `z < 1` real), we use the linear transformation `z → 1 - z` or `z → z/(z - 1)` to
//! shift into the convergent region.

use crate::error::{NumericError, NumericResult};

fn series(a: f64, b: f64, c: f64, z: f64, max_iter: usize) -> NumericResult<f64> {
    if c <= 0.0 && c == c.floor() {
        return Err(NumericError::OutOfDomain {
            value: c,
            function: "hypergeometric_2f1 (c is non-positive integer)".into(),
        });
    }
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    for k in 0..max_iter {
        let kf = k as f64;
        term *= (a + kf) * (b + kf) / ((c + kf) * (kf + 1.0)) * z;
        sum += term;
        if term.abs() < 1.0e-16 * sum.abs() {
            return Ok(sum);
        }
    }
    Err(NumericError::NotConverged {
        iter: max_iter,
        residual: term.abs(),
    })
}

/// ₂F₁(a, b; c; z).
pub fn hypergeometric_2f1(a: f64, b: f64, c: f64, z: f64) -> NumericResult<f64> {
    if z.is_nan() || a.is_nan() || b.is_nan() || c.is_nan() {
        return Err(NumericError::InvalidParameter("NaN argument".into()));
    }
    if z.abs() < 0.5 {
        return series(a, b, c, z, 500);
    }
    if z > -0.5 && z < 0.95 {
        return series(a, b, c, z, 1000);
    }
    if (0.95..1.0).contains(&z) {
        // Use connection: ₂F₁(a,b;c;z) = Γ(c) Γ(c-a-b)/(Γ(c-a) Γ(c-b)) ₂F₁(a,b;a+b-c+1;1-z)
        //                              + Γ(c) Γ(a+b-c)/(Γ(a) Γ(b)) (1-z)^{c-a-b} ₂F₁(c-a,c-b;c-a-b+1;1-z)
        // valid when c - a - b is non-integer.
        let s1 = series(a, b, a + b - c + 1.0, 1.0 - z, 1000)?;
        let s2 = series(c - a, c - b, c - a - b + 1.0, 1.0 - z, 1000)?;
        let cab = c - a - b;
        let g_c = gamma_lanczos(c);
        let g_cab = gamma_lanczos(cab);
        let g_ca = gamma_lanczos(c - a);
        let g_cb = gamma_lanczos(c - b);
        let g_a = gamma_lanczos(a);
        let g_b = gamma_lanczos(b);
        Ok(g_c * g_cab / (g_ca * g_cb) * s1
            + g_c * gamma_lanczos(-cab) / (g_a * g_b) * (1.0 - z).powf(cab) * s2)
    } else if z <= -0.5 {
        // Linear transformation: ₂F₁(a,b;c;z) = (1-z)^{-a} ₂F₁(a, c-b; c; z/(z-1))
        let w = z / (z - 1.0);
        let s = series(a, c - b, c, w, 1000)?;
        Ok((1.0 - z).powf(-a) * s)
    } else {
        Err(NumericError::OutOfDomain {
            value: z,
            function: "hypergeometric_2f1 (z must be in (-∞, 1))".into(),
        })
    }
}

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
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma_lanczos(1.0 - x))
    } else {
        let x_shift = x - 1.0;
        let mut a = 0.999_999_999_999_809_93_f64;
        for (i, &pi) in p.iter().enumerate() {
            a += pi / (x_shift + (i as f64) + 1.0);
        }
        let t = x_shift + p.len() as f64 - 0.5;
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(x_shift + 0.5) * (-t).exp() * a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f21_zero_is_one() {
        let r = hypergeometric_2f1(1.5, 2.5, 3.5, 0.0).expect("ok");
        assert!((r - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn f21_geometric() {
        // ₂F₁(1, 1; 1; z) = 1 / (1 - z)
        let z = 0.3_f64;
        let r = hypergeometric_2f1(1.0, 1.0, 1.0, z).expect("ok");
        assert!((r - 1.0 / (1.0 - z)).abs() < 1.0e-10);
    }

    #[test]
    fn f21_log_identity() {
        // ₂F₁(1, 1; 2; z) = -ln(1 - z)/z
        let z = 0.4_f64;
        let r = hypergeometric_2f1(1.0, 1.0, 2.0, z).expect("ok");
        assert!((r + (1.0 - z).ln() / z).abs() < 1.0e-10);
    }
}
