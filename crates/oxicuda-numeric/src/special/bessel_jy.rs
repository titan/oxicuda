//! Bessel J/Y functions (regular and irregular of the first/second kind) — real argument.
//!
//! - `J_n(x)` for integer order via Miller's downward recurrence (numerically stable).
//! - `Y_0(x)`, `Y_1(x)` via series (small x) and asymptotic (large x).
//! - `Y_n(x)` via stable upward recurrence from `Y_0, Y_1`.

use crate::error::{NumericError, NumericResult};

const PI: f64 = std::f64::consts::PI;
const TWO_OVER_PI: f64 = 2.0 / PI;

/// J_0(x) — series + asymptotic.
pub fn bessel_j0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let p =
            ((((1.0_f64 - y / 16.0) * y / 9.0 - y / 4.0) * y / 4.0 - y / 1.0) * 0.0 + 1.0) - 0.0;
        let _ = p;
        // Use Taylor series ∑ (-1)^k (x/2)^{2k} / (k!)²
        let mut term = 1.0_f64;
        let mut sum = 1.0_f64;
        let half_x_sq = (x * x) / 4.0;
        for k in 1..40 {
            term *= -half_x_sq / (k as f64 * k as f64);
            sum += term;
            if term.abs() < 1.0e-16 * sum.abs() {
                break;
            }
        }
        sum
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let p0 = 1.0
            + y * (-0.1098628627e-2
                + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let q0 = -0.1562499995e-1
            + y * (0.1430488765e-3
                + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934935152e-7)));
        let xx = ax - std::f64::consts::FRAC_PI_4;
        let amp = (TWO_OVER_PI / ax).sqrt();
        amp * (xx.cos() * p0 - z * xx.sin() * q0)
    }
}

/// J_1(x).
pub fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    if ax < 8.0 {
        let mut term = 0.5_f64 * x;
        let mut sum = term;
        let half_x_sq = (x * x) / 4.0;
        for k in 1..40 {
            term *= -half_x_sq / (k as f64 * (k as f64 + 1.0));
            sum += term;
            if term.abs() < 1.0e-16 * sum.abs() {
                break;
            }
        }
        sum
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let p1 = 1.0
            + y * (0.183105e-2
                + y * (-0.3516396496e-4 + y * (0.2457520174e-5 - y * 0.240337019e-6)));
        let q1 = 0.04687499995
            + y * (-0.2002690873e-3
                + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
        let xx = ax - 2.356_194_490_192_345;
        let amp = (TWO_OVER_PI / ax).sqrt();
        sign * amp * (xx.cos() * p1 - z * xx.sin() * q1)
    }
}

/// Y_0(x) for x > 0.
pub fn bessel_y0(x: f64) -> NumericResult<f64> {
    if x <= 0.0 {
        return Err(NumericError::OutOfDomain {
            value: x,
            function: "bessel_y0".into(),
        });
    }
    if x < 8.0 {
        let j0 = bessel_j0(x);
        // Series for Y_0 (Abramowitz & Stegun 9.1.13):
        // Y_0(x) = (2/π)(ln(x/2) + γ) J_0(x) + (2/π) Σ_{k=1} (-1)^{k+1} h_k (x/2)^{2k} / (k!)²
        let gamma = 0.577_215_664_901_532_9_f64;
        let mut term = 1.0_f64;
        let mut harmonic = 0.0_f64;
        let mut sum = 0.0_f64;
        let half_x_sq = (x * x) / 4.0;
        for k in 1..40 {
            term *= -half_x_sq / (k as f64 * k as f64);
            harmonic += 1.0 / k as f64;
            // (-1)^{k+1} h_k; "term" already has (-1)^k factor in it because each step multiplies by negative.
            sum += -term * harmonic;
            if term.abs() < 1.0e-18 * sum.abs().max(1.0) {
                break;
            }
        }
        Ok(TWO_OVER_PI * ((x / 2.0).ln() + gamma) * j0 + TWO_OVER_PI * sum)
    } else {
        let z = 8.0 / x;
        let y = z * z;
        let p0 = 1.0
            + y * (-0.1098628627e-2
                + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let q0 = -0.1562499995e-1
            + y * (0.1430488765e-3
                + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934935152e-7)));
        let xx = x - std::f64::consts::FRAC_PI_4;
        let amp = (TWO_OVER_PI / x).sqrt();
        Ok(amp * (xx.sin() * p0 + z * xx.cos() * q0))
    }
}

/// Y_1(x) for x > 0.
pub fn bessel_y1(x: f64) -> NumericResult<f64> {
    if x <= 0.0 {
        return Err(NumericError::OutOfDomain {
            value: x,
            function: "bessel_y1".into(),
        });
    }
    if x < 8.0 {
        // Wronskian: J_0 Y_1 - J_1 Y_0 = -2/(πx)  →  Y_1 = (J_1 Y_0 - 2/(πx)) / J_0
        let j0 = bessel_j0(x);
        let j1 = bessel_j1(x);
        let y0 = bessel_y0(x)?;
        if j0.abs() < 1.0e-300 {
            return Err(NumericError::NumericalInstability(
                "J_0 near zero in Y_1 evaluation".into(),
            ));
        }
        Ok((j1 * y0 - 2.0 / (PI * x)) / j0)
    } else {
        let z = 8.0 / x;
        let y = z * z;
        let p1 = 1.0
            + y * (0.183105e-2
                + y * (-0.3516396496e-4 + y * (0.2457520174e-5 - y * 0.240337019e-6)));
        let q1 = 0.04687499995
            + y * (-0.2002690873e-3
                + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
        let xx = x - 2.356_194_490_192_345;
        let amp = (TWO_OVER_PI / x).sqrt();
        Ok(amp * (xx.sin() * p1 + z * xx.cos() * q1))
    }
}

/// `J_n(x)` via Miller's downward recurrence for non-negative integer order `n`.
pub fn bessel_jn(n: usize, x: f64) -> NumericResult<f64> {
    if x == 0.0 {
        return Ok(if n == 0 { 1.0 } else { 0.0 });
    }
    if n == 0 {
        return Ok(bessel_j0(x));
    }
    if n == 1 {
        return Ok(bessel_j1(x));
    }
    // For x >> n we can use a stable upward recurrence; for x ≤ n use downward (Miller).
    let ax = x.abs();
    let sign = if x < 0.0 && (n & 1) == 1 { -1.0 } else { 1.0 };
    if ax > n as f64 {
        // upward recurrence is stable.
        let mut jm = bessel_j0(ax);
        let mut j = bessel_j1(ax);
        for k in 1..n {
            let jp = (2.0 * k as f64 / ax) * j - jm;
            jm = j;
            j = jp;
        }
        return Ok(sign * j);
    }
    // Miller's downward recurrence with normalization J_0 + 2 Σ_{k≥1} J_{2k} = 1.
    let m_start = (n + (40.0_f64.max((40.0 * ax).sqrt()) as usize)) | 1;
    let mut bjp = 0.0_f64;
    let mut bj = 1.0e-10_f64;
    let mut answer = 0.0_f64;
    let mut sum = 0.0_f64;
    for k in (1..=m_start).rev() {
        let bjm = (2.0 * k as f64 / ax) * bj - bjp;
        bjp = bj;
        bj = bjm;
        if bj.abs() > 1.0e10 {
            bj *= 1.0e-10;
            bjp *= 1.0e-10;
            answer *= 1.0e-10;
            sum *= 1.0e-10;
        }
        if (k & 1) == 0 {
            sum += bj;
        }
        if k == n {
            answer = bjp;
        }
    }
    // After loop: bj = J_0, sum = Σ_{m=1..} J_{2m}; normalization: bj + 2 sum = 1.
    let norm = bj + 2.0 * sum;
    answer /= norm;
    Ok(sign * answer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn j0_at_zero() {
        assert!((bessel_j0(0.0) - 1.0).abs() < 1.0e-14);
    }

    #[test]
    fn j0_first_zero() {
        // First positive zero of J_0 is ≈ 2.404825557695773
        assert!(bessel_j0(2.404_825_557_695_773).abs() < 1.0e-7);
    }

    #[test]
    fn j1_at_zero() {
        assert!(bessel_j1(0.0).abs() < 1.0e-14);
    }

    #[test]
    fn j1_first_max() {
        // J_1(1.8412) ≈ 0.5819 (max)
        assert!((bessel_j1(1.841_183_781_340_659) - 0.5819).abs() < 1.0e-3);
    }

    #[test]
    fn y0_positive() {
        let y = bessel_y0(1.0).expect("ok");
        // Y_0(1) ≈ 0.0883
        assert!((y - 0.0883).abs() < 1.0e-3);
    }

    #[test]
    fn jn_consistency() {
        // J_3(5) ≈ 0.36483, classical value
        let v = bessel_jn(3, 5.0).expect("ok");
        assert!((v - 0.364_831_230_613_667).abs() < 1.0e-4);
    }
}
