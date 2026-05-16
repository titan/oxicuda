//! Lambert W function via Halley iteration.

use crate::error::{NumericError, NumericResult};

const E_INV: f64 = 0.367_879_441_171_442_3;

/// Principal branch `W_0(x)` for `x ≥ -1/e`.
pub fn lambert_w0(x: f64) -> NumericResult<f64> {
    if x < -E_INV {
        return Err(NumericError::OutOfDomain {
            value: x,
            function: "lambert_w0 (x < -1/e)".into(),
        });
    }
    if x == 0.0 {
        return Ok(0.0);
    }
    let mut w = if x < 1.0 {
        let p = (2.0 * (1.0 + std::f64::consts::E * x)).sqrt();
        -1.0 + p - p * p / 3.0 + 11.0 / 72.0 * p.powi(3)
    } else {
        x.ln() - x.ln().abs().max(1.0).ln()
    };
    for _ in 0..50 {
        let ew = w.exp();
        let f = w * ew - x;
        let denom = ew * (w + 1.0) - (w + 2.0) * f / (2.0 * (w + 1.0));
        if denom.abs() < 1.0e-300 {
            break;
        }
        let step = f / denom;
        w -= step;
        if step.abs() < 1.0e-14 * w.abs().max(1.0) {
            return Ok(w);
        }
    }
    Ok(w)
}

/// Branch `W_{-1}(x)` for `-1/e ≤ x < 0`.
pub fn lambert_wm1(x: f64) -> NumericResult<f64> {
    if !(-E_INV..0.0).contains(&x) {
        return Err(NumericError::OutOfDomain {
            value: x,
            function: "lambert_wm1 (require -1/e ≤ x < 0)".into(),
        });
    }
    let mut w = if x > -1.0e-3 {
        (-x).ln() - (-(-x).ln()).ln()
    } else {
        let l1 = (-x).ln();
        let l2 = (-l1).ln();
        l1 - l2 + l2 / l1
    };
    for _ in 0..50 {
        let ew = w.exp();
        let f = w * ew - x;
        let denom = ew * (w + 1.0) - (w + 2.0) * f / (2.0 * (w + 1.0));
        if denom.abs() < 1.0e-300 {
            break;
        }
        let step = f / denom;
        w -= step;
        if step.abs() < 1.0e-14 * w.abs().max(1.0) {
            return Ok(w);
        }
    }
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w0_of_e() {
        let r = lambert_w0(std::f64::consts::E).expect("ok");
        assert!((r - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn w0_of_zero() {
        let r = lambert_w0(0.0).expect("ok");
        assert!(r.abs() < 1.0e-12);
    }

    #[test]
    fn w0_of_neg_inv_e() {
        let r = lambert_w0(-E_INV).expect("ok");
        assert!((r + 1.0).abs() < 1.0e-3);
    }

    #[test]
    fn wm1_negative() {
        let r = lambert_wm1(-0.1).expect("ok");
        assert!(r < -1.0);
    }

    #[test]
    fn w0_satisfies_defining_eq() {
        let x = 2.0_f64;
        let w = lambert_w0(x).expect("ok");
        assert!((w * w.exp() - x).abs() < 1.0e-10);
    }
}
