//! Complete elliptic integrals K(m) and E(m) via the arithmetic-geometric mean (AGM).
//!
//! For modulus `m` (`= k²`) with `0 ≤ m < 1`:
//! - K(m) = π / (2 · AGM(1, √(1 - m)))
//! - E(m) = K(m) · (1 - Σ 2^{i-1} c_i²)
//!
//! where `(a_i, b_i, c_i)` is the AGM sequence with `c_i = a_i - b_i` after substitution.

use crate::error::{NumericError, NumericResult};

/// Complete elliptic integral of the first kind `K(m)`.
pub fn elliptic_k(m: f64) -> NumericResult<f64> {
    if !(0.0..1.0).contains(&m) {
        if m == 1.0 {
            return Err(NumericError::OutOfDomain {
                value: m,
                function: "elliptic_k (K(1) diverges)".into(),
            });
        }
        return Err(NumericError::OutOfDomain {
            value: m,
            function: "elliptic_k (require 0 ≤ m < 1)".into(),
        });
    }
    let mut a = 1.0_f64;
    let mut b = (1.0 - m).sqrt();
    for _ in 0..50 {
        let an = 0.5 * (a + b);
        let bn = (a * b).sqrt();
        let stop = (a - b).abs() < 1.0e-15 * a.abs();
        a = an;
        b = bn;
        if stop {
            break;
        }
    }
    let _ = b;
    Ok(std::f64::consts::FRAC_PI_2 / a)
}

/// Complete elliptic integral of the second kind `E(m)`.
pub fn elliptic_e(m: f64) -> NumericResult<f64> {
    if !(0.0..=1.0).contains(&m) {
        return Err(NumericError::OutOfDomain {
            value: m,
            function: "elliptic_e (require 0 ≤ m ≤ 1)".into(),
        });
    }
    if m == 1.0 {
        return Ok(1.0);
    }
    let mut a = 1.0_f64;
    let mut b = (1.0 - m).sqrt();
    let mut c_sq_sum = m;
    let mut pow2 = 1.0_f64;
    for _ in 0..50 {
        let an = 0.5 * (a + b);
        let bn = (a * b).sqrt();
        let c = 0.5 * (a - b);
        pow2 *= 2.0;
        c_sq_sum += pow2 * c * c;
        let stop = c.abs() < 1.0e-15 * a.abs();
        a = an;
        b = bn;
        if stop {
            break;
        }
    }
    let _ = b;
    let k = std::f64::consts::FRAC_PI_2 / a;
    Ok(k * (1.0 - 0.5 * c_sq_sum))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn k_at_zero() {
        let r = elliptic_k(0.0).expect("ok");
        assert!((r - PI / 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn e_at_zero() {
        let r = elliptic_e(0.0).expect("ok");
        assert!((r - PI / 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn e_at_one() {
        let r = elliptic_e(1.0).expect("ok");
        assert!((r - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn k_half_approx() {
        // K(0.5) ≈ 1.8540746773013719
        let r = elliptic_k(0.5).expect("ok");
        assert!((r - 1.854_074_677_301_371_9).abs() < 1.0e-10);
    }

    #[test]
    fn e_half_approx() {
        // E(0.5) ≈ 1.3506438810476755
        let r = elliptic_e(0.5).expect("ok");
        assert!((r - 1.350_643_881_047_675_5).abs() < 1.0e-10);
    }
}
