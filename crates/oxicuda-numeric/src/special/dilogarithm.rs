//! Dilogarithm `Li₂(z)` for real `z ≤ 1`.
//!
//! `Li₂(z) = Σ_{k≥1} z^k / k²` for `|z| ≤ 1`. For other regions, use the reflection identities:
//! - `Li₂(z) + Li₂(1−z) = π²/6 − ln(z) ln(1−z)`
//! - `Li₂(z) + Li₂(z/(z−1)) = −½ ln²(1−z)`

use crate::error::{NumericError, NumericResult};

const PI: f64 = std::f64::consts::PI;
const PI_SQ_OVER_6: f64 = PI * PI / 6.0;

fn series(z: f64) -> f64 {
    let mut term = z;
    let mut sum = 0.0_f64;
    for k in 1..200 {
        let t = term / (k as f64 * k as f64);
        sum += t;
        term *= z;
        if t.abs() < 1.0e-18 * sum.abs().max(1.0) {
            break;
        }
    }
    sum
}

/// Dilogarithm `Li₂(z)` for real argument.
pub fn dilogarithm(z: f64) -> NumericResult<f64> {
    if z > 1.0 {
        return Err(NumericError::OutOfDomain {
            value: z,
            function: "dilogarithm (real z ≤ 1)".into(),
        });
    }
    if z == 0.0 {
        return Ok(0.0);
    }
    if z == 1.0 {
        return Ok(PI_SQ_OVER_6);
    }
    if (z + 1.0).abs() < 1.0e-15 {
        return Ok(-PI * PI / 12.0);
    }
    if z >= 0.5 {
        // use Li₂(z) + Li₂(1 - z) = π²/6 - ln(z) ln(1-z)
        let one_minus = 1.0 - z;
        return Ok(PI_SQ_OVER_6 - z.ln() * one_minus.ln() - series(one_minus));
    }
    if z >= -0.5 {
        return Ok(series(z));
    }
    if z >= -1.0 {
        // use squared identity: 2 Li₂(z) + 2 Li₂(-z) = Li₂(z²); thus
        // Li₂(z) = ½ Li₂(z²) - Li₂(-z) for -1 ≤ z < 0.
        let z_sq = z * z;
        let pos = if (-z) < 0.5 {
            series(-z)
        } else {
            PI_SQ_OVER_6 - (-z).ln() * (1.0 + z).ln() - series(1.0 + z)
        };
        let sq = series(z_sq);
        return Ok(0.5 * sq - pos);
    }
    // z < -1: use Li₂(z) = -Li₂(1/z) - π²/6 - ½ ln²(-z)
    let inv_z = 1.0 / z;
    let li_inv = if inv_z >= -0.5 {
        series(inv_z)
    } else {
        // recursive: but |1/z| < 1
        series(inv_z)
    };
    Ok(-li_inv - PI_SQ_OVER_6 - 0.5 * (-z).ln().powi(2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn li2_zero() {
        let r = dilogarithm(0.0).expect("ok");
        assert!(r.abs() < 1.0e-14);
    }

    #[test]
    fn li2_one() {
        let r = dilogarithm(1.0).expect("ok");
        assert!((r - PI_SQ_OVER_6).abs() < 1.0e-12);
    }

    #[test]
    fn li2_neg_one() {
        // Li₂(-1) = -π²/12
        let r = dilogarithm(-1.0).expect("ok");
        assert!((r + PI * PI / 12.0).abs() < 1.0e-10);
    }

    #[test]
    fn li2_half() {
        // Li₂(1/2) = π²/12 - ln²(2)/2
        let r = dilogarithm(0.5).expect("ok");
        let expected = PI * PI / 12.0 - 0.5 * (2.0_f64).ln().powi(2);
        assert!((r - expected).abs() < 1.0e-10);
    }
}
