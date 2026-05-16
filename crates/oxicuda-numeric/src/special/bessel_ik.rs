//! Modified Bessel functions I/K (real argument).
//!
//! Polynomial approximations from Abramowitz & Stegun §9.8.

use crate::error::{NumericError, NumericResult};

/// I_0(x).
pub fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75).powi(2);
        1.0 + y
            * (3.5156229
                + y * (3.0899424
                    + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
    } else {
        let y = 3.75 / ax;
        let amp = ax.exp() / ax.sqrt();
        amp * (0.39894228
            + y * (0.01328592
                + y * (0.00225319
                    + y * (-0.00157565
                        + y * (0.00916281
                            + y * (-0.02057706
                                + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))))
    }
}

/// I_1(x).
pub fn bessel_i1(x: f64) -> f64 {
    let ax = x.abs();
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    if ax < 3.75 {
        let y = (x / 3.75).powi(2);
        let s = 0.5
            + y * (0.87890594
                + y * (0.51498869
                    + y * (0.15084934 + y * (0.02658733 + y * (0.00301532 + y * 0.00032411)))));
        x * s
    } else {
        let y = 3.75 / ax;
        let amp = ax.exp() / ax.sqrt();
        let s = 0.39894228
            + y * (-0.03988024
                + y * (-0.00362018
                    + y * (0.00163801
                        + y * (-0.01031555
                            + y * (0.02282967
                                + y * (-0.02895312 + y * (0.01787654 - y * 0.00420059)))))));
        sign * amp * s
    }
}

/// K_0(x) for x > 0.
pub fn bessel_k0(x: f64) -> NumericResult<f64> {
    if x <= 0.0 {
        return Err(NumericError::OutOfDomain {
            value: x,
            function: "bessel_k0".into(),
        });
    }
    if x <= 2.0 {
        let y = (x / 2.0).powi(2);
        let s = -0.57721566
            + y * (0.42278420
                + y * (0.23069756
                    + y * (0.03488590 + y * (0.00262698 + y * (0.00010750 + y * 0.00000740)))));
        Ok(-(x / 2.0).ln() * bessel_i0(x) + s)
    } else {
        let y = 2.0 / x;
        let amp = (-x).exp() / x.sqrt();
        let s = 1.25331414
            + y * (-0.07832358
                + y * (0.02189568
                    + y * (-0.01062446 + y * (0.00587872 + y * (-0.00251540 + y * 0.00053208)))));
        Ok(amp * s)
    }
}

/// K_1(x) for x > 0.
pub fn bessel_k1(x: f64) -> NumericResult<f64> {
    if x <= 0.0 {
        return Err(NumericError::OutOfDomain {
            value: x,
            function: "bessel_k1".into(),
        });
    }
    if x <= 2.0 {
        let y = (x / 2.0).powi(2);
        let s = 1.0
            + y * (0.15443144
                + y * (-0.67278579
                    + y * (-0.18156897 + y * (-0.01919402 + y * (-0.00110404 - y * 0.00004686)))));
        Ok((x / 2.0).ln() * bessel_i1(x) + s / x)
    } else {
        let y = 2.0 / x;
        let amp = (-x).exp() / x.sqrt();
        let s = 1.25331414
            + y * (0.23498619
                + y * (-0.03655620
                    + y * (0.01504268 + y * (-0.00780353 + y * (0.00325614 - y * 0.00068245)))));
        Ok(amp * s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i0_at_zero() {
        assert!((bessel_i0(0.0) - 1.0).abs() < 1.0e-14);
    }

    #[test]
    fn i0_at_one() {
        assert!((bessel_i0(1.0) - 1.266_065_877_752_008).abs() < 1.0e-4);
    }

    #[test]
    fn i1_at_zero() {
        assert!(bessel_i1(0.0).abs() < 1.0e-14);
    }

    #[test]
    fn k0_positive() {
        let v = bessel_k0(1.0).expect("ok");
        assert!((v - 0.421_024_438_240_708).abs() < 1.0e-4);
    }

    #[test]
    fn k0_neg_err() {
        let v = bessel_k0(-1.0);
        assert!(matches!(v, Err(NumericError::OutOfDomain { .. })));
    }

    #[test]
    fn k1_positive() {
        let v = bessel_k1(1.0).expect("ok");
        assert!((v - 0.601_907_230_197_235).abs() < 1.0e-4);
    }
}
