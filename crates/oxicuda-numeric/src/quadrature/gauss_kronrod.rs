//! Gauss-Kronrod G7-K15 quadrature pair on `[-1, 1]`.

use crate::error::{NumericError, NumericResult};

const KRONROD_NODES_15: [f64; 15] = [
    -0.991_455_371_120_812_6,
    -0.949_107_912_342_758_5,
    -0.864_864_423_359_769_1,
    -0.741_531_185_599_394_4,
    -0.586_087_235_467_691_1,
    -0.405_845_151_377_397_2,
    -0.207_784_955_007_898_45,
    0.0,
    0.207_784_955_007_898_45,
    0.405_845_151_377_397_2,
    0.586_087_235_467_691_1,
    0.741_531_185_599_394_4,
    0.864_864_423_359_769_1,
    0.949_107_912_342_758_5,
    0.991_455_371_120_812_6,
];

const KRONROD_WEIGHTS_15: [f64; 15] = [
    0.022_935_322_010_529_22,
    0.063_092_092_629_978_55,
    0.104_790_010_322_250_18,
    0.140_653_259_715_525_92,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_4,
    0.204_432_940_075_298_9,
    0.209_482_141_084_727_82,
    0.204_432_940_075_298_9,
    0.190_350_578_064_785_4,
    0.169_004_726_639_267_9,
    0.140_653_259_715_525_92,
    0.104_790_010_322_250_18,
    0.063_092_092_629_978_55,
    0.022_935_322_010_529_22,
];

const GAUSS_WEIGHTS_7: [f64; 7] = [
    0.129_484_966_168_869_69,
    0.279_705_391_489_276_67,
    0.381_830_050_505_118_94,
    0.417_959_183_673_469_4,
    0.381_830_050_505_118_94,
    0.279_705_391_489_276_67,
    0.129_484_966_168_869_69,
];

/// Apply G7-K15 quadrature on `[a, b]`. Returns `(integral_k15, error_estimate)`.
pub fn gauss_kronrod_g7k15<F>(f: F, a: f64, b: f64) -> NumericResult<(f64, f64)>
where
    F: Fn(f64) -> NumericResult<f64>,
{
    if !a.is_finite() || !b.is_finite() {
        return Err(NumericError::InvalidParameter("non-finite limits".into()));
    }
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let mut sum_k = 0.0_f64;
    let mut sum_g = 0.0_f64;
    for (i, (x, w)) in KRONROD_NODES_15
        .iter()
        .zip(KRONROD_WEIGHTS_15.iter())
        .enumerate()
    {
        let fx = f(mid + half * x)?;
        sum_k += w * fx;
        if i & 1 == 1 {
            let gi = i / 2;
            sum_g += GAUSS_WEIGHTS_7[gi] * fx;
        }
    }
    let k_val = half * sum_k;
    let g_val = half * sum_g;
    let raw = (k_val - g_val).abs();
    let err = (200.0 * raw).powf(1.5);
    Ok((k_val, err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn gk_arctan() {
        let f = |x: f64| -> NumericResult<f64> { Ok(1.0 / (1.0 + x * x)) };
        let (r, _err) = gauss_kronrod_g7k15(f, 0.0, 1.0).expect("ok");
        assert!((r - PI / 4.0).abs() < 1.0e-9);
    }

    #[test]
    fn gk_polynomial_x10() {
        let f = |x: f64| -> NumericResult<f64> { Ok(x.powi(10)) };
        let (r, _e) = gauss_kronrod_g7k15(f, -1.0, 1.0).expect("ok");
        assert!((r - 2.0 / 11.0).abs() < 1.0e-10);
    }

    #[test]
    fn gk_error_small_for_smooth() {
        let f = |x: f64| -> NumericResult<f64> { Ok(x.exp()) };
        let (_v, e) = gauss_kronrod_g7k15(f, 0.0, 1.0).expect("ok");
        assert!(e < 1.0e-3);
    }
}
