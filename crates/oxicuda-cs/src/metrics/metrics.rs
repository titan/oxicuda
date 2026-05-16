//! Sparsity, recovery error, support recovery rate, MSE, and PSNR metrics.

use crate::error::{CsError, CsResult};

/// Count entries `|x_i| > epsilon`.
pub fn sparsity(x: &[f64], epsilon: f64) -> usize {
    x.iter().filter(|&&v| v.abs() > epsilon).count()
}

/// Recovery error `||x̂ - x||₂ / ||x||₂` (relative).
///
/// Returns `||x̂ - x||₂` if `||x||₂ < 1e-300`.
pub fn recovery_error(x_hat: &[f64], x: &[f64]) -> CsResult<f64> {
    if x_hat.len() != x.len() {
        return Err(CsError::DimensionMismatch {
            a: x_hat.len(),
            b: x.len(),
        });
    }
    let mut num_sq = 0.0_f64;
    let mut den_sq = 0.0_f64;
    for (xi_hat, xi) in x_hat.iter().zip(x.iter()) {
        let d = xi_hat - xi;
        num_sq += d * d;
        den_sq += xi * xi;
    }
    let den = den_sq.sqrt();
    if den < 1.0e-300 {
        Ok(num_sq.sqrt())
    } else {
        Ok(num_sq.sqrt() / den)
    }
}

/// Support recovery rate: fraction of true support indices present in estimated support.
///
/// Inputs are index sets (already sorted or unsorted). Returns `|S_true ∩ S_hat| / |S_true|`.
pub fn support_recovery_rate(s_true: &[usize], s_hat: &[usize]) -> CsResult<f64> {
    if s_true.is_empty() {
        return Ok(1.0);
    }
    let mut hat_set: Vec<usize> = s_hat.to_vec();
    hat_set.sort();
    hat_set.dedup();
    let mut hit = 0usize;
    for &t in s_true {
        if hat_set.binary_search(&t).is_ok() {
            hit += 1;
        }
    }
    Ok(hit as f64 / s_true.len() as f64)
}

/// Mean squared error `(1/n) ||x̂ - x||²`.
pub fn mean_squared_error(x_hat: &[f64], x: &[f64]) -> CsResult<f64> {
    if x_hat.len() != x.len() {
        return Err(CsError::DimensionMismatch {
            a: x_hat.len(),
            b: x.len(),
        });
    }
    if x.is_empty() {
        return Err(CsError::EmptyInput);
    }
    let mut s = 0.0_f64;
    for (a, b) in x_hat.iter().zip(x.iter()) {
        let d = a - b;
        s += d * d;
    }
    Ok(s / x.len() as f64)
}

/// Normalised MSE `||x̂ - x||² / ||x||²`.
pub fn normalized_mse(x_hat: &[f64], x: &[f64]) -> CsResult<f64> {
    if x_hat.len() != x.len() {
        return Err(CsError::DimensionMismatch {
            a: x_hat.len(),
            b: x.len(),
        });
    }
    if x.is_empty() {
        return Err(CsError::EmptyInput);
    }
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (a, b) in x_hat.iter().zip(x.iter()) {
        let d = a - b;
        num += d * d;
        den += b * b;
    }
    if den < 1.0e-300 {
        Ok(num)
    } else {
        Ok(num / den)
    }
}

/// PSNR `20 log₁₀(peak / sqrt(MSE))` in dB. `peak` defaults to 1.0 for normalised signals.
pub fn psnr(x_hat: &[f64], x: &[f64], peak: f64) -> CsResult<f64> {
    let mse = mean_squared_error(x_hat, x)?;
    if mse < 1.0e-300 {
        return Ok(f64::INFINITY);
    }
    if peak <= 0.0 {
        return Err(CsError::InvalidParameter(format!(
            "PSNR peak must be > 0; got {peak}"
        )));
    }
    Ok(20.0 * (peak / mse.sqrt()).log10())
}

/// Signal-to-noise ratio in dB: `10 log₁₀(||x||² / ||x̂ - x||²)`.
pub fn snr(x_hat: &[f64], x: &[f64]) -> CsResult<f64> {
    if x_hat.len() != x.len() {
        return Err(CsError::DimensionMismatch {
            a: x_hat.len(),
            b: x.len(),
        });
    }
    let mut sig = 0.0_f64;
    let mut err = 0.0_f64;
    for (a, b) in x_hat.iter().zip(x.iter()) {
        sig += b * b;
        let d = a - b;
        err += d * d;
    }
    if err < 1.0e-300 {
        return Ok(f64::INFINITY);
    }
    if sig < 1.0e-300 {
        return Ok(f64::NEG_INFINITY);
    }
    Ok(10.0 * (sig / err).log10())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparsity_basic() {
        let x = [1.0, 0.0, -2.0, 0.0001, 3.0];
        assert_eq!(sparsity(&x, 1.0e-3), 3);
    }

    #[test]
    fn recovery_error_exact() {
        let x = [1.0, 2.0];
        let x_hat = [1.0, 2.0];
        assert!(recovery_error(&x_hat, &x).expect("ok") < 1.0e-12);
    }

    #[test]
    fn recovery_error_relative() {
        let x = [3.0, 4.0]; // norm 5
        let x_hat = [3.0, 5.0]; // diff [0, 1] norm 1
        let e = recovery_error(&x_hat, &x).expect("ok");
        assert!((e - 0.2).abs() < 1.0e-12);
    }

    #[test]
    fn support_recovery_partial() {
        let st = vec![1usize, 3, 5];
        let sh = vec![3usize, 5, 7];
        let r = support_recovery_rate(&st, &sh).expect("ok");
        assert!((r - 2.0 / 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn mse_basic() {
        let x = [1.0, 2.0, 3.0];
        let x_hat = [2.0, 2.0, 3.0];
        let m = mean_squared_error(&x_hat, &x).expect("ok");
        assert!((m - 1.0 / 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn psnr_exact_inf() {
        let x = [1.0, 2.0];
        let p = psnr(&x, &x, 1.0).expect("ok");
        assert!(p.is_infinite());
    }
}
