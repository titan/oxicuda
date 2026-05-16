//! 1D total-variation proximal operator.
//!
//! Solves `min_x ½ ||x − y||² + λ Σ |x_{i+1} − x_i|`.
//!
//! Implements Condat's O(n) algorithm (Condat 2013, "A direct algorithm for 1-D total variation
//! denoising").

use crate::error::{CvxError, CvxResult};

/// Condat exact 1-D TV denoising in O(n).
///
/// Reference:
/// L. Condat, "A direct algorithm for 1D total variation denoising",
/// IEEE Signal Processing Letters, 2013.
pub fn prox_tv_1d(y: &[f64], lambda: f64) -> CvxResult<Vec<f64>> {
    if y.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "TV prox requires lambda ≥ 0, got {lambda}"
        )));
    }
    let n = y.len();
    let mut x = vec![0.0_f64; n];
    if n == 1 || lambda == 0.0 {
        x.clone_from_slice(y);
        return Ok(x);
    }
    // Condat 1-D TV algorithm with full state.
    // Variables:
    //   k        : current index
    //   k0       : start of current segment under consideration
    //   k_minus  : last index where we hit the negative bound
    //   k_plus   : last index where we hit the positive bound
    //   v_min, v_max : running segment min/max bounds
    //   u_min, u_max : accumulated slack with respect to those bounds
    let mut k = 0usize;
    let mut k0 = 0usize;
    let mut k_minus = 0usize;
    let mut k_plus = 0usize;
    let mut v_min = y[0] - lambda;
    let mut v_max = y[0] + lambda;
    let mut u_min = lambda;
    let mut u_max = -lambda;
    loop {
        if k == n - 1 {
            // Terminate.
            if u_min < 0.0 {
                let val = v_min;
                for slot in x.iter_mut().take(k_minus + 1).skip(k0) {
                    *slot = val;
                }
                k = k_minus + 1;
                k0 = k;
                k_minus = k;
                v_min = y[k];
                u_min = lambda;
                u_max = y[k] + lambda - v_max;
            } else if u_max > 0.0 {
                let val = v_max;
                for slot in x.iter_mut().take(k_plus + 1).skip(k0) {
                    *slot = val;
                }
                k = k_plus + 1;
                k0 = k;
                k_plus = k;
                v_max = y[k];
                u_max = -lambda;
                u_min = y[k] - lambda - v_min;
            } else {
                let val = v_min + u_min / (k - k0 + 1) as f64;
                for slot in x.iter_mut().take(n).skip(k0) {
                    *slot = val;
                }
                return Ok(x);
            }
            continue;
        }
        let yk1 = y[k + 1];
        if yk1 + u_min < v_min - lambda {
            // Negative jump: emit segment at v_min.
            let val = v_min;
            for slot in x.iter_mut().take(k_minus + 1).skip(k0) {
                *slot = val;
            }
            k = k_minus + 1;
            k0 = k;
            k_minus = k;
            k_plus = k;
            v_min = y[k];
            v_max = y[k] + 2.0 * lambda;
            u_min = lambda;
            u_max = -lambda;
        } else if yk1 + u_max > v_max + lambda {
            // Positive jump.
            let val = v_max;
            for slot in x.iter_mut().take(k_plus + 1).skip(k0) {
                *slot = val;
            }
            k = k_plus + 1;
            k0 = k;
            k_minus = k;
            k_plus = k;
            v_max = y[k];
            v_min = y[k] - 2.0 * lambda;
            u_min = lambda;
            u_max = -lambda;
        } else {
            // Continue segment: update slacks.
            k += 1;
            u_min += yk1 - v_min;
            u_max += yk1 - v_max;
            if u_min >= lambda {
                v_min += (u_min - lambda) / (k - k0 + 1) as f64;
                u_min = lambda;
                k_minus = k;
            }
            if u_max <= -lambda {
                v_max += (u_max + lambda) / (k - k0 + 1) as f64;
                u_max = -lambda;
                k_plus = k;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tv_zero_lambda_recovers() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let x = prox_tv_1d(&y, 0.0).expect("ok");
        for (xi, yi) in x.iter().zip(y.iter()) {
            assert!((xi - yi).abs() < 1.0e-12);
        }
    }

    #[test]
    fn tv_constant_signal_unchanged() {
        let y = vec![1.0; 6];
        let x = prox_tv_1d(&y, 0.5).expect("ok");
        for xi in &x {
            assert!((xi - 1.0).abs() < 1.0e-9, "got xi={xi}");
        }
    }

    #[test]
    fn tv_reduces_jumps() {
        let y = vec![0.1, -0.05, 0.05, 0.9, 1.05, 0.95];
        let x = prox_tv_1d(&y, 0.5).expect("ok");
        let mean_left: f64 = x[0..3].iter().sum::<f64>() / 3.0;
        let var_left: f64 = x[0..3].iter().map(|v| (v - mean_left).powi(2)).sum();
        let var_y_left: f64 = y[0..3]
            .iter()
            .map(|v| (v - y[0..3].iter().sum::<f64>() / 3.0).powi(2))
            .sum();
        assert!(var_left <= var_y_left + 1.0e-10);
    }

    #[test]
    fn tv_single_element() {
        let y = vec![3.5];
        let x = prox_tv_1d(&y, 1.0).expect("ok");
        assert_eq!(x, vec![3.5]);
    }
}
