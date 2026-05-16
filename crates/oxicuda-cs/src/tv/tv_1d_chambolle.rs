//! 1D TV denoising via Chambolle's primal-dual (Chambolle 2004) on the dual problem.
//!
//! Solve `min_x  ½ ||x − y||² + λ TV(x)` where `TV(x) = Σ_j |x_{j+1} − x_j|`.
//! Dual formulation: `min_p ½ ||y − D^T p||² s.t. ||p||_∞ ≤ λ`, then `x = y − D^T p`.

use crate::error::{CsError, CsResult};

/// 1D TV denoising of signal `y` with regularisation parameter `lambda`.
///
/// Returns the denoised signal of the same length. Uses projected gradient on dual.
pub fn tv_1d_chambolle(y: &[f64], lambda: f64, max_iter: usize, tol: f64) -> CsResult<Vec<f64>> {
    if y.is_empty() {
        return Err(CsError::EmptyInput);
    }
    if lambda < 0.0 {
        return Err(CsError::InvalidParameter("lambda must be ≥ 0".into()));
    }
    if lambda == 0.0 {
        return Ok(y.to_vec());
    }
    let n = y.len();
    // Dual variable p of length n-1.
    let mut p = vec![0.0_f64; n.saturating_sub(1)];
    // Step size: tau ≤ 1/8 for stability when ||D||² = 4 (but use 1/4 for 1D).
    let tau = 0.249_f64;
    let mut x = y.to_vec();
    for _ in 0..max_iter {
        // x = y - D^T p
        x.copy_from_slice(y);
        for j in 0..p.len() {
            x[j] += p[j];
            x[j + 1] -= p[j];
        }
        // p_new = clip(p + tau * D x, -λ, λ); D x = x_{j+1} - x_j.
        let mut delta = 0.0_f64;
        for j in 0..p.len() {
            let dx = x[j + 1] - x[j];
            let p_new = (p[j] + tau * dx).clamp(-lambda, lambda);
            let dp = (p_new - p[j]).abs();
            if dp > delta {
                delta = dp;
            }
            p[j] = p_new;
        }
        if delta < tol {
            break;
        }
    }
    // Final reconstruction.
    let mut x_out = y.to_vec();
    for j in 0..p.len() {
        x_out[j] += p[j];
        x_out[j + 1] -= p[j];
    }
    Ok(x_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tv_1d_no_noise_passes_constant() {
        let y = vec![5.0, 5.0, 5.0, 5.0];
        let x = tv_1d_chambolle(&y, 1.0, 100, 1.0e-9).expect("ok");
        for &xi in &x {
            assert!((xi - 5.0).abs() < 1.0e-6);
        }
    }

    #[test]
    fn tv_1d_step_edge() {
        // Two flat regions of length 4 each; small noise.
        let y = vec![1.0, 1.05, 0.95, 1.02, 5.0, 4.97, 5.03, 5.0];
        let x = tv_1d_chambolle(&y, 0.5, 500, 1.0e-9).expect("ok");
        // First half values should be close together; same for second half.
        let mean_lo = x[..4].iter().sum::<f64>() / 4.0;
        let mean_hi = x[4..].iter().sum::<f64>() / 4.0;
        for &v in &x[..4] {
            assert!((v - mean_lo).abs() < 0.2);
        }
        for &v in &x[4..] {
            assert!((v - mean_hi).abs() < 0.2);
        }
        assert!(mean_hi > mean_lo + 1.0);
    }
}
