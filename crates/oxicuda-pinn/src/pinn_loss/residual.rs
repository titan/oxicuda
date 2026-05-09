//! PDE residual loss computation.

use crate::error::{PinnError, PinnResult};

/// Mean-squared PDE residual loss: `(1/n) Σ r_i²`.
pub fn pde_residual_loss(residuals: &[f32]) -> PinnResult<f32> {
    if residuals.is_empty() {
        return Err(PinnError::EmptyCollocationSet);
    }
    let mse: f32 = residuals.iter().map(|&r| r * r).sum::<f32>() / residuals.len() as f32;
    if !mse.is_finite() {
        return Err(PinnError::NanEncountered {
            location: "pde_residual_loss",
        });
    }
    Ok(mse)
}

/// Compute residuals at collocation points.
///
/// `points` is a flat `[n × d]` array. `residual_fn` takes a `d`-dim point.
pub fn compute_residuals<F>(
    points: &[f32],
    n: usize,
    d: usize,
    residual_fn: F,
) -> PinnResult<Vec<f32>>
where
    F: Fn(&[f32]) -> f32,
{
    if points.is_empty() || n == 0 {
        return Err(PinnError::EmptyCollocationSet);
    }
    if points.len() != n * d {
        return Err(PinnError::DimensionMismatch {
            expected: n * d,
            got: points.len(),
        });
    }

    let residuals: Vec<f32> = (0..n)
        .map(|i| residual_fn(&points[i * d..(i + 1) * d]))
        .collect();

    Ok(residuals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_residuals_zero_loss() {
        let r = vec![0.0_f32; 10];
        let loss = pde_residual_loss(&r).unwrap();
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn constant_residuals_mse() {
        let r = vec![2.0_f32; 4]; // MSE = 4.0
        let loss = pde_residual_loss(&r).unwrap();
        assert!(
            (loss - 4.0).abs() < 1e-6,
            "MSE of [2,2,2,2] = 4, got {loss}"
        );
    }

    #[test]
    fn mixed_residuals_mse() {
        let r = vec![1.0_f32, -1.0, 2.0, -2.0]; // MSE = (1+1+4+4)/4 = 2.5
        let loss = pde_residual_loss(&r).unwrap();
        assert!((loss - 2.5).abs() < 1e-6);
    }

    #[test]
    fn single_residual() {
        let r = vec![3.0_f32];
        let loss = pde_residual_loss(&r).unwrap();
        assert!((loss - 9.0).abs() < 1e-6);
    }

    #[test]
    fn empty_residuals_error() {
        let result = pde_residual_loss(&[]);
        assert!(matches!(result, Err(PinnError::EmptyCollocationSet)));
    }

    #[test]
    fn compute_residuals_correct_shape() {
        let pts = vec![0.0_f32, 1.0, 2.0, 3.0]; // n=2, d=2
        let res = compute_residuals(&pts, 2, 2, |p| p[0] + p[1]).unwrap();
        assert_eq!(res.len(), 2);
        assert!((res[0] - 1.0).abs() < 1e-6);
        assert!((res[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn compute_residuals_zero_fn() {
        let pts = vec![1.0_f32; 10];
        let res = compute_residuals(&pts, 5, 2, |_| 0.0).unwrap();
        assert!(res.iter().all(|&r| r == 0.0));
    }

    #[test]
    fn compute_residuals_empty_error() {
        let result = compute_residuals(&[], 0, 2, |_| 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn compute_residuals_dim_mismatch_error() {
        let pts = vec![1.0_f32; 5]; // n=3, d=2 → expects 6
        let result = compute_residuals(&pts, 3, 2, |_| 0.0);
        assert!(result.is_err());
    }
}
