//! Initial condition loss computation.

use crate::error::{PinnError, PinnResult};

/// Initial condition MSE loss: `(1/M)Σ(u_initial_i - u_target_i)²`.
pub fn ic_loss(u_initial: &[f32], u_target: &[f32]) -> PinnResult<f32> {
    if u_initial.is_empty() {
        return Err(PinnError::EmptyCollocationSet);
    }
    if u_initial.len() != u_target.len() {
        return Err(PinnError::DimensionMismatch {
            expected: u_initial.len(),
            got: u_target.len(),
        });
    }

    let mse: f32 = u_initial
        .iter()
        .zip(u_target.iter())
        .map(|(&u, &t)| (u - t).powi(2))
        .sum::<f32>()
        / u_initial.len() as f32;

    if !mse.is_finite() {
        return Err(PinnError::NanEncountered {
            location: "ic_loss",
        });
    }
    Ok(mse)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ic_loss_zero_at_target() {
        let u = vec![1.0_f32, 2.0, 3.0];
        let loss = ic_loss(&u, &u)
            .expect("initial condition loss computation should succeed when u equals target");
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn ic_loss_constant_error() {
        let u = vec![1.0_f32; 4];
        let t = vec![0.0_f32; 4];
        let loss = ic_loss(&u, &t)
            .expect("initial condition loss computation should succeed for constant-error inputs");
        assert!((loss - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ic_loss_non_negative() {
        let u = vec![0.5_f32, -0.3, 1.2];
        let t = vec![0.0_f32, 0.5, -1.0];
        let loss = ic_loss(&u, &t).expect(
            "initial condition loss computation should succeed for valid mixed-sign inputs",
        );
        assert!(loss >= 0.0);
    }

    #[test]
    fn ic_loss_empty_error() {
        let result = ic_loss(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn ic_loss_dim_mismatch_error() {
        let result = ic_loss(&[1.0, 2.0], &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn ic_loss_single_element() {
        let loss = ic_loss(&[3.0], &[0.0])
            .expect("initial condition loss computation should succeed for single-element inputs");
        assert!((loss - 9.0).abs() < 1e-6);
    }
}
