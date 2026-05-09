//! Boundary condition loss computation.

use crate::error::{PinnError, PinnResult};

/// Type of boundary condition.
#[derive(Debug, Clone)]
pub enum BcType {
    /// `(1/M)Σ|u - g|²`
    Dirichlet,
    /// Neumann in x-direction: gradient penalty
    NeumannX,
    /// Neumann in y-direction: gradient penalty
    NeumannY,
}

/// Compute boundary condition loss.
///
/// For `Dirichlet`: `(1/M)Σ(predictions_i - targets_i)²`.
/// For `NeumannX`/`NeumannY`: `(1/M)Σ(predictions_i - targets_i)²` (targets = prescribed flux).
pub fn bc_loss(predictions: &[f32], targets: &[f32], bc_type: BcType) -> PinnResult<f32> {
    if predictions.is_empty() {
        return Err(PinnError::EmptyCollocationSet);
    }
    if predictions.len() != targets.len() {
        return Err(PinnError::DimensionMismatch {
            expected: predictions.len(),
            got: targets.len(),
        });
    }

    let loss: f32 = match bc_type {
        BcType::Dirichlet | BcType::NeumannX | BcType::NeumannY => {
            let mse: f32 = predictions
                .iter()
                .zip(targets.iter())
                .map(|(&p, &t)| (p - t).powi(2))
                .sum::<f32>()
                / predictions.len() as f32;
            mse
        }
    };

    if !loss.is_finite() {
        return Err(PinnError::NanEncountered {
            location: "bc_loss",
        });
    }
    Ok(loss)
}

/// Compute Dirichlet BC loss on boundary predictions vs zero.
pub fn bc_loss_zero(predictions: &[f32]) -> PinnResult<f32> {
    let targets = vec![0.0_f32; predictions.len()];
    bc_loss(predictions, &targets, BcType::Dirichlet)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirichlet_zero_targets_mse() {
        let preds = vec![0.0_f32; 5];
        let targets = vec![0.0_f32; 5];
        let loss = bc_loss(&preds, &targets, BcType::Dirichlet).unwrap();
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn dirichlet_constant_error() {
        let preds = vec![1.0_f32; 4];
        let targets = vec![0.0_f32; 4];
        let loss = bc_loss(&preds, &targets, BcType::Dirichlet).unwrap();
        assert!((loss - 1.0).abs() < 1e-6);
    }

    #[test]
    fn neumann_x_loss_formula() {
        let preds = vec![2.0_f32, 1.0];
        let targets = vec![0.0_f32, 0.0];
        let loss = bc_loss(&preds, &targets, BcType::NeumannX).unwrap();
        assert!((loss - 2.5).abs() < 1e-6, "Expected 2.5, got {loss}");
    }

    #[test]
    fn neumann_y_loss_finite() {
        let preds = vec![0.5_f32; 3];
        let targets = vec![1.0_f32; 3];
        let loss = bc_loss(&preds, &targets, BcType::NeumannY).unwrap();
        assert!(loss.is_finite());
    }

    #[test]
    fn bc_loss_non_negative() {
        let preds = vec![-1.0_f32, 2.0, -3.0];
        let targets = vec![0.5_f32; 3];
        let loss = bc_loss(&preds, &targets, BcType::Dirichlet).unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn bc_loss_empty_error() {
        let result = bc_loss(&[], &[], BcType::Dirichlet);
        assert!(result.is_err());
    }

    #[test]
    fn bc_loss_dim_mismatch_error() {
        let result = bc_loss(&[1.0, 2.0], &[1.0], BcType::Dirichlet);
        assert!(result.is_err());
    }

    #[test]
    fn bc_loss_zero_helper() {
        let preds = vec![0.5_f32, -0.5];
        let loss = bc_loss_zero(&preds).unwrap();
        assert!((loss - 0.25).abs() < 1e-6);
    }
}
