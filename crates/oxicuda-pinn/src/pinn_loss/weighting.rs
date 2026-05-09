//! NTK-style adaptive loss weighting.

use crate::error::{PinnError, PinnResult};

/// NTK-style adaptive loss weights.
///
/// Weights are updated via: `λᵢ ← α·λᵢ + (1-α)/||∇Lᵢ||`
#[derive(Debug, Clone)]
pub struct AdaptiveWeights {
    pub lambda_pde: f32,
    pub lambda_bc: f32,
    pub lambda_ic: f32,
    alpha: f32,
}

impl AdaptiveWeights {
    /// Create new adaptive weights with equal initial weights.
    pub fn new(alpha: f32) -> PinnResult<Self> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(PinnError::InvalidWeight { weight: alpha });
        }
        Ok(Self {
            lambda_pde: 1.0,
            lambda_bc: 1.0,
            lambda_ic: 1.0,
            alpha,
        })
    }

    /// Update weights based on gradient norms.
    ///
    /// `λᵢ ← α·λᵢ + (1-α) / (||∇Lᵢ|| + ε)`
    pub fn update(
        &mut self,
        grad_norm_pde: f32,
        grad_norm_bc: f32,
        grad_norm_ic: f32,
    ) -> PinnResult<()> {
        if !grad_norm_pde.is_finite() || !grad_norm_bc.is_finite() || !grad_norm_ic.is_finite() {
            return Err(PinnError::NanEncountered {
                location: "adaptive_weights_update",
            });
        }
        let eps = 1e-8_f32;
        let beta = 1.0 - self.alpha;
        self.lambda_pde = self.alpha * self.lambda_pde + beta / (grad_norm_pde + eps);
        self.lambda_bc = self.alpha * self.lambda_bc + beta / (grad_norm_bc + eps);
        self.lambda_ic = self.alpha * self.lambda_ic + beta / (grad_norm_ic + eps);
        Ok(())
    }

    /// Compute weighted total loss.
    pub fn weighted_loss(&self, l_pde: f32, l_bc: f32, l_ic: f32) -> f32 {
        self.lambda_pde * l_pde + self.lambda_bc * l_bc + self.lambda_ic * l_ic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_weights_initial_equal() {
        let w = AdaptiveWeights::new(0.9).unwrap();
        assert_eq!(w.lambda_pde, 1.0);
        assert_eq!(w.lambda_bc, 1.0);
        assert_eq!(w.lambda_ic, 1.0);
    }

    #[test]
    fn adaptive_weights_update_changes_values() {
        let mut w = AdaptiveWeights::new(0.9).unwrap();
        let before = w.lambda_pde;
        w.update(0.5, 1.0, 2.0).unwrap();
        assert!(
            (w.lambda_pde - before).abs() > 1e-6,
            "Update should change lambda_pde"
        );
    }

    #[test]
    fn adaptive_weights_stay_positive() {
        let mut w = AdaptiveWeights::new(0.95).unwrap();
        for _ in 0..10 {
            w.update(1.0, 2.0, 0.5).unwrap();
        }
        assert!(w.lambda_pde > 0.0);
        assert!(w.lambda_bc > 0.0);
        assert!(w.lambda_ic > 0.0);
    }

    #[test]
    fn adaptive_weights_weighted_loss_finite() {
        let w = AdaptiveWeights::new(0.8).unwrap();
        let loss = w.weighted_loss(0.5, 0.3, 0.2);
        assert!(loss.is_finite());
    }

    #[test]
    fn adaptive_weights_invalid_alpha_error() {
        assert!(AdaptiveWeights::new(1.5).is_err());
        assert!(AdaptiveWeights::new(-0.1).is_err());
    }

    #[test]
    fn adaptive_weights_nan_update_error() {
        let mut w = AdaptiveWeights::new(0.9).unwrap();
        let result = w.update(f32::NAN, 1.0, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn adaptive_weighted_loss_formula() {
        let w = AdaptiveWeights {
            lambda_pde: 2.0,
            lambda_bc: 3.0,
            lambda_ic: 1.0,
            alpha: 0.9,
        };
        let loss = w.weighted_loss(1.0, 1.0, 1.0);
        assert!((loss - 6.0).abs() < 1e-6);
    }
}
