//! BYOL — Grill et al. 2020 — Bootstrap Your Own Latent.
//!
//! BYOL avoids explicit negative pairs by training an *online* network to
//! predict the projection of a *target* (momentum) network on the same input.
//! The loss is the L2-normalised cosine distance
//! ```text
//!     L = ‖p̂ − ẑ_target‖² = 2 − 2 · cos(p, z_target)
//! ```
//! where `p̂ = p/‖p‖` and `ẑ_target = z_target/‖z_target‖`. The target stream
//! receives a stop-gradient (`sg`) so only the online network is updated.
//!
//! This module contains the loss helper and a thin [`ByolPredictor`] struct
//! that wraps the prediction MLP head (an additional two-layer MLP placed on
//! top of the online projection — required only for the online branch).

use crate::error::{SslError, SslResult};
use crate::head::predictor::PredictorHead;

/// Cosine-MSE loss between online predictions `p` and target projections `z`.
///
/// Both inputs are `[N, D]` row-major; both are L2-normalised internally.
/// Returns the mean across the batch.
///
/// # Errors
/// - [`SslError::EmptyInput`] when `n == 0` or `d == 0`.
/// - [`SslError::DimensionMismatch`] when shapes disagree.
pub fn byol_loss(p: &[f32], z: &[f32], n: usize, d: usize) -> SslResult<f32> {
    if n == 0 || d == 0 {
        return Err(SslError::EmptyInput);
    }
    if p.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: p.len(),
        });
    }
    if z.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: z.len(),
        });
    }
    let mut total = 0.0_f64;
    for i in 0..n {
        let p_row = &p[i * d..(i + 1) * d];
        let z_row = &z[i * d..(i + 1) * d];
        let p_norm = (p_row.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-12);
        let z_norm = (z_row.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-12);
        let mut dot = 0.0_f32;
        for (a, b) in p_row.iter().zip(z_row.iter()) {
            dot += a * b;
        }
        let cos = dot / (p_norm * z_norm);
        total += (2.0 - 2.0 * cos as f64).max(0.0);
    }
    Ok((total / n as f64) as f32)
}

/// BYOL predictor head wrapper.
///
/// Wraps a [`PredictorHead`] for ergonomic use in a BYOL training loop:
/// `loss = byol_loss(predictor(z_online), sg(z_target))`.
#[derive(Debug, Clone)]
pub struct ByolPredictor {
    /// Underlying predictor MLP.
    pub predictor: PredictorHead,
}

impl ByolPredictor {
    /// New BYOL predictor from a [`PredictorHead`].
    #[must_use]
    pub fn new(predictor: PredictorHead) -> Self {
        Self { predictor }
    }

    /// Apply the predictor to a batch of online projections.
    ///
    /// # Errors
    /// Propagates [`PredictorHead::forward`] errors.
    pub fn forward(&self, z_online: &[f32]) -> SslResult<Vec<f32>> {
        self.predictor.forward(z_online)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn byol_loss_identical_inputs_zero() {
        let z: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 + 1.0).collect();
        let l = byol_loss(&z, &z, 4, 4).unwrap();
        assert!(l.abs() < 1e-4, "l = {l}");
    }

    #[test]
    fn byol_loss_orthogonal_pair_two() {
        // (1, 0) vs (0, 1) → cos = 0 → loss = 2
        let p = vec![1.0_f32, 0.0];
        let z = vec![0.0_f32, 1.0];
        let l = byol_loss(&p, &z, 1, 2).unwrap();
        assert!((l - 2.0).abs() < 1e-5, "l = {l}");
    }

    #[test]
    fn byol_loss_anti_parallel_pair_four() {
        // (1, 0) vs (-1, 0) → cos = -1 → loss = 4
        let p = vec![1.0_f32, 0.0];
        let z = vec![-1.0_f32, 0.0];
        let l = byol_loss(&p, &z, 1, 2).unwrap();
        assert!((l - 4.0).abs() < 1e-5, "l = {l}");
    }

    #[test]
    fn byol_loss_invariant_to_scale() {
        let p = vec![1.0_f32, 0.0, 0.0];
        let z = vec![1.0_f32, 0.0, 0.0];
        let l1 = byol_loss(&p, &z, 1, 3).unwrap();
        let p2 = vec![10.0_f32, 0.0, 0.0];
        let z2 = vec![100.0_f32, 0.0, 0.0];
        let l2 = byol_loss(&p2, &z2, 1, 3).unwrap();
        assert!((l1 - l2).abs() < 1e-5);
    }

    #[test]
    fn byol_loss_zero_input_safe() {
        let p = vec![0.0_f32, 0.0];
        let z = vec![1.0_f32, 0.0];
        let l = byol_loss(&p, &z, 1, 2).unwrap();
        assert!(l.is_finite());
    }

    #[test]
    fn byol_loss_rejects_dim_mismatch() {
        let p = vec![1.0_f32, 0.0];
        let z = vec![1.0_f32, 0.0, 0.0];
        assert!(byol_loss(&p, &z, 1, 2).is_err());
    }

    #[test]
    fn byol_loss_rejects_empty() {
        let r = byol_loss(&[], &[], 0, 0);
        assert!(r.is_err());
    }

    #[test]
    fn byol_predictor_round_trip_shape() {
        let mut rng = LcgRng::new(0);
        let pred = PredictorHead::new(8, 4, 8, &mut rng).unwrap();
        let online = ByolPredictor::new(pred);
        let z = vec![0.1_f32; 8];
        let p = online.forward(&z).unwrap();
        assert_eq!(p.len(), 8);
    }
}
