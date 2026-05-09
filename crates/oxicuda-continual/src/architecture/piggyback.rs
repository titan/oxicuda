//! Piggyback: Task-specific learned binary masks over a frozen base network.
//!
//! Implements the method from:
//! Mallya et al. "Piggyback: Adapting a Single Network to Multiple Tasks by
//! Learning to Mask Weights." ECCV 2018.
//!
//! Each task learns a real-valued mask that is binarized at a threshold.
//! The effective weights are `w_eff = w_base ⊙ binarize(m)`, keeping the
//! base network frozen while each task adapts a small mask.

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;

/// Configuration for Piggyback masking.
#[derive(Debug, Clone)]
pub struct PiggybackConfig {
    /// Number of parameters in the base network.
    pub base_dim: usize,
    /// Binarization threshold: `m_i = 1 if r_i > threshold else 0`.
    pub threshold: f32,
}

impl Default for PiggybackConfig {
    fn default() -> Self {
        Self {
            base_dim: 256,
            threshold: 0.0,
        }
    }
}

impl PiggybackConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> ContinualResult<()> {
        if !self.threshold.is_finite() {
            return Err(ContinualError::InvalidThreshold {
                threshold: self.threshold,
            });
        }
        if self.base_dim == 0 {
            return Err(ContinualError::EmptyInput);
        }
        Ok(())
    }
}

/// Real-valued mask for a Piggyback task.
///
/// The mask is binarized at `config.threshold` before application.
#[derive(Debug, Clone)]
pub struct PiggybackMask {
    /// Continuous real-valued mask entries (learned during training).
    pub real_mask: Vec<f32>,
    /// Task identifier.
    pub task_id: usize,
}

impl PiggybackMask {
    /// Create a new mask initialized from a uniform distribution in [-1, 1]
    /// using the provided RNG.
    #[must_use]
    pub fn random_init(dim: usize, task_id: usize, rng: &mut LcgRng) -> Self {
        let real_mask = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        Self { real_mask, task_id }
    }
}

/// Binarize a real-valued mask at a threshold.
///
/// `m_i = 1 if r_i > threshold else 0`
pub fn binarize_mask(real_mask: &[f32], threshold: f32) -> ContinualResult<Vec<u8>> {
    if !threshold.is_finite() {
        return Err(ContinualError::InvalidThreshold { threshold });
    }
    Ok(real_mask
        .iter()
        .map(|&r| if r > threshold { 1u8 } else { 0u8 })
        .collect())
}

/// Compute the effective weights for a Piggyback task.
///
/// `w_eff[i] = base_weights[i] * binarize(mask.real_mask[i], threshold)`
///
/// Returns the effective weight vector.
pub fn piggyback_forward(
    weights: &[f32],
    mask: &PiggybackMask,
    threshold: f32,
) -> ContinualResult<Vec<f32>> {
    if weights.len() != mask.real_mask.len() {
        return Err(ContinualError::DimensionMismatch {
            expected: weights.len(),
            got: mask.real_mask.len(),
        });
    }
    let binary = binarize_mask(&mask.real_mask, threshold)?;
    let result = weights
        .iter()
        .zip(binary.iter())
        .map(|(&w, &m)| w * (m as f32))
        .collect();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binarize_at_threshold() {
        let real_mask = vec![-1.0_f32, -0.1, 0.0, 0.1, 1.0];
        let bin = binarize_mask(&real_mask, 0.0).unwrap();
        assert_eq!(bin, vec![0, 0, 0, 1, 1]);
    }

    #[test]
    fn binarize_all_above_threshold() {
        let real_mask = vec![1.0_f32, 2.0, 3.0];
        let bin = binarize_mask(&real_mask, -5.0).unwrap();
        assert_eq!(bin, vec![1, 1, 1]);
    }

    #[test]
    fn binarize_all_below_threshold() {
        let real_mask = vec![-1.0_f32, -2.0, -3.0];
        let bin = binarize_mask(&real_mask, 0.0).unwrap();
        assert_eq!(bin, vec![0, 0, 0]);
    }

    #[test]
    fn piggyback_forward_preserves_base_weight_scale() {
        let weights = vec![2.0_f32, 3.0, 4.0, 5.0];
        let mask = PiggybackMask {
            real_mask: vec![1.0, -1.0, 1.0, -1.0],
            task_id: 0,
        };
        let effective = piggyback_forward(&weights, &mask, 0.0).unwrap();
        // mask = [1, 0, 1, 0] → effective = [2, 0, 4, 0]
        assert_eq!(effective[0], 2.0);
        assert_eq!(effective[1], 0.0);
        assert_eq!(effective[2], 4.0);
        assert_eq!(effective[3], 0.0);
    }

    #[test]
    fn piggyback_forward_all_active() {
        let weights = vec![1.5_f32; 4];
        let mask = PiggybackMask {
            real_mask: vec![1.0_f32; 4],
            task_id: 1,
        };
        let effective = piggyback_forward(&weights, &mask, 0.0).unwrap();
        for &v in &effective {
            assert!(
                (v - 1.5).abs() < 1e-6,
                "All-active mask should pass weights unchanged"
            );
        }
    }

    #[test]
    fn different_tasks_different_masks() {
        let mut rng = LcgRng::new(42);
        let mask0 = PiggybackMask::random_init(8, 0, &mut rng);
        let mask1 = PiggybackMask::random_init(8, 1, &mut rng);
        // Different random seeds should produce different masks with high probability
        let same = mask0
            .real_mask
            .iter()
            .zip(mask1.real_mask.iter())
            .all(|(a, b)| (a - b).abs() < 1e-8);
        assert!(!same, "Different tasks should have different masks");
    }

    #[test]
    fn piggyback_forward_dimension_mismatch() {
        let weights = vec![1.0_f32; 4];
        let mask = PiggybackMask {
            real_mask: vec![1.0_f32; 3],
            task_id: 0,
        };
        assert!(piggyback_forward(&weights, &mask, 0.0).is_err());
    }

    #[test]
    fn binarize_invalid_threshold_returns_err() {
        let real_mask = vec![1.0_f32];
        assert!(binarize_mask(&real_mask, f32::NAN).is_err());
        assert!(binarize_mask(&real_mask, f32::INFINITY).is_err());
    }

    #[test]
    fn piggyback_config_validate() {
        let cfg = PiggybackConfig {
            base_dim: 0,
            threshold: 0.0,
        };
        assert!(cfg.validate().is_err());
        let cfg_nan = PiggybackConfig {
            base_dim: 16,
            threshold: f32::NAN,
        };
        assert!(cfg_nan.validate().is_err());
    }
}
