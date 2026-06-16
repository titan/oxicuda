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

// ─── Stochastic Binary Forward (Straight-Through Estimator) ──────────────────

/// Sigmoid activation: σ(x) = 1 / (1 + exp(−x)).
#[inline]
#[must_use]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Sigmoid derivative: σ'(x) = σ(x) · (1 − σ(x)).
#[inline]
fn sigmoid_prime(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

/// Stochastic binary forward pass with straight-through estimator (STE).
///
/// During the **forward pass** each mask element is sampled as a Bernoulli
/// draw:  `b_i ~ Bernoulli(σ(m_i))`.
///
/// The **straight-through gradient** approximation treats the Bernoulli sample
/// as if it were the continuous sigmoid, so the STE multiplier for parameter
/// `m_i` is:
///
/// ```text
/// ste_grad_i = upstream_gradient_i · σ'(m_i)
/// ```
///
/// # Parameters
/// - `mask_values`: real-valued mask parameters (before sigmoid), length n.
/// - `upstream_gradient`: loss gradient w.r.t. effective weights, length n.
/// - `rng`: deterministic LCG RNG for Bernoulli sampling.
///
/// # Returns
/// `(binary_mask, ste_gradients)` both of length n.
///
/// # Errors
/// Returns [`ContinualError::DimensionMismatch`] if lengths differ.
/// Returns [`ContinualError::EmptyInput`] if the inputs are empty.
pub fn stochastic_binary_forward(
    mask_values: &[f64],
    upstream_gradient: &[f64],
    rng: &mut LcgRng,
) -> ContinualResult<(Vec<bool>, Vec<f64>)> {
    if mask_values.is_empty() {
        return Err(ContinualError::EmptyInput);
    }
    if upstream_gradient.len() != mask_values.len() {
        return Err(ContinualError::DimensionMismatch {
            expected: mask_values.len(),
            got: upstream_gradient.len(),
        });
    }

    let n = mask_values.len();
    let mut binary_mask = Vec::with_capacity(n);
    let mut ste_gradients = Vec::with_capacity(n);

    for i in 0..n {
        let prob = sigmoid(mask_values[i]);
        // Bernoulli sample: b = 1 if U(0,1) < prob
        let u = rng.next_f32() as f64;
        binary_mask.push(u < prob);

        // STE gradient: upstream_grad * σ'(m_i)
        let sp = sigmoid_prime(mask_values[i]);
        ste_gradients.push(upstream_gradient[i] * sp);
    }

    Ok((binary_mask, ste_gradients))
}

/// Soft (deterministic) binary forward pass for inference.
///
/// Returns the sigmoid of each mask value — no sampling is performed.
/// Used when the model is in evaluation mode where stochasticity is undesirable.
///
/// # Returns
/// A `Vec<f64>` of length equal to `mask_values` with all values in (0, 1).
#[must_use]
pub fn soft_binary_forward(mask_values: &[f64]) -> Vec<f64> {
    mask_values.iter().map(|&m| sigmoid(m)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binarize_at_threshold() {
        let real_mask = vec![-1.0_f32, -0.1, 0.0, 0.1, 1.0];
        let bin = binarize_mask(&real_mask, 0.0)
            .expect("mask binarization should succeed with valid inputs");
        assert_eq!(bin, vec![0, 0, 0, 1, 1]);
    }

    #[test]
    fn binarize_all_above_threshold() {
        let real_mask = vec![1.0_f32, 2.0, 3.0];
        let bin = binarize_mask(&real_mask, -5.0)
            .expect("mask binarization should succeed with valid inputs");
        assert_eq!(bin, vec![1, 1, 1]);
    }

    #[test]
    fn binarize_all_below_threshold() {
        let real_mask = vec![-1.0_f32, -2.0, -3.0];
        let bin = binarize_mask(&real_mask, 0.0)
            .expect("mask binarization should succeed with valid inputs");
        assert_eq!(bin, vec![0, 0, 0]);
    }

    #[test]
    fn piggyback_forward_preserves_base_weight_scale() {
        let weights = vec![2.0_f32, 3.0, 4.0, 5.0];
        let mask = PiggybackMask {
            real_mask: vec![1.0, -1.0, 1.0, -1.0],
            task_id: 0,
        };
        let effective = piggyback_forward(&weights, &mask, 0.0)
            .expect("piggyback forward should succeed with valid weights");
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
        let effective = piggyback_forward(&weights, &mask, 0.0)
            .expect("piggyback forward should succeed with valid weights");
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

    // ── STE / stochastic binary forward tests ─────────────────────────────────

    #[test]
    fn stochastic_binary_forward_output_is_bool() {
        let mut rng = LcgRng::new(42);
        let mask = vec![0.5_f64, -1.0, 2.0, 0.0, -2.0];
        let upstream = vec![1.0_f64; 5];
        let (bin, _) = stochastic_binary_forward(&mask, &upstream, &mut rng)
            .expect("stochastic binary forward should succeed");
        // Every element must be a valid bool — this just checks length.
        assert_eq!(bin.len(), 5);
        // All values are bool by type; the assertion is implicit.
    }

    #[test]
    fn ste_gradients_same_length_as_mask() {
        let mut rng = LcgRng::new(7);
        let mask = vec![0.1_f64; 8];
        let upstream = vec![1.0_f64; 8];
        let (_, ste) = stochastic_binary_forward(&mask, &upstream, &mut rng)
            .expect("stochastic binary forward should succeed");
        assert_eq!(ste.len(), 8);
    }

    #[test]
    fn soft_binary_forward_all_in_open_unit_interval() {
        let mask = vec![-10.0_f64, -1.0, 0.0, 1.0, 10.0];
        let out = soft_binary_forward(&mask);
        for &v in &out {
            assert!(v > 0.0 && v < 1.0, "soft forward must be in (0,1), got {v}");
        }
    }

    #[test]
    fn soft_binary_forward_at_zero_is_half() {
        let out = soft_binary_forward(&[0.0_f64]);
        assert!(
            (out[0] - 0.5).abs() < 1e-12,
            "sigmoid(0) must be 0.5, got {}",
            out[0]
        );
    }

    #[test]
    fn sigmoid_values_at_key_points() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12, "sigmoid(0) = 0.5");
        assert!(
            sigmoid(100.0) > 0.9999,
            "sigmoid(+∞) should approach 1.0, got {}",
            sigmoid(100.0)
        );
        assert!(
            sigmoid(-100.0) < 1e-4,
            "sigmoid(−∞) should approach 0.0, got {}",
            sigmoid(-100.0)
        );
    }

    #[test]
    fn stochastic_binary_forward_extreme_positive_mostly_true() {
        // With mask = +10 the Bernoulli probability is ≈ 1.0 → almost all True.
        let mut rng = LcgRng::new(99);
        let mask = vec![10.0_f64; 100];
        let upstream = vec![1.0_f64; 100];
        let (bin, _) = stochastic_binary_forward(&mask, &upstream, &mut rng)
            .expect("stochastic binary forward should succeed");
        let n_true = bin.iter().filter(|&&b| b).count();
        assert!(
            n_true >= 95,
            "extreme positive mask should give mostly True, got {n_true}/100"
        );
    }

    #[test]
    fn stochastic_binary_forward_extreme_negative_mostly_false() {
        // With mask = -10 the Bernoulli probability is ≈ 0 → almost all False.
        let mut rng = LcgRng::new(17);
        let mask = vec![-10.0_f64; 100];
        let upstream = vec![1.0_f64; 100];
        let (bin, _) = stochastic_binary_forward(&mask, &upstream, &mut rng)
            .expect("stochastic binary forward should succeed");
        let n_false = bin.iter().filter(|&&b| !b).count();
        assert!(
            n_false >= 95,
            "extreme negative mask should give mostly False, got {n_false}/100"
        );
    }

    #[test]
    fn ste_gradient_same_sign_as_upstream_gradient() {
        // sigmoid'(x) > 0 always, so ste_grad_i and upstream_grad_i share sign.
        let mut rng = LcgRng::new(55);
        let mask = vec![0.0_f64, 1.0, -1.0, 3.0, -3.0];
        let upstream = vec![1.0_f64, -2.0, 3.0, -4.0, 5.0];
        let (_, ste) = stochastic_binary_forward(&mask, &upstream, &mut rng)
            .expect("stochastic binary forward should succeed");
        for (i, (&up, &st)) in upstream.iter().zip(ste.iter()).enumerate() {
            let same_sign = up * st >= 0.0;
            assert!(
                same_sign,
                "STE gradient must share sign with upstream gradient at index {i}: up={up} ste={st}"
            );
        }
    }
}
