//! SimCLR — Chen et al. 2020 — symmetric NT-Xent contrastive loss.
//!
//! Given two augmented views `(z_a, z_b)` of the same `N` items, computes the
//! symmetric InfoNCE loss with cosine similarity at temperature τ:
//!
//! ```text
//!   L = (1/2N) Σ_i [ −log p_{ab}(i, i) − log p_{ba}(i, i) ]
//!   p_{ab}(i, j) = exp(s_{ij}/τ) / Σ_k exp(s_{ik}/τ)
//! ```

use crate::contrastive::info_nce::info_nce_loss;
use crate::error::{SslError, SslResult};

/// Configuration for SimCLR.
#[derive(Debug, Clone)]
pub struct SimClrConfig {
    /// Temperature for InfoNCE (default 0.1).
    pub temperature: f32,
}

impl Default for SimClrConfig {
    fn default() -> Self {
        Self { temperature: 0.1 }
    }
}

impl SimClrConfig {
    /// Create a validated SimCLR config.
    ///
    /// # Errors
    /// [`SslError::InvalidTemperature`] if `temperature <= 0` or non-finite.
    pub fn new(temperature: f32) -> SslResult<Self> {
        if !(temperature.is_finite() && temperature > 0.0) {
            return Err(SslError::InvalidTemperature { temp: temperature });
        }
        Ok(Self { temperature })
    }
}

/// Compute the SimCLR symmetric NT-Xent loss.
///
/// `z_a` and `z_b` are `[N, D]` row-major projection matrices for two
/// augmented views of the same `N` items (positive pairs are diagonal).
/// Returns `(loss, accuracy@1)`.
///
/// # Errors
/// Propagates errors from [`info_nce_loss`].
pub fn simclr_loss(
    z_a: &[f32],
    z_b: &[f32],
    n: usize,
    d: usize,
    config: &SimClrConfig,
) -> SslResult<(f32, f32)> {
    info_nce_loss(z_a, z_b, n, d, config.temperature)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simclr_default_temperature() {
        let cfg = SimClrConfig::default();
        assert!((cfg.temperature - 0.1).abs() < 1e-7);
    }

    #[test]
    fn simclr_new_validates_temperature() {
        assert!(SimClrConfig::new(0.0).is_err());
        assert!(SimClrConfig::new(-1.0).is_err());
        assert!(SimClrConfig::new(f32::NAN).is_err());
        assert!(SimClrConfig::new(0.5).is_ok());
    }

    #[test]
    fn simclr_loss_distinct_paired_views_low() {
        // Distinct rows so off-diagonal cosine similarity is small; identical
        // pair so diagonal cosine is 1.
        let n = 4;
        let d = 8;
        let mut z = vec![0.0_f32; n * d];
        for i in 0..n {
            z[i * d + i] = 1.0;
        }
        let cfg = SimClrConfig::default();
        let (loss, acc) = simclr_loss(&z, &z, n, d, &cfg).unwrap();
        assert!(loss < 0.5);
        assert!((acc - 1.0).abs() < 1e-6);
    }

    #[test]
    fn simclr_loss_random_views_finite() {
        let n = 16;
        let d = 32;
        let z_a: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.0123).sin()).collect();
        let z_b: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.0451).cos()).collect();
        let cfg = SimClrConfig::default();
        let (loss, _) = simclr_loss(&z_a, &z_b, n, d, &cfg).unwrap();
        assert!(loss.is_finite());
    }

    #[test]
    fn simclr_temperature_affects_loss() {
        let n = 4;
        let d = 8;
        let z: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.05).collect();
        let z2: Vec<f32> = z.iter().map(|v| v + 0.1).collect();
        let high_t = SimClrConfig { temperature: 1.0 };
        let low_t = SimClrConfig { temperature: 0.05 };
        let (l_high, _) = simclr_loss(&z, &z2, n, d, &high_t).unwrap();
        let (l_low, _) = simclr_loss(&z, &z2, n, d, &low_t).unwrap();
        // Lower temperature sharpens softmax; well-aligned positives → lower loss.
        assert!(l_low <= l_high + 1e-3, "l_low={l_low}, l_high={l_high}");
    }
}
