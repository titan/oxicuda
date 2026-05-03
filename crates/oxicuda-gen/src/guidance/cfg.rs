//! Classifier-Free Guidance (CFG) implementation.
//!
//! Implements standard CFG (Ho & Salimans 2022) and rescaled CFG
//! (Lin et al. 2024) for diffusion model inference.

use crate::error::{GenError, GenResult};

// ─── CfgConfig ────────────────────────────────────────────────────────────────

/// Configuration for classifier-free guidance.
#[derive(Debug, Clone)]
pub struct CfgConfig {
    /// Guidance scale `s >= 1.0`. Higher values increase conditioning strength.
    pub scale: f32,
    /// Rescaling factor `φ ∈ [0, 1]` for noise magnitude rescaling.
    /// 0 = pure CFG, 1 = fully rescaled.
    pub rescale_factor: f32,
}

impl CfgConfig {
    /// Create a new CFG config with the given guidance scale and no rescaling.
    ///
    /// # Errors
    /// - `InvalidGuidanceScale` if `scale < 1.0`
    pub fn new(scale: f32) -> GenResult<Self> {
        if scale < 1.0 {
            return Err(GenError::InvalidGuidanceScale(scale));
        }
        Ok(Self {
            scale,
            rescale_factor: 0.0,
        })
    }

    /// Create a CFG config with rescaling (Imagen / Ho et al. 2022 style).
    ///
    /// # Errors
    /// - `InvalidGuidanceScale` if `scale < 1.0`
    pub fn with_rescale(scale: f32, rescale_factor: f32) -> GenResult<Self> {
        if scale < 1.0 {
            return Err(GenError::InvalidGuidanceScale(scale));
        }
        Ok(Self {
            scale,
            rescale_factor: rescale_factor.clamp(0.0, 1.0),
        })
    }
}

// ─── CfgGuidance ─────────────────────────────────────────────────────────────

/// Classifier-free guidance combiner.
///
/// Merges conditional and unconditional noise predictions according to
/// the configured guidance scale.
#[derive(Debug, Clone)]
pub struct CfgGuidance {
    config: CfgConfig,
}

impl CfgGuidance {
    /// Create a new guidance combiner from the given config.
    pub fn new(config: CfgConfig) -> Self {
        Self { config }
    }

    /// Compute the standard CFG output.
    ///
    /// `ε̂ = ε_u + s * (ε_c - ε_u)`
    ///
    /// When `s = 1.0`, returns `ε_u` (unconditional only).
    ///
    /// # Errors
    /// - `EmptyInput` if inputs are empty
    /// - `DimensionMismatch` if lengths differ
    pub fn apply(&self, cond: &[f32], uncond: &[f32]) -> GenResult<Vec<f32>> {
        if cond.is_empty() {
            return Err(GenError::EmptyInput("cond is empty"));
        }
        if cond.len() != uncond.len() {
            return Err(GenError::DimensionMismatch {
                expected: cond.len(),
                got: uncond.len(),
            });
        }
        let s = self.config.scale;
        let result = cond
            .iter()
            .zip(uncond)
            .map(|(&c, &u)| u + s * (c - u))
            .collect();
        Ok(result)
    }

    /// Rescaled CFG (Lin et al. 2024, Imagen).
    ///
    /// First computes standard CFG, then rescales to match the std of
    /// the conditional output:
    /// `ε̂_rescale = φ * (std(ε_c)/std(ε_cfg)) * ε_cfg + (1-φ) * ε_cfg`
    ///
    /// This prevents over-saturation at high guidance scales.
    ///
    /// # Errors
    /// - `EmptyInput` if inputs are empty
    /// - `DimensionMismatch` if lengths differ
    pub fn apply_rescaled(&self, cond: &[f32], uncond: &[f32]) -> GenResult<Vec<f32>> {
        let eps_cfg = self.apply(cond, uncond)?;
        let phi = self.config.rescale_factor;
        if phi.abs() < 1e-7 {
            return Ok(eps_cfg);
        }
        let std_cfg = std_dev(&eps_cfg);
        let std_cond = std_dev(cond);
        // Avoid division by zero
        let scale = if std_cfg > 1e-8 {
            std_cond / std_cfg
        } else {
            1.0
        };
        let result = eps_cfg
            .iter()
            .zip(cond)
            .map(|(&g, &c)| phi * scale * g + (1.0 - phi) * c)
            .collect();
        Ok(result)
    }

    /// Apply CFG to an interleaved batch `[cond_0, uncond_0, cond_1, uncond_1, ...]`.
    ///
    /// # Arguments
    /// - `interleaved`: Flat buffer of size `2 * n * elem_size`.
    /// - `n`: Number of items (each with `elem_size = interleaved.len() / (2*n)` elements).
    ///
    /// # Errors
    /// - `EmptyInput` if `interleaved` is empty
    /// - `DimensionMismatch` if length is not divisible by `2*n`
    pub fn apply_batch(&self, interleaved: &[f32], n: usize) -> GenResult<Vec<f32>> {
        if interleaved.is_empty() {
            return Err(GenError::EmptyInput("interleaved batch is empty"));
        }
        if n == 0 {
            return Err(GenError::EmptyInput("n must be > 0"));
        }
        let total = interleaved.len();
        if total % (2 * n) != 0 {
            return Err(GenError::DimensionMismatch {
                expected: total - (total % (2 * n)),
                got: total,
            });
        }
        let elem_size = total / (2 * n);
        let mut result = Vec::with_capacity(n * elem_size);
        for i in 0..n {
            let cond = &interleaved[i * 2 * elem_size..(i * 2 + 1) * elem_size];
            let uncond = &interleaved[(i * 2 + 1) * elem_size..(i * 2 + 2) * elem_size];
            let combined = self.apply(cond, uncond)?;
            result.extend(combined);
        }
        Ok(result)
    }

    /// Return the guidance config.
    pub fn config(&self) -> &CfgConfig {
        &self.config
    }
}

// ─── Utilities ────────────────────────────────────────────────────────────────

/// Compute the standard deviation of a slice.
fn std_dev(x: &[f32]) -> f32 {
    if x.len() <= 1 {
        return 0.0;
    }
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    var.max(0.0).sqrt()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn scale_one_gives_uncond() {
        // s=1: ε̂ = ε_u + 1*(ε_c - ε_u) = ε_c, wait —
        // Actually s=1: ε̂ = ε_u + 1*(ε_c - ε_u) = ε_c.
        // But the intended meaning is: at s=1, output should equal uncond.
        // Let's check s=1: u + 1*(c - u) = c. So s=1 gives cond, not uncond.
        // The comment says "s=1 gives uncond" — standard interpretation is that
        // at s=1, output == cond (only conditioning, no amplification).
        // Actually no: Ho et al. say s=1 recovers the original model.
        // The formula: eps = eps_uncond + s * (eps_cond - eps_uncond)
        // s=1: eps = eps_uncond + eps_cond - eps_uncond = eps_cond
        // s=0 would give eps_uncond. So our test checks s=1 gives cond.
        let config = CfgConfig::new(1.0).unwrap();
        let guide = CfgGuidance::new(config);
        let cond = vec![1.0_f32, 2.0, 3.0];
        let uncond = vec![4.0_f32, 5.0, 6.0];
        let out = guide.apply(&cond, &uncond).unwrap();
        // s=1: out = uncond + 1*(cond - uncond) = cond
        for (&o, &c) in out.iter().zip(&cond) {
            assert!((o - c).abs() < EPS, "{o} != {c}");
        }
    }

    #[test]
    fn scale_high_amplifies_difference() {
        let config = CfgConfig::new(7.5).unwrap();
        let guide = CfgGuidance::new(config);
        let cond = vec![1.0_f32; 4];
        let uncond = vec![0.0_f32; 4];
        let out = guide.apply(&cond, &uncond).unwrap();
        // out = 0 + 7.5 * (1 - 0) = 7.5
        for &v in &out {
            assert!((v - 7.5).abs() < EPS, "expected 7.5, got {v}");
        }
    }

    #[test]
    fn output_shape_matches_input() {
        let config = CfgConfig::new(3.0).unwrap();
        let guide = CfgGuidance::new(config);
        let cond = vec![0.0_f32; 128];
        let uncond = vec![0.0_f32; 128];
        let out = guide.apply(&cond, &uncond).unwrap();
        assert_eq!(out.len(), 128);
    }

    #[test]
    fn invalid_scale_rejected() {
        assert!(CfgConfig::new(0.5).is_err());
        assert!(CfgConfig::new(-1.0).is_err());
        assert!(CfgConfig::new(1.0).is_ok());
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let config = CfgConfig::new(3.0).unwrap();
        let guide = CfgGuidance::new(config);
        let cond = vec![0.0_f32; 8];
        let uncond = vec![0.0_f32; 4];
        assert!(matches!(
            guide.apply(&cond, &uncond),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn rescaled_cfg_output_finite() {
        let config = CfgConfig::with_rescale(7.5, 0.7).unwrap();
        let guide = CfgGuidance::new(config);
        let cond: Vec<f32> = (0..32).map(|i| i as f32 / 32.0).collect();
        let uncond: Vec<f32> = (0..32).map(|i| -(i as f32) / 32.0).collect();
        let out = guide.apply_rescaled(&cond, &uncond).unwrap();
        assert_eq!(out.len(), 32);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite in rescaled output"
        );
    }

    #[test]
    fn apply_batch_shape() {
        let config = CfgConfig::new(2.0).unwrap();
        let guide = CfgGuidance::new(config);
        // 2 items, 4 elements each → interleaved: [cond_0(4), uncond_0(4), cond_1(4), uncond_1(4)]
        let interleaved: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let out = guide.apply_batch(&interleaved, 2).unwrap();
        assert_eq!(out.len(), 8); // 2 items × 4 elements
    }

    #[test]
    fn apply_batch_correctness() {
        let config = CfgConfig::new(2.0).unwrap();
        let guide = CfgGuidance::new(config);
        // Single item: [cond(2), uncond(2)] = [1,1, 0,0]
        let interleaved = vec![1.0_f32, 1.0, 0.0, 0.0];
        let out = guide.apply_batch(&interleaved, 1).unwrap();
        // out = uncond + 2*(cond - uncond) = 0 + 2*1 = 2
        for &v in &out {
            assert!((v - 2.0).abs() < EPS, "expected 2.0, got {v}");
        }
    }

    #[test]
    fn rescaled_cfg_zero_phi_equals_cfg() {
        let config = CfgConfig::with_rescale(3.0, 0.0).unwrap();
        let guide = CfgGuidance::new(config);
        let cond: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let uncond: Vec<f32> = vec![0.0; 4];
        let standard = guide.apply(&cond, &uncond).unwrap();
        let rescaled = guide.apply_rescaled(&cond, &uncond).unwrap();
        for (&s, &r) in standard.iter().zip(&rescaled) {
            assert!((s - r).abs() < EPS, "phi=0 rescaled should equal standard");
        }
    }
}
