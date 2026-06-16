//! Classifier guidance for diffusion model inference.
//!
//! Implements gradient-based classifier guidance (Dhariwal & Nichol 2021)
//! and the classifier-free guidance (CFG) combination formula as a
//! standalone module distinct from the CFG combiner in `cfg.rs`.
//!
//! # Reference
//! Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis",
//! NeurIPS 2021. <https://arxiv.org/abs/2105.05233>

use crate::error::{GenError, GenResult};

// ─── ClassifierGuidanceConfig ─────────────────────────────────────────────────

/// Configuration for classifier gradient guidance.
#[derive(Debug, Clone)]
pub struct ClassifierGuidanceConfig {
    /// Guidance scale `γ` (guidance strength). May be any finite `f64`.
    pub guidance_scale: f64,
}

// ─── ClassifierGuidance ───────────────────────────────────────────────────────

/// Classifier guidance combiner.
///
/// Applies classifier gradients to a noise prediction or combines
/// conditional and unconditional predictions via classifier-free guidance.
#[derive(Debug, Clone)]
pub struct ClassifierGuidance {
    config: ClassifierGuidanceConfig,
}

impl ClassifierGuidance {
    /// Create a new classifier guidance instance.
    pub fn new(config: ClassifierGuidanceConfig) -> Self {
        Self { config }
    }

    /// Apply classifier gradient guidance to the unconditional noise prediction.
    ///
    /// Computes the guided noise estimate:
    /// ```text
    /// ε_guided[i] = ε_uncond[i] - σ_t * γ * ∇log p(y|x_t)[i]
    /// ```
    ///
    /// When `σ_t = 0`, the gradient term vanishes and the output equals
    /// `eps_uncond` unchanged.
    ///
    /// # Arguments
    /// - `eps_uncond`: Unconditional noise prediction.
    /// - `classifier_grad`: ∇log p(y|x_t), the classifier gradient w.r.t. `x_t`.
    /// - `sigma_t`: Noise level at timestep `t` (must be finite).
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if `eps_uncond` is empty
    /// - [`GenError::DimensionMismatch`] if lengths differ
    pub fn apply(
        &self,
        eps_uncond: &[f64],
        classifier_grad: &[f64],
        sigma_t: f64,
    ) -> GenResult<Vec<f64>> {
        if eps_uncond.is_empty() {
            return Err(GenError::EmptyInput("eps_uncond must not be empty"));
        }
        if eps_uncond.len() != classifier_grad.len() {
            return Err(GenError::DimensionMismatch {
                expected: eps_uncond.len(),
                got: classifier_grad.len(),
            });
        }

        let gamma = self.config.guidance_scale;
        let result = eps_uncond
            .iter()
            .zip(classifier_grad.iter())
            .map(|(&eps, &grad)| eps - sigma_t * gamma * grad)
            .collect();

        Ok(result)
    }

    /// Combine conditional and unconditional noise predictions via CFG.
    ///
    /// Applies the classifier-free guidance formula:
    /// ```text
    /// ε_cfg[i] = ε_uncond[i] + γ * (ε_cond[i] - ε_uncond[i])
    /// ```
    ///
    /// When `γ = 0`, output equals `eps_uncond`.
    /// When `γ = 1`, output equals `eps_cond`.
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if `eps_uncond` is empty
    /// - [`GenError::DimensionMismatch`] if lengths differ
    pub fn cfg_combine(&self, eps_uncond: &[f64], eps_cond: &[f64]) -> GenResult<Vec<f64>> {
        if eps_uncond.is_empty() {
            return Err(GenError::EmptyInput("eps_uncond must not be empty"));
        }
        if eps_uncond.len() != eps_cond.len() {
            return Err(GenError::DimensionMismatch {
                expected: eps_uncond.len(),
                got: eps_cond.len(),
            });
        }

        let gamma = self.config.guidance_scale;
        let result = eps_uncond
            .iter()
            .zip(eps_cond.iter())
            .map(|(&u, &c)| u + gamma * (c - u))
            .collect();

        Ok(result)
    }

    /// Return the classifier guidance configuration.
    pub fn config(&self) -> &ClassifierGuidanceConfig {
        &self.config
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn make_guide(scale: f64) -> ClassifierGuidance {
        ClassifierGuidance::new(ClassifierGuidanceConfig {
            guidance_scale: scale,
        })
    }

    #[test]
    fn apply_output_shape() {
        let guide = make_guide(1.0);
        let eps_uncond = vec![0.0_f64; 64];
        let grad = vec![1.0_f64; 64];
        let out = guide
            .apply(&eps_uncond, &grad, 0.5)
            .expect("apply should succeed");
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn apply_output_finite() {
        let guide = make_guide(7.5);
        let eps_uncond: Vec<f64> = (0..32).map(|i| (i as f64 - 16.0) * 0.1).collect();
        let grad: Vec<f64> = (0..32).map(|i| i as f64 * 0.01).collect();
        let out = guide
            .apply(&eps_uncond, &grad, 0.3)
            .expect("apply should succeed");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "apply output[{i}]={v} not finite");
        }
    }

    #[test]
    fn gamma_zero_returns_uncond() {
        // γ=0: ε_guided = ε_uncond - σ_t * 0 * grad = ε_uncond
        let guide = make_guide(0.0);
        let eps_uncond = vec![1.0_f64, 2.0, 3.0, 4.0];
        let grad = vec![100.0_f64; 4];
        let out = guide
            .apply(&eps_uncond, &grad, 1.0)
            .expect("apply should succeed");
        for (i, (&o, &u)) in out.iter().zip(&eps_uncond).enumerate() {
            assert!((o - u).abs() < EPS, "gamma=0: out[{i}]={o} != uncond={u}");
        }
    }

    #[test]
    fn cfg_gamma_zero_returns_uncond() {
        let guide = make_guide(0.0);
        let eps_uncond = vec![1.0_f64, 2.0, 3.0];
        let eps_cond = vec![10.0_f64, 20.0, 30.0];
        let out = guide
            .cfg_combine(&eps_uncond, &eps_cond)
            .expect("cfg_combine should succeed");
        // γ=0: out = uncond + 0*(cond - uncond) = uncond
        for (i, (&o, &u)) in out.iter().zip(&eps_uncond).enumerate() {
            assert!(
                (o - u).abs() < EPS,
                "cfg gamma=0: out[{i}]={o} != uncond={u}"
            );
        }
    }

    #[test]
    fn cfg_gamma_one_returns_cond() {
        let guide = make_guide(1.0);
        let eps_uncond = vec![1.0_f64, 2.0, 3.0];
        let eps_cond = vec![10.0_f64, 20.0, 30.0];
        let out = guide
            .cfg_combine(&eps_uncond, &eps_cond)
            .expect("cfg_combine should succeed");
        // γ=1: out = uncond + 1*(cond - uncond) = cond
        for (i, (&o, &c)) in out.iter().zip(&eps_cond).enumerate() {
            assert!((o - c).abs() < EPS, "cfg gamma=1: out[{i}]={o} != cond={c}");
        }
    }

    #[test]
    fn cfg_interpolates() {
        // γ=0.5: out = uncond + 0.5*(cond - uncond) = midpoint
        let guide = make_guide(0.5);
        let eps_uncond = vec![0.0_f64, 0.0, 0.0];
        let eps_cond = vec![2.0_f64, 4.0, 6.0];
        let out = guide
            .cfg_combine(&eps_uncond, &eps_cond)
            .expect("cfg_combine should succeed");
        let expected = [1.0_f64, 2.0, 3.0];
        for (i, (&o, &e)) in out.iter().zip(&expected).enumerate() {
            assert!(
                (o - e).abs() < EPS,
                "cfg gamma=0.5: out[{i}]={o} expected {e}"
            );
        }
    }

    #[test]
    fn dim_mismatch_error() {
        let guide = make_guide(3.0);
        let eps_uncond = vec![0.0_f64; 8];
        let grad = vec![0.0_f64; 16];
        let r = guide.apply(&eps_uncond, &grad, 0.5);
        assert!(
            matches!(r, Err(GenError::DimensionMismatch { .. })),
            "mismatched lengths should fail"
        );

        let eps_cond = vec![0.0_f64; 16];
        let r2 = guide.cfg_combine(&eps_uncond, &eps_cond);
        assert!(
            matches!(r2, Err(GenError::DimensionMismatch { .. })),
            "cfg_combine mismatched lengths should fail"
        );
    }

    #[test]
    fn sigma_zero_no_effect() {
        // σ_t=0: gradient term vanishes → output == eps_uncond
        let guide = make_guide(100.0);
        let eps_uncond = vec![1.0_f64, -2.0, 3.5, -4.0];
        let grad = vec![999.0_f64; 4]; // large gradient that should be nullified
        let out = guide
            .apply(&eps_uncond, &grad, 0.0)
            .expect("apply should succeed");
        for (i, (&o, &u)) in out.iter().zip(&eps_uncond).enumerate() {
            assert!(
                (o - u).abs() < EPS,
                "sigma=0: out[{i}]={o} should equal uncond={u}"
            );
        }
    }

    #[test]
    fn guidance_scale_negative_ok() {
        // Negative scale is valid (repulsion from class); should not error
        let guide = make_guide(-2.0);
        let eps_uncond = vec![0.5_f64; 8];
        let grad = vec![1.0_f64; 8];
        let result = guide.apply(&eps_uncond, &grad, 0.5);
        assert!(result.is_ok(), "negative guidance_scale should not error");
        let out = result.expect("result should be present");
        assert_eq!(out.len(), 8);
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}]={v} must be finite");
        }
    }

    #[test]
    fn apply_correctness() {
        // Verify formula: ε_guided = ε_uncond - σ_t * γ * grad
        let guide = make_guide(2.0);
        let eps_uncond = vec![1.0_f64, 1.0, 1.0];
        let grad = vec![1.0_f64, 1.0, 1.0];
        let sigma_t = 0.5;
        let out = guide
            .apply(&eps_uncond, &grad, sigma_t)
            .expect("apply should succeed");
        // expected: 1 - 0.5*2*1 = 0.0
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - 0.0).abs() < EPS,
                "apply correctness[{i}]={v} expected 0.0"
            );
        }
    }
}
