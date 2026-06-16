//! Adaptive CFG schedule for dynamic guidance scale during inference.
//!
//! Allows the guidance scale to vary across denoising steps,
//! enabling techniques like time-varying guidance (TVG).

use crate::error::{GenError, GenResult};
use crate::guidance::cfg::{CfgConfig, CfgGuidance};

// ─── AdaptiveCfgPolicy ────────────────────────────────────────────────────────

/// Policy for adapting the CFG guidance scale over denoising steps.
#[derive(Debug, Clone)]
pub enum AdaptiveCfgPolicy {
    /// Constant scale throughout all steps.
    Constant(f32),
    /// Linear interpolation from `start` at step 0 to `end` at the final step.
    Linear { start: f32, end: f32 },
    /// Cosine annealing from `start` to `end`.
    Cosine { start: f32, end: f32 },
    /// Step-wise constant: at each listed `(step, scale)` pair, the scale
    /// applies from that step until the next one. Pairs must be sorted by step.
    StepWise { steps: Vec<(usize, f32)> },
}

// ─── AdaptiveCfgScheduler ─────────────────────────────────────────────────────

/// Adaptive CFG scheduler that varies the guidance scale across denoising steps.
///
/// Useful for techniques like time-varying guidance (TVG) where early steps
/// use a higher scale for structure and later steps use a lower scale for detail.
#[derive(Debug, Clone)]
pub struct AdaptiveCfgScheduler {
    policy: AdaptiveCfgPolicy,
    total_steps: usize,
}

impl AdaptiveCfgScheduler {
    /// Create a new adaptive scheduler.
    ///
    /// # Arguments
    /// - `policy`: The scale schedule policy.
    /// - `total_steps`: Total number of denoising steps.
    pub fn new(policy: AdaptiveCfgPolicy, total_steps: usize) -> Self {
        Self {
            policy,
            total_steps: total_steps.max(1),
        }
    }

    /// Compute the guidance scale at the given denoising step index.
    ///
    /// # Errors
    /// - `InvalidTimestep` if `step >= total_steps`
    /// - `InvalidGuidanceScale` if the computed scale is < 0 (for step-wise policy)
    pub fn scale_at(&self, step: usize) -> GenResult<f32> {
        if step >= self.total_steps {
            return Err(GenError::InvalidTimestep {
                t: step,
                max_t: self.total_steps,
            });
        }
        let scale = match &self.policy {
            AdaptiveCfgPolicy::Constant(s) => *s,
            AdaptiveCfgPolicy::Linear { start, end } => {
                let t = if self.total_steps <= 1 {
                    0.0
                } else {
                    step as f32 / (self.total_steps - 1) as f32
                };
                start + t * (end - start)
            }
            AdaptiveCfgPolicy::Cosine { start, end } => {
                let t = if self.total_steps <= 1 {
                    0.0
                } else {
                    step as f32 / (self.total_steps - 1) as f32
                };
                let cos_t = (t * std::f32::consts::PI).cos();
                end + (start - end) * (cos_t + 1.0) * 0.5
            }
            AdaptiveCfgPolicy::StepWise { steps } => {
                // Find the last (step_threshold, scale) pair where threshold <= step
                let mut result = 1.0_f32;
                for &(threshold, s) in steps {
                    if step >= threshold {
                        result = s;
                    }
                }
                result
            }
        };
        Ok(scale.max(1.0)) // clamp to minimum valid guidance scale
    }

    /// Apply CFG at the given step with the adaptive scale.
    ///
    /// # Errors
    /// - `InvalidTimestep` if `step >= total_steps`
    /// - `InvalidGuidanceScale` if computed scale is invalid
    /// - All errors from `CfgGuidance::apply`
    pub fn apply_at_step(&self, cond: &[f32], uncond: &[f32], step: usize) -> GenResult<Vec<f32>> {
        let scale = self.scale_at(step)?;
        let config = CfgConfig::new(scale)?;
        let guide = CfgGuidance::new(config);
        guide.apply(cond, uncond)
    }

    /// Return all scales for all steps.
    ///
    /// Useful for visualisation and debugging.
    pub fn all_scales(&self) -> GenResult<Vec<f32>> {
        (0..self.total_steps).map(|s| self.scale_at(s)).collect()
    }

    /// Return the total number of steps.
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Return a reference to the policy.
    pub fn policy(&self) -> &AdaptiveCfgPolicy {
        &self.policy
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    #[test]
    fn constant_policy_same_scale() {
        let sched = AdaptiveCfgScheduler::new(AdaptiveCfgPolicy::Constant(5.0), 10);
        for i in 0..10 {
            let s = sched
                .scale_at(i)
                .expect("scale_at should succeed for valid step index in range 0..total_steps");
            assert!((s - 5.0).abs() < EPS, "step {i}: expected 5.0, got {s}");
        }
    }

    #[test]
    fn linear_policy_boundary_values() {
        let sched = AdaptiveCfgScheduler::new(
            AdaptiveCfgPolicy::Linear {
                start: 7.0,
                end: 3.0,
            },
            10,
        );
        let s0 = sched
            .scale_at(0)
            .expect("scale_at step 0 should succeed for linear policy boundary check");
        let s9 = sched.scale_at(9).expect(
            "scale_at final step 9 should succeed for 10-step linear policy boundary check",
        );
        assert!((s0 - 7.0).abs() < EPS, "start: {s0}");
        assert!((s9 - 3.0).abs() < EPS, "end: {s9}");
    }

    #[test]
    fn linear_policy_monotone_decreasing() {
        let sched = AdaptiveCfgScheduler::new(
            AdaptiveCfgPolicy::Linear {
                start: 7.0,
                end: 3.0,
            },
            10,
        );
        let scales: Vec<f32> = (0..10)
            .map(|i| sched.scale_at(i).expect("scale_at should succeed"))
            .collect();
        for w in scales.windows(2) {
            assert!(
                w[1] <= w[0] + EPS,
                "scale should decrease: {} > {}",
                w[1],
                w[0]
            );
        }
    }

    #[test]
    fn cosine_policy_boundary_values() {
        let sched = AdaptiveCfgScheduler::new(
            AdaptiveCfgPolicy::Cosine {
                start: 8.0,
                end: 2.0,
            },
            100,
        );
        let s0 = sched.scale_at(0).expect("scale_at should succeed");
        let s99 = sched.scale_at(99).expect("scale_at should succeed");
        assert!((s0 - 8.0).abs() < EPS, "cosine start: {s0}");
        assert!((s99 - 2.0).abs() < EPS, "cosine end: {s99}");
    }

    #[test]
    fn stepwise_policy_correct_segments() {
        let sched = AdaptiveCfgScheduler::new(
            AdaptiveCfgPolicy::StepWise {
                steps: vec![(0, 7.0), (5, 3.0), (8, 1.5)],
            },
            10,
        );
        assert!((sched.scale_at(0).expect("scale_at should succeed") - 7.0).abs() < EPS);
        assert!((sched.scale_at(4).expect("scale_at should succeed") - 7.0).abs() < EPS);
        assert!((sched.scale_at(5).expect("scale_at should succeed") - 3.0).abs() < EPS);
        assert!((sched.scale_at(7).expect("scale_at should succeed") - 3.0).abs() < EPS);
        assert!((sched.scale_at(8).expect("scale_at should succeed") - 1.5).abs() < EPS);
        assert!((sched.scale_at(9).expect("scale_at should succeed") - 1.5).abs() < EPS);
    }

    #[test]
    fn invalid_step_rejected() {
        let sched = AdaptiveCfgScheduler::new(AdaptiveCfgPolicy::Constant(5.0), 10);
        assert!(matches!(
            sched.scale_at(10),
            Err(GenError::InvalidTimestep { .. })
        ));
    }

    #[test]
    fn apply_at_step_output_shape() {
        let sched = AdaptiveCfgScheduler::new(AdaptiveCfgPolicy::Constant(3.0), 10);
        let cond = vec![1.0_f32; 32];
        let uncond = vec![0.0_f32; 32];
        let out = sched
            .apply_at_step(&cond, &uncond, 5)
            .expect("apply_at_step should succeed");
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn all_scales_count() {
        let sched = AdaptiveCfgScheduler::new(AdaptiveCfgPolicy::Constant(2.0), 20);
        let scales = sched.all_scales().expect("all_scales should succeed");
        assert_eq!(scales.len(), 20);
    }

    #[test]
    fn scale_minimum_clamped_to_one() {
        // Even if policy would give < 1.0, clamp to 1.0
        let sched = AdaptiveCfgScheduler::new(
            AdaptiveCfgPolicy::Linear {
                start: 2.0,
                end: 0.5,
            },
            10,
        );
        for i in 0..10 {
            let s = sched.scale_at(i).expect("scale_at should succeed");
            assert!(s >= 1.0, "scale below 1.0 at step {i}: {s}");
        }
    }
}
