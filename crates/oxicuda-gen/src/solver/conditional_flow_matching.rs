//! Conditional Flow Matching (CFM) implementation.
//!
//! Implements the OT-CFM (Optimal Transport Conditional Flow Matching) framework
//! from Lipman et al. 2022 and Liu et al. 2022. The flow interpolates linearly
//! between source distribution `x_0` and target distribution `x_1` with an
//! independent Gaussian path variance controlled by `σ_min`.
//!
//! # Reference
//! Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.
//! <https://arxiv.org/abs/2210.02747>

use crate::error::{GenError, GenResult};

// ─── CfmConfig ────────────────────────────────────────────────────────────────

/// Configuration for Conditional Flow Matching.
#[derive(Debug, Clone)]
pub struct CfmConfig {
    /// Minimum path standard deviation `σ_min ∈ (0, 1)`.
    ///
    /// Controls the Gaussian path width around the conditional trajectory.
    /// Smaller values produce sharper flows but require more precise models.
    pub sigma_min: f32,
}

// ─── ConditionalFlowMatching ──────────────────────────────────────────────────

/// Conditional Flow Matching sampler and loss utility.
///
/// Provides:
/// - [`sample_xt`](Self::sample_xt): Interpolate noisy sample at time `t`.
/// - [`target_velocity`](Self::target_velocity): Compute ground-truth velocity.
/// - [`cfm_loss`](Self::cfm_loss): MSE between predicted and target velocity.
/// - [`euler_step`](Self::euler_step): Single Euler ODE integration step.
/// - [`sample_trajectory`](Self::sample_trajectory): Full trajectory from t=0→1.
#[derive(Debug, Clone)]
pub struct ConditionalFlowMatching {
    config: CfmConfig,
}

impl ConditionalFlowMatching {
    /// Create a new CFM sampler with the given configuration.
    ///
    /// # Errors
    /// - [`GenError::InvalidBetaSchedule`] if `sigma_min <= 0` or `sigma_min >= 1`
    pub fn new(config: CfmConfig) -> GenResult<Self> {
        if config.sigma_min <= 0.0 || config.sigma_min >= 1.0 {
            return Err(GenError::InvalidBetaSchedule);
        }
        Ok(Self { config })
    }

    /// Sample the noisy interpolant `x_t` at continuous time `t ∈ [0, 1]`.
    ///
    /// Implements the OT conditional probability path:
    /// ```text
    /// x_t = (1 - (1 - σ_min) * t) * x_0 + t * x_1
    /// ```
    ///
    /// At `t=0`, returns `x_0`; at `t=1`, returns `x_1`.
    ///
    /// # Errors
    /// - [`GenError::InvalidFlowTime`] if `t ∉ [0, 1]`
    /// - [`GenError::EmptyInput`] if `x_0` is empty
    /// - [`GenError::DimensionMismatch`] if `x_0` and `x_1` lengths differ
    pub fn sample_xt(&self, x_0: &[f32], x_1: &[f32], t: f32) -> GenResult<Vec<f32>> {
        if !(0.0..=1.0).contains(&t) {
            return Err(GenError::InvalidFlowTime(t));
        }
        if x_0.is_empty() {
            return Err(GenError::EmptyInput("x_0 must not be empty"));
        }
        if x_0.len() != x_1.len() {
            return Err(GenError::DimensionMismatch {
                expected: x_0.len(),
                got: x_1.len(),
            });
        }

        let sigma_min = self.config.sigma_min;
        let coeff_0 = 1.0 - (1.0 - sigma_min) * t;
        let coeff_1 = t;

        let result = x_0
            .iter()
            .zip(x_1.iter())
            .map(|(&x0, &x1)| coeff_0 * x0 + coeff_1 * x1)
            .collect();

        Ok(result)
    }

    /// Compute the constant conditional target velocity `u_t(x | x_0, x_1)`.
    ///
    /// For the linear OT path, the velocity field is time-independent:
    /// ```text
    /// u_t = x_1 - (1 - σ_min) * x_0
    /// ```
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if `x_0` is empty
    /// - [`GenError::DimensionMismatch`] if lengths differ
    pub fn target_velocity(&self, x_0: &[f32], x_1: &[f32]) -> GenResult<Vec<f32>> {
        if x_0.is_empty() {
            return Err(GenError::EmptyInput("x_0 must not be empty"));
        }
        if x_0.len() != x_1.len() {
            return Err(GenError::DimensionMismatch {
                expected: x_0.len(),
                got: x_1.len(),
            });
        }

        let sigma_min = self.config.sigma_min;
        let result = x_0
            .iter()
            .zip(x_1.iter())
            .map(|(&x0, &x1)| x1 - (1.0 - sigma_min) * x0)
            .collect();

        Ok(result)
    }

    /// Compute the element-wise MSE CFM training loss.
    ///
    /// ```text
    /// L = mean( (v_theta[i] - u_t[i])^2 )
    /// ```
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if `v_theta` is empty
    /// - [`GenError::DimensionMismatch`] if lengths differ
    pub fn cfm_loss(&self, v_theta: &[f32], u_t: &[f32]) -> GenResult<f32> {
        if v_theta.is_empty() {
            return Err(GenError::EmptyInput("v_theta must not be empty"));
        }
        if v_theta.len() != u_t.len() {
            return Err(GenError::DimensionMismatch {
                expected: v_theta.len(),
                got: u_t.len(),
            });
        }

        let n = v_theta.len() as f32;
        let sum: f32 = v_theta
            .iter()
            .zip(u_t.iter())
            .map(|(&v, &u)| (v - u) * (v - u))
            .sum();

        Ok(sum / n)
    }

    /// Perform a single Euler integration step along the velocity field.
    ///
    /// ```text
    /// x_{t+dt} = x_t + dt * velocity
    /// ```
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if `x_t` is empty
    /// - [`GenError::DimensionMismatch`] if lengths differ
    pub fn euler_step(&self, x_t: &[f32], velocity: &[f32], dt: f32) -> GenResult<Vec<f32>> {
        if x_t.is_empty() {
            return Err(GenError::EmptyInput("x_t must not be empty"));
        }
        if x_t.len() != velocity.len() {
            return Err(GenError::DimensionMismatch {
                expected: x_t.len(),
                got: velocity.len(),
            });
        }

        let result = x_t
            .iter()
            .zip(velocity.iter())
            .map(|(&x, &v)| x + dt * v)
            .collect();

        Ok(result)
    }

    /// Integrate the velocity ODE from `t=0` to `t=1` using Euler steps.
    ///
    /// The trajectory is produced by applying `n_steps` uniform Euler steps
    /// with `dt = 1.0 / n_steps`, driven by the provided velocity closure.
    ///
    /// # Arguments
    /// - `x_0`: Initial sample at `t=0`, shape `[D]`.
    /// - `velocity_fn`: Closure `|x: &[f32], t: f32| -> GenResult<Vec<f32>>`
    ///   that returns the predicted velocity at the current `(x, t)`.
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if `n_steps == 0` or `x_0` is empty
    /// - Any error propagated from `velocity_fn`
    pub fn sample_trajectory<F>(
        &self,
        x_0: &[f32],
        n_steps: usize,
        velocity_fn: &mut F,
    ) -> GenResult<Vec<f32>>
    where
        F: FnMut(&[f32], f32) -> GenResult<Vec<f32>>,
    {
        if n_steps == 0 {
            return Err(GenError::EmptyInput("n_steps must be > 0"));
        }
        if x_0.is_empty() {
            return Err(GenError::EmptyInput("x_0 must not be empty"));
        }

        let dt = 1.0_f32 / n_steps as f32;
        let mut x_t = x_0.to_vec();

        for step in 0..n_steps {
            let t = step as f32 * dt;
            let vel = velocity_fn(&x_t, t)?;
            x_t = self.euler_step(&x_t, &vel, dt)?;
        }

        Ok(x_t)
    }

    /// Return the CFM configuration.
    pub fn config(&self) -> &CfmConfig {
        &self.config
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn make_cfm() -> ConditionalFlowMatching {
        ConditionalFlowMatching::new(CfmConfig { sigma_min: 0.01 }).expect("new should succeed")
    }

    #[test]
    fn sample_xt_at_t0() {
        let cfm = make_cfm();
        let x_0 = vec![1.0_f32, 2.0, 3.0];
        let x_1 = vec![4.0_f32, 5.0, 6.0];
        let x_t = cfm
            .sample_xt(&x_0, &x_1, 0.0)
            .expect("sample_xt should succeed");
        for (&xt, &x0) in x_t.iter().zip(&x_0) {
            assert!((xt - x0).abs() < EPS, "t=0: x_t={xt} should equal x_0={x0}");
        }
    }

    #[test]
    fn sample_xt_at_t1() {
        let cfm = make_cfm();
        let x_0 = vec![1.0_f32, 2.0, 3.0];
        let x_1 = vec![4.0_f32, 5.0, 6.0];
        let x_t = cfm
            .sample_xt(&x_0, &x_1, 1.0)
            .expect("sample_xt should succeed");
        // At t=1: x_t = sigma_min * x_0 + x_1, but sigma_min=0.01 so ≈ x_1
        // Actually: coeff_0 = 1 - (1-sigma_min)*1 = sigma_min, coeff_1 = 1
        let sigma_min = 0.01_f32;
        for (i, (&xt, (&x0, &x1))) in x_t.iter().zip(x_0.iter().zip(x_1.iter())).enumerate() {
            let expected = sigma_min * x0 + x1;
            assert!(
                (xt - expected).abs() < EPS,
                "t=1: x_t[{i}]={xt} should equal sigma_min*x_0+x_1={expected}"
            );
        }
    }

    #[test]
    fn sample_xt_output_shape() {
        let cfm = make_cfm();
        let x_0 = vec![0.0_f32; 128];
        let x_1 = vec![1.0_f32; 128];
        let x_t = cfm
            .sample_xt(&x_0, &x_1, 0.5)
            .expect("sample_xt should succeed");
        assert_eq!(x_t.len(), 128);
    }

    #[test]
    fn target_velocity_shape() {
        let cfm = make_cfm();
        let x_0 = vec![0.0_f32; 64];
        let x_1 = vec![1.0_f32; 64];
        let vel = cfm
            .target_velocity(&x_0, &x_1)
            .expect("target_velocity should succeed");
        assert_eq!(vel.len(), 64);
    }

    #[test]
    fn target_velocity_constant() {
        // Velocity should be time-independent; compute twice and compare
        let cfm = make_cfm();
        let x_0: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let x_1: Vec<f32> = (0..32).map(|i| i as f32 * 0.2 + 1.0).collect();
        let vel_a = cfm
            .target_velocity(&x_0, &x_1)
            .expect("target_velocity should succeed");
        let vel_b = cfm
            .target_velocity(&x_0, &x_1)
            .expect("target_velocity should succeed");
        for (i, (&a, &b)) in vel_a.iter().zip(&vel_b).enumerate() {
            assert!(
                (a - b).abs() < 1e-7,
                "velocity[{i}] changed between calls: {a} vs {b}"
            );
        }
    }

    #[test]
    fn cfm_loss_zero_for_perfect() {
        let cfm = make_cfm();
        let x_0: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let x_1: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let u_t = cfm
            .target_velocity(&x_0, &x_1)
            .expect("target_velocity should succeed");
        // Perfect prediction: v_theta == u_t
        let loss = cfm.cfm_loss(&u_t, &u_t).expect("cfm_loss should succeed");
        assert!(
            loss.abs() < EPS,
            "loss for perfect prediction should be 0, got {loss}"
        );
    }

    #[test]
    fn euler_step_shape() {
        let cfm = make_cfm();
        let x_t = vec![0.0_f32; 48];
        let vel = vec![1.0_f32; 48];
        let out = cfm
            .euler_step(&x_t, &vel, 0.01)
            .expect("euler_step should succeed");
        assert_eq!(out.len(), 48);
    }

    #[test]
    fn sample_trajectory_shape() {
        let cfm = make_cfm();
        let x_0 = vec![0.0_f32; 16];
        let mut vel_fn = |_x: &[f32], _t: f32| -> GenResult<Vec<f32>> { Ok(vec![1.0_f32; 16]) };
        let out = cfm
            .sample_trajectory(&x_0, 10, &mut vel_fn)
            .expect("sample_trajectory should succeed");
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn sigma_min_invalid_error() {
        // sigma_min <= 0 → Err
        let r1 = ConditionalFlowMatching::new(CfmConfig { sigma_min: 0.0 });
        assert!(
            matches!(r1, Err(GenError::InvalidBetaSchedule)),
            "sigma_min=0 should fail"
        );
        // sigma_min < 0 → Err
        let r2 = ConditionalFlowMatching::new(CfmConfig { sigma_min: -0.1 });
        assert!(
            matches!(r2, Err(GenError::InvalidBetaSchedule)),
            "sigma_min<0 should fail"
        );
        // sigma_min >= 1 → Err
        let r3 = ConditionalFlowMatching::new(CfmConfig { sigma_min: 1.0 });
        assert!(
            matches!(r3, Err(GenError::InvalidBetaSchedule)),
            "sigma_min=1 should fail"
        );
    }

    #[test]
    fn t_out_of_range_error() {
        let cfm = make_cfm();
        let x_0 = vec![0.0_f32; 4];
        let x_1 = vec![1.0_f32; 4];
        let r = cfm.sample_xt(&x_0, &x_1, 1.5);
        assert!(
            matches!(r, Err(GenError::InvalidFlowTime(_))),
            "t=1.5 should fail with InvalidFlowTime"
        );
        let r2 = cfm.sample_xt(&x_0, &x_1, -0.1);
        assert!(
            matches!(r2, Err(GenError::InvalidFlowTime(_))),
            "t=-0.1 should fail with InvalidFlowTime"
        );
    }

    #[test]
    fn n_steps_zero_error() {
        let cfm = make_cfm();
        let x_0 = vec![0.0_f32; 8];
        let mut vel_fn = |_x: &[f32], _t: f32| -> GenResult<Vec<f32>> { Ok(vec![0.0_f32; 8]) };
        let r = cfm.sample_trajectory(&x_0, 0, &mut vel_fn);
        assert!(
            matches!(r, Err(GenError::EmptyInput(_))),
            "n_steps=0 should fail with EmptyInput"
        );
    }

    #[test]
    fn euler_step_correctness() {
        // x + dt * v with known values
        let cfm = make_cfm();
        let x_t = vec![1.0_f32, 2.0, 3.0];
        let vel = vec![10.0_f32, 20.0, 30.0];
        let out = cfm
            .euler_step(&x_t, &vel, 0.1)
            .expect("euler_step should succeed");
        // expected: [1 + 0.1*10, 2 + 0.1*20, 3 + 0.1*30] = [2, 4, 6]
        let expected = [2.0_f32, 4.0, 6.0];
        for (i, (&o, &e)) in out.iter().zip(&expected).enumerate() {
            assert!((o - e).abs() < EPS, "euler[{i}]={o} expected {e}");
        }
    }

    #[test]
    fn dim_mismatch_error() {
        let cfm = make_cfm();
        let x_0 = vec![0.0_f32; 8];
        let x_1 = vec![1.0_f32; 16];
        assert!(
            matches!(
                cfm.sample_xt(&x_0, &x_1, 0.5),
                Err(GenError::DimensionMismatch { .. })
            ),
            "mismatched x_0/x_1 should fail"
        );
        assert!(
            matches!(
                cfm.target_velocity(&x_0, &x_1),
                Err(GenError::DimensionMismatch { .. })
            ),
            "mismatched x_0/x_1 for target_velocity should fail"
        );
    }
}
