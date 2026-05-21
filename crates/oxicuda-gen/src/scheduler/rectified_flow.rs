//! Rectified Flow scheduler.
//!
//! Implements Rectified Flow (Liu, Gong & Liu, ICLR 2023, "Flow Straight and
//! Fast: Learning to Generate and Transfer Data with Rectified Flow").
//!
//! The core idea is to learn an ODE whose trajectories are as straight as
//! possible. A linear interpolation path
//!
//! ```text
//! X_t = (1 - t) · x0 + t · x1 ,  t ∈ [0, 1]
//! ```
//!
//! is constructed between a source sample `x0` (typically noise) and a target
//! sample `x1` (data). The velocity field is trained to match the **constant**
//! direction `x1 − x0` along this path. Sampling integrates
//!
//! ```text
//! dX/dt = v(X_t, t)
//! ```
//!
//! from `t = 0` to `t = 1`. Because a perfectly-trained field is constant in
//! time, the trajectory is a straight line and a forward Euler integrator
//! reproduces it exactly with even a single step.
//!
//! "Reflow" regenerates training pairs `(x0, x1_hat)` by integrating the
//! current flow forward from `x0`; retraining on these straighter pairs
//! progressively reduces trajectory curvature.
//!
//! # Reference
//! Liu, Gong & Liu, "Flow Straight and Fast: Learning to Generate and Transfer
//! Data with Rectified Flow", ICLR 2023.

use crate::error::{GenError, GenResult};

// ─── RectifiedFlowConfig ────────────────────────────────────────────────────────

/// Configuration for the [`RectifiedFlow`] scheduler.
#[derive(Debug, Clone, PartialEq)]
pub struct RectifiedFlowConfig {
    /// Dimensionality of the state vectors. Must be ≥ 1.
    pub dim: usize,
    /// Number of Euler/Heun integration steps used during sampling. Must be ≥ 1.
    pub n_steps: usize,
    /// When `true`, use Heun's (RK2 / trapezoidal) corrector; otherwise plain
    /// forward Euler.
    pub heun: bool,
}

// ─── RectifiedFlow ───────────────────────────────────────────────────────────────

/// Rectified Flow scheduler over linear interpolation paths.
///
/// Provides the interpolation path, the constant target velocity, the flow
/// matching loss, an Euler/Heun ODE sampler, reflow pair generation, and a
/// straightness diagnostic.
///
/// # Reference
/// Liu, Gong & Liu, "Flow Straight and Fast: Learning to Generate and Transfer
/// Data with Rectified Flow", ICLR 2023.
#[derive(Debug, Clone)]
pub struct RectifiedFlow {
    /// Configuration controlling dimensionality and the ODE solver.
    pub cfg: RectifiedFlowConfig,
}

impl RectifiedFlow {
    /// Build a new [`RectifiedFlow`] from the given configuration.
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if `dim == 0` or `n_steps == 0`.
    pub fn new(cfg: RectifiedFlowConfig) -> GenResult<Self> {
        if cfg.dim == 0 {
            return Err(GenError::EmptyInput("dim must be >= 1"));
        }
        if cfg.n_steps == 0 {
            return Err(GenError::EmptyInput("n_steps must be >= 1"));
        }
        Ok(Self { cfg })
    }

    /// Validate that a slice has length equal to the configured `dim`.
    #[inline]
    fn check_len(&self, slice: &[f32]) -> GenResult<()> {
        if slice.len() != self.cfg.dim {
            return Err(GenError::DimensionMismatch {
                expected: self.cfg.dim,
                got: slice.len(),
            });
        }
        Ok(())
    }

    /// Linear interpolation path `X_t = (1 − t) · x0 + t · x1` (element-wise).
    ///
    /// For `t = 0` this returns `x0`; for `t = 1` it returns `x1`.
    ///
    /// # Errors
    /// - [`GenError::DimensionMismatch`] if either input differs from `dim`.
    pub fn interpolate(&self, x0: &[f32], x1: &[f32], t: f32) -> GenResult<Vec<f32>> {
        self.check_len(x0)?;
        self.check_len(x1)?;
        let out = x0
            .iter()
            .zip(x1)
            .map(|(&a, &b)| (1.0 - t) * a + t * b)
            .collect();
        Ok(out)
    }

    /// Constant target velocity `x1 − x0` (element-wise).
    ///
    /// This is the regression target for the velocity network: along the linear
    /// path the exact velocity is the time-independent direction `x1 − x0`.
    ///
    /// # Errors
    /// - [`GenError::DimensionMismatch`] if either input differs from `dim`.
    pub fn target_velocity(&self, x0: &[f32], x1: &[f32]) -> GenResult<Vec<f32>> {
        self.check_len(x0)?;
        self.check_len(x1)?;
        let out = x0.iter().zip(x1).map(|(&a, &b)| b - a).collect();
        Ok(out)
    }

    /// Flow matching loss: mean squared error between the predicted velocity and
    /// the constant target `x1 − x0`.
    ///
    /// ```text
    /// L = (1/dim) · Σᵢ (v_pred[i] − (x1[i] − x0[i]))²
    /// ```
    ///
    /// # Errors
    /// - [`GenError::DimensionMismatch`] if any input differs from `dim`.
    pub fn flow_loss(&self, v_pred: &[f32], x0: &[f32], x1: &[f32]) -> GenResult<f32> {
        self.check_len(v_pred)?;
        self.check_len(x0)?;
        self.check_len(x1)?;
        let mut acc = 0.0_f32;
        for ((&vp, &a), &b) in v_pred.iter().zip(x0).zip(x1) {
            let target = b - a;
            let diff = vp - target;
            acc += diff * diff;
        }
        Ok(acc / self.cfg.dim as f32)
    }

    /// Integrate `dX/dt = v(X_t, t)` from `t = 0` to `t = 1` starting at
    /// `x_init`, returning the terminal state.
    ///
    /// With `cfg.heun == false` a forward Euler integrator is used:
    /// ```text
    /// dt = 1/n_steps
    /// for i in 0..n_steps:
    ///     t = i · dt
    ///     v = velocity(X, t)
    ///     X += dt · v
    /// ```
    ///
    /// With `cfg.heun == true` a Heun (RK2 / trapezoidal) corrector is applied:
    /// ```text
    /// v1 = velocity(X, t)
    /// X_pred = X + dt · v1
    /// v2 = velocity(X_pred, t + dt)
    /// X += dt · (v1 + v2) / 2
    /// ```
    ///
    /// For the exact constant field `v ≡ x1 − x0`, the path is a straight line
    /// and either integrator reproduces `x1` exactly regardless of `n_steps`.
    ///
    /// # Errors
    /// - [`GenError::DimensionMismatch`] if `x_init` or any closure output
    ///   differs from `dim`.
    pub fn sample<F>(&self, x_init: &[f32], velocity: F) -> GenResult<Vec<f32>>
    where
        F: Fn(&[f32], f32) -> Vec<f32>,
    {
        self.check_len(x_init)?;
        let dt = 1.0 / self.cfg.n_steps as f32;
        let mut x = x_init.to_vec();
        for i in 0..self.cfg.n_steps {
            let t = i as f32 * dt;
            let v1 = velocity(&x, t);
            self.check_len(&v1)?;
            if self.cfg.heun {
                let x_pred: Vec<f32> = x.iter().zip(&v1).map(|(&xi, &vi)| xi + dt * vi).collect();
                let v2 = velocity(&x_pred, t + dt);
                self.check_len(&v2)?;
                for ((xi, &a), &b) in x.iter_mut().zip(&v1).zip(&v2) {
                    *xi += dt * 0.5 * (a + b);
                }
            } else {
                for (xi, &vi) in x.iter_mut().zip(&v1) {
                    *xi += dt * vi;
                }
            }
        }
        Ok(x)
    }

    /// Generate a reflow pair endpoint `x1_hat` by integrating the current flow
    /// forward from `x0`.
    ///
    /// This is identical to [`Self::sample`] started from `x0`; the resulting
    /// `(x0, x1_hat)` pair is straighter than the original coupling and is used
    /// to retrain the velocity network in the Reflow procedure.
    ///
    /// # Errors
    /// - [`GenError::DimensionMismatch`] if `x0` or any closure output differs
    ///   from `dim`.
    pub fn reflow_pair<F>(&self, x0: &[f32], velocity: F) -> GenResult<Vec<f32>>
    where
        F: Fn(&[f32], f32) -> Vec<f32>,
    {
        self.sample(x0, velocity)
    }

    /// Straightness diagnostic: average squared deviation of the velocity field
    /// from the constant direction `x1 − x0` along the linear path.
    ///
    /// ```text
    /// S = (1/n_steps) · Σ_{s=0}^{n_steps-1} ‖ v(X_{t_s}, t_s) − (x1 − x0) ‖²
    /// ```
    ///
    /// where `t_s = s / n_steps` and `X_{t_s} = (1 − t_s) · x0 + t_s · x1`.
    /// Lower values indicate a straighter flow; for the exact constant field
    /// this is `0`.
    ///
    /// # Errors
    /// - [`GenError::DimensionMismatch`] if `x0`, `x1`, or any closure output
    ///   differs from `dim`.
    pub fn straightness<F>(&self, x0: &[f32], x1: &[f32], velocity: F) -> GenResult<f32>
    where
        F: Fn(&[f32], f32) -> Vec<f32>,
    {
        self.check_len(x0)?;
        self.check_len(x1)?;
        let target = self.target_velocity(x0, x1)?;
        let mut acc = 0.0_f32;
        for s in 0..self.cfg.n_steps {
            let t_s = s as f32 / self.cfg.n_steps as f32;
            let x_ts = self.interpolate(x0, x1, t_s)?;
            let v = velocity(&x_ts, t_s);
            self.check_len(&v)?;
            for (&vi, &ti) in v.iter().zip(&target) {
                let diff = vi - ti;
                acc += diff * diff;
            }
        }
        Ok(acc / self.cfg.n_steps as f32)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    const TINY: f32 = 1e-6;

    fn make_flow(dim: usize, n_steps: usize, heun: bool) -> RectifiedFlow {
        RectifiedFlow::new(RectifiedFlowConfig { dim, n_steps, heun }).unwrap()
    }

    #[test]
    fn interpolate_at_t0_gives_x0() {
        let rf = make_flow(3, 10, false);
        let x0 = vec![1.0_f32, 2.0, 3.0];
        let x1 = vec![4.0_f32, 5.0, 6.0];
        let out = rf.interpolate(&x0, &x1, 0.0).unwrap();
        for (&o, &a) in out.iter().zip(&x0) {
            assert!((o - a).abs() < EPS, "t=0: {o} != {a}");
        }
    }

    #[test]
    fn interpolate_at_t1_gives_x1() {
        let rf = make_flow(3, 10, false);
        let x0 = vec![1.0_f32, 2.0, 3.0];
        let x1 = vec![4.0_f32, 5.0, 6.0];
        let out = rf.interpolate(&x0, &x1, 1.0).unwrap();
        for (&o, &b) in out.iter().zip(&x1) {
            assert!((o - b).abs() < EPS, "t=1: {o} != {b}");
        }
    }

    #[test]
    fn interpolate_at_t_half_is_midpoint() {
        let rf = make_flow(4, 10, false);
        let x0 = vec![0.0_f32, 2.0, -4.0, 10.0];
        let x1 = vec![2.0_f32, 4.0, 0.0, 0.0];
        let out = rf.interpolate(&x0, &x1, 0.5).unwrap();
        let expected = [1.0_f32, 3.0, -2.0, 5.0];
        for (&o, &e) in out.iter().zip(&expected) {
            assert!((o - e).abs() < EPS, "midpoint: {o} != {e}");
        }
    }

    #[test]
    fn target_velocity_is_x1_minus_x0() {
        let rf = make_flow(3, 10, false);
        let x0 = vec![1.0_f32, 2.0, 3.0];
        let x1 = vec![4.0_f32, 6.0, 0.0];
        let v = rf.target_velocity(&x0, &x1).unwrap();
        let expected = [3.0_f32, 4.0, -3.0];
        for (&vi, &e) in v.iter().zip(&expected) {
            assert!((vi - e).abs() < EPS, "target_velocity: {vi} != {e}");
        }
    }

    #[test]
    fn flow_loss_zero_when_pred_equals_target() {
        let rf = make_flow(3, 10, false);
        let x0 = vec![1.0_f32, 2.0, 3.0];
        let x1 = vec![4.0_f32, 6.0, 0.0];
        let v_pred = rf.target_velocity(&x0, &x1).unwrap();
        let loss = rf.flow_loss(&v_pred, &x0, &x1).unwrap();
        assert!(loss.abs() < TINY, "loss should be 0, got {loss}");
    }

    #[test]
    fn flow_loss_mse_hand_example() {
        // x1 - x0 = [1, 1], v_pred = [2, 3] => diffs [1, 2] => (1 + 4)/2 = 2.5
        let rf = make_flow(2, 10, false);
        let x0 = vec![0.0_f32, 0.0];
        let x1 = vec![1.0_f32, 1.0];
        let v_pred = vec![2.0_f32, 3.0];
        let loss = rf.flow_loss(&v_pred, &x0, &x1).unwrap();
        assert!((loss - 2.5).abs() < EPS, "loss should be 2.5, got {loss}");
    }

    #[test]
    fn sample_const_field_recovers_x1_one_step() {
        let rf = make_flow(3, 1, false);
        let x0 = vec![0.0_f32, -1.0, 2.0];
        let x1 = vec![3.0_f32, 5.0, -2.0];
        let target = rf.target_velocity(&x0, &x1).unwrap();
        let out = rf.sample(&x0, |_, _| target.clone()).unwrap();
        for (&o, &b) in out.iter().zip(&x1) {
            assert!((o - b).abs() < EPS, "1-step Euler const field: {o} != {b}");
        }
    }

    #[test]
    fn sample_const_field_recovers_x1_ten_steps() {
        let rf = make_flow(3, 10, false);
        let x0 = vec![0.0_f32, -1.0, 2.0];
        let x1 = vec![3.0_f32, 5.0, -2.0];
        let target = rf.target_velocity(&x0, &x1).unwrap();
        let out = rf.sample(&x0, |_, _| target.clone()).unwrap();
        for (&o, &b) in out.iter().zip(&x1) {
            assert!((o - b).abs() < EPS, "10-step Euler const field: {o} != {b}");
        }
    }

    #[test]
    fn sample_heun_const_field_recovers_x1() {
        let rf = make_flow(3, 5, true);
        let x0 = vec![1.0_f32, 1.0, 1.0];
        let x1 = vec![4.0_f32, -2.0, 0.0];
        let target = rf.target_velocity(&x0, &x1).unwrap();
        let out = rf.sample(&x0, |_, _| target.clone()).unwrap();
        for (&o, &b) in out.iter().zip(&x1) {
            assert!((o - b).abs() < EPS, "Heun const field: {o} != {b}");
        }
    }

    #[test]
    fn straightness_zero_for_const_field() {
        let rf = make_flow(3, 20, false);
        let x0 = vec![0.0_f32, 1.0, -1.0];
        let x1 = vec![2.0_f32, -3.0, 4.0];
        let target = rf.target_velocity(&x0, &x1).unwrap();
        let s = rf.straightness(&x0, &x1, |_, _| target.clone()).unwrap();
        assert!(s.abs() < TINY, "straightness should be 0, got {s}");
    }

    #[test]
    fn straightness_positive_for_curved_field() {
        // A curved field: scale the constant direction by a time-dependent
        // factor that is not identically 1, so the velocity deviates from
        // the constant target along the path.
        let rf = make_flow(2, 16, false);
        let x0 = vec![0.0_f32, 0.0];
        let x1 = vec![1.0_f32, 1.0];
        let target = [1.0_f32, 1.0];
        let s = rf
            .straightness(&x0, &x1, |_, t| {
                // rotation-like time-varying scaling of the direction
                let c = (2.0 * std::f32::consts::PI * t).cos();
                target.iter().map(|&d| c * d).collect()
            })
            .unwrap();
        assert!(
            s > 1e-3,
            "curved field should have positive straightness: {s}"
        );
    }

    #[test]
    fn reflow_pair_equals_sample_from_x0() {
        let rf = make_flow(3, 7, false);
        let x0 = vec![0.5_f32, -0.5, 1.5];
        let velocity = |x: &[f32], _t: f32| x.iter().map(|&xi| 0.3 * xi + 0.1).collect();
        let via_sample = rf.sample(&x0, velocity).unwrap();
        let via_reflow = rf.reflow_pair(&x0, velocity).unwrap();
        for (&a, &b) in via_sample.iter().zip(&via_reflow) {
            assert!((a - b).abs() < TINY, "reflow != sample: {a} vs {b}");
        }
    }

    #[test]
    fn dim_one_works() {
        let rf = make_flow(1, 4, false);
        let x0 = vec![2.0_f32];
        let x1 = vec![5.0_f32];
        let target = rf.target_velocity(&x0, &x1).unwrap();
        let out = rf.sample(&x0, |_, _| target.clone()).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - 5.0).abs() < EPS, "dim=1 recover x1: {}", out[0]);
    }

    #[test]
    fn sample_output_len_equals_dim() {
        let rf = make_flow(5, 3, true);
        let x0 = vec![1.0_f32; 5];
        let out = rf.sample(&x0, |x, _| x.to_vec()).unwrap();
        assert_eq!(out.len(), 5, "output length should equal dim");
    }

    #[test]
    fn err_interpolate_dim_mismatch() {
        let rf = make_flow(3, 10, false);
        let x0 = vec![1.0_f32, 2.0, 3.0];
        let x1 = vec![4.0_f32, 5.0];
        assert!(matches!(
            rf.interpolate(&x0, &x1, 0.5),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_n_steps_zero() {
        let cfg = RectifiedFlowConfig {
            dim: 3,
            n_steps: 0,
            heun: false,
        };
        assert!(matches!(
            RectifiedFlow::new(cfg),
            Err(GenError::EmptyInput(_))
        ));
    }

    #[test]
    fn err_dim_zero() {
        let cfg = RectifiedFlowConfig {
            dim: 0,
            n_steps: 4,
            heun: false,
        };
        assert!(matches!(
            RectifiedFlow::new(cfg),
            Err(GenError::EmptyInput(_))
        ));
    }

    #[test]
    fn err_empty_input_to_sample() {
        let rf = make_flow(3, 4, false);
        let empty: Vec<f32> = Vec::new();
        assert!(matches!(
            rf.sample(&empty, |x, _| x.to_vec()),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_closure_wrong_length() {
        let rf = make_flow(3, 4, false);
        let x0 = vec![1.0_f32, 2.0, 3.0];
        // closure returns a vector of the wrong length
        assert!(matches!(
            rf.sample(&x0, |_, _| vec![0.0_f32, 0.0]),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_straightness_closure_wrong_length() {
        let rf = make_flow(3, 4, false);
        let x0 = vec![1.0_f32, 2.0, 3.0];
        let x1 = vec![4.0_f32, 5.0, 6.0];
        assert!(matches!(
            rf.straightness(&x0, &x1, |_, _| vec![0.0_f32]),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn sample_deterministic() {
        let rf = make_flow(4, 6, true);
        let x0 = vec![0.1_f32, 0.2, 0.3, 0.4];
        let velocity = |x: &[f32], t: f32| x.iter().map(|&xi| xi * 0.5 + t).collect();
        let a = rf.sample(&x0, velocity).unwrap();
        let b = rf.sample(&x0, velocity).unwrap();
        for (&ai, &bi) in a.iter().zip(&b) {
            assert!((ai - bi).abs() < TINY, "non-deterministic: {ai} vs {bi}");
        }
    }

    #[test]
    fn sample_n_steps_one_euler_single_step() {
        // n_steps=1 Euler should equal x_init + 1.0 * velocity(x_init, 0).
        let rf = make_flow(3, 1, false);
        let x0 = vec![1.0_f32, 2.0, 3.0];
        let velocity = |x: &[f32], _t: f32| x.iter().map(|&xi| xi * 2.0).collect::<Vec<_>>();
        let out = rf.sample(&x0, velocity).unwrap();
        // dt = 1, v = [2,4,6] => x = [1+2, 2+4, 3+6] = [3, 6, 9]
        let expected = [3.0_f32, 6.0, 9.0];
        for (&o, &e) in out.iter().zip(&expected) {
            assert!((o - e).abs() < EPS, "single Euler step: {o} != {e}");
        }
    }

    #[test]
    fn flow_loss_dim_mismatch_errors() {
        let rf = make_flow(3, 4, false);
        let v_pred = vec![1.0_f32, 2.0];
        let x0 = vec![0.0_f32, 0.0, 0.0];
        let x1 = vec![1.0_f32, 1.0, 1.0];
        assert!(matches!(
            rf.flow_loss(&v_pred, &x0, &x1),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn straightness_average_normalised_by_n_steps() {
        // Constant deviation field: velocity = target + constant offset c per
        // element. Then each step contributes ‖c‖² and the average is ‖c‖².
        let rf = make_flow(2, 8, false);
        let x0 = vec![0.0_f32, 0.0];
        let x1 = vec![1.0_f32, 1.0]; // target = [1, 1]
        let s = rf
            .straightness(&x0, &x1, |_, _| vec![2.0_f32, 1.0])
            .unwrap();
        // deviation = [2-1, 1-1] = [1, 0] => ‖.‖² = 1 each step => average 1
        assert!((s - 1.0).abs() < EPS, "straightness average: {s}");
    }
}
