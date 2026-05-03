//! Flow Matching scheduler.
//!
//! Implements conditional flow matching (Lipman et al. 2022; Albergo & Vanden-Eijnden 2022)
//! with linear and optimal transport paths.

use crate::error::{GenError, GenResult};

// ─── FlowMatchingPath ─────────────────────────────────────────────────────────

/// The interpolation path for flow matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowMatchingPath {
    /// Linear interpolation: `x_t = (1-t)*x_0 + t*x_1`.
    Linear,
    /// Optimal transport path (straight paths for Gaussian marginals).
    OptimalTransport,
}

// ─── FlowMatchingScheduler ────────────────────────────────────────────────────

/// Scheduler for flow matching / continuous normalizing flows.
///
/// Provides interpolation, vector field computation, and ODE solver steps
/// for training and inference with flow matching objectives.
///
/// # Reference
/// Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.
#[derive(Debug, Clone)]
pub struct FlowMatchingScheduler {
    num_steps: usize,
    path: FlowMatchingPath,
    sigma_min: f32,
}

impl FlowMatchingScheduler {
    /// Create a new flow matching scheduler with linear path.
    #[must_use]
    pub fn new(num_steps: usize) -> Self {
        Self {
            num_steps: num_steps.max(1),
            path: FlowMatchingPath::Linear,
            sigma_min: 1e-4,
        }
    }

    /// Create a scheduler with a specified path type.
    #[must_use]
    pub fn with_path(num_steps: usize, path: FlowMatchingPath) -> Self {
        Self {
            num_steps: num_steps.max(1),
            path,
            sigma_min: 1e-4,
        }
    }

    /// Create a scheduler with full parameter control.
    #[must_use]
    pub fn with_params(num_steps: usize, path: FlowMatchingPath, sigma_min: f32) -> Self {
        Self {
            num_steps: num_steps.max(1),
            path,
            sigma_min: sigma_min.max(0.0),
        }
    }

    /// Linear interpolation: `x_t = (1-t)*x_0 + t*x_1`, `t ∈ [0,1]`.
    ///
    /// For `t=0`, returns `x_0`. For `t=1`, returns `x_1`.
    ///
    /// # Errors
    /// - `InvalidFlowTime` if `t ∉ [0, 1]`
    /// - `DimensionMismatch` if shapes differ
    /// - `EmptyInput` if inputs are empty
    pub fn interpolate(&self, x_0: &[f32], x_1: &[f32], t: f32) -> GenResult<Vec<f32>> {
        if x_0.is_empty() {
            return Err(GenError::EmptyInput("x_0 is empty"));
        }
        if !(0.0..=1.0).contains(&t) {
            return Err(GenError::InvalidFlowTime(t));
        }
        if x_0.len() != x_1.len() {
            return Err(GenError::DimensionMismatch {
                expected: x_0.len(),
                got: x_1.len(),
            });
        }
        let result = match self.path {
            FlowMatchingPath::Linear => x_0
                .iter()
                .zip(x_1)
                .map(|(&a, &b)| (1.0 - t) * a + t * b)
                .collect(),
            FlowMatchingPath::OptimalTransport => {
                // OT path with noise scaling: x_t = (1 - (1-σ_min)*t)*x_0 + t*x_1
                let sigma_t = 1.0 - (1.0 - self.sigma_min) * t;
                x_0.iter()
                    .zip(x_1)
                    .map(|(&a, &b)| sigma_t * a + t * b)
                    .collect()
            }
        };
        Ok(result)
    }

    /// Conditional vector field: `v(x_t, t | x_0, x_1) = x_1 - x_0` (linear path).
    ///
    /// For OT path: `v = x_1 - (1-σ_min)*x_0`.
    ///
    /// # Errors
    /// - `DimensionMismatch` if shapes differ
    /// - `EmptyInput` if inputs are empty
    pub fn vector_field(&self, x_0: &[f32], x_1: &[f32]) -> GenResult<Vec<f32>> {
        if x_0.is_empty() {
            return Err(GenError::EmptyInput("x_0 is empty"));
        }
        if x_0.len() != x_1.len() {
            return Err(GenError::DimensionMismatch {
                expected: x_0.len(),
                got: x_1.len(),
            });
        }
        let result = match self.path {
            FlowMatchingPath::Linear => x_0.iter().zip(x_1).map(|(&a, &b)| b - a).collect(),
            FlowMatchingPath::OptimalTransport => x_0
                .iter()
                .zip(x_1)
                .map(|(&a, &b)| b - (1.0 - self.sigma_min) * a)
                .collect(),
        };
        Ok(result)
    }

    /// Euler ODE step: `x_{t+dt} = x_t + dt * v(x_t, t)`.
    ///
    /// # Errors
    /// - `DimensionMismatch` if shapes differ
    /// - `EmptyInput` if inputs are empty
    pub fn euler_step(&self, x_t: &[f32], velocity: &[f32], dt: f32) -> GenResult<Vec<f32>> {
        if x_t.is_empty() {
            return Err(GenError::EmptyInput("x_t is empty"));
        }
        if x_t.len() != velocity.len() {
            return Err(GenError::DimensionMismatch {
                expected: x_t.len(),
                got: velocity.len(),
            });
        }
        let result = x_t
            .iter()
            .zip(velocity)
            .map(|(&x, &v)| x + dt * v)
            .collect();
        Ok(result)
    }

    /// Heun (RK2) step for improved accuracy.
    ///
    /// Uses trapezoidal rule: `x_{t+dt} = x_t + dt * (v_t + v_{t+dt}) / 2`
    ///
    /// # Errors
    /// - `DimensionMismatch` if shapes differ
    /// - `EmptyInput` if inputs are empty
    pub fn heun_step(
        &self,
        x_t: &[f32],
        v_t: &[f32],
        v_tp1: &[f32],
        dt: f32,
    ) -> GenResult<Vec<f32>> {
        if x_t.is_empty() {
            return Err(GenError::EmptyInput("x_t is empty"));
        }
        if x_t.len() != v_t.len() {
            return Err(GenError::DimensionMismatch {
                expected: x_t.len(),
                got: v_t.len(),
            });
        }
        if x_t.len() != v_tp1.len() {
            return Err(GenError::DimensionMismatch {
                expected: x_t.len(),
                got: v_tp1.len(),
            });
        }
        let result = x_t
            .iter()
            .zip(v_t)
            .zip(v_tp1)
            .map(|((&x, &vt), &vt1)| x + dt * 0.5 * (vt + vt1))
            .collect();
        Ok(result)
    }

    /// Compute the ODE trajectory timesteps: `[0, 1/T, 2/T, ..., 1]`.
    ///
    /// Returns `num_steps + 1` values evenly spaced in `[0, 1]`.
    pub fn timesteps(&self) -> Vec<f32> {
        (0..=self.num_steps)
            .map(|i| i as f32 / self.num_steps as f32)
            .collect()
    }

    /// Return the number of ODE steps.
    pub fn num_steps(&self) -> usize {
        self.num_steps
    }

    /// Return the path type.
    pub fn path(&self) -> FlowMatchingPath {
        self.path
    }

    /// Return the minimum noise level σ_min.
    pub fn sigma_min(&self) -> f32 {
        self.sigma_min
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn ones(n: usize) -> Vec<f32> {
        vec![1.0_f32; n]
    }

    fn zeros(n: usize) -> Vec<f32> {
        vec![0.0_f32; n]
    }

    #[test]
    fn interpolate_at_t0_gives_x0() {
        let sched = FlowMatchingScheduler::new(100);
        let x0 = vec![1.0_f32, 2.0, 3.0];
        let x1 = vec![4.0_f32, 5.0, 6.0];
        let out = sched.interpolate(&x0, &x1, 0.0).unwrap();
        for (&o, &a) in out.iter().zip(&x0) {
            assert!((o - a).abs() < EPS, "{o} != {a}");
        }
    }

    #[test]
    fn interpolate_at_t1_gives_x1() {
        let sched = FlowMatchingScheduler::new(100);
        let x0 = vec![1.0_f32, 2.0, 3.0];
        let x1 = vec![4.0_f32, 5.0, 6.0];
        let out = sched.interpolate(&x0, &x1, 1.0).unwrap();
        for (&o, &b) in out.iter().zip(&x1) {
            assert!((o - b).abs() < EPS, "{o} != {b}");
        }
    }

    #[test]
    fn interpolate_at_t_half() {
        let sched = FlowMatchingScheduler::new(100);
        let x0 = zeros(4);
        let x1 = ones(4);
        let out = sched.interpolate(&x0, &x1, 0.5).unwrap();
        for &v in &out {
            assert!((v - 0.5).abs() < EPS, "midpoint should be 0.5, got {v}");
        }
    }

    #[test]
    fn interpolate_invalid_t() {
        let sched = FlowMatchingScheduler::new(100);
        let x0 = zeros(4);
        let x1 = ones(4);
        assert!(matches!(
            sched.interpolate(&x0, &x1, 1.5),
            Err(GenError::InvalidFlowTime(_))
        ));
        assert!(matches!(
            sched.interpolate(&x0, &x1, -0.1),
            Err(GenError::InvalidFlowTime(_))
        ));
    }

    #[test]
    fn vector_field_linear_is_x1_minus_x0() {
        let sched = FlowMatchingScheduler::new(100);
        let x0 = vec![1.0_f32, 2.0, 3.0];
        let x1 = vec![4.0_f32, 5.0, 6.0];
        let v = sched.vector_field(&x0, &x1).unwrap();
        for (&vi, (&a, &b)) in v.iter().zip(x0.iter().zip(&x1)) {
            assert!((vi - (b - a)).abs() < EPS, "{vi} != {}", b - a);
        }
    }

    #[test]
    fn euler_step_correctness() {
        let sched = FlowMatchingScheduler::new(10);
        let x_t = vec![0.0_f32, 0.0, 0.0];
        let velocity = vec![1.0_f32, 2.0, 3.0];
        let dt = 0.1;
        let out = sched.euler_step(&x_t, &velocity, dt).unwrap();
        let expected = vec![0.1, 0.2, 0.3];
        for (&o, &e) in out.iter().zip(&expected) {
            assert!((o - e).abs() < EPS, "{o} != {e}");
        }
    }

    #[test]
    fn heun_step_symmetric() {
        // With v_t = v_{t+1}, heun should equal euler
        let sched = FlowMatchingScheduler::new(10);
        let x_t = vec![1.0_f32, 2.0, 3.0];
        let v = vec![0.5_f32, -0.5, 1.0];
        let dt = 0.1;
        let euler = sched.euler_step(&x_t, &v, dt).unwrap();
        let heun = sched.heun_step(&x_t, &v, &v, dt).unwrap();
        for (&e, &h) in euler.iter().zip(&heun) {
            assert!((e - h).abs() < EPS, "heun != euler when v_t == v_t+1");
        }
    }

    #[test]
    fn timesteps_boundary_values() {
        let sched = FlowMatchingScheduler::new(10);
        let ts = sched.timesteps();
        assert_eq!(ts.len(), 11, "should have 11 values for 10 steps");
        assert!((ts[0] - 0.0).abs() < EPS, "first timestep should be 0");
        assert!((ts[10] - 1.0).abs() < EPS, "last timestep should be 1");
    }

    #[test]
    fn timesteps_uniformly_spaced() {
        let sched = FlowMatchingScheduler::new(10);
        let ts = sched.timesteps();
        for w in ts.windows(2) {
            assert!(
                (w[1] - w[0] - 0.1).abs() < EPS,
                "non-uniform spacing: {} - {}",
                w[1],
                w[0]
            );
        }
    }

    #[test]
    fn ot_path_interpolation_different_from_linear() {
        let linear = FlowMatchingScheduler::new(100);
        let ot = FlowMatchingScheduler::with_path(100, FlowMatchingPath::OptimalTransport);
        let x0 = vec![1.0_f32, 2.0, 3.0];
        let x1 = vec![4.0_f32, 5.0, 6.0];
        let out_lin = linear.interpolate(&x0, &x1, 0.5).unwrap();
        let out_ot = ot.interpolate(&x0, &x1, 0.5).unwrap();
        // They should differ due to sigma_min correction
        let diff: f32 = out_lin
            .iter()
            .zip(&out_ot)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(diff > 1e-6, "OT and linear should differ: diff={diff}");
    }

    #[test]
    fn euler_roundtrip_const_velocity() {
        // With constant velocity v = x1 - x0, Euler from t=0 to t=1 should give x1
        let sched = FlowMatchingScheduler::new(100);
        let x0 = vec![0.0_f32, 0.0];
        let x1 = vec![1.0_f32, 1.0];
        let v = sched.vector_field(&x0, &x1).unwrap();
        let mut x = x0.clone();
        let dt = 1.0 / 100.0;
        for _ in 0..100 {
            x = sched.euler_step(&x, &v, dt).unwrap();
        }
        for (&xi, &target) in x.iter().zip(&x1) {
            assert!(
                (xi - target).abs() < 1e-4,
                "Euler roundtrip failed: {xi} != {target}"
            );
        }
    }
}
