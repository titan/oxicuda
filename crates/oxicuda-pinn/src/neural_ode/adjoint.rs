//! Continuous adjoint method for Neural ODE gradient computation.

use crate::error::{PinnError, PinnResult};
use crate::neural_ode::solvers::{OdeRhsFn, integrate_fixed, rk4_step};

/// Configuration for a Neural ODE layer.
pub struct NeuralOdeConfig {
    pub dim: usize,
    pub n_params: usize,
}

/// Trajectory checkpoint type: list of `(t, y)` pairs.
pub type Trajectory = Vec<(f32, Vec<f32>)>;

/// Forward integrate ODE from t0 to t1, storing the trajectory.
///
/// Returns `(y_final, checkpoints)` where `checkpoints` is a list of `(t, y)` pairs.
pub fn node_forward(
    rhs: OdeRhsFn,
    t0: f32,
    t1: f32,
    y0: &[f32],
    h: f32,
) -> PinnResult<(Vec<f32>, Trajectory)> {
    let (times, states) = integrate_fixed(rhs, t0, t1, y0, h)?;
    let checkpoints: Trajectory = times.into_iter().zip(states.clone()).collect();
    let y_final = states.last().cloned().unwrap_or_else(|| y0.to_vec());
    Ok((y_final, checkpoints))
}

/// Compute `dL/dθ` via the continuous adjoint method.
///
/// Algorithm:
/// 1. Start with `a(t1) = dL/dy(t1)` (initial adjoint state).
/// 2. Integrate backward: `da/dt = -aᵀ · ∂f/∂y`.
/// 3. Accumulate `dL/dθ = -∫ aᵀ · ∂f/∂θ dt`.
///
/// The Jacobian `dfdy_fn` returns a flat `[dim × dim]` matrix (row-major).
/// The `dfdtheta_fn` returns a flat `[n_params]` vector.
pub fn node_adjoint_grad(
    rhs: OdeRhsFn,
    dfdy_fn: &dyn Fn(f32, &[f32]) -> Vec<f32>,
    dfdtheta_fn: &dyn Fn(f32, &[f32]) -> Vec<f32>,
    trajectory: &[(f32, Vec<f32>)],
    dl_dy_final: &[f32],
    h: f32,
) -> PinnResult<Vec<f32>> {
    if trajectory.is_empty() {
        return Err(PinnError::EmptyInput);
    }
    let dim = dl_dy_final.len();
    let n_params = {
        let t0_traj = trajectory[0].0;
        let y0_traj = &trajectory[0].1;
        dfdtheta_fn(t0_traj, y0_traj).len()
    };

    // Adjoint state a(t1) = dL/dy(t1)
    let mut a = dl_dy_final.to_vec();
    // dL/dtheta accumulator
    let mut dl_dtheta = vec![0.0_f32; n_params];

    // Traverse trajectory in reverse (backward time integration)
    let n = trajectory.len();
    for k in (0..n.saturating_sub(1)).rev() {
        let (t_next, y_next) = &trajectory[k + 1];
        let (t_cur, _y_cur) = &trajectory[k];
        let dt = t_cur - t_next; // negative (backward)

        // Adjoint ODE: da/dt = -a^T * df/dy
        // We do one RK4 backward step with effective step dt (negative → backward)
        // Use magnitude h for sub-stepping
        let t_start = *t_next;
        let t_end = *t_cur;

        // Build adjoint RHS: da/dt = -J^T * a  (where J = df/dy)
        let a_adj = a.clone();
        let y_adj = y_next.clone();

        let adjoint_rhs = |t_loc: f32, a_loc: &[f32], dadt: &mut [f32]| {
            let jac = dfdy_fn(t_loc, &y_adj);
            // da/dt[i] = -sum_j J[j,i] * a[j]  (J^T * a)
            for i in 0..dim {
                let mut s = 0.0_f32;
                for j in 0..dim {
                    if j * dim + i < jac.len() {
                        s += jac[j * dim + i] * a_loc[j];
                    }
                }
                dadt[i] = -s;
            }
            // also compute one RHS eval to get f at this point (for chain rule)
            let _ = rhs;
        };

        // Integrate adjoint backward with fixed-step RK4
        let h_back = (t_end - t_start).abs().min(h.abs());
        if h_back < 1e-12 {
            continue;
        }
        // Single RK4 step backward (step = t_end - t_start which is negative)
        let step_sign = if t_end >= t_start { 1.0_f32 } else { -1.0_f32 };
        let h_eff = h_back * step_sign;
        // But we're integrating from t_next to t_cur; since k is going backward
        // t_cur < t_next so dt = t_cur - t_next < 0, so h_eff is negative → backward
        let h_signed = dt.signum() * h_back;
        let a_new = rk4_step(&adjoint_rhs, t_start, &a_adj, h_signed);
        let _ = h_eff;

        if a_new.iter().any(|v| !v.is_finite()) {
            return Err(PinnError::SolverDivergence {
                reason: "NaN in adjoint backward pass",
            });
        }

        // Accumulate dL/dθ: += -∫ a^T * df/dθ dt ≈ -a^T * df/dθ * dt
        let dfdth = dfdtheta_fn(t_start, &y_adj);
        if dfdth.len() != n_params {
            return Err(PinnError::DimensionMismatch {
                expected: n_params,
                got: dfdth.len(),
            });
        }
        for p in 0..n_params {
            let mut s = 0.0_f32;
            for j in 0..dim {
                if p < dfdth.len() {
                    // approximate: a^T * (df/dθ columns for param p)
                    // Here dfdtheta returns [n_params], interpret as sum over dim
                    s += a_adj[j.min(dim - 1)] * dfdth[p] / dim as f32;
                }
            }
            dl_dtheta[p] -= s * (t_end - t_start).abs();
        }

        a = a_new;
    }

    Ok(dl_dtheta)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exp_decay(_t: f32, y: &[f32], dydt: &mut [f32]) {
        dydt[0] = -y[0];
    }

    #[test]
    fn node_forward_shape() {
        let (y_final, checkpoints) = node_forward(&exp_decay, 0.0, 1.0, &[1.0], 0.1).unwrap();
        assert_eq!(y_final.len(), 1);
        assert!(!checkpoints.is_empty());
    }

    #[test]
    fn node_forward_exp_decay_accurate() {
        let (y_final, _) = node_forward(&exp_decay, 0.0, 1.0, &[1.0], 0.01).unwrap();
        let expected = (-1.0_f32).exp();
        assert!((y_final[0] - expected).abs() < 1e-4);
    }

    #[test]
    fn adjoint_grad_returns_correct_shape() {
        let (_, traj) = node_forward(&exp_decay, 0.0, 1.0, &[1.0], 0.1).unwrap();
        let dfdy = |_t: f32, _y: &[f32]| vec![-1.0_f32]; // df/dy = -1 for dy/dt = -y
        let dfdth = |_t: f32, _y: &[f32]| vec![0.0_f32; 2]; // 2 dummy params
        let dl_dth = node_adjoint_grad(&exp_decay, &dfdy, &dfdth, &traj, &[1.0], 0.1).unwrap();
        assert_eq!(dl_dth.len(), 2);
    }

    #[test]
    fn adjoint_grad_finite_values() {
        let (_, traj) = node_forward(&exp_decay, 0.0, 0.5, &[1.0], 0.05).unwrap();
        let dfdy = |_t: f32, _y: &[f32]| vec![-1.0_f32];
        let dfdth = |_t: f32, _y: &[f32]| vec![1.0_f32];
        let dl_dth = node_adjoint_grad(&exp_decay, &dfdy, &dfdth, &traj, &[1.0], 0.05).unwrap();
        assert!(
            dl_dth.iter().all(|v| v.is_finite()),
            "dL/dθ not finite: {:?}",
            dl_dth
        );
    }

    #[test]
    fn adjoint_empty_traj_error() {
        let dfdy = |_t: f32, _y: &[f32]| vec![-1.0_f32];
        let dfdth = |_t: f32, _y: &[f32]| vec![0.0_f32];
        let result = node_adjoint_grad(&exp_decay, &dfdy, &dfdth, &[], &[1.0], 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn node_forward_invalid_h_error() {
        let result = node_forward(&exp_decay, 0.0, 1.0, &[1.0], 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn node_forward_multidim() {
        fn two_dim(_t: f32, y: &[f32], dy: &mut [f32]) {
            dy[0] = -y[0];
            dy[1] = -2.0 * y[1];
        }
        let (y_final, _) = node_forward(&two_dim, 0.0, 1.0, &[1.0, 1.0], 0.01).unwrap();
        assert_eq!(y_final.len(), 2);
        assert!((y_final[0] - (-1.0_f32).exp()).abs() < 1e-4);
        assert!((y_final[1] - (-2.0_f32).exp()).abs() < 1e-3);
    }

    #[test]
    fn adjoint_2d_returns_correct_n_params() {
        fn two_dim_ode(_t: f32, y: &[f32], dy: &mut [f32]) {
            dy[0] = -y[0];
            dy[1] = -y[1];
        }
        let (_, traj) = node_forward(&two_dim_ode, 0.0, 0.5, &[1.0, 1.0], 0.1).unwrap();
        let dfdy = |_t: f32, _y: &[f32]| vec![-1.0_f32, 0.0, 0.0, -1.0]; // 2x2 identity * -1
        let dfdth = |_t: f32, _y: &[f32]| vec![0.1_f32, 0.2_f32, 0.3_f32]; // 3 params
        let result = node_adjoint_grad(&two_dim_ode, &dfdy, &dfdth, &traj, &[1.0, 1.0], 0.1);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3);
    }
}
