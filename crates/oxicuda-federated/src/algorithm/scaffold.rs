//! SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.
//!
//! Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for
//! Federated Learning", ICML 2020.
//!
//! SCAFFOLD uses control variates to correct for client drift caused by
//! heterogeneous local data distributions.

use crate::error::{FedError, FedResult};

/// Server-side state for SCAFFOLD.
#[derive(Debug, Clone)]
pub struct ScaffoldState {
    /// Current global model parameters.
    pub global_params: Vec<f32>,
    /// Server control variate `c` (corrects global drift).
    pub server_control: Vec<f32>,
    /// Number of completed server aggregation rounds.
    pub round: usize,
}

impl ScaffoldState {
    /// Initialize SCAFFOLD state with zero parameters and zero server control.
    #[must_use]
    pub fn new(n_params: usize) -> Self {
        Self {
            global_params: vec![0.0_f32; n_params],
            server_control: vec![0.0_f32; n_params],
            round: 0,
        }
    }

    /// Initialize from existing parameters.
    #[must_use]
    pub fn from_params(params: Vec<f32>) -> Self {
        let n = params.len();
        Self {
            global_params: params,
            server_control: vec![0.0_f32; n],
            round: 0,
        }
    }
}

/// Per-client state for SCAFFOLD.
#[derive(Debug, Clone)]
pub struct ScaffoldClientState {
    /// Client control variate `c_i`.
    pub client_control: Vec<f32>,
}

impl ScaffoldClientState {
    /// Initialize with zero control variate.
    #[must_use]
    pub fn new(n_params: usize) -> Self {
        Self {
            client_control: vec![0.0_f32; n_params],
        }
    }
}

/// Perform one SCAFFOLD client update.
///
/// After local training, computes:
/// - `delta_y = local_params_after − global_params`   (parameter delta)
/// - `c_i_new = c_i − c + delta_y / (K * lr)`         (control variate update, Option 2)
/// - `delta_c = c_i_new − c_i`
///
/// Returns `(delta_y, delta_c)` for server-side aggregation.
///
/// # Arguments
/// - `state` — current server state (global params + server control)
/// - `client_state` — client's local control variate (modified in-place)
/// - `local_params_after` — client model params after local SGD steps
/// - `local_steps` — number of local SGD steps K (must be > 0)
/// - `lr` — client learning rate
///
/// # Errors
/// Returns `DimensionMismatch` if lengths differ or `Internal` if
/// `local_steps == 0` or `lr` is non-positive.
pub fn scaffold_client_update(
    state: &ScaffoldState,
    client_state: &mut ScaffoldClientState,
    local_params_after: &[f32],
    local_steps: usize,
    lr: f32,
) -> FedResult<(Vec<f32>, Vec<f32>)> {
    let n = state.global_params.len();
    if local_params_after.len() != n {
        return Err(FedError::DimensionMismatch {
            expected: n,
            got: local_params_after.len(),
        });
    }
    if client_state.client_control.len() != n {
        return Err(FedError::DimensionMismatch {
            expected: n,
            got: client_state.client_control.len(),
        });
    }
    if local_steps == 0 {
        return Err(FedError::Internal(
            "local_steps must be > 0 for SCAFFOLD update".into(),
        ));
    }
    if !(lr > 0.0 && lr.is_finite()) {
        return Err(FedError::Internal(
            "learning rate must be positive and finite".into(),
        ));
    }

    let k_lr = local_steps as f32 * lr;

    // delta_y = local_after - global
    let delta_y: Vec<f32> = local_params_after
        .iter()
        .zip(state.global_params.iter())
        .map(|(&la, &g)| la - g)
        .collect();

    // Option 2 control update: c_i_new = c_i - c + delta_y / (K * lr)
    // delta_c = c_i_new - c_i = -c + delta_y / (K * lr)
    let mut delta_c = vec![0.0_f32; n];
    for i in 0..n {
        let c_i_new = client_state.client_control[i] - state.server_control[i] + delta_y[i] / k_lr;
        delta_c[i] = c_i_new - client_state.client_control[i];
        client_state.client_control[i] = c_i_new;
    }

    Ok((delta_y, delta_c))
}

/// Perform SCAFFOLD server aggregation.
///
/// Updates:
/// - `global += lr_server * mean(delta_y_i)`
/// - `c += mean(delta_c_i)`  (server control variate update)
///
/// # Errors
/// Returns errors if delta collections are empty, have mismatched lengths,
/// or if `lr_server` is non-positive.
pub fn scaffold_server_aggregate(
    state: &mut ScaffoldState,
    delta_ys: &[Vec<f32>],
    delta_cs: &[Vec<f32>],
    lr_server: f32,
) -> FedResult<()> {
    if delta_ys.is_empty() {
        return Err(FedError::EmptyClientList);
    }
    if delta_ys.len() != delta_cs.len() {
        return Err(FedError::DimensionMismatch {
            expected: delta_ys.len(),
            got: delta_cs.len(),
        });
    }
    if !(lr_server > 0.0 && lr_server.is_finite()) {
        return Err(FedError::Internal(
            "server learning rate must be positive and finite".into(),
        ));
    }
    let n = state.global_params.len();
    let n_clients = delta_ys.len() as f32;

    // Validate dimensions
    for (dy, dc) in delta_ys.iter().zip(delta_cs.iter()) {
        if dy.len() != n {
            return Err(FedError::DimensionMismatch {
                expected: n,
                got: dy.len(),
            });
        }
        if dc.len() != n {
            return Err(FedError::DimensionMismatch {
                expected: n,
                got: dc.len(),
            });
        }
    }

    // Accumulate mean delta_y and mean delta_c
    let mut mean_dy = vec![0.0_f64; n];
    let mut mean_dc = vec![0.0_f64; n];
    for (dy, dc) in delta_ys.iter().zip(delta_cs.iter()) {
        for i in 0..n {
            mean_dy[i] += dy[i] as f64;
            mean_dc[i] += dc[i] as f64;
        }
    }

    let inv_n = 1.0 / n_clients as f64;
    for i in 0..n {
        state.global_params[i] += (lr_server as f64 * mean_dy[i] * inv_n) as f32;
        state.server_control[i] += (mean_dc[i] * inv_n) as f32;
    }

    state.round += 1;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffold_state_new() {
        let s = ScaffoldState::new(4);
        assert_eq!(s.global_params.len(), 4);
        assert!(s.server_control.iter().all(|&v| v == 0.0));
        assert_eq!(s.round, 0);
    }

    #[test]
    fn scaffold_client_update_direction() {
        let state = ScaffoldState::from_params(vec![1.0f32, 2.0, 3.0]);
        let mut client_state = ScaffoldClientState::new(3);
        let local_after = vec![1.5f32, 2.5, 3.5];
        let (dy, _dc) = scaffold_client_update(&state, &mut client_state, &local_after, 5, 0.01)
            .expect("test invariant: valid scaffold update");
        // delta_y should point away from global (local_after > global)
        for &d in &dy {
            assert!(d > 0.0, "delta_y should be positive");
        }
    }

    #[test]
    fn scaffold_client_update_dimension_mismatch() {
        let state = ScaffoldState::new(3);
        let mut client_state = ScaffoldClientState::new(3);
        let local_after = vec![1.0f32, 2.0]; // wrong size
        assert!(matches!(
            scaffold_client_update(&state, &mut client_state, &local_after, 5, 0.01),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn scaffold_client_update_zero_steps_error() {
        let state = ScaffoldState::new(2);
        let mut client_state = ScaffoldClientState::new(2);
        let local_after = vec![1.0f32, 2.0];
        assert!(matches!(
            scaffold_client_update(&state, &mut client_state, &local_after, 0, 0.01),
            Err(FedError::Internal(_))
        ));
    }

    #[test]
    fn scaffold_server_aggregate_updates_global() {
        let mut state = ScaffoldState::new(2);
        let delta_ys = vec![vec![0.1f32, 0.2], vec![0.3f32, 0.4]];
        let delta_cs = vec![vec![0.01f32, 0.02], vec![0.03f32, 0.04]];
        scaffold_server_aggregate(&mut state, &delta_ys, &delta_cs, 1.0)
            .expect("test invariant: valid server aggregate");
        assert_eq!(state.round, 1);
        // mean delta_y = [0.2, 0.3], global += 1.0 * [0.2, 0.3]
        assert!((state.global_params[0] - 0.2).abs() < 1e-5);
        assert!((state.global_params[1] - 0.3).abs() < 1e-5);
    }

    #[test]
    fn scaffold_server_aggregate_empty_error() {
        let mut state = ScaffoldState::new(2);
        assert!(matches!(
            scaffold_server_aggregate(&mut state, &[], &[], 1.0),
            Err(FedError::EmptyClientList)
        ));
    }
}
