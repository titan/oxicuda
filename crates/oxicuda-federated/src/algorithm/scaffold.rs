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

// ─── Drift diagnostics ───────────────────────────────────────────────────────

/// L2 norm `‖c_i − c‖` between a client's control variate and the server's.
///
/// In SCAFFOLD this quantity measures *client drift*: a large value means the
/// client's local objective pulls the model in a direction far from the global
/// consensus, which is exactly what the control variates correct. Monitoring
/// it per round surfaces stragglers / heterogeneous clients.
///
/// # Errors
/// Returns [`FedError::DimensionMismatch`] if the two control variates differ
/// in length.
pub fn control_variate_drift(
    client_state: &ScaffoldClientState,
    state: &ScaffoldState,
) -> FedResult<f32> {
    let c_i = &client_state.client_control;
    let c = &state.server_control;
    if c_i.len() != c.len() {
        return Err(FedError::DimensionMismatch {
            expected: c.len(),
            got: c_i.len(),
        });
    }
    let sq: f64 = c_i
        .iter()
        .zip(c.iter())
        .map(|(&a, &b)| {
            let d = (a - b) as f64;
            d * d
        })
        .sum();
    Ok(sq.sqrt() as f32)
}

/// A fixed-width histogram of gradient (or update) L2 norms across clients,
/// together with summary statistics, for SCAFFOLD drift monitoring.
#[derive(Debug, Clone, PartialEq)]
pub struct DriftDiagnostics {
    /// Per-client values that were binned (e.g. `‖c_i − c‖` or `‖Δy_i‖`).
    pub values: Vec<f32>,
    /// Histogram bin counts, `bins.len() == n_bins`.
    pub bins: Vec<usize>,
    /// Lower edge of the histogram range (the minimum observed value).
    pub min: f32,
    /// Upper edge of the histogram range (the maximum observed value).
    pub max: f32,
    /// Arithmetic mean of `values`.
    pub mean: f32,
    /// Population standard deviation of `values`.
    pub std: f32,
}

impl DriftDiagnostics {
    /// Width of a single histogram bin (`0` when all values are equal).
    #[must_use]
    pub fn bin_width(&self) -> f32 {
        if self.bins.is_empty() {
            return 0.0;
        }
        (self.max - self.min) / self.bins.len() as f32
    }
}

/// Build a [`DriftDiagnostics`] histogram from per-client scalar values.
///
/// Each value is placed into one of `n_bins` equal-width buckets spanning
/// `[min(values), max(values)]`; the maximum value lands in the last bin. Use
/// it on the output of [`control_variate_drift`] across clients, or on a set of
/// pre-computed update norms, to obtain a gradient-norm histogram.
///
/// # Errors
/// Returns [`FedError::EmptyClientList`] if `values` is empty,
/// [`FedError::Internal`] if `n_bins == 0`, or `FedError::NanEncountered`
/// (mapped to `Internal`) if any value is non-finite.
pub fn gradient_norm_histogram(values: &[f32], n_bins: usize) -> FedResult<DriftDiagnostics> {
    if values.is_empty() {
        return Err(FedError::EmptyClientList);
    }
    if n_bins == 0 {
        return Err(FedError::Internal(
            "gradient_norm_histogram requires n_bins > 0".into(),
        ));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(FedError::Internal(
            "gradient_norm_histogram: non-finite value encountered".into(),
        ));
    }

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in values {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }

    let mut bins = vec![0usize; n_bins];
    let span = max - min;
    if span <= 0.0 {
        // All values identical → drop everything in the first bin.
        bins[0] = values.len();
    } else {
        let inv_width = n_bins as f32 / span;
        for &v in values {
            let mut idx = ((v - min) * inv_width) as usize;
            if idx >= n_bins {
                idx = n_bins - 1; // clamp the maximum into the last bin
            }
            bins[idx] += 1;
        }
    }

    let n = values.len() as f64;
    let mean = values.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = values
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;

    Ok(DriftDiagnostics {
        values: values.to_vec(),
        bins,
        min,
        max,
        mean: mean as f32,
        std: var.sqrt() as f32,
    })
}

/// Compute the control-variate drift `‖c_i − c‖` for every client and bin the
/// results into a [`DriftDiagnostics`] histogram in one call.
///
/// # Errors
/// Propagates [`control_variate_drift`] and [`gradient_norm_histogram`] errors;
/// returns [`FedError::EmptyClientList`] if `clients` is empty.
pub fn scaffold_drift_diagnostics(
    clients: &[ScaffoldClientState],
    state: &ScaffoldState,
    n_bins: usize,
) -> FedResult<DriftDiagnostics> {
    if clients.is_empty() {
        return Err(FedError::EmptyClientList);
    }
    let drifts: Vec<f32> = clients
        .iter()
        .map(|c| control_variate_drift(c, state))
        .collect::<FedResult<Vec<f32>>>()?;
    gradient_norm_histogram(&drifts, n_bins)
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

    #[test]
    fn control_variate_drift_matches_l2() {
        let mut state = ScaffoldState::new(3);
        state.server_control = vec![1.0, 1.0, 1.0];
        let mut client = ScaffoldClientState::new(3);
        client.client_control = vec![1.0, 1.0, 4.0]; // diff = (0,0,3) → norm 3
        let d = control_variate_drift(&client, &state).expect("drift");
        assert!((d - 3.0).abs() < 1e-5);
    }

    #[test]
    fn control_variate_drift_zero_when_aligned() {
        let state = ScaffoldState::new(4);
        let client = ScaffoldClientState::new(4);
        let d = control_variate_drift(&client, &state).expect("drift");
        assert!(d.abs() < 1e-6, "aligned controls have zero drift");
    }

    #[test]
    fn gradient_norm_histogram_bins_correctly() {
        // Values 0,1,2,3,4 into 5 bins over [0,4] → one per bin.
        let values = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0];
        let hist = gradient_norm_histogram(&values, 5).expect("hist");
        assert_eq!(hist.bins, vec![1, 1, 1, 1, 1]);
        assert!((hist.min - 0.0).abs() < 1e-6);
        assert!((hist.max - 4.0).abs() < 1e-6);
        assert!((hist.mean - 2.0).abs() < 1e-6);
        // total count preserved
        assert_eq!(hist.bins.iter().sum::<usize>(), values.len());
    }

    #[test]
    fn gradient_norm_histogram_identical_values() {
        let values = vec![2.5_f32; 6];
        let hist = gradient_norm_histogram(&values, 4).expect("hist");
        assert_eq!(hist.bins[0], 6, "identical values fall in the first bin");
        assert!((hist.std).abs() < 1e-6);
    }

    #[test]
    fn gradient_norm_histogram_skewed_distribution() {
        // Most clients have small drift, one is an outlier → last bin gets it.
        let values = vec![0.1_f32, 0.2, 0.1, 0.15, 10.0];
        let hist = gradient_norm_histogram(&values, 5).expect("hist");
        assert_eq!(hist.bins[0], 4, "the four small values cluster in bin 0");
        assert_eq!(*hist.bins.last().expect("last"), 1, "outlier in last bin");
        assert!(hist.std > 0.0);
    }

    #[test]
    fn gradient_norm_histogram_rejects_bad_input() {
        assert!(matches!(
            gradient_norm_histogram(&[], 4),
            Err(FedError::EmptyClientList)
        ));
        assert!(matches!(
            gradient_norm_histogram(&[1.0, 2.0], 0),
            Err(FedError::Internal(_))
        ));
        assert!(matches!(
            gradient_norm_histogram(&[1.0, f32::NAN], 4),
            Err(FedError::Internal(_))
        ));
    }

    #[test]
    fn scaffold_drift_diagnostics_end_to_end() {
        let mut state = ScaffoldState::new(2);
        state.server_control = vec![0.0, 0.0];
        // Three clients with drifts 0, 1, 2 from the (zero) server control.
        let mut c0 = ScaffoldClientState::new(2);
        c0.client_control = vec![0.0, 0.0]; // drift 0
        let mut c1 = ScaffoldClientState::new(2);
        c1.client_control = vec![1.0, 0.0]; // drift 1
        let mut c2 = ScaffoldClientState::new(2);
        c2.client_control = vec![0.0, 2.0]; // drift 2
        let clients = vec![c0, c1, c2];
        let diag = scaffold_drift_diagnostics(&clients, &state, 4).expect("diag");
        assert_eq!(diag.values.len(), 3);
        assert!((diag.min - 0.0).abs() < 1e-5);
        assert!((diag.max - 2.0).abs() < 1e-5);
        assert_eq!(diag.bins.iter().sum::<usize>(), 3);
        assert!(diag.bin_width() > 0.0);
    }

    #[test]
    fn scaffold_drift_diagnostics_empty_clients_error() {
        let state = ScaffoldState::new(2);
        assert!(matches!(
            scaffold_drift_diagnostics(&[], &state, 4),
            Err(FedError::EmptyClientList)
        ));
    }
}
