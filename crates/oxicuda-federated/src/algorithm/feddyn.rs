//! FedDyn: Federated learning with dynamic regularization.
//!
//! Acar et al., "Federated Learning Based on Dynamic Regularization",
//! ICLR 2021.
//!
//! FedDyn removes the *objective inconsistency* of FedAvg: even when clients run
//! their local solver to optimality, the global stationary point coincides with
//! the true global minimiser. Each client `k` augments its local loss with a
//! **dynamic** linear term (a running gradient state `h_k`) plus a quadratic
//! proximal anchor to the current server model:
//!
//! ```text
//! R_k(θ) = L_k(θ) − ⟨h_k, θ⟩ + (α/2)·‖θ − θ_server‖²
//! ```
//!
//! After local optimisation produces `θ_k`, the client refreshes its state
//! `h_k ← h_k − α·(θ_k − θ_server)`. The server keeps an aggregate state `h`
//! and updates
//!
//! ```text
//! h     ← h − (α/m)·Σ_{k∈P}(θ_k − θ_server_old)
//! θ_new ← mean_{k∈P}(θ_k) − (1/α)·h
//! ```
//!
//! where `m` is the **total** number of devices and `P` the participating set
//! (full participation, `|P| = m`, is the common case and the one exercised by
//! the analytic convergence test).

use crate::error::{FedError, FedResult};

/// Configuration for FedDyn training.
#[derive(Debug, Clone)]
pub struct FedDynConfig {
    /// Total number of devices `m` in the federation.
    pub n_clients: usize,
    /// Dynamic-regularization coefficient `α > 0`.
    pub alpha: f32,
    /// Client local learning rate `> 0` (for gradient-based local solvers).
    pub learning_rate: f32,
    /// Number of local optimisation epochs.
    pub local_epochs: usize,
}

impl FedDynConfig {
    /// Create a validated FedDyn configuration.
    ///
    /// # Errors
    /// - [`FedError::EmptyClientList`] if `n_clients == 0`.
    /// - [`FedError::InvalidProximalMu`] if `alpha` is not positive/finite.
    /// - [`FedError::InvalidWeight`] if `learning_rate` is not positive/finite.
    pub fn new(
        n_clients: usize,
        alpha: f32,
        learning_rate: f32,
        local_epochs: usize,
    ) -> FedResult<Self> {
        if n_clients == 0 {
            return Err(FedError::EmptyClientList);
        }
        if !(alpha > 0.0 && alpha.is_finite()) {
            return Err(FedError::InvalidProximalMu);
        }
        if !(learning_rate > 0.0 && learning_rate.is_finite()) {
            return Err(FedError::InvalidWeight {
                weight: learning_rate,
            });
        }
        Ok(Self {
            n_clients,
            alpha,
            learning_rate,
            local_epochs,
        })
    }
}

/// Server-side state for FedDyn.
#[derive(Debug, Clone)]
pub struct FedDynState {
    /// Current global model parameters `θ_server`.
    pub global_params: Vec<f32>,
    /// Server aggregate gradient state `h` (same dimension as the model).
    pub h: Vec<f32>,
    /// Number of completed aggregation rounds.
    pub round: usize,
}

impl FedDynState {
    /// Initialise a zeroed FedDyn server state with `n_params` parameters.
    #[must_use]
    pub fn new(n_params: usize) -> Self {
        Self {
            global_params: vec![0.0_f32; n_params],
            h: vec![0.0_f32; n_params],
            round: 0,
        }
    }

    /// Initialise from existing global parameters with a zero `h` state.
    #[must_use]
    pub fn from_params(params: Vec<f32>) -> Self {
        let n = params.len();
        Self {
            global_params: params,
            h: vec![0.0_f32; n],
            round: 0,
        }
    }
}

/// Per-client persistent state: the running gradient term `h_k`.
#[derive(Debug, Clone)]
pub struct FedDynClientState {
    /// Unique client identifier.
    pub client_id: usize,
    /// Running local gradient state `h_k` (same dimension as the model).
    pub local_grad: Vec<f32>,
}

impl FedDynClientState {
    /// Initialise a zeroed client state.
    #[must_use]
    pub fn new(client_id: usize, n_params: usize) -> Self {
        Self {
            client_id,
            local_grad: vec![0.0_f32; n_params],
        }
    }
}

/// Stateless handle providing the FedDyn primitives.
pub struct FedDyn;

impl FedDyn {
    /// Gradient of the dynamically regularised local objective `R_k`.
    ///
    /// Given the client's base loss gradient `∇L_k(θ)` (`base_grad`), the running
    /// state `h_k`, the current point `θ`, and the server anchor `θ_server`,
    /// returns `∇R_k(θ) = ∇L_k(θ) − h_k + α·(θ − θ_server)`.
    ///
    /// # Errors
    /// [`FedError::DimensionMismatch`] if the slices differ in length.
    pub fn regularized_gradient(
        base_grad: &[f32],
        local_grad_state: &[f32],
        theta: &[f32],
        theta_server: &[f32],
        alpha: f32,
    ) -> FedResult<Vec<f32>> {
        let n = base_grad.len();
        if local_grad_state.len() != n || theta.len() != n || theta_server.len() != n {
            return Err(FedError::DimensionMismatch {
                expected: n,
                got: local_grad_state.len(),
            });
        }
        let mut out = vec![0.0_f32; n];
        for (((o, &g), &h), (&t, &ts)) in out
            .iter_mut()
            .zip(base_grad.iter())
            .zip(local_grad_state.iter())
            .zip(theta.iter().zip(theta_server.iter()))
        {
            *o = g - h + alpha * (t - ts);
        }
        Ok(out)
    }

    /// Refresh a client's running gradient state after local optimisation:
    /// `h_k ← h_k − α·(θ_k − θ_server)`.
    ///
    /// # Errors
    /// [`FedError::DimensionMismatch`] if any slice length disagrees with `h_k`.
    pub fn update_local_grad_state(
        state: &mut FedDynClientState,
        theta_local: &[f32],
        theta_server: &[f32],
        alpha: f32,
    ) -> FedResult<()> {
        let n = state.local_grad.len();
        if theta_local.len() != n || theta_server.len() != n {
            return Err(FedError::DimensionMismatch {
                expected: n,
                got: theta_local.len(),
            });
        }
        for (h, (&t, &ts)) in state
            .local_grad
            .iter_mut()
            .zip(theta_local.iter().zip(theta_server.iter()))
        {
            *h -= alpha * (t - ts);
        }
        Ok(())
    }

    /// Closed-form minimiser of `R_k` for a separable quadratic local loss
    /// `L_k(θ) = Σ_i (q_i/2)·(θ_i − a_i)²`.
    ///
    /// Per coordinate the regularised objective is quadratic with stationary
    /// point `θ_i = (q_i·a_i + h_{k,i} + α·θ_server,i) / (q_i + α)`. Useful as a
    /// strong (exact) local solver for analytic tests and as a reference for
    /// gradient-based solvers.
    ///
    /// # Errors
    /// - [`FedError::DimensionMismatch`] if the slices differ in length.
    /// - [`FedError::InvalidProximalMu`] if `alpha` is not positive/finite.
    /// - [`FedError::InvalidWeight`] if any `q_i + α` is non-positive.
    pub fn solve_quadratic_local(
        curvature: &[f32],
        optimum: &[f32],
        local_grad_state: &[f32],
        theta_server: &[f32],
        alpha: f32,
    ) -> FedResult<Vec<f32>> {
        let n = curvature.len();
        if optimum.len() != n || local_grad_state.len() != n || theta_server.len() != n {
            return Err(FedError::DimensionMismatch {
                expected: n,
                got: optimum.len(),
            });
        }
        if !(alpha > 0.0 && alpha.is_finite()) {
            return Err(FedError::InvalidProximalMu);
        }
        let mut out = vec![0.0_f32; n];
        for ((o, (&q, &a)), (&h, &ts)) in out
            .iter_mut()
            .zip(curvature.iter().zip(optimum.iter()))
            .zip(local_grad_state.iter().zip(theta_server.iter()))
        {
            let denom = q + alpha;
            if !(denom > 0.0 && denom.is_finite()) {
                return Err(FedError::InvalidWeight { weight: denom });
            }
            *o = (q * a + h + alpha * ts) / denom;
        }
        Ok(out)
    }

    /// One round of FedDyn server aggregation (full or partial participation).
    ///
    /// `client_params` are the locally optimised models `θ_k` of the
    /// **participating** clients. The server state is updated in place:
    ///
    /// ```text
    /// h     ← h − (α/m)·Σ_k(θ_k − θ_server_old)
    /// θ_new ← mean_k(θ_k) − (1/α)·h
    /// ```
    ///
    /// where `m = cfg.n_clients` is the total device count. For full
    /// participation `|P| = m`.
    ///
    /// # Errors
    /// - [`FedError::EmptyClientList`] if `client_params` is empty.
    /// - [`FedError::DimensionMismatch`] if any client vector length differs from
    ///   the global model.
    /// - [`FedError::InsufficientClients`] if more participants than `n_clients`.
    pub fn aggregate(
        state: &mut FedDynState,
        client_params: &[Vec<f32>],
        cfg: &FedDynConfig,
    ) -> FedResult<()> {
        if client_params.is_empty() {
            return Err(FedError::EmptyClientList);
        }
        let n_params = state.global_params.len();
        for theta in client_params {
            if theta.len() != n_params {
                return Err(FedError::DimensionMismatch {
                    expected: n_params,
                    got: theta.len(),
                });
            }
        }
        let participants = client_params.len();
        if participants > cfg.n_clients {
            return Err(FedError::InsufficientClients {
                min: participants,
                got: cfg.n_clients,
            });
        }

        let alpha = cfg.alpha as f64;
        let m = cfg.n_clients as f64;
        let inv_p = 1.0 / participants as f64;
        let theta_old = state.global_params.clone();

        // Σ_k(θ_k − θ_old) and Σ_k θ_k.
        let mut delta_sum = vec![0.0_f64; n_params];
        let mut theta_sum = vec![0.0_f64; n_params];
        for theta in client_params {
            for ((d, s), (&t, &old)) in delta_sum
                .iter_mut()
                .zip(theta_sum.iter_mut())
                .zip(theta.iter().zip(theta_old.iter()))
            {
                *d += (t as f64) - (old as f64);
                *s += t as f64;
            }
        }

        // h ← h − (α/m)·Σ(θ_k − θ_old)
        for (h, &d) in state.h.iter_mut().zip(delta_sum.iter()) {
            *h = (*h as f64 - (alpha / m) * d) as f32;
        }
        // θ_new ← mean(θ_k) − (1/α)·h
        for ((param, &s), &h) in state
            .global_params
            .iter_mut()
            .zip(theta_sum.iter())
            .zip(state.h.iter())
        {
            *param = (s * inv_p - (h as f64) / alpha) as f32;
        }

        state.round += 1;
        Ok(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validation() {
        assert!(matches!(
            FedDynConfig::new(0, 1.0, 0.1, 1),
            Err(FedError::EmptyClientList)
        ));
        assert!(matches!(
            FedDynConfig::new(3, 0.0, 0.1, 1),
            Err(FedError::InvalidProximalMu)
        ));
        assert!(matches!(
            FedDynConfig::new(3, -1.0, 0.1, 1),
            Err(FedError::InvalidProximalMu)
        ));
        assert!(matches!(
            FedDynConfig::new(3, 1.0, 0.0, 1),
            Err(FedError::InvalidWeight { .. })
        ));
        assert!(FedDynConfig::new(3, 1.0, 0.1, 5).is_ok());
    }

    #[test]
    fn state_new_zeros() {
        let s = FedDynState::new(4);
        assert_eq!(s.global_params.len(), 4);
        assert_eq!(s.h.len(), 4);
        assert!(s.global_params.iter().all(|&v| v == 0.0));
        assert!(s.h.iter().all(|&v| v == 0.0));
        assert_eq!(s.round, 0);
    }

    #[test]
    fn regularized_gradient_formula() {
        // base_grad − h_k + α(θ − θ_server)
        let base = vec![1.0_f32, 2.0];
        let h = vec![0.5_f32, -0.5];
        let theta = vec![3.0_f32, 1.0];
        let server = vec![1.0_f32, 1.0];
        let g = FedDyn::regularized_gradient(&base, &h, &theta, &server, 2.0).expect("grad");
        // dim0: 1 − 0.5 + 2·(3−1) = 0.5 + 4 = 4.5
        // dim1: 2 − (−0.5) + 2·(1−1) = 2.5
        assert!((g[0] - 4.5).abs() < 1e-5);
        assert!((g[1] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn regularized_gradient_dim_mismatch() {
        assert!(matches!(
            FedDyn::regularized_gradient(&[1.0, 2.0], &[1.0], &[1.0, 2.0], &[1.0, 2.0], 1.0),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn update_local_grad_state_formula() {
        let mut st = FedDynClientState::new(0, 2);
        // h_k ← 0 − α(θ_k − θ_server); α=1, θ_k=[2,4], server=[0,1]
        FedDyn::update_local_grad_state(&mut st, &[2.0, 4.0], &[0.0, 1.0], 1.0).expect("update");
        assert!((st.local_grad[0] - (-2.0)).abs() < 1e-5);
        assert!((st.local_grad[1] - (-3.0)).abs() < 1e-5);
    }

    #[test]
    fn solve_quadratic_local_formula() {
        // θ_i = (q·a + h + α·θ_server) / (q + α)
        // q=2, a=3, h=0, server=0, α=2 → (6+0+0)/4 = 1.5
        let sol =
            FedDyn::solve_quadratic_local(&[2.0], &[3.0], &[0.0], &[0.0], 2.0).expect("solve");
        assert!((sol[0] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn aggregate_single_round_formula() {
        // m=2 clients, full participation, α=1, θ_old=0.
        // θ_0=[2,0], θ_1=[0,4].
        // Σδ = [2,4]; h = 0 − (1/2)·[2,4] = [−1,−2]
        // mean θ = [1,2]; θ_new = [1,2] − h = [1−(−1), 2−(−2)] = [2,4]
        let mut state = FedDynState::from_params(vec![0.0_f32, 0.0]);
        let cfg = FedDynConfig::new(2, 1.0, 0.1, 1).expect("cfg");
        FedDyn::aggregate(&mut state, &[vec![2.0, 0.0], vec![0.0, 4.0]], &cfg).expect("agg");
        assert!((state.h[0] - (-1.0)).abs() < 1e-5);
        assert!((state.h[1] - (-2.0)).abs() < 1e-5);
        assert!((state.global_params[0] - 2.0).abs() < 1e-5);
        assert!((state.global_params[1] - 4.0).abs() < 1e-5);
        assert_eq!(state.round, 1);
    }

    #[test]
    fn aggregate_empty_errors() {
        let mut state = FedDynState::new(3);
        let cfg = FedDynConfig::new(3, 1.0, 0.1, 1).expect("cfg");
        assert!(matches!(
            FedDyn::aggregate(&mut state, &[], &cfg),
            Err(FedError::EmptyClientList)
        ));
    }

    #[test]
    fn aggregate_dim_mismatch_errors() {
        let mut state = FedDynState::new(3);
        let cfg = FedDynConfig::new(3, 1.0, 0.1, 1).expect("cfg");
        assert!(matches!(
            FedDyn::aggregate(&mut state, &[vec![1.0, 2.0]], &cfg),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn aggregate_too_many_participants_errors() {
        let mut state = FedDynState::new(2);
        let cfg = FedDynConfig::new(1, 1.0, 0.1, 1).expect("cfg");
        let updates = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
        assert!(matches!(
            FedDyn::aggregate(&mut state, &updates, &cfg),
            Err(FedError::InsufficientClients { .. })
        ));
    }

    /// Drive a full FedDyn simulation with exact local solvers on heterogeneous
    /// quadratics and return the converged global model.
    fn run_simulation(
        curvatures: &[Vec<f32>],
        optima: &[Vec<f32>],
        alpha: f32,
        rounds: usize,
    ) -> Vec<f32> {
        let n_clients = curvatures.len();
        let n_params = curvatures[0].len();
        let cfg = FedDynConfig::new(n_clients, alpha, 0.1, 1).expect("cfg");
        let mut state = FedDynState::new(n_params);
        let mut clients: Vec<FedDynClientState> = (0..n_clients)
            .map(|i| FedDynClientState::new(i, n_params))
            .collect();

        for _round in 0..rounds {
            let server = state.global_params.clone();
            let mut thetas = Vec::with_capacity(n_clients);
            for (k, client) in clients.iter_mut().enumerate() {
                // Exact local minimisation of the regularised objective.
                let theta_k = FedDyn::solve_quadratic_local(
                    &curvatures[k],
                    &optima[k],
                    &client.local_grad,
                    &server,
                    alpha,
                )
                .expect("local solve");
                FedDyn::update_local_grad_state(client, &theta_k, &server, alpha)
                    .expect("state update");
                thetas.push(theta_k);
            }
            FedDyn::aggregate(&mut state, &thetas, &cfg).expect("aggregate");
        }
        state.global_params
    }

    #[test]
    fn converges_to_curvature_weighted_minimum() {
        // Heterogeneous quadratics: the true global minimiser is the
        // curvature-weighted mean Σ q_k a_k / Σ q_k — NOT the plain mean of a_k.
        let curvatures = vec![vec![1.0_f32, 1.0], vec![3.0_f32, 3.0], vec![1.0_f32, 1.0]];
        let optima = vec![vec![0.0_f32, 0.0], vec![4.0_f32, 2.0], vec![-1.0_f32, 5.0]];
        // dim0: (1·0 + 3·4 + 1·−1)/5 = 11/5 = 2.2 ; plain mean = (0+4−1)/3 = 1.0
        // dim1: (1·0 + 3·2 + 1·5)/5 = 11/5 = 2.2 ; plain mean = (0+2+5)/3 ≈ 2.333
        let result = run_simulation(&curvatures, &optima, 1.0, 300);
        assert!((result[0] - 2.2).abs() < 2e-2, "dim0={}", result[0]);
        assert!((result[1] - 2.2).abs() < 2e-2, "dim1={}", result[1]);
        // Distinct from the plain (objective-inconsistent FedAvg-of-optima) mean.
        assert!(
            (result[0] - 1.0).abs() > 0.5,
            "FedDyn must beat the plain mean on dim0: {}",
            result[0]
        );
    }

    #[test]
    fn homogeneous_curvature_recovers_plain_mean() {
        // Equal curvature → curvature-weighted mean == plain mean of optima.
        let curvatures = vec![vec![2.0_f32], vec![2.0_f32], vec![2.0_f32]];
        let optima = vec![vec![0.0_f32], vec![4.0_f32], vec![-1.0_f32]];
        // weighted == plain mean = (0+4−1)/3 = 1.0
        let result = run_simulation(&curvatures, &optima, 1.0, 300);
        assert!((result[0] - 1.0).abs() < 2e-2, "result={}", result[0]);
    }

    #[test]
    fn round_increments() {
        let mut state = FedDynState::new(2);
        let cfg = FedDynConfig::new(1, 1.0, 0.1, 1).expect("cfg");
        assert_eq!(state.round, 0);
        FedDyn::aggregate(&mut state, &[vec![0.1, 0.2]], &cfg).expect("agg");
        assert_eq!(state.round, 1);
    }
}
