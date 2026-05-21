//! Ditto: Personalised Federated Learning.
//!
//! Li et al., "Ditto: Fair and Robust Federated Learning Through Personalization",
//! ICML 2021.
//!
//! Each client maintains both a **global** model (updated via FedAvg-style weighted
//! averaging) and a **personal** model kept locally. The personal model is updated
//! by proximal gradient descent anchored toward the global model:
//!
//! ```text
//!   v_i ← v_i − η_p · (∇L_i(v_i) + λ(v_i − w))
//! ```
//!
//! where `w` is the current global model and `λ > 0` controls the pull strength.

use crate::error::{FedError, FedResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for Ditto.
#[derive(Debug, Clone)]
pub struct DittoConfig {
    /// Number of clients.
    pub n_clients: usize,
    /// Proximal strength λ > 0. Controls pull of personal model toward global model.
    pub lambda: f32,
    /// Server-side learning rate for global model aggregation.
    pub global_lr: f32,
    /// Client-side learning rate for personal model updates.
    pub personal_lr: f32,
    /// Number of personal gradient steps per round.
    pub n_personal_steps: usize,
}

// ─── Client Update ────────────────────────────────────────────────────────────

/// Per-client update for global model aggregation.
#[derive(Debug, Clone)]
pub struct DittoClientUpdate {
    /// Index of the client that produced this update.
    pub client_id: usize,
    /// Local delta: `w_before − w_after_local_SGD`, used for FedAvg-style aggregation.
    pub global_delta: Vec<f32>,
    /// Number of local training samples (used for weighted averaging).
    pub n_samples: usize,
}

// ─── State ────────────────────────────────────────────────────────────────────

/// Combined global + personal model state for Ditto.
#[derive(Debug, Clone)]
pub struct DittoState {
    /// Global parameters shared across clients.
    pub global_params: Vec<f32>,
    /// Personal parameters, one `Vec<f32>` per client.
    pub personal_params: Vec<Vec<f32>>,
    /// Current round number.
    pub round: usize,
}

impl DittoState {
    /// Initialise state with all-zero global and personal parameters.
    #[must_use]
    pub fn new(n_clients: usize, n_params: usize) -> Self {
        Self {
            global_params: vec![0.0_f32; n_params],
            personal_params: vec![vec![0.0_f32; n_params]; n_clients],
            round: 0,
        }
    }

    /// Initialise from an existing global parameter vector.
    ///
    /// Each client's personal model is initialised as a copy of the global model.
    #[must_use]
    pub fn from_params(n_clients: usize, params: Vec<f32>) -> Self {
        let personal_params = vec![params.clone(); n_clients];
        Self {
            global_params: params,
            personal_params,
            round: 0,
        }
    }
}

// ─── Algorithm ───────────────────────────────────────────────────────────────

/// Ditto algorithm implementation.
pub struct Ditto;

impl Ditto {
    /// Aggregate global client updates using weighted FedAvg.
    ///
    /// Updates `state.global_params` in place:
    ///
    /// ```text
    ///   w ← w − global_lr · (1/N_total) · Σ_i n_i · Δ_i
    /// ```
    ///
    /// where `N_total = Σ n_i` and `Δ_i = global_delta_i`.
    ///
    /// # Errors
    /// - [`FedError::EmptyClientList`] if `updates` is empty.
    /// - [`FedError::DimensionMismatch`] if any `global_delta.len() ≠ global_params.len()`.
    /// - [`FedError::InvalidWeight`] if any `n_samples == 0`.
    pub fn aggregate_global(
        state: &mut DittoState,
        updates: &[DittoClientUpdate],
        cfg: &DittoConfig,
    ) -> FedResult<()> {
        if updates.is_empty() {
            return Err(FedError::EmptyClientList);
        }

        let n_params = state.global_params.len();

        // Validate all updates before mutating state.
        for upd in updates {
            if upd.global_delta.len() != n_params {
                return Err(FedError::DimensionMismatch {
                    expected: n_params,
                    got: upd.global_delta.len(),
                });
            }
            if upd.n_samples == 0 {
                return Err(FedError::InvalidWeight { weight: 0.0 });
            }
        }

        // Compute total sample count.
        let total_samples: usize = updates.iter().map(|u| u.n_samples).sum();
        // total_samples > 0 is guaranteed because every n_samples > 0 was validated above.

        // Aggregate: delta_agg[j] = Σ_i (n_i / N_total) * Δ_i[j]
        let mut delta_agg = vec![0.0_f64; n_params];
        let inv_total = 1.0_f64 / total_samples as f64;
        for upd in updates {
            let weight = upd.n_samples as f64 * inv_total;
            for (acc, &d) in delta_agg.iter_mut().zip(upd.global_delta.iter()) {
                *acc += weight * d as f64;
            }
        }

        // Apply: w[j] -= global_lr * delta_agg[j]
        let lr = cfg.global_lr as f64;
        for (p, agg) in state.global_params.iter_mut().zip(delta_agg.iter()) {
            *p -= (lr * agg) as f32;
        }

        state.round += 1;
        Ok(())
    }

    /// Update the personal model for `client_id` via proximal gradient descent.
    ///
    /// Runs `n_personal_steps` steps of:
    ///
    /// ```text
    ///   v ← v − personal_lr · (∇L_i(v) + λ · (v − w))
    /// ```
    ///
    /// The `grad_fn` closure takes the current personal params as a slice and must
    /// return the gradient `∇L_i(v)` as a `Vec<f32>` of identical length.
    ///
    /// # Errors
    /// - [`FedError::InvalidClientUtility`] if `client_id ≥ n_clients`.
    /// - [`FedError::DimensionMismatch`] if `grad_fn` returns a vector of wrong length.
    pub fn update_personal<F>(
        state: &mut DittoState,
        client_id: usize,
        grad_fn: F,
        cfg: &DittoConfig,
    ) -> FedResult<()>
    where
        F: Fn(&[f32]) -> Vec<f32>,
    {
        if client_id >= cfg.n_clients {
            return Err(FedError::InvalidClientUtility);
        }

        let n_params = state.global_params.len();

        for _ in 0..cfg.n_personal_steps {
            // Clone current personal params so we can pass a slice to grad_fn.
            let v: Vec<f32> = state.personal_params[client_id].clone();

            let grad = grad_fn(&v);
            if grad.len() != n_params {
                return Err(FedError::DimensionMismatch {
                    expected: n_params,
                    got: grad.len(),
                });
            }

            // Compute proximal gradient and apply update.
            let personal = &mut state.personal_params[client_id];
            let global = &state.global_params;
            let lr = cfg.personal_lr;
            let lambda = cfg.lambda;
            for j in 0..n_params {
                let proximal_grad = grad[j] + lambda * (v[j] - global[j]);
                personal[j] -= lr * proximal_grad;
            }
        }

        Ok(())
    }

    /// Compute the proximal objective for `client_id`:
    ///
    /// ```text
    ///   F_i(v) = loss_fn(v) + (λ/2) · ||v − w||²
    /// ```
    ///
    /// `loss_fn` takes the current personal params `v` and returns a scalar loss.
    ///
    /// # Errors
    /// - [`FedError::InvalidClientUtility`] if `client_id ≥ n_clients`.
    pub fn proximal_objective<F>(
        state: &DittoState,
        client_id: usize,
        loss_fn: F,
        cfg: &DittoConfig,
    ) -> FedResult<f32>
    where
        F: Fn(&[f32]) -> f32,
    {
        if client_id >= cfg.n_clients {
            return Err(FedError::InvalidClientUtility);
        }

        let v = &state.personal_params[client_id];
        let w = &state.global_params;

        let loss = loss_fn(v);

        let sq_dist: f32 = v
            .iter()
            .zip(w.iter())
            .map(|(&vi, &wi)| (vi - wi).powi(2))
            .sum();
        let prox = (cfg.lambda / 2.0) * sq_dist;

        Ok(loss + prox)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg(
        n_clients: usize,
        lambda: f32,
        global_lr: f32,
        personal_lr: f32,
        steps: usize,
    ) -> DittoConfig {
        DittoConfig {
            n_clients,
            lambda,
            global_lr,
            personal_lr,
            n_personal_steps: steps,
        }
    }

    // ─── Test 1: single client global aggregation ─────────────────────────────
    #[test]
    fn ditto_global_agg_single_client() {
        let mut state = DittoState::from_params(1, vec![5.0_f32, 3.0]);
        let cfg = make_cfg(1, 0.1, 1.0, 0.01, 1);
        let updates = vec![DittoClientUpdate {
            client_id: 0,
            global_delta: vec![2.0_f32, 1.0],
            n_samples: 10,
        }];
        Ditto::aggregate_global(&mut state, &updates, &cfg).unwrap();
        // w[j] -= global_lr * (n/N) * delta[j] = 1.0 * 1.0 * 2.0 = 2.0
        assert!((state.global_params[0] - 3.0).abs() < 1e-5);
        assert!((state.global_params[1] - 2.0).abs() < 1e-5);
    }

    // ─── Test 2: homogeneous clients equal to FedAvg ─────────────────────────
    #[test]
    fn ditto_global_agg_homogeneous() {
        let mut state = DittoState::from_params(2, vec![10.0_f32, 20.0]);
        let cfg = make_cfg(2, 0.1, 1.0, 0.01, 1);
        let delta = vec![2.0_f32, 4.0];
        let updates = vec![
            DittoClientUpdate {
                client_id: 0,
                global_delta: delta.clone(),
                n_samples: 5,
            },
            DittoClientUpdate {
                client_id: 1,
                global_delta: delta.clone(),
                n_samples: 5,
            },
        ];
        Ditto::aggregate_global(&mut state, &updates, &cfg).unwrap();
        // Equal weights, same delta -> same result as single update with that delta
        // delta_agg = 0.5*2 + 0.5*2 = 2.0
        assert!((state.global_params[0] - 8.0).abs() < 1e-4);
        assert!((state.global_params[1] - 16.0).abs() < 1e-4);
    }

    // ─── Test 3: weighted aggregation 3:1 ────────────────────────────────────
    #[test]
    fn ditto_global_agg_weighted() {
        let mut state = DittoState::from_params(2, vec![0.0_f32]);
        let cfg = make_cfg(2, 0.1, 1.0, 0.01, 1);
        let updates = vec![
            DittoClientUpdate {
                client_id: 0,
                global_delta: vec![4.0_f32],
                n_samples: 3,
            },
            DittoClientUpdate {
                client_id: 1,
                global_delta: vec![0.0_f32],
                n_samples: 1,
            },
        ];
        Ditto::aggregate_global(&mut state, &updates, &cfg).unwrap();
        // delta_agg = (3/4)*4 + (1/4)*0 = 3.0; w -= 1.0 * 3.0 = -3.0
        assert!((state.global_params[0] - (-3.0)).abs() < 1e-4);
    }

    // ─── Test 4: round increments ─────────────────────────────────────────────
    #[test]
    fn ditto_round_increments() {
        let mut state = DittoState::new(1, 2);
        let cfg = make_cfg(1, 0.1, 0.01, 0.01, 1);
        assert_eq!(state.round, 0);
        let updates = vec![DittoClientUpdate {
            client_id: 0,
            global_delta: vec![0.0, 0.0],
            n_samples: 1,
        }];
        Ditto::aggregate_global(&mut state, &updates, &cfg).unwrap();
        assert_eq!(state.round, 1);
        Ditto::aggregate_global(&mut state, &updates, &cfg).unwrap();
        assert_eq!(state.round, 2);
    }

    // ─── Test 5: personal update moves toward global when lambda > 0 ─────────
    #[test]
    fn ditto_personal_update_moves_toward_global() {
        // global = 0, personal = 1, grad = 0 → proximal force pulls personal toward 0
        let mut state = DittoState::from_params(1, vec![0.0_f32]);
        state.personal_params[0] = vec![1.0_f32];
        let cfg = make_cfg(1, 1.0, 0.01, 0.5, 1);
        Ditto::update_personal(&mut state, 0, |_v| vec![0.0_f32], &cfg).unwrap();
        // proximal_grad = 0 + 1*(1 - 0) = 1; new v = 1 - 0.5*1 = 0.5
        assert!((state.personal_params[0][0] - 0.5).abs() < 1e-5);
    }

    // ─── Test 6: n_personal_steps applied correctly ───────────────────────────
    #[test]
    fn ditto_personal_update_n_steps() {
        // global = 0, personal = 8, grad_fn = 0, lambda = 1, lr = 0.5
        // After each step v *= (1 - lr*lambda) = 0.5
        // After 3 steps: 8 * 0.5^3 = 1.0
        let mut state = DittoState::from_params(1, vec![0.0_f32]);
        state.personal_params[0] = vec![8.0_f32];
        let cfg = make_cfg(1, 1.0, 0.01, 0.5, 3);
        Ditto::update_personal(&mut state, 0, |_v| vec![0.0_f32], &cfg).unwrap();
        assert!((state.personal_params[0][0] - 1.0).abs() < 1e-4);
    }

    // ─── Test 7: zero gradient — purely proximal convergence to global ────────
    #[test]
    fn ditto_personal_zero_gradient() {
        let init = vec![100.0_f32, 200.0];
        let mut state = DittoState::from_params(1, vec![0.0_f32, 0.0]);
        state.personal_params[0] = init;
        let cfg = make_cfg(1, 1.0, 0.01, 0.1, 20);
        Ditto::update_personal(&mut state, 0, |_v| vec![0.0_f32, 0.0], &cfg).unwrap();
        // After 20 steps of shrinking by (1 - 0.1*1)^20 = 0.9^20 ≈ 0.122
        // params should be strictly closer to 0 than their initial values
        assert!(state.personal_params[0][0].abs() < 20.0);
        assert!(state.personal_params[0][1].abs() < 40.0);
    }

    // ─── Test 8: lambda = 0 — only gradient, no proximal pull ────────────────
    #[test]
    fn ditto_personal_lambda_zero() {
        let mut state = DittoState::from_params(1, vec![0.0_f32]);
        state.personal_params[0] = vec![10.0_f32];
        // With lambda=0, only the constant gradient 2.0 drives the update.
        let cfg = make_cfg(1, 0.0, 0.01, 0.5, 1);
        Ditto::update_personal(&mut state, 0, |_v| vec![2.0_f32], &cfg).unwrap();
        // v = 10 - 0.5 * (2 + 0*(10-0)) = 10 - 1 = 9
        assert!((state.personal_params[0][0] - 9.0).abs() < 1e-5);
    }

    // ─── Test 9: proximal objective at global → prox term == 0 ───────────────
    #[test]
    fn ditto_proximal_objective_at_global() {
        let params = vec![3.0_f32, -1.0, 2.0];
        let state = DittoState::from_params(1, params.clone());
        // personal == global, so prox term is 0
        let cfg = make_cfg(1, 5.0, 0.01, 0.01, 1);
        let obj = Ditto::proximal_objective(&state, 0, |_v| 7.0_f32, &cfg).unwrap();
        assert!((obj - 7.0).abs() < 1e-5);
    }

    // ─── Test 10: proximal objective non-zero when v ≠ w ─────────────────────
    #[test]
    fn ditto_proximal_objective_nonzero() {
        let mut state = DittoState::from_params(1, vec![0.0_f32]);
        state.personal_params[0] = vec![2.0_f32];
        let cfg = make_cfg(1, 1.0, 0.01, 0.01, 1);
        let obj = Ditto::proximal_objective(&state, 0, |_v| 0.0_f32, &cfg).unwrap();
        // prox = (1.0/2) * (2-0)^2 = 2.0; loss = 0 → obj = 2.0
        assert!((obj - 2.0).abs() < 1e-5);
    }

    // ─── Test 11: DittoState::new initialises zeros ───────────────────────────
    #[test]
    fn ditto_state_new_zeros() {
        let state = DittoState::new(3, 5);
        assert_eq!(state.global_params.len(), 5);
        assert!(state.global_params.iter().all(|&v| v == 0.0));
        assert_eq!(state.personal_params.len(), 3);
        for p in &state.personal_params {
            assert_eq!(p.len(), 5);
            assert!(p.iter().all(|&v| v == 0.0));
        }
        assert_eq!(state.round, 0);
    }

    // ─── Test 12: DittoState::from_params copies global to personal ──────────
    #[test]
    fn ditto_state_from_params() {
        let params = vec![1.0_f32, 2.0, 3.0];
        let state = DittoState::from_params(2, params.clone());
        assert_eq!(state.global_params, params);
        for p in &state.personal_params {
            assert_eq!(*p, params);
        }
    }

    // ─── Test 13: empty updates → EmptyClientList ─────────────────────────────
    #[test]
    fn ditto_err_empty_updates() {
        let mut state = DittoState::new(1, 2);
        let cfg = make_cfg(1, 0.1, 0.01, 0.01, 1);
        assert!(matches!(
            Ditto::aggregate_global(&mut state, &[], &cfg),
            Err(FedError::EmptyClientList)
        ));
    }

    // ─── Test 14: wrong delta length → DimensionMismatch ─────────────────────
    #[test]
    fn ditto_err_dim_mismatch_delta() {
        let mut state = DittoState::new(1, 3);
        let cfg = make_cfg(1, 0.1, 0.01, 0.01, 1);
        let updates = vec![DittoClientUpdate {
            client_id: 0,
            global_delta: vec![1.0, 2.0], // wrong length (expected 3)
            n_samples: 1,
        }];
        assert!(matches!(
            Ditto::aggregate_global(&mut state, &updates, &cfg),
            Err(FedError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));
    }

    // ─── Test 15: invalid client_id → InvalidClientUtility ───────────────────
    #[test]
    fn ditto_err_invalid_client_id() {
        let mut state = DittoState::new(2, 3);
        let cfg = make_cfg(2, 0.1, 0.01, 0.01, 1);
        assert!(matches!(
            Ditto::update_personal(&mut state, 5, |_v| vec![0.0, 0.0, 0.0], &cfg),
            Err(FedError::InvalidClientUtility)
        ));
        assert!(matches!(
            Ditto::proximal_objective(&state, 5, |_v| 0.0, &cfg),
            Err(FedError::InvalidClientUtility)
        ));
    }

    // ─── Test 16: n_samples == 0 → InvalidWeight ─────────────────────────────
    #[test]
    fn ditto_err_zero_samples() {
        let mut state = DittoState::new(1, 2);
        let cfg = make_cfg(1, 0.1, 0.01, 0.01, 1);
        let updates = vec![DittoClientUpdate {
            client_id: 0,
            global_delta: vec![0.0, 0.0],
            n_samples: 0,
        }];
        assert!(matches!(
            Ditto::aggregate_global(&mut state, &updates, &cfg),
            Err(FedError::InvalidWeight { weight: 0.0 })
        ));
    }

    // ─── Test 17: multiple clients with different grad_fns diverge ────────────
    #[test]
    fn ditto_multiple_clients_different_personal() {
        // 3 clients start at same point; each pushed by different constant gradient.
        let mut state = DittoState::new(3, 1);
        let cfg = make_cfg(3, 0.0, 0.01, 0.1, 5);
        // Client 0: pushed by +1 gradient
        Ditto::update_personal(&mut state, 0, |_v| vec![1.0_f32], &cfg).unwrap();
        // Client 1: pushed by -1 gradient
        Ditto::update_personal(&mut state, 1, |_v| vec![-1.0_f32], &cfg).unwrap();
        // Client 2: pushed by +3 gradient
        Ditto::update_personal(&mut state, 2, |_v| vec![3.0_f32], &cfg).unwrap();

        let v0 = state.personal_params[0][0];
        let v1 = state.personal_params[1][0];
        let v2 = state.personal_params[2][0];
        assert!(v0 < v1, "v0={v0}, v1={v1}: client 0 should be < client 1");
        assert!(v2 < v0, "v2={v2}, v0={v0}: client 2 should be < client 0");
    }

    // ─── Test 18: aggregate_global does not affect personal params ────────────
    #[test]
    fn ditto_aggregate_does_not_affect_personal() {
        let mut state = DittoState::from_params(2, vec![1.0_f32, 2.0, 3.0]);
        // Push personal params to a different position.
        state.personal_params[0] = vec![10.0, 20.0, 30.0];
        state.personal_params[1] = vec![-1.0, -2.0, -3.0];
        let personal_before: Vec<Vec<f32>> = state.personal_params.clone();

        let cfg = make_cfg(2, 0.1, 0.5, 0.01, 1);
        let updates = vec![
            DittoClientUpdate {
                client_id: 0,
                global_delta: vec![1.0, 1.0, 1.0],
                n_samples: 5,
            },
            DittoClientUpdate {
                client_id: 1,
                global_delta: vec![1.0, 1.0, 1.0],
                n_samples: 5,
            },
        ];
        Ditto::aggregate_global(&mut state, &updates, &cfg).unwrap();

        assert_eq!(state.personal_params, personal_before);
    }
}
