//! FedNova: Federated Normalized Averaging.
//!
//! Wang et al., "Tackling the Objective Inconsistency Problem in Heterogeneous
//! Federated Optimization", NeurIPS 2020.
//!
//! FedNova corrects for client drift caused by heterogeneous numbers of local SGD
//! steps by normalizing each client's update by its effective local step size τ_i
//! before aggregation, then re-scaling by the global effective step size τ_eff.

use crate::error::{FedError, FedResult};

/// Configuration for FedNova training.
#[derive(Debug, Clone)]
pub struct FedNovaConfig {
    /// Total number of participating clients.
    pub n_clients: usize,
    /// Global (server) learning rate `η_g`.
    pub global_learning_rate: f32,
    /// Client momentum coefficient `μ ∈ [0, 1)`. Set to `0.0` for vanilla SGD.
    pub momentum: f32,
    /// Client weight-decay coefficient `λ ≥ 0`.
    pub weight_decay: f32,
}

/// Per-client update produced after local SGD.
#[derive(Debug, Clone)]
pub struct FedNovaClientUpdate {
    /// Unique identifier of this client.
    pub client_id: usize,
    /// Parameter difference `θ_global − θ_local` (positive = moved in gradient direction).
    pub local_delta: Vec<f32>,
    /// Number of local SGD steps `K_i` (must be > 0).
    pub local_steps: usize,
    /// Local dataset size `n_i` (must be > 0).
    pub n_samples: usize,
}

/// Server-side state for FedNova.
#[derive(Debug, Clone)]
pub struct FedNovaState {
    /// Current global model parameters.
    pub global_params: Vec<f32>,
    /// Number of completed aggregation rounds.
    pub round: usize,
}

impl FedNovaState {
    /// Initialize a new FedNova state with zero global parameters.
    #[must_use]
    pub fn new(n_params: usize) -> Self {
        Self {
            global_params: vec![0.0_f32; n_params],
            round: 0,
        }
    }

    /// Initialize from existing parameters.
    #[must_use]
    pub fn from_params(params: Vec<f32>) -> Self {
        Self {
            global_params: params,
            round: 0,
        }
    }
}

/// Stateless handle providing all FedNova primitives.
pub struct FedNova;

impl FedNova {
    /// Effective local step size τ_i for a client with `k` local SGD steps.
    ///
    /// - No momentum (`|μ| < 1e-6`): τ_i = K
    /// - With momentum: τ_i = K − μ·(1 − μ^K) / (1 − μ)
    ///   (geometric series correction from Wang et al. 2020, §3.1)
    #[must_use]
    pub fn tau_i(k: usize, momentum: f32, _weight_decay: f32) -> f32 {
        if momentum.abs() < 1e-6 {
            k as f32
        } else {
            // τ_i = K - μ*(1 - μ^K)/(1 - μ)
            let mu = momentum;
            let k_f = k as f32;
            k_f - mu * (1.0 - mu.powi(k as i32)) / (1.0 - mu)
        }
    }

    /// Normalize a client's local delta by its effective step size τ_i.
    ///
    /// Returns `a_i = delta / τ_i`.
    ///
    /// # Errors
    /// - [`FedError::InvalidWeight`] if τ_i is (near-)zero.
    pub fn normalize_delta(
        delta: &[f32],
        k: usize,
        momentum: f32,
        weight_decay: f32,
    ) -> FedResult<Vec<f32>> {
        let tau = Self::tau_i(k, momentum, weight_decay);
        if tau.abs() < 1e-8 {
            return Err(FedError::InvalidWeight { weight: tau });
        }
        let inv_tau = 1.0 / tau;
        Ok(delta.iter().map(|&d| d * inv_tau).collect())
    }

    /// Compute the global effective step size `τ_eff` from a set of client updates.
    ///
    /// τ_eff = n_total / Σ_i(n_i / K_i)
    ///
    /// This equals 1 / Σ_i(p_i / K_i) where p_i = n_i / n_total.
    ///
    /// # Errors
    /// - [`FedError::EmptyClientList`] if total sample count is zero.
    pub fn effective_step_size(updates: &[FedNovaClientUpdate]) -> FedResult<f32> {
        let n_total: usize = updates.iter().map(|u| u.n_samples).sum();
        if n_total == 0 {
            return Err(FedError::EmptyClientList);
        }
        // Σ_i(n_i / K_i)
        let denom: f64 = updates
            .iter()
            .map(|u| u.n_samples as f64 / u.local_steps as f64)
            .sum();
        if denom.abs() < 1e-15 {
            return Err(FedError::InvalidWeight {
                weight: denom as f32,
            });
        }
        Ok((n_total as f64 / denom) as f32)
    }

    /// Perform one round of FedNova server aggregation.
    ///
    /// Updates `state.global_params` using the normalized-averaging rule:
    ///
    /// ```text
    /// d_agg = Σ_i p_i · (delta_i / τ_i)
    /// τ_eff = 1 / Σ_i(p_i / τ_i)
    /// θ_new = θ_old - η_g · τ_eff · d_agg
    /// ```
    ///
    /// # Errors
    /// - [`FedError::EmptyClientList`] — updates is empty or total n_samples is 0.
    /// - [`FedError::DimensionMismatch`] — any `local_delta.len() ≠ n_params`.
    /// - [`FedError::InvalidWeight`] — any `local_steps == 0` or τ_i ≈ 0.
    pub fn aggregate(
        state: &mut FedNovaState,
        updates: &[FedNovaClientUpdate],
        cfg: &FedNovaConfig,
    ) -> FedResult<()> {
        if updates.is_empty() {
            return Err(FedError::EmptyClientList);
        }

        let n_params = state.global_params.len();

        // Validate all updates.
        for upd in updates {
            if upd.local_delta.len() != n_params {
                return Err(FedError::DimensionMismatch {
                    expected: n_params,
                    got: upd.local_delta.len(),
                });
            }
            if upd.local_steps == 0 {
                return Err(FedError::InvalidWeight { weight: 0.0 });
            }
            if upd.n_samples == 0 {
                return Err(FedError::EmptyClientList);
            }
        }

        let n_total: usize = updates.iter().map(|u| u.n_samples).sum();
        let n_total_f64 = n_total as f64;

        // Compute Σ_i(p_i / τ_i) and accumulate d_agg = Σ_i p_i * a_i.
        let mut sum_p_over_tau = 0.0_f64;
        let mut d_agg = vec![0.0_f64; n_params];

        for upd in updates {
            let p_i = upd.n_samples as f64 / n_total_f64;
            let tau = Self::tau_i(upd.local_steps, cfg.momentum, cfg.weight_decay) as f64;
            if tau.abs() < 1e-8 {
                return Err(FedError::InvalidWeight { weight: tau as f32 });
            }
            sum_p_over_tau += p_i / tau;
            // a_i = delta_i / tau_i; accumulate p_i * a_i.
            for (acc, &d) in d_agg.iter_mut().zip(upd.local_delta.iter()) {
                *acc += p_i * (d as f64) / tau;
            }
        }

        if sum_p_over_tau.abs() < 1e-15 {
            return Err(FedError::InvalidWeight {
                weight: sum_p_over_tau as f32,
            });
        }

        // τ_eff = 1 / Σ_i(p_i / τ_i)
        let tau_eff = 1.0 / sum_p_over_tau;
        let eta_g = cfg.global_learning_rate as f64;

        // θ_new = θ_old - η_g * τ_eff * d_agg
        for (param, &da) in state.global_params.iter_mut().zip(d_agg.iter()) {
            *param = (*param as f64 - eta_g * tau_eff * da) as f32;
        }

        state.round += 1;
        Ok(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg(n: usize) -> FedNovaConfig {
        FedNovaConfig {
            n_clients: n,
            global_learning_rate: 1.0,
            momentum: 0.0,
            weight_decay: 0.0,
        }
    }

    fn make_update(id: usize, delta: Vec<f32>, k: usize, n: usize) -> FedNovaClientUpdate {
        FedNovaClientUpdate {
            client_id: id,
            local_delta: delta,
            local_steps: k,
            n_samples: n,
        }
    }

    // ── Test 1: homogeneous_k_equals_fedavg ──────────────────────────────────
    #[test]
    fn homogeneous_k_equals_fedavg() {
        // With all clients having the same K and equal sample counts,
        // FedNova reduces to FedAvg with η_g / K scaling.
        // θ_new = θ - η_g * τ_eff * d_agg
        //       = θ - 1 * K * mean(delta_i / K)
        //       = θ - mean(delta_i)
        let mut state = FedNovaState::from_params(vec![0.0_f32; 2]);
        let updates = vec![
            make_update(0, vec![2.0_f32, 4.0], 2, 10),
            make_update(1, vec![4.0_f32, 2.0], 2, 10),
        ];
        let cfg = make_cfg(2);
        FedNova::aggregate(&mut state, &updates, &cfg)
            .expect("test invariant: homogeneous k aggregate");
        // d_agg = mean([2/2, 4/2], [4/2, 2/2]) = [1.5, 1.5]
        // τ_eff = 2  (all K=2)
        // θ_new = 0 - 1 * 2 * [1.5, 1.5] = [-3, -3]
        assert!(
            (state.global_params[0] - (-3.0)).abs() < 1e-4,
            "param[0]={}",
            state.global_params[0]
        );
        assert!(
            (state.global_params[1] - (-3.0)).abs() < 1e-4,
            "param[1]={}",
            state.global_params[1]
        );
    }

    // ── Test 2: tau_i_no_momentum ────────────────────────────────────────────
    #[test]
    fn tau_i_no_momentum() {
        let tau = FedNova::tau_i(7, 0.0, 0.0);
        assert!((tau - 7.0).abs() < 1e-5, "tau_i no momentum must equal K");
    }

    // ── Test 3: tau_i_with_momentum ──────────────────────────────────────────
    #[test]
    fn tau_i_with_momentum() {
        let k = 5_usize;
        let mu = 0.9_f32;
        let tau = FedNova::tau_i(k, mu, 0.0);
        // τ < K due to geometric-series correction.
        assert!(tau < k as f32, "tau_i with momentum must be < K, got {tau}");
        assert!(tau > 0.0, "tau_i with momentum must be positive");
    }

    // ── Test 4: normalize_delta_length ───────────────────────────────────────
    #[test]
    fn normalize_delta_length() {
        let delta = vec![1.0_f32, 2.0, 3.0, 4.0];
        let norm = FedNova::normalize_delta(&delta, 3, 0.0, 0.0)
            .expect("test invariant: normalize_delta_length");
        assert_eq!(norm.len(), delta.len());
    }

    // ── Test 5: normalize_delta_scale ────────────────────────────────────────
    #[test]
    fn normalize_delta_scale() {
        let delta = vec![2.0_f32, 4.0];
        let norm = FedNova::normalize_delta(&delta, 2, 0.0, 0.0)
            .expect("test invariant: normalize_delta_scale");
        assert!((norm[0] - 1.0).abs() < 1e-5, "norm[0] should be 1.0");
        assert!((norm[1] - 2.0).abs() < 1e-5, "norm[1] should be 2.0");
    }

    // ── Test 6: effective_step_size_uniform ──────────────────────────────────
    #[test]
    fn effective_step_size_uniform() {
        // All n_i=10, K_i=3 → τ_eff = n_total / Σ(n_i/K_i) = 30 / (3*10/3) = 30/10 = 3
        let updates = vec![
            make_update(0, vec![0.0], 3, 10),
            make_update(1, vec![0.0], 3, 10),
            make_update(2, vec![0.0], 3, 10),
        ];
        let tau_eff = FedNova::effective_step_size(&updates)
            .expect("test invariant: effective_step_size_uniform");
        assert!(
            (tau_eff - 3.0).abs() < 1e-4,
            "uniform K=3 → τ_eff=3, got {tau_eff}"
        );
    }

    // ── Test 7: effective_step_size_weighted ─────────────────────────────────
    #[test]
    fn effective_step_size_weighted() {
        // n_0=10, K_0=2; n_1=10, K_1=4
        // τ_eff = 20 / (10/2 + 10/4) = 20 / (5 + 2.5) = 20/7.5 ≈ 2.667
        let updates = vec![
            make_update(0, vec![0.0], 2, 10),
            make_update(1, vec![0.0], 4, 10),
        ];
        let tau_eff = FedNova::effective_step_size(&updates)
            .expect("test invariant: effective_step_size_weighted");
        let expected = 20.0_f32 / 7.5;
        assert!(
            (tau_eff - expected).abs() < 1e-3,
            "weighted τ_eff={tau_eff}, expected {expected}"
        );
    }

    // ── Test 8: aggregate_updates_state ──────────────────────────────────────
    #[test]
    fn aggregate_updates_state() {
        let mut state = FedNovaState::from_params(vec![5.0_f32, 5.0]);
        let updates = vec![make_update(0, vec![1.0_f32, 1.0], 1, 1)];
        let cfg = make_cfg(1);
        FedNova::aggregate(&mut state, &updates, &cfg)
            .expect("test invariant: aggregate updates state");
        // After aggregation, global_params must have changed.
        assert!(
            state.global_params[0] != 5.0 || state.global_params[1] != 5.0,
            "global_params must change after aggregate"
        );
    }

    // ── Test 9: aggregate_round_increments ───────────────────────────────────
    #[test]
    fn aggregate_round_increments() {
        let mut state = FedNovaState::new(2);
        let updates = vec![make_update(0, vec![0.1_f32, 0.2], 1, 5)];
        let cfg = make_cfg(1);
        assert_eq!(state.round, 0);
        FedNova::aggregate(&mut state, &updates, &cfg)
            .expect("test invariant: aggregate_round_increments");
        assert_eq!(state.round, 1);
    }

    // ── Test 10: single_client_aggregate ─────────────────────────────────────
    #[test]
    fn single_client_aggregate() {
        let mut state = FedNovaState::from_params(vec![0.0_f32; 3]);
        let updates = vec![make_update(0, vec![1.0_f32, 2.0, 3.0], 5, 100)];
        let cfg = make_cfg(1);
        FedNova::aggregate(&mut state, &updates, &cfg)
            .expect("test invariant: single client aggregate");
        // τ_eff=5, p_0=1, a_0=[0.2,0.4,0.6], d_agg=[0.2,0.4,0.6]
        // θ_new = 0 - 1*5*[0.2,0.4,0.6] = [-1,-2,-3]
        assert!((state.global_params[0] - (-1.0)).abs() < 1e-4);
        assert!((state.global_params[1] - (-2.0)).abs() < 1e-4);
        assert!((state.global_params[2] - (-3.0)).abs() < 1e-4);
    }

    // ── Test 11: heterogeneous_k_corrects ────────────────────────────────────
    #[test]
    fn heterogeneous_k_corrects() {
        // Two clients with different K produce a different result from naive FedAvg.
        let mut state_nova = FedNovaState::from_params(vec![0.0_f32; 2]);
        let mut state_avg = FedNovaState::from_params(vec![0.0_f32; 2]);

        let delta_0 = vec![2.0_f32, 2.0];
        let delta_1 = vec![8.0_f32, 8.0];

        let updates = vec![
            make_update(0, delta_0.clone(), 1, 10), // K=1
            make_update(1, delta_1.clone(), 4, 10), // K=4
        ];
        let cfg_nova = make_cfg(2);

        FedNova::aggregate(&mut state_nova, &updates, &cfg_nova)
            .expect("test invariant: heterogeneous k fednova");

        // Naive FedAvg: θ_new = 0 - mean(delta) = -[(2+8)/2, (2+8)/2] = [-5, -5]
        let mean_d = (2.0_f32 + 8.0) / 2.0;
        state_avg.global_params[0] -= mean_d;
        state_avg.global_params[1] -= mean_d;

        // FedNova should differ from naive FedAvg for heterogeneous K.
        let nova_param = state_nova.global_params[0];
        let avg_param = state_avg.global_params[0];
        assert!(
            (nova_param - avg_param).abs() > 1e-3,
            "FedNova should differ from FedAvg for heterogeneous K: nova={nova_param}, avg={avg_param}"
        );
    }

    // ── Test 12: global_params_len_preserved ─────────────────────────────────
    #[test]
    fn global_params_len_preserved() {
        let n = 10_usize;
        let mut state = FedNovaState::new(n);
        let updates = vec![make_update(0, vec![0.5_f32; n], 3, 20)];
        let cfg = make_cfg(1);
        FedNova::aggregate(&mut state, &updates, &cfg)
            .expect("test invariant: global_params_len_preserved");
        assert_eq!(state.global_params.len(), n);
    }

    // ── Test 13: err_empty_updates ───────────────────────────────────────────
    #[test]
    fn err_empty_updates() {
        let mut state = FedNovaState::new(4);
        let cfg = make_cfg(0);
        assert!(matches!(
            FedNova::aggregate(&mut state, &[], &cfg),
            Err(FedError::EmptyClientList)
        ));
    }

    // ── Test 14: err_delta_mismatch ──────────────────────────────────────────
    #[test]
    fn err_delta_mismatch() {
        let mut state = FedNovaState::new(4);
        // delta has wrong length.
        let updates = vec![make_update(0, vec![1.0_f32, 2.0], 3, 10)];
        let cfg = make_cfg(1);
        assert!(matches!(
            FedNova::aggregate(&mut state, &updates, &cfg),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    // ── Test 15: err_zero_local_steps ────────────────────────────────────────
    #[test]
    fn err_zero_local_steps() {
        let mut state = FedNovaState::new(2);
        let updates = vec![make_update(0, vec![0.1_f32, 0.2], 0, 10)];
        let cfg = make_cfg(1);
        assert!(matches!(
            FedNova::aggregate(&mut state, &updates, &cfg),
            Err(FedError::InvalidWeight { .. })
        ));
    }

    // ── Test 16: err_zero_n_samples ──────────────────────────────────────────
    #[test]
    fn err_zero_n_samples() {
        let mut state = FedNovaState::new(2);
        let updates = vec![make_update(0, vec![0.1_f32, 0.2], 3, 0)];
        let cfg = make_cfg(1);
        let result = FedNova::aggregate(&mut state, &updates, &cfg);
        assert!(result.is_err(), "zero n_samples must return error");
    }

    // ── Test 17: fednova_state_new ────────────────────────────────────────────
    #[test]
    fn fednova_state_new() {
        let state = FedNovaState::new(10);
        assert_eq!(state.global_params.len(), 10);
        assert!(
            state.global_params.iter().all(|&v| v == 0.0),
            "new state must have zero params"
        );
        assert_eq!(state.round, 0);
    }
}
