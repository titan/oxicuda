//! FedAvg: Federated Averaging algorithm.
//!
//! McMahan et al., "Communication-Efficient Learning of Deep Networks from
//! Decentralized Data", AISTATS 2017.

use crate::error::{FedError, FedResult};

/// Configuration for FedAvg training.
#[derive(Debug, Clone)]
pub struct FedAvgConfig {
    /// Total number of clients in the federation.
    pub n_clients: usize,
    /// Fraction of clients selected per round (in (0, 1]).
    pub fraction: f32,
    /// Number of local SGD steps per client per round.
    pub local_epochs: usize,
    /// Client-side learning rate.
    pub learning_rate: f32,
}

impl FedAvgConfig {
    /// Create a validated FedAvg configuration.
    ///
    /// Returns `Err` if any parameter is invalid.
    pub fn new(
        n_clients: usize,
        fraction: f32,
        local_epochs: usize,
        learning_rate: f32,
    ) -> FedResult<Self> {
        if n_clients == 0 {
            return Err(FedError::EmptyClientList);
        }
        if !(fraction > 0.0 && fraction <= 1.0 && fraction.is_finite()) {
            return Err(FedError::InvalidWeight { weight: fraction });
        }
        if !(learning_rate > 0.0 && learning_rate.is_finite()) {
            return Err(FedError::InvalidWeight {
                weight: learning_rate,
            });
        }
        Ok(Self {
            n_clients,
            fraction,
            local_epochs,
            learning_rate,
        })
    }
}

/// Server-side state for FedAvg.
#[derive(Debug, Clone)]
pub struct FedAvgState {
    /// Current global model parameters.
    pub global_params: Vec<f32>,
    /// Number of completed aggregation rounds.
    pub round: usize,
}

impl FedAvgState {
    /// Initialize a new FedAvg state with zero global parameters.
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

    /// Perform weighted FedAvg aggregation.
    ///
    /// `updates` is a slice of `(client_params, client_weight)` pairs.
    /// The global model is updated as: `global = Σ(w_i * params_i) / Σw_i`.
    ///
    /// # Errors
    /// Returns `Err` if any update has wrong dimension, if weights are
    /// non-positive/non-finite, or if the client list is empty.
    pub fn aggregate(&mut self, updates: &[(Vec<f32>, f32)]) -> FedResult<()> {
        if updates.is_empty() {
            return Err(FedError::EmptyClientList);
        }

        let n_params = self.global_params.len();

        // Validate all updates first
        let mut weight_sum = 0.0_f64;
        for (params, w) in updates {
            if params.len() != n_params {
                return Err(FedError::DimensionMismatch {
                    expected: n_params,
                    got: params.len(),
                });
            }
            if !(*w > 0.0 && w.is_finite()) {
                return Err(FedError::InvalidWeight { weight: *w });
            }
            weight_sum += *w as f64;
        }

        if weight_sum <= 0.0 {
            return Err(FedError::InvalidWeight { weight: 0.0 });
        }

        // Weighted average
        let mut new_params = vec![0.0_f64; n_params];
        for (params, w) in updates {
            let w64 = *w as f64;
            for (acc, &p) in new_params.iter_mut().zip(params.iter()) {
                *acc += w64 * p as f64;
            }
        }

        let inv_weight_sum = 1.0 / weight_sum;
        for (out, acc) in self.global_params.iter_mut().zip(new_params.iter()) {
            *out = (*acc * inv_weight_sum) as f32;
        }

        self.round += 1;
        Ok(())
    }

    /// Compute the number of clients to select per round based on fraction.
    #[must_use]
    pub fn clients_per_round(&self, config: &FedAvgConfig) -> usize {
        let n = (config.n_clients as f32 * config.fraction).ceil() as usize;
        n.max(1).min(config.n_clients)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fedavg_state_new_zeros() {
        let s = FedAvgState::new(4);
        assert_eq!(s.global_params.len(), 4);
        assert!(s.global_params.iter().all(|&v| v == 0.0));
        assert_eq!(s.round, 0);
    }

    #[test]
    fn fedavg_aggregate_equal_weights() {
        let mut state = FedAvgState::from_params(vec![0.0; 3]);
        let updates = vec![
            (vec![1.0f32, 2.0, 3.0], 0.5f32),
            (vec![3.0f32, 4.0, 5.0], 0.5f32),
        ];
        state
            .aggregate(&updates)
            .expect("test invariant: valid aggregate");
        assert!((state.global_params[0] - 2.0).abs() < 1e-5);
        assert!((state.global_params[1] - 3.0).abs() < 1e-5);
        assert!((state.global_params[2] - 4.0).abs() < 1e-5);
        assert_eq!(state.round, 1);
    }

    #[test]
    fn fedavg_aggregate_dimension_mismatch() {
        let mut state = FedAvgState::new(3);
        let updates = vec![(vec![1.0f32, 2.0], 1.0f32)];
        assert!(matches!(
            state.aggregate(&updates),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn fedavg_aggregate_empty_list() {
        let mut state = FedAvgState::new(3);
        assert!(matches!(
            state.aggregate(&[]),
            Err(FedError::EmptyClientList)
        ));
    }

    #[test]
    fn fedavg_aggregate_invalid_weight() {
        let mut state = FedAvgState::new(2);
        let updates = vec![(vec![1.0f32, 2.0], -1.0f32)];
        assert!(matches!(
            state.aggregate(&updates),
            Err(FedError::InvalidWeight { .. })
        ));
    }

    #[test]
    fn fedavg_config_new_valid() {
        let cfg = FedAvgConfig::new(10, 0.5, 5, 0.01).expect("test invariant: valid config");
        assert_eq!(cfg.n_clients, 10);
    }

    #[test]
    fn fedavg_config_new_invalid_fraction() {
        assert!(FedAvgConfig::new(10, 0.0, 5, 0.01).is_err());
        assert!(FedAvgConfig::new(10, 1.5, 5, 0.01).is_err());
    }

    #[test]
    fn fedavg_clients_per_round() {
        let cfg = FedAvgConfig::new(10, 0.3, 5, 0.01).expect("test invariant: valid config");
        let state = FedAvgState::new(4);
        assert_eq!(state.clients_per_round(&cfg), 3);
    }
}
