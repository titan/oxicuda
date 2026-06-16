//! Signed-SGD: 1-bit gradient compression via coordinate-wise signing.
//!
//! Bernstein et al., "signSGD: Compressed Optimisation for Non-Convex Problems",
//! ICML 2018.
//!
//! Clients compress gradients to 1 bit per coordinate (the sign). The server
//! aggregates via a coordinate-wise **majority vote** weighted by client sample counts:
//!
//! ```text
//!   vote_j = Σ_i (n_i · sign(g_ij))
//!   d_j    = sign(vote_j)          (0 if tied)
//!   θ_j   -= η · d_j
//! ```
//!
//! This achieves a factor-of-32 compression (f32 → i8 sign per coordinate) with
//! convergence guarantees under bounded gradient variance conditions.

use crate::error::{FedError, FedResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for Signed-SGD.
#[derive(Debug, Clone)]
pub struct SignedSgdConfig {
    /// Number of clients.
    pub n_clients: usize,
    /// Server learning rate η > 0.
    pub server_lr: f32,
}

// ─── Client Update ────────────────────────────────────────────────────────────

/// Per-client compressed update (signed gradient).
#[derive(Debug, Clone)]
pub struct SignedSgdUpdate {
    /// Index of the client that produced this update.
    pub client_id: usize,
    /// Sign of each gradient coordinate: `+1i8` or `-1i8`.
    pub signed_grad: Vec<i8>,
    /// Number of local training samples (used for weighted majority vote).
    pub n_samples: usize,
}

// ─── State ────────────────────────────────────────────────────────────────────

/// Server-side parameter state for Signed-SGD.
#[derive(Debug, Clone)]
pub struct SignedSgdState {
    /// Current model parameters.
    pub params: Vec<f32>,
    /// Completed aggregation rounds.
    pub round: usize,
}

impl SignedSgdState {
    /// Initialise with all-zero parameters.
    #[must_use]
    pub fn new(n_params: usize) -> Self {
        Self {
            params: vec![0.0_f32; n_params],
            round: 0,
        }
    }

    /// Initialise from an existing parameter vector.
    #[must_use]
    pub fn from_params(params: Vec<f32>) -> Self {
        Self { params, round: 0 }
    }
}

// ─── Algorithm ───────────────────────────────────────────────────────────────

/// Signed-SGD algorithm.
pub struct SignedSgd;

impl SignedSgd {
    /// Compress a `f32` gradient vector to `i8` signs.
    ///
    /// `sign(g_j) = +1` if `g_j ≥ 0`, `-1` if `g_j < 0`.
    #[must_use]
    pub fn compress(grad: &[f32]) -> Vec<i8> {
        grad.iter()
            .map(|&g| if g >= 0.0 { 1_i8 } else { -1_i8 })
            .collect()
    }

    /// Decompress a signed gradient to `f32` values scaled by `scale`.
    ///
    /// Returns `scale * s_j` for each coordinate.
    #[must_use]
    pub fn decompress(signed: &[i8], scale: f32) -> Vec<f32> {
        signed.iter().map(|&s| scale * s as f32).collect()
    }

    /// Coordinate-wise weighted majority vote aggregation with parameter update.
    ///
    /// For each coordinate `j`:
    ///
    /// ```text
    ///   vote_j      = Σ_i (n_i · signed_grad_i[j])   (i64 to avoid overflow)
    ///   direction_j = +1.0 if vote_j > 0
    ///               = -1.0 if vote_j < 0
    ///               =  0.0 if vote_j == 0 (tie → no update)
    ///   params[j]  -= server_lr · direction_j
    /// ```
    ///
    /// # Errors
    /// - [`FedError::EmptyClientList`] if `updates` is empty.
    /// - [`FedError::DimensionMismatch`] if any `signed_grad.len() ≠ params.len()`.
    /// - [`FedError::InvalidWeight`] if any `n_samples == 0`.
    pub fn aggregate(
        state: &mut SignedSgdState,
        updates: &[SignedSgdUpdate],
        cfg: &SignedSgdConfig,
    ) -> FedResult<()> {
        if updates.is_empty() {
            return Err(FedError::EmptyClientList);
        }

        let n_params = state.params.len();

        // Validate all updates before mutating state.
        for upd in updates {
            if upd.signed_grad.len() != n_params {
                return Err(FedError::DimensionMismatch {
                    expected: n_params,
                    got: upd.signed_grad.len(),
                });
            }
            if upd.n_samples == 0 {
                return Err(FedError::InvalidWeight { weight: 0.0 });
            }
        }

        // Accumulate weighted votes using i64 to avoid overflow for large cohorts.
        let mut votes = vec![0_i64; n_params];
        for upd in updates {
            let weight = upd.n_samples as i64;
            for (vote, &s) in votes.iter_mut().zip(upd.signed_grad.iter()) {
                *vote += weight * s as i64;
            }
        }

        // Apply direction to parameters.
        let lr = cfg.server_lr;
        for (p, &v) in state.params.iter_mut().zip(votes.iter()) {
            let direction = match v.cmp(&0) {
                std::cmp::Ordering::Greater => 1.0_f32,
                std::cmp::Ordering::Less => -1.0_f32,
                std::cmp::Ordering::Equal => 0.0_f32,
            };
            *p -= lr * direction;
        }

        state.round += 1;
        Ok(())
    }

    /// Compute the compression error between the original gradient and the
    /// decompressed signed gradient scaled by the per-dimension RMS norm.
    ///
    /// Returns `0.0` if the gradient is all-zero (no signal to measure error
    /// against).  Otherwise:
    ///
    /// ```text
    ///   scale  = ||g||₂ / √dim
    ///   g̃_j   = scale · sign(g_j)
    ///   error  = ||g - g̃||₂ / √dim
    /// ```
    #[must_use]
    pub fn compression_error(grad: &[f32]) -> f32 {
        let dim = grad.len();
        if dim == 0 {
            return 0.0;
        }

        let sq_sum: f32 = grad.iter().map(|&g| g * g).sum();
        if sq_sum == 0.0 {
            return 0.0;
        }

        let l2_norm = sq_sum.sqrt() / (dim as f32).sqrt();

        let signed = Self::compress(grad);
        let decompressed = Self::decompress(&signed, l2_norm);

        let err_sq: f32 = grad
            .iter()
            .zip(decompressed.iter())
            .map(|(&g, &gd)| (g - gd).powi(2))
            .sum();

        err_sq.sqrt() / (dim as f32).sqrt()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Test 1: all positive → all +1 ───────────────────────────────────────
    #[test]
    fn signed_sgd_compress_positive() {
        let grad = vec![1.0_f32, 0.5, 100.0, 0.001];
        let signs = SignedSgd::compress(&grad);
        assert!(signs.iter().all(|&s| s == 1_i8));
    }

    // ─── Test 2: all negative → all -1 ───────────────────────────────────────
    #[test]
    fn signed_sgd_compress_negative() {
        let grad = vec![-1.0_f32, -0.5, -100.0, -0.001];
        let signs = SignedSgd::compress(&grad);
        assert!(signs.iter().all(|&s| s == -1_i8));
    }

    // ─── Test 3: zero → +1 (non-negative branch) ─────────────────────────────
    #[test]
    fn signed_sgd_compress_zero() {
        let signs = SignedSgd::compress(&[0.0_f32]);
        assert_eq!(signs, vec![1_i8]);
    }

    // ─── Test 4: mixed values ─────────────────────────────────────────────────
    #[test]
    fn signed_sgd_compress_mixed() {
        let grad = vec![1.0_f32, -1.0, 0.5, -0.5];
        let signs = SignedSgd::compress(&grad);
        assert_eq!(signs, vec![1_i8, -1, 1, -1]);
    }

    // ─── Test 5: decompress scales correctly ──────────────────────────────────
    #[test]
    fn signed_sgd_decompress() {
        let signed = vec![1_i8, -1];
        let decompressed = SignedSgd::decompress(&signed, 0.5);
        assert!((decompressed[0] - 0.5).abs() < 1e-6);
        assert!((decompressed[1] - (-0.5)).abs() < 1e-6);
    }

    // ─── Test 6: compress then decompress with scale=1.0 → ±1.0 ─────────────
    #[test]
    fn signed_sgd_roundtrip_binary() {
        let grad = vec![3.5_f32, -2.0, 0.0, -7.0];
        let signs = SignedSgd::compress(&grad);
        let decompressed = SignedSgd::decompress(&signs, 1.0);
        for d in &decompressed {
            assert!(d.abs() == 1.0, "expected ±1.0, got {d}");
        }
    }

    // ─── Test 7: single client, positive signs → params decrease ─────────────
    #[test]
    fn signed_sgd_aggregate_single_client() {
        let mut state = SignedSgdState::from_params(vec![5.0_f32, 3.0]);
        let cfg = SignedSgdConfig {
            n_clients: 1,
            server_lr: 0.1,
        };
        let updates = vec![SignedSgdUpdate {
            client_id: 0,
            signed_grad: vec![1_i8, 1],
            n_samples: 10,
        }];
        SignedSgd::aggregate(&mut state, &updates, &cfg).expect("aggregate should succeed");
        assert!((state.params[0] - 4.9).abs() < 1e-5);
        assert!((state.params[1] - 2.9).abs() < 1e-5);
    }

    // ─── Test 8: 3 clients, 2 positive + 1 negative → majority +1 wins ───────
    #[test]
    fn signed_sgd_aggregate_majority() {
        let mut state = SignedSgdState::from_params(vec![0.0_f32]);
        let cfg = SignedSgdConfig {
            n_clients: 3,
            server_lr: 1.0,
        };
        let updates = vec![
            SignedSgdUpdate {
                client_id: 0,
                signed_grad: vec![1_i8],
                n_samples: 1,
            },
            SignedSgdUpdate {
                client_id: 1,
                signed_grad: vec![1_i8],
                n_samples: 1,
            },
            SignedSgdUpdate {
                client_id: 2,
                signed_grad: vec![-1_i8],
                n_samples: 1,
            },
        ];
        SignedSgd::aggregate(&mut state, &updates, &cfg).expect("aggregate should succeed");
        // vote = 1+1-1 = 1 > 0 → direction = +1 → params -= 1*1 = -1
        assert!((state.params[0] - (-1.0)).abs() < 1e-5);
    }

    // ─── Test 9: 2 clients +1 and -1 → tie → no update ───────────────────────
    #[test]
    fn signed_sgd_aggregate_tie() {
        let mut state = SignedSgdState::from_params(vec![7.0_f32]);
        let cfg = SignedSgdConfig {
            n_clients: 2,
            server_lr: 1.0,
        };
        let updates = vec![
            SignedSgdUpdate {
                client_id: 0,
                signed_grad: vec![1_i8],
                n_samples: 1,
            },
            SignedSgdUpdate {
                client_id: 1,
                signed_grad: vec![-1_i8],
                n_samples: 1,
            },
        ];
        SignedSgd::aggregate(&mut state, &updates, &cfg).expect("aggregate should succeed");
        // vote = 0 → direction = 0 → params unchanged
        assert!((state.params[0] - 7.0).abs() < 1e-5);
    }

    // ─── Test 10: weighted majority — n_samples [3,1] ─────────────────────────
    #[test]
    fn signed_sgd_aggregate_weighted_majority() {
        let mut state = SignedSgdState::from_params(vec![0.0_f32]);
        let cfg = SignedSgdConfig {
            n_clients: 2,
            server_lr: 1.0,
        };
        let updates = vec![
            // positive sign with weight 3
            SignedSgdUpdate {
                client_id: 0,
                signed_grad: vec![1_i8],
                n_samples: 3,
            },
            // negative sign with weight 1
            SignedSgdUpdate {
                client_id: 1,
                signed_grad: vec![-1_i8],
                n_samples: 1,
            },
        ];
        SignedSgd::aggregate(&mut state, &updates, &cfg).expect("aggregate should succeed");
        // vote = 3*1 + 1*(-1) = 2 > 0 → direction = +1 → params -= 1
        assert!((state.params[0] - (-1.0)).abs() < 1e-5);
    }

    // ─── Test 11: round increments after aggregate ────────────────────────────
    #[test]
    fn signed_sgd_aggregate_round_increments() {
        let mut state = SignedSgdState::new(1);
        let cfg = SignedSgdConfig {
            n_clients: 1,
            server_lr: 0.01,
        };
        let updates = vec![SignedSgdUpdate {
            client_id: 0,
            signed_grad: vec![1_i8],
            n_samples: 1,
        }];
        assert_eq!(state.round, 0);
        SignedSgd::aggregate(&mut state, &updates, &cfg).expect("aggregate should succeed");
        assert_eq!(state.round, 1);
        SignedSgd::aggregate(&mut state, &updates, &cfg).expect("aggregate should succeed");
        assert_eq!(state.round, 2);
    }

    // ─── Test 12: SignedSgdState::new → all params zero ──────────────────────
    #[test]
    fn signed_sgd_state_new_zeros() {
        let state = SignedSgdState::new(4);
        assert_eq!(state.params.len(), 4);
        assert!(state.params.iter().all(|&v| v == 0.0));
        assert_eq!(state.round, 0);
    }

    // ─── Test 13: all-zero gradient → compression error == 0 ─────────────────
    #[test]
    fn signed_sgd_compression_error_zero_grad() {
        let grad = vec![0.0_f32; 10];
        let err = SignedSgd::compression_error(&grad);
        assert_eq!(err, 0.0);
    }

    // ─── Test 14: compression error is bounded by scale ──────────────────────
    #[test]
    fn signed_sgd_compression_error_bounded() {
        // For a gradient that is already ±constant, error should be near zero.
        let grad = vec![2.0_f32; 16];
        let err = SignedSgd::compression_error(&grad);
        // All values are the same positive constant → sign = +1 for all.
        // scale = ||g||/√dim = 2*√16/√16 = 2; decompressed = 2 for all.
        // error = |g - g̃| / √dim = 0.
        assert!(err < 1e-4, "expected near-zero error, got {err}");
    }

    // ─── Test 15: empty updates → EmptyClientList ─────────────────────────────
    #[test]
    fn signed_sgd_err_empty_updates() {
        let mut state = SignedSgdState::new(3);
        let cfg = SignedSgdConfig {
            n_clients: 1,
            server_lr: 0.01,
        };
        assert!(matches!(
            SignedSgd::aggregate(&mut state, &[], &cfg),
            Err(FedError::EmptyClientList)
        ));
    }

    // ─── Test 16: wrong signed_grad length → DimensionMismatch ───────────────
    #[test]
    fn signed_sgd_err_dim_mismatch() {
        let mut state = SignedSgdState::new(4);
        let cfg = SignedSgdConfig {
            n_clients: 1,
            server_lr: 0.01,
        };
        let updates = vec![SignedSgdUpdate {
            client_id: 0,
            signed_grad: vec![1_i8, -1], // wrong length (expected 4)
            n_samples: 1,
        }];
        assert!(matches!(
            SignedSgd::aggregate(&mut state, &updates, &cfg),
            Err(FedError::DimensionMismatch {
                expected: 4,
                got: 2
            })
        ));
    }

    // ─── Test 17: n_samples == 0 → InvalidWeight ─────────────────────────────
    #[test]
    fn signed_sgd_err_zero_samples() {
        let mut state = SignedSgdState::new(2);
        let cfg = SignedSgdConfig {
            n_clients: 1,
            server_lr: 0.01,
        };
        let updates = vec![SignedSgdUpdate {
            client_id: 0,
            signed_grad: vec![1_i8, -1],
            n_samples: 0,
        }];
        assert!(matches!(
            SignedSgd::aggregate(&mut state, &updates, &cfg),
            Err(FedError::InvalidWeight { weight: 0.0 })
        ));
    }

    // ─── Test 18: 5 identical clients unanimous → params shift uniformly ──────
    #[test]
    fn signed_sgd_homogeneous_clients() {
        let init = vec![10.0_f32, -5.0, 0.0];
        let mut state = SignedSgdState::from_params(init.clone());
        let cfg = SignedSgdConfig {
            n_clients: 5,
            server_lr: 0.5,
        };
        // All 5 clients agree: [+1, -1, +1]
        let updates: Vec<_> = (0..5)
            .map(|id| SignedSgdUpdate {
                client_id: id,
                signed_grad: vec![1_i8, -1, 1],
                n_samples: 10,
            })
            .collect();
        SignedSgd::aggregate(&mut state, &updates, &cfg).expect("aggregate should succeed");
        // Unanimous majority: direction = [+1, -1, +1]
        // params -= 0.5 * direction
        assert!(
            (state.params[0] - 9.5).abs() < 1e-5,
            "params[0]={}",
            state.params[0]
        );
        assert!(
            (state.params[1] - (-4.5)).abs() < 1e-5,
            "params[1]={}",
            state.params[1]
        );
        assert!(
            (state.params[2] - (-0.5)).abs() < 1e-5,
            "params[2]={}",
            state.params[2]
        );
    }
}
