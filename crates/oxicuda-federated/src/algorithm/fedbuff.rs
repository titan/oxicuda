//! FedBuff: Federated Learning with Buffered Asynchronous Aggregation.
//!
//! Nguyen, J. et al. (2022). "Federated Learning with Buffered Asynchronous
//! Aggregation." *AISTATS 2022*.
//!
//! The server maintains a buffer of the K most-recent client updates. Clients
//! submit their local deltas asynchronously (each tagged with the global
//! round number `client_round` they were trained on). When the buffer is
//! full the server performs one staleness-weighted aggregation step,
//! applies the result with global learning rate η_g, advances the round,
//! and empties the buffer.
//!
//! ## Aggregation rule
//!
//! For each update `i` in the buffer at server round `r`:
//!
//! ```text
//! staleness_i = max(0, r − update_i.client_round)
//! weight_i    = 1 / (1 + α · staleness_i)
//! ```
//!
//! The new global model is
//!
//! ```text
//! θ_{r+1} = θ_r − η_g · (Σ_i weight_i · delta_i) / (Σ_i weight_i).
//! ```
//!
//! With `α = 0` all updates are treated equally (uniform mean). With α > 0,
//! older updates are down-weighted, which matches the variance-reduction
//! behaviour of FedAsync (Xie et al., 2019).

use crate::error::{FedError, FedResult};
use std::collections::VecDeque;

/// Configuration for FedBuff buffered asynchronous aggregation.
#[derive(Debug, Clone)]
pub struct FedBuffConfig {
    /// Number of buffered client updates required to trigger one
    /// aggregation step. Must be ≥ 1.
    pub buffer_size: usize,
    /// Staleness penalty α ≥ 0. Weight of an update with staleness s is
    /// `1 / (1 + α · s)`.
    pub staleness_alpha: f32,
    /// Global (server) learning rate η_g applied to the aggregated delta.
    pub global_learning_rate: f32,
}

/// A single asynchronous client submission.
#[derive(Debug, Clone)]
pub struct BufferedUpdate {
    /// Parameter difference `θ_global − θ_local_post_training` (a "negative
    /// gradient" in the FedAvg sense; positive entries pull θ toward the
    /// local optimum).
    pub client_delta: Vec<f32>,
    /// The global round number the client started training from.
    pub client_round: usize,
}

/// Server-side state for FedBuff.
///
/// Owns the global parameter vector, the per-round counter, the FIFO
/// buffer of pending updates, and the configuration.
#[derive(Debug, Clone)]
pub struct FedBuffState {
    /// Current global model parameters θ_r.
    pub global_params: Vec<f32>,
    /// Number of completed aggregation rounds.
    pub round: usize,
    /// FIFO of pending client submissions.
    pub buffer: VecDeque<BufferedUpdate>,
    /// Aggregation configuration.
    pub cfg: FedBuffConfig,
}

impl FedBuffState {
    /// Construct a validated FedBuff server state.
    ///
    /// # Errors
    /// - [`FedError::EmptyClientList`] if `global_params` is empty.
    /// - [`FedError::InsufficientClients`] if `buffer_size = 0`.
    /// - [`FedError::InvalidWeight`] if `staleness_alpha < 0` or non-finite.
    /// - [`FedError::InvalidNoiseMultiplier`] if `global_learning_rate` is
    ///   not finite.
    pub fn new(global_params: Vec<f32>, cfg: FedBuffConfig) -> FedResult<Self> {
        if global_params.is_empty() {
            return Err(FedError::EmptyClientList);
        }
        if cfg.buffer_size == 0 {
            return Err(FedError::InsufficientClients { min: 1, got: 0 });
        }
        if !(cfg.staleness_alpha >= 0.0 && cfg.staleness_alpha.is_finite()) {
            return Err(FedError::InvalidWeight {
                weight: cfg.staleness_alpha,
            });
        }
        if !cfg.global_learning_rate.is_finite() {
            return Err(FedError::InvalidNoiseMultiplier);
        }
        Ok(Self {
            global_params,
            round: 0,
            buffer: VecDeque::with_capacity(cfg.buffer_size),
            cfg,
        })
    }

    /// Submit one asynchronous client update.
    ///
    /// Pushes the update into the FIFO buffer; if the buffer reaches
    /// `buffer_size`, immediately runs [`Self::aggregate`] and returns
    /// `true`. Otherwise returns `false`.
    ///
    /// # Errors
    /// - [`FedError::DimensionMismatch`] if `update.client_delta.len() ≠
    ///   global_params.len()`.
    pub fn submit(&mut self, update: BufferedUpdate) -> FedResult<bool> {
        if update.client_delta.len() != self.global_params.len() {
            return Err(FedError::DimensionMismatch {
                expected: self.global_params.len(),
                got: update.client_delta.len(),
            });
        }
        self.buffer.push_back(update);
        if self.buffer.len() >= self.cfg.buffer_size {
            self.aggregate()?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Aggregate the currently buffered updates into the global model.
    ///
    /// ```text
    /// θ_new = θ_old − η_g · (Σ_i w_i · delta_i) / (Σ_i w_i)
    /// where w_i = 1 / (1 + α · staleness_i),
    /// staleness_i = max(0, self.round − update.client_round).
    /// ```
    ///
    /// Empties the buffer and increments `round`. If the weighted sum of
    /// weights collapses to (near-)zero (cannot occur for finite non-negative
    /// staleness, but defended against), returns an error.
    ///
    /// # Errors
    /// - [`FedError::EmptyClientList`] if the buffer is empty.
    /// - [`FedError::InvalidWeight`] if `Σ w_i ≤ 0` (numerical guard).
    pub fn aggregate(&mut self) -> FedResult<()> {
        if self.buffer.is_empty() {
            return Err(FedError::EmptyClientList);
        }
        let n_params = self.global_params.len();
        let alpha = self.cfg.staleness_alpha as f64;
        let r = self.round;

        let mut weighted = vec![0.0_f64; n_params];
        let mut sum_w = 0.0_f64;

        for upd in self.buffer.iter() {
            // Re-check delta length defensively, even though `submit` ensures it.
            if upd.client_delta.len() != n_params {
                return Err(FedError::DimensionMismatch {
                    expected: n_params,
                    got: upd.client_delta.len(),
                });
            }
            let staleness = if upd.client_round > r {
                0.0_f64
            } else {
                (r - upd.client_round) as f64
            };
            let w = 1.0 / (1.0 + alpha * staleness);
            sum_w += w;
            for (acc, &d) in weighted.iter_mut().zip(upd.client_delta.iter()) {
                *acc += w * d as f64;
            }
        }

        if sum_w <= 0.0 || !sum_w.is_finite() {
            return Err(FedError::InvalidWeight {
                weight: sum_w as f32,
            });
        }

        let eta = self.cfg.global_learning_rate as f64;
        let inv_sum = 1.0 / sum_w;
        for (p, &acc) in self.global_params.iter_mut().zip(weighted.iter()) {
            *p = (*p as f64 - eta * acc * inv_sum) as f32;
        }

        self.buffer.clear();
        self.round += 1;
        Ok(())
    }

    /// Return `true` iff the buffer has reached `buffer_size` and is
    /// therefore "full" / ready to aggregate.
    #[must_use]
    pub fn buffer_full(&self) -> bool {
        self.buffer.len() >= self.cfg.buffer_size
    }

    /// Return the current number of buffered updates.
    #[must_use]
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(buffer_size: usize, alpha: f32, eta: f32) -> FedBuffConfig {
        FedBuffConfig {
            buffer_size,
            staleness_alpha: alpha,
            global_learning_rate: eta,
        }
    }

    fn make_update(delta: Vec<f32>, client_round: usize) -> BufferedUpdate {
        BufferedUpdate {
            client_delta: delta,
            client_round,
        }
    }

    // ── Test 1: new_buffer_empty ─────────────────────────────────────────────
    #[test]
    fn new_buffer_empty() {
        let s = FedBuffState::new(vec![0.0_f32; 4], cfg(3, 0.0, 1.0))
            .expect("test invariant: valid state");
        assert_eq!(s.buffer_len(), 0);
        assert!(!s.buffer_full());
        assert_eq!(s.round, 0);
        assert_eq!(s.global_params.len(), 4);
    }

    // ── Test 2: submit_adds_to_buffer ────────────────────────────────────────
    #[test]
    fn submit_adds_to_buffer() {
        let mut s = FedBuffState::new(vec![0.0_f32; 2], cfg(3, 0.0, 1.0)).expect("test invariant");
        let triggered = s.submit(make_update(vec![1.0, 1.0], 0)).expect("submit");
        assert!(!triggered);
        assert_eq!(s.buffer_len(), 1);
        let triggered = s.submit(make_update(vec![2.0, 2.0], 0)).expect("submit");
        assert!(!triggered);
        assert_eq!(s.buffer_len(), 2);
    }

    // ── Test 3: buffer_full_at_size ──────────────────────────────────────────
    #[test]
    fn buffer_full_at_size() {
        // Use buffer_size=4 and submit 3 manually so we can observe `buffer_full`
        // before the auto-aggregate kicks in.
        let mut s = FedBuffState::new(vec![0.0_f32; 2], cfg(4, 0.0, 1.0)).expect("state");
        for _ in 0..3 {
            assert!(
                !s.submit(make_update(vec![0.0, 0.0], 0)).expect("submit"),
                "should not trigger"
            );
        }
        assert!(!s.buffer_full());
        assert_eq!(s.buffer_len(), 3);
    }

    // ── Test 4: submit_triggers_aggregate ────────────────────────────────────
    #[test]
    fn submit_triggers_aggregate() {
        let mut s = FedBuffState::new(vec![0.0_f32; 2], cfg(2, 0.0, 1.0)).expect("state");
        let r1 = s.submit(make_update(vec![1.0, 1.0], 0)).expect("submit");
        assert!(!r1);
        assert_eq!(s.round, 0);
        let r2 = s.submit(make_update(vec![1.0, 1.0], 0)).expect("submit");
        assert!(r2, "second submit should trigger aggregate");
        assert_eq!(s.buffer_len(), 0, "buffer should be cleared");
        assert_eq!(s.round, 1, "round should increment");
    }

    // ── Test 5: aggregate_constant_delta_direction ───────────────────────────
    #[test]
    fn aggregate_constant_delta_direction() {
        // With α=0 and all deltas equal to a constant c, the weighted mean
        // is c, so θ_new = θ_old − η · c.
        let mut s = FedBuffState::new(vec![5.0_f32, 5.0], cfg(2, 0.0, 0.5)).expect("state");
        s.submit(make_update(vec![1.0, 1.0], 0)).expect("submit");
        let triggered = s.submit(make_update(vec![1.0, 1.0], 0)).expect("submit");
        assert!(triggered);
        // θ_new = 5.0 − 0.5·1.0 = 4.5
        for v in &s.global_params {
            assert!(
                (*v - 4.5).abs() < 1e-5,
                "expected 4.5, got {v} (params={:?})",
                s.global_params
            );
        }
    }

    // ── Test 6: aggregate_uniform_weights_when_alpha_zero ────────────────────
    #[test]
    fn aggregate_uniform_weights_when_alpha_zero() {
        // α=0 ⇒ all weights = 1 ⇒ θ_new − θ_old = −η · mean(deltas).
        let mut s = FedBuffState::new(vec![0.0_f32], cfg(3, 0.0, 1.0)).expect("state");
        // Server is on round 0; submit three updates from different rounds.
        s.submit(make_update(vec![3.0], 0)).expect("s1");
        s.submit(make_update(vec![6.0], 0)).expect("s2");
        let triggered = s.submit(make_update(vec![9.0], 0)).expect("s3");
        assert!(triggered);
        // mean = (3+6+9)/3 = 6 → θ_new = 0 − 1·6 = −6
        assert!(
            (s.global_params[0] - (-6.0)).abs() < 1e-4,
            "got {}",
            s.global_params[0]
        );
    }

    // ── Test 7: higher_staleness_lower_weight ────────────────────────────────
    #[test]
    fn higher_staleness_lower_weight() {
        // Advance server to round 5, then aggregate a buffer that mixes
        // a fresh update (client_round=5, staleness=0) and a stale one
        // (client_round=0, staleness=5).
        let mut s = FedBuffState::new(vec![0.0_f32], cfg(2, 1.0, 1.0)).expect("state");
        s.round = 5;
        // Fresh update has delta=0, stale update has delta=10.
        s.submit(make_update(vec![0.0], 5)).expect("fresh");
        let triggered = s.submit(make_update(vec![10.0], 0)).expect("stale");
        assert!(triggered);
        // w_fresh = 1, w_stale = 1/(1+5) = 1/6 ⇒ weighted mean = (0+10/6)/(1+1/6) = (10/6)/(7/6) = 10/7.
        // θ_new = 0 − 1 · 10/7 = −10/7 ≈ −1.4286
        let expected = -10.0_f32 / 7.0;
        assert!(
            (s.global_params[0] - expected).abs() < 1e-4,
            "got {}, expected {expected}",
            s.global_params[0]
        );
    }

    // ── Test 8: round_increments_after_aggregate ─────────────────────────────
    #[test]
    fn round_increments_after_aggregate() {
        let mut s = FedBuffState::new(vec![0.0_f32; 3], cfg(1, 0.0, 1.0)).expect("state");
        assert_eq!(s.round, 0);
        // buffer_size=1 ⇒ each submit triggers aggregate.
        for r in 1..=4 {
            let t = s
                .submit(make_update(vec![0.1, 0.1, 0.1], 0))
                .expect("submit");
            assert!(t);
            assert_eq!(s.round, r);
        }
    }

    // ── Test 9: deterministic_aggregation ────────────────────────────────────
    #[test]
    fn deterministic_aggregation() {
        // Two identical sequences of submits yield bit-identical state.
        let make = || {
            let mut s = FedBuffState::new(vec![0.0_f32; 2], cfg(3, 0.5, 1.0)).expect("state");
            s.submit(make_update(vec![1.0, 0.5], 0)).expect("s1");
            s.submit(make_update(vec![2.0, 1.0], 0)).expect("s2");
            s.submit(make_update(vec![3.0, 1.5], 0)).expect("s3");
            s
        };
        let a = make();
        let b = make();
        assert_eq!(a.global_params, b.global_params);
        assert_eq!(a.round, b.round);
        assert_eq!(a.buffer_len(), b.buffer_len());
    }

    // ── Test 10: err_buffer_size_zero ────────────────────────────────────────
    #[test]
    fn err_buffer_size_zero() {
        assert!(matches!(
            FedBuffState::new(vec![0.0_f32; 2], cfg(0, 0.0, 1.0)),
            Err(FedError::InsufficientClients { .. })
        ));
    }

    // ── Test 11: err_negative_alpha ──────────────────────────────────────────
    #[test]
    fn err_negative_alpha() {
        assert!(matches!(
            FedBuffState::new(vec![0.0_f32; 2], cfg(3, -0.1, 1.0)),
            Err(FedError::InvalidWeight { .. })
        ));
        assert!(matches!(
            FedBuffState::new(vec![0.0_f32; 2], cfg(3, f32::NAN, 1.0)),
            Err(FedError::InvalidWeight { .. })
        ));
    }

    // ── Test 12: err_delta_length_mismatch ───────────────────────────────────
    #[test]
    fn err_delta_length_mismatch() {
        let mut s = FedBuffState::new(vec![0.0_f32; 4], cfg(2, 0.0, 1.0)).expect("state");
        let r = s.submit(make_update(vec![0.1, 0.2], 0));
        assert!(matches!(r, Err(FedError::DimensionMismatch { .. })));
        assert_eq!(s.buffer_len(), 0, "failed submit must not add");
    }

    // ── Test 13: err_empty_global_params ─────────────────────────────────────
    #[test]
    fn err_empty_global_params() {
        assert!(matches!(
            FedBuffState::new(vec![], cfg(2, 0.0, 1.0)),
            Err(FedError::EmptyClientList)
        ));
    }

    // ── Test 14: err_aggregate_on_empty_buffer ───────────────────────────────
    #[test]
    fn err_aggregate_on_empty_buffer() {
        let mut s = FedBuffState::new(vec![0.0_f32; 2], cfg(3, 0.0, 1.0)).expect("state");
        assert!(matches!(s.aggregate(), Err(FedError::EmptyClientList)));
    }

    // ── Test 15: buffer_len_tracking ─────────────────────────────────────────
    #[test]
    fn buffer_len_tracking() {
        let mut s = FedBuffState::new(vec![0.0_f32; 1], cfg(4, 0.0, 1.0)).expect("state");
        assert_eq!(s.buffer_len(), 0);
        s.submit(make_update(vec![0.0], 0)).expect("1");
        assert_eq!(s.buffer_len(), 1);
        s.submit(make_update(vec![0.0], 0)).expect("2");
        assert_eq!(s.buffer_len(), 2);
        s.submit(make_update(vec![0.0], 0)).expect("3");
        assert_eq!(s.buffer_len(), 3);
        let triggered = s.submit(make_update(vec![0.0], 0)).expect("4");
        assert!(triggered);
        assert_eq!(s.buffer_len(), 0, "buffer cleared after aggregation");
    }

    // ── Test 16: client_round_greater_than_round_clamped ─────────────────────
    #[test]
    fn client_round_greater_than_round_clamped() {
        // If the client somehow reports a future round, staleness clamps to 0.
        let mut s = FedBuffState::new(vec![0.0_f32], cfg(1, 100.0, 1.0)).expect("state");
        // staleness_alpha=100 would crush any non-zero staleness; we want
        // to see weight=1 because future-round ⇒ staleness=0.
        let triggered = s.submit(make_update(vec![5.0], 999)).expect("submit");
        assert!(triggered);
        // weighted mean = 5.0 (weight=1), θ_new = 0 − 1·5 = −5
        assert!(
            (s.global_params[0] - (-5.0)).abs() < 1e-5,
            "got {}",
            s.global_params[0]
        );
    }

    // ── Test 17: multiple_aggregation_cycles ─────────────────────────────────
    #[test]
    fn multiple_aggregation_cycles() {
        // buffer_size=2, η=1, α=0: each pair of submits subtracts the
        // mean of that pair.
        let mut s = FedBuffState::new(vec![0.0_f32], cfg(2, 0.0, 1.0)).expect("state");
        // Round 1: mean([1, 3]) = 2 ⇒ θ = 0 − 2 = −2
        s.submit(make_update(vec![1.0], 0)).expect("a");
        s.submit(make_update(vec![3.0], 0)).expect("b");
        assert_eq!(s.round, 1);
        assert!((s.global_params[0] - (-2.0)).abs() < 1e-5);
        // Round 2: mean([2, 6]) = 4 ⇒ θ = −2 − 4 = −6
        s.submit(make_update(vec![2.0], 1)).expect("c");
        s.submit(make_update(vec![6.0], 1)).expect("d");
        assert_eq!(s.round, 2);
        assert!((s.global_params[0] - (-6.0)).abs() < 1e-5);
    }

    // ── Test 18: weighted_mean_hand_check ────────────────────────────────────
    #[test]
    fn weighted_mean_hand_check() {
        // Three updates with α=0.5:
        //   (delta=2, staleness=0) ⇒ w = 1
        //   (delta=4, staleness=2) ⇒ w = 1/(1+1) = 0.5
        //   (delta=6, staleness=4) ⇒ w = 1/(1+2) = 1/3
        // Σw = 1 + 0.5 + 1/3 = 11/6
        // Σw·d = 2 + 2 + 2 = 6
        // weighted mean = 6 / (11/6) = 36/11 ≈ 3.2727
        // θ_new = 0 − 1·36/11 = −36/11
        let mut s = FedBuffState::new(vec![0.0_f32], cfg(3, 0.5, 1.0)).expect("state");
        s.round = 4;
        s.submit(make_update(vec![2.0], 4)).expect("fresh");
        s.submit(make_update(vec![4.0], 2)).expect("mid");
        let triggered = s.submit(make_update(vec![6.0], 0)).expect("stale");
        assert!(triggered);
        let expected = -36.0_f32 / 11.0;
        assert!(
            (s.global_params[0] - expected).abs() < 1e-4,
            "got {}, expected {expected}",
            s.global_params[0]
        );
    }

    // ── Test 19: err_global_lr_non_finite ────────────────────────────────────
    #[test]
    fn err_global_lr_non_finite() {
        assert!(matches!(
            FedBuffState::new(vec![0.0_f32; 2], cfg(3, 0.0, f32::INFINITY)),
            Err(FedError::InvalidNoiseMultiplier)
        ));
        assert!(matches!(
            FedBuffState::new(vec![0.0_f32; 2], cfg(3, 0.0, f32::NAN)),
            Err(FedError::InvalidNoiseMultiplier)
        ));
    }

    // ── Test 20: buffer_full_property_after_submit ───────────────────────────
    #[test]
    fn buffer_full_property_after_submit() {
        // After triggering an aggregation the buffer is empty so
        // `buffer_full()` is false even if buffer_size was 1.
        let mut s = FedBuffState::new(vec![0.0_f32], cfg(1, 0.0, 1.0)).expect("state");
        assert!(!s.buffer_full());
        s.submit(make_update(vec![0.5], 0)).expect("submit");
        assert!(!s.buffer_full(), "buffer should be empty after aggregate");
    }
}
