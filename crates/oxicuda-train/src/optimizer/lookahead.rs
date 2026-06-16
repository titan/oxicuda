//! Lookahead optimizer wrapper (Zhang et al., 2019).
//!
//! "Lookahead Optimizer: k steps forward, 1 step back", NeurIPS 2019.
//!
//! Lookahead maintains two sets of weights: **fast weights** updated by an
//! arbitrary inner optimizer (e.g. Adam, SGD) every step, and **slow weights**
//! that are nudged toward the fast weights every `k` steps:
//!
//! ```text
//! every step:        φ ← inner_optimizer(φ)            // fast weights
//! every k steps:     θ ← θ + α · (φ − θ)               // slow update
//!                    φ ← θ                             // resynchronise fast = slow
//! ```
//!
//! By interpolating along the trajectory of the inner optimizer, Lookahead
//! reduces variance and improves robustness to hyperparameters with negligible
//! overhead.  This wrapper is **optimizer-agnostic**: the caller runs whatever
//! inner step they like on `fast_params`, then calls [`Lookahead::step`] to
//! apply the slow-weight bookkeeping in-place.

use crate::error::{TrainError, TrainResult};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the [`Lookahead`] wrapper.
#[derive(Debug, Clone)]
pub struct LookaheadConfig {
    /// Synchronisation period: slow weights update every `k` fast steps.
    pub k: usize,
    /// Slow-weight step size α ∈ (0, 1].  `α = 1` copies fast weights exactly;
    /// `α → 0` keeps slow weights nearly frozen.
    pub alpha: f32,
}

impl Default for LookaheadConfig {
    fn default() -> Self {
        Self { k: 5, alpha: 0.5 }
    }
}

// ─── Optimizer ───────────────────────────────────────────────────────────────

/// Lookahead slow-weight tracker wrapping an externally-driven inner optimizer.
pub struct Lookahead {
    slow_weights: Vec<f32>,
    step_count: usize,
    config: LookaheadConfig,
}

impl Lookahead {
    /// Create a `Lookahead` wrapper initialised with the current parameters
    /// as the slow weights.
    ///
    /// # Errors
    ///
    /// * [`TrainError::NotSupported`] if `config.k == 0`.
    /// * [`TrainError::Internal`] if `config.alpha ∉ (0, 1]`.
    /// * [`TrainError::EmptyParams`] if `params` is empty.
    pub fn new(params: &[f32], config: LookaheadConfig) -> TrainResult<Self> {
        if params.is_empty() {
            return Err(TrainError::EmptyParams);
        }
        if config.k == 0 {
            return Err(TrainError::NotSupported {
                msg: "Lookahead k must be >= 1".into(),
            });
        }
        if !(config.alpha > 0.0 && config.alpha <= 1.0) {
            return Err(TrainError::Internal {
                msg: format!("Lookahead alpha must be in (0, 1], got {}", config.alpha),
            });
        }
        Ok(Self {
            slow_weights: params.to_vec(),
            step_count: 0,
            config,
        })
    }

    /// Record one fast-optimizer step and, every `k` steps, perform the slow
    /// update and resynchronise `fast_params` to the slow weights in-place.
    ///
    /// Call this **after** the inner optimizer has already updated
    /// `fast_params` for the current step.
    ///
    /// Returns `true` if a synchronisation occurred on this call.
    ///
    /// # Errors
    ///
    /// * [`TrainError::ParamCountMismatch`] if `fast_params.len()` differs from
    ///   the slow-weight length.
    pub fn step(&mut self, fast_params: &mut [f32]) -> TrainResult<bool> {
        if fast_params.len() != self.slow_weights.len() {
            return Err(TrainError::ParamCountMismatch {
                expected: self.slow_weights.len(),
                got: fast_params.len(),
            });
        }
        self.step_count += 1;
        if self.step_count % self.config.k == 0 {
            let alpha = self.config.alpha;
            for (slow, fast) in self.slow_weights.iter_mut().zip(fast_params.iter_mut()) {
                // θ ← θ + α·(φ − θ)
                *slow += alpha * (*fast - *slow);
                // φ ← θ
                *fast = *slow;
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Number of fast steps recorded so far.
    #[must_use]
    #[inline]
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Immutable view of the slow weights.
    #[must_use]
    #[inline]
    pub fn slow_weights(&self) -> &[f32] {
        &self.slow_weights
    }

    /// Synchronisation period `k`.
    #[must_use]
    #[inline]
    pub fn k(&self) -> usize {
        self.config.k
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(k: usize, alpha: f32) -> LookaheadConfig {
        LookaheadConfig { k, alpha }
    }

    #[test]
    fn slow_weights_init_to_params() {
        let p = vec![1.0_f32, 2.0, 3.0];
        let la = Lookahead::new(&p, cfg(5, 0.5)).expect("valid config");
        assert_eq!(la.slow_weights(), &p[..]);
    }

    #[test]
    fn no_sync_before_k() {
        let p = vec![0.0_f32; 4];
        let mut la = Lookahead::new(&p, cfg(5, 0.5)).expect("valid");
        let mut fast = vec![0.0_f32; 4];
        for _ in 0..4 {
            // Pretend inner optimizer advanced fast weights.
            for f in &mut fast {
                *f += 1.0;
            }
            let synced = la.step(&mut fast).expect("valid len");
            assert!(!synced, "should not sync before k steps");
        }
        // Fast weights untouched by Lookahead (no resync yet).
        assert!(fast.iter().all(|&v| (v - 4.0).abs() < 1e-6));
    }

    #[test]
    fn sync_at_k() {
        let p = vec![0.0_f32; 2];
        let mut la = Lookahead::new(&p, cfg(3, 0.5)).expect("valid");
        let mut fast = vec![0.0_f32; 2];
        let mut synced_at = None;
        for s in 1..=3 {
            for f in &mut fast {
                *f += 2.0;
            }
            if la.step(&mut fast).expect("valid") {
                synced_at = Some(s);
            }
        }
        assert_eq!(synced_at, Some(3), "sync should occur exactly at step k=3");
    }

    #[test]
    fn alpha_one_copies_fast() {
        let p = vec![0.0_f32; 2];
        let mut la = Lookahead::new(&p, cfg(1, 1.0)).expect("valid");
        let mut fast = vec![5.0_f32, 7.0];
        let synced = la.step(&mut fast).expect("valid");
        assert!(synced);
        // α=1 → slow becomes fast exactly.
        assert_eq!(la.slow_weights(), &[5.0, 7.0]);
        assert_eq!(fast, vec![5.0, 7.0]);
    }

    #[test]
    fn alpha_small_keeps_slow_near_frozen() {
        let p = vec![0.0_f32; 1];
        let mut la = Lookahead::new(&p, cfg(1, 0.01)).expect("valid");
        let mut fast = vec![100.0_f32];
        la.step(&mut fast).expect("valid");
        // slow = 0 + 0.01*(100 - 0) = 1.0; fast resynced to 1.0
        assert!((la.slow_weights()[0] - 1.0).abs() < 1e-5);
        assert!((fast[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn len_mismatch_error() {
        let p = vec![0.0_f32; 4];
        let mut la = Lookahead::new(&p, cfg(2, 0.5)).expect("valid");
        let mut fast = vec![0.0_f32; 3];
        assert!(matches!(
            la.step(&mut fast),
            Err(TrainError::ParamCountMismatch { .. })
        ));
    }

    #[test]
    fn k_zero_error() {
        let p = vec![0.0_f32; 4];
        assert!(matches!(
            Lookahead::new(&p, cfg(0, 0.5)),
            Err(TrainError::NotSupported { .. })
        ));
    }

    #[test]
    fn alpha_out_of_range_error() {
        let p = vec![0.0_f32; 4];
        assert!(matches!(
            Lookahead::new(&p, cfg(2, 0.0)),
            Err(TrainError::Internal { .. })
        ));
        assert!(matches!(
            Lookahead::new(&p, cfg(2, 1.5)),
            Err(TrainError::Internal { .. })
        ));
    }

    #[test]
    fn empty_params_error() {
        assert!(matches!(
            Lookahead::new(&[], cfg(2, 0.5)),
            Err(TrainError::EmptyParams)
        ));
    }

    #[test]
    fn multiple_syncs() {
        let p = vec![0.0_f32; 1];
        let mut la = Lookahead::new(&p, cfg(2, 0.5)).expect("valid");
        let mut fast = vec![0.0_f32; 1];
        let mut sync_count = 0;
        for _ in 0..10 {
            fast[0] += 1.0;
            if la.step(&mut fast).expect("valid") {
                sync_count += 1;
            }
        }
        // 10 steps / k=2 → 5 syncs.
        assert_eq!(sync_count, 5);
        assert_eq!(la.step_count(), 10);
    }

    #[test]
    fn returns_sync_flag_consistently() {
        let p = vec![0.0_f32; 1];
        let mut la = Lookahead::new(&p, cfg(4, 0.5)).expect("valid");
        let mut fast = vec![0.0_f32; 1];
        let flags: Vec<bool> = (1..=8)
            .map(|_| {
                fast[0] += 1.0;
                la.step(&mut fast).expect("valid")
            })
            .collect();
        assert_eq!(
            flags,
            vec![false, false, false, true, false, false, false, true]
        );
    }
}
