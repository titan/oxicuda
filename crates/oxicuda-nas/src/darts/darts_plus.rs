//! DARTS+: early-stopping criterion to prevent skip-connection collapse.
//!
//! Reference: Liang, Zhang, Sun, He, Huang, Zhuang & Li, "DARTS+: Improved
//! Differentiable Architecture Search with Early Stopping", 2019
//! (arXiv:1909.06035).
//!
//! # Background — the skip-connection collapse failure mode
//!
//! Vanilla [`crate::darts::cell::DartsCell`] training optimises the continuous
//! architecture parameters `α ∈ ℝ^{n_edges × n_ops}` jointly with the
//! convolutional weights via bi-level descent. As training proceeds, the
//! identity / `skip-connect` op accumulates a disproportionate share of the
//! softmax mass on more and more edges: it is parameter-free, so the inner
//! weight loop never penalises it, and the outer loop is happy to choose the
//! cheapest op once the network is large enough to fit the validation set.
//! Empirically, by the end of a long DARTS run, almost every edge selects
//! `skip-connect`, the derived cell degenerates to a residual stack, and the
//! retrained accuracy drops well below shorter-trained runs.
//!
//! # DARTS+ rule
//!
//! Liang et al. propose a simple, hyperparameter-light criterion to *freeze*
//! the architecture parameters before this collapse:
//!
//! 1. After each epoch compute, for the current `α`, the **skip-count** =
//!    number of edges whose argmax-over-ops equals the index of the
//!    `skip-connect` op.
//! 2. Maintain a counter `epochs_above` of *consecutive* epochs in which
//!    skip-count `>` `skip_threshold`. A sub-threshold epoch **resets** the
//!    counter to zero.
//! 3. Once `epochs_above` reaches `patience`, set `frozen = true` and stop
//!    updating `α`. The derived discrete cell at this freeze point is the
//!    final architecture.
//!
//! The full state machine is encapsulated by [`DartsPlusState`]; the caller
//! is responsible for calling [`DartsPlusState::update`] *once per epoch*
//! with the current row-major `α` of shape `[n_edges, n_ops]`.
//!
//! # Tie-breaking
//!
//! `argmax` is computed left-to-right with strict `>` so the **first** maximum
//! wins on ties. This matches Rust's `Iterator::position_max` convention (and
//! `slice::iter().enumerate().max_by(...)` with stable ordering).

use crate::error::{NasError, NasResult};

// ─── DartsPlusConfig ──────────────────────────────────────────────────────────

/// Configuration for [`DartsPlusState`].
///
/// All fields are validated by [`DartsPlusState::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DartsPlusConfig {
    /// Number of edges in the cell whose `α` parameters are being monitored.
    /// Must be `>= 1`.
    pub n_edges: usize,
    /// Number of candidate ops per edge (the second dimension of `α`).
    /// Must be `>= 1`.
    pub n_ops: usize,
    /// Index of the `skip-connect` op inside the [`crate::ops::primitives`]
    /// op list for this cell. Must satisfy `skip_op_idx < n_ops`.
    pub skip_op_idx: usize,
    /// Skip-count threshold. An epoch counts toward `epochs_above` only when
    /// `skip_count > skip_threshold` (strict inequality). Any non-negative
    /// `usize` is allowed; `0` means *any* edge selecting skip triggers.
    pub skip_threshold: usize,
    /// Number of consecutive above-threshold epochs that triggers freezing.
    /// Must be `>= 1` (a value of `0` would freeze before any epoch ran).
    pub patience: usize,
}

impl DartsPlusConfig {
    fn validate(&self) -> NasResult<()> {
        if self.n_edges == 0 {
            return Err(NasError::InvalidNumNodes { min: 1, got: 0 });
        }
        if self.n_ops == 0 {
            return Err(NasError::InvalidNumOps);
        }
        if self.skip_op_idx >= self.n_ops {
            return Err(NasError::InvalidRank {
                rank: self.skip_op_idx,
                dim: self.n_ops,
            });
        }
        if self.patience == 0 {
            return Err(NasError::InvalidArchEncoding);
        }
        Ok(())
    }
}

// ─── DartsPlusState ───────────────────────────────────────────────────────────

/// Persistent state of the DARTS+ early-stopping monitor.
///
/// Constructed via [`DartsPlusState::new`], driven by per-epoch calls to
/// [`DartsPlusState::update`]. Once [`DartsPlusState::is_frozen`] returns
/// `true`, the caller should stop applying gradient updates to `α` and
/// derive the discrete cell with
/// [`crate::darts::derive::derive_discrete_cell`].
#[derive(Debug, Clone)]
pub struct DartsPlusState {
    cfg: DartsPlusConfig,
    history: Vec<usize>,
    epochs_above: usize,
    frozen: bool,
}

impl DartsPlusState {
    /// Construct a fresh state with an empty history and the counter at zero.
    ///
    /// # Errors
    /// Propagates the `DartsPlusConfig::validate` errors.
    pub fn new(cfg: DartsPlusConfig) -> NasResult<Self> {
        cfg.validate()?;
        Ok(Self {
            cfg,
            history: Vec::new(),
            epochs_above: 0,
            frozen: false,
        })
    }

    /// Count, in the given `α`, the number of edges whose argmax-over-ops
    /// equals `cfg.skip_op_idx`.
    ///
    /// `alpha` is interpreted as row-major `[n_edges, n_ops]`. The argmax of
    /// each row is taken with strict-greater-than comparison, so the
    /// *first* maximum wins on ties.
    ///
    /// # Errors
    /// * [`NasError::DimensionMismatch`] if `alpha.len() != n_edges * n_ops`.
    pub fn count_skip(&self, alpha: &[f32]) -> NasResult<usize> {
        let expected = self.cfg.n_edges.saturating_mul(self.cfg.n_ops);
        if alpha.len() != expected {
            return Err(NasError::DimensionMismatch {
                expected,
                got: alpha.len(),
            });
        }
        let n_ops = self.cfg.n_ops;
        let skip = self.cfg.skip_op_idx;
        let mut count = 0usize;
        for e in 0..self.cfg.n_edges {
            let row_start = e * n_ops;
            // Linear argmax with first-max-wins tie-break. We rely on row
            // length == n_ops >= 1 (validated by `new`).
            let mut best_val = match alpha.get(row_start) {
                Some(&v) => v,
                None => return Err(NasError::InvalidArchEncoding),
            };
            let mut best_idx = 0usize;
            for k in 1..n_ops {
                let v = match alpha.get(row_start + k) {
                    Some(&v) => v,
                    None => return Err(NasError::InvalidArchEncoding),
                };
                if v > best_val {
                    best_val = v;
                    best_idx = k;
                }
            }
            if best_idx == skip {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Append one epoch's skip-count to the history, update the
    /// above-threshold counter, and flip `frozen` if `patience` is reached.
    ///
    /// Once `self.frozen == true`, every subsequent call is a *no-op* with
    /// respect to `epochs_above` (the counter is held constant) and `frozen`
    /// (it stays `true`). The current skip-count is still recorded in
    /// `history` so callers can plot the curve through the freeze point.
    ///
    /// # Errors
    /// Propagates [`DartsPlusState::count_skip`] errors.
    pub fn update(&mut self, alpha: &[f32]) -> NasResult<()> {
        let sk = self.count_skip(alpha)?;
        self.history.push(sk);
        if self.frozen {
            // Frozen: do not touch epochs_above and do not re-freeze.
            return Ok(());
        }
        if sk > self.cfg.skip_threshold {
            self.epochs_above = self.epochs_above.saturating_add(1);
            if self.epochs_above >= self.cfg.patience {
                self.frozen = true;
            }
        } else {
            // A sub-threshold epoch *resets* the consecutive counter.
            self.epochs_above = 0;
        }
        Ok(())
    }

    /// Whether the architecture parameters should be frozen.
    #[must_use]
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Current count of *consecutive* above-threshold epochs.
    #[must_use]
    pub fn epochs_above_threshold(&self) -> usize {
        self.epochs_above
    }

    /// Read-only view of every skip-count recorded so far, in epoch order.
    #[must_use]
    pub fn history(&self) -> &[usize] {
        &self.history
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(
        n_edges: usize,
        n_ops: usize,
        skip_idx: usize,
        threshold: usize,
        patience: usize,
    ) -> DartsPlusState {
        DartsPlusState::new(DartsPlusConfig {
            n_edges,
            n_ops,
            skip_op_idx: skip_idx,
            skip_threshold: threshold,
            patience,
        })
        .expect("test invariant: valid darts+ cfg")
    }

    /// Build an alpha where every row's argmax is `skip_idx`.
    fn alpha_all_skip(n_edges: usize, n_ops: usize, skip_idx: usize) -> Vec<f32> {
        let mut a = vec![0.0_f32; n_edges * n_ops];
        for e in 0..n_edges {
            for k in 0..n_ops {
                a[e * n_ops + k] = if k == skip_idx { 5.0 } else { -1.0 };
            }
        }
        a
    }

    /// Build an alpha where every row's argmax is some other (non-skip) op.
    fn alpha_no_skip(n_edges: usize, n_ops: usize, skip_idx: usize) -> Vec<f32> {
        let mut a = vec![0.0_f32; n_edges * n_ops];
        let alt = (skip_idx + 1) % n_ops;
        for e in 0..n_edges {
            for k in 0..n_ops {
                a[e * n_ops + k] = if k == alt { 5.0 } else { -1.0 };
            }
        }
        a
    }

    #[test]
    fn count_skip_all_rows_select_skip() {
        let st = mk(6, 4, 2, 1, 3);
        let a = alpha_all_skip(6, 4, 2);
        let c = st.count_skip(&a).expect("test invariant: count_skip");
        assert_eq!(c, 6);
    }

    #[test]
    fn count_skip_no_row_selects_skip() {
        let st = mk(5, 3, 1, 1, 3);
        let a = alpha_no_skip(5, 3, 1);
        let c = st.count_skip(&a).expect("test invariant: count_skip");
        assert_eq!(c, 0);
    }

    #[test]
    fn count_skip_hand_constructed_mixed() {
        // n_edges = 4, n_ops = 3, skip_idx = 0.
        // Row 0: argmax = 0 (skip).
        // Row 1: argmax = 2 (not skip).
        // Row 2: argmax = 0 (skip).
        // Row 3: argmax = 1 (not skip).
        // Expected count = 2.
        let st = mk(4, 3, 0, 0, 1);
        let a: Vec<f32> = vec![
            1.0, 0.0, -1.0, // row 0
            0.0, 0.5, 1.2, // row 1
            2.5, 1.0, 0.5, // row 2
            0.0, 1.0, -1.0, // row 3
        ];
        let c = st.count_skip(&a).expect("test invariant: count_skip");
        assert_eq!(c, 2);
    }

    #[test]
    fn new_state_initial_invariants() {
        let st = mk(3, 4, 0, 1, 2);
        assert!(!st.is_frozen());
        assert_eq!(st.epochs_above_threshold(), 0);
        assert!(st.history().is_empty());
    }

    #[test]
    fn update_below_threshold_keeps_counter_zero() {
        // threshold = 5, skip_count = 0 (no_skip alpha) → below threshold.
        let mut st = mk(4, 3, 0, 5, 3);
        let a = alpha_no_skip(4, 3, 0);
        for _ in 0..5 {
            st.update(&a).expect("test invariant: update");
            assert_eq!(st.epochs_above_threshold(), 0);
            assert!(!st.is_frozen());
        }
        assert_eq!(st.history().len(), 5);
        assert!(st.history().iter().all(|&c| c == 0));
    }

    #[test]
    fn update_above_threshold_increments_counter() {
        // threshold = 1, skip_count = 4 (all skip) → above.
        let mut st = mk(4, 3, 0, 1, 5);
        let a = alpha_all_skip(4, 3, 0);
        st.update(&a).expect("test invariant: update 1");
        assert_eq!(st.epochs_above_threshold(), 1);
        st.update(&a).expect("test invariant: update 2");
        assert_eq!(st.epochs_above_threshold(), 2);
        assert!(!st.is_frozen());
    }

    #[test]
    fn freeze_triggers_when_patience_reached() {
        // patience = 3, threshold = 1, all-skip → always above.
        let mut st = mk(4, 3, 0, 1, 3);
        let a = alpha_all_skip(4, 3, 0);
        st.update(&a).expect("test invariant: u1");
        assert!(!st.is_frozen());
        st.update(&a).expect("test invariant: u2");
        assert!(!st.is_frozen());
        st.update(&a).expect("test invariant: u3");
        assert!(st.is_frozen());
        assert_eq!(st.epochs_above_threshold(), 3);
    }

    #[test]
    fn once_frozen_subsequent_updates_are_noop() {
        let mut st = mk(4, 3, 0, 1, 2);
        let a_above = alpha_all_skip(4, 3, 0);
        let a_below = alpha_no_skip(4, 3, 0);
        st.update(&a_above).expect("test invariant: u1");
        st.update(&a_above).expect("test invariant: u2");
        assert!(st.is_frozen());
        let frozen_counter = st.epochs_above_threshold();
        // Below threshold *after* freeze must not reset the counter.
        st.update(&a_below).expect("test invariant: u3");
        assert!(st.is_frozen());
        assert_eq!(st.epochs_above_threshold(), frozen_counter);
        // Above threshold *after* freeze must not increment.
        st.update(&a_above).expect("test invariant: u4");
        assert!(st.is_frozen());
        assert_eq!(st.epochs_above_threshold(), frozen_counter);
        // History continues to grow regardless.
        assert_eq!(st.history().len(), 4);
    }

    #[test]
    fn sub_threshold_epoch_resets_counter() {
        // patience = 4 (so we don't accidentally freeze). After two above
        // epochs, one below epoch must reset to 0.
        let mut st = mk(4, 3, 0, 1, 4);
        let a_above = alpha_all_skip(4, 3, 0);
        let a_below = alpha_no_skip(4, 3, 0);
        st.update(&a_above).expect("test invariant: u1");
        st.update(&a_above).expect("test invariant: u2");
        assert_eq!(st.epochs_above_threshold(), 2);
        st.update(&a_below).expect("test invariant: u3");
        assert_eq!(st.epochs_above_threshold(), 0);
        assert!(!st.is_frozen());
    }

    #[test]
    fn history_records_each_update_in_order() {
        // Verify that the recorded skip-count for each epoch matches
        // count_skip on the alpha we pass in.
        let mut st = mk(3, 3, 0, 10, 5);
        let a_skip = alpha_all_skip(3, 3, 0); // skip count = 3
        let a_no = alpha_no_skip(3, 3, 0); // skip count = 0
        st.update(&a_skip).expect("test invariant: u1");
        st.update(&a_no).expect("test invariant: u2");
        st.update(&a_skip).expect("test invariant: u3");
        assert_eq!(st.history(), &[3, 0, 3]);
    }

    #[test]
    fn deterministic_no_rng_use() {
        // Two parallel runs on the same alpha sequence must agree exactly
        // (and we are not using any rng).
        let mut sa = mk(3, 3, 0, 1, 2);
        let mut sb = mk(3, 3, 0, 1, 2);
        let a_above = alpha_all_skip(3, 3, 0);
        let a_below = alpha_no_skip(3, 3, 0);
        for alpha in [&a_above, &a_above, &a_below, &a_above, &a_above, &a_above] {
            sa.update(alpha).expect("test invariant: a");
            sb.update(alpha).expect("test invariant: b");
            assert_eq!(sa.is_frozen(), sb.is_frozen());
            assert_eq!(sa.epochs_above_threshold(), sb.epochs_above_threshold());
            assert_eq!(sa.history(), sb.history());
        }
    }

    #[test]
    fn err_count_skip_wrong_length() {
        let st = mk(4, 3, 0, 1, 2);
        let bad = vec![0.0_f32; 11]; // expected 12
        let r = st.count_skip(&bad);
        assert!(matches!(
            r,
            Err(NasError::DimensionMismatch {
                expected: 12,
                got: 11,
            })
        ));
    }

    #[test]
    fn err_n_edges_zero() {
        let r = DartsPlusState::new(DartsPlusConfig {
            n_edges: 0,
            n_ops: 4,
            skip_op_idx: 0,
            skip_threshold: 1,
            patience: 2,
        });
        assert!(matches!(
            r,
            Err(NasError::InvalidNumNodes { min: 1, got: 0 })
        ));
    }

    #[test]
    fn err_n_ops_zero() {
        let r = DartsPlusState::new(DartsPlusConfig {
            n_edges: 4,
            n_ops: 0,
            skip_op_idx: 0,
            skip_threshold: 1,
            patience: 2,
        });
        assert!(matches!(r, Err(NasError::InvalidNumOps)));
    }

    #[test]
    fn err_skip_op_idx_out_of_range() {
        let r = DartsPlusState::new(DartsPlusConfig {
            n_edges: 4,
            n_ops: 3,
            skip_op_idx: 3,
            skip_threshold: 1,
            patience: 2,
        });
        assert!(matches!(r, Err(NasError::InvalidRank { rank: 3, dim: 3 })));
    }

    #[test]
    fn err_patience_zero() {
        let r = DartsPlusState::new(DartsPlusConfig {
            n_edges: 4,
            n_ops: 3,
            skip_op_idx: 0,
            skip_threshold: 1,
            patience: 0,
        });
        assert!(matches!(r, Err(NasError::InvalidArchEncoding)));
    }

    #[test]
    fn argmax_tie_break_first_max_wins_for_skip_detection() {
        // skip_idx = 0. Row's first op tied with op 2 at the maximum.
        // First-max-wins => argmax = 0 = skip_idx. Expected skip_count = 1.
        let st = mk(1, 3, 0, 0, 1);
        let a: Vec<f32> = vec![1.0, 0.5, 1.0]; // ties between idx 0 and 2
        let c = st.count_skip(&a).expect("test invariant: count_skip");
        assert_eq!(c, 1);
    }

    #[test]
    fn argmax_tie_break_skip_loses_when_earlier_op_ties() {
        // skip_idx = 2. Row has tied max between idx 0 and idx 2.
        // First-max-wins => argmax = 0, not skip. Expected skip_count = 0.
        let st = mk(1, 3, 2, 0, 1);
        let a: Vec<f32> = vec![1.0, 0.5, 1.0];
        let c = st.count_skip(&a).expect("test invariant: count_skip");
        assert_eq!(c, 0);
    }

    #[test]
    fn threshold_zero_any_skip_edge_triggers() {
        // threshold = 0 means *any* skip-edge counts as above-threshold,
        // because the predicate is strict (> 0).
        let mut st = mk(3, 3, 0, 0, 2);
        let mut a = alpha_no_skip(3, 3, 0);
        // Flip the first edge's argmax to skip.
        a[0] = 10.0; // row 0, op 0 → now skip
        a[1] = -1.0;
        a[2] = -1.0;
        let c = st.count_skip(&a).expect("test invariant: count_skip");
        assert_eq!(c, 1);
        st.update(&a).expect("test invariant: update");
        assert_eq!(st.epochs_above_threshold(), 1);
        st.update(&a).expect("test invariant: update");
        assert!(st.is_frozen());
    }
}
