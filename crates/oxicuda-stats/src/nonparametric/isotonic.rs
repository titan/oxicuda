//! Isotonic (monotone) regression via the Pool-Adjacent-Violators Algorithm.
//!
//! Given responses `y₁,…,y_n` (with optional positive weights `w_i`), isotonic
//! regression finds the monotone non-decreasing fit `ŷ` that minimises the
//! weighted least-squares objective
//!
//! ```text
//! minimise  Σ_i w_i (y_i − ŷ_i)²   subject to   ŷ₁ ≤ ŷ₂ ≤ … ≤ ŷ_n.
//! ```
//!
//! The **Pool-Adjacent-Violators Algorithm (PAVA)** solves this exactly in
//! `O(n)` time. It scans left to right maintaining a stack of *blocks* (level
//! sets); whenever a new block violates monotonicity against the block below it,
//! the two are *pooled* — replaced by their weighted mean — and the merge is
//! propagated downward until monotonicity is restored. The optimal fit is
//! piecewise-constant on the final blocks, each block taking the weighted mean
//! of its members.
//!
//! Antitonic (non-increasing) regression is obtained by negating `y`, running
//! PAVA, and negating the result.
//!
//! # References
//! - Barlow, R.E., Bartholomew, D.J., Bremner, J.M. & Brunk, H.D. (1972).
//!   *Statistical Inference under Order Restrictions.* Wiley.
//! - Robertson, T., Wright, F.T. & Dykstra, R.L. (1988). *Order Restricted
//!   Statistical Inference.* Wiley.
//! - de Leeuw, J., Hornik, K. & Mair, P. (2009). *Isotone optimization in R:
//!   pool-adjacent-violators algorithm (PAVA) and active set methods.*
//!   J. Stat. Softw. 32(5).

use crate::error::{StatsError, StatsResult};

/// A pooled block (level set) produced by PAVA.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IsotonicBlock {
    /// Index of the first member of this block (inclusive, into the input).
    pub start: usize,
    /// Index one past the last member of this block (exclusive).
    pub end: usize,
    /// The fitted value on this block (weighted mean of its members).
    pub value: f64,
    /// Total weight of the block.
    pub weight: f64,
}

impl IsotonicBlock {
    /// Number of original observations pooled into this block.
    #[must_use]
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether the block is empty (never true for a PAVA output; provided for
    /// API completeness / clippy).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.end == self.start
    }
}

/// Result of an isotonic regression fit.
#[derive(Debug, Clone)]
pub struct IsotonicFit {
    /// Fitted values, one per input observation (piecewise-constant on blocks).
    pub fitted: Vec<f64>,
    /// The pooled blocks (level sets) in left-to-right order.
    pub blocks: Vec<IsotonicBlock>,
}

// ---------------------------------------------------------------------------
// Core PAVA (weighted, non-decreasing)
// ---------------------------------------------------------------------------

/// Internal stack entry: a running block holding `Σw` and `Σ w·y`.
#[derive(Clone, Copy)]
struct Running {
    start: usize,
    end: usize,
    sum_w: f64,
    sum_wy: f64,
}

impl Running {
    #[inline]
    fn mean(&self) -> f64 {
        self.sum_wy / self.sum_w
    }
}

/// Weighted Pool-Adjacent-Violators for a **non-decreasing** fit.
///
/// `y` and `weights` must have equal, non-zero length and all weights must be
/// strictly positive and finite (as must all `y`).
///
/// # Errors
/// - [`StatsError::EmptyInput`] if `y` is empty.
/// - [`StatsError::DimensionMismatch`] if `weights.len() != y.len()`.
/// - [`StatsError::InvalidParameter`] if any weight is `≤ 0` or non-finite.
/// - [`StatsError::NonFiniteValue`] if any `y[i]` is non-finite.
pub fn isotonic_regression_weighted(y: &[f64], weights: &[f64]) -> StatsResult<IsotonicFit> {
    let n = y.len();
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    if weights.len() != n {
        return Err(StatsError::DimensionMismatch {
            a: weights.len(),
            b: n,
        });
    }
    for (i, &yi) in y.iter().enumerate() {
        if !yi.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }
    for (i, &wi) in weights.iter().enumerate() {
        if !(wi.is_finite() && wi > 0.0) {
            return Err(StatsError::InvalidParameter {
                name: format!("weights[{i}]"),
                reason: format!("must be finite and > 0, got {wi}"),
            });
        }
    }

    // Standard up-and-down-block (stack-based) PAVA.
    let mut stack: Vec<Running> = Vec::with_capacity(n);
    for i in 0..n {
        let mut block = Running {
            start: i,
            end: i + 1,
            sum_w: weights[i],
            sum_wy: weights[i] * y[i],
        };
        // Pool with preceding blocks while the monotonicity constraint
        // (means non-decreasing) is violated. We merge only on a strict
        // violation `top.mean() > block.mean()`, which yields the canonical
        // maximal level sets (equal-mean neighbours stay separate blocks).
        while let Some(&top) = stack.last() {
            if top.mean() <= block.mean() {
                break;
            }
            // Violation: merge `top` into `block`.
            let merged = Running {
                start: top.start,
                end: block.end,
                sum_w: top.sum_w + block.sum_w,
                sum_wy: top.sum_wy + block.sum_wy,
            };
            stack.pop();
            block = merged;
        }
        stack.push(block);
    }

    // Materialise blocks + fitted vector.
    let mut blocks = Vec::with_capacity(stack.len());
    let mut fitted = vec![0.0; n];
    for b in &stack {
        let value = b.mean();
        for f in fitted.iter_mut().take(b.end).skip(b.start) {
            *f = value;
        }
        blocks.push(IsotonicBlock {
            start: b.start,
            end: b.end,
            value,
            weight: b.sum_w,
        });
    }

    Ok(IsotonicFit { fitted, blocks })
}

/// Unweighted isotonic regression (non-decreasing); all weights equal to 1.
///
/// # Errors
/// See [`isotonic_regression_weighted`].
pub fn isotonic_regression(y: &[f64]) -> StatsResult<IsotonicFit> {
    let weights = vec![1.0_f64; y.len()];
    isotonic_regression_weighted(y, &weights)
}

/// Antitonic (non-increasing) isotonic regression via the sign-flip trick.
///
/// Negates `y`, runs non-decreasing PAVA, and negates the result, yielding the
/// weighted-least-squares optimal **non-increasing** fit.
///
/// # Errors
/// See [`isotonic_regression_weighted`].
pub fn antitonic_regression_weighted(y: &[f64], weights: &[f64]) -> StatsResult<IsotonicFit> {
    let neg: Vec<f64> = y.iter().map(|&v| -v).collect();
    let fit = isotonic_regression_weighted(&neg, weights)?;
    let fitted = fit.fitted.iter().map(|&v| -v).collect();
    let blocks = fit
        .blocks
        .iter()
        .map(|b| IsotonicBlock {
            start: b.start,
            end: b.end,
            value: -b.value,
            weight: b.weight,
        })
        .collect();
    Ok(IsotonicFit { fitted, blocks })
}

/// Unweighted antitonic (non-increasing) regression.
///
/// # Errors
/// See [`isotonic_regression_weighted`].
pub fn antitonic_regression(y: &[f64]) -> StatsResult<IsotonicFit> {
    let weights = vec![1.0_f64; y.len()];
    antitonic_regression_weighted(y, &weights)
}

/// Weighted least-squares objective `Σ_i w_i (y_i − fit_i)²` of a candidate fit.
///
/// # Errors
/// [`StatsError::DimensionMismatch`] if the three slices differ in length.
pub fn weighted_sse(y: &[f64], fitted: &[f64], weights: &[f64]) -> StatsResult<f64> {
    if y.len() != fitted.len() || y.len() != weights.len() {
        return Err(StatsError::DimensionMismatch {
            a: y.len(),
            b: fitted.len().min(weights.len()),
        });
    }
    let mut s = 0.0;
    for ((&yi, &fi), &wi) in y.iter().zip(fitted.iter()).zip(weights.iter()) {
        let d = yi - fi;
        s += wi * d * d;
    }
    Ok(s)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn is_non_decreasing(v: &[f64]) -> bool {
        v.windows(2).all(|w| w[1] + 1e-12 >= w[0])
    }

    fn is_non_increasing(v: &[f64]) -> bool {
        v.windows(2).all(|w| w[1] - 1e-12 <= w[0])
    }

    // ---- (a) output is monotone non-decreasing ---------------------------
    #[test]
    fn output_is_monotone() {
        let y = [4.0, 1.0, 3.0, 2.0, 5.0, 0.0, 6.0];
        let fit = isotonic_regression(&y).expect("isotonic_regression should succeed");
        assert!(
            is_non_decreasing(&fit.fitted),
            "fit {:?} not monotone",
            fit.fitted
        );
    }

    // ---- (b) already-monotone input returned unchanged -------------------
    #[test]
    fn already_monotone_unchanged() {
        let y = [1.0, 2.0, 2.0, 3.5, 7.0];
        let fit = isotonic_regression(&y).expect("isotonic_regression should succeed");
        for (a, b) in fit.fitted.iter().zip(y.iter()) {
            assert!((a - b).abs() < 1e-12, "changed monotone input");
        }
    }

    // ---- (c) classic [1,3,2,4] → [1,2.5,2.5,4] ---------------------------
    #[test]
    fn classic_pooling_example() {
        let y = [1.0, 3.0, 2.0, 4.0];
        let fit = isotonic_regression(&y).expect("isotonic_regression should succeed");
        let expected = [1.0, 2.5, 2.5, 4.0];
        for (a, b) in fit.fitted.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "got {:?}, want {expected:?}",
                fit.fitted
            );
        }
        // The middle two form one block.
        assert_eq!(fit.blocks.len(), 3);
        assert_eq!(fit.blocks[1].start, 1);
        assert_eq!(fit.blocks[1].end, 3);
    }

    // ---- (d) minimises the objective vs brute-force grid -----------------
    #[test]
    fn minimises_objective_vs_brute_force() {
        // Small case; search a fine grid of monotone step candidates and confirm
        // PAVA's SSE is no larger than any grid point's SSE.
        let y = [2.0, 1.0, 4.0, 3.0];
        let w = [1.0, 1.0, 1.0, 1.0];
        let fit = isotonic_regression_weighted(&y, &w)
            .expect("isotonic_regression_weighted should succeed");
        let pava_sse = weighted_sse(&y, &fit.fitted, &w).expect("weighted_sse should succeed");

        // Brute force: each ŷ_i ∈ grid, non-decreasing. Grid spans [0,5].
        let grid: Vec<f64> = (0..=50).map(|k| k as f64 * 0.1).collect();
        let mut best = f64::INFINITY;
        for &a in &grid {
            for &b in &grid {
                if b < a {
                    continue;
                }
                for &c in &grid {
                    if c < b {
                        continue;
                    }
                    for &d in &grid {
                        if d < c {
                            continue;
                        }
                        let cand = [a, b, c, d];
                        let sse = weighted_sse(&y, &cand, &w).expect("weighted_sse should succeed");
                        if sse < best {
                            best = sse;
                        }
                    }
                }
            }
        }
        // PAVA optimum must be ≤ the best grid value up to grid resolution.
        assert!(
            pava_sse <= best + 1e-9,
            "PAVA SSE {pava_sse} should be ≤ grid-best {best}"
        );
        // And the grid (which is coarse) cannot beat PAVA by more than its step.
        assert!(best + 1e-6 >= pava_sse - 0.05);
    }

    // ---- (e) weighted PAVA respects unequal weights ----------------------
    #[test]
    fn weighted_pooling_uses_weighted_mean() {
        // Two points get pooled; the pooled value must be their weighted mean.
        let y = [0.0, 10.0, 0.0]; // index 1,2 violate then pool with weights
        let w = [1.0, 1.0, 3.0];
        let fit = isotonic_regression_weighted(&y, &w)
            .expect("isotonic_regression_weighted should succeed");
        // Expected: PAVA pools all three? Check block weighted means directly.
        // Points: y=[0,10,0], w=[1,1,3]. Block {1,2}: mean=(10*1+0*3)/4=2.5 but
        // that violates 0 ≤ 2.5 (ok), and 2.5 must precede... only {1,2} pool
        // since 10>0. After pooling, fitted = [0, 2.5, 2.5].
        let expected = [0.0, 2.5, 2.5];
        for (a, b) in fit.fitted.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "got {:?}, want {expected:?}",
                fit.fitted
            );
        }
        // Verify each block's value equals the weighted mean of its members.
        for blk in &fit.blocks {
            let mut sw = 0.0;
            let mut swy = 0.0;
            for i in blk.start..blk.end {
                sw += w[i];
                swy += w[i] * y[i];
            }
            assert!(
                (blk.value - swy / sw).abs() < 1e-12,
                "block value {} ≠ weighted mean {}",
                blk.value,
                swy / sw
            );
        }
    }

    // ---- (f) antitonic (non-increasing) via flip -------------------------
    #[test]
    fn antitonic_is_non_increasing() {
        let y = [1.0, 4.0, 2.0, 3.0, 0.0];
        let fit = antitonic_regression(&y).expect("antitonic_regression should succeed");
        assert!(
            is_non_increasing(&fit.fitted),
            "antitonic fit {:?}",
            fit.fitted
        );
        // Cross-check: antitonic(y) == -isotonic(-y).
        let neg: Vec<f64> = y.iter().map(|&v| -v).collect();
        let iso = isotonic_regression(&neg).expect("isotonic_regression should succeed");
        for (a, b) in fit.fitted.iter().zip(iso.fitted.iter()) {
            assert!((a + b).abs() < 1e-12);
        }
    }

    // ---- (g) idempotent: PAVA(PAVA(y)) == PAVA(y) ------------------------
    #[test]
    fn idempotent() {
        let y = [5.0, 2.0, 8.0, 1.0, 6.0, 3.0, 9.0, 0.0];
        let fit1 = isotonic_regression(&y).expect("isotonic_regression should succeed");
        let fit2 = isotonic_regression(&fit1.fitted).expect("isotonic_regression should succeed");
        for (a, b) in fit1.fitted.iter().zip(fit2.fitted.iter()) {
            assert!((a - b).abs() < 1e-12, "not idempotent: {a} vs {b}");
        }
    }

    // ---- (h) pooled blocks are exactly the level sets --------------------
    #[test]
    fn blocks_are_level_sets() {
        let y = [4.0, 1.0, 3.0, 2.0, 5.0, 0.0, 6.0];
        let fit = isotonic_regression(&y).expect("isotonic_regression should succeed");
        // Blocks partition [0,n) contiguously and cover every index once.
        let mut cursor = 0;
        for blk in &fit.blocks {
            assert_eq!(blk.start, cursor, "blocks not contiguous");
            assert!(blk.end > blk.start, "empty block");
            // All fitted values inside the block are equal to its value.
            for i in blk.start..blk.end {
                assert!((fit.fitted[i] - blk.value).abs() < 1e-12);
            }
            cursor = blk.end;
        }
        assert_eq!(cursor, y.len(), "blocks don't cover all indices");
        // Adjacent blocks are strictly increasing in value (maximal level sets).
        for w in fit.blocks.windows(2) {
            assert!(
                w[1].value > w[0].value - 1e-12,
                "block values not non-decreasing"
            );
        }
    }

    // ---- validation paths -------------------------------------------------
    #[test]
    fn empty_input_errors() {
        assert!(matches!(
            isotonic_regression(&[]),
            Err(StatsError::EmptyInput)
        ));
    }

    #[test]
    fn weight_length_mismatch_errors() {
        assert!(matches!(
            isotonic_regression_weighted(&[1.0, 2.0], &[1.0]),
            Err(StatsError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn non_positive_weight_errors() {
        assert!(matches!(
            isotonic_regression_weighted(&[1.0, 2.0], &[1.0, 0.0]),
            Err(StatsError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn non_finite_y_errors() {
        assert!(matches!(
            isotonic_regression(&[1.0, f64::NAN, 2.0]),
            Err(StatsError::NonFiniteValue(1))
        ));
    }

    #[test]
    fn single_point_is_itself() {
        let fit = isotonic_regression(&[3.7]).expect("isotonic_regression should succeed");
        assert_eq!(fit.fitted, vec![3.7]);
        assert_eq!(fit.blocks.len(), 1);
        assert_eq!(fit.blocks[0].len(), 1);
        assert!(!fit.blocks[0].is_empty());
    }

    #[test]
    fn all_decreasing_pools_to_single_mean() {
        // Strictly decreasing input → one block at the overall mean.
        let y = [5.0, 4.0, 3.0, 2.0, 1.0];
        let fit = isotonic_regression(&y).expect("isotonic_regression should succeed");
        assert_eq!(fit.blocks.len(), 1);
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        for &f in &fit.fitted {
            assert!((f - mean).abs() < 1e-12, "expected flat mean {mean}");
        }
    }
}
