//! Multi-objective alignment: Pareto-front computation and scalarisation.
//!
//! References:
//! * Bai et al. 2022, "Training a Helpful and Harmless Assistant with RLHF",
//!   arXiv:2204.05862 — the canonical helpfulness/harmlessness trade-off.
//! * Rame et al. 2023, "Rewarded Soups", arXiv:2306.04488 — weight-space
//!   interpolation across multiple reward objectives.
//!
//! Modern alignment optimises several, often-conflicting, reward objectives at
//! once — most commonly **helpfulness** and **harmlessness**: a maximally
//! helpful policy may answer harmful requests, and a maximally harmless one may
//! refuse benign ones. There is no single best policy; instead there is a
//! *Pareto front* of policies, each of which cannot improve one objective
//! without degrading another. This module provides the two operations such a
//! pipeline needs:
//!
//! 1. **Pareto-front extraction** — given a set of candidate policies, each
//!    scored on `m` objectives (higher = better), return the indices of the
//!    non-dominated candidates. Candidate `a` *dominates* `b` iff `a` is ≥ `b`
//!    on every objective and strictly greater on at least one.
//! 2. **Scalarisation** — collapse the objective vector to a single scalar so a
//!    point on the front can be selected, via a non-negative **weighted sum**
//!    `Σ_j w_j r_j` (a linear scalarisation) or a **Chebyshev** (weighted
//!    max-regret) scalarisation `−max_j w_j (z_j − r_j)` against an ideal point
//!    `z`, which — unlike the weighted sum — can reach concave parts of the
//!    front.
//!
//! All routines are deterministic and validate shapes / weights up front.

use crate::error::{RlhfError, RlhfResult};

// ── Dominance ───────────────────────────────────────────────────────────────

/// Whether objective vector `a` Pareto-dominates `b` (maximisation): `a_j ≥ b_j`
/// for all `j` and `a_j > b_j` for at least one `j`.
///
/// `a` and `b` must have equal, non-zero length (callers validate).
fn dominates(a: &[f32], b: &[f32]) -> bool {
    let mut strictly_better = false;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x < y {
            return false;
        }
        if x > y {
            strictly_better = true;
        }
    }
    strictly_better
}

/// Validate a candidate matrix: non-empty, every row of equal non-zero width,
/// no NaN.
fn validate_candidates(candidates: &[Vec<f32>]) -> RlhfResult<usize> {
    if candidates.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    let m = candidates[0].len();
    if m == 0 {
        return Err(RlhfError::EmptyInput);
    }
    for row in candidates {
        if row.len() != m {
            return Err(RlhfError::DimensionMismatch {
                expected: m,
                got: row.len(),
            });
        }
        if row.iter().any(|x| x.is_nan()) {
            return Err(RlhfError::NanEncountered);
        }
    }
    Ok(m)
}

/// Indices of the non-dominated (Pareto-optimal) candidates, in ascending order.
///
/// Each candidate is a length-`m` objective vector (higher = better). The
/// returned indices identify candidates not dominated by any other.
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`] for no candidates / zero objectives,
/// [`RlhfError::DimensionMismatch`] for ragged rows, and
/// [`RlhfError::NanEncountered`] for NaN scores.
pub fn pareto_front(candidates: &[Vec<f32>]) -> RlhfResult<Vec<usize>> {
    validate_candidates(candidates)?;
    let mut front = Vec::new();
    for (i, ci) in candidates.iter().enumerate() {
        let dominated = candidates
            .iter()
            .enumerate()
            .any(|(j, cj)| j != i && dominates(cj, ci));
        if !dominated {
            front.push(i);
        }
    }
    Ok(front)
}

// ── Scalarisation ───────────────────────────────────────────────────────────

/// Non-negative weighted-sum (linear) scalarisation `Σ_j w_j · r_j`.
///
/// `weights` must be non-negative, finite, not all zero, and the same length as
/// `objectives`.
///
/// # Errors
///
/// Returns [`RlhfError::DimensionMismatch`] for a length mismatch,
/// [`RlhfError::InvalidLambda`] for a negative / non-finite / all-zero weight
/// vector, and [`RlhfError::NanEncountered`] for NaN objectives.
pub fn weighted_sum(objectives: &[f32], weights: &[f32]) -> RlhfResult<f32> {
    if objectives.len() != weights.len() {
        return Err(RlhfError::DimensionMismatch {
            expected: objectives.len(),
            got: weights.len(),
        });
    }
    if objectives.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    let mut wsum = 0.0_f32;
    for &w in weights {
        if !w.is_finite() || w < 0.0 {
            return Err(RlhfError::InvalidLambda { lambda: w });
        }
        wsum += w;
    }
    if wsum <= 0.0 {
        return Err(RlhfError::InvalidLambda { lambda: 0.0 });
    }
    let mut acc = 0.0_f32;
    for (&r, &w) in objectives.iter().zip(weights.iter()) {
        if r.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        acc += w * r;
    }
    Ok(acc)
}

/// Chebyshev (weighted max-regret) scalarisation against an ideal point.
///
/// Returns `−max_j w_j · (ideal_j − r_j)`: the negated, weighted worst-case
/// shortfall from the ideal. Larger (closer to 0) is better, and a candidate
/// equal to the ideal scores `0`. Unlike [`weighted_sum`], this can select
/// points on concave regions of the Pareto front.
///
/// # Errors
///
/// Same shape / weight validation as [`weighted_sum`], plus
/// [`RlhfError::NanEncountered`] for NaN inputs.
pub fn chebyshev_scalarisation(
    objectives: &[f32],
    ideal: &[f32],
    weights: &[f32],
) -> RlhfResult<f32> {
    let m = objectives.len();
    if ideal.len() != m {
        return Err(RlhfError::DimensionMismatch {
            expected: m,
            got: ideal.len(),
        });
    }
    if weights.len() != m {
        return Err(RlhfError::DimensionMismatch {
            expected: m,
            got: weights.len(),
        });
    }
    if m == 0 {
        return Err(RlhfError::EmptyInput);
    }
    let mut wsum = 0.0_f32;
    for &w in weights {
        if !w.is_finite() || w < 0.0 {
            return Err(RlhfError::InvalidLambda { lambda: w });
        }
        wsum += w;
    }
    if wsum <= 0.0 {
        return Err(RlhfError::InvalidLambda { lambda: 0.0 });
    }
    let mut worst = f32::NEG_INFINITY;
    for ((&r, &z), &w) in objectives.iter().zip(ideal.iter()).zip(weights.iter()) {
        if r.is_nan() || z.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        let regret = w * (z - r);
        if regret > worst {
            worst = regret;
        }
    }
    Ok(-worst)
}

/// Select the index of the Pareto-front candidate that maximises the weighted
/// sum scalarisation. A convenience that composes [`pareto_front`] and
/// [`weighted_sum`].
///
/// Ties resolve to the lowest front index.
///
/// # Errors
///
/// Propagates errors from [`pareto_front`] and [`weighted_sum`].
pub fn select_by_weighted_sum(candidates: &[Vec<f32>], weights: &[f32]) -> RlhfResult<usize> {
    let front = pareto_front(candidates)?;
    let mut best_idx = *front.first().ok_or(RlhfError::EmptyInput)?;
    let mut best_score = f32::NEG_INFINITY;
    for &i in &front {
        let score = weighted_sum(&candidates[i], weights)?;
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    Ok(best_idx)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. dominates: clear domination.
    #[test]
    fn dominates_basic() {
        assert!(dominates(&[2.0, 2.0], &[1.0, 1.0]));
        assert!(dominates(&[2.0, 1.0], &[1.0, 1.0])); // equal on one, better on one
        assert!(!dominates(&[2.0, 0.0], &[1.0, 1.0])); // trade-off → no domination
        assert!(!dominates(&[1.0, 1.0], &[1.0, 1.0])); // equal → not strict
    }

    // 2. Pareto front of a 2-objective trade-off keeps the non-dominated set.
    #[test]
    fn pareto_front_tradeoff() {
        // Candidates (helpfulness, harmlessness):
        let c = vec![
            vec![1.0, 0.0], // A: max helpful, min harmless
            vec![0.0, 1.0], // B: min helpful, max harmless
            vec![0.5, 0.5], // C: balanced
            vec![0.2, 0.2], // D: dominated by C
        ];
        let front = pareto_front(&c).expect("front");
        // D (index 3) is dominated by C; A, B, C are mutually non-dominated.
        assert_eq!(front, vec![0, 1, 2]);
    }

    // 3. A single dominating candidate is the sole front member.
    #[test]
    fn pareto_front_single_dominator() {
        let c = vec![vec![3.0, 3.0], vec![1.0, 1.0], vec![2.0, 0.0]];
        let front = pareto_front(&c).expect("front");
        assert_eq!(front, vec![0]);
    }

    // 4. All-equal candidates: none dominates another → all on the front.
    #[test]
    fn pareto_front_all_equal() {
        let c = vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]];
        let front = pareto_front(&c).expect("front");
        assert_eq!(front, vec![0, 1, 2]);
    }

    // 5. weighted_sum computes the linear combination.
    #[test]
    fn weighted_sum_basic() {
        let s = weighted_sum(&[2.0, 4.0], &[0.25, 0.75]).expect("ws");
        assert!((s - (0.25 * 2.0 + 0.75 * 4.0)).abs() < 1e-6);
    }

    // 6. Equal weights average behaviour selects the most balanced front point.
    #[test]
    fn select_balanced_with_equal_weights() {
        let c = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.6, 0.6]];
        let idx = select_by_weighted_sum(&c, &[0.5, 0.5]).expect("select");
        // Balanced point scores 0.6 vs 0.5 for the extremes.
        assert_eq!(idx, 2);
    }

    // 7. Skewed weights select the favoured extreme.
    #[test]
    fn select_extreme_with_skewed_weights() {
        let c = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
        // Heavily weight objective 0 (helpfulness).
        let idx = select_by_weighted_sum(&c, &[0.9, 0.1]).expect("select");
        assert_eq!(idx, 0);
    }

    // 8. Chebyshev scores 0 at the ideal and negative below it.
    #[test]
    fn chebyshev_zero_at_ideal() {
        let ideal = [1.0, 1.0];
        let at = chebyshev_scalarisation(&[1.0, 1.0], &ideal, &[1.0, 1.0]).expect("cheb");
        assert!(at.abs() < 1e-6, "at-ideal should score 0, got {at}");
        let below = chebyshev_scalarisation(&[0.5, 0.8], &ideal, &[1.0, 1.0]).expect("cheb");
        assert!(below < 0.0, "below ideal should be negative, got {below}");
        // worst shortfall is on objective 0: 1*(1-0.5)=0.5 → score -0.5.
        assert!((below - (-0.5)).abs() < 1e-6);
    }

    // 9. Chebyshev can pick a concave-front point a weighted sum misses.
    #[test]
    fn chebyshev_prefers_balanced_concave() {
        // Concave front: extremes (1,0)/(0,1) and a balanced (0.6,0.6).
        // Equal-weight Chebyshev against ideal (1,1):
        let ideal = [1.0, 1.0];
        let extreme = chebyshev_scalarisation(&[1.0, 0.0], &ideal, &[0.5, 0.5]).expect("a");
        let balanced = chebyshev_scalarisation(&[0.6, 0.6], &ideal, &[0.5, 0.5]).expect("b");
        // extreme regret = 0.5*(1-0)=0.5; balanced regret = 0.5*(1-0.6)=0.2.
        assert!(
            balanced > extreme,
            "Chebyshev should favour the balanced point"
        );
    }

    // 10. Ragged candidate rows rejected.
    #[test]
    fn ragged_rows_error() {
        let c = vec![vec![1.0, 2.0], vec![1.0]];
        assert!(matches!(
            pareto_front(&c),
            Err(RlhfError::DimensionMismatch { .. })
        ));
    }

    // 11. NaN candidate rejected.
    #[test]
    fn nan_candidate_error() {
        let c = vec![vec![1.0, f32::NAN]];
        assert!(matches!(pareto_front(&c), Err(RlhfError::NanEncountered)));
    }

    // 12. Empty candidates / zero objectives rejected.
    #[test]
    fn empty_inputs_error() {
        let empty: Vec<Vec<f32>> = vec![];
        assert!(matches!(pareto_front(&empty), Err(RlhfError::EmptyInput)));
        let zero_obj = vec![vec![]];
        assert!(matches!(
            pareto_front(&zero_obj),
            Err(RlhfError::EmptyInput)
        ));
    }

    // 13. Negative / all-zero weights rejected by scalarisations.
    #[test]
    fn invalid_weights_error() {
        assert!(matches!(
            weighted_sum(&[1.0, 2.0], &[-0.1, 0.5]),
            Err(RlhfError::InvalidLambda { .. })
        ));
        assert!(matches!(
            weighted_sum(&[1.0, 2.0], &[0.0, 0.0]),
            Err(RlhfError::InvalidLambda { .. })
        ));
        assert!(matches!(
            chebyshev_scalarisation(&[1.0], &[1.0], &[0.0]),
            Err(RlhfError::InvalidLambda { .. })
        ));
    }

    // 14. Weight/objective length mismatch rejected.
    #[test]
    fn weight_length_mismatch_error() {
        assert!(matches!(
            weighted_sum(&[1.0, 2.0], &[0.5]),
            Err(RlhfError::DimensionMismatch { .. })
        ));
    }
}
