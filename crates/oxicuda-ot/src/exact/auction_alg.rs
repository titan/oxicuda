//! Auction algorithm for the linear assignment problem (Bertsekas 1988).
//!
//! The assignment problem seeks a permutation `σ` of `{0, …, n−1}` maximising
//! the total benefit `Σ_i A[i, σ(i)]` (equivalently minimising a cost), where
//! `A ∈ ℝ^{n × n}` is a dense benefit matrix. Bertsekas' auction algorithm
//! solves it by an economic analogy: "persons" (rows) competitively bid for
//! "objects" (columns) whose `prices` rise as demand increases.
//!
//! At each round every unassigned person `i` finds the two objects with the
//! highest *net* value `A[i, j] − price[j]`. It claims its best object and
//! raises that object's price by the bidding increment
//!
//! ```text
//! bid = (best_value − second_value) + ε ,
//! ```
//!
//! displacing whoever previously held the object. Terminating with a positive
//! `ε` yields an assignment whose total benefit is within `n·ε` of the optimum;
//! **ε-scaling** drives `ε` geometrically toward `1/(n+1)`, at which point an
//! integer-benefit problem's solution is provably exact (complementary
//! slackness holds to within `ε < 1/n`).
//!
//! This is an alternative to the network-simplex solver of [`crate::exact`] and
//! is competitive for dense small-to-medium `n × n` problems.

use crate::error::{OtError, OtResult};

/// Configuration for the auction assignment solver.
#[derive(Debug, Clone)]
pub struct AuctionConfig {
    /// Initial bidding increment `ε₀` (must be `> 0`).
    ///
    /// A common heuristic sets `ε₀ ≈ max|A| / 5`; the default works for
    /// benefit magnitudes near unity.
    pub eps_init: f32,
    /// Geometric decay factor applied to `ε` after each scaling phase
    /// (must lie in `(0, 1)`).
    pub eps_decay: f32,
    /// Terminal increment `ε_min`. Phases stop once `ε ≤ ε_min`.
    pub eps_min: f32,
    /// Maximum number of bidding rounds per scaling phase.
    pub max_iter: usize,
}

impl Default for AuctionConfig {
    fn default() -> Self {
        Self {
            eps_init: 1.0,
            eps_decay: 0.25,
            eps_min: 1e-3,
            max_iter: 100_000,
        }
    }
}

/// Result of the auction assignment solver.
#[derive(Debug, Clone)]
pub struct AuctionResult {
    /// `assignment[i] = j` means person `i` is matched to object `j`.
    pub assignment: Vec<usize>,
    /// Final object prices, length `n`.
    pub prices: Vec<f32>,
    /// Total benefit `Σ_i A[i, assignment[i]]`.
    pub benefit: f32,
    /// Total number of bidding rounds performed across all scaling phases.
    pub iters: usize,
}

/// Validate a square benefit matrix.
fn validate(benefit: &[f32], n: usize, cfg: &AuctionConfig) -> OtResult<()> {
    if n == 0 {
        return Err(OtError::EmptyInput);
    }
    if benefit.len() != n * n {
        return Err(OtError::MarginalMismatch {
            m: n,
            n,
            a_len: benefit.len(),
            b_len: n * n,
        });
    }
    for &v in benefit {
        if !v.is_finite() {
            return Err(OtError::Internal {
                msg: "non-finite benefit entry".to_string(),
            });
        }
    }
    if cfg.eps_init <= 0.0 {
        return Err(OtError::BadEpsilon { eps: cfg.eps_init });
    }
    if cfg.eps_min <= 0.0 {
        return Err(OtError::BadEpsilon { eps: cfg.eps_min });
    }
    if !(0.0..1.0).contains(&cfg.eps_decay) || cfg.eps_decay <= 0.0 {
        return Err(OtError::Internal {
            msg: "eps_decay must lie in (0, 1)".to_string(),
        });
    }
    Ok(())
}

/// Run one forward-auction phase at a fixed `ε`, mutating the assignment and
/// prices in place. Returns the number of rounds consumed, or an error if the
/// round budget is exhausted before all persons are assigned.
fn auction_phase(
    benefit: &[f32],
    n: usize,
    eps: f32,
    prices: &mut [f32],
    assignment: &mut [Option<usize>],
    owner: &mut [Option<usize>],
    max_iter: usize,
) -> OtResult<usize> {
    let mut rounds = 0_usize;
    loop {
        // Collect every currently-unassigned person.
        let mut any_unassigned = false;
        for i in 0..n {
            if assignment[i].is_some() {
                continue;
            }
            any_unassigned = true;

            // Find the best and second-best net value for person `i`.
            let mut best_j = 0_usize;
            let mut best_val = f32::NEG_INFINITY;
            let mut second_val = f32::NEG_INFINITY;
            let row = i * n;
            for j in 0..n {
                let net = benefit[row + j] - prices[j];
                if net > best_val {
                    second_val = best_val;
                    best_val = net;
                    best_j = j;
                } else if net > second_val {
                    second_val = net;
                }
            }

            // Bidding increment: how much the best object can be over-priced
            // while remaining at least as good as the runner-up, plus `ε`.
            let gap = if second_val.is_finite() {
                best_val - second_val
            } else {
                // Single object case: bid an `ε` step above current price.
                0.0
            };
            let bid = gap + eps;
            prices[best_j] += bid;

            // Re-assign object `best_j` to person `i`, evicting the old owner.
            if let Some(prev) = owner[best_j].take() {
                assignment[prev] = None;
            }
            owner[best_j] = Some(i);
            assignment[i] = Some(best_j);

            rounds += 1;
            if rounds > max_iter {
                return Err(OtError::NotConverged {
                    iter: max_iter,
                    tol: eps,
                });
            }
        }
        if !any_unassigned {
            break;
        }
    }
    Ok(rounds)
}

/// Solve the dense linear assignment problem by maximising total benefit.
///
/// `benefit` is the `n × n` benefit matrix in row-major order; entry
/// `benefit[i*n + j]` is the value of matching person `i` to object `j`.
/// Returns the optimal (to within ε-scaling tolerance) assignment.
pub fn auction_assignment(
    benefit: &[f32],
    n: usize,
    cfg: &AuctionConfig,
) -> OtResult<AuctionResult> {
    validate(benefit, n, cfg)?;

    let mut prices = vec![0.0_f32; n];
    let mut assignment: Vec<Option<usize>> = vec![None; n];
    let mut owner: Vec<Option<usize>> = vec![None; n];
    let mut total_rounds = 0_usize;

    // ε-scaling: repeatedly solve with a shrinking increment, reusing the
    // prices (but clearing the assignment so persons can re-bid more cheaply).
    let mut eps = cfg.eps_init.max(cfg.eps_min);
    loop {
        for slot in assignment.iter_mut() {
            *slot = None;
        }
        for slot in owner.iter_mut() {
            *slot = None;
        }
        total_rounds += auction_phase(
            benefit,
            n,
            eps,
            &mut prices,
            &mut assignment,
            &mut owner,
            cfg.max_iter,
        )?;
        if eps <= cfg.eps_min {
            break;
        }
        eps = (eps * cfg.eps_decay).max(cfg.eps_min);
    }

    // Extract the final assignment (every slot is filled after termination).
    let mut final_assignment = vec![0_usize; n];
    let mut benefit_total = 0.0_f32;
    for i in 0..n {
        let j = assignment[i].ok_or_else(|| OtError::Internal {
            msg: "auction terminated with an unassigned person".to_string(),
        })?;
        final_assignment[i] = j;
        benefit_total += benefit[i * n + j];
    }

    Ok(AuctionResult {
        assignment: final_assignment,
        prices,
        benefit: benefit_total,
        iters: total_rounds,
    })
}

/// Convenience wrapper that *minimises* total cost instead of maximising
/// benefit, by negating the cost matrix internally.
///
/// `cost[i*n + j]` is the cost of assigning person `i` to object `j`. Returns
/// the assignment together with its total cost `Σ_i cost[i, σ(i)]`.
pub fn auction_min_cost(
    cost: &[f32],
    n: usize,
    cfg: &AuctionConfig,
) -> OtResult<(Vec<usize>, f32)> {
    if n == 0 {
        return Err(OtError::EmptyInput);
    }
    if cost.len() != n * n {
        return Err(OtError::MarginalMismatch {
            m: n,
            n,
            a_len: cost.len(),
            b_len: n * n,
        });
    }
    let benefit: Vec<f32> = cost.iter().map(|&c| -c).collect();
    let res = auction_assignment(&benefit, n, cfg)?;
    let mut total = 0.0_f32;
    for (i, &j) in res.assignment.iter().enumerate() {
        total += cost[i * n + j];
    }
    Ok((res.assignment, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force optimal assignment benefit for tiny `n` (n ≤ 8), for testing.
    fn brute_force_max(benefit: &[f32], n: usize) -> f32 {
        let mut perm: Vec<usize> = (0..n).collect();
        let mut best = f32::NEG_INFINITY;
        permute(&mut perm, 0, &mut |p| {
            let mut s = 0.0_f32;
            for (i, &j) in p.iter().enumerate() {
                s += benefit[i * n + j];
            }
            if s > best {
                best = s;
            }
        });
        best
    }

    fn permute<F: FnMut(&[usize])>(arr: &mut [usize], k: usize, f: &mut F) {
        if k == arr.len() {
            f(arr);
            return;
        }
        for i in k..arr.len() {
            arr.swap(k, i);
            permute(arr, k + 1, f);
            arr.swap(k, i);
        }
    }

    fn is_permutation(a: &[usize], n: usize) -> bool {
        let mut seen = vec![false; n];
        for &j in a {
            if j >= n || seen[j] {
                return false;
            }
            seen[j] = true;
        }
        true
    }

    #[test]
    fn identity_benefit_recovers_diagonal() {
        // Diagonal benefits dominate → identity assignment.
        let n = 4;
        let mut benefit = vec![0.0_f32; n * n];
        for i in 0..n {
            benefit[i * n + i] = 10.0;
        }
        let res = auction_assignment(&benefit, n, &AuctionConfig::default()).expect("ok");
        for i in 0..n {
            assert_eq!(res.assignment[i], i);
        }
        assert!((res.benefit - 40.0).abs() < 1e-3);
    }

    #[test]
    fn output_is_a_valid_permutation() {
        let n = 5;
        let benefit = vec![
            3.0_f32, 1.0, 2.0, 5.0, 0.0, 2.0, 4.0, 1.0, 0.0, 3.0, 1.0, 0.0, 6.0, 2.0, 1.0, 4.0,
            2.0, 1.0, 3.0, 5.0, 0.0, 3.0, 2.0, 4.0, 6.0,
        ];
        let res = auction_assignment(&benefit, n, &AuctionConfig::default()).expect("ok");
        assert!(is_permutation(&res.assignment, n));
    }

    #[test]
    fn matches_brute_force_optimum_small() {
        let n = 5;
        let benefit = vec![
            7.0_f32, 2.0, 1.0, 9.0, 4.0, 3.0, 8.0, 5.0, 2.0, 6.0, 1.0, 4.0, 9.0, 3.0, 2.0, 6.0,
            5.0, 2.0, 7.0, 8.0, 4.0, 3.0, 6.0, 1.0, 9.0,
        ];
        let res = auction_assignment(&benefit, n, &AuctionConfig::default()).expect("ok");
        let opt = brute_force_max(&benefit, n);
        // ε-scaling guarantees benefit ≥ opt − n·ε_min.
        assert!(
            res.benefit >= opt - (n as f32) * 1e-3 - 1e-3,
            "auction {} vs brute {}",
            res.benefit,
            opt
        );
    }

    #[test]
    fn min_cost_finds_cheapest_assignment() {
        // Cheapest is the anti-diagonal here.
        let n = 3;
        let cost = vec![9.0_f32, 9.0, 1.0, 9.0, 1.0, 9.0, 1.0, 9.0, 9.0];
        let (assign, total) = auction_min_cost(&cost, n, &AuctionConfig::default()).expect("ok");
        assert_eq!(assign, vec![2, 1, 0]);
        assert!((total - 3.0).abs() < 1e-3);
    }

    #[test]
    fn single_element_trivial() {
        let res = auction_assignment(&[5.0_f32], 1, &AuctionConfig::default()).expect("ok");
        assert_eq!(res.assignment, vec![0]);
        assert!((res.benefit - 5.0).abs() < 1e-6);
    }

    #[test]
    fn empty_rejected() {
        let res = auction_assignment(&[], 0, &AuctionConfig::default());
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn wrong_shape_rejected() {
        let res = auction_assignment(&[1.0_f32, 2.0, 3.0], 2, &AuctionConfig::default());
        assert!(matches!(res, Err(OtError::MarginalMismatch { .. })));
    }

    #[test]
    fn non_finite_rejected() {
        let res = auction_assignment(&[1.0_f32, f32::NAN, 0.0, 1.0], 2, &AuctionConfig::default());
        assert!(matches!(res, Err(OtError::Internal { .. })));
    }

    #[test]
    fn bad_epsilon_rejected() {
        let cfg = AuctionConfig {
            eps_init: 0.0,
            ..AuctionConfig::default()
        };
        let res = auction_assignment(&[1.0_f32, 0.0, 0.0, 1.0], 2, &cfg);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn bad_decay_rejected() {
        let cfg = AuctionConfig {
            eps_decay: 1.5,
            ..AuctionConfig::default()
        };
        let res = auction_assignment(&[1.0_f32, 0.0, 0.0, 1.0], 2, &cfg);
        assert!(matches!(res, Err(OtError::Internal { .. })));
    }

    #[test]
    fn prices_are_nonnegative_after_scaling() {
        let n = 4;
        let benefit = vec![
            2.0_f32, 5.0, 1.0, 0.0, 3.0, 1.0, 4.0, 2.0, 1.0, 2.0, 3.0, 5.0, 4.0, 0.0, 2.0, 1.0,
        ];
        let res = auction_assignment(&benefit, n, &AuctionConfig::default()).expect("ok");
        // Auction prices only ever increase from zero.
        for &p in &res.prices {
            assert!(p >= -1e-6, "negative price {p}");
        }
    }

    #[test]
    fn larger_random_problem_is_permutation_and_near_optimal() {
        // Deterministic pseudo-random benefit via a small LCG.
        let n = 8;
        let mut state = 12345_u64;
        let mut benefit = vec![0.0_f32; n * n];
        for v in benefit.iter_mut() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *v = ((state >> 40) & 0xff) as f32 / 16.0;
        }
        let res = auction_assignment(&benefit, n, &AuctionConfig::default()).expect("ok");
        assert!(is_permutation(&res.assignment, n));
        let opt = brute_force_max(&benefit, n);
        assert!(
            res.benefit >= opt - (n as f32) * 1e-3 - 1e-2,
            "auction {} vs brute {}",
            res.benefit,
            opt
        );
    }
}
