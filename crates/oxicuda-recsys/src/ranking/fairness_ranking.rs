//! Fairness-aware exposure ranking (demographic parity of exposure).
//!
//! Reference: Ashudeep Singh, Thorsten Joachims, "Fairness of Exposure in
//! Rankings", KDD 2018.
//!
//! # Idea
//!
//! In a ranking, **exposure** is governed by position bias: the item at rank
//! `r` (0-indexed) is examined with weight `v(r) = 1 / log2(r + 2)` — the
//! familiar DCG discount. A pure relevance sort hands almost all exposure to
//! the very top, which can starve a protected group of items even when the
//! group carries a substantial share of the total relevance. The
//! *demographic-parity-of-exposure* criterion asks that each group `g` receive
//! exposure proportional to its share of relevance,
//! `exposure_g / E ≈ R_g / R_total`, where `E = Σ_r v(r)` is the total
//! exposure budget and `R_g` the group's summed relevance.
//!
//! # Method
//!
//! [`FairnessRanker`] performs a deterministic, position-by-position greedy
//! re-ranking. At each rank it selects the still-unplaced item that maximises a
//! convex blend of a relevance term and a fairness term:
//!
//! ```text
//! score(item) = (1 - λ) · rel(item)/rel_max
//!             +      λ  · ( merit_share(g) − allocated_exposure(g)/E )
//! ```
//!
//! * With `λ = 0` the rule reduces to sorting by relevance (descending).
//! * With `λ > 0` items from groups whose *allocated* exposure trails their
//!   *merit share* receive a bonus and are pulled to higher-exposure
//!   positions, trading a little relevance for a more balanced exposure
//!   allocation.
//!
//! Ties are broken by the lower item index, making the ranking deterministic.

use crate::error::{RecsysError, RecsysResult};

/// Position-bias exposure weight for a 0-indexed rank: `1 / log2(rank + 2)`.
#[inline]
#[must_use]
pub fn position_weight(rank: usize) -> f32 {
    1.0 / ((rank as f32) + 2.0).log2()
}

/// Configuration for [`FairnessRanker`].
#[derive(Debug, Clone)]
pub struct FairnessRankerConfig {
    /// Trade-off weight `λ ∈ [0, 1]`. `0` is a pure relevance sort; larger
    /// values prioritise exposure parity.
    pub fairness_weight: f32,
}

impl Default for FairnessRankerConfig {
    fn default() -> Self {
        Self {
            fairness_weight: 0.5,
        }
    }
}

/// Deterministic fairness-of-exposure re-ranker.
pub struct FairnessRanker {
    /// Configuration the ranker was built from.
    pub cfg: FairnessRankerConfig,
}

impl FairnessRanker {
    /// Construct a ranker.
    ///
    /// # Errors
    /// [`RecsysError::InvalidLossWeight`] when `fairness_weight` is not a
    /// finite value in `[0, 1]`.
    pub fn new(cfg: FairnessRankerConfig) -> RecsysResult<Self> {
        let w = cfg.fairness_weight;
        if !w.is_finite() || !(0.0..=1.0).contains(&w) {
            return Err(RecsysError::InvalidLossWeight { w });
        }
        Ok(Self { cfg })
    }

    /// Exposure weight at a 0-indexed `rank`.
    #[must_use]
    pub fn position_weight(&self, rank: usize) -> f32 {
        position_weight(rank)
    }

    /// The exposure weights `v(0), v(1), …, v(n-1)` (monotonically
    /// decreasing).
    #[must_use]
    pub fn exposure_weights(&self, n: usize) -> Vec<f32> {
        (0..n).map(position_weight).collect()
    }

    /// Produce a fairness-aware ranking as an ordered list of item indices.
    ///
    /// # Errors
    /// - [`RecsysError::DimensionMismatch`] when `relevances` and `groups`
    ///   differ in length.
    /// - [`RecsysError::EmptyInput`] when the inputs are empty.
    /// - [`RecsysError::InvalidConfig`] when `n_groups == 0`.
    /// - [`RecsysError::ItemOutOfBounds`] when any group id `>= n_groups`.
    pub fn rank(
        &self,
        relevances: &[f32],
        groups: &[usize],
        n_groups: usize,
    ) -> RecsysResult<Vec<usize>> {
        if relevances.len() != groups.len() {
            return Err(RecsysError::DimensionMismatch {
                expected: relevances.len(),
                got: groups.len(),
            });
        }
        if relevances.is_empty() {
            return Err(RecsysError::EmptyInput);
        }
        if n_groups == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "n_groups must be >= 1".into(),
            });
        }
        for &g in groups {
            if g >= n_groups {
                return Err(RecsysError::ItemOutOfBounds {
                    idx: g,
                    n: n_groups,
                });
            }
        }

        let n = relevances.len();
        let lambda = self.cfg.fairness_weight;
        let scale = relevances
            .iter()
            .fold(0.0_f32, |m, &r| m.max(r.abs()))
            .max(1e-9);

        let mut r_g = vec![0.0_f32; n_groups];
        for (&g, &rel) in groups.iter().zip(relevances.iter()) {
            r_g[g] += rel.max(0.0);
        }
        let r_total: f32 = r_g.iter().sum();
        let m_g: Vec<f32> = r_g
            .iter()
            .map(|&rg| {
                if r_total > 1e-12 {
                    rg / r_total
                } else {
                    1.0 / n_groups as f32
                }
            })
            .collect();
        let e_total: f32 = self.exposure_weights(n).iter().sum::<f32>().max(1e-12);

        let mut allocated = vec![0.0_f32; n_groups];
        let mut placed = vec![false; n];
        let mut ranking = Vec::with_capacity(n);

        for p in 0..n {
            let w_p = position_weight(p);
            let mut best: Option<usize> = None;
            let mut best_score = f32::NEG_INFINITY;
            for item in 0..n {
                if placed[item] {
                    continue;
                }
                let g = groups[item];
                let rel_term = relevances[item] / scale;
                let fair_term = m_g[g] - allocated[g] / e_total;
                let combined = (1.0 - lambda) * rel_term + lambda * fair_term;
                if combined > best_score {
                    best_score = combined;
                    best = Some(item);
                }
            }
            let chosen = best.ok_or(RecsysError::Internal {
                msg: "rank: no candidate available".into(),
            })?;
            placed[chosen] = true;
            allocated[groups[chosen]] += w_p;
            ranking.push(chosen);
        }
        Ok(ranking)
    }

    /// Realised exposure per group for a given `ranking`. The number of groups
    /// is inferred from `groups` as `max(groups) + 1`.
    ///
    /// # Errors
    /// - [`RecsysError::EmptyInput`] when either argument is empty.
    /// - [`RecsysError::ItemOutOfBounds`] when a ranked item index is out of
    ///   range for `groups`.
    pub fn group_exposure(&self, ranking: &[usize], groups: &[usize]) -> RecsysResult<Vec<f32>> {
        if ranking.is_empty() || groups.is_empty() {
            return Err(RecsysError::EmptyInput);
        }
        let n_groups = groups
            .iter()
            .copied()
            .max()
            .ok_or(RecsysError::EmptyInput)?
            + 1;
        let mut exposure = vec![0.0_f32; n_groups];
        for (p, &item) in ranking.iter().enumerate() {
            let g = *groups.get(item).ok_or(RecsysError::ItemOutOfBounds {
                idx: item,
                n: groups.len(),
            })?;
            if let Some(slot) = exposure.get_mut(g) {
                *slot += position_weight(p);
            }
        }
        Ok(exposure)
    }

    /// Exposure-disparity metric: the L1 deviation of each group's exposure
    /// share from its merit (relevance) share,
    /// `Σ_g | exposure_g/E − R_g/R_total |`. Lower is fairer; `0` is perfect
    /// demographic parity of exposure.
    ///
    /// # Errors
    /// - [`RecsysError::DimensionMismatch`] when the three slices differ in
    ///   length.
    /// - [`RecsysError::EmptyInput`] when they are empty.
    /// - [`RecsysError::ItemOutOfBounds`] when a ranked index is out of range.
    pub fn exposure_disparity(
        &self,
        ranking: &[usize],
        groups: &[usize],
        relevances: &[f32],
    ) -> RecsysResult<f32> {
        if ranking.len() != groups.len() || groups.len() != relevances.len() {
            return Err(RecsysError::DimensionMismatch {
                expected: groups.len(),
                got: ranking.len().max(relevances.len()),
            });
        }
        if ranking.is_empty() {
            return Err(RecsysError::EmptyInput);
        }
        let exposure = self.group_exposure(ranking, groups)?;
        let n_groups = exposure.len();

        let mut r_g = vec![0.0_f32; n_groups];
        for (&g, &rel) in groups.iter().zip(relevances.iter()) {
            if let Some(slot) = r_g.get_mut(g) {
                *slot += rel.max(0.0);
            }
        }
        let r_total: f32 = r_g.iter().sum();
        let e_total: f32 = exposure.iter().sum::<f32>().max(1e-12);

        let mut disparity = 0.0_f32;
        for (g, &exp_g) in exposure.iter().enumerate() {
            let merit = if r_total > 1e-12 {
                r_g.get(g).copied().unwrap_or(0.0) / r_total
            } else {
                1.0 / n_groups as f32
            };
            disparity += (exp_g / e_total - merit).abs();
        }
        Ok(disparity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranker(weight: f32) -> FairnessRanker {
        FairnessRanker::new(FairnessRankerConfig {
            fairness_weight: weight,
        })
        .expect("value should be present")
    }

    #[test]
    fn lambda_zero_is_relevance_sort() {
        let r = ranker(0.0);
        let relevances = vec![0.2_f32, 0.9, 0.5, 0.7, 0.1];
        let groups = vec![0_usize, 1, 0, 1, 0];
        let ranking = r
            .rank(&relevances, &groups, 2)
            .expect("rank should succeed");
        assert_eq!(
            ranking,
            vec![1, 3, 2, 0, 4],
            "λ=0 must sort by relevance desc"
        );
    }

    #[test]
    fn fairness_reduces_disparity() {
        // Group 0 holds the three top-relevance items, group 1 the three
        // slightly-lower items. A pure relevance sort clusters group 0 at the
        // top; turning fairness on interleaves and balances exposure.
        let relevances = vec![0.90_f32, 0.85, 0.80, 0.79, 0.78, 0.77];
        let groups = vec![0_usize, 0, 0, 1, 1, 1];

        let plain = ranker(0.0);
        let fair = ranker(0.5);
        let rank_plain = plain
            .rank(&relevances, &groups, 2)
            .expect("rank should succeed");
        let rank_fair = fair
            .rank(&relevances, &groups, 2)
            .expect("rank should succeed");

        let disp_plain = plain
            .exposure_disparity(&rank_plain, &groups, &relevances)
            .expect("value should be present");
        let disp_fair = fair
            .exposure_disparity(&rank_fair, &groups, &relevances)
            .expect("value should be present");
        assert!(
            disp_fair < disp_plain,
            "fairness should reduce disparity: plain {disp_plain}, fair {disp_fair}"
        );
    }

    #[test]
    fn ranking_is_a_permutation() {
        let r = ranker(0.7);
        let relevances = vec![0.5_f32, 0.1, 0.9, 0.3, 0.8, 0.2, 0.6];
        let groups = vec![0_usize, 1, 2, 1, 0, 2, 1];
        let mut ranking = r
            .rank(&relevances, &groups, 3)
            .expect("rank should succeed");
        assert_eq!(ranking.len(), relevances.len());
        ranking.sort_unstable();
        assert_eq!(ranking, (0..relevances.len()).collect::<Vec<_>>());
    }

    #[test]
    fn exposure_weights_decrease_with_rank() {
        let r = ranker(0.5);
        let w = r.exposure_weights(8);
        assert_eq!(w.len(), 8);
        assert!((w[0] - 1.0).abs() < 1e-6, "rank-0 weight must be 1");
        for p in 1..w.len() {
            assert!(w[p] < w[p - 1], "weight must strictly decrease at rank {p}");
        }
    }

    #[test]
    fn group_exposure_sums_to_total_budget() {
        let r = ranker(0.5);
        let relevances = vec![0.9_f32, 0.5, 0.7, 0.2];
        let groups = vec![0_usize, 1, 0, 1];
        let ranking = r
            .rank(&relevances, &groups, 2)
            .expect("rank should succeed");
        let exposure = r
            .group_exposure(&ranking, &groups)
            .expect("group_exposure should succeed");
        let total: f32 = exposure.iter().sum();
        let budget: f32 = r.exposure_weights(4).iter().sum();
        assert!(
            (total - budget).abs() < 1e-5,
            "group exposures must sum to the total budget"
        );
    }

    #[test]
    fn err_length_mismatch_and_empty() {
        let r = ranker(0.5);
        assert!(matches!(
            r.rank(&[0.5, 0.2], &[0], 1),
            Err(RecsysError::DimensionMismatch { .. })
        ));
        assert!(matches!(r.rank(&[], &[], 1), Err(RecsysError::EmptyInput)));
        assert!(matches!(
            r.rank(&[0.5], &[0], 0),
            Err(RecsysError::InvalidConfig { .. })
        ));
        assert!(matches!(
            r.rank(&[0.5], &[3], 2),
            Err(RecsysError::ItemOutOfBounds { .. })
        ));
    }

    #[test]
    fn err_invalid_weight() {
        assert!(matches!(
            FairnessRanker::new(FairnessRankerConfig {
                fairness_weight: 1.5
            }),
            Err(RecsysError::InvalidLossWeight { .. })
        ));
        assert!(matches!(
            FairnessRanker::new(FairnessRankerConfig {
                fairness_weight: -0.1
            }),
            Err(RecsysError::InvalidLossWeight { .. })
        ));
        assert!(matches!(
            FairnessRanker::new(FairnessRankerConfig {
                fairness_weight: f32::NAN
            }),
            Err(RecsysError::InvalidLossWeight { .. })
        ));
    }

    #[test]
    fn disparity_non_negative_and_fair_not_worse() {
        // Two symmetric (equal-merit) groups. Discrete log-discounted exposure
        // over four slots cannot be split perfectly evenly, so zero disparity
        // is unreachable — but the fair ranking must never be *worse* than the
        // plain relevance sort, and disparity is always non-negative.
        let relevances = vec![0.8_f32, 0.8, 0.6, 0.6];
        let groups = vec![0_usize, 1, 0, 1];

        let plain = ranker(0.0);
        let fair = ranker(0.5);
        let rank_plain = plain
            .rank(&relevances, &groups, 2)
            .expect("rank should succeed");
        let rank_fair = fair
            .rank(&relevances, &groups, 2)
            .expect("rank should succeed");
        let disp_plain = plain
            .exposure_disparity(&rank_plain, &groups, &relevances)
            .expect("value should be present");
        let disp_fair = fair
            .exposure_disparity(&rank_fair, &groups, &relevances)
            .expect("value should be present");
        assert!(
            disp_plain >= 0.0 && disp_fair >= 0.0,
            "disparity must be >= 0"
        );
        assert!(
            disp_fair <= disp_plain + 1e-6,
            "fair ranking must not increase disparity: plain {disp_plain}, fair {disp_fair}"
        );
    }
}
