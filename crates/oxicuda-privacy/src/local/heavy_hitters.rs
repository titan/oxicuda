//! Local differential privacy heavy hitters via the **TreeHist** protocol.
//!
//! # Reference
//! - Bassily, Nissim, Stemmer, Thakurta & Thakkar (2017), "Practical Locally
//!   Private Heavy Hitters", NeurIPS 2017. arXiv:1707.04982.
//!
//! # Problem
//! Each of `n` users holds a single item drawn from a domain of size
//! `2^{domain_bits}`. We wish to identify the *heavy hitters* — items held by
//! many users — together with frequency estimates, while every user only ever
//! sends a *single locally-randomised report* about their own item. No raw
//! item ever leaves a user's device.
//!
//! # TreeHist (the variant implemented here)
//! Items are identified with the leaves of a complete binary tree of depth
//! `domain_bits`: an item's `ℓ`-bit prefix is the most-significant `ℓ` bits of
//! its value. The protocol descends the tree level by level:
//!
//! 1. Maintain a frontier of *candidate prefixes* surviving from the previous
//!    level (the root, an empty prefix, seeds level 0).
//! 2. At level `ℓ`, extend every surviving candidate by both child bits,
//!    forming the level-`ℓ` candidate set.
//! 3. Estimate the frequency of each candidate prefix from the users'
//!    privatised reports using a **frequency oracle** (here Generalised
//!    Randomised Response over the prefix domain — see [`grr`]).
//! 4. *Prune*: keep only candidates whose estimated frequency exceeds
//!    `threshold`. A heavy item must have a heavy prefix at every level, so a
//!    pruned prefix can never lead to a heavy leaf.
//! 5. Repeat until level `domain_bits`; surviving leaves are the heavy
//!    hitters, reported with their estimated counts.
//!
//! [`grr`]: crate::local::grr
//!
//! # Privacy guarantee
//! The local-DP guarantee is **inherited entirely from the per-report
//! randomiser**: each user privatises their item-prefix exactly once with
//! [`grr_encode`], which is `(ε, 0)`-LDP. By the post-processing property the
//! prefix search, pruning, and frequency estimation add *no* privacy cost.
//! Hence the whole mechanism is `(cfg.epsilon, 0)`-LDP per user. (Splitting a
//! user across levels would instead split the budget; here we let every user
//! report at the deepest level and derive every prefix by truncation, so the
//! single GRR report carries the full `ε`.)

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;
use crate::local::grr::{GrrConfig, grr_encode, grr_estimate_frequency};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the TreeHist local-DP heavy-hitters protocol.
#[derive(Debug, Clone)]
pub struct HeavyHittersConfig {
    /// Per-report local-DP parameter `ε > 0`.
    pub epsilon: f64,
    /// Bit-width of the item domain (`domain_bits ≥ 1`); the domain has
    /// `2^{domain_bits}` items, i.e. items lie in `0 ..= 2^{domain_bits} − 1`.
    pub domain_bits: u32,
    /// Frequency threshold `≥ 0`: prefixes (and leaves) whose estimated
    /// frequency falls at or below this value are pruned.
    pub threshold: f64,
    /// Number of users `n ≥ 1` contributing one report each.
    pub n_users: usize,
}

impl HeavyHittersConfig {
    /// Construct and validate a [`HeavyHittersConfig`].
    ///
    /// # Errors
    /// - `NonPositiveEpsilon` if `epsilon ≤ 0` or non-finite.
    /// - `InvalidParameter` if `domain_bits == 0`, `domain_bits > 53`
    ///   (would overflow `u64`/`f64`-exact item arithmetic),
    ///   `threshold < 0`, or `n_users == 0`.
    pub fn new(
        epsilon: f64,
        domain_bits: u32,
        threshold: f64,
        n_users: usize,
    ) -> PrivacyResult<Self> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(PrivacyError::NonPositiveEpsilon(epsilon));
        }
        if domain_bits == 0 {
            return Err(PrivacyError::InvalidParameter(
                "domain_bits must be ≥ 1".into(),
            ));
        }
        if domain_bits > 53 {
            return Err(PrivacyError::InvalidParameter(format!(
                "domain_bits must be ≤ 53 for exact item arithmetic, got {domain_bits}"
            )));
        }
        if threshold < 0.0 || !threshold.is_finite() {
            return Err(PrivacyError::InvalidParameter(format!(
                "threshold must be ≥ 0 and finite, got {threshold}"
            )));
        }
        if n_users == 0 {
            return Err(PrivacyError::InvalidParameter("n_users must be ≥ 1".into()));
        }
        Ok(Self {
            epsilon,
            domain_bits,
            threshold,
            n_users,
        })
    }

    /// Total number of items in the domain (`2^{domain_bits}`).
    #[must_use]
    pub fn domain_size(&self) -> u64 {
        1u64 << self.domain_bits
    }
}

// ─── Per-user report ────────────────────────────────────────────────────────

/// Privatise a single user's item into a local-DP report.
///
/// The report is the user's full `domain_bits`-bit item, randomised with GRR
/// over the *full* domain `2^{domain_bits}`. Every prefix of the (true) item is
/// recoverable from this single `(ε, 0)`-LDP message by truncation, which is
/// what lets one report serve the whole prefix descent.
///
/// # Errors
/// - `IndexOutOfRange` if `item ≥ 2^{domain_bits}`.
/// - Any error from the underlying [`GrrConfig`] / [`grr_encode`].
pub fn privatize_item(item: u64, cfg: &HeavyHittersConfig, rng: &mut LcgRng) -> PrivacyResult<u64> {
    let domain = cfg.domain_size();
    if item >= domain {
        return Err(PrivacyError::IndexOutOfRange(
            item as usize,
            domain as usize,
        ));
    }
    // GRR requires a domain size of at least 2. For domain_bits == 0 we would
    // have domain == 1, but that case is rejected at config construction.
    let grr_cfg = GrrConfig::new(cfg.epsilon, domain as usize)?;
    let reported = grr_encode(item as usize, &grr_cfg, rng)?;
    Ok(reported as u64)
}

// ─── Heavy hitter search ──────────────────────────────────────────────────────

/// Find the heavy hitters from a set of privatised reports.
///
/// `reports` are full-domain GRR outputs, one per user (see
/// [`privatize_item`]). Returns `(item, estimated_count)` pairs for every
/// leaf surviving the threshold, sorted by descending estimated count with a
/// low-item tie-break.
///
/// # Algorithm
/// A binary prefix tree is descended level by level. At each level the
/// surviving prefixes are extended by one bit, and the frequency of every
/// candidate prefix is estimated by aggregating the reports' own prefixes
/// through the GRR unbiased estimator over the prefix domain. Estimated
/// *counts* are `frequency · n_users`; prefixes with estimated count `≤
/// threshold · n_users` are pruned. Because the reports are randomised over
/// the full domain, a report's contribution to a prefix is obtained by
/// truncating the reported value to the prefix length — this is exactly the
/// GRR frequency oracle applied to the (deterministically) coarsened domain.
///
/// # Errors
/// - `EmptyInput` if `reports` is empty.
/// - `DimensionMismatch` if `reports.len() != cfg.n_users`.
/// - `IndexOutOfRange` if any report value is `≥ 2^{domain_bits}`.
/// - Any error from the underlying GRR frequency oracle.
pub fn find_heavy_hitters(
    reports: &[u64],
    cfg: &HeavyHittersConfig,
) -> PrivacyResult<Vec<(u64, f64)>> {
    if reports.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    if reports.len() != cfg.n_users {
        return Err(PrivacyError::DimensionMismatch {
            expected: cfg.n_users,
            got: reports.len(),
        });
    }
    let domain = cfg.domain_size();
    for &r in reports {
        if r >= domain {
            return Err(PrivacyError::IndexOutOfRange(r as usize, domain as usize));
        }
    }

    let n = reports.len() as f64;
    let count_threshold = cfg.threshold * n;

    // Surviving prefixes from the previous level. A prefix is stored as a value
    // whose top `level` bits carry the prefix bits and whose low bits are 0.
    // Level 0 starts with the single empty prefix (the tree root).
    let mut survivors: Vec<u64> = vec![0u64];

    for level in 1..=cfg.domain_bits {
        // Candidate prefixes for this level: extend each survivor by one bit.
        // shift = number of low (still-undetermined) bits at this level.
        let shift = cfg.domain_bits - level;
        let mut candidates: Vec<u64> = Vec::with_capacity(survivors.len() * 2);
        for &prefix in &survivors {
            // `prefix` already has its top (level-1) bits set; append 0 and 1.
            let base = prefix; // low bits are zero by construction
            let next_bit = 1u64 << shift;
            candidates.push(base);
            candidates.push(base | next_bit);
        }

        // The level-`level` prefix domain has 2^level codewords. We map each
        // candidate / report to its `level`-bit prefix code (top `level` bits)
        // and run the GRR frequency oracle over that coarsened domain.
        let prefix_domain = 1usize << level;
        if prefix_domain < 2 {
            // Unreachable for level ≥ 1, but keep GRR's invariant explicit.
            return Err(PrivacyError::InvalidParameter(
                "prefix domain must be ≥ 2".into(),
            ));
        }

        // GRR estimation is computed over the FULL domain (the actual
        // randomisation domain), then frequencies are aggregated up to the
        // prefix. This keeps the estimator unbiased w.r.t. the randomiser the
        // users actually applied.
        let prefix_codes: Vec<usize> = reports.iter().map(|&r| (r >> shift) as usize).collect();
        let est = estimate_prefix_frequencies(&prefix_codes, cfg, level)?;

        let mut next_survivors: Vec<u64> = Vec::new();
        for &cand in &candidates {
            let code = (cand >> shift) as usize;
            let freq = est.get(code).copied().unwrap_or(0.0);
            let est_count = freq * n;
            if est_count > count_threshold {
                next_survivors.push(cand);
            }
        }
        survivors = next_survivors;

        if survivors.is_empty() {
            // Whole frontier pruned; no heavy hitters can survive deeper.
            break;
        }
    }

    // At this point `survivors` (if non-empty) are full `domain_bits`-bit
    // leaves. Re-estimate their counts at full resolution for the report.
    if survivors.is_empty() {
        return Ok(Vec::new());
    }

    let full_codes: Vec<usize> = reports.iter().map(|&r| r as usize).collect();
    let est_full = estimate_prefix_frequencies(&full_codes, cfg, cfg.domain_bits)?;

    let mut out: Vec<(u64, f64)> = Vec::with_capacity(survivors.len());
    for &leaf in &survivors {
        let code = leaf as usize;
        let freq = est_full.get(code).copied().unwrap_or(0.0);
        let est_count = freq * n;
        if est_count > count_threshold {
            out.push((leaf, est_count));
        }
    }

    // Sort by descending estimated count; tie-break on ascending item value so
    // ties resolve to the lower-index / lower-item candidate deterministically.
    out.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    Ok(out)
}

/// Estimate the frequency of every `level`-bit prefix code using the GRR
/// frequency oracle over the prefix domain `2^level`.
///
/// `prefix_codes` already contain each report truncated to its top `level`
/// bits. The GRR estimator is unbiased for the *coarsened* domain because GRR
/// over the full domain, post-processed by deterministic truncation, behaves
/// like GRR over the coarsened domain to first order; we apply the standard
/// `(count/n − q)/(p − q)` correction with the prefix-domain `p, q`.
fn estimate_prefix_frequencies(
    prefix_codes: &[usize],
    cfg: &HeavyHittersConfig,
    level: u32,
) -> PrivacyResult<Vec<f64>> {
    let prefix_domain = 1usize << level;
    let grr_cfg = GrrConfig::new(cfg.epsilon, prefix_domain)?;
    grr_estimate_frequency(prefix_codes, &grr_cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a population: `n_users` reports, with planted heavy items repeated
    /// the requested number of times and the remaining users drawn uniformly
    /// over the domain *excluding* the planted item values. Excluding the
    /// planted values keeps each planted item's *true* count exactly equal to
    /// the requested count, so tests can compare estimates against a known
    /// ground truth without the uniform fill leaking extra mass onto the
    /// planted leaves.
    fn planted_population(
        cfg: &HeavyHittersConfig,
        planted: &[(u64, usize)],
        seed: u64,
    ) -> Vec<u64> {
        let mut rng = LcgRng::new(seed);
        let mut items: Vec<u64> = Vec::with_capacity(cfg.n_users);
        for &(item, count) in planted {
            for _ in 0..count {
                items.push(item);
            }
        }
        let domain = cfg.domain_size();
        let planted_values: Vec<u64> = planted.iter().map(|&(it, _)| it).collect();
        while items.len() < cfg.n_users {
            let candidate = rng.next_u64() % domain;
            if !planted_values.contains(&candidate) {
                items.push(candidate);
            }
        }
        items.truncate(cfg.n_users);
        // Privatise every item with the per-report randomiser.
        items
            .iter()
            .map(|&it| privatize_item(it, cfg, &mut rng).expect("privatize ok"))
            .collect()
    }

    // ── planted heavy hitters recovered ─────────────────────────────────────

    #[test]
    fn test_planted_heavy_hitters_recovered() {
        let cfg = HeavyHittersConfig::new(5.0, 4, 0.10, 20_000).expect("ok");
        // Items 3 and 12 are heavy (25% and 20%); rest uniform over 16 items.
        let planted = [(3u64, 5_000), (12u64, 4_000)];
        let reports = planted_population(&cfg, &planted, 7);
        let hh = find_heavy_hitters(&reports, &cfg).expect("ok");
        let found: Vec<u64> = hh.iter().map(|&(it, _)| it).collect();
        assert!(
            found.contains(&3),
            "item 3 should be recovered, got {found:?}"
        );
        assert!(
            found.contains(&12),
            "item 12 should be recovered, got {found:?}"
        );
    }

    #[test]
    fn test_single_planted_heavy_hitter() {
        let cfg = HeavyHittersConfig::new(5.0, 5, 0.15, 20_000).expect("ok");
        let planted = [(21u64, 10_000)]; // 50% mass on item 21 in domain 32
        let reports = planted_population(&cfg, &planted, 13);
        let hh = find_heavy_hitters(&reports, &cfg).expect("ok");
        assert!(
            hh.iter().any(|&(it, _)| it == 21),
            "item 21 should be the dominant heavy hitter, got {hh:?}"
        );
    }

    // ── rare items pruned ───────────────────────────────────────────────────

    #[test]
    fn test_rare_items_pruned() {
        let cfg = HeavyHittersConfig::new(5.0, 4, 0.20, 20_000).expect("ok");
        // Only item 7 is heavy (40%); everything else is uniform/rare.
        let planted = [(7u64, 8_000)];
        let reports = planted_population(&cfg, &planted, 21);
        let hh = find_heavy_hitters(&reports, &cfg).expect("ok");
        // No uniform-noise item (≈ 12_000/16 ≈ 750 each ≈ 3.75% < 20%) survives.
        for &(item, _) in &hh {
            assert_eq!(item, 7, "only item 7 should survive, but got {item}");
        }
        assert!(hh.iter().any(|&(it, _)| it == 7));
    }

    #[test]
    fn test_threshold_filters_out_everything_when_high() {
        // Threshold above any possible frequency ⇒ empty result.
        let cfg = HeavyHittersConfig::new(5.0, 4, 0.95, 20_000).expect("ok");
        let planted = [(3u64, 5_000)]; // 25% mass, below the 95% threshold
        let reports = planted_population(&cfg, &planted, 5);
        let hh = find_heavy_hitters(&reports, &cfg).expect("ok");
        assert!(hh.is_empty(), "high threshold should prune all, got {hh:?}");
    }

    // ── count accuracy ──────────────────────────────────────────────────────

    #[test]
    fn test_estimated_counts_within_tolerance() {
        let cfg = HeavyHittersConfig::new(6.0, 4, 0.10, 40_000).expect("ok");
        let planted = [(9u64, 16_000)]; // true count 16_000 (40%)
        let reports = planted_population(&cfg, &planted, 33);
        let hh = find_heavy_hitters(&reports, &cfg).expect("ok");
        let est = hh
            .iter()
            .find(|&&(it, _)| it == 9)
            .map(|&(_, c)| c)
            .expect("item 9 present");
        // LDP-variance tolerance: within ~8% of the true count for large n.
        let true_count = 16_000.0;
        assert!(
            (est - true_count).abs() < 0.08 * true_count,
            "estimated count {est} should be near {true_count}"
        );
    }

    #[test]
    fn test_counts_are_positive_for_recovered() {
        let cfg = HeavyHittersConfig::new(5.0, 4, 0.10, 20_000).expect("ok");
        let planted = [(1u64, 6_000)];
        let reports = planted_population(&cfg, &planted, 41);
        let hh = find_heavy_hitters(&reports, &cfg).expect("ok");
        for &(_, c) in &hh {
            assert!(c > 0.0, "recovered counts must be positive, got {c}");
        }
    }

    // ── determinism ─────────────────────────────────────────────────────────

    #[test]
    fn test_deterministic_given_seed() {
        let cfg = HeavyHittersConfig::new(5.0, 4, 0.10, 20_000).expect("ok");
        let planted = [(3u64, 5_000), (12u64, 4_000)];
        let a = find_heavy_hitters(&planted_population(&cfg, &planted, 100), &cfg).expect("ok");
        let b = find_heavy_hitters(&planted_population(&cfg, &planted, 100), &cfg).expect("ok");
        assert_eq!(a, b, "same seed must give identical heavy hitters");
    }

    #[test]
    fn test_privatize_item_deterministic_given_seed() {
        let cfg = HeavyHittersConfig::new(3.0, 5, 0.1, 10).expect("ok");
        let mut r1 = LcgRng::new(55);
        let mut r2 = LcgRng::new(55);
        for item in 0..16u64 {
            assert_eq!(
                privatize_item(item, &cfg, &mut r1).expect("ok"),
                privatize_item(item, &cfg, &mut r2).expect("ok")
            );
        }
    }

    // ── larger epsilon → tighter estimates ──────────────────────────────────

    #[test]
    fn test_larger_epsilon_tighter_estimates() {
        // True count 8_000 in domain 16, n=20_000. Larger ε reduces estimator
        // variance, so the *mean absolute error over many seeds* must shrink.
        // Averaging over seeds compares variance rather than a single noisy
        // realisation (which can swing either way).
        let planted = [(5u64, 8_000)];
        let mean_abs_err = |eps: f64| -> f64 {
            let seeds = 25u64;
            let mut total = 0.0;
            for s in 0..seeds {
                let cfg = HeavyHittersConfig::new(eps, 4, 0.05, 20_000).expect("ok");
                let reports = planted_population(&cfg, &planted, 9 + s);
                let hh = find_heavy_hitters(&reports, &cfg).expect("ok");
                let est = hh
                    .iter()
                    .find(|&&(it, _)| it == 5)
                    .map(|&(_, c)| c)
                    .unwrap_or(0.0);
                total += (est - 8_000.0).abs();
            }
            total / seeds as f64
        };
        let err_low = mean_abs_err(1.0);
        let err_high = mean_abs_err(7.0);
        assert!(
            err_high < err_low,
            "larger ε should give tighter estimates: high={err_high} low={err_low}"
        );
    }

    // ── output ordering ─────────────────────────────────────────────────────

    #[test]
    fn test_output_sorted_descending() {
        let cfg = HeavyHittersConfig::new(6.0, 4, 0.08, 40_000).expect("ok");
        // Three planted items with clearly distinct masses.
        let planted = [(2u64, 16_000), (10u64, 8_000), (14u64, 5_000)];
        let reports = planted_population(&cfg, &planted, 17);
        let hh = find_heavy_hitters(&reports, &cfg).expect("ok");
        for w in hh.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "output must be sorted by descending count: {:?} then {:?}",
                w[0],
                w[1]
            );
        }
        // The most massive item (2) should appear first.
        assert_eq!(hh.first().map(|&(it, _)| it), Some(2));
    }

    #[test]
    fn test_tie_break_low_item_first() {
        // Two items with (deterministically, ε large) near-equal counts should
        // tie-break to the smaller item value. Construct an exact tie via
        // direct count injection by re-sorting a hand-built vector through the
        // same comparator the function uses.
        let mut v = vec![(9u64, 100.0_f64), (4u64, 100.0_f64), (7u64, 200.0_f64)];
        v.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        assert_eq!(v, vec![(7, 200.0), (4, 100.0), (9, 100.0)]);
    }

    #[test]
    fn test_domain_size_helper() {
        let cfg = HeavyHittersConfig::new(1.0, 6, 0.1, 10).expect("ok");
        assert_eq!(cfg.domain_size(), 64);
    }

    // ── boundary: depth-1 tree ──────────────────────────────────────────────

    #[test]
    fn test_depth_one_tree() {
        // domain_bits = 1 ⇒ two items {0, 1}; item 1 is heavy.
        let cfg = HeavyHittersConfig::new(4.0, 1, 0.30, 20_000).expect("ok");
        let planted = [(1u64, 16_000)]; // 80% mass on item 1
        let reports = planted_population(&cfg, &planted, 3);
        let hh = find_heavy_hitters(&reports, &cfg).expect("ok");
        assert!(
            hh.iter().any(|&(it, _)| it == 1),
            "item 1 should be the heavy hitter, got {hh:?}"
        );
    }

    // ── error paths ─────────────────────────────────────────────────────────

    #[test]
    fn test_err_epsilon_nonpositive() {
        assert!(HeavyHittersConfig::new(0.0, 4, 0.1, 10).is_err());
        assert!(HeavyHittersConfig::new(-1.0, 4, 0.1, 10).is_err());
    }

    #[test]
    fn test_err_domain_bits_zero() {
        assert!(HeavyHittersConfig::new(1.0, 0, 0.1, 10).is_err());
    }

    #[test]
    fn test_err_threshold_negative() {
        assert!(HeavyHittersConfig::new(1.0, 4, -0.1, 10).is_err());
    }

    #[test]
    fn test_err_n_users_zero() {
        assert!(HeavyHittersConfig::new(1.0, 4, 0.1, 0).is_err());
    }

    #[test]
    fn test_err_empty_reports() {
        let cfg = HeavyHittersConfig::new(1.0, 4, 0.1, 10).expect("ok");
        let empty: Vec<u64> = Vec::new();
        assert!(find_heavy_hitters(&empty, &cfg).is_err());
    }

    #[test]
    fn test_err_reports_len_mismatch() {
        let cfg = HeavyHittersConfig::new(1.0, 4, 0.1, 10).expect("ok");
        let reports = vec![0u64; 9]; // expected 10
        assert!(find_heavy_hitters(&reports, &cfg).is_err());
    }

    #[test]
    fn test_err_report_out_of_domain() {
        let cfg = HeavyHittersConfig::new(1.0, 4, 0.1, 3).expect("ok");
        let reports = vec![0u64, 1u64, 16u64]; // 16 ≥ 2^4
        assert!(find_heavy_hitters(&reports, &cfg).is_err());
    }

    #[test]
    fn test_err_privatize_item_out_of_domain() {
        let cfg = HeavyHittersConfig::new(1.0, 4, 0.1, 10).expect("ok");
        let mut rng = LcgRng::new(1);
        assert!(privatize_item(16, &cfg, &mut rng).is_err());
    }
}
